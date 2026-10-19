#!/usr/bin/env python3
"""Four-round, compute-matched feedback experiment; immutable research outputs.

prepare -> train -> evaluate -> report. No outer scores are computed until ALL
training/controller histories have been sealed. Original data and runs are read
only. Interrupted, incomplete files are retained for diagnosis, never deleted.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from scripts.run_rsi_nested import idle_gpus
from tfree_detection.expert_student import make_expert_student, predict_scores, restore_checkpoint, save_checkpoint, torch
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.rsi_continual import FeedbackSampler, error_state, make_partitions, materialize, measure, retention_gate
from tfree_detection.rsi_refinement import POLICIES, refinement_loss

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('scripts/run_rsi_continual.py', 'src/tfree_detection/rsi_continual.py',
           'src/tfree_detection/rsi_refinement.py', 'src/tfree_detection/rsi_feedback.py',
           'src/tfree_detection/rsi_evaluation.py', 'src/tfree_detection/expert_student.py',
           'src/tfree_detection/expert_supervision.py', 'src/tfree_detection/student.py',
           'scripts/run_rsi_nested.py')


def read(path):
    return json.loads(path.read_text())


def hashes():
    return {p: sha256_file(ROOT/p) for p in SOURCES}


def prepare(output, dataset):
    output, dataset = contained_path(ROOT, output), contained_path(ROOT, dataset)
    if output.exists():
        raise FileExistsError('Select a fresh output directory')
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    frozen = contained_path(ROOT, manifest['frozen_bundle'])
    records = {r['case_number']: r for r in map(json.loads, (frozen/'cases.jsonl').read_text().splitlines())}
    groups = sorted({f['held_out_group'] for f in manifest['folds']})
    folds = []
    for i, group in enumerate(groups):
        part = make_partitions(manifest['entries'], group)
        inventory = {}
        for role in ('train', 'control', 'outer'):
            entries, _, _ = materialize(manifest['entries'], arrays, records, part[role])
            inventory[role] = [{'case_number': e['case_number'], 'group': e['group'],
                                'geometry': e['geometry'], 'rows': e['original_rows'],
                                'positive_pixels': e['human_positive_pixels'],
                                'point_count': len(e['points_xy']) if e['geometry'] == 'point' else None} for e in entries]
        folds.append({'fold_id': f'outer-{i:02d}', 'partition': part, 'inventory': inventory})
    protocol = {
        'format': 'phyevo-bounded-continual-development-v1', 'source_hashes': hashes(),
        'dataset': str(dataset.relative_to(ROOT)), 'manifest_sha256': sha256_file(dataset/'manifest.json'),
        'frozen_records_sha256': sha256_file(frozen/'cases.jsonl'),
        'folds': folds, 'seeds': [23, 41, 67], 'arms': ['fixed_replay', 'error_feedback'],
        'rounds': 4, 'steps_per_round': 400, 'batch_size': 32, 'patch_size': 128,
        'base_channels': 16, 'learning_rate': .001, 'gradient_clip': 5., 'device': 'cuda:0', 'threads': 2,
        'base_policy': 'photometric', 'loss_changed': False, 'architecture_changed': False,
        'feedback': 'weight=min(5,1+ceil(2*miss_fraction+2*forgetting)); 50% base source mixture; all positives plus two copies of missed positives',
        'gate': {'minimum_gain': .005, 'maximum_loss': .01},
        'gate_branches': ['student_expert', 'student_reranked_physics_plus_rescue'],
        'evaluation': {'point_budget_per_512': 50, 'point_tolerance_px': 6, 'point_min_distance_px': 4, 'area_budget_fraction': .1},
        'partition_rule': 'outer whole acquisition; internal control last20% frames with2-frame gap, or bottom tile with128px guard for single images',
        'research_scope': 'reused development cohort; internal same-source control is not source-independent validation',
        'primary_comparison': 'round4 error-feedback gated vs round4 fixed-replay gated; paired by outer acquisition and seed',
        'secondary_comparisons': 'all four rounds, latest student and fixed50% fusion, gate-only ablation, physics; report regressions',
        'candidate_training_continues_after_gate_rejection': True,
        'production_approved': False, 'precision_f1_identifiable': False,
        'outer_scoring_requires_all_controller_histories_sealed': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
    }
    output.mkdir(parents=True)
    write_new_json(output/'protocol.json', protocol)
    print(json.dumps({'prepared': str(output), 'jobs': len(folds)*len(protocol['seeds'])*2,
                      'round_checkpoints': len(folds)*len(protocol['seeds'])*2*4}))


def load_protocol(output):
    protocol = read(output/'protocol.json')
    dataset = contained_path(ROOT, protocol['dataset'])
    if protocol['source_hashes'] != hashes() or protocol['manifest_sha256'] != sha256_file(dataset/'manifest.json'):
        raise ValueError('Code or dataset changed after registration')
    return protocol, dataset


def save_sealed(path, value, inputs):
    write_new_json(path, value)
    write_new_json(path.with_suffix('.seal.json'), {'sha256': sha256_file(path),
                   'inputs': {str(p.relative_to(ROOT)): sha256_file(p) for p in inputs}})


def verified(path):
    seal = read(path.with_suffix('.seal.json'))
    if sha256_file(path) != seal['sha256']:
        raise ValueError(f'Changed artifact: {path}')
    for name, digest in seal['inputs'].items():
        if sha256_file(contained_path(ROOT, name)) != digest:
            raise ValueError(f'Changed input: {name}')
    return read(path)


def run_path(output, arm, seed, fold):
    return output/arm/f'seed-{seed}'/fold['fold_id']


def all_jobs(protocol):
    return [(a, s, f) for s in protocol['seeds'] for a in protocol['arms'] for f in protocol['folds']]


def score(model, entry, planes, device):
    return predict_scores(model, planes[entry['case_number']]['image'], entry['geometry_index'],
                          device=device, normalized=True, tile_size=256, overlap=64)


def train_worker(output, arm, seed, fold, *, resume=False):
    protocol, dataset = load_protocol(output)
    run = run_path(output, arm, seed, fold)
    if (run/'training_complete.json').exists():
        if not resume:
            raise FileExistsError(run)
        verified(run/'training_complete.json')
        return
    if run.exists() and not resume:
        raise FileExistsError('Existing run requires explicit resume')
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    frozen = contained_path(ROOT, manifest['frozen_bundle'])
    if sha256_file(frozen/'cases.jsonl') != protocol['frozen_records_sha256']:
        raise ValueError('Frozen prediction records changed')
    records = {r['case_number']: r for r in map(json.loads, (frozen/'cases.jsonl').read_text().splitlines())}
    part = fold['partition']
    train, planes, _ = materialize(manifest['entries'], arrays, records, part['train'])
    control, control_planes, control_candidates = materialize(manifest['entries'], arrays, records, part['control'])
    # Outer pixels and annotations are never passed to the learner/controller.
    del arrays, manifest, records
    device = protocol['device']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_expert_student(protocol['base_channels']).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=protocol['learning_rate'])
    state, hard, incumbent = {}, {}, {}
    phash = sha256_file(output/'protocol.json')
    identity = {'protocol_sha256': phash, 'arm': arm, 'seed': seed, 'fold_id': fold['fold_id']}
    run.mkdir(parents=True, exist_ok=True)
    round_paths = []
    for r in range(1, protocol['rounds']+1):
        rd = run/f'round-{r:02d}'
        checkpoint = rd/'checkpoint'
        completed = rd/'feedback.json'
        parent = run/f'round-{r-1:02d}'/'feedback.json' if r > 1 else None
        parent_hash = sha256_file(parent) if parent else None
        sampler = FeedbackSampler(train, planes, excluded_groups=[part['outer_group']],
                                  state=state, hard_masks=hard, enabled=arm == 'error_feedback')
        if checkpoint.exists():
            model, optimizer, rng, meta = restore_checkpoint(checkpoint, device=device, resume=True)
            if any(meta.get(k) != v for k, v in identity.items()) or meta['parent_feedback_sha256'] != parent_hash or meta['round'] != r:
                raise ValueError('Checkpoint/controller identity mismatch')
        else:
            model.train()
            began = time.monotonic()
            for j in range(1, protocol['steps_per_round']+1):
                step = (r-1)*protocol['steps_per_round']+j
                batch, _ = sampler.sample(rng, batch_size=protocol['batch_size'], patch_size=protocol['patch_size'],
                                          device=device, augmentation_seed=seed*1000000+step)
                optimizer.zero_grad(set_to_none=True)
                loss, _ = refinement_loss(model(batch['image']), batch, POLICIES[protocol['base_policy']])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), protocol['gradient_clip'])
                optimizer.step()
                if j % 100 == 0:
                    event = {'round': r, 'step': step, 'loss': float(loss.detach().cpu()), 'seconds': time.monotonic()-began}
                    with (run/'training_events.jsonl').open('a') as stream:
                        stream.write(json.dumps(event)+'\n')
                    print(json.dumps({**identity, **event}), flush=True)
            save_checkpoint(checkpoint, model, optimizer, rng, {
                **identity, 'round': r, 'step': r*protocol['steps_per_round'],
                'base_channels': protocol['base_channels'], 'learning_rate': protocol['learning_rate'],
                'parent_feedback_sha256': parent_hash, 'sampler_audit': sampler.audit,
                'source_probabilities': sampler.probabilities(), 'train_partition': part['train'],
                'train_groups': sorted({e['group'] for e in train}), 'geometry_support': ['point', 'interface']})
        if completed.exists():
            feedback = verified(completed)
        else:
            new_state, masks = {}, {}
            for entry in train:
                n = entry['case_number']
                values = score(model, entry, planes, device)
                new_state[str(n)], masks[str(n)] = error_state(entry, planes[n], values, state.get(str(n), {}).get('best_coverage', 0.))
            rows = [measure(e, control_planes[e['case_number']], control_candidates[e['case_number']],
                            score(model, e, control_planes, device)) for e in control]
            updated, decisions = retention_gate(rows, incumbent, r, **protocol['gate'])
            mask_path = rd/'training_hard_positives.npz'
            with mask_path.open('xb') as stream:
                np.savez_compressed(stream, **masks)
            feedback = {**identity, 'round': r, 'parent_feedback_sha256': parent_hash,
                        'state': new_state, 'incumbent': updated, 'decisions': decisions,
                        'control_rows': rows, 'uses_outer_results': False,
                        'feedback_applied_to_next_training_round': arm == 'error_feedback' and r < protocol['rounds'],
                        'research_only': True}
            inputs = [output/'protocol.json', checkpoint/'checkpoint.json', checkpoint/'state.npz', mask_path]
            if parent:
                inputs.extend([parent, parent.with_suffix('.seal.json')])
            save_sealed(completed, feedback, inputs)
        state, incumbent = feedback['state'], feedback['incumbent']
        with np.load(rd/'training_hard_positives.npz', allow_pickle=False) as archive:
            hard = {int(k): archive[k] for k in archive.files}
        round_paths.extend([completed, completed.with_suffix('.seal.json')])
        print(f'{arm} seed={seed} {fold["fold_id"]} round={r} sealed; routes={incumbent}', flush=True)
    save_sealed(run/'training_complete.json', {**identity, 'rounds': protocol['rounds'], 'outer_evaluated': False}, round_paths)


def evaluate_worker(output, arm, seed, fold, *, resume=False):
    protocol, dataset = load_protocol(output)
    verified(output/'all_training_sealed.json')
    run = run_path(output, arm, seed, fold)
    verified(run/'training_complete.json')
    result_path = run/'outer_evaluation.json'
    if result_path.exists():
        if not resume:
            raise FileExistsError(result_path)
        verified(result_path)
        return
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    frozen = contained_path(ROOT, manifest['frozen_bundle'])
    if sha256_file(frozen/'cases.jsonl') != protocol['frozen_records_sha256']:
        raise ValueError('Frozen records changed')
    records = {r['case_number']: r for r in map(json.loads, (frozen/'cases.jsonl').read_text().splitlines())}
    entries, planes, candidates = materialize(manifest['entries'], arrays, records, fold['partition']['outer'])
    all_rows, inputs, prior = [], [output/'all_training_sealed.json', run/'training_complete.json'], {}
    for r in range(1, protocol['rounds']+1):
        rd = run/f'round-{r:02d}'
        feedback = verified(rd/'feedback.json')
        model, _, _, _ = restore_checkpoint(rd/'checkpoint', device=protocol['device'])
        inputs.extend([rd/'feedback.json', rd/'feedback.seal.json', rd/'checkpoint'/'checkpoint.json', rd/'checkpoint'/'state.npz'])
        target = rd/'outer_scores'
        target.mkdir(exist_ok=True)
        for entry in entries:
            n = entry['case_number']
            path = target/f'case-{n:03d}.npz'
            binding = path.with_suffix('.json')
            if path.exists():
                if not resume:
                    raise FileExistsError(path)
                verified(binding)
                with np.load(path, allow_pickle=False) as archive:
                    values = archive['score']
            else:
                values = score(model, entry, planes, protocol['device'])
                with path.open('xb') as stream:
                    np.savez_compressed(stream, score=values)
                save_sealed(binding, {'case_number': n, 'round': r, 'arm': arm, 'seed': seed},
                            [path, rd/'checkpoint'/'checkpoint.json', output/'all_training_sealed.json'])
            row = measure(entry, planes[n], candidates[n], values)
            route = feedback['incumbent'][entry['geometry']]
            source = row if route['round'] in (0, r) else prior[(route['round'], n)]
            row['methods']['gated_release'] = source['methods'][route['branch']]
            row.update({'round': r, 'arm': arm, 'seed': seed, 'route': route, 'fold_id': fold['fold_id']})
            all_rows.append(row)
            prior[(r, n)] = row
            inputs.extend([path, binding, binding.with_suffix('.seal.json')])
    save_sealed(result_path, {'rows': all_rows, 'selection_precedes_outer_evaluation': True}, inputs)


def seal_training(output, protocol):
    inputs = [output/'protocol.json']
    for a, s, f in all_jobs(protocol):
        run = run_path(output, a, s, f)
        verified(run/'training_complete.json')
        for r in range(1, protocol['rounds']+1):
            verified(run/f'round-{r:02d}'/'feedback.json')
        inputs.extend([run/'training_complete.json', run/'training_complete.seal.json'])
    path = output/'all_training_sealed.json'
    if path.exists():
        verified(path)
    else:
        save_sealed(path, {'jobs': len(all_jobs(protocol)), 'sealed_at_utc': datetime.now(timezone.utc).isoformat(),
                          'outer_evaluation_may_start': True}, inputs)


def launch(output, stage, gpus, *, resume=False):
    protocol, _ = load_protocol(output)
    gpu_ids = [int(g) for g in gpus.split(',')]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError('Specify unique GPUs')
    idle_gpus(gpu_ids)
    jobs = []
    receipt = 'training_complete.json' if stage == 'train' else 'outer_evaluation.json'
    if stage == 'evaluate':
        seal_training(output, protocol)
    for a, s, f in all_jobs(protocol):
        target = run_path(output, a, s, f)/receipt
        if target.exists() and resume:
            verified(target)
        else:
            jobs.append((a, s, f))
    workers = output/'workers'
    workers.mkdir(exist_ok=True)
    active, failures = {}, []
    def event(value):
        with (workers/'process_events.jsonl').open('a') as stream:
            stream.write(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(), **value})+'\n')
    event({'event': 'controller_started', 'stage': stage, 'pid': os.getpid(), 'jobs': len(jobs)})
    while jobs or active:
        for gpu in gpu_ids:
            if gpu in active or not jobs:
                continue
            if shutil.disk_usage(ROOT).free < 15*1024**3:
                failures.append('disk_guard'); jobs.clear(); break
            idle_gpus([gpu])
            a, s, f = jobs.pop(0)
            temp = tempfile.mkdtemp(prefix=f'continual-gpu{gpu}-', dir=ROOT/'.codex-tmp')
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'TMPDIR': temp,
                   'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONDONTWRITEBYTECODE': '1',
                   'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'}
            cmd = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--output', str(output),
                   '--stage', stage, '--arm', a, '--seed', str(s), '--fold', f['fold_id']]
            if resume:
                cmd.append('--resume')
            log = (workers/f'{stage}--{a}--{s}--{f["fold_id"]}.log').open('a')
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[gpu] = process, log
            event({'event': 'started', 'stage': stage, 'pid': process.pid, 'gpu': gpu, 'arm': a, 'seed': s, 'fold': f['fold_id']})
            print(f'{stage} GPU {gpu}: {a} seed={s} {f["fold_id"]} PID={process.pid}', flush=True)
        for gpu, (p, log) in list(active.items()):
            code = p.poll()
            if code is not None:
                log.close(); del active[gpu]
                event({'event': 'finished', 'pid': p.pid, 'gpu': gpu, 'returncode': code})
                print(f'{stage} GPU {gpu}: PID={p.pid} exit={code}', flush=True)
                if code:
                    failures.append({'pid': p.pid, 'returncode': code})
        if active:
            time.sleep(2)
    if failures:
        raise RuntimeError(f'Failures retained, no files removed: {failures}')
    if stage == 'train':
        seal_training(output, protocol)


def summarize(output):
    protocol, _ = load_protocol(output)
    seal_training(output, protocol)
    grouped, inputs = defaultdict(list), []
    for arm, seed, fold in all_jobs(protocol):
        path = run_path(output, arm, seed, fold)/'outer_evaluation.json'
        data = verified(path)
        inputs.extend([path, path.with_suffix('.seal.json')])
        for row in data['rows']:
            for method, metric in row['methods'].items():
                grouped[(row['group'], row['geometry'], arm, seed, row['round'], method)].append(metric['positive_coverage'])
    summaries = defaultdict(list)
    for (group, geometry, arm, seed, r, method), values in grouped.items():
        summaries[(group, geometry, arm, r, method)].append({'seed': seed, 'coverage': float(np.mean(values)), 'cases': len(values)})
    rows = [{'group': k[0], 'geometry': k[1], 'arm': k[2], 'round': k[3], 'method': k[4],
             'mean_coverage': float(np.mean([v['coverage'] for v in values])), 'seeds': sorted(values, key=lambda v: v['seed'])}
            for k, values in sorted(summaries.items())]
    comparisons = []
    for group in sorted({r['group'] for r in rows}):
        selected = {(r['arm'], r['method']): r for r in rows if r['group'] == group and r['round'] == protocol['rounds']}
        a, b = selected[('error_feedback', 'gated_release')], selected[('fixed_replay', 'gated_release')]
        comparisons.append({'group': group, 'geometry': a['geometry'], 'feedback_gated': a['mean_coverage'],
                            'fixed_gated': b['mean_coverage'], 'difference_pp': 100*(a['mean_coverage']-b['mean_coverage']),
                            'seed_differences_pp': [100*(x['coverage']-y['coverage']) for x, y in zip(a['seeds'], b['seeds'])]})
    save_sealed(output/'summary.json', {'rows': rows, 'primary_comparison': comparisons,
                'metric': 'known_positive_coverage_at_fixed_budget_not_accuracy_precision_or_F1',
                'independent_sources': 5, 'frames_and_seeds_are_not_independent_sources': True,
                'evidence': 'retrospective_development_source_holdout', 'production_approved': False}, inputs)
    print(json.dumps(comparisons, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'train', 'evaluate', 'worker', 'summarize'])
    parser.add_argument('--output', type=Path, default=Path('outputs/rsi_cycle_003'))
    parser.add_argument('--dataset', type=Path, default=Path('outputs/expert_cycle_001/learning_dataset_v1'))
    parser.add_argument('--gpus', default='2,3,4,5,6,7')
    parser.add_argument('--arm', choices=['fixed_replay', 'error_feedback'])
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold')
    parser.add_argument('--stage', choices=['train', 'evaluate'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = contained_path(ROOT, args.output)
    if args.command == 'prepare':
        prepare(output, args.dataset)
    elif args.command in ('train', 'evaluate'):
        launch(output, args.command, args.gpus, resume=args.resume)
    elif args.command == 'summarize':
        summarize(output)
    else:
        protocol, _ = load_protocol(output)
        if args.arm not in protocol['arms'] or args.seed not in protocol['seeds']:
            raise ValueError('Unregistered worker')
        fold = next(f for f in protocol['folds'] if f['fold_id'] == args.fold)
        torch.set_num_threads(protocol['threads'])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        (train_worker if args.stage == 'train' else evaluate_worker)(output, args.arm, args.seed, fold, resume=args.resume)


if __name__ == '__main__':
    main()
