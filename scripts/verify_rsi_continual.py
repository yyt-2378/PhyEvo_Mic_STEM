#!/usr/bin/env python3
"""Independently replay controller decisions, hashes, and held-out metrics."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json

import numpy as np

from scripts.run_rsi_continual import ROOT, all_jobs, load_protocol, read, run_path, save_sealed, verified
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file
from tfree_detection.rsi_continual import materialize, measure, retention_gate, validate_partitions


def verify_run_job(job):
    output, arm, seed, fold, recompute = job
    output = Path(output)
    protocol, dataset = load_protocol(output)
    run = run_path(output, arm, seed, fold)
    verified(run/'training_complete.json')
    incumbent, accepted, rejected, inputs = {}, 0, 0, []
    for r in range(1, protocol['rounds']+1):
        rd = run/f'round-{r:02d}'
        feedback = verified(rd/'feedback.json')
        new, decisions = retention_gate(feedback['control_rows'], incumbent, r, **protocol['gate'])
        if new != feedback['incumbent'] or decisions != feedback['decisions'] or feedback['uses_outer_results']:
            raise ValueError('Controller decisions do not reproduce')
        incumbent = new
        accepted += sum(d['accepted'] for d in decisions)
        rejected += sum(not d['accepted'] for d in decisions)
        meta = read(rd/'checkpoint'/'checkpoint.json')
        if meta['train_partition'] != fold['partition']['train'] or meta['arm'] != arm or meta['seed'] != seed:
            raise ValueError('Training identity mismatch')
        if sum(meta['sampler_audit']['by_case'].values()) != protocol['batch_size']*protocol['steps_per_round']:
            raise ValueError('Unequal gradient exposure')
        if not set(map(int, meta['sampler_audit']['by_case'])) <= {x['case_number'] for x in fold['partition']['train']}:
            raise ValueError('Non-training case sampled')
        if fold['partition']['outer_group'] in meta['sampler_audit']['by_source']:
            raise ValueError('Outer source sampled')
        expected_parent = sha256_file(run/f'round-{r-1:02d}'/'feedback.json') if r > 1 else None
        if feedback['parent_feedback_sha256'] != expected_parent or meta['parent_feedback_sha256'] != expected_parent:
            raise ValueError('Recursive lineage broken')
        inputs.extend([rd/'feedback.json', rd/'feedback.seal.json'])
    evaluation = verified(run/'outer_evaluation.json')
    rows = evaluation['rows']
    expected = {(r, x['case_number']) for r in range(1, protocol['rounds']+1) for x in fold['partition']['outer']}
    if {(r['round'], r['case_number']) for r in rows} != expected or len(rows) != len(expected):
        raise ValueError('Missing or duplicated outer outcomes')
    if recompute:
        manifest, arrays = load_learning_dataset(ROOT, dataset)
        records = {r['case_number']: r for r in map(json.loads, (contained_path(ROOT, manifest['frozen_bundle'])/'cases.jsonl').read_text().splitlines())}
        entries, planes, candidates = materialize(manifest['entries'], arrays, records, fold['partition']['outer'])
        by_id = {e['case_number']: e for e in entries}
        prior = {}
        for row in rows:
            r, n = row['round'], row['case_number']
            rd = run/f'round-{r:02d}'
            with np.load(rd/'outer_scores'/f'case-{n:03d}.npz', allow_pickle=False) as archive:
                score = archive['score']
            measured = measure(by_id[n], planes[n], candidates[n], score)
            route = read(rd/'feedback.json')['incumbent'][by_id[n]['geometry']]
            source = measured if route['round'] in (0, r) else prior[(route['round'], n)]
            measured['methods']['gated_release'] = source['methods'][route['branch']]
            if measured['methods'] != row['methods'] or route != row['route']:
                raise ValueError(f'Outer metric does not reproduce: {arm} {seed} {fold["fold_id"]} {r} {n}')
            if any(m['precision'] is not None or m['f1'] is not None for m in row['methods'].values()):
                raise ValueError('Unsupported precision/F1 claim')
            prior[(r, n)] = measured
    return {'arm': arm, 'seed': seed, 'fold_id': fold['fold_id'], 'accepted': accepted,
            'rejected': rejected, 'outer_rows': len(rows), 'metrics_recomputed': recompute}


def verify(output, target, *, workers=4, recompute=True):
    protocol, dataset = load_protocol(output)
    manifest, _ = load_learning_dataset(ROOT, dataset)
    verified(output/'all_training_sealed.json')
    verified(output/'summary.json')
    for f in protocol['folds']:
        validate_partitions(manifest['entries'], f['partition'])
    pairs = []
    for seed in protocol['seeds']:
        for fold in protocol['folds']:
            counts = {}
            for r in (1, 2, 3, 4):
                paths = [run_path(output, a, seed, fold)/f'round-{r:02d}'/'checkpoint'/'state.npz' for a in protocol['arms']]
                with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
                    keys = [k for k in a.files if k.startswith('model:') or k.startswith('optimizer:') or k.startswith('rng:')]
                    counts[r] = sum(not np.array_equal(a[k], b[k]) for k in keys)
                if r == 1 and counts[r]:
                    raise ValueError('First round arms were not exactly compute matched')
            pairs.append({'seed': seed, 'fold': fold['fold_id'], 'different_arrays_by_round': counts})
    jobs = [(str(output), a, s, f, recompute) for a, s, f in all_jobs(protocol)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        runs = []
        for item in pool.map(verify_run_job, jobs):
            runs.append(item); print(json.dumps(item), flush=True)
    result = {'protocol_sha256': sha256_file(output/'protocol.json'), 'models': len(runs),
              'checkpoints': len(runs)*protocol['rounds'], 'outer_rows': sum(x['outer_rows'] for x in runs),
              'first_round_exactly_matched': True, 'pairs': pairs, 'runs': runs,
              'accepted_control_updates': sum(x['accepted'] for x in runs),
              'rejected_control_updates': sum(x['rejected'] for x in runs),
              'all_outer_metrics_recomputed': recompute, 'data_files_unchanged': True,
              'production_approved': False}
    save_sealed(target, result, [output/'protocol.json', output/'all_training_sealed.json', output/'summary.json'])
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs', 'pairs')}, indent=2))


def runtime_replay(output, target, device):
    from tfree_detection.config import load_config
    from tfree_detection.framework import TrainingFreeDetectionFramework
    from tfree_detection.expert_student import torch
    from tfree_detection.rsi_continual_runtime import detect_with_continual
    from tfree_detection.rsi_runtime import save_rsi_detection
    from tfree_detection.rsi_feedback import feedback_predictions
    protocol, dataset = load_protocol(output)
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    frozen = contained_path(ROOT, manifest['frozen_bundle'])
    records = {r['case_number']: r for r in map(json.loads, (frozen/'cases.jsonl').read_text().splitlines())}
    entries = {e['case_number']: e for e in manifest['entries']}
    torch.set_num_threads(2)
    folder = output/'runtime_validation'
    folder.mkdir(exist_ok=False)
    results, inputs = [], []
    for n in (63, 103, 135):
        record, entry = records[n], entries[n]
        sample = record['sample']
        fold = next(f for f in protocol['folds'] if f['partition']['outer_group'] == entry['group'])
        binding = output/'bindings'/f'error_feedback--seed-23--{fold["fold_id"]}.json'
        config = load_config(contained_path(ROOT, record['prediction']['config_path']))
        if config.stage2.sam3.enabled:
            raise ValueError('Raw replay must not provision external models')
        source = contained_path(ROOT, sample['source_path'])
        options = {'frame_index': sample['frame_index'], 'dataset_path': sample['dataset_path']}
        framework = TrainingFreeDetectionFramework(config)
        native = framework.detect(source, **options)
        detected = detect_with_continual(framework, source, project_root=ROOT, binding=binding,
                                         experimental_research=True, student_device=device, **options)
        if detected.metadata['status'] != 'continual_research':
            raise ValueError(f'Unexpected runtime fallback: {detected.metadata}')
        physical = {}
        for key in ('normalized_image', 'score_map', 'probability_map', 'mask', 'reliability_map', 'decision_map'):
            a, b = getattr(native, key), getattr(detected.physics, key)
            physical[key] = bool(np.array_equal(a, b, equal_nan=True)) if a is not None else b is None
        if not all(physical.values()):
            raise ValueError('Native physics changed')
        rd = detected.metadata['route']['round']
        path = run_path(output, 'error_feedback', 23, fold)/f'round-{rd:02d}'/'outer_scores'/f'case-{n:03d}.npz'
        with np.load(path, allow_pickle=False) as archive:
            saved = archive['score']
        score_equal = np.array_equal(saved, detected.student_score)
        candidates = feedback_predictions(arrays[n]['image'], arrays[n]['teacher'], record['prediction']['candidates'],
                                          saved, entry['geometry'])
        equality = {}
        for method, p in candidates.items():
            q = detected.candidates[method]
            equality[method] = bool(np.array_equal(p.points_xy, q.points_xy) if p.points_xy is not None else np.array_equal(p.mask, q.mask))
        case_dir = folder/f'case-{n:03d}'
        save_rsi_detection(detected, config, case_dir, project_root=ROOT)
        receipt = {'case_number': n, 'device': device, 'seed': 23, 'route': detected.metadata['route'],
                   'physical_arrays_exact': physical, 'student_score_exact': bool(score_equal),
                   'student_score_max_abs_difference': float(np.max(np.abs(saved-detected.student_score))),
                   'candidate_arrays_exact': equality}
        results.append(receipt)
        inputs.extend([binding, path, case_dir/'rsi_detection.json', source])
        print(json.dumps(receipt), flush=True)
    save_sealed(target, {'replays': results, 'all_exact': all(r['student_score_exact'] and all(r['candidate_arrays_exact'].values()) for r in results)}, inputs)
    if not all(r['student_score_exact'] and all(r['candidate_arrays_exact'].values()) for r in results):
        raise ValueError('Runtime numerical mismatch retained in receipt')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('outputs/rsi_cycle_003'))
    p.add_argument('--target', type=Path, default=Path('outputs/rsi_cycle_003/verification.json'))
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--hashes-only', action='store_true')
    p.add_argument('--runtime', action='store_true')
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    if a.runtime:
        runtime_replay(contained_path(ROOT, a.output), contained_path(ROOT, a.target), a.device)
    else:
        verify(contained_path(ROOT, a.output), contained_path(ROOT, a.target), workers=a.workers, recompute=not a.hashes_only)


if __name__ == '__main__':
    main()
