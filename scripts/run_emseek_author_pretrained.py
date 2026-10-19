#!/usr/bin/env python3
"""Author EMSeek weights: fixed no-prompt and train-only reference evaluations.

Register before scoring; preserve the frozen best collection and Dennler run.
No LLM service, training, test-label prompting, or choosing a winner per test case.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zlib
from zipfile import ZipFile
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from scripts import run_author_pretrained as base
from scripts.emseek_author_weights import load_model, predict, WEIGHTS, VENDOR
from scripts.fetch_emseek_author_weights import DEST, EXPECTED
from scripts.run_rsi_nested import idle_gpus
from scripts.verify_rsi_cycle import require
from tfree_detection.expert_supervision import contained_path, sha256_file, write_new_json, load_learning_dataset
from tfree_detection.fair_baselines import reference_bank
from tfree_detection.rsi_evaluation import EvaluationProtocol, Prediction, evaluate_case, student_points, student_mask, matched_expert_indices
from benchmarks.run_dennler_reference_baseline import _binary_counts, _counts_to_metrics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = 'outputs/author_pretrained_emseek_20261014_v2'
SOURCE_NAMES = [
    'scripts/run_emseek_author_pretrained.py', 'scripts/emseek_author_weights.py',
    'scripts/fetch_emseek_author_weights.py', 'scripts/run_author_pretrained.py',
    'scripts/run_fair_dense.py', 'src/tfree_detection/rsi_evaluation.py',
    'src/tfree_detection/expert_supervision.py', 'src/tfree_detection/fair_baselines.py',
    'src/tfree_detection/review_dataset.py', 'benchmarks/run_dennler_reference_baseline.py',
    'vendor/fair_baselines_20261014/emseek/emseek/models/unetpp_ref.py',
    'vendor/fair_baselines_20261014/emseek/emseek/agents/segmentor_agent.py',
    'vendor/fair_baselines_20261014/emseek/cfg.py',
]


def read(path):
    return json.loads(path.read_text())


def variants_for_task(task):
    result = [('general_no_prompt', 'unet_ref.pth', None)]
    if task['kind'] == 'wse2' or (task['kind'] == 'expert' and task['geometry'] == 'point'):
        result.append(('task2_no_prompt', 'task2.pth', None))
    if task['kind'] == 'fecral':
        result.append(('task5_no_prompt', 'task5.pth', None))
    if task['kind'] in ('expert', 'wse2'):
        key = task['group'] if task['kind'] == 'expert' else 'external-wse2'
        result.append(('general_train_reference', 'unet_ref.pth', key))
    return result


def training_references(entries, arrays, excluded):
    train = [e for e in entries if e['action'] == 'correct_geometry'
             and e['geometry'] in ('point', 'interface') and e['group'] not in excluded]
    bank, audit = reference_bank(train, arrays, excluded_groups=excluded)
    require(not any(a['group'] in excluded for a in audit), 'Query source in reference bank')
    return bank, audit


def corpus_overlap(tasks):
    inventory_path = ROOT / 'vendor/author_pretrained_20261014_v1/emseek/data_archive_inventory.json'
    inventory = {r['name']: r for r in read(inventory_path)['members']}
    checks = []
    for task in tasks:
        if task['kind'] not in ('wse2', 'fecral'):
            continue
        for field in ('image', 'label'):
            path = contained_path(ROOT, task['input'][field])
            require(sha256_file(path) == task['input'][field + '_sha256'], 'Corpus source drift')
            prefix = 'atomic_defects/WSe2' if task['kind'] == 'wse2' else 'metal_alloy_defects/FeCrAl'
            name = f'EM_data/{prefix}/{field}/{path.name}'
            item = inventory.get(name)
            payload = path.read_bytes()
            checks.append({'task_id': task['id'], 'field': field, 'member': name,
                           'published_member_exists': item is not None,
                           'same_size_and_zip_crc': item is not None and len(payload) == item['bytes']
                           and zlib.crc32(payload) == item['CRC'],
                           'local_sha256': hashlib.sha256(payload).hexdigest()})
    return {'inventory_sha256': sha256_file(inventory_path), 'checks': checks,
            'training_split_known': False, 'new_unseen_material_claim': False,
            'interpretation': 'WSe2/FeCrAl originate from the EMSeek published corpus; author train/test membership unknown. CRC+size agreement is not a cryptographic remote byte hash.'}


def prepare(output):
    require(not output.exists(), 'Use a new output directory')
    old = base.checked(ROOT / base.DEFAULT)
    require(sha256_file(base.ARCHIVE) == old['archive_sha256'], 'Dense input changed')
    discovery = read(DEST / 'discovery.json')
    weight_receipts = read(DEST / 'weights.json')
    model_checks = []
    torch.set_num_threads(2)
    for name in sorted(EXPECTED):
        model, receipt = load_model(name)
        model_checks.append(receipt)
        del model
    dataset = contained_path(ROOT, old['dataset'])
    manifest, arrays = load_learning_dataset(ROOT, dataset)
    eligible = [e for e in manifest['entries'] if e['action'] == 'correct_geometry' and e['geometry'] in ('point', 'interface')]
    groups = sorted({e['group'] for e in eligible})
    references = {}; buffers = {}
    for index, key in enumerate(groups + ['external-wse2']):
        excluded = [key] if key != 'external-wse2' else []
        bank, audit = training_references(eligible, arrays, excluded)
        references[key] = {'excluded_groups': excluded, 'selection': 'existing deterministic three-training-positive-exemplars per geometry; no query labels',
                           'audit': audit, 'arrays': {str(g): f'reference-{index}-{g}' for g in bank}}
        for geometry, value in bank.items():
            # Frozen learning images are [0,1]. Re-encode each reference as RGB-compatible uint8.
            buffers[references[key]['arrays'][str(geometry)]] = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
    tasks = old['tasks']; jobs = []
    for task in tasks:
        for variant, weight, refkey in variants_for_task(task):
            geometry = (next(e['geometry_index'] for e in eligible if e['case_number'] == task['case_number'])
                        if task['kind'] == 'expert' else 1)
            jobs.append({'index': len(jobs), 'id': task['id'] + '--' + variant, 'task': task,
                         'variant': variant, 'weight': weight, 'reference_key': refkey, 'geometry_index': geometry})
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'references.npz').open('xb') as stream:
        np.savez_compressed(stream, **buffers)
    overlap = corpus_overlap(tasks)
    write_new_json(output / 'corpus_overlap.json', overlap)
    protocol = {'format': 'emseek-author-direct-pretrained-v2', 'registered_at_utc': datetime.now(timezone.utc).isoformat(),
                'jobs': jobs, 'references': references, 'references_sha256': sha256_file(output / 'references.npz'),
                'dataset': old['dataset'], 'dataset_manifest_sha256': sha256_file(dataset / 'manifest.json'),
                'base_protocol': str((ROOT / base.DEFAULT / 'protocol.json').relative_to(ROOT)),
                'base_protocol_sha256': sha256_file(ROOT / base.DEFAULT / 'protocol.json'),
                'best_release_manifest_sha256': sha256_file(ROOT / 'outputs/best_release_20261014_v1/manifest.json'),
                'archive_sha256': old['archive_sha256'], 'evaluation': old['evaluation'],
                'weight_receipts': weight_receipts, 'model_checks': model_checks, 'discovery': discovery,
                'source_sha256': {n: sha256_file(ROOT / n) for n in SOURCE_NAMES},
                'corpus_overlap_sha256': sha256_file(output / 'corpus_overlap.json'),
                'input': 'same raw image planes as Dennler direct evaluation; PIL RGB conversion; official 1024x1024 resize and ImageNet normalization',
                'native_output': 'official two-class argmax at1024, nearest-resized to source coordinates',
                'score_output': 'foreground softmax, bilinear resized to source coordinates',
                'reference_output': 'arithmetic mean of three train-only reference probabilities; threshold0.5 for ensemble mask; explicitly prompted adaptation',
                'native_point_adapter': 'four-connected component centroids; no component filtering; separate from fixed-budget local maxima',
                'task_selection': 'task2 for atomic defect tasks, task5 only FeCrAl; no task-specific interface checkpoint is published',
                'no_prompt': 'official optional-bbox omitted path; no artificial center prompt',
                'training_performed': False, 'query_labels_used_for_inference': False,
                'query_label_dependent_selection': False, 'llm_agent_platform_evaluated': False,
                'equal_training_resources': False, 'new_blind_test': False,
                'pretraining_overlap': 'WSe2 and FeCrAl are in the author published corpus; exact training split unavailable; expert sources not proven absent from all author training',
                'unrelated_weights_not_evaluated': ['task1: all-atom columns, not defects', 'task3: single-atom catalysts', 'task4: nanoparticles']}
    write_new_json(output / 'protocol.json', protocol)
    print('REGISTERED', len(jobs), 'image/variant conditions from', len(tasks), 'images', flush=True)


def checked(output):
    p = read(output / 'protocol.json')
    for name, digest in p['source_sha256'].items():
        require(sha256_file(contained_path(ROOT, name)) == digest, 'Registered inference code changed')
    require(sha256_file(output / 'references.npz') == p['references_sha256'], 'Reference bank changed')
    require(sha256_file(output / 'corpus_overlap.json') == p['corpus_overlap_sha256'], 'Corpus audit changed')
    require(sha256_file(contained_path(ROOT, p['base_protocol'])) == p['base_protocol_sha256'], 'Source protocol changed')
    require(sha256_file(ROOT / 'outputs/best_release_20261014_v1/manifest.json') == p['best_release_manifest_sha256'], 'Best results changed')
    for receipt in p['weight_receipts']['weights']:
        require(sha256_file(WEIGHTS / receipt['name']) == receipt['sha256'], 'Author weights changed')
    return p


def worker(output, shard, shards):
    p = checked(output)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    cache = {}; failures = []; completed = 0
    with np.load(output / 'references.npz', allow_pickle=False) as refs, ZipFile(base.ARCHIVE) as archive:
        for job in p['jobs'][shard::shards]:
            target = output / 'predictions' / job['id']
            require(not target.exists(), 'Existing condition preserved; no implicit rerun')
            target.mkdir(parents=True)
            try:
                if job['weight'] not in cache:
                    cache[job['weight']] = load_model(job['weight'], 'cuda:0')
                model, mapping = cache[job['weight']]
                image = base.infer_image(job['task'], archive)
                begin = time.monotonic()
                if job['reference_key'] is None:
                    values, mask = predict(model, image, device='cuda:0')
                else:
                    meta = p['references'][job['reference_key']]
                    require(job['task'].get('group') not in {a['group'] for a in meta['audit']}, 'Reference leakage')
                    patches = refs[meta['arrays'][str(job['geometry_index'])]]
                    values = np.mean([predict(model, image, patch=patch, device='cuda:0')[0] for patch in patches], axis=0)
                    mask = (values >= 0.5).astype(np.uint8)
                with (target / 'prediction.npz').open('xb') as stream:
                    np.savez_compressed(stream, score=values, mask=mask)
                write_new_json(target / 'receipt.json', {'job': job, 'status': 'success', 'shape': list(image.shape[:2]),
                               'model': mapping, 'seconds': time.monotonic() - begin, 'query_labels_read': False,
                               'protocol_sha256': sha256_file(output / 'protocol.json'),
                               'prediction_sha256': sha256_file(target / 'prediction.npz')})
                completed += 1
                print('DONE', shard, completed, job['id'], flush=True)
            except Exception as error:
                failures.append(job['id'])
                write_new_json(target / 'failure.json', {'job': job, 'error': repr(error)})
                print('FAILED', job['id'], repr(error), flush=True)
                break
    write_new_json(output / f'shard-{shard:02d}.json', {'shard': shard, 'completed': completed, 'failed': failures})
    if failures:
        raise RuntimeError('Author inference failed; partial outputs preserved')


def launch(output, gpu_ids):
    p = checked(output); idle_gpus(gpu_ids)
    logs = output / 'workers'; logs.mkdir(exist_ok=False)
    active = []; journal = []
    for shard, gpu in enumerate(gpu_ids):
        log = (logs / f'shard-{shard:02d}.log').open('x')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
               'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'TMPDIR': str(ROOT / '.codex-tmp'), 'TORCH_HOME': str(ROOT / '.codex-tmp/torch')}
        command = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker', '--output', str(output),
                   '--shard', str(shard), '--shards', str(len(gpu_ids))]
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active.append((proc, log)); journal.append({'pid': proc.pid, 'gpu': gpu, 'shard': shard, 'command': command})
    write_new_json(logs / 'processes.json', {'workers': journal, 'protocol_sha256': sha256_file(output / 'protocol.json')})
    while any(proc.poll() is None for proc, _ in active):
        count = len(list((output / 'predictions').glob('*/receipt.json'))) if (output / 'predictions').exists() else 0
        print('PROGRESS', count, '/', len(p['jobs']), flush=True)
        time.sleep(10)
    codes = [proc.wait() for proc, _ in active]
    for _, log in active:
        log.close()
    write_new_json(output / 'execution.json', {'exitcodes': codes})
    require(not any(codes), 'At least one shard failed')


def point_metrics(score, mask, truth, ep):
    points = student_points(score, ep.point_budget(score.shape), ep.point_min_distance_px)
    labels, count = ndimage.label(mask > 0)
    centers = (np.asarray(ndimage.center_of_mass(mask > 0, labels, range(1, count + 1))).reshape(-1, 2)[:, ::-1]
               if count else np.empty((0, 2)))
    return {name: {'matched': len(matched_expert_indices(value, truth, ep.point_tolerance_px)),
                   'predicted': len(value), 'truth': len(truth)}
            for name, value in [('budgeted_score_points', points), ('native_component_points', centers)]}


def compute_results(output):
    p = checked(output)
    require(not any(read(output / 'execution.json')['exitcodes']), 'Incomplete workers')
    manifest, arrays = load_learning_dataset(ROOT, contained_path(ROOT, p['dataset']))
    require(sha256_file(contained_path(ROOT, p['dataset']) / 'manifest.json') == p['dataset_manifest_sha256'], 'Labels changed')
    entries = {e['case_number']: e for e in manifest['entries']}
    ep = EvaluationProtocol(**p['evaluation']); expert = []; dense = []; wse = []; normals = []
    with ZipFile(base.ARCHIVE) as archive:
        for job in p['jobs']:
            task = job['task']; target = output / 'predictions' / job['id']; receipt = read(target / 'receipt.json')
            require(receipt['job'] == job and receipt['status'] == 'success' and not receipt['query_labels_read'], 'Wrong inference identity')
            require(receipt['protocol_sha256'] == sha256_file(output / 'protocol.json'), 'Wrong prediction protocol')
            require(sha256_file(target / 'prediction.npz') == receipt['prediction_sha256'], 'Prediction drift')
            with np.load(target / 'prediction.npz', allow_pickle=False) as values:
                score, mask = values['score'], values['mask'] > 0
            tag = {'id': task['id'], 'variant': job['variant'], 'kind': task['kind']}
            if task['kind'] == 'expert':
                entry = entries[task['case_number']]; geometry = entry['geometry']
                if geometry == 'point':
                    prediction = Prediction(geometry, points_xy=student_points(score, ep.point_budget(score.shape), ep.point_min_distance_px), budget=ep.point_budget(score.shape))
                else:
                    prediction = Prediction(geometry, mask=student_mask(score, ep.area_budget(score.shape)), budget=ep.area_budget(score.shape))
                predictions = {'equal_budget': prediction}
                if geometry != 'point':
                    predictions['native_region'] = Prediction(geometry, mask=mask)
                metric = evaluate_case(entry, arrays[task['case_number']], predictions, protocol=ep)
                expert.append({**tag, **metric}); continue
            if task['kind'] == 'wse2':
                path = contained_path(ROOT, task['input']['label'])
                require(sha256_file(path) == task['input']['label_sha256'], 'WSe2 labels changed')
                truth = base.label_points(path)
                wse.append({**tag, 'methods': point_metrics(score, mask, truth, ep)})
                truth_mask = np.asarray(Image.open(path).convert('RGB')).any(-1)
                dense.append({**tag, 'border': 0, 'subset': 'author_published_corpus', 'metrics': _counts_to_metrics(_binary_counts(mask, truth_mask))})
                continue
            if task['kind'] == 'dennler':
                _, truth = base.load_image(archive, task['input'], truth=True)
            else:
                path = contained_path(ROOT, task['input']['label'])
                require(sha256_file(path) == task['input']['label_sha256'], 'FeCrAl labels changed')
                truth = np.asarray(Image.open(path).convert('L')) > 0
            for border in ([0, 12] if task['kind'] == 'dennler' else [0]):
                sl = (slice(border, -border), slice(border, -border)) if border else (slice(None), slice(None))
                dense.append({**tag, 'border': border, 'subset': task['input'].get('author_split', 'author_published_corpus'),
                              'metrics': _counts_to_metrics(_binary_counts(mask[sl], truth[sl] > 0))})
            if task['kind'] == 'dennler' and task['input']['category'] == 'normal':
                normals.append({**tag, 'false_positive_area': float(mask.mean())})
    cover = defaultdict(list); areas = defaultdict(list); counts = defaultdict(lambda: defaultdict(int)); point_counts = defaultdict(lambda: defaultdict(int))
    for record in expert:
        for method, metric in record['methods'].items():
            key = (record['group'], record['variant'], method)
            cover[key].append(metric['positive_coverage'])
            areas[key].append(metric.get('selected_area_fraction', 0))
    for record in dense:
        for subset in ['all', record['subset']]:
            for key, value in record['metrics'].items():
                if key.endswith('_pixels'):
                    counts[record['kind'], record['variant'], record['border'], subset][key] += value
    for record in wse:
        for method, values in record['methods'].items():
            for key, value in values.items():
                point_counts[record['variant'], method][key] += value
    point_summary = []
    for (variant, method), values in point_counts.items():
        tp, predicted, truth = values['matched'], values['predicted'], values['truth']
        point_summary.append({'variant': variant, 'method': method, **values,
                              'precision': tp / max(1, predicted), 'recall': tp / max(1, truth), 'f1': 2 * tp / max(1, predicted + truth)})
    return {'protocol_sha256': sha256_file(output / 'protocol.json'), 'successful_conditions': len(p['jobs']),
            'expert_summary': [{'group': key[0], 'variant': key[1], 'method': key[2], 'cases': len(values),
                                'mean': float(np.mean(values)), 'mean_selected_area': float(np.mean(areas[key]))} for key, values in cover.items()],
            'dense_summary': [{'kind': key[0], 'variant': key[1], 'border': key[2], 'subset': key[3], 'metrics': _counts_to_metrics(dict(value))} for key, value in counts.items()],
            'wse2_summary': point_summary, 'normal_cases': normals, 'expert_cases': expert, 'dense_cases': dense, 'wse2_cases': wse}


def score_and_seal(output):
    result = compute_results(output)
    write_new_json(output / 'results.json', result)
    write_new_json(output / 'seal.json', {'files': {str(path.relative_to(output)): sha256_file(path) for path in output.rglob('*') if path.is_file()}})
    print(json.dumps({k: v for k, v in result.items() if k.endswith('summary')}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=['prepare', 'launch', 'worker', 'score', 'verify'])
    parser.add_argument('--output', default=DEFAULT); parser.add_argument('--gpus', default='2,3,4,5,6,7')
    parser.add_argument('--shard', type=int); parser.add_argument('--shards', type=int)
    args = parser.parse_args(); out = contained_path(ROOT, args.output)
    if args.command == 'prepare': prepare(out)
    elif args.command == 'launch': launch(out, [int(i) for i in args.gpus.split(',')])
    elif args.command == 'worker': worker(out, args.shard, args.shards)
    elif args.command == 'score': score_and_seal(out)
    else:
        require(compute_results(out) == read(out / 'results.json'), 'Metrics no longer match saved results')
        for name, digest in read(out / 'seal.json')['files'].items():
            require(sha256_file(contained_path(out, name)) == digest, 'Sealed evidence changed')
        print('VERIFIED all registered conditions and freshly recomputed metrics', flush=True)
