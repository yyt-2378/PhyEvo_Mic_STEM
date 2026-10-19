#!/usr/bin/env python3
"""Recheck all EMSeek sources/metrics and optionally replay fixed GPU exemplars."""
import argparse
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import torch
from scripts import run_emseek_author_pretrained as run
from scripts.emseek_author_weights import load_model, predict
from scripts.run_rsi_nested import idle_gpus
from scripts.verify_rsi_cycle import require, junit_counts
from tfree_detection.expert_supervision import contained_path, sha256_file, load_learning_dataset, write_new_json


def verify(output, replay_gpu=None, record=False):
    p = run.checked(output)
    expected = run.compute_results(output)
    require(expected == run.read(output / 'results.json'), 'Fresh metric replay differs')
    for name, digest in run.read(output / 'seal.json')['files'].items():
        require(sha256_file(contained_path(output, name)) == digest, 'Sealed evidence changed')
    require(len(p['jobs']) == len({job['id'] for job in p['jobs']}) == 788, 'Condition inventory incomplete')
    require(len({job['task']['id'] for job in p['jobs']}) == 321, 'Image inventory incomplete')
    checks = {row['checkpoint']: row for row in p['model_checks']}
    for job in p['jobs']:
        receipt = run.read(output / 'predictions' / job['id'] / 'receipt.json')
        require(receipt['model'] == checks[job['weight']], 'Loaded model differs from registered author checkpoint')
        with np.load(output / 'predictions' / job['id'] / 'prediction.npz', allow_pickle=False) as array:
            require(array['score'].shape == array['mask'].shape == tuple(receipt['shape']), 'Source-coordinate shape mismatch')
            require(np.isfinite(array['score']).all() and array['score'].min() >= 0 and array['score'].max() <= 1, 'Invalid probability plane')
            require(set(np.unique(array['mask'])) <= {0, 1}, 'Invalid binary segmentation')
    shards = [run.read(output / f'shard-{i:02d}.json') for i in range(6)]
    require(sum(row['completed'] for row in shards) == 788 and all(not row['failed'] for row in shards), 'Incomplete shards')
    dataset = contained_path(run.ROOT, p['dataset']); manifest, arrays = load_learning_dataset(run.ROOT, dataset)
    entries = manifest['entries']; by_number = {e['case_number']: e for e in entries}
    reference_checks = 0
    with np.load(output / 'references.npz', allow_pickle=False) as saved:
        for key, meta in p['references'].items():
            bank, audit = run.training_references(entries, arrays, meta['excluded_groups'])
            require(audit == meta['audit'], 'Reference selection cannot be reconstructed')
            query_files = {e['source_file_sha256'] for e in entries if e['group'] in meta['excluded_groups']}
            for record in audit:
                require(by_number[record['case_number']]['source_file_sha256'] not in query_files, 'Reference reuses a query source file')
            for geometry, value in bank.items():
                expected = np.rint(np.clip(value, 0, 1) * 255).astype(np.uint8)
                require(np.array_equal(expected, saved[meta['arrays'][str(geometry)]]), 'Saved prompt pixels differ')
                reference_checks += len(expected)
    source_hashes = {}
    for job in p['jobs']:
        task = job['task']
        if task['kind'] == 'expert':
            source_hashes[task['sample']['source_path']] = task['source_sha256']
        elif task['kind'] != 'dennler':
            for field in ('image', 'label'):
                source_hashes[task['input'][field]] = task['input'][field + '_sha256']
    for name, digest in source_hashes.items():
        require(sha256_file(contained_path(run.ROOT, name)) == digest, 'Original image or label changed')
    require(sha256_file(run.base.ARCHIVE) == p['archive_sha256'], 'Dense archive changed')
    overlap = run.corpus_overlap([job['task'] for job in p['jobs'] if job['variant'] == 'general_no_prompt'])
    require(overlap == run.read(output / 'corpus_overlap.json'), 'Published-corpus overlap no longer agrees')
    replay = []
    if replay_gpu is not None:
        idle_gpus([replay_gpu]); torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        # First registered condition per cohort/variant, never select by its measured performance.
        selected = {}
        for job in p['jobs']:
            selected.setdefault((job['task']['kind'], job['variant']), job)
        models = {}
        device = f'cuda:{replay_gpu}'
        with np.load(output / 'references.npz', allow_pickle=False) as refs, ZipFile(run.base.ARCHIVE) as archive:
            for job in selected.values():
                if job['weight'] not in models:
                    models[job['weight']], _ = load_model(job['weight'], device)
                model = models[job['weight']]; image = run.base.infer_image(job['task'], archive)
                if job['reference_key'] is None:
                    values, mask = predict(model, image, device=device)
                else:
                    meta = p['references'][job['reference_key']]
                    patches = refs[meta['arrays'][str(job['geometry_index'])]]
                    values = np.mean([predict(model, image, patch=patch, device=device)[0] for patch in patches], axis=0)
                    mask = (values >= 0.5).astype(np.uint8)
                with np.load(output / 'predictions' / job['id'] / 'prediction.npz', allow_pickle=False) as old:
                    require(np.array_equal(old['score'], values) and np.array_equal(old['mask'], mask), 'GPU replay differs')
                replay.append({'id': job['id'], 'bitwise_identical': True})
    tests = junit_counts(output / 'final_tests.xml')
    require(tests == {'tests': 28, 'failures': 0, 'errors': 0, 'skipped': 0}, 'Focused tests incomplete')
    result = {'images': 321, 'successful_conditions': 788, 'all_metrics_recomputed_identically': True,
              'model_checks': checks, 'weight_conditions': dict(Counter(job['weight'] for job in p['jobs'])),
              'reference_images_reconstructed_exactly': reference_checks, 'query_source_overlap_in_references': False,
              'source_files_rehashed': len(source_hashes), 'same_author_corpus_file_checks': len(overlap['checks']),
              'author_corpus_size_crc_matches': sum(row['same_size_and_zip_crc'] for row in overlap['checks']),
              'author_training_split_known': False, 'gpu_sample_replay': replay, 'tests': tests,
              'results_sha256': sha256_file(output / 'results.json'), 'verifier_sha256': sha256_file(Path(__file__))}
    if record:
        write_new_json(output / 'verification.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default=run.DEFAULT)
    parser.add_argument('--replay-gpu', type=int); parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    verify(contained_path(run.ROOT, args.output), args.replay_gpu, args.record)
