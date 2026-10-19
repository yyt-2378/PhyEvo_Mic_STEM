"""Actual bounded replay of the exported parent, not another RSI proposal."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .contracts import file_hash, protected_snapshot, verify_snapshot, write_new
from .field_storage import ArrayPool, allocated_bytes, exact_child
from .field_storage_runtime import FieldStorageSandbox
from .study_io import load_study
from .retained_portfolio import ROOT, AUTHORIZED_SOURCE, ACCEPTED_HASH, export
from .stability_diagnostics import transform_array


def historical_route(source, relative, branch, observed_hashes):
    """Stress summaries omit prose; read the actual route, bind its current bytes."""
    path = exact_child(source, source / relative)
    payload = path.read_bytes()
    value = json.loads(payload)
    if value.get('branch') != branch or not isinstance(value.get('reason'), str):
        raise ValueError('Historical route disagrees with the sealed selected branch')
    observed_hashes[relative] = hashlib.sha256(payload).hexdigest()
    return value


def replay(destination):
    destination = exact_child(ROOT, destination)
    if not destination.is_dir() or not (destination / 'RUN_SCOPE.json').exists():
        raise ValueError('Declared exclusive scope required')
    source = AUTHORIZED_SOURCE
    config = json.loads((source.parent / 'protocol.json').read_text())
    protected = protected_snapshot(ROOT)
    write_new(destination / 'protected_before.json', protected)
    frozen = destination / 'frozen_parent'
    bundle = export(ROOT, source, 'requests/r06/result.json', ACCEPTED_HASH, frozen)
    candidate = bundle['source_candidate']
    manifest_path = ROOT / 'outputs/rsi_v4_storage_migration_20261017_v1/cleanup_plan_20261018/retained_dependencies.json'
    retention = json.loads(manifest_path.read_text())['verified_core_files']
    reference_hashes = {}

    def reference(relative, expected=None):
        p = exact_child(source, source / relative)
        rel = str(p.relative_to(source.parent))
        expected = expected or retention.get(rel)
        # Retention records must bind every reference; no best-score reselection.
        if isinstance(expected, dict):
            expected = expected['sha256']
        if not expected or file_hash(p) != expected:
            raise ValueError('Historical reference hash mismatch: ' + rel)
        reference_hashes[relative] = expected
        return p

    fields = exact_child(ROOT, ROOT / config['fields'])
    if file_hash(fields / 'dataset.json') != config['fields_manifest_sha256']:
        raise ValueError('Frozen input manifest changed')
    _, _, samples = load_study(fields)
    original = json.loads(reference(f'system_jobs/{candidate}-development/result.json').read_text())
    expected_keys = [r['case'] for r in original['rows']]
    by_key = {s.key: s for s in samples}
    if set(by_key) != set(expected_keys) or len(expected_keys) != len(set(expected_keys)):
        raise ValueError('Historical original cohort does not match fields')
    pool = ArrayPool(ROOT, destination / 'field_pool', max_pool_bytes=256 * 1024**2,
                     minimum_free_bytes=8 * 1024**3)
    modes = [('original', 'development', False, None),
             ('reverse', 'development', True, None),
             ('rot90', 'stability-rot90', False, 'rot90')]
    results = []
    for name, historical_split, reverse, transform in modes:
        historical = original if historical_split == 'development' else json.loads(
            reference(f'system_jobs/{candidate}-{historical_split}/result.json').read_text())
        hrows = historical['rows']
        if len(hrows) != len(expected_keys):
            raise ValueError('Transform cohort count changed')
        jobs = list(enumerate(expected_keys))
        if reverse:
            jobs.reverse()
        inputs = destination / (name + '_input'); inputs.mkdir()
        records = []
        expected = []
        for index, (old_index, key) in enumerate(jobs):
            sample = by_key[key]
            fields_to_stage = dict(sample.fields)
            if transform:
                fields_to_stage = {k: np.ascontiguousarray(transform_array(v, transform))
                                   for k, v in fields_to_stage.items()}
            if any(k in fields_to_stage for k in ('human', 'known', 'points_xy', 'label')):
                raise ValueError('Labels entered inference inputs')
            records.append(pool.stage(inputs, f'sample-{index:04d}.fields.json', fields_to_stage))
            old_row = hrows[old_index]
            # Retain the original per-case order, not an answer-selected subset.
            if transform and not old_row['case'].endswith(key):
                raise ValueError('Transform case mapping is not the declared original order')
            ref = historical['predictions'][old_row['case']]
            expected.append((key, old_row, reference(ref['path'], ref['sha256'])))
        write_new(inputs / 'request.json', {'mode': 'predict', 'seed': 41, 'training_steps': 0,
                  'samples': records, 'state_file': False, 'device': 'cpu'})
        output = destination / (name + '_output')
        print(json.dumps({'phase': name, 'cases': len(records),
                          'scope_allocated_bytes': allocated_bytes(destination)}), flush=True)
        receipt = FieldStorageSandbox(ROOT, ROOT / '.venvs/sam2', timeout=900).run(
            frozen / 'program', inputs, output)
        if receipt['status'] != 'ok':
            write_new(destination / (name + '_failed.json'), {'execution': receipt,
                      'stderr_tail': (output / 'stderr.log').read_text()[-4000:]})
            raise RuntimeError('Isolated retained portfolio execution failed')
        rows = []
        failures = []
        for index, (key, old, expected_path) in enumerate(expected):
            path = output / f'prediction-{index:04d}.npz'
            with np.load(path, allow_pickle=False) as new, np.load(expected_path, allow_pickle=False) as previous:
                equal = {k: np.array_equal(new[k], previous[k]) for k in ('score', 'support')}
                maximum_difference = float(np.max(np.abs(new['score'].astype(float) - previous['score'].astype(float))))
            route = json.loads((output / f'route-{index:04d}.json').read_text())
            old_index = expected_keys.index(key)
            expected_route = historical_route(source,
                f'system_jobs/{candidate}-{historical_split}/output/route-{old_index:04d}.json',
                old['branch'], reference_hashes)
            route_equal = route == expected_route
            if not all(equal.values()) or not route_equal:
                failures.append({'case': key, 'equal': equal, 'route_equal': route_equal,
                                 'maximum_score_difference': maximum_difference})
            rows.append({'case': key, 'reference_sha256': file_hash(expected_path),
                         'prediction_sha256': file_hash(path), 'equal_arrays': equal,
                         'equal_route': route_equal, 'branch': route['branch']})
        report = {'mode': name, 'cases': len(rows), 'actual_execution': True,
                  'all_arrays_and_routes_equal': not failures, 'failures': failures,
                  'rows': rows, 'execution': receipt,
                  'reference_summary': historical['summary'], 'new_scientific_gain': False}
        write_new(destination / (name + '_verification.json'), report)
        results.append({k: v for k, v in report.items() if k not in ('rows', 'execution')})
        if failures:
            raise ValueError('Portfolio packaging is not equivalent; do not use as a new baseline')
    for path, expected in bundle['exported_hashes'].items():
        if file_hash(frozen / 'program' / path) != expected:
            raise ValueError('Exported program changed during replay')
    for relative, expected in bundle['source_hashes'].items():
        if file_hash(source / relative) != expected:
            raise ValueError('Source program changed during replay')
    for relative, expected in reference_hashes.items():
        if file_hash(source / relative) != expected:
            raise ValueError('Historical prediction changed during replay')
    verified = verify_snapshot(ROOT, protected)
    final = {'status': 'verified_parent_ready_for_new_study', 'source_candidate': candidate,
             'new_scientific_gain': False, 'new_research_proposals': 0,
             'source_code_and_arrays_unchanged': True, 'original_all_cases_replayed': len(samples),
             'total_actual_case_inferences': sum(r['cases'] for r in results),
             'source_accepted_result_sha256': ACCEPTED_HASH, 'mode_results': results,
             'input_pool_bytes': allocated_bytes(pool.root),
             'whole_scope_allocated_bytes': allocated_bytes(destination),
             'protected_check': verified, 'new_program_manifest_sha256': file_hash(frozen / 'manifest.json'),
             'new_study_must_freeze_protocol_before_any_proposals': True,
             'preexisting_final_transforms_not_evaluated': True}
    write_new(destination / 'parent_replay_completed.json', final)
    print(json.dumps({k: v for k, v in final.items() if k != 'mode_results'}), flush=True)


def verify_existing(destination):
    """Resume only postprocessing of already completed, recorded executions."""
    destination = exact_child(ROOT, destination)
    frozen = destination / 'frozen_parent'
    bundle = json.loads((frozen / 'manifest.json').read_text())
    source = AUTHORIZED_SOURCE
    if bundle['source_episode'] != str(source) or bundle['source_accepted_result'] != 'requests/r06/result.json':
        raise ValueError('Unexpected parent provenance')
    if file_hash(source / bundle['source_accepted_result']) != ACCEPTED_HASH:
        raise ValueError('Accepted source changed')
    for name, h in bundle['exported_hashes'].items():
        if file_hash(frozen / 'program' / name) != h:
            raise ValueError('Exported program changed')
    for rel, h in bundle['source_hashes'].items():
        if file_hash(exact_child(source, source / rel)) != h:
            raise ValueError('Retained source changed')
    candidate = bundle['source_candidate']
    retention = json.loads((ROOT / 'outputs/rsi_v4_storage_migration_20261017_v1/cleanup_plan_20261018/retained_dependencies.json').read_text())['verified_core_files']
    observed_hashes = {}
    checked_rows = []
    summaries = {}
    execution_bindings = {}
    for mode, split, reverse in [('original', 'development', False),
                                 ('reverse', 'development', True),
                                 ('rot90', 'stability-rot90', False)]:
        reference_path = source / f'system_jobs/{candidate}-{split}/result.json'
        h = retention[str(reference_path.relative_to(source.parent))]
        if isinstance(h, dict):
            h = h['sha256']
        if file_hash(reference_path) != h:
            raise ValueError('Bound historical result changed')
        historical = json.loads(reference_path.read_text())
        observed_hashes[str(reference_path.relative_to(source))] = h
        inputs, output = destination / (mode + '_input'), destination / (mode + '_output')
        execution = json.loads((output / 'execution.json').read_text())
        if execution['status'] != 'ok' or execution['returncode'] != 0:
            raise ValueError('Inference has not successfully completed; do not infer completion from file presence')
        execution_bindings[mode] = {'execution_sha256': file_hash(output / 'execution.json'),
                                   'worker_sha256': execution['worker_sha256'],
                                   'storage_runtime_sha256': file_hash(output / 'field_storage_runtime.json')}
        binding = json.loads((output / 'field_storage_runtime.json').read_text())
        for name, expected in binding['input_request_and_manifest_sha256'].items():
            if file_hash(exact_child(inputs, inputs / name)) != expected:
                raise ValueError('Executed input manifest changed')
        request = json.loads((inputs / 'request.json').read_text())
        indexes = list(range(len(historical['rows'])))
        if reverse:
            indexes.reverse()
        if len(request['samples']) != len(indexes) or len(indexes) != 51:
            raise ValueError('Full historical cohort required')
        for index, old_index in enumerate(indexes):
            old = historical['rows'][old_index]
            ref = historical['predictions'][old['case']]
            expected_path = exact_child(source, source / ref['path'])
            if file_hash(expected_path) != ref['sha256']:
                raise ValueError('Historical prediction hash changed')
            observed_hashes[ref['path']] = ref['sha256']
            actual_path = output / f'prediction-{index:04d}.npz'
            with np.load(actual_path, allow_pickle=False) as new, np.load(expected_path, allow_pickle=False) as previous:
                if not all(np.array_equal(new[k], previous[k]) for k in ('score', 'support')):
                    raise ValueError('Actual inference changed: ' + old['case'])
            expected_route = historical_route(source,
                f'system_jobs/{candidate}-{split}/output/route-{old_index:04d}.json',
                old['branch'], observed_hashes)
            route = json.loads((output / f'route-{index:04d}.json').read_text())
            if route != expected_route:
                raise ValueError('Actual route changed: ' + old['case'])
            checked_rows.append({'mode': mode, 'case': old['case'], 'branch': route['branch'],
                                 'new_prediction_sha256': file_hash(actual_path),
                                 'reference_prediction_sha256': ref['sha256']})
        summaries[mode] = historical['summary']
        print(json.dumps({'verified_mode': mode, 'cases': len(indexes), 'arrays_and_routes_equal': True}), flush=True)
    for rel, h in observed_hashes.items():
        if file_hash(source / rel) != h:
            raise ValueError('Read-only reference changed during validation')
    protected = json.loads((destination / 'protected_before.json').read_text())
    report = {'status': 'verified_parent_ready_for_new_study', 'source_candidate': candidate,
              'actual_completed_inferences_verified': len(checked_rows), 'inference_rerun_for_this_verification': False,
              'new_research_proposals': 0, 'new_scientific_gain': False,
              'all_arrays_and_full_route_records_equal': True, 'source_code_and_arrays_unchanged': True,
              'source_accepted_result_sha256': ACCEPTED_HASH, 'execution_bindings': execution_bindings,
              'reference_summaries': summaries, 'rows': checked_rows,
              'protected_check': verify_snapshot(ROOT, protected),
              'scope_allocated_bytes': allocated_bytes(destination),
              'reason_field_compatibility': 'Stress rows omit reason; raw historical route JSON is compared. '
                  'Scientific selected-branch and prediction hashes stay bound to the original result.',
              'original_failed_controller_receipt_preserved': True,
              'frozen_final_transforms_not_evaluated': True, 'production_promoted': False}
    write_new(destination / 'parent_replay_verified_v2.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'reference_summaries', 'execution_bindings')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--verify-existing', action='store_true')
    args = parser.parse_args()
    (verify_existing if args.verify_existing else replay)(args.destination)
