"""Seal all composition searches, then execute unchanged held-transform probes.

No candidate selection, fitting, or feedback into the sealed search. These are
held-label-side transformation checks, not independent unseen acquisitions.
"""
import argparse
from pathlib import Path

import numpy as np

from . import composition_study as study
from . import composition_retention_audit as retention
from .contracts import file_hash, write_new
from .evaluation import summarize
from .field_storage import ArrayPool, exact_child
from .prediction_storage import PredictionPool
from .prediction_storage_runtime import SharedPredictionSandbox, prediction_binding


def require_search_open(scope):
    if (scope / 'composition_search_seal.json').exists():
        raise ValueError('Search is sealed; no further proposals or candidate execution')


def implementations():
    from . import composition_continuation
    return prediction_binding() | {
        'composition_study.py': file_hash(study.__file__),
        'composition_retention_audit.py': file_hash(retention.__file__),
        'composition_final_audit.py': file_hash(__file__),
        'composition_continuation.py': file_hash(composition_continuation.__file__),
    }


def search_evidence(config):
    """Bind every attempted proposal, not just the successful trajectories."""
    ledger, selected = {}, {}
    from .composition_continuation import phase_ledger
    ledger.update(phase_ledger())
    for name in ('prospective_protocol.json', 'frozen_measurements.json',
                 'parent_replay_verified_v2.json', 'protected_before.json'):
        ledger[name] = file_hash(study.SCOPE / name)
    fixed = study.read(study.SCOPE / 'frozen_measurements.json')
    frozen_program = study.SCOPE / 'frozen_parent/program'
    frozen_hashes = study.read(study.SCOPE / 'frozen_parent/manifest.json')['exported_hashes']
    for name, expected in frozen_hashes.items():
        if file_hash(frozen_program / name) != expected:
            raise ValueError('Frozen program changed')
    frozen = {'identity': 'frozen', 'program': str(frozen_program.relative_to(study.ROOT)),
              'program_hashes': frozen_hashes, 'state': None,
              'original_summary': summarize(fixed['original'])}
    for arm in config['arms']:
        for seed in config['seeds']:
            key = f'{arm}-s{seed}'
            chosen = frozen
            records = study.history(arm, seed)  # Reject any nonterminal attempt.
            if not records:
                raise ValueError('Every declared trajectory needs actual search evidence')
            for folder, result in records:
                study.checked_proposal(folder, config)
                for name in ('request.json', 'request_binding.json', 'response.json',
                             'execution_started.json', 'result.json', 'result_binding.json'):
                    ledger[str((folder / name).relative_to(study.SCOPE))] = file_hash(folder / name)
                for name, expected in result.get('program_hashes', {}).items():
                    if file_hash(folder / 'program' / name) != expected:
                        raise ValueError('Attempt executable changed')
                state = result.get('state')
                if state and file_hash(exact_child(study.ROOT, study.ROOT / state['path'])) != state['sha256']:
                    raise ValueError('Fitted state changed')
                if result['decision']['accepted']:
                    chosen = {'identity': result['identity'],
                        'program': str((folder / 'program').relative_to(study.ROOT)),
                        'program_hashes': result['program_hashes'], 'state': state,
                        'original_summary': result['summary']}
            selected[key] = chosen
    return {'ledger': ledger, 'selected': selected, 'frozen': frozen}


def seal():
    require_search_open(study.SCOPE)
    config = study.validate_study()
    evidence = search_evidence(config)
    audit_path, audit = retention.audit()
    if not audit['development_prerequisites_met']:
        raise ValueError('Repeated gains and actual later inheritance prerequisites not met')
    value = {'format': 'composition-joint-search-seal-v1', **evidence,
        'retention_audit': str(audit_path.relative_to(study.SCOPE)),
        'retention_audit_sha256': file_hash(audit_path),
        'qualifying_full_seeds': audit['qualifying_full_seeds'],
        'implementations': implementations(),
        'held_back_transforms': config['held_back_transforms'],
        'all_declared_trajectories_included': True,
        'training_allowed_after_seal': False, 'selection_after_audit': False,
        'new_blind_dataset': False, 'goal_fully_achieved': False}
    path = study.SCOPE / 'composition_search_seal.json'
    write_new(path, value)
    print(study.json.dumps({'seal': str(path), 'qualifying': audit['qualifying_full_seeds']}), flush=True)
    return value


def verify_seal():
    config = study.validate_study()
    value = study.read(study.SCOPE / 'composition_search_seal.json')
    if value['implementations'] != implementations():
        raise ValueError('Sealed audit implementation changed')
    actual = search_evidence(config)
    if any(value[k] != actual[k] for k in ('ledger', 'selected', 'frozen')):
        raise ValueError('Search evidence or selected system changed after sealing')
    audit_path = exact_child(study.SCOPE, study.SCOPE / value['retention_audit'])
    audit = study.read(audit_path)
    if (file_hash(audit_path) != value['retention_audit_sha256']
            or not audit['development_prerequisites_met']
            or audit['qualifying_full_seeds'] != value['qualifying_full_seeds']
            or value['held_back_transforms'] != config['held_back_transforms']):
        raise ValueError('Joint seal prerequisites changed')
    return config, value


def held_decision(full, frozen, route, original, protocol):
    """Evaluate a fixed selected system; never choose a winner using these values."""
    if not (set(full) == set(frozen) == set(route) == set(original)):
        raise ValueError('Unmatched held-transform cohorts')
    if not set(protocol.validation_groups) <= set(full):
        raise ValueError('Missing weight-fit-excluded source')
    if not all(np.isfinite(v['mean']) and 0 <= v['mean'] <= 1
               for rows in (full, frozen, route, original) for v in rows.values()):
        raise ValueError('Invalid held-transform coverage')
    gain = lambda ref: float(np.mean([full[g]['mean'] - ref[g]['mean']
                                      for g in protocol.validation_groups]))
    whole_gain = float(np.mean([full[g]['mean'] - frozen[g]['mean'] for g in full]))
    regressions = [{'source': g, 'reference': name,
                    'drop': ref[g]['mean'] - full[g]['mean']}
                   for name, ref in (('original_selected', original), ('frozen_transformed', frozen))
                   for g in full if ref[g]['mean'] - full[g]['mean'] > protocol.maximum_group_regression]
    return {'passed': not regressions and whole_gain >= protocol.minimum_gain
                and gain(frozen) >= protocol.minimum_gain and gain(route) >= protocol.minimum_gain,
            'whole_gain_over_frozen': whole_gain,
            'nontraining_gain_over_frozen': gain(frozen),
            'nontraining_gain_over_route': gain(route), 'regressions': regressions,
            'statistical_significance_claimed': False}


def execute():
    config, sealed = verify_seal()
    destination = exact_child(study.ROOT, study.SCOPE / 'composition_final_audit')
    destination.mkdir(exist_ok=False)  # A partial run is evidence, not permission to overwrite.
    seal_hash = file_hash(study.SCOPE / 'composition_search_seal.json')
    write_new(destination / 'started.json', {'search_seal_sha256': seal_hash,
        'implementations': implementations(), 'training_steps': 0})
    # Keep held-transform staging separate from the immutable development pool.
    # Each pool stays bounded at 256 MiB; the outer study's cumulative 1 GiB
    # storage guard still includes both. No development object is replaced.
    pool = ArrayPool(study.ROOT, destination / 'input_pool',
        max_pool_bytes=256*1024**2, minimum_free_bytes=config['minimum_free_bytes'])
    predictions = PredictionPool(study.ROOT, study.SCOPE / 'prediction_pool',
        max_pool_bytes=config['maximum_prediction_pool_bytes'], minimum_free_bytes=config['minimum_free_bytes'])
    sandbox = SharedPredictionSandbox(study.ROOT, study.ROOT / config['run_configuration']['runtime'], timeout=900,
        prediction_pool=predictions, max_written_prediction_bytes=64*1024**2)
    results, measured, failures = [], 0, []
    for seed in config['seeds']:
        protocol = study.protocol_for(config, seed)
        for transform in config['held_back_transforms']:
            samples = study.samples_for(config, transform)
            systems = {'frozen': sealed['frozen'],
                       'route': sealed['selected'][f'route-s{seed}'],
                       'full': sealed['selected'][f'full-s{seed}']}
            outcomes = {}
            for arm, system in systems.items():
                folder = exact_child(study.ROOT, destination / f'{arm}-s{seed}-{transform}')
                folder.mkdir(exist_ok=False)
                program = exact_child(study.ROOT, study.ROOT / system['program'])
                state = None if system['state'] is None else exact_child(study.ROOT, study.ROOT / system['state']['path'])
                study.stage(folder / 'input', samples, pool, protocol, state=state, fit=False, steps=0)
                print(study.json.dumps({'phase': 'held_transform', 'seed': seed,
                                       'arm': arm, 'transform': transform}), flush=True)
                execution = sandbox.run(program, folder / 'input', folder / 'output')
                value = {'identity': system['identity'], 'execution': execution,
                         'training_steps': 0, 'actual_execution': True,
                         'same_acquisition_transform_not_new_source': True}
                if execution['status'] == 'ok':
                    rows = study.measure(folder / 'output', samples, protocol)
                    value.update(rows=rows, summary=summarize(rows))
                    measured += len(rows)
                else:
                    failures.append(str(folder.relative_to(study.SCOPE)))
                write_new(folder / 'result.json', value)
                outcomes[arm] = value
            decision = (held_decision(outcomes['full']['summary'], outcomes['frozen']['summary'],
                         outcomes['route']['summary'], systems['full']['original_summary'], protocol)
                        if all('summary' in v for v in outcomes.values()) else
                        {'passed': False, 'reason': 'actual_execution_failed'})
            results.append({'seed': seed, 'transform': transform, 'decision': decision})
    verify_seal()
    if file_hash(study.SCOPE / 'composition_search_seal.json') != seal_hash:
        raise ValueError('Search seal changed during final execution')
    passed = [seed for seed in sealed['qualifying_full_seeds']
              if all(r['decision']['passed'] for r in results if r['seed'] == seed)]
    value = {'search_seal_sha256': seal_hash, 'all_results': results,
        'actual_prediction_arrays': measured, 'failed_executions': failures,
        'qualifying_seeds_passing_all_final_transforms': passed,
        'numerical_completion_prerequisites_met': not failures and len(passed) >=
            config['success']['independent_full_trajectories'],
        'goal_fully_achieved': False, 'requires_requirement_by_requirement_handoff_audit': True,
        'new_blind_dataset': False, 'no_final_result_used_for_selection_or_training': True,
        'production_promoted': False, 'protected_check': study.validate_study() == config}
    write_new(destination / 'results.json', value)
    print(study.json.dumps({'final_audit': str(destination / 'results.json'),
                           'passed_seeds': passed, 'actual_inferences': measured}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('seal', 'execute'))
    parser.add_argument('--run-config', required=True)
    args = parser.parse_args()
    study.configure_run(args.run_config)
    seal() if args.command == 'seal' else execute()
