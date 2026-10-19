"""Recompute development evidence and inspect actual compositional inheritance.

This is not final held-transform evaluation and cannot mark the goal complete.
It reads predictions, programs and worker traces, not just an 'accepted' label.
"""
from collections import Counter

import numpy as np

from . import composition_study as study
from .contracts import file_hash, write_new
from .evaluation import summarize
from .expansion_state import expansion_admission
from .stability_state import stability_admission


def domain_field_provenance(config, samples):
    """Match input domain IDs to the pre-existing image-only physical output."""
    physical = study.ROOT / 'outputs/interface_improvement_20261015_v1'
    manifest = study.read(physical / 'physical_audit.json')
    generators = ('scripts/run_interface_improvement.py', 'src/tfree_detection/tasks/interface.py')
    for name in generators:
        if file_hash(study.ROOT / name) != manifest['source_hashes'][name]:
            raise ValueError('Physical domain generator differs from its original manifest')
    entries = {r['key']: r for r in study.read(study.ROOT / config['fields'] / 'dataset.json')['samples']}
    for sample in samples:
        original = physical / 'physical' / (sample.key + '.npz')
        if file_hash(original) != entries[sample.key]['physical_sha256']:
            raise ValueError('Bound physical input changed')
        with np.load(original, allow_pickle=False) as values:
            if not np.array_equal(sample.fields['domain_labels'], values['labels']):
                raise ValueError('Domain field does not reproduce image-derived physical labels')
    return {'cases_checked': len(samples), 'all_domain_arrays_exact': True,
            'generator_hashes_match_original_manifest': True,
            'generator_files': list(generators),
            'physical_manifest_sha256': file_hash(physical / 'physical_audit.json'),
            'meaning': 'image-derived reciprocal-signature clusters, not expert annotations'}


def trace_calls(folder, result, mode):
    """Count recorded calls; a declared parent alone is not execution evidence."""
    calls = Counter()
    for index in range(len(result[mode]['rows'])):
        trace = study.read(folder / (mode + '_output') / f'composition-trace-{index:04d}.json')
        if trace['node'] != result['identity']:
            raise ValueError('Execution trace belongs to another candidate')
        for call in trace['calls']:
            if call['call'] == 'parent' and call['node'] != 'frozen':
                calls[call['node']] += 1
    return dict(calls)


def trajectory_requirements(attempts, criteria):
    accepted = [r for r in attempts if r['accepted']]
    later_use = []
    for earlier in accepted:
        witnesses = [r['round'] for r in attempts if r['round'] > earlier['round']
                     and r['ancestor_execution_calls'].get(earlier['identity'], 0) > 0]
        later_use.append({'identity': earlier['identity'], 'round': earlier['round'],
                          'later_executed_rounds': witnesses})
    last = accepted[-1] if accepted else None
    return {'accepted_updates': len(accepted), 'later_use': later_use,
            'latest_accepted': None if last is None else last['identity'],
            'qualifies_for_final_audit': bool(
                len(accepted) >= criteria['accepted_updates_per_successful_trajectory']
                and all(r['later_executed_rounds'] for r in later_use)
                and last['nontraining_gain_over_frozen'] >=
                    criteria['minimum_nontraining_gain_over_frozen_and_route']
                and last['nontraining_gain_over_route'] >=
                    criteria['minimum_nontraining_gain_over_frozen_and_route']
                and last['nontraining_gain_over_old_ceiling'] >=
                    criteria['minimum_new_capacity_over_old_casewise_ceiling'])}


def audit():
    config = study.validate_study()
    fixed = study.read(study.SCOPE / 'frozen_measurements.json')
    cohorts = {mode: study.samples_for(config, None if mode == 'original' else mode)
               for mode in ('original', 'rot90')}
    provenance = domain_field_provenance(config, cohorts['original'])
    ceilings = {r['case']: r['ceiling'] for r in fixed['ceiling']}
    summaries = []
    pending = []
    remeasured_predictions = 0
    # Controls first, so comparisons use actually admitted route systems.
    route_scores = {}
    for arm in ('route', 'full'):
        for seed in config['seeds']:
            protocol = study.protocol_for(config, seed)
            frozen = [dict(r, seed=seed) for r in fixed['original']]
            frozen_summary = summarize(frozen)
            current, parent = frozen, 'frozen'
            best = {g: v['mean'] for g, v in frozen_summary.items()}
            rotated_best = {g: v['mean'] for g, v in summarize(fixed['rot90']).items()}
            attempts = []
            for folder in sorted((study.SCOPE / f'{arm}-s{seed}').glob('r[0-9][0-9]')):
                number = int(folder.name[1:])
                if not (folder / 'result_binding.json').exists():
                    pending.append(str(folder.relative_to(study.SCOPE)))
                    continue
                result = study.verified_result(folder)
                binding, _ = study.checked_proposal(folder, config)
                if result['parent'] != parent or binding['parent'] != parent:
                    raise ValueError('Attempt did not inherit the last admitted state')
                if result['status'] == 'execution_failed':
                    if result['decision']['accepted']:
                        raise ValueError('Failed execution cannot be admitted')
                    attempts.append({'round': number, 'identity': result['identity'],
                        'accepted': False, 'ancestor_execution_calls': {}, 'execution_failed': True})
                    continue
                for name, expected in result['program_hashes'].items():
                    if file_hash(folder / 'program' / name) != expected:
                        raise ValueError('Executable or ancestor package changed')
                if result['state'] and file_hash(study.ROOT / result['state']['path']) != result['state']['sha256']:
                    raise ValueError('Numerical state changed')
                probes, counts = {}, Counter()
                for mode in ('original', 'rot90'):
                    output = folder / (mode + '_output')
                    if study.read(output / 'execution.json')['status'] != 'ok':
                        raise ValueError('Successful worker receipt required')
                    rows = study.measure(output, cohorts[mode], protocol)
                    if rows != result[mode]['rows'] or summarize(rows) != result[mode]['summary']:
                        raise ValueError('Stored metrics differ from actual prediction arrays')
                    probes[mode] = dict(result[mode], rows=rows)
                    counts.update(trace_calls(folder, result, mode))
                    remeasured_predictions += len(rows)
                quality = expansion_admission(probes['original']['rows'], current, frozen, protocol, best)
                decision = stability_admission(quality, probes['original']['rows'],
                    {'rot90': probes['rot90']}, ('rot90',), protocol.maximum_group_regression,
                    {'rot90': rotated_best})
                if decision != result['decision']:
                    raise ValueError('Admission decision does not reproduce')
                measured = summarize(probes['original']['rows'])
                novel = {g: float(np.mean([r['coverage'] - ceilings[r['case']]
                    for r in probes['original']['rows'] if r['group'] == g])) for g in measured}
                if novel != result['new_capacity']['gain_over_conservative_old_casewise_ceiling']:
                    raise ValueError('Old-library capacity comparison changed')
                references = route_scores.get(seed, frozen_summary)
                row = {'round': number, 'identity': result['identity'], 'parent': parent,
                    'accepted': decision['accepted'], 'ancestor_execution_calls': dict(counts),
                    'nontraining_gain_over_frozen': float(np.mean([
                        measured[g]['mean']-frozen_summary[g]['mean'] for g in protocol.validation_groups])),
                    'nontraining_gain_over_route': float(np.mean([
                        measured[g]['mean']-references[g]['mean'] for g in protocol.validation_groups])),
                    'nontraining_gain_over_old_ceiling': float(np.mean([
                        novel[g] for g in protocol.validation_groups])),
                    'result_sha256': file_hash(folder / 'result.json')}
                attempts.append(row)
                if decision['accepted']:
                    reverse = folder / 'reverse_output'
                    if study.read(reverse / 'execution.json')['status'] != 'ok':
                        raise ValueError('Accepted update lacks successful actual reverse replay')
                    for i in range(len(cohorts['original'])):
                        with np.load(folder / 'original_output' / f'prediction-{i:04d}.npz', allow_pickle=False) as a:
                            with np.load(reverse / f'prediction-{len(cohorts["original"])-1-i:04d}.npz', allow_pickle=False) as b:
                                if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                                    raise ValueError('Accepted reverse replay differs')
                    remeasured_predictions += len(cohorts['original'])
                    current, parent = probes['original']['rows'], result['identity']
                    for g in best:
                        best[g] = max(best[g], measured[g]['mean'])
                        rotated_best[g] = max(rotated_best[g], summarize(probes['rot90']['rows'])[g]['mean'])
            if arm == 'route':
                route_scores[seed] = summarize(current)
            summaries.append({'arm': arm, 'seed': seed, 'attempts': attempts,
                              **trajectory_requirements(attempts, config['success'])})
    qualifying = [r['seed'] for r in summaries if r['arm'] == 'full' and r['qualifies_for_final_audit']]
    value = {'protocol_sha256': file_hash(study.SCOPE / 'prospective_protocol.json'),
        'auditor_sha256': file_hash(__file__), 'trajectories': summaries,
        'pending_attempts': pending, 'remeasured_prediction_arrays': remeasured_predictions,
        'qualifying_full_seeds': qualifying,
        'development_prerequisites_met': not pending and len(qualifying) >=
            config['success']['independent_full_trajectories'],
        'final_transforms_evaluated': False, 'goal_fully_achieved': False,
        'call_trace_proves_execution_not_causal_contribution': True,
        'domain_field_provenance': provenance,
        'new_blind_dataset': False, 'production_promoted': False,
        'protected_check': study.validate_study() == config}
    destination = study.SCOPE / 'retention_audits'
    destination.mkdir(exist_ok=True)
    path = destination / f'audit-{len(list(destination.glob("audit-*.json")))+1:04d}.json'
    write_new(path, value)
    print(study.json.dumps({'audit': str(path), 'qualifying': qualifying,
                           'remeasured': remeasured_predictions, 'pending': pending}), flush=True)
    return path, value


if __name__ == '__main__':
    audit()
