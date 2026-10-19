"""Prospectively register one extra research call without rewriting phase one.

This changes neither the scientific API nor admission. It does not manufacture
an inheritance witness: a new independent proposal must still actually run.
"""
import argparse
from pathlib import Path

from . import composition_study as study
from .contracts import file_hash, write_new
from .field_storage import exact_child


def phase_path(arm, seed, number):
    return exact_child(study.ROOT, study.SCOPE / 'continuation_phases' /
                       f'{arm}-s{seed}-r{number:02d}.json')


def budget_binding(config, arm, seed, number):
    if arm not in config['arms'] or seed not in config['seeds'] or number < 1:
        raise ValueError('Undeclared trajectory or round')
    if number <= config['rounds_per_arm']:
        return None
    path = phase_path(arm, seed, number)
    if not path.is_file():
        raise ValueError('Study proposal budget exhausted; separate preregistration required')
    phase = study.read(path)
    expected = {'format': 'composition-single-call-continuation-v1',
                'arm': arm, 'seed': seed, 'round': number,
                'previous_round': number - 1, 'additional_model_calls': 1,
                'initial_round_budget': config['rounds_per_arm'],
                'original_protocol_sha256': file_hash(study.SCOPE / 'prospective_protocol.json'),
                'scientific_protocol_changed': False,
                'additional_call_counts_as_initial_budget': False}
    if any(phase.get(k) != v for k, v in expected.items()):
        raise ValueError('Continuation contract differs from its declared single call')
    for name, expected_hash in phase['prior_evidence'].items():
        if file_hash(exact_child(study.SCOPE, study.SCOPE / name)) != expected_hash:
            raise ValueError('Continuation prior evidence changed')
    previous = study.completed_before(arm, seed, number)
    accepted = [r for _, r in previous if r['decision']['accepted']]
    if not accepted or accepted[-1]['identity'] != phase['parent']:
        raise ValueError('Continuation must start from the latest admitted state')
    return {'file': str(path.relative_to(study.SCOPE)), 'sha256': file_hash(path),
            'phase_model_call': 1, 'initial_round_budget': config['rounds_per_arm'],
            'parent': phase['parent']}


def check_binding(config, binding):
    phase = budget_binding(config, binding['arm'], binding['seed'], binding['round'])
    if binding.get('continuation_phase') != phase:
        raise ValueError('Request continuation binding missing or changed')
    if phase is not None and binding['parent'] != phase['parent']:
        raise ValueError('Request does not inherit its registered continuation parent')
    return phase


def phase_ledger():
    directory = study.SCOPE / 'continuation_phases'
    return {str(p.relative_to(study.SCOPE)): file_hash(exact_child(study.SCOPE, p))
            for p in sorted(directory.glob('*.json'))}


def register(arm, seed):
    from .composition_final_audit import require_search_open, search_evidence
    require_search_open(study.SCOPE)
    config = study.validate_study()
    if arm not in config['arms'] or seed not in config['seeds']:
        raise ValueError('Undeclared trajectory')
    # This also rejects nonterminal requests on any other declared trajectory.
    evidence = search_evidence(config)
    previous = study.history(arm, seed)
    number = len(previous) + 1
    if number <= config['rounds_per_arm']:
        raise ValueError('Initial proposal budget has not been exhausted')
    accepted = [r for _, r in previous if r['decision']['accepted']]
    if len(accepted) < config['success']['accepted_updates_per_successful_trajectory']:
        raise ValueError('Continuation requires repeated gains and a retained admitted state')
    folder = exact_child(study.ROOT, study.SCOPE / f'{arm}-s{seed}' / f'r{number:02d}')
    if folder.exists():
        raise ValueError('Continuation must be registered before its request exists')
    path = phase_path(arm, seed, number)
    path.parent.mkdir(exist_ok=True)
    value = {'format': 'composition-single-call-continuation-v1',
        'arm': arm, 'seed': seed, 'round': number, 'previous_round': number-1,
        'parent': accepted[-1]['identity'], 'additional_model_calls': 1,
        'initial_round_budget': config['rounds_per_arm'],
        'original_protocol_sha256': file_hash(study.SCOPE / 'prospective_protocol.json'),
        'prior_evidence': evidence['ledger'],
        'prior_selected_systems': {k: v['identity'] for k, v in evidence['selected'].items()},
        'reason': 'Actual next research iteration after an upgrade at the end of phase one.',
        'model': config['model'], 'reasoning_effort': config['reasoning_effort'],
        'scientific_protocol_changed': False, 'additional_call_counts_as_initial_budget': False,
        'admission_and_inheritance_requirements_unchanged': True,
        'candidate_outcome_prescribed': False, 'final_results_visible': False,
        'not_a_new_independent_trajectory': True,
        'storage_limit_unchanged_bytes': config['cumulative_growth_bytes'],
        'controller_sha256': file_hash(study.__file__),
        'registration_implementation_sha256': file_hash(__file__)}
    write_new(path, value)
    budget_binding(config, arm, seed, number)
    print(study.json.dumps({'continuation_phase': str(path), 'round': number,
                           'additional_calls': 1, 'parent': value['parent']}), flush=True)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-config', required=True)
    parser.add_argument('--arm', choices=('full', 'route'), default='full')
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    study.configure_run(args.run_config)
    register(args.arm, args.seed)
