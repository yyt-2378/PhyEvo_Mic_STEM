"""Synthetic-only checks for explicit phase budgets, bindings and closure."""
import pytest

from . import composition_continuation as continuation
from . import composition_final_audit as final
from . import composition_study as study
from .contracts import file_hash, write_new


@pytest.fixture
def phase_fixture(tmp_path, monkeypatch):
    scope = tmp_path / 'scope'
    scope.mkdir()
    config = {'arms': ['full'], 'seeds': [67], 'rounds_per_arm': 2,
              'success': {'accepted_updates_per_successful_trajectory': 2},
              'model': 'synthetic', 'reasoning_effort': 'xhigh',
              'cumulative_growth_bytes': 1024**3,
              'evaluation': {'maximum_training_steps': 4096}}
    write_new(scope / 'prospective_protocol.json', config)
    records, ledger = [], {}
    for number in (1, 2):
        folder = scope / 'full-s67' / f'r{number:02d}'
        folder.mkdir(parents=True)
        result = {'identity': f'node-{number}', 'decision': {'accepted': True}}
        write_new(folder / 'result.json', result)
        write_new(folder / 'response.json', {'synthetic': True})
        write_new(folder / 'request_binding.json', {'synthetic': True})
        bound_names = ('result.json', 'response.json', 'request_binding.json')
        write_new(folder / 'result_binding.json', {name: file_hash(folder / name) for name in bound_names})
        for name in bound_names + ('result_binding.json',):
            ledger[str((folder / name).relative_to(scope))] = file_hash(folder / name)
        records.append((folder, result))
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    monkeypatch.setattr(study, 'SCOPE', scope)
    monkeypatch.setattr(study, 'validate_study', lambda: config)
    monkeypatch.setattr(study, 'history', lambda arm, seed: records)
    monkeypatch.setattr(final, 'search_evidence', lambda cfg: {
        'ledger': ledger, 'selected': {'full-s67': {'identity': 'node-2'}}})
    return scope, config, records


def test_initial_budget_is_unchanged_and_extra_call_requires_advance_registration(phase_fixture):
    scope, config, _ = phase_fixture
    before = file_hash(scope / 'prospective_protocol.json')
    assert continuation.budget_binding(config, 'full', 67, 2) is None
    with pytest.raises(ValueError, match='budget exhausted'):
        continuation.budget_binding(config, 'full', 67, 3)
    path = continuation.register('full', 67)
    phase = continuation.budget_binding(config, 'full', 67, 3)
    assert phase['parent'] == 'node-2'
    assert phase['sha256'] == file_hash(path)
    assert file_hash(scope / 'prospective_protocol.json') == before
    assert study.read(path)['additional_model_calls'] == 1
    assert not study.read(path)['additional_call_counts_as_initial_budget']
    with pytest.raises(ValueError, match='budget exhausted'):
        continuation.budget_binding(config, 'full', 67, 4)
    with pytest.raises(FileExistsError):
        continuation.register('full', 67)


def test_request_must_bind_the_phase_and_exact_admitted_parent(phase_fixture):
    _, config, _ = phase_fixture
    continuation.register('full', 67)
    phase = continuation.budget_binding(config, 'full', 67, 3)
    binding = {'arm': 'full', 'seed': 67, 'round': 3, 'parent': 'node-2',
               'continuation_phase': phase}
    assert continuation.check_binding(config, binding) == phase
    with pytest.raises(ValueError, match='binding'):
        continuation.check_binding(config, dict(binding, continuation_phase=None))
    with pytest.raises(ValueError, match='binding'):
        continuation.check_binding(config, dict(binding,
            continuation_phase=dict(phase, sha256='0'*64)))
    with pytest.raises(ValueError, match='parent'):
        continuation.check_binding(config, dict(binding, parent='node-1'))


def test_cannot_retroactively_register_after_request_or_after_final_seal(phase_fixture):
    scope, _, _ = phase_fixture
    (scope / 'full-s67/r03').mkdir()
    with pytest.raises(ValueError, match='before its request'):
        continuation.register('full', 67)
    assert not (scope / 'continuation_phases').exists()
    write_new(scope / 'composition_search_seal.json', {})
    with pytest.raises(ValueError, match='sealed'):
        continuation.register('full', 67)


def test_bound_prior_evidence_must_still_match(phase_fixture, monkeypatch):
    scope, config, _ = phase_fixture
    path = continuation.register('full', 67)
    original_hash = continuation.file_hash
    changed = scope / 'full-s67/r01/result.json'
    monkeypatch.setattr(continuation, 'file_hash',
                        lambda p: '0'*64 if p == changed else original_hash(p))
    with pytest.raises(ValueError, match='prior evidence changed'):
        continuation.budget_binding(config, 'full', 67, 3)
    assert original_hash(path) == file_hash(path)


def test_no_extra_budget_without_repeated_admitted_gains(phase_fixture):
    scope, _, records = phase_fixture
    records[-1][1]['decision']['accepted'] = False
    with pytest.raises(ValueError, match='repeated gains'):
        continuation.register('full', 67)
    assert not (scope / 'continuation_phases').exists()


def test_rejected_attempt_does_not_replace_the_retained_parent(phase_fixture):
    scope, config, records = phase_fixture
    folder = scope / 'full-s67/r03'
    folder.mkdir()
    result = {'identity': 'rejected-node', 'decision': {'accepted': False}}
    write_new(folder / 'result.json', result)
    for name in ('response.json', 'request_binding.json'):
        write_new(folder / name, {'synthetic': True})
    write_new(folder / 'result_binding.json', {name: file_hash(folder / name)
        for name in ('result.json', 'response.json', 'request_binding.json')})
    records.append((folder, result))
    continuation.register('full', 67)
    assert continuation.budget_binding(config, 'full', 67, 4)['parent'] == 'node-2'


def test_scientific_proposal_checker_enforces_extra_phase_binding(phase_fixture):
    scope, config, _ = phase_fixture
    continuation.register('full', 67)
    phase = continuation.budget_binding(config, 'full', 67, 3)
    folder = scope / 'full-s67/r03'
    folder.mkdir()
    write_new(folder / 'request.json', {'synthetic': True})
    response = {k: 'synthetic' for k in ('observation', 'mechanism', 'action', 'applicability', 'falsifier')}
    response.update(training_steps=0,
                    pipeline_code="def predict(s,w,c):\n return {'score':s['image'],'support':s['valid']}\n")
    write_new(folder / 'response.json', response)
    binding = {'arm': 'full', 'seed': 67, 'round': 3, 'parent': 'node-2',
               'continuation_phase': phase, 'request_sha256': file_hash(folder / 'request.json'),
               'protocol_sha256': file_hash(scope / 'prospective_protocol.json')}
    write_new(folder / 'request_binding.json', binding)
    assert study.checked_proposal(folder, config) == (binding, response)
    assert continuation.phase_ledger() == {phase['file']: phase['sha256']}
