from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from . import composition_final_audit as final
from .contracts import Protocol, file_hash, write_new
from .executor import Sample


def summary(a, b, c):
    return {k: {'mean': v} for k, v in zip(('A', 'B', 'C'), (a, b, c))}


P = SimpleNamespace(validation_groups=('A', 'B'), maximum_group_regression=.002,
                    minimum_gain=.002)


def test_held_gain_must_survive_on_both_matched_references_and_original():
    old = summary(.7, .6, .8)
    good = summary(.71, .62, .8)
    assert final.held_decision(good, old, old, good, P)['passed']
    assert not final.held_decision(good, old, summary(.75, .65, .8), good, P)['passed']
    assert not final.held_decision(good, old, old, summary(.72, .62, .8), P)['passed']


def test_training_source_gain_cannot_hide_nontraining_regression():
    old = summary(.7, .6, .5)
    assert not final.held_decision(summary(.699, .599, .9), old, old, old, P)['passed']
    with pytest.raises(ValueError, match='Unmatched'):
        final.held_decision({'A': {'mean': .9}}, old, old, old, P)
    with pytest.raises(ValueError, match='Invalid'):
        final.held_decision(summary(float('nan'), .6, .5), old, old, old, P)


def test_seal_prevents_more_search_even_before_final_measurement(tmp_path):
    final.require_search_open(tmp_path)
    write_new(tmp_path / 'composition_search_seal.json', {'final_results_not_yet_known': True})
    with pytest.raises(ValueError, match='sealed'):
        final.require_search_open(tmp_path)


def test_incomplete_development_cannot_create_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(final.study, 'SCOPE', tmp_path)
    monkeypatch.setattr(final.study, 'validate_study', lambda: {})
    monkeypatch.setattr(final, 'search_evidence', lambda config: {})
    monkeypatch.setattr(final.retention, 'audit', lambda: (
        tmp_path / 'audit.json', {'development_prerequisites_met': False}))
    with pytest.raises(ValueError, match='prerequisites'):
        final.seal()
    assert not (tmp_path / 'composition_search_seal.json').exists()


def test_extra_attempt_after_sealing_invalidates_final_execution(tmp_path, monkeypatch):
    original = {'ledger': {'attempt': 'old'}, 'selected': {}, 'frozen': {}}
    monkeypatch.setattr(final.study, 'SCOPE', tmp_path)
    monkeypatch.setattr(final.study, 'validate_study', lambda: {})
    monkeypatch.setattr(final, 'implementations', lambda: {'test': 'same'})
    write_new(tmp_path / 'composition_search_seal.json', dict(original, implementations={'test': 'same'}))
    monkeypatch.setattr(final, 'search_evidence', lambda config: dict(original, ledger={'attempt': 'new'}))
    with pytest.raises(ValueError, match='after sealing'):
        final.verify_seal()


def test_controller_entry_points_reject_search_after_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(final.study, 'SCOPE', tmp_path)
    write_new(tmp_path / 'composition_search_seal.json', {})
    with pytest.raises(ValueError, match='sealed'):
        final.study.request('full', 23)
    with pytest.raises(ValueError, match='sealed'):
        final.study.execute('full', 23, 1)


def test_final_driver_uses_actual_sandbox_predictions_without_fit_or_cherry_picking(tmp_path, monkeypatch):
    """Synthetic only; verify all seeds/arms/transforms really run and zero gain fails."""
    real_root = final.study.ROOT
    real_sandbox = final.SharedPredictionSandbox
    scope = tmp_path / 'scope'
    scope.mkdir()
    development_pool = scope / 'field_pool'
    development_pool.mkdir()
    write_new(development_pool / 'preserved.json', {'development_inputs_are_immutable': True})
    development_pool_hash = file_hash(development_pool / 'preserved.json')
    program = scope / 'program'
    program.mkdir()
    code = "def fit(s,c):\n raise AssertionError('final audit must not fit')\ndef predict(s,w,c):\n assert c['training_steps']==0\n assert 'human' not in s and 'known' not in s\n return {'score':s['image']+w['delta'], 'support':s['valid']}\n"
    with (program / 'pipeline.py').open('x') as stream:
        stream.write(code)
    state = scope / 'state.npz'
    with state.open('xb') as stream:
        np.savez_compressed(stream, delta=np.array([0.], dtype=np.float32))
    protocol = Protocol(('C',), ('A', 'B'), ('held-A',), seeds=(23, 41))
    config = {'seeds': [23, 41], 'held_back_transforms': ['flip_lr', 'rot180'],
              'minimum_free_bytes': 0, 'maximum_prediction_pool_bytes': 4*1024**2,
              'success': {'independent_full_trajectories': 2}}
    image = np.arange(20, dtype=np.float32).reshape(4, 5)
    valid = np.ones(image.shape, bool)
    human = image >= 18
    samples = [Sample('sample-'+g, g, 'interface', {'image': image, 'valid': valid},
                      human, human.astype(float), np.empty((0, 2))) for g in ('A', 'B', 'C')]
    system = {'identity': 'synthetic', 'program': str(program.relative_to(tmp_path)),
              'program_hashes': {'pipeline.py': file_hash(program / 'pipeline.py')},
              'state': {'path': str(state.relative_to(tmp_path)), 'sha256': file_hash(state)},
              'original_summary': summary(1., 1., 1.)}
    sealed = {'frozen': system, 'qualifying_full_seeds': [23, 41],
              'selected': {f'{arm}-s{seed}': system for arm in ('route', 'full') for seed in config['seeds']}}
    write_new(scope / 'composition_search_seal.json', sealed)
    monkeypatch.setattr(final.study, 'ROOT', tmp_path)
    monkeypatch.setattr(final.study, 'SCOPE', scope)
    monkeypatch.setattr(final, 'verify_seal', lambda: (config, sealed))
    monkeypatch.setattr(final.study, 'validate_study', lambda: config)
    monkeypatch.setattr(final.study, 'protocol_for', lambda cfg, seed: replace(protocol, seeds=(seed,)))
    def transformed(cfg, transform):
        move = lambda a: np.ascontiguousarray(np.fliplr(a) if transform == 'flip_lr' else np.rot90(a, 2))
        return [replace(s, fields={k: move(v) for k,v in s.fields.items()},
                        human=move(s.human), known=move(s.known)) for s in samples]
    monkeypatch.setattr(final.study, 'samples_for', transformed)
    monkeypatch.setattr(final, 'SharedPredictionSandbox', lambda root, runtime, **kw:
                        real_sandbox(real_root, real_root / '.venvs/sam2', **kw))
    final.execute()
    assert file_hash(development_pool / 'preserved.json') == development_pool_hash
    assert list(development_pool.iterdir()) == [development_pool / 'preserved.json']
    final_pool = scope / 'composition_final_audit/input_pool'
    assert final.study.read(final_pool / 'policy.json')['max_pool_bytes'] == 256*1024**2
    result = final.study.read(scope / 'composition_final_audit/results.json')
    assert result['actual_prediction_arrays'] == 36
    assert not result['failed_executions']
    assert len(result['all_results']) == 4
    assert not result['qualifying_seeds_passing_all_final_transforms']
    assert not result['numerical_completion_prerequisites_met']
    assert not result['goal_fully_achieved']
    for path in (scope / 'composition_final_audit').glob('*/input/request.json'):
        request = final.study.read(path)
        assert request['mode'] == 'predict' and request['training_steps'] == 0
        assert request['state_file']
        assert file_hash(path.with_name('state.npz')) == file_hash(state)
