"""Input borrowing changes storage, not inference, replay or admission."""
from dataclasses import replace

import pytest

from . import input_reuse_storage as reuse
from .input_reuse_continuation import estimate
from .field_storage import allocated_bytes
from .test_system_state import fixture, prepare_initial, proposal
from .test_storage_continuation import small_policy
from . import test_stable_study as science


def legacy_runner(tmp_path):
    archive, original = fixture(tmp_path)
    original.development_transforms = ('rot90',)
    initial, _ = prepare_initial(original)
    runner = reuse.adapt_runner(original, tmp_path / 'new-pool', small_policy())
    return archive, runner, initial


def test_existing_inputs_are_computed_and_linked_without_importing_payload_into_pool(tmp_path):
    archive, runner, initial = legacy_runner(tmp_path)
    try:
        before = allocated_bytes(runner.sandbox.prediction_pool.root)
        child = runner.register_system(proposal(initial), runner.frozen_state(initial))
        _, result = runner.execute(child)
        assert 'rows' in result, result
        sharing = result['execution']['prediction_storage']['sharing']
        assert sharing['computed_predictions'] == sharing['borrowed_input_predictions'] == 2
        assert sharing['new_prediction_payload_bytes'] == sharing['new_canonical_objects'] == 0
        assert not sharing['inference_skipped']
        assert allocated_bytes(runner.sandbox.prediction_pool.root) == before
        _, replay = runner.execute(child, replay=True)
        assert 'rows' in replay, replay
        assert replay['execution']['prediction_storage']['sharing']['borrowed_input_predictions'] == 2
    finally:
        archive.close()


def test_unknown_branch_and_training_reserves_are_not_removed(tmp_path):
    archive, runner, initial = legacy_runner(tmp_path)
    try:
        branch = {'name':'b2','pipeline_code':'def predict(s,x,c): return {}','training_steps':2,
                  'parent_branch':None,'replaces':None}
        p = proposal(initial, operation='add_branch', new_branch=branch, active_branches=['b0','b1','b2'])
        before, tail = allocated_bytes(tmp_path / 'new-pool'), archive.verify()
        plan = estimate(runner, p, runner.frozen_state(initial)['library'], small_policy())
        assert plan['input_pair_payload_reservation_removed'] > 0
        expected = sum(sum(s.fields['image'].size*5+16384 for s in runner.cohorts(item['split']))
                       for item in plan['missing_predictions'])
        assert plan['planned_prediction_bytes'] == expected and expected > 0
        assert plan['checkpoint_bytes_bound'] == 256*1024**2
        assert archive.verify() == tail and allocated_bytes(tmp_path / 'new-pool') == before
    finally:
        archive.close()


def test_mutating_router_is_rejected_by_original_contract(tmp_path):
    archive, runner, initial = legacy_runner(tmp_path)
    try:
        p = proposal(initial, router_code="def route(s,b,x,c):\n b['b0']['score'][:]=0\n return {'branch':'b0','reason':'mutation'}\n")
        child = runner.register_system(p, runner.frozen_state(initial))
        _, result = runner.execute(child)
        assert 'changed a capability output' in result['error']
        assert archive.incumbent() == initial
    finally:
        archive.close()


def test_borrowed_hash_is_not_exposed_for_other_images(tmp_path):
    archive, runner, initial = legacy_runner(tmp_path)
    try:
        state = runner.frozen_state(initial)
        branches = {name: runner.predict_capability(ref, 'development')['predictions'] for name,ref in state['library'].items()}
        inputs = tmp_path / 'staged'
        runner.stage(inputs, runner.cohorts('development'), branches=branches)
        pool = reuse.InputReusePool(runner.sandbox.prediction_pool, inputs)
        assert pool.borrowed and pool.snapshot(['f'*64]) == {}
        key = next(iter(pool.borrowed))
        assert pool.snapshot([key])
        source = next(iter(pool.borrowed[key].values()))
        with source.open('ab') as stream:
            stream.write(b'synthetic deliberate tamper')
        with pytest.raises(ValueError, match='Borrowed input changed'):
            reuse.InputReusePool(runner.sandbox.prediction_pool, inputs)
    finally:
        archive.close()


def test_novel_payload_still_faces_prewrite_cap(tmp_path):
    archive, runner, initial = legacy_runner(tmp_path)
    try:
        runner.sandbox = replace(runner.sandbox, max_written_prediction_bytes=1)
        cap = runner.capability({'pipeline_code':"def predict(s,x,c): return {'score':s['base']+2,'support':s['valid']}",
                                 'training_steps':0,'parent_state':None})
        with pytest.raises(RuntimeError, match='budget exceeded before write'):
            runner.predict_capability(cap, 'development')
        cap_folder = runner.checked_ref(cap).parent
        assert not list((cap_folder / 'development/output').glob('prediction-*.npz'))
        logs = list(tmp_path.rglob('stderr.log'))
        assert any('budget exceeded before write' in p.read_text() for p in logs)
    finally:
        archive.close()


def patch_science(monkeypatch):
    original = science.experiment
    def experiment(tmp_path, patch):
        initial, load = original(tmp_path, patch)
        def adapted(*args, **kwargs):
            values = load(*args, **kwargs)
            return (*values[:-1], reuse.adapt_runner(values[-1], tmp_path / 'reuse-pool', small_policy()))
        patch.setattr(science.study, 'load', adapted)
        return initial, adapted
    monkeypatch.setattr(science, 'experiment', experiment)


def test_two_real_synthetic_updates_and_next_inheritance(monkeypatch, tmp_path):
    patch_science(monkeypatch)
    science.test_executed_improvements_are_accepted_and_actually_inherited_twice(tmp_path, monkeypatch)


def test_rotation_regression_is_rejected(monkeypatch, tmp_path):
    patch_science(monkeypatch)
    science.test_original_gain_with_rotation_failure_is_banked_but_never_deployed(tmp_path, monkeypatch)


def test_invalid_original_reply_is_not_repaired(monkeypatch, tmp_path):
    patch_science(monkeypatch)
    science.test_invalid_reply_keeps_raw_bytes_and_consumes_round(tmp_path, monkeypatch)
