"""The same frozen decision loop with bounded, additive storage transport."""
import json
from pathlib import Path

import pytest

from . import stable_study as study
from . import test_stable_study as scientific_tests
from .contracts import file_hash, write_new
from .field_storage import allocated_bytes
from .storage_continuation import POLICY, adapt_runner, binding, configure, estimate, installed
from .test_system_state import fixture, prepare_initial, proposal


def small_policy():
    return POLICY | {'minimum_free_bytes': 0, 'max_input_pool_bytes': 16*1024**2,
                     'max_prediction_pool_bytes': 16*1024**2, 'metadata_reserve_bytes': 1024**2}


def test_volume_estimate_does_not_create_arrays_or_change_research_state(tmp_path):
    archive, original = fixture(tmp_path)
    original.development_transforms = ('rot90',)
    try:
        initial, _ = prepare_initial(original)
        runner = adapt_runner(original, tmp_path/'storage', small_policy())
        before = archive.verify(); count = allocated_bytes(tmp_path/'storage')
        plan = estimate(runner, proposal(initial), runner.frozen_state(initial)['library'], small_policy())
        assert plan['input_pool_fits'] and plan['prediction_pool_fits']
        assert plan['total_new_bytes_bound'] > 0
        assert archive.verify() == before and allocated_bytes(tmp_path/'storage') == count
        assert not list(runner.array_pool.root.glob('*.npz'))
    finally:
        archive.close()


def test_training_and_unknown_outputs_have_explicit_storage_reserve(tmp_path):
    archive, original = fixture(tmp_path)
    original.development_transforms = ('rot90',)
    try:
        initial, _ = prepare_initial(original)
        runner = adapt_runner(original, tmp_path/'storage', small_policy())
        branch = {'name':'b2','pipeline_code':'def predict(s,x,c): return {}', 'training_steps':2,
                  'parent_branch':None,'replaces':None}
        plan = estimate(runner, proposal(initial, operation='add_branch', new_branch=branch,
                        active_branches=['b0','b1','b2']), runner.frozen_state(initial)['library'], small_policy())
        assert plan['checkpoint_bytes_bound'] == 256*1024**2
        assert plan['uncomputed_input_raw_bound'] > 0 and plan['training_jobs'] == 1
        assert any(p['branch']=='b2' for p in plan['missing_predictions'])
    finally:
        archive.close()


def use_storage_fixture(monkeypatch):
    original = scientific_tests.experiment
    def experiment(tmp_path, patch):
        initial, base_load = original(tmp_path, patch)
        def load(*args, **kwargs):
            values=base_load(*args, **kwargs)
            return (*values[:-1], adapt_runner(values[-1], tmp_path/'storage', small_policy()))
        patch.setattr(study, 'load', load)
        return initial, load
    monkeypatch.setattr(scientific_tests, 'experiment', experiment)


def test_same_two_real_executions_admit_and_inherit_with_new_storage(tmp_path, monkeypatch):
    use_storage_fixture(monkeypatch)
    scientific_tests.test_executed_improvements_are_accepted_and_actually_inherited_twice(tmp_path, monkeypatch)
    assert (tmp_path/'storage/fields/policy.json').is_file()
    assert len(list((tmp_path/'storage/predictions/objects').glob('*.npz'))) > 0


def test_rotation_failure_still_rejects_and_banks_the_candidate(tmp_path, monkeypatch):
    use_storage_fixture(monkeypatch)
    scientific_tests.test_original_gain_with_rotation_failure_is_banked_but_never_deployed(tmp_path, monkeypatch)


def test_invalid_reply_still_consumes_original_scientific_round(tmp_path, monkeypatch):
    use_storage_fixture(monkeypatch)
    scientific_tests.test_invalid_reply_keeps_raw_bytes_and_consumes_round(tmp_path, monkeypatch)


def test_explicit_binding_is_idempotent_and_drift_is_refused(tmp_path):
    write_new(tmp_path/'protocol.json', {'synthetic': True})
    preflight = study.ROOT/'outputs/rsi_v4_storage_migration_20261017_v1/prediction_storage_preflight.json'
    runtime=configure(tmp_path,preflight); before=file_hash(runtime/'binding.json')
    assert configure(tmp_path,preflight)==runtime and file_hash(runtime/'binding.json')==before
    assert json.loads((runtime/'binding.json').read_text())['implementations']==binding()
    bad=tmp_path/'different-proof.json'
    proof=json.loads(preflight.read_text()); proof['implementations']['worker.py']='0'*64
    write_new(bad,proof)
    with pytest.raises(ValueError,match='implementation changed'):
        configure(tmp_path,bad)


def test_context_manager_restores_original_loader_on_exception(tmp_path):
    original = study.load
    with pytest.raises(RuntimeError, match='synthetic'):
        with installed(tmp_path/'storage',small_policy()):
            assert study.load != original
            raise RuntimeError('synthetic')
    assert study.load == original
