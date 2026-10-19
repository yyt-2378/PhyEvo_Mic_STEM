"""Small storage/integration checks; no synthetic metric is scientific evidence."""
from dataclasses import replace
import json
import os

import pytest

from . import bounded_continuation as bounded
from . import test_stable_study as science
from .bank_study import analysis_replay
from .contracts import file_hash, write_new
from .field_storage import allocated_bytes
from .storage_budget import StorageBudget
from .test_system_state import fixture, prepare_initial, proposal
from .test_storage_continuation import small_policy


def credit_fixture(tmp_path):
    scope = tmp_path/'scope'; scope.mkdir()
    for name in ('a.bin', 'b.bin'):
        with (scope/name).open('xb') as stream:
            stream.write(b'x' * 131072)
    original = StorageBudget(tmp_path, scope, max_growth_bytes=65536, minimum_free_bytes=0)
    before = original.check()
    reclaimed = (scope/'b.bin').stat().st_blocks * 512
    os.link(scope/'a.bin', scope/'replacement.link')
    os.replace(scope/'replacement.link', scope/'b.bin')
    receipt = tmp_path/'credit.json'
    write_new(receipt, {'status': 'completed', 'reclaimed_allocated_bytes': reclaimed,
        'changed_contents': 0, 'missing_paths': 0, 'reclaimed_space_is_not_additional_experiment_budget': True})
    adjusted = bounded.ReclaimAwareBudget(tmp_path, scope, max_growth_bytes=65536, minimum_free_bytes=0,
        reclamation_receipt=receipt, receipt_sha256=file_hash(receipt))
    return scope, original, adjusted, before, receipt


def test_cleanup_cannot_reset_or_enlarge_the_original_growth_allowance(tmp_path):
    scope, original, adjusted, before, _ = credit_fixture(tmp_path)
    assert original.check()['growth_bytes'] == 0
    assert adjusted.check()['growth_bytes'] == before['growth_bytes']
    with (scope/'new-result.bin').open('xb') as stream:
        stream.write(b'z' * 60000)
    assert original.check()['reason'] is None
    assert adjusted.check()['reason'] == 'growth_limit'
    assert adjusted.policy == original.policy


def test_changed_cleanup_receipt_is_refused(tmp_path):
    _, _, adjusted, _, receipt = credit_fixture(tmp_path)
    with receipt.open('a') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='receipt changed'):
        adjusted.check()


def test_planner_preserves_worst_case_and_records_conditional_replay_estimate(tmp_path):
    archive, original = fixture(tmp_path); original.development_transforms = ('rot90',)
    try:
        initial, _ = prepare_initial(original)
        r = bounded.adapt_runner(original, tmp_path/'pool', small_policy())
        branch = {'name': 'b2', 'pipeline_code': 'def predict(s,x,c): return {}',
                  'training_steps': 0, 'parent_branch': None, 'replaces': None}
        p = proposal(initial, operation='add_branch', new_branch=branch, active_branches=['b0','b1','b2'])
        tail = archive.verify(); size = allocated_bytes(tmp_path/'pool')
        plan = bounded.estimate(r, p, r.frozen_state(initial)['library'], small_policy())
        reserve = plan['replay_raw_bound_already_in_original_estimate']
        assert reserve > 0 and plan['prediction_bytes_bound'] - plan['planned_prediction_bytes'] == reserve
        assert plan['all_replay_inference_still_executed']
        assert archive.verify() == tail and allocated_bytes(tmp_path/'pool') == size
    finally:
        archive.close()


def actual_runner(tmp_path):
    archive, original = fixture(tmp_path)
    original.development_transforms = ('rot90',)
    return archive, bounded.adapt_runner(original, tmp_path/'pool', small_policy())


def test_signed_zero_arrays_still_pass_original_numeric_replay_contract(tmp_path):
    archive, r = actual_runner(tmp_path)
    try:
        code = "import numpy as np\nn=0\ndef predict(s,x,c):\n global n\n n+=1\n return {'score':np.full(s['image'].shape,-0.0 if n==1 else 0.0,dtype=np.float32),'support':s['valid']}\n"
        cap = r.capability({'pipeline_code': code, 'training_steps': 0, 'parent_state': None})
        first = r.predict_capability(cap, 'development')
        replay = json.loads(r.checked_ref(analysis_replay(r, cap)).read_text())
        assert replay['exact_reverse_replay']
        assert first['predictions']['A']['sha256'] != replay['predictions']['A']['sha256']
    finally:
        archive.close()


def test_actual_case_order_dependence_still_fails_scientific_replay(tmp_path):
    archive, r = actual_runner(tmp_path)
    try:
        code = "import numpy as np\nn=0\ndef predict(s,x,c):\n global n\n n+=1\n return {'score':np.full(s['image'].shape,n,dtype=np.float32),'support':s['valid']}\n"
        cap = r.capability({'pipeline_code': code, 'training_steps': 0, 'parent_state': None})
        r.predict_capability(cap, 'development')
        with pytest.raises(RuntimeError, match='changes with case order'):
            analysis_replay(r, cap)
    finally:
        archive.close()


def test_worker_stops_novel_prediction_before_exceeding_write_limit(tmp_path):
    archive, r = actual_runner(tmp_path)
    try:
        r.sandbox = replace(r.sandbox, max_written_prediction_bytes=1)
        program = tmp_path/'program'; program.mkdir()
        with (program/'pipeline.py').open('x') as stream:
            stream.write("def predict(s,x,c):\n return {'score':s['base'],'support':s['valid']}\n")
        r.stage(tmp_path/'input', r.samples[:1])
        result = r.sandbox.run(program, tmp_path/'input', tmp_path/'output')
        assert result['status'] == 'failed'
        assert not list((tmp_path/'output').glob('prediction-*.npz'))
        assert 'budget exceeded before write' in (tmp_path/'output/stderr.log').read_text()
    finally:
        archive.close()


def patch_science(monkeypatch):
    original = science.experiment
    def experiment(tmp_path, patch):
        initial, load = original(tmp_path, patch)
        def adapted(*args, **kwargs):
            values = load(*args, **kwargs)
            return (*values[:-1], bounded.adapt_runner(values[-1], tmp_path/'pool', small_policy()))
        patch.setattr(science.study, 'load', adapted)
        return initial, adapted
    monkeypatch.setattr(science, 'experiment', experiment)


def test_two_actual_synthetic_updates_and_next_inheritance_are_preserved(tmp_path, monkeypatch):
    patch_science(monkeypatch)
    science.test_executed_improvements_are_accepted_and_actually_inherited_twice(tmp_path, monkeypatch)


def test_rotation_failure_is_still_rejected(tmp_path, monkeypatch):
    patch_science(monkeypatch)
    science.test_original_gain_with_rotation_failure_is_banked_but_never_deployed(tmp_path, monkeypatch)


def test_invalid_original_reply_is_not_repaired(tmp_path, monkeypatch):
    patch_science(monkeypatch)
    science.test_invalid_reply_keeps_raw_bytes_and_consumes_round(tmp_path, monkeypatch)
