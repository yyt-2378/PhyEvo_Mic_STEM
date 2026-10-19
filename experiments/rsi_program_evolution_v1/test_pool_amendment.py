"""Only the explicitly authorized pool ceiling changes; science stays frozen."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from . import pool_amendment_continuation as amendment
from .contracts import file_hash, write_new
from .field_storage import ArrayPool
from .test_system_state import fixture, prepare_initial, proposal


def approval(directory, policy_sha):
    return {'format': amendment.FORMAT, 'study_scope': str(directory),
        'approved_by': 'user', 'approval_text': 'Explicit synthetic test authorization',
        'preserve_existing_files': True, 'scientific_protocol_unchanged': True,
        'previous_policy': dict(amendment.BASE), 'effective_policy': dict(amendment.POLICY),
        'original_pool_policy_sha256': policy_sha}


def test_only_authorized_pool_changes():
    before = deepcopy(amendment.BASE)
    effective = amendment.validate_authorization(approval('/scope', 'sha'), '/scope', 'sha')
    assert effective['max_prediction_pool_bytes'] == 512*1024**2
    assert effective['max_growth_bytes'] == 1024**3
    assert {k for k in before if before[k] != effective[k]} == {'max_prediction_pool_bytes'}
    assert amendment.BASE == before


@pytest.mark.parametrize('key', ['max_growth_bytes', 'max_input_pool_bytes',
    'max_written_prediction_bytes', 'metadata_reserve_bytes', 'minimum_free_bytes',
    'max_prediction_pool_bytes'])
def test_other_limits_or_more_than_512_mib_are_refused(key):
    value = approval('/scope', 'sha'); value['effective_policy'][key] += 1
    with pytest.raises(ValueError, match='authorization'):
        amendment.validate_authorization(value, '/scope', 'sha')


@pytest.mark.parametrize('key,value', [('approved_by', 'agent'), ('approval_text', ''),
    ('study_scope', '/another'), ('original_pool_policy_sha256', 'changed'),
    ('preserve_existing_files', False), ('scientific_protocol_unchanged', False)])
def test_wrong_authority_scope_or_immutable_binding_is_refused(key, value):
    record = approval('/scope', 'sha'); record[key] = value
    with pytest.raises(ValueError, match='authorization'):
        amendment.validate_authorization(record, '/scope', 'sha')


def guard_fixture(tmp_path):
    old = ArrayPool(tmp_path, tmp_path/'storage_runtime_members_v1/predictions',
        max_pool_bytes=amendment.BASE['max_prediction_pool_bytes'],
        minimum_free_bytes=amendment.BASE['minimum_free_bytes'])
    path = tmp_path/'authorization.json'
    write_new(path, approval(tmp_path, file_hash(old.root/'policy.json')))
    return old, path, amendment.AuthorizedGuard(old, path)


def test_amendment_never_rewrites_historical_policy(tmp_path):
    old, path, guard = guard_fixture(tmp_path)
    original = (old.root/'policy.json').read_bytes()
    with guard.locked():
        assert guard.policy['max_pool_bytes'] == 512*1024**2
        assert old.policy['max_pool_bytes'] == 256*1024**2
    assert (old.root/'policy.json').read_bytes() == original
    changed_copy = guard.policy; changed_copy['max_pool_bytes'] = 9999999999
    assert guard.policy['max_pool_bytes'] == 512*1024**2


@pytest.mark.parametrize('target', ['authorization', 'historical_policy'])
def test_changed_binding_is_detected_before_use(tmp_path, target):
    old, path, guard = guard_fixture(tmp_path)
    victim = path if target == 'authorization' else old.root/'policy.json'
    # Owned synthetic fixture only: byte change without altering JSON meaning.
    with victim.open('a') as stream: stream.write(' ')
    with pytest.raises(ValueError, match='changed during execution'):
        with guard.locked(): pass
    with pytest.raises(ValueError, match='changed during execution'):
        _ = guard.policy


def runner_fixture(tmp_path):
    archive, original = fixture(tmp_path)
    original.development_transforms = ('rot90',)
    initial, _ = prepare_initial(original)
    directory = tmp_path/'storage_runtime_members_v1'
    first = amendment.prior.storage.adapt_runner(original, directory, amendment.BASE)
    authorization = tmp_path/'authorization.json'
    write_new(authorization, approval(tmp_path, file_hash(directory/'predictions/policy.json')))
    runner = amendment.adapt_runner(original, directory, authorization)
    return archive, runner, initial


def test_actual_input_reuse_still_computes_outputs_and_exact_replay(tmp_path):
    archive, runner, initial = runner_fixture(tmp_path)
    try:
        old_hash = file_hash(runner.sandbox.prediction_pool.root/'policy.json')
        child = runner.register_system(proposal(initial), runner.frozen_state(initial))
        _, receipt = runner.execute(child)
        assert 'rows' in receipt, receipt
        sharing = receipt['execution']['prediction_storage']['sharing']
        assert sharing['computed_predictions'] == sharing['borrowed_input_predictions'] == 2
        assert sharing['new_prediction_payload_bytes'] == 0 and not sharing['inference_skipped']
        _, replay = runner.execute(child, replay=True)
        assert receipt['summary'] == replay['summary']
        assert replay['execution']['prediction_storage']['sharing']['new_prediction_payload_bytes'] == 0
        assert file_hash(runner.sandbox.prediction_pool.root/'policy.json') == old_hash
    finally:
        archive.close()


def test_actual_worker_prewrite_cap_remains_in_force(tmp_path):
    archive, runner, _ = runner_fixture(tmp_path)
    try:
        runner.sandbox = replace(runner.sandbox, max_written_prediction_bytes=1)
        cap = runner.capability({'pipeline_code': "def predict(s,x,c): return {'score':s['base']+2,'support':s['valid']}",
                                'training_steps':0, 'parent_state':None})
        with pytest.raises(RuntimeError, match='budget exceeded before write'):
            runner.predict_capability(cap, 'development')
        assert not list((runner.checked_ref(cap).parent/'development/output').glob('prediction-*.npz'))
    finally:
        archive.close()


def test_new_limit_is_used_not_a_reset_pool(tmp_path, monkeypatch):
    archive, runner, _ = runner_fixture(tmp_path)
    try:
        # Simulate allocated usage between old and new limits, without allocating 300 MiB.
        monkeypatch.setattr(amendment.bounded, 'allocated_bytes', lambda root: 300*1024**2)
        cap = runner.capability({'pipeline_code': "def predict(s,x,c): return {'score':s['base']+2,'support':s['valid']}",
                                'training_steps':0, 'parent_state':None})
        receipt = runner.predict_capability(cap, 'development')
        assert 'rows' in receipt, receipt
        proof = json.loads((runner.checked_ref(cap).parent/'development/output/remaining_capacity.json').read_text())
        assert proof['remaining_pool_bytes_before_job'] == 212*1024**2
        assert proof['new_payload_limit_bytes'] == amendment.BASE['max_written_prediction_bytes']
    finally:
        archive.close()


def test_even_authorized_limit_stops_before_worker_when_exhausted(tmp_path, monkeypatch):
    archive, runner, _ = runner_fixture(tmp_path)
    try:
        monkeypatch.setattr(amendment.bounded, 'allocated_bytes', lambda root: 512*1024**2)
        cap = runner.capability({'pipeline_code': "def predict(s,x,c): return {'score':s['base']+2,'support':s['valid']}",
                                'training_steps':0, 'parent_state':None})
        with pytest.raises(OSError, match='capacity exhausted before worker launch'):
            runner.predict_capability(cap, 'development')
    finally:
        archive.close()
