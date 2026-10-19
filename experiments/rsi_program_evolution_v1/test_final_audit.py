import pytest

from experiments.rsi_program_evolution_v1.archive import Archive
from experiments.rsi_program_evolution_v1.final_audit import assert_frozen
from experiments.rsi_program_evolution_v1.test_core import protocol, proposal, trial_receipt


def frozen(tmp_path, training_steps=0):
    a = Archive(tmp_path / "experiment", protocol(), create=True)
    identity = a.add(proposal(training_steps=training_steps))
    trial = a.record_trial(identity, "validation", "ok", trial_receipt())
    a.select(identity, True, "initial", trial)
    seal = a.event("search_sealed", {"incumbent": identity})
    c = a.candidate(identity)
    return a, {"seal": seal, "protocol_sha256": a.protocol.identity, "incumbent": identity, "programs": {
        identity: {"pipeline": c["code_hash"], "policy": c["policy_hash"]}}}


def test_joint_audit_freeze_allows_only_audit_execution(tmp_path):
    a, spec = frozen(tmp_path)
    assert_frozen(a, spec)
    a.record_trial(spec["incumbent"], "audit", "ok", trial_receipt(.1))
    assert_frozen(a, spec)
    a.event("model_call_started", {"unexpected": True})
    with pytest.raises(ValueError, match="Development activity"):
        assert_frozen(a, spec)
    a.close()


def test_final_audit_cannot_retrain_the_candidate(tmp_path):
    a, spec = frozen(tmp_path, training_steps=1)
    with pytest.raises(ValueError, match="inference-only"):
        assert_frozen(a, spec)
    a.close()


def test_final_audit_refuses_wrong_seal(tmp_path):
    a, spec = frozen(tmp_path)
    with pytest.raises(ValueError, match="seal"):
        assert_frozen(a, {**spec, "seal": "invented"})
    a.close()


def test_final_audit_refuses_changed_protocol(tmp_path):
    a, spec = frozen(tmp_path)
    with pytest.raises(ValueError, match="protocol differs"):
        assert_frozen(a, {**spec, "protocol_sha256": "different"})
    a.close()
