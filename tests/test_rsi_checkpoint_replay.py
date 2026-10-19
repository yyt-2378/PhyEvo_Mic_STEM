import json

import pytest

from scripts.replay_rsi_checkpoints import (
    gate_transition, idle_gpus, immutable_json, score_record_valid, validate_checkpoint,
)
from tfree_detection.expert_supervision import sha256_file


def test_average_gain_must_not_hide_source_regression():
    old = {"a": {23: 0.1, 41: 0.1}, "b": {23: 0.5, 41: 0.5}}
    new = {"a": {23: 0.5, 41: 0.5}, "b": {23: 0.4, 41: 0.4}}
    result = gate_transition(old, new)
    assert result["equal_source_mean_gain"] > 0
    assert result["mean_hides_source_regression"]
    assert result["regressed_sources"] == ["b"]
    assert result["decision"] == "reject_update_retain_incumbent"
    assert not result["production_release_eligible"]


def test_seed_regression_blocks_even_when_source_mean_improves():
    result = gate_transition({"a": {23: 0.5, 41: 0.5}}, {"a": {23: 0.45, 41: 0.65}})
    assert result["equal_source_mean_gain"] > 0
    assert result["regressed_sources"] == []
    assert result["seed_unstable_sources"] == ["a"]
    assert not result["descriptive_metric_gate_passed"]


def test_all_retained_and_mean_improved_is_metric_only_pass():
    result = gate_transition({"a": {23: 0.2}, "b": {23: 0.4}}, {"a": {23: 0.3}, "b": {23: 0.4}})
    assert result["descriptive_metric_gate_passed"]
    assert not result["production_release_eligible"]


@pytest.mark.parametrize("old,new", [({}, {}), ({"a": {23: 1}}, {"b": {23: 1}}),
                                      ({"a": {23: 1}}, {"a": {41: 1}}),
                                      ({"a": {23: 1}}, {"a": {23: float('nan')}})])
def test_gate_rejects_incompatible_evidence(old, new):
    with pytest.raises(ValueError):
        gate_transition(old, new)


def test_checkpoint_identity_and_group_isolation():
    fold = {"train_case_numbers": [1, 2], "validation_case_numbers": [3], "held_out_group": "test"}
    meta = {"seed": 23, "step": 400, "variant": "teacher_plus_expert", "protocol_sha256": "p",
            "learning_manifest_sha256": "m", "train_case_numbers": [1, 2], "held_out_group": "test",
            "train_groups": ["train"]}
    validate_checkpoint(meta, fold, 23, 400, "p", "m")
    with pytest.raises(ValueError, match="identity"):
        validate_checkpoint(meta, fold, 23, 800, "p", "m")
    with pytest.raises(ValueError, match="source group"):
        validate_checkpoint({**meta, "train_groups": ["test"]}, fold, 23, 400, "p", "m")


def test_resume_cannot_overwrite_or_reuse_tampered_output(tmp_path):
    path = tmp_path / "result.json"
    immutable_json(path, {"x": 1})
    immutable_json(path, {"x": 1})
    assert json.loads(path.read_text()) == {"x": 1}
    with pytest.raises(ValueError, match="changed existing"):
        immutable_json(path, {"x": 2})
    record = {"score_sha256": sha256_file(path), "protocol_sha256": "p", "checkpoint_sha256": "c"}
    score_record_valid(record, path, "p", "c")
    with pytest.raises(ValueError, match="lineage"):
        score_record_valid(record, path, "p", "different")


def test_gpu_guard_protects_user_and_other_agent_devices(monkeypatch):
    monkeypatch.setattr("scripts.replay_rsi_checkpoints.subprocess.check_output",
                        lambda *args, **kwargs: "0, 50000, 40\n2, 17, 0\n3, 17, 0\n4, 3000, 5\n")
    idle_gpus([3])
    for requested in ([0], [2], [3, 3]):
        with pytest.raises(ValueError, match="reserved"):
            idle_gpus(requested)
    with pytest.raises(ValueError, match="occupied"):
        idle_gpus([4])
