"""The sandbox replicates a mechanism without importing the previous answers."""
import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/replay_rsi_sandbox.py"
    spec = importlib.util.spec_from_file_location("rsi_sandbox_under_test", path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def test_input_whitelist_does_not_include_historical_solutions():
    m = module()
    names = m.input_inventory({"jobs": [{"index": 0, "train_cases": [2, 3], "test_cases": [1, 2]}]})
    assert len(names) == 6
    assert len(names) == len(set(names))
    assert set(names) == {"protocol.json", "baseline/manifest.json", "cache/job-00/complete.json",
                         "cache/job-00/case-001.npz", "cache/job-00/case-002.npz", "cache/job-00/case-003.npz"}
    assert not any(s in name for name in names for s in ("searches", "audit", "candidate", "results"))


def test_pipeline_generates_searches_before_audit_and_independent_verification():
    names = [name for name, _ in module().stage_commands(Path("outputs/new-sandbox"), "1,2")]
    assert names == ["search", "seal", "audit", "verify", "release"]


def test_cannot_overwrite_existing_successful_directory(tmp_path):
    with pytest.raises(ValueError, match="exclusive"):
        module().prepare(tmp_path / "source", tmp_path)


def test_semantic_comparison_ignores_timing_but_not_learned_parameters():
    m = module()
    assert m.semantic({"elapsed_seconds": 1, "weights": [2]}) == m.semantic({"elapsed_seconds": 4, "weights": [2]})
    assert m.semantic({"weights": [2]}) != m.semantic({"weights": [3]})
    assert m.semantic({"winner": 1}) != m.semantic({"winner": 2})


@pytest.mark.parametrize("updates,verified,preserved,passed", [(2, True, True, True), (1, True, True, False),
    (2, False, True, False), (2, True, False, False), (0, True, True, False)])
def test_feasibility_does_not_become_general_rsi_claim(updates, verified, preserved, passed):
    m = module()
    curves = [{"search": {"arm": "feedback"}, "trajectory": [{"winner": None}] + [{"winner": 0}] * updates}]
    result = m.assess(curves, verified, preserved)
    assert result["bounded_loop_feasibility_reproduced"] is passed
    assert not result["independent_positive_rsi_proven"]
    assert not result["historical_human_geometry_repair_rediscovered"]
    assert not result["general_conversation_agent_replaced"]


def test_corrupt_input_is_rejected(tmp_path):
    m = module()
    with pytest.raises(ValueError, match="Changed protected"):
        m.check_hashes(tmp_path, {"missing.json": "0" * 64})
