import importlib.util
from pathlib import Path


def test_json_gate_replay_matches_tuple_serialization_without_relaxing_values():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_autonomous_rsi.py"
    spec = importlib.util.spec_from_file_location("autonomous_verifier", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    actual = {"accepted": False, "seed_regressions": [("a", "23")], "macro_gain": .123}
    stored = {"accepted": False, "seed_regressions": [["a", "23"]], "macro_gain": .123}
    assert module.same_json(actual, stored)
    assert not module.same_json(actual, {**stored, "accepted": True})
    assert not module.same_json(actual, {**stored, "macro_gain": .1230000000001})
