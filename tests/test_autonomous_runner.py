import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[1] / "scripts/run_autonomous_rsi.py"
    spec = importlib.util.spec_from_file_location("autonomous_runner_under_test", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_outer_cases_unavailable_to_development_loader_even_without_files(tmp_path):
    runner = module()
    with pytest.raises(ValueError, match="withheld"):
        runner.load_job(tmp_path, {}, {"stage": "outer"})


def test_outer_audit_requires_all_searches_closed_first(tmp_path, monkeypatch):
    runner = module()
    monkeypatch.setattr(runner, "checked", lambda out: {})
    with pytest.raises(FileNotFoundError, match="search_seal.json"):
        runner.audit_worker(tmp_path, 0)


def test_runtime_requires_explicit_opt_in_before_reading_binding(tmp_path):
    from tfree_detection.autonomous_runtime import AutonomousResearchRuntime
    with pytest.raises(ValueError, match="opt-in"):
        AutonomousResearchRuntime(tmp_path, "absent.json")


def test_controller_source_contract_includes_learning_governor_and_interpreter():
    runner = module()
    assert {"src/tfree_detection/autonomous_program.py", "src/tfree_detection/autonomous_rsi.py",
            "scripts/run_autonomous_rsi.py"} <= set(runner.CODE)


def test_one_command_pipeline_cannot_audit_before_search_sealing():
    path = Path(__file__).resolve().parents[1] / "scripts/run_autonomous_rsi_cycle.py"
    spec = importlib.util.spec_from_file_location("autonomous_pipeline", path)
    pipeline = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipeline)
    commands = pipeline.stage_commands(Path("outputs/new-exclusive-cycle"), "1,2")
    names = [name for name, _ in commands]
    assert names.index("freeze") < names.index("search") < names.index("seal-searches") < names.index("audit")
    assert names.index("verify") < names.index("release") < names.index("report")
    assert all("--output" in command for _, command in commands)
