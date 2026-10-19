"""Report checks use synthetic fixtures, never local manuscripts or live data."""
from dataclasses import asdict

import pytest

from experiments.rsi_program_evolution_v1.archive import Archive
from experiments.rsi_program_evolution_v1.contracts import write_new
from experiments.rsi_program_evolution_v1.meta_report import collect
from experiments.rsi_program_evolution_v1.test_core import protocol, proposal, trial_receipt


def fixture(tmp_path, *, mismatched_summary=False, continuation_policy=None):
    study = tmp_path / "study"
    study.mkdir()
    archive = Archive(study / "archive", protocol(), create=True)
    parent = archive.add(proposal())
    meta = archive.add(proposal(action="tune", parents=(parent,), target="research_policy"))
    def result(cid, value):
        summary = {"B": {"mean": value}}
        trial = archive.record_trial(cid, "validation", "ok", {
            **trial_receipt(value), "summary": summary, "actual_execution": True, "executions": [{"seconds": 1.0}]})
        return {"candidate": cid, "trial": trial, "summary": summary, "gain_per_call": max(0., value - .5)}
    base = result(parent, .5)
    child = archive.add(proposal(action="tune", parents=(parent,), intervention="synthetic child"))
    run = result(child, .55)
    run["baseline_trial"] = base["trial"]
    if mismatched_summary:
        run["summary"] = {"B": {"mean": .99}}
    archive.close()
    write_new(study / "study.json", {"allowed_requests": []})
    write_new(study / "meta_result.json", {"pairs": [{"control": run, "candidate": run}],
        "candidate_policy": meta, "accepted": True, "selected_policy": meta,
        "selected_task_program": child, "policy_selection": {"paired_gain_per_call": .01}})
    if continuation_policy:
        write_new(study / "continuation-result.json", {"policy": continuation_policy})
    return study


def test_meta_report_uses_recorded_task_trials(tmp_path):
    report = collect(fixture(tmp_path))
    assert report["table"][0]["starting_coverage"] == .5
    assert report["table"][0]["candidate"]["coverage"] == .55
    assert report["task_execution_seconds"] == 2
    assert report["continuation"] is None
    assert report["token_usage"] is None and not report["statistical_significance_claimed"]


def test_meta_report_rejects_changed_summary(tmp_path):
    with pytest.raises(ValueError, match="changed after evaluation"):
        collect(fixture(tmp_path, mismatched_summary=True))


def test_meta_report_checks_actual_policy_inheritance(tmp_path):
    with pytest.raises(ValueError, match="did not use"):
        collect(fixture(tmp_path, continuation_policy="unapproved-policy"))
