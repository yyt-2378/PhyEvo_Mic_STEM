"""Meta policy admission has a different experimental unit from task admission."""
from dataclasses import replace

import pytest

from experiments.rsi_program_evolution_v1.archive import Archive
from experiments.rsi_program_evolution_v1.test_core import protocol, proposal, trial_receipt


def setup_archive(tmp_path):
    a = Archive(tmp_path / "experiment", protocol(), create=True)
    base = a.add(proposal())
    trial = a.record_trial(base, "validation", "ok", {**trial_receipt(), "actual_execution": True})
    a.select(base, True, "baseline", trial)
    a.event("baseline_registered", {"candidate": base, "trial": trial})
    meta = a.add(proposal(action="tune", parents=(base,), target="research_policy",
                         intervention="Change research strategy"))
    return a, base, meta


def meta_receipt(archive, base, meta):
    baseline_trial = archive.events("baseline_registered")[0]["body"]["trial"]
    control_trial = archive.record_trial(base, "validation", "ok", {**trial_receipt(.52), "actual_execution": True})
    candidate_trial = archive.record_trial(meta, "validation", "ok", {**trial_receipt(.54), "actual_execution": True})
    common = {"model": "same-model", "input_sha256": "same-input",
              "starting_program_sha256": archive.candidate(base)["code_hash"],
              "baseline_trial": baseline_trial,
              "budget": {"model_calls": 2, "training_steps": 100}}
    return {"actual_execution": True, "pairs": [{
        "control": {**common, "gain_per_call": .01, "execution_ids": [control_trial]},
        "candidate": {**common, "gain_per_call": .02, "execution_ids": [candidate_trial]}}]}


def test_task_gain_does_not_promote_research_strategy(tmp_path):
    archive, base, meta = setup_archive(tmp_path)
    trial = archive.record_trial(meta, "validation", "ok", trial_receipt(.9))
    with pytest.raises(ValueError, match="separate paired"):
        archive.select_policy(meta, trial)
    archive.close()


def test_meta_requires_matched_model_data_and_budget(tmp_path):
    archive, base, meta = setup_archive(tmp_path)
    r = meta_receipt(archive, base, meta); r["pairs"][0]["candidate"]["budget"] = {"model_calls": 20}
    trial = archive.record_trial(meta, "meta_validation", "ok", r)
    with pytest.raises(ValueError, match="Unmatched"):
        archive.select_policy(meta, trial)
    archive.close()


def test_meta_candidate_can_be_admitted_independently(tmp_path):
    archive, base, meta = setup_archive(tmp_path)
    trial = archive.record_trial(meta, "meta_validation", "ok", meta_receipt(archive, base, meta))
    assert archive.select_policy(meta, trial)
    assert archive.incumbent() == base
    assert archive.events("policy_selection")[-1]["body"]["incumbent"] == meta
    assert archive.events("policy_selection")[-1]["body"]["trial"] == trial
    archive.close()


def test_meta_cannot_claim_gain_without_matching_execution(tmp_path):
    archive, base, meta = setup_archive(tmp_path)
    r = meta_receipt(archive, base, meta)
    r["pairs"][0]["candidate"]["gain_per_call"] = .9
    trial = archive.record_trial(meta, "meta_validation", "ok", r)
    with pytest.raises(ValueError, match="disagrees"):
        archive.select_policy(meta, trial)
    archive.close()
