from copy import deepcopy

import numpy as np
import pytest

import tfree_detection.autonomous_admission as admission


def family(candidate=.6):
    protocol = {"groups": ["a", "b", "c"], "jobs": [], "minimum_macro_gain": .002,
                "source_regression_tolerance": 1e-8, "seed_regression_tolerance": .01}
    audits = []
    for i, group in enumerate(protocol["groups"]):
        records = []
        for seed in (23, 41, 67):
            protocol["jobs"].append({"stage": "outer", "outer_group": group, "seed": seed, "test_cases": [i]})
            records.append({"group": group, "case_number": i, "seed": seed, "coverage": candidate,
                            "selected_pixels": 10, "budget": 10})
        audits.append({"search": {"outer_group": group}, "rows": records,
                       "baseline_rows": [{**r, "coverage": .5} for r in records]})
    return protocol, audits


def test_family_gate_passes_a_genuine_nonregressing_improvement():
    p, a = family()
    assert admission.family_decision(p, a)["accepted"]


def test_source_gain_cannot_be_used_to_admit_a_failed_family():
    p, a = family(.8)
    for r in a[0]["rows"]:
        r["coverage"] = .49
    gate = admission.family_decision(p, a)
    assert not gate["accepted"] and gate["source_regressions"] == ["a"]
    with pytest.raises(ValueError, match="subset"):
        admission.family_decision(p, a[1:])


@pytest.mark.parametrize("change", ["missing", "duplicate", "negative_pixels", "bad_budget", "nan"])
def test_incomplete_or_invalid_metrics_cannot_pass_admission(change):
    p, a = family()
    if change == "missing":
        a[0]["rows"].pop()
    elif change == "duplicate":
        a[0]["rows"].append(deepcopy(a[0]["rows"][0]))
    elif change == "negative_pixels":
        a[0]["rows"][0]["selected_pixels"] = -1
    elif change == "bad_budget":
        a[0]["rows"][0]["selected_pixels"] = 11
    else:
        a[0]["rows"][0]["coverage"] = float("nan")
    with pytest.raises(ValueError):
        admission.family_decision(p, a)


def runtime(monkeypatch, tmp_path, *, accepted=False, mode="active", failure=None, bad_mask=False):
    counts = {"incumbent_constructed": 0, "incumbent_predict": 0, "candidate_constructed": 0, "candidate_predict": 0}
    monkeypatch.setattr(admission, "load_admission", lambda root, path: {
        "gate": {"accepted": accepted, "reason": "passed" if accepted else "gain_or_nonregression_failed"},
        "bindings": [{"job_index": 0, "candidate": {"path": "candidate.json"}, "incumbent": {"path": "base.json"}}]})

    class Base:
        def predict(self, image):
            counts["incumbent_predict"] += 1
            mask = np.zeros(image.shape, bool); mask.flat[:3] = True
            return {"mask": mask, "budget": int(np.ceil(.1 * image.size)), "claims": {"production_approved": False}}

    class Candidate:
        def predict(self, image):
            counts["candidate_predict"] += 1
            if failure:
                raise failure
            mask = np.zeros(image.shape, bool); mask.flat[-4:] = True
            if bad_mask:
                mask[:] = True
            return {"mask": mask, "budget": int(np.ceil(.1 * image.size)), "status": "experimental_program_lineage"}

    def base_factory(*args):
        counts["incumbent_constructed"] += 1
        return Base()

    def candidate_factory(*args):
        counts["candidate_constructed"] += 1
        return Candidate()
    monkeypatch.setattr(admission, "_incumbent_runtime", base_factory)
    monkeypatch.setattr(admission, "_candidate_runtime", candidate_factory)
    value = admission.AdmittedRSIRuntime(tmp_path, "admission.json", 0, mode=mode, allow_experimental=True)
    return value, counts


def test_rejected_update_never_constructed_or_evaluated_in_active_mode(monkeypatch, tmp_path):
    model, counts = runtime(monkeypatch, tmp_path)
    result = model.predict(np.zeros((16, 16)))
    assert counts["candidate_constructed"] == counts["candidate_predict"] == 0
    assert result["rsi_admission"]["selected"] == "incumbent"
    assert not result["rsi_admission"]["candidate_evaluated"]
    assert result["mask"].sum() == 3


@pytest.mark.parametrize("accepted", [False, True])
def test_shadow_never_takes_over_primary_even_if_admitted(monkeypatch, tmp_path, accepted):
    model, counts = runtime(monkeypatch, tmp_path, accepted=accepted, mode="shadow")
    result = model.predict(np.zeros((16, 16)))
    assert counts["candidate_predict"] == 1
    assert result["mask"].sum() == 3 and result["shadow_mask"].sum() == 4
    assert not np.array_equal(result["mask"], result["shadow_mask"])
    assert result["rsi_admission"]["selected"] == "incumbent"


def test_admitted_candidate_can_become_active(monkeypatch, tmp_path):
    model, counts = runtime(monkeypatch, tmp_path, accepted=True)
    result = model.predict(np.zeros((16, 16)))
    assert counts["candidate_constructed"] == 1 and result["mask"].sum() == 4
    assert result["rsi_admission"]["selected"] == "candidate"


@pytest.mark.parametrize("failure", [ValueError("invalid field"), RuntimeError("device error"), OSError("missing candidate")])
def test_admitted_runtime_failure_preserves_incumbent(monkeypatch, tmp_path, failure):
    model, _ = runtime(monkeypatch, tmp_path, accepted=True, failure=failure)
    result = model.predict(np.zeros((16, 16)))
    assert result["mask"].sum() == 3
    assert result["rsi_admission"]["selected"] == "incumbent"
    assert result["rsi_admission"]["candidate_error"]


def test_budget_violation_in_admitted_candidate_is_rejected(monkeypatch, tmp_path):
    model, _ = runtime(monkeypatch, tmp_path, accepted=True, bad_mask=True)
    result = model.predict(np.zeros((16, 16)))
    assert result["mask"].sum() == 3 and result["rsi_admission"]["candidate_error"]


def test_no_opt_in_no_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(admission, "load_admission", lambda *args: pytest.fail("must not load"))
    with pytest.raises(ValueError, match="opt-in"):
        admission.AdmittedRSIRuntime(tmp_path, "not_there.json", 0)


def test_invalid_image_fails_before_model_execution(monkeypatch, tmp_path):
    model, counts = runtime(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="finite"):
        model.predict(np.full((16, 16), np.nan))
    assert counts["incumbent_predict"] == counts["candidate_predict"] == 0


def test_nonfinite_extra_candidate_field_is_not_hidden():
    x = np.zeros((16, 16), np.float32)
    with pytest.raises(ValueError, match="Nonfinite"):
        admission.validate_prediction({"mask": x.astype(bool), "budget": 26, "score": x + np.nan}, x)
