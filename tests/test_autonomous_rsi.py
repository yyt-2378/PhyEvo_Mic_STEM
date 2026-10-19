from copy import deepcopy

import numpy as np
import pytest
import torch

from tfree_detection.autonomous_rsi import (case_coverage, diagnose, fit_candidate,
    prepare_case, promotion, propose, scores)


def fixture_case(number=1, group="train"):
    rng = np.random.default_rng(number)
    image = rng.normal(size=(32, 32)).astype(np.float32)
    base = np.zeros_like(image); base[:, 10:14] = 1
    student = np.zeros_like(image); student[:, 18:22] = 1
    human = np.zeros_like(image, dtype=bool); human[:, 19:21] = True
    physical = {"score": base, "variation": np.abs(image), "valid": np.ones_like(human)}
    return prepare_case(number, group, image, physical, student, base,
                        np.ones_like(human), human, physical_route=True, band=False, resolution=32)


def rows(value, group="a"):
    return [{"group": group, "case_number": 1, "seed": seed, "coverage": value,
             "selected_pixels": 10, "budget": 10} for seed in [23, 41, 67]]


def test_sparse_positive_learning_changes_readout_without_mutating_labels():
    torch.set_num_threads(1)
    case = fixture_case(); before = case.human.clone()
    candidate = {"programs": [{"op": "field", "name": "student"}],
                 "learning_rate": .05, "temperature": .5, "retention": .01, "hard_positive_weight": 2}
    stage = fit_candidate([case], [[]], candidate, seed=23, steps=60)
    assert torch.equal(case.human, before)
    assert stage["train_groups"] == ["train"] and not stage["unknown_as_negative_labels"]
    assert case_coverage(case, [stage])[0] > case_coverage(case, [])[0]


def test_missing_labels_cannot_be_fabricated_as_background():
    case = fixture_case(); case.small_human.zero_()
    candidate = {"programs": [{"op": "field", "name": "student"}],
                 "learning_rate": .01, "temperature": .5, "retention": .1, "hard_positive_weight": 2}
    with pytest.raises(ValueError, match="No expert-positive"):
        fit_candidate([case], [[]], candidate, seed=23, steps=1)


def test_critic_generates_auditable_program_hypotheses_not_named_source_rules():
    case = fixture_case()
    candidates, receipt = propose([case], [[]], [], np.random.default_rng(4), arm="feedback", count=2, pool_size=4)
    assert len(candidates) == 2 and len(receipt["pool"]) == 4
    assert receipt["evidence"]["groups"] == {"train": 0.0}
    assert all(c["hypothesis"]["association_used_for_selection"] for c in candidates)
    assert all(not c["unknown_as_negative_labels"] for c in candidates)


def test_parameter_control_keeps_fixed_basis_instead_of_synthesized_operators():
    c, _ = propose([fixture_case()], [[]], [], np.random.default_rng(4), arm="parameter_only", count=2, pool_size=4)
    assert all(p["op"] == "field" for v in c for p in v["programs"])
    assert all(not v["uses_parent_feedback"] for v in c)


def test_macro_gain_cannot_hide_other_source_regression():
    baseline = rows(.5) + rows(.5, "b")
    candidate = rows(.9) + rows(.49, "b")
    decision = promotion(candidate, baseline, baseline)
    assert not decision["accepted"] and decision["source_regressions"] == ["b"]


def test_promotion_preserves_parent_and_original_baseline():
    baseline = rows(.6)
    assert not promotion(rows(.55), rows(.5), baseline)["accepted"]
    assert promotion(rows(.62), baseline, baseline)["accepted"]
    assert not promotion(rows(.601), baseline, baseline)["accepted"]


def test_budget_violation_refused_even_if_coverage_grows():
    baseline = rows(.5); candidate = rows(.9)
    candidate[0]["selected_pixels"] = 11
    assert promotion(candidate, baseline, baseline)["reason"] == "invalid_metric_or_area_budget"


def test_governor_refuses_missing_duplicate_or_mismatched_rows():
    baseline = rows(.5)
    for bad in [rows(.9)[:-1], rows(.9) + rows(.9)[:1]]:
        with pytest.raises(ValueError):
            promotion(bad, baseline, baseline)


def test_corrupt_readout_is_refused_and_parent_remains_usable():
    case = fixture_case(); old = scores(case, []).clone()
    stage = {"programs": [{"op": "field", "name": "base"}], "weights": [[float("nan"), 0, 0]]}
    with pytest.raises(ValueError):
        scores(case, [stage])
    assert torch.equal(scores(case, []), old)


def test_feedback_diagnosis_uses_current_parent_and_changes_after_success():
    case = fixture_case()
    stage = {"programs": [{"op": "field", "name": "student"}], "weights": [[2., 0., 0.]]}
    before = diagnose([case], [[]]); after = diagnose([case], [[stage]])
    assert after["source_macro_coverage"] > before["source_macro_coverage"]
