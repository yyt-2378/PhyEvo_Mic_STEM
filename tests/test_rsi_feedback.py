import copy

import numpy as np
import pytest

from tfree_detection.rsi_evaluation import EvaluationProtocol
from tfree_detection.rsi_feedback import (
    BRANCHES, FEEDBACK_BRANCH, embedding_reference_context, feedback_predictions,
    fit_feedback_selector, selected_configuration,
)


def test_feedback_reranks_physical_support_without_modifying_it_or_budget():
    image = np.zeros((20, 20))
    teacher = np.zeros_like(image, dtype=bool)
    teacher[:4] = True
    before = teacher.copy()
    score = np.linspace(0, 1, 400).reshape(image.shape)
    results = feedback_predictions(image, teacher, [], score, "interface", protocol=EvaluationProtocol(area_budget_fraction=.1))
    result = results[FEEDBACK_BRANCH]
    assert result.mask.sum() == 40 and result.budget == 40
    assert (result.mask & teacher).sum() == 20
    np.testing.assert_array_equal(teacher, before)
    np.testing.assert_array_equal(results["physics_native"].mask, before)
    assert result.metadata["student_scores_are_physical_evidence"] is False
    assert not np.array_equal(result.mask, results["hybrid_fixed_budget"].mask)


def test_point_feedback_is_same_coordinate_space_and_rescues_empty_teacher():
    score = np.zeros((64,64))
    score[5,6], score[30,31] = .9, .8
    result = feedback_predictions(score, np.zeros_like(score), [], score, "point",
                                  protocol=EvaluationProtocol(point_budget_per_512=128))[FEEDBACK_BRANCH]
    assert result.budget == 2
    np.testing.assert_array_equal(result.points_xy, [[6,5], [31,30]])


def test_embedding_neighbors_never_infer_chemical_identity_or_calibrated_ood():
    records = [{"case_number": 1, "group": "a", "geometry": "point"},
               {"case_number": 2, "group": "a", "geometry": "point"},
               {"case_number": 3, "group": "b", "geometry": "interface"}]
    bank = np.array([[1.,0], [.99,.01], [0,1.]])
    near = embedding_reference_context([1,0], bank, records, "point")
    far = embedding_reference_context([0,1], bank, records, "point")
    assert near["neighbors"][0]["case_number"] == 1
    assert not near["novelty_veto"] and far["novelty_veto"]
    assert near["reference_acquisition_count"] == 1
    assert near["calibrated_ood"] is False and near["chemical_identity_inferred"] is False
    single = embedding_reference_context([0,1], bank, records, "interface")
    assert single["threshold"] is None and single["novelty_veto"]
    assert embedding_reference_context([1,0], bank, records, "line")["reason"] == "no_geometry_reference"
    with pytest.raises(ValueError):
        embedding_reference_context([0,0], bank, records, "point")


def rows():
    result = []
    for group, case in (("a", 1), ("b", 2)):
        for step in (400,1200):
            for seed in (23,41,67):
                scores = {b: .1 for b in BRANCHES}
                scores[FEEDBACK_BRANCH] = .5 if step == 400 else .2
                result.append({"outer_group": "heldout", "group": group, "case_number": case,
                               "model_train_groups": ["c"], "policy": "test", "step": step, "seed": seed,
                               "main_comparison_eligible": True, "geometry": "interface", "teacher_empty": False,
                               "model_geometry_support": ["interface"], "embedding_context": {"novelty_veto": False},
                               "methods": {b: {"positive_coverage": v} for b,v in scores.items()}})
    return result


def test_nested_selection_uses_inner_groups_and_prefers_early_checkpoint_on_inner_evidence():
    selector = fit_feedback_selector(rows(), outer_group="heldout")
    config = selected_configuration(selector, geometry="interface", teacher_empty=False)
    assert config["step"] == 400 and config["branch"] == FEEDBACK_BRANCH
    assert config["embedding_veto"] is False
    assert selected_configuration(selector, geometry="point", teacher_empty=False) is None
    assert selector["selection_uses_outer_labels"] is False


def test_selector_rejects_leakage_duplicates_single_source_and_nonretention():
    data = rows()
    with pytest.raises(ValueError, match="Duplicate"):
        fit_feedback_selector(data+[data[0]], outer_group="heldout")
    corrupted = copy.deepcopy(data)
    corrupted[0]["model_train_groups"].append("heldout")
    with pytest.raises(ValueError, match="leakage"):
        fit_feedback_selector(corrupted, outer_group="heldout")
    one = fit_feedback_selector([r for r in data if r["group"] == "a"], outer_group="heldout")
    assert selected_configuration(one, geometry="interface", teacher_empty=False) is None
    for row in data:
        if row["group"] == "b":
            row["methods"][FEEDBACK_BRANCH]["positive_coverage"] = 0
    nonretained = fit_feedback_selector(data, outer_group="heldout")
    assert selected_configuration(nonretained, geometry="interface", teacher_empty=False) is None
