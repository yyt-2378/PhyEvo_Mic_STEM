import copy

import numpy as np
import pytest

from tfree_detection.expert_supervision import (
    closed_interface, group_folds, normalize_image, supervision_from_record, validate_admission,
)


def record(hint="point", action="correct_geometry", points=None):
    return {"revision": {"image_shape": [32, 32], "geometry_hint": hint, "action": action,
                         "geometry": {"points_xy": points or [[3, 4]]}},
            "prediction": {"status": "completed"}, "sample": {"source_partition": "development_test"}}


def test_expert_point_is_supervised_when_teacher_is_empty():
    target = supervision_from_record(record(), np.zeros((32, 32)))
    assert target.positive[4, 3]
    assert target.known[4, 3] == 1
    assert not target.teacher_weight.any()
    assert target.known[20, 20] == 0  # unknown, not an expert negative


def test_human_override_removes_conflicting_teacher_prior_near_point():
    teacher = np.zeros((32, 32), bool)
    teacher[20, 20] = True
    target = supervision_from_record(record(), teacher)
    assert target.teacher_weight[4, 3] == 0
    assert target.teacher_weight[20, 20] == 1


def test_closed_interface_is_band_not_outline():
    points = [[4, 10], [28, 10], [28, 20], [4, 20], [4, 10]]
    assert closed_interface(np.array(points), (32, 32))
    target = supervision_from_record(record("interface", points=points), np.zeros((32, 32)))
    assert target.positive[15, 16]
    assert target.semantic_kind == "interface_band_roi_not_centerline"
    assert target.known[1, 1] == 0


def test_open_interface_remains_trace():
    target = supervision_from_record(record("interface", points=[[2, 10], [28, 10]]), np.zeros((32, 32)))
    assert target.semantic_kind == "positive_interface_trace"
    assert not target.positive[20, 20]


@pytest.mark.parametrize("action", ["reject_prediction", "artifact", "unknown"])
def test_unresolved_action_is_not_background_truth(action):
    target = supervision_from_record(record(action=action), np.ones((32, 32)))
    assert not target.known.any()
    assert not target.teacher_weight.any()


def test_explicit_no_defect_is_the_only_dense_human_negative():
    target = supervision_from_record(record(action="no_defect"), np.ones((32, 32)))
    assert target.known.min() == 1
    assert not target.positive.any()


def test_illegal_source_geometry_fails():
    with pytest.raises(ValueError, match="bounds"):
        supervision_from_record(record(points=[[32, 2]]), np.zeros((32, 32)))


def test_retirement_is_explicit_and_never_admits_locked_or_calibration():
    r = record()
    with pytest.raises(ValueError, match="retirement"):
        validate_admission([r], retire_development=False)
    validate_admission([r], retire_development=True)
    for partition in ("locked_external_test", "calibration"):
        protected = copy.deepcopy(r)
        protected["sample"]["source_partition"] = partition
        with pytest.raises(ValueError, match="protected"):
            validate_admission([protected], retire_development=True)
    assert r["sample"]["source_partition"] == "development_test"


def test_folds_keep_entire_acquisition_together_and_exclude_single_source_task():
    entries = [{"case_number": i, "group": group, "geometry": geometry, "human_known_pixels": 1}
               for i, (group, geometry) in enumerate([("A", "point"), ("A", "point"),
                                                      ("B", "point"), ("C", "region")])]
    folds = group_folds(entries)
    assert len(folds) == 2
    for fold in folds:
        assert fold["held_out_group"] not in fold["train_groups"]
        assert set(fold["train_case_numbers"]).isdisjoint(fold["validation_case_numbers"])
    assert folds[0]["validation_case_numbers"] == [0, 1]


def test_normalization_preserves_shape_and_rejects_nonfinite():
    assert normalize_image(np.ones((8, 8))).shape == (8, 8)
    with pytest.raises(ValueError):
        normalize_image(np.full((8, 8), np.nan))
