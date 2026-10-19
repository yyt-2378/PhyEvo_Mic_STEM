import numpy as np
import pytest

from scripts.evaluate_rsi_capabilities import across_seeds, complementarity, run, spatial_controls
from tfree_detection.rsi_evaluation import EvaluationProtocol


def test_spatial_controls_repeat_without_labels_and_obey_budget():
    image = np.zeros((128, 128))
    protocol = EvaluationProtocol()
    a = spatial_controls(image, "point", 63, protocol, 0)
    b = spatial_controls(image, "point", 63, protocol, 0)
    for name in a:
        np.testing.assert_array_equal(a[name].points_xy, b[name].points_xy)
        assert len(a[name].points_xy) <= protocol.point_budget(image.shape)
        if "border" in name:
            width = int(name[-2:])
            xy = a[name].points_xy
            distance = np.minimum.reduce([xy[:, 0], xy[:, 1], 127 - xy[:, 0], 127 - xy[:, 1]])
            assert (distance < width).all()


def test_uniform_area_control_has_exact_budget():
    controls = spatial_controls(np.zeros((64, 80)), "interface", 135, EvaluationProtocol(), 0)
    assert list(controls) == ["control_uniform"]
    assert controls["control_uniform"].mask.sum() == 512


def test_seed_aggregation_preserves_unique_case_count_not_replicates():
    def summary(value):
        return {"strata": [{"dimension": "source_group", "value": "same_acquisition", "geometry": "point",
                            "case_count": 99, "independent_source_group_count": 1,
                            "methods": {"student_expert": {"micro_positive_coverage": value,
                                                           "macro_case_positive_coverage": value}}}]}
    result = across_seeds({23: summary(.1), 41: summary(.2), 67: summary(.3)})[0]
    assert result["unique_case_count"] == 99
    assert result["independent_source_group_count"] == 1
    assert result["seed_count"] == 3
    assert result["methods"]["student_expert"]["micro_positive_coverage_mean"] == pytest.approx(.2)
    assert result["methods"]["student_expert"]["micro_positive_coverage_seed_std"] == pytest.approx(.1)


def test_complementarity_is_descriptive_not_equal_cost_oracle():
    row = {"main_comparison_eligible": True, "geometry": "point",
           "methods": {"physics_budgeted": {"matched_expert_indices": [0, 1]},
                       "student_expert": {"matched_expert_indices": [1, 2, 3]}}}
    result = complementarity(row, {}, {})
    assert result["both_matched_positive_count"] == 1
    assert result["physics_only_positive_count"] == 1
    assert result["student_only_positive_count"] == 2
    assert result["not_a_deployable_oracle_router_or_equal_budget_union"]


def test_existing_output_cannot_be_overwritten(tmp_path):
    output = tmp_path / "existing-evaluation"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        run(tmp_path, output)
