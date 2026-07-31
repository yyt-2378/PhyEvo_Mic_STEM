import numpy as np
import pytest

from tfree_detection.evaluation import (
    boundary_refinement_comparison,
    boundary_metrics,
    expert_correction_metrics,
    grouped_bootstrap_interval,
    label_efficiency_auc,
    macro_f1,
    region_metrics,
    site_metrics,
    topology_metrics,
)
from tfree_detection.review import ReviewAction, ReviewResponse


def test_region_and_boundary_metrics_are_exact_for_identical_masks() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 18:46] = True
    assert region_metrics(mask, mask)["iou"] == 1.0
    boundary = boundary_metrics(mask, mask, tolerance_pixels=1)
    assert boundary["boundary_f1"] == 1.0
    assert boundary["average_symmetric_surface_distance"] == 0.0
    assert boundary["hausdorff_distance_95"] == 0.0
    assert topology_metrics(mask, mask)["absolute_component_count_error"] == 0


def test_boundary_metric_detects_shift_and_site_matching_is_one_to_one() -> None:
    truth = np.zeros((64, 64), dtype=bool)
    truth[20:44, 20:44] = True
    shifted = np.zeros_like(truth)
    shifted[20:44, 24:48] = True
    boundary = boundary_metrics(shifted, truth, tolerance_pixels=1)
    assert boundary["boundary_f1"] < 1
    assert boundary["average_symmetric_surface_distance"] > 0
    assert boundary["hausdorff_distance_95"] > 0

    sites = site_metrics(
        np.asarray([[10.2, 10.1], [20.0, 20.0], [50.0, 50.0]]),
        np.asarray([[10.0, 10.0], [20.4, 20.2]]),
        tolerance_pixels=1.0,
    )
    assert sites["matched_sites"] == 2
    assert sites["site_precision"] == pytest.approx(2 / 3)
    assert sites["site_recall"] == 1.0


def test_boundary_refinement_comparison_quantifies_gain_and_safe_fallback():
    truth = np.zeros((64, 64), dtype=bool)
    truth[16:48, 18:46] = True
    coarse = np.zeros_like(truth)
    coarse[16:48, 22:50] = True
    core = np.zeros_like(truth)
    core[24:40, 26:38] = True

    improved = boundary_refinement_comparison(
        coarse,
        truth,
        truth,
        tolerance_pixels=1,
        physics_core=core,
    )
    assert improved["boundary_f1_delta"] > 0
    assert improved["assd_delta"] < 0
    assert improved["hd95_delta"] < 0
    assert improved["physics_core_recall"] == 1.0

    fallback = boundary_refinement_comparison(
        coarse,
        coarse,
        truth,
        tolerance_pixels=1,
        fallback_used=True,
    )
    assert fallback["fallback_equality_verified"] is True
    assert fallback["boundary_f1_delta"] == 0


def test_expert_correction_metrics_preserve_assistance_condition():
    responses = [
        ReviewResponse(
            response_id="assisted-adjust",
            candidate_id="candidate-a",
            reviewer_id="expert-a",
            action=ReviewAction.ADJUST_CONTOUR,
            defect_labels=("amorphous",),
            annotation_duration_seconds=8.0,
            model_assistance_shown=True,
            geometry_payload={"points_xy": [[0, 0], [1, 0], [1, 1]]},
        ),
        ReviewResponse(
            response_id="blind-accept",
            candidate_id="candidate-b",
            reviewer_id="expert-b",
            action=ReviewAction.ACCEPT,
            defect_labels=("amorphous",),
            annotation_duration_seconds=12.0,
            model_assistance_shown=False,
        ),
    ]
    metrics = expert_correction_metrics(responses)
    assert metrics["assisted"]["geometry_correction_rate"] == 1.0
    assert metrics["blinded"]["geometry_correction_rate"] == 0.0
    assert metrics["assisted_minus_blinded_mean_seconds"] == -4.0


def test_classification_efficiency_and_grouped_interval_contracts() -> None:
    score = macro_f1(
        ["a", "a", "b", "b"],
        ["a", "b", "b", "b"],
    )
    assert 0 < score < 1
    auc = label_efficiency_auc([0, 10, 30], [0.2, 0.6, 0.8])
    assert 0.2 < auc < 0.8
    interval = grouped_bootstrap_interval(
        {
            "acquisition-a": [0.7, 0.8],
            "acquisition-b": [0.5, 0.6],
            "acquisition-c": [0.9],
        },
        resamples=200,
        seed=4,
    )
    assert interval["group_count"] == 3
    assert interval["lower"] <= interval["mean"] <= interval["upper"]
