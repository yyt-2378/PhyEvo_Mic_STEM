"""Fair-budget and label-leakage invariants for the capability comparison."""
import copy

import numpy as np
import pytest

from tfree_detection.rsi_evaluation import (
    EvaluationProtocol, Prediction, aggregate_capabilities, budgeted_physics_mask,
    coordinate_hash_order, evaluate_case, matched_expert_indices, observable_features,
    physics_points, predict_methods, student_mask, student_points,
)


def fixture(geometry="point", *, empty=False):
    image = np.random.default_rng(8).random((64, 64)).astype(np.float32)
    teacher = np.zeros_like(image, dtype=bool)
    candidates = []
    if not empty:
        teacher[10:16, 10:16] = True
        teacher[30:36, 30:36] = True
        candidates = [{"centroid_xy": [12., 12.], "reliability": 0.8},
                      {"centroid_xy": [32., 32.], "reliability": 0.4}]
    score = np.zeros_like(image)
    for x, y, value in [(12, 12, 0.9), (52, 52, 0.8), (42, 22, 0.7), (22, 42, 0.6)]:
        score[y, x] = value
    human = np.zeros_like(teacher)
    human[52, 52] = True
    entry = {"case_number": 1, "group": "g", "geometry": geometry,
             "action": "correct_geometry", "semantic_kind": "sparse_positive_points",
             "points_xy": [[52, 52]]}
    plane = {"image": image, "teacher": teacher, "human": human}
    protocol = EvaluationProtocol(point_budget_per_512=256, area_budget_fraction=0.10)
    return entry, plane, candidates, score, protocol


def predictions(geometry="point", *, empty=False, protocol=None):
    entry, plane, candidates, score, default = fixture(geometry, empty=empty)
    protocol = protocol or default
    outputs = predict_methods(plane["image"], plane["teacher"], candidates,
                              score * 0.5, score, geometry, protocol=protocol)
    return entry, plane, outputs, protocol


@pytest.mark.parametrize("density", [20, 50, 100])
def test_point_budget_matches_previous_round(density):
    protocol = EvaluationProtocol(point_budget_per_512=density)
    assert protocol.point_budget((512, 512)) == density
    assert protocol.point_budget((1024, 1024)) == 4 * density
    assert protocol.point_budget((123, 157)) == max(1, round(density * 123 * 157 / 512**2))


@pytest.mark.parametrize("fraction", [0.05, 0.10, 0.20])
def test_area_budget_support_is_bounded_and_empty_never_filled(fraction):
    p = EvaluationProtocol(area_budget_fraction=fraction)
    entry, plane, output, _ = predictions("interface", empty=True, protocol=p)
    for name, prediction in output.items():
        if name.startswith("physics"):
            assert not prediction.mask.any()
        else:
            assert prediction.mask.sum() == p.area_budget(plane["image"].shape)
    np.testing.assert_array_equal(output["hybrid_fixed_budget"].mask, output["student_expert"].mask)


def test_point_empty_teacher_has_no_fabricated_points():
    _, _, output, _ = predictions(empty=True)
    assert len(output["physics_native"].points_xy) == len(output["physics_budgeted"].points_xy) == 0
    np.testing.assert_array_equal(output["hybrid_fixed_budget"].points_xy, output["student_expert"].points_xy)


def test_native_centroids_preserved_rank_is_reliability_then_coordinate():
    candidates = [{"centroid_xy": [7., 8.], "reliability": 0.5},
                  {"centroid_xy": [9., 3.], "reliability": 0.5},
                  {"centroid_xy": [5., 2.], "reliability": 0.1},
                  {"centroid_xy": [3., 3.], "reliability": 0.5}]
    np.testing.assert_array_equal(physics_points(candidates, (10, 10)), [[3, 3], [9, 3], [7, 8], [5, 2]])


def test_hybrid_point_duplicates_suppressed_and_total_budget_retained():
    _, _, output, protocol = predictions()
    hybrid = output["hybrid_fixed_budget"].points_xy
    assert len(hybrid) == 4
    distances = np.max(np.abs(hybrid[:, None] - hybrid[None]), axis=-1)
    assert (distances[np.triu_indices(len(hybrid), 1)] >= protocol.point_min_distance_px).all()
    assert sum(np.all(hybrid == [12, 12], axis=1)) == 1


def test_hybrid_area_does_not_double_budget_or_mutate_support():
    entry, plane, candidates, score, protocol = fixture("interface")
    before = plane["teacher"].copy()
    outputs = predict_methods(plane["image"], plane["teacher"], candidates, score, score,
                              "interface", protocol=protocol)
    assert outputs["hybrid_fixed_budget"].mask.sum() == protocol.area_budget(score.shape)
    assert np.all(outputs["physics_budgeted"].mask <= before)
    np.testing.assert_array_equal(plane["teacher"], before)


def test_coordinate_hash_thinning_is_stable_not_top_left_and_nested():
    mask = np.ones((40, 50), bool)
    first = budgeted_physics_mask(mask, 100)
    second = budgeted_physics_mask(mask, 200)
    np.testing.assert_array_equal(first, budgeted_physics_mask(mask, 100))
    assert np.all(first <= second)
    assert first.sum() == 100
    assert first[20:].sum() > 20
    assert len(np.unique(coordinate_hash_order(mask))) == mask.size


def test_score_ties_are_stable_but_not_claimed_calibration():
    score = np.ones((10, 10), dtype=float)
    a = student_mask(score, 10)
    np.testing.assert_array_equal(a, student_mask(score, 10))
    assert a.sum() == 10
    _, _, output, _ = predictions("interface")
    assert all(prediction.metadata["calibrated"] is False for prediction in output.values())


def test_matching_is_one_to_one_and_uses_pixel_tolerance():
    assert matched_expert_indices([[5, 5]], [[5, 5], [6, 5]]) == [0]
    assert matched_expert_indices([[5, 5]], [[11, 5]], tolerance=6) == [0]
    assert matched_expert_indices([[5, 5]], [[11.01, 5]], tolerance=6) == []


def test_evaluation_labels_do_not_change_prediction_selection():
    entry, plane, output, protocol = predictions()
    before = {name: prediction.points_xy.copy() for name, prediction in output.items()}
    report = evaluate_case(entry, plane, output, protocol=protocol)
    changed_entry = {**entry, "points_xy": [[1, 1]]}
    changed_plane = {**plane, "human": np.eye(64, dtype=bool)}
    other = evaluate_case(changed_entry, changed_plane, output, protocol=protocol)
    for name, prediction in output.items():
        np.testing.assert_array_equal(before[name], prediction.points_xy)
        assert report["methods"][name]["precision"] is None
        assert report["methods"][name]["f1"] is None
    assert report["observable_features"] == other["observable_features"]
    assert report["methods"]["student_expert"]["positive_coverage"] != other["methods"]["student_expert"]["positive_coverage"]


def test_accepted_teacher_truth_is_excluded_from_primary_comparison():
    entry, plane, output, protocol = predictions()
    entry.update(action="accept_prediction", semantic_kind="accepted_teacher_positive", points_xy=[])
    plane["human"] = plane["teacher"].copy()
    report = evaluate_case(entry, plane, output, protocol=protocol)
    assert not report["main_comparison_eligible"]
    assert all(m["positive_coverage"] is None for m in report["methods"].values())
    assert aggregate_capabilities([report])["excluded_case_count"] == 1


def test_group_aggregation_does_not_pool_points_and_pixels_or_invent_independence():
    reports = []
    for geometry in ("point", "interface"):
        entry, plane, output, protocol = predictions(geometry)
        report = evaluate_case(entry, plane, output, protocol=protocol)
        reports.append(report)
    result = aggregate_capabilities(reports)
    strata = [row for row in result["strata"] if row["dimension"] == "source_group"]
    assert len(strata) == 2
    assert {row["geometry"] for row in strata} == {"point", "interface"}
    assert all(row["independent_source_group_count"] == 1 for row in strata)
    assert result["frames_and_seeds_are_not_independent_acquisitions"]


def test_native_area_cost_is_separate_from_fixed_budget_cost():
    entry, plane, output, protocol = predictions("interface")
    report = evaluate_case(entry, plane, output, protocol=protocol)
    assert report["methods"]["physics_native"]["budget"] is None
    assert not report["methods"]["physics_native"]["selection_metadata"]["budget_matched"]
    assert report["methods"]["student_expert"]["selected_area_fraction"] >= 0.10


def test_observable_features_are_finite_on_constant_image():
    features = observable_features(np.zeros((32, 32)), np.zeros((32, 32), bool))
    assert all(np.isfinite(value) for value in features.values())
    assert features["fft_top20_power_fraction"] == features["laplacian_noise_proxy"] == 0


@pytest.mark.parametrize("kwargs", [{"area_budget_fraction": 0}, {"area_budget_fraction": 1.1},
                                    {"hybrid_teacher_fraction": -0.1}, {"point_budget_per_512": np.nan},
                                    {"point_tolerance_px": -1}, {"point_min_distance_px": 0}])
def test_invalid_protocol_rejected(kwargs):
    with pytest.raises(ValueError):
        EvaluationProtocol(**kwargs)


def test_inconsistent_teacher_candidates_rejected_not_silently_reconstructed():
    _, plane, candidates, score, protocol = fixture()
    with pytest.raises(ValueError, match="no native"):
        predict_methods(plane["image"], plane["teacher"], [], score, score, "point", protocol=protocol)
    with pytest.raises(ValueError, match="empty physics"):
        predict_methods(plane["image"], np.zeros_like(plane["teacher"]), candidates,
                        score, score, "point", protocol=protocol)


def test_malformed_inputs_and_overbudget_predictions_rejected():
    entry, plane, output, protocol = predictions()
    malformed = copy.deepcopy(output)
    malformed["student_expert"].points_xy[0] = [-1, 10]
    with pytest.raises(ValueError, match="outside"):
        evaluate_case(entry, plane, malformed, protocol=protocol)
    output["student_expert"].budget = 0
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_case(entry, plane, output, protocol=protocol)
    with pytest.raises(ValueError, match="finite"):
        student_points(np.full((10, 10), np.nan), 4)
