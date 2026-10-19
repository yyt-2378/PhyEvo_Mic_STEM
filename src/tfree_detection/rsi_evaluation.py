"""Label-independent, equal-budget physics/student comparison for controlled RSI.

This module deliberately does not learn a router. Predictions are constructed
without expert labels; labels enter only ``evaluate_case``. Frozen physics masks
are native *supports*, not dense scores: a budgeted support is a deterministic
coordinate-hash subsample and never grows an empty physics prediction. Sparse
positive annotations cannot establish precision, F1, or an absence of defects.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import laplace
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max


METHODS = ("physics_native", "physics_budgeted", "student_teacher_only",
           "student_expert", "hybrid_fixed_budget")


@dataclass(frozen=True)
class EvaluationProtocol:
    point_budget_per_512: float = 50
    point_tolerance_px: float = 6.0
    area_budget_fraction: float = 0.10
    hybrid_teacher_fraction: float = 0.5
    point_min_distance_px: int = 4

    def __post_init__(self):
        if not math.isfinite(self.point_budget_per_512) or self.point_budget_per_512 <= 0:
            raise ValueError("Point density must be finite and positive")
        if not math.isfinite(self.point_tolerance_px) or self.point_tolerance_px < 0:
            raise ValueError("Point tolerance must be finite and nonnegative")
        if not 0 < self.area_budget_fraction <= 1:
            raise ValueError("Area fraction must be in (0, 1]")
        if not 0 <= self.hybrid_teacher_fraction <= 1:
            raise ValueError("Teacher fraction must be in [0, 1]")
        if self.point_min_distance_px < 1:
            raise ValueError("Point minimum distance must be positive")

    def point_budget(self, shape: tuple[int, int]) -> int:
        # Match the existing expert_student.evaluate_case rounding exactly.
        return max(1, round(self.point_budget_per_512 * math.prod(shape) / 512**2))

    def area_budget(self, shape: tuple[int, int]) -> int:
        return max(1, math.ceil(self.area_budget_fraction * math.prod(shape)))


@dataclass
class Prediction:
    geometry: str
    points_xy: np.ndarray | None = None
    mask: np.ndarray | None = None
    budget: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _finite_plane(value: np.ndarray, name: str, shape=None) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite 2-D plane")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} shape differs from the source image")
    return value


def _xy(value, shape=None) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    if not value.size:
        return np.empty((0, 2), dtype=float)
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("Points must be finite source-pixel x,y pairs")
    if shape is not None and ((value < 0).any() or
                             (value[:, 0] > shape[1] - 1).any() or
                             (value[:, 1] > shape[0] - 1).any()):
        raise ValueError("Point outside source image")
    return value


def physics_points(candidates: Sequence[Mapping[str, Any]], shape) -> np.ndarray:
    """Rank existing centroids only; reliability is not treated as probability."""
    ranked = []
    for index, candidate in enumerate(candidates):
        if "centroid_xy" not in candidate:
            raise ValueError("Native point candidate lacks centroid_xy")
        point = _xy([candidate["centroid_xy"]], shape)[0]
        reliability = float(candidate.get("reliability", 0.0))
        if not math.isfinite(reliability):
            raise ValueError("Nonfinite physics candidate reliability")
        ranked.append((-reliability, float(point[1]), float(point[0]), index, point))
    return _xy([row[-1] for row in sorted(ranked, key=lambda row: row[:4])])


def student_points(score: np.ndarray, budget: int, min_distance: int = 4) -> np.ndarray:
    if budget <= 0:
        return np.empty((0, 2), dtype=float)
    score = _finite_plane(score, "student score")
    yx = peak_local_max(score, min_distance=min_distance, threshold_abs=0.0,
                        exclude_border=False, num_peaks=budget)
    return yx[:, ::-1].astype(float)


def coordinate_hash_order(mask: np.ndarray) -> np.ndarray:
    """Fixed splitmix64 coordinate ranks, independent of annotations/intensity.

    This is only a reproducible uniform-like thinning control, not an invented
    physical confidence ranking. Array-index ties are resolved stably.
    """
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    values = indices.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return indices[np.argsort(values, kind="stable")]


def budgeted_physics_mask(mask: np.ndarray, budget: int) -> np.ndarray:
    mask = _finite_plane(mask, "physics mask").astype(bool)
    output = np.zeros_like(mask)
    if budget > 0:
        output.ravel()[coordinate_hash_order(mask)[:budget]] = True
    return output


def student_mask(score: np.ndarray, budget: int, *, exclude=None) -> np.ndarray:
    score = _finite_plane(score, "student score")
    available = np.ones(score.shape, dtype=bool) if exclude is None else ~np.asarray(exclude, dtype=bool)
    if available.shape != score.shape:
        raise ValueError("Exclusion mask shape mismatch")
    indices = np.flatnonzero(available)
    selected = indices[np.argsort(-score.ravel()[indices], kind="stable")[:max(0, budget)]]
    output = np.zeros(score.shape, dtype=bool)
    output.ravel()[selected] = True
    return output


def _append_unique(selected: list, candidates, limit: int, min_distance: float) -> None:
    for point in candidates:
        if len(selected) >= limit:
            return
        # Chebyshev distance matches peak_local_max's square exclusion region.
        if selected and (np.max(np.abs(np.asarray(selected) - point), axis=1) < min_distance).any():
            continue
        selected.append(point)


def hybrid_points(teacher_xy, expert_score, budget: int, protocol: EvaluationProtocol) -> np.ndarray:
    teacher_xy = _xy(teacher_xy)
    reserved = int(math.floor(budget * protocol.hybrid_teacher_fraction))
    selected: list[np.ndarray] = []
    _append_unique(selected, teacher_xy, reserved, protocol.point_min_distance_px)
    # Oversample to fill places removed by cross-model near-duplicate suppression.
    student_xy = student_points(expert_score, max(budget * 4, budget), protocol.point_min_distance_px)
    _append_unique(selected, student_xy, budget, protocol.point_min_distance_px)
    _append_unique(selected, teacher_xy, budget, protocol.point_min_distance_px)
    return _xy(selected)


def predict_methods(image, teacher_mask, teacher_candidates, teacher_only_score,
                    expert_score, geometry: str, *,
                    protocol: EvaluationProtocol | None = None) -> dict[str, Prediction]:
    """Create all comparison predictions; no labels or held-out group allowed."""
    protocol = protocol or EvaluationProtocol()
    image = _finite_plane(image, "image")
    teacher = _finite_plane(teacher_mask, "physics mask", image.shape).astype(bool)
    scores = {"student_teacher_only": _finite_plane(teacher_only_score, "teacher-only student", image.shape),
              "student_expert": _finite_plane(expert_score, "expert student", image.shape)}
    if geometry not in {"point", "interface", "region", "line"}:
        raise ValueError("Unsupported geometry")
    common = {"selection_uses_expert_labels": False, "calibrated": False}
    if geometry == "point":
        native = physics_points(teacher_candidates, image.shape)
        if len(native) and not teacher.any():
            raise ValueError("Nonempty native candidates contradict an empty physics mask")
        if teacher.any() and not len(native):
            raise ValueError("Point physics mask has no native candidate centroids")
        budget = protocol.point_budget(image.shape)
        result = {
            "physics_native": Prediction(geometry, points_xy=native.copy(), metadata={
                **common, "selection": "all_original_centroids", "budget_matched": False}),
            "physics_budgeted": Prediction(geometry, points_xy=native[:budget].copy(), budget=budget,
                metadata={**common, "selection": "native_reliability_then_y_x", "budget_matched": True}),
        }
        for name, score in scores.items():
            result[name] = Prediction(geometry, points_xy=student_points(
                score, budget, protocol.point_min_distance_px), budget=budget,
                metadata={**common, "selection": "student_local_maxima", "budget_matched": True})
        result["hybrid_fixed_budget"] = Prediction(geometry, points_xy=hybrid_points(
            native, scores["student_expert"], budget, protocol), budget=budget,
            metadata={**common, "selection": "fixed_teacher_quota_then_student_then_teacher_backfill",
                      "teacher_fraction": protocol.hybrid_teacher_fraction, "budget_matched": True})
    else:
        budget = protocol.area_budget(image.shape)
        result = {
            "physics_native": Prediction(geometry, mask=teacher.copy(), metadata={
                **common, "selection": "all_original_support", "budget_matched": False}),
            "physics_budgeted": Prediction(geometry, mask=budgeted_physics_mask(teacher, budget), budget=budget,
                metadata={**common, "selection": "support_only_coordinate_hash_thinning_not_dense_score",
                          "budget_matched": True}),
        }
        for name, score in scores.items():
            result[name] = Prediction(geometry, mask=student_mask(score, budget), budget=budget,
                metadata={**common, "selection": "student_top_score_pixels", "budget_matched": True})
        reserved = int(math.floor(budget * protocol.hybrid_teacher_fraction))
        hybrid = budgeted_physics_mask(teacher, reserved)
        hybrid |= student_mask(scores["student_expert"], budget - int(hybrid.sum()), exclude=hybrid)
        result["hybrid_fixed_budget"] = Prediction(geometry, mask=hybrid, budget=budget,
            metadata={**common, "selection": "fixed_teacher_support_quota_then_student_backfill",
                      "teacher_fraction": protocol.hybrid_teacher_fraction, "budget_matched": True})
    return result


def observable_features(image: np.ndarray, teacher_mask: np.ndarray) -> dict[str, float]:
    """Input-only diagnostics, not calibrated SNR/crystallinity estimates."""
    image = _finite_plane(image, "image").astype(float)
    teacher = _finite_plane(teacher_mask, "physics mask", image.shape).astype(bool)
    p05, p95 = np.percentile(image, [5, 95])
    lap = laplace(image)
    noise = np.median(np.abs(lap - np.median(lap))) / (0.67448975 * math.sqrt(20))
    # Fixed-size striding caps diagnostic work without resizing model inputs.
    step = max(1, math.ceil(max(image.shape) / 512))
    sample = image[::step, ::step]
    centered = (sample - sample.mean()) * np.outer(np.hanning(sample.shape[0]), np.hanning(sample.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered)))**2
    cy, cx = np.array(power.shape) // 2
    power[max(0, cy - 2):cy + 3, max(0, cx - 2):cx + 3] = 0
    flat = power.ravel()
    top = np.sort(flat)[-min(20, flat.size):].sum()
    return {"robust_contrast_p95_p05": float(p95 - p05),
            "laplacian_noise_proxy": float(noise),
            "fft_top20_power_fraction": float(top / max(float(flat.sum()), 1e-30)),
            "physics_area_fraction": float(teacher.mean()),
            "image_height_px": int(image.shape[0]), "image_width_px": int(image.shape[1])}


def matched_expert_indices(predicted_xy, expert_xy, tolerance: float = 6.0) -> list[int]:
    predicted, expert = _xy(predicted_xy), _xy(expert_xy)
    if not len(predicted) or not len(expert):
        return []
    distances = cdist(predicted, expert)
    cost = np.where(distances <= tolerance, distances / max(tolerance, 1e-8), 1e6)
    row, col = linear_sum_assignment(cost)
    return sorted(int(c) for r, c in zip(row, col) if distances[r, c] <= tolerance)


def evaluate_case(entry: Mapping, plane: Mapping, predictions: Mapping[str, Prediction], *,
                  protocol: EvaluationProtocol | None = None) -> dict:
    protocol = protocol or EvaluationProtocol()
    image = _finite_plane(plane["image"], "image")
    teacher = _finite_plane(plane["teacher"], "physics mask", image.shape).astype(bool)
    human = _finite_plane(plane["human"], "human support", image.shape).astype(bool)
    geometry = entry["geometry"]
    # Accepted teacher geometry is circular evaluation truth and kept audit-only.
    eligible = entry["action"] == "correct_geometry" and bool(human.any())
    points = _xy(entry.get("points_xy", []), image.shape) if geometry == "point" else None
    if geometry == "point" and eligible and not len(points):
        raise ValueError("Corrected point annotation has no points")
    result = {"case_number": entry["case_number"], "group": entry["group"], "geometry": geometry,
              "action": entry["action"], "semantic_kind": entry.get("semantic_kind"),
              "main_comparison_eligible": eligible,
              "exclusion_reason": None if eligible else "not_independent_corrected_positive_geometry",
              "teacher_empty": not bool(teacher.any()),
              "observable_features": observable_features(image, teacher),
              "protocol": asdict(protocol), "methods": {}}
    for name, prediction in predictions.items():
        if prediction.geometry != geometry:
            raise ValueError("Prediction/annotation geometry mismatch")
        metric = {"precision": None, "f1": None, "negative_label_completeness_verified": False,
                  "budget": prediction.budget, "selection_metadata": dict(prediction.metadata),
                  "annotated_positive_count": 0, "matched_positive_count": 0,
                  "positive_coverage": None}
        if geometry == "point":
            predicted = _xy(prediction.points_xy, image.shape)
            metric["candidate_count"] = len(predicted)
            if prediction.budget is not None and len(predicted) > prediction.budget:
                raise ValueError("Point prediction exceeds declared budget")
            if eligible:
                matches = matched_expert_indices(predicted, points, protocol.point_tolerance_px)
                metric.update({"metric": "marked_point_recall", "annotated_positive_count": len(points),
                               "matched_positive_count": len(matches), "matched_expert_indices": matches,
                               "positive_coverage": len(matches) / len(points)})
        else:
            mask = _finite_plane(prediction.mask, "prediction support", image.shape).astype(bool)
            metric.update({"selected_pixels": int(mask.sum()), "selected_area_fraction": float(mask.mean())})
            if prediction.budget is not None and int(mask.sum()) > prediction.budget:
                raise ValueError("Area prediction exceeds declared budget")
            if eligible:
                matches = int((mask & human).sum())
                metric.update({"metric": "annotated_positive_roi_coverage",
                               "annotated_positive_count": int(human.sum()),
                               "matched_positive_count": matches, "positive_coverage": matches / int(human.sum())})
        result["methods"][name] = metric
    if geometry == "point" and eligible:
        distance = np.minimum.reduce([points[:, 0], points[:, 1], image.shape[1] - 1 - points[:, 0],
                                      image.shape[0] - 1 - points[:, 1]])
        result["annotation_diagnostics_not_routing_features"] = {"fraction_within_24px_border": float((distance < 24).mean())}
    return result


def aggregate_capabilities(case_results: Sequence[Mapping]) -> dict:
    """Descriptive strata only; seeds/frames are not independent acquisitions."""
    rows: dict[tuple, list] = {}
    for case in case_results:
        if not case["main_comparison_eligible"]:
            continue
        geometry = case["geometry"]
        features = case["observable_features"]
        strata = [("source_group", case["group"]), ("geometry", geometry),
                  ("teacher_empty", str(case["teacher_empty"]).lower()),
                  ("noise_proxy_fixed_bin", "ge_0.08" if features["laplacian_noise_proxy"] >= 0.08 else "lt_0.08"),
                  ("fft_concentration_fixed_bin", "ge_0.20" if features["fft_top20_power_fraction"] >= 0.20 else "lt_0.20")]
        # Do not pool incompatible point counts and area pixels in one statistic.
        for dimension, value in strata:
            rows.setdefault((dimension, value, geometry), []).append(case)
    summaries = []
    for (dimension, value, geometry), cases in sorted(rows.items()):
        methods = {}
        for name in sorted(set.intersection(*(set(c["methods"]) for c in cases))):
            values = [c["methods"][name] for c in cases]
            count = sum(v["annotated_positive_count"] for v in values)
            matches = sum(v["matched_positive_count"] for v in values)
            methods[name] = {"matched_positive_count": matches, "annotated_positive_count": count,
                             "micro_positive_coverage": matches / count if count else None,
                             "macro_case_positive_coverage": float(np.mean([v["positive_coverage"] for v in values])),
                             "precision": None, "f1": None}
        summaries.append({"dimension": dimension, "value": value, "geometry": geometry,
                          "case_count": len(cases), "independent_source_group_count": len({c["group"] for c in cases}),
                          "methods": methods})
    return {"strata": summaries, "included_case_count": sum(bool(c["main_comparison_eligible"]) for c in case_results),
            "excluded_case_count": sum(not c["main_comparison_eligible"] for c in case_results),
            "inference": "descriptive_assisted_silver_not_confirmatory_significance",
            "frames_and_seeds_are_not_independent_acquisitions": True,
            "physics_native_is_not_budget_matched": True,
            "physics_area_budget_is_support_thinning_not_dense_model_ranking": True}
