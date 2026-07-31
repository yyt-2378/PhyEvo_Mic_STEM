"""Submission-oriented point, region, line, interface, and efficiency metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment


def region_metrics(
    prediction: np.ndarray, truth: np.ndarray
) -> Dict[str, float]:
    predicted = np.asarray(prediction, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("Prediction and truth masks must have equal shape")
    intersection = int(np.sum(predicted & target))
    union = int(np.sum(predicted | target))
    predicted_count = int(np.sum(predicted))
    target_count = int(np.sum(target))
    precision = intersection / max(predicted_count, 1)
    recall = intersection / max(target_count, 1)
    return {
        "iou": intersection / max(union, 1),
        "dice": 2 * intersection / max(predicted_count + target_count, 1),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    if not np.any(values):
        return values.copy()
    return values & ~binary_erosion(values)


def boundary_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    tolerance_pixels: float = 2.0,
) -> Dict[str, float]:
    predicted = np.asarray(prediction, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if predicted.shape != target.shape:
        raise ValueError("Prediction and truth masks must have equal shape")
    if tolerance_pixels < 0:
        raise ValueError("Boundary tolerance cannot be negative")
    predicted_boundary = _boundary(predicted)
    target_boundary = _boundary(target)
    if not np.any(predicted_boundary) or not np.any(target_boundary):
        perfect_empty = not np.any(predicted_boundary) and not np.any(
            target_boundary
        )
        return {
            "boundary_precision": float(perfect_empty),
            "boundary_recall": float(perfect_empty),
            "boundary_f1": float(perfect_empty),
            "average_symmetric_surface_distance": (
                0.0 if perfect_empty else float("inf")
            ),
            "hausdorff_distance_95": (
                0.0 if perfect_empty else float("inf")
            ),
        }
    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_prediction = distance_transform_edt(~predicted_boundary)
    predicted_distances = distance_to_target[predicted_boundary]
    target_distances = distance_to_prediction[target_boundary]
    precision = float(
        np.mean(predicted_distances <= tolerance_pixels)
    )
    recall = float(np.mean(target_distances <= tolerance_pixels))
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": (
            2 * precision * recall / max(precision + recall, 1e-12)
        ),
        "average_symmetric_surface_distance": float(
            0.5
            * (np.mean(predicted_distances) + np.mean(target_distances))
        ),
        "hausdorff_distance_95": float(
            max(
                np.percentile(predicted_distances, 95),
                np.percentile(target_distances, 95),
            )
        ),
    }


def boundary_refinement_comparison(
    coarse_prediction: np.ndarray,
    final_prediction: np.ndarray,
    truth: np.ndarray,
    *,
    tolerance_pixels: float = 2.0,
    physics_core: np.ndarray | None = None,
    negative_reference: np.ndarray | None = None,
    fallback_used: bool = False,
) -> Dict[str, float | bool | Dict[str, float]]:
    """Compare final geometry with its own physics-only starting point."""

    coarse = np.asarray(coarse_prediction, dtype=bool)
    final = np.asarray(final_prediction, dtype=bool)
    target = np.asarray(truth, dtype=bool)
    if coarse.shape != final.shape or final.shape != target.shape:
        raise ValueError("Coarse, final, and truth masks must have equal shape")
    coarse_metrics = boundary_metrics(
        coarse, target, tolerance_pixels=tolerance_pixels
    )
    final_metrics = boundary_metrics(
        final, target, tolerance_pixels=tolerance_pixels
    )
    payload: Dict[str, float | bool | Dict[str, float]] = {
        "coarse": coarse_metrics,
        "final": final_metrics,
        "boundary_f1_delta": (
            final_metrics["boundary_f1"] - coarse_metrics["boundary_f1"]
        ),
        "assd_delta": (
            final_metrics["average_symmetric_surface_distance"]
            - coarse_metrics["average_symmetric_surface_distance"]
        ),
        "hd95_delta": (
            final_metrics["hausdorff_distance_95"]
            - coarse_metrics["hausdorff_distance_95"]
        ),
        "fallback_used": bool(fallback_used),
        "fallback_equality_verified": bool(
            not fallback_used or np.array_equal(final, coarse)
        ),
    }
    if physics_core is not None:
        core = np.asarray(physics_core, dtype=bool)
        if core.shape != final.shape:
            raise ValueError("physics_core must match prediction shape")
        payload["physics_core_recall"] = float(
            np.logical_and(final, core).sum() / max(int(core.sum()), 1)
        )
    if negative_reference is not None:
        negative = np.asarray(negative_reference, dtype=bool)
        if negative.shape != final.shape:
            raise ValueError(
                "negative_reference must match prediction shape"
            )
        payload["negative_leakage"] = float(
            np.logical_and(final, negative).sum()
            / max(int(final.sum()), 1)
        )
    return payload


def expert_correction_metrics(
    responses: Sequence[Any],
) -> Dict[str, Any]:
    """Summarize geometry-correction rate and time by review condition."""

    if not responses:
        raise ValueError("Expert correction metrics require review responses")
    adjustment_actions = {
        "adjust_point",
        "adjust_line",
        "adjust_contour",
    }
    grouped: Dict[str, list[tuple[float, bool]]] = {
        "assisted": [],
        "blinded": [],
    }
    for response in responses:
        duration = float(response.annotation_duration_seconds)
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Review durations must be finite and positive")
        action = getattr(response.action, "value", response.action)
        condition = (
            "assisted"
            if bool(response.model_assistance_shown)
            else "blinded"
        )
        grouped[condition].append(
            (duration, str(action) in adjustment_actions)
        )

    def summarize(values: list[tuple[float, bool]]) -> Dict[str, Any]:
        if not values:
            return {
                "response_count": 0,
                "geometry_correction_count": 0,
                "geometry_correction_rate": None,
                "mean_annotation_duration_seconds": None,
                "median_annotation_duration_seconds": None,
            }
        durations = np.asarray([item[0] for item in values], dtype=float)
        corrections = np.asarray(
            [item[1] for item in values], dtype=bool
        )
        return {
            "response_count": len(values),
            "geometry_correction_count": int(corrections.sum()),
            "geometry_correction_rate": float(corrections.mean()),
            "mean_annotation_duration_seconds": float(durations.mean()),
            "median_annotation_duration_seconds": float(
                np.median(durations)
            ),
        }

    assisted = summarize(grouped["assisted"])
    blinded = summarize(grouped["blinded"])
    assisted_mean = assisted["mean_annotation_duration_seconds"]
    blinded_mean = blinded["mean_annotation_duration_seconds"]
    return {
        "assisted": assisted,
        "blinded": blinded,
        "assisted_minus_blinded_mean_seconds": (
            float(assisted_mean - blinded_mean)
            if assisted_mean is not None and blinded_mean is not None
            else None
        ),
        "interpretation": (
            "A negative assisted-minus-blinded duration indicates lower "
            "review time with model geometry; causal claims require a "
            "counterbalanced reviewer design."
        ),
    }


def site_metrics(
    predicted_xy: np.ndarray,
    truth_xy: np.ndarray,
    *,
    tolerance_pixels: float,
) -> Dict[str, float]:
    predicted = np.asarray(predicted_xy, dtype=float).reshape(-1, 2)
    target = np.asarray(truth_xy, dtype=float).reshape(-1, 2)
    if tolerance_pixels <= 0:
        raise ValueError("Site matching tolerance must be positive")
    if not len(predicted) or not len(target):
        matched = 0
        distances = np.empty(0, dtype=float)
    else:
        pairwise = np.linalg.norm(
            predicted[:, None, :] - target[None, :, :], axis=-1
        )
        rows, columns = linear_sum_assignment(pairwise)
        selected = pairwise[rows, columns] <= tolerance_pixels
        matched = int(np.sum(selected))
        distances = pairwise[rows[selected], columns[selected]]
    precision = matched / max(len(predicted), 1)
    recall = matched / max(len(target), 1)
    return {
        "matched_sites": matched,
        "site_precision": precision,
        "site_recall": recall,
        "site_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "mean_localization_error_pixels": (
            float(np.mean(distances)) if distances.size else float("nan")
        ),
        "p95_localization_error_pixels": (
            float(np.percentile(distances, 95))
            if distances.size
            else float("nan")
        ),
    }


def topology_metrics(
    prediction: np.ndarray, truth: np.ndarray
) -> Dict[str, float]:
    predicted_labels, predicted_components = label(
        np.asarray(prediction, dtype=bool)
    )
    truth_labels, truth_components = label(np.asarray(truth, dtype=bool))
    del predicted_labels, truth_labels
    return {
        "predicted_component_count": int(predicted_components),
        "truth_component_count": int(truth_components),
        "absolute_component_count_error": abs(
            int(predicted_components) - int(truth_components)
        ),
    }


def macro_f1(
    predicted_labels: Sequence[str],
    truth_labels: Sequence[str],
    classes: Sequence[str] | None = None,
) -> float:
    predicted = np.asarray(predicted_labels, dtype=object)
    truth = np.asarray(truth_labels, dtype=object)
    if predicted.shape != truth.shape or not len(predicted):
        raise ValueError("Predicted and truth labels need equal non-zero length")
    class_names = list(
        classes
        if classes is not None
        else sorted(set(predicted.tolist()) | set(truth.tolist()))
    )
    scores = []
    for name in class_names:
        true_positive = int(np.sum((predicted == name) & (truth == name)))
        false_positive = int(np.sum((predicted == name) & (truth != name)))
        false_negative = int(np.sum((predicted != name) & (truth == name)))
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        scores.append(
            2 * precision * recall / max(precision + recall, 1e-12)
        )
    return float(np.mean(scores))


def label_efficiency_auc(
    annotation_costs: Sequence[float],
    performance: Sequence[float],
) -> float:
    costs = np.asarray(annotation_costs, dtype=float)
    scores = np.asarray(performance, dtype=float)
    if costs.shape != scores.shape or len(costs) < 2:
        raise ValueError("Label-efficiency curve needs at least two matched points")
    if np.any(np.diff(costs) <= 0):
        raise ValueError("Annotation costs must increase strictly")
    normalized_cost = (costs - costs[0]) / (costs[-1] - costs[0])
    return float(np.trapz(scores, normalized_cost))


def grouped_bootstrap_interval(
    values_by_group: Mapping[str, Sequence[float]],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap acquisition/sample/lab groups, never individual patches."""

    if resamples < 100:
        raise ValueError("Grouped bootstrap needs at least 100 resamples")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    groups = sorted(values_by_group)
    if len(groups) < 2:
        raise ValueError("Grouped bootstrap needs at least two groups")
    group_means = np.asarray(
        [np.mean(values_by_group[group]) for group in groups], dtype=float
    )
    if not np.all(np.isfinite(group_means)):
        raise ValueError("Grouped metric values must be finite")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(
        group_means,
        size=(resamples, len(group_means)),
        replace=True,
    )
    bootstrap_means = np.mean(sampled, axis=1)
    alpha = 0.5 * (1 - confidence)
    lower, upper = np.quantile(bootstrap_means, [alpha, 1 - alpha])
    return {
        "mean": float(np.mean(group_means)),
        "lower": float(lower),
        "upper": float(upper),
        "confidence": confidence,
        "group_count": len(groups),
    }
