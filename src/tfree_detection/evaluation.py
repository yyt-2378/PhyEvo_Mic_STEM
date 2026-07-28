"""Submission-oriented point, region, line, interface, and efficiency metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

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
