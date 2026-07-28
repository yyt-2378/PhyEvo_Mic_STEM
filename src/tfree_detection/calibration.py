"""Gold-only candidate calibration, risk-coverage, and split conformal sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize_scalar

from .evidence import LabelTier
from .governance import DataPartition


@dataclass(frozen=True)
class CalibrationExample:
    candidate_id: str
    raw_score: float
    is_true_defect: bool
    partition: DataPartition
    label_tier: LabelTier
    split_group_id: str
    domain_id: str = "unspecified"

    def validate(self) -> None:
        if self.partition != DataPartition.CALIBRATION:
            raise ValueError(
                "Calibration fitting accepts only the calibration partition"
            )
        if self.label_tier != LabelTier.GOLD:
            raise ValueError("Calibration fitting accepts only Gold labels")
        if not np.isfinite(self.raw_score):
            raise ValueError("Calibration score must be finite")
        if not self.split_group_id:
            raise ValueError("Calibration split_group_id is required")


def _arrays(
    examples: Iterable[CalibrationExample],
) -> tuple[np.ndarray, np.ndarray]:
    values = list(examples)
    if not values:
        raise ValueError("At least one calibration example is required")
    for example in values:
        example.validate()
    scores = np.asarray([item.raw_score for item in values], dtype=float)
    labels = np.asarray([item.is_true_defect for item in values], dtype=float)
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration requires positive and negative Gold examples")
    return scores, labels


@dataclass
class TemperatureCalibrator:
    """Platt-style scalar temperature for raw logits."""

    temperature: float = 1.0
    fitted: bool = False

    def fit(
        self, examples: Sequence[CalibrationExample]
    ) -> "TemperatureCalibrator":
        logits, labels = _arrays(examples)

        def objective(log_temperature: float) -> float:
            temperature = float(np.exp(log_temperature))
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(logits / temperature, -40, 40))
            )
            probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
            return float(
                -np.mean(
                    labels * np.log(probabilities)
                    + (1 - labels) * np.log(1 - probabilities)
                )
            )

        optimum = minimize_scalar(
            objective, bounds=(-5.0, 5.0), method="bounded"
        )
        if not optimum.success:
            raise RuntimeError("Temperature calibration did not converge")
        self.temperature = float(np.exp(optimum.x))
        self.fitted = True
        return self

    def predict(self, raw_scores: np.ndarray | Sequence[float]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("TemperatureCalibrator must be fitted first")
        values = np.asarray(raw_scores, dtype=float)
        return (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(values / max(self.temperature, 1e-8), -40, 40)
                )
            )
        ).astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "temperature_scaling",
            "temperature": self.temperature,
            "fitted": self.fitted,
            "input_semantics": "raw_logit",
        }


@dataclass
class IsotonicCalibrator:
    """Dependency-light pool-adjacent-violators isotonic calibration."""

    thresholds: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    probabilities: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    fitted: bool = False

    def fit(
        self, examples: Sequence[CalibrationExample]
    ) -> "IsotonicCalibrator":
        scores, labels = _arrays(examples)
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_labels = labels[order]

        unique_scores, inverse = np.unique(sorted_scores, return_inverse=True)
        sums = np.bincount(inverse, weights=sorted_labels).astype(float)
        weights = np.bincount(inverse).astype(float)
        blocks = [
            [index, index, sums[index], weights[index]]
            for index in range(len(unique_scores))
        ]
        index = 0
        while index < len(blocks) - 1:
            left_mean = blocks[index][2] / blocks[index][3]
            right_mean = blocks[index + 1][2] / blocks[index + 1][3]
            if left_mean <= right_mean:
                index += 1
                continue
            blocks[index][1] = blocks[index + 1][1]
            blocks[index][2] += blocks[index + 1][2]
            blocks[index][3] += blocks[index + 1][3]
            blocks.pop(index + 1)
            index = max(index - 1, 0)

        probabilities = np.zeros(len(unique_scores), dtype=float)
        for start, end, label_sum, weight in blocks:
            probabilities[int(start) : int(end) + 1] = label_sum / weight
        self.thresholds = unique_scores
        self.probabilities = probabilities
        self.fitted = True
        return self

    def predict(self, raw_scores: np.ndarray | Sequence[float]) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted first")
        values = np.asarray(raw_scores, dtype=float)
        indices = np.searchsorted(self.thresholds, values, side="right") - 1
        indices = np.clip(indices, 0, len(self.probabilities) - 1)
        return self.probabilities[indices].astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "isotonic_pava",
            "thresholds": self.thresholds.tolist(),
            "probabilities": self.probabilities.tolist(),
            "fitted": self.fitted,
            "input_semantics": "monotonic_candidate_score",
        }


@dataclass
class SplitConformalBinary:
    """Binary prediction sets under the calibration/test exchangeability assumption."""

    alpha: float = 0.10
    quantile: float = 1.0
    fitted: bool = False

    def fit(
        self,
        probabilities: Sequence[float] | np.ndarray,
        labels: Sequence[bool] | np.ndarray,
    ) -> "SplitConformalBinary":
        values = np.asarray(probabilities, dtype=float)
        truth = np.asarray(labels, dtype=bool)
        if len(values) != len(truth) or not len(values):
            raise ValueError("Probabilities and labels must have equal non-zero length")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if np.any(values < 0) or np.any(values > 1):
            raise ValueError("Probabilities must be in [0, 1]")
        true_class_probability = np.where(truth, values, 1 - values)
        nonconformity = 1 - true_class_probability
        rank = int(np.ceil((len(values) + 1) * (1 - self.alpha)))
        rank = min(max(rank, 1), len(values))
        self.quantile = float(np.partition(nonconformity, rank - 1)[rank - 1])
        self.fitted = True
        return self

    def predict_sets(
        self, probabilities: Sequence[float] | np.ndarray
    ) -> list[tuple[int, ...]]:
        if not self.fitted:
            raise RuntimeError("SplitConformalBinary must be fitted first")
        values = np.asarray(probabilities, dtype=float)
        output: list[tuple[int, ...]] = []
        for probability in values:
            labels = []
            if probability <= self.quantile:
                labels.append(0)
            if 1 - probability <= self.quantile:
                labels.append(1)
            output.append(tuple(labels))
        return output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "split_conformal_binary",
            "alpha": self.alpha,
            "quantile": self.quantile,
            "fitted": self.fitted,
            "assumption": (
                "Calibration and deployment examples must be exchangeable; "
                "domain shift requires OOD detection and recalibration."
            ),
        }


def brier_score(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=float)
    if values.shape != truth.shape:
        raise ValueError("Probabilities and labels must have the same shape")
    return float(np.mean((values - truth) ** 2))


def expected_calibration_error(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    bins: int = 10,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=float)
    if values.shape != truth.shape or not values.size:
        raise ValueError("Probabilities and labels must have equal non-zero shape")
    if bins < 2:
        raise ValueError("ECE requires at least two bins")
    edges = np.linspace(0, 1, bins + 1)
    total = len(values)
    error = 0.0
    for index in range(bins):
        selected = (values >= edges[index]) & (
            values <= edges[index + 1]
            if index == bins - 1
            else values < edges[index + 1]
        )
        if not np.any(selected):
            continue
        error += (
            np.sum(selected)
            / total
            * abs(float(np.mean(values[selected]) - np.mean(truth[selected])))
        )
    return float(error)


def risk_coverage_curve(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
) -> Dict[str, list[float]]:
    values = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels, dtype=bool)
    if values.shape != truth.shape or not values.size:
        raise ValueError("Probabilities and labels must have equal non-zero shape")
    confidence = np.maximum(values, 1 - values)
    predictions = values >= 0.5
    order = np.argsort(confidence)[::-1]
    correct = (predictions[order] == truth[order]).astype(float)
    accepted = np.arange(1, len(values) + 1)
    risk = 1 - np.cumsum(correct) / accepted
    return {
        "coverage": (accepted / len(values)).astype(float).tolist(),
        "risk": risk.astype(float).tolist(),
        "confidence_threshold": confidence[order].astype(float).tolist(),
    }


def calibration_report(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[bool] | np.ndarray,
    *,
    bins: int = 10,
) -> Dict[str, Any]:
    return {
        "sample_count": int(len(probabilities)),
        "brier_score": brier_score(probabilities, labels),
        "expected_calibration_error": expected_calibration_error(
            probabilities, labels, bins=bins
        ),
        "risk_coverage": risk_coverage_curve(probabilities, labels),
    }
