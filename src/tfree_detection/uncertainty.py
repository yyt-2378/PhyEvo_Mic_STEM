"""Domain-shift detection and explicit uncertainty decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence

import numpy as np
from scipy.stats import rankdata

from .governance import DataPartition


@dataclass(frozen=True)
class ReferenceEmbedding:
    asset_id: str
    embedding: np.ndarray
    partition: DataPartition
    split_group_id: str

    def validate_for_fit(self) -> None:
        if self.partition != DataPartition.TRAIN:
            raise ValueError("OOD reference fitting accepts only train assets")
        if not self.split_group_id:
            raise ValueError("OOD reference split_group_id is required")
        if self.embedding.ndim != 1 or not np.all(np.isfinite(self.embedding)):
            raise ValueError("OOD reference embedding must be finite and 1-D")


@dataclass(frozen=True)
class OODCalibrationEmbedding:
    asset_id: str
    embedding: np.ndarray
    partition: DataPartition
    is_in_domain: bool

    def validate(self) -> None:
        if self.partition != DataPartition.CALIBRATION:
            raise ValueError(
                "OOD threshold calibration accepts only calibration assets"
            )
        if self.embedding.ndim != 1 or not np.all(np.isfinite(self.embedding)):
            raise ValueError("OOD calibration embedding must be finite and 1-D")


@dataclass
class MahalanobisOODDetector:
    """Shrinkage Mahalanobis detector with a held-out in-domain threshold."""

    shrinkage: float = 0.10
    threshold_quantile: float = 0.95
    center: np.ndarray | None = None
    precision: np.ndarray | None = None
    threshold: float | None = None
    embedding_dimension: int | None = None

    def fit(
        self, references: Sequence[ReferenceEmbedding]
    ) -> "MahalanobisOODDetector":
        if len(references) < 2:
            raise ValueError("OOD reference fitting requires at least two assets")
        for reference in references:
            reference.validate_for_fit()
        matrix = np.stack([reference.embedding for reference in references])
        if len({row.shape for row in matrix}) != 1:
            raise ValueError("OOD reference embeddings have inconsistent shapes")
        self.center = np.mean(matrix, axis=0)
        centered = matrix - self.center
        covariance = np.atleast_2d(np.cov(centered, rowvar=False, bias=False))
        if covariance.shape != (matrix.shape[1], matrix.shape[1]):
            covariance = np.eye(matrix.shape[1], dtype=float)
        diagonal = np.diag(np.diag(covariance))
        scale = float(np.trace(covariance) / max(matrix.shape[1], 1))
        regularizer = max(scale, 1e-6) * np.eye(matrix.shape[1])
        covariance = (
            (1 - self.shrinkage) * covariance
            + self.shrinkage * diagonal
            + 1e-6 * regularizer
        )
        self.precision = np.linalg.pinv(covariance)
        self.embedding_dimension = int(matrix.shape[1])
        self.threshold = None
        return self

    def distances(
        self, embeddings: np.ndarray | Sequence[np.ndarray]
    ) -> np.ndarray:
        if self.center is None or self.precision is None:
            raise RuntimeError("MahalanobisOODDetector must be fitted first")
        matrix = np.asarray(embeddings, dtype=float)
        if matrix.ndim == 1:
            matrix = matrix[None, :]
        if matrix.shape[1] != self.embedding_dimension:
            raise ValueError("OOD embedding dimension does not match the reference")
        centered = matrix - self.center
        squared = np.einsum(
            "ni,ij,nj->n", centered, self.precision, centered
        )
        return np.sqrt(np.clip(squared, 0, None))

    def calibrate_threshold(
        self, examples: Sequence[OODCalibrationEmbedding]
    ) -> "MahalanobisOODDetector":
        if not 0.5 < self.threshold_quantile < 1:
            raise ValueError("OOD threshold_quantile must be in (0.5, 1)")
        if not examples:
            raise ValueError("OOD threshold calibration requires held-out assets")
        for example in examples:
            example.validate()
        in_domain = [example for example in examples if example.is_in_domain]
        if not in_domain:
            raise ValueError("OOD threshold needs held-out in-domain examples")
        distances = self.distances(
            np.stack([example.embedding for example in in_domain])
        )
        self.threshold = float(
            np.quantile(distances, self.threshold_quantile, method="higher")
        )
        self.threshold = max(self.threshold, 1e-8)
        return self

    def ood_scores(
        self, embeddings: np.ndarray | Sequence[np.ndarray]
    ) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("OOD threshold must be calibrated first")
        distance = self.distances(embeddings)
        return (distance / self.threshold).astype(np.float32)

    def predict(
        self, embeddings: np.ndarray | Sequence[np.ndarray]
    ) -> np.ndarray:
        return self.ood_scores(embeddings) > 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": "shrinkage_mahalanobis",
            "shrinkage": self.shrinkage,
            "threshold_quantile": self.threshold_quantile,
            "threshold": self.threshold,
            "embedding_dimension": self.embedding_dimension,
            "threshold_source": "held_out_in_domain_calibration",
        }


def roc_auc(
    scores: Sequence[float] | np.ndarray,
    is_positive: Sequence[bool] | np.ndarray,
) -> float:
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(is_positive, dtype=bool)
    positives = int(np.sum(labels))
    negatives = int(np.sum(~labels))
    if values.shape != labels.shape or not positives or not negatives:
        raise ValueError("ROC AUC requires matched positive and negative examples")
    ranks = rankdata(values, method="average")
    positive_rank_sum = float(np.sum(ranks[labels]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def average_precision(
    scores: Sequence[float] | np.ndarray,
    is_positive: Sequence[bool] | np.ndarray,
) -> float:
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(is_positive, dtype=bool)
    if values.shape != labels.shape or not np.any(labels):
        raise ValueError("Average precision requires matched positive examples")
    order = np.argsort(values)[::-1]
    sorted_labels = labels[order].astype(float)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float(np.sum(precision * sorted_labels) / np.sum(sorted_labels))


def ood_report(
    scores: Sequence[float] | np.ndarray,
    is_out_of_domain: Sequence[bool] | np.ndarray,
) -> Dict[str, float]:
    return {
        "ood_auroc": roc_auc(scores, is_out_of_domain),
        "ood_average_precision": average_precision(
            scores, is_out_of_domain
        ),
    }


@dataclass(frozen=True)
class UncertaintyComponents:
    """Keep data, model, and domain uncertainty separately inspectable."""

    data_uncertainty: float
    model_uncertainty: float
    domain_uncertainty: float

    def validate(self) -> None:
        for name, value in (
            ("data_uncertainty", self.data_uncertainty),
            ("model_uncertainty", self.model_uncertainty),
            ("domain_uncertainty", self.domain_uncertainty),
        ):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")

    def conservative_union(self) -> float:
        self.validate()
        return float(
            1
            - (1 - self.data_uncertainty)
            * (1 - self.model_uncertainty)
            * (1 - self.domain_uncertainty)
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "data_uncertainty": self.data_uncertainty,
            "model_uncertainty": self.model_uncertainty,
            "domain_uncertainty": self.domain_uncertainty,
            "conservative_union": self.conservative_union(),
        }
