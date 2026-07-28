"""Partition-safe disagreement-driven active-learning acquisition."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .governance import DataPartition


@dataclass(frozen=True)
class AcquisitionCandidate:
    candidate_id: str
    partition: DataPartition
    student_uncertainty: float
    teacher_student_disagreement: float
    ood_score: float
    diversity_score: float
    physics_violation: float
    annotation_cost_seconds: float
    embedding: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if not self.candidate_id:
            raise ValueError("Acquisition candidate_id is required")
        for name, value in (
            ("student_uncertainty", self.student_uncertainty),
            (
                "teacher_student_disagreement",
                self.teacher_student_disagreement,
            ),
            ("ood_score", self.ood_score),
            ("diversity_score", self.diversity_score),
            ("physics_violation", self.physics_violation),
        ):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.annotation_cost_seconds)
            or self.annotation_cost_seconds <= 0
        ):
            raise ValueError("annotation_cost_seconds must be positive")
        if self.embedding is not None and (
            self.embedding.ndim != 1
            or not np.all(np.isfinite(self.embedding))
        ):
            raise ValueError("Acquisition embedding must be finite and 1-D")


@dataclass(frozen=True)
class AcquisitionWeights:
    uncertainty: float = 1.0
    disagreement: float = 1.5
    ood: float = 1.0
    diversity: float = 0.8
    physics_violation: float = 1.0
    annotation_cost: float = 0.7


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    acquisition_score: float
    annotation_cost_seconds: float
    component_ranks: Mapping[str, float]
    metadata: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class AcquisitionPlan:
    selected: tuple[RankedCandidate, ...]
    excluded_by_partition: Mapping[str, int]
    annotation_budget_seconds: float
    expected_cost_seconds: float
    strategy: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected": [asdict(item) for item in self.selected],
            "excluded_by_partition": dict(self.excluded_by_partition),
            "annotation_budget_seconds": self.annotation_budget_seconds,
            "expected_cost_seconds": self.expected_cost_seconds,
            "strategy": self.strategy,
        }


def teacher_student_disagreement(
    teacher_soft_mask: np.ndarray,
    student_probability: np.ndarray,
    *,
    teacher_evidence: np.ndarray | None = None,
    student_evidence: np.ndarray | None = None,
    evidence_reliability: np.ndarray | None = None,
) -> float:
    """Combine mask disagreement with optional reliability-weighted evidence error."""

    teacher = np.asarray(teacher_soft_mask, dtype=float)
    student = np.asarray(student_probability, dtype=float)
    if teacher.shape != student.shape:
        raise ValueError("Teacher and student masks must have equal shape")
    mask_disagreement = float(np.mean(np.abs(teacher - student)))
    if teacher_evidence is None or student_evidence is None:
        return mask_disagreement
    expected = np.asarray(teacher_evidence, dtype=float)
    predicted = np.asarray(student_evidence, dtype=float)
    if expected.shape != predicted.shape:
        raise ValueError("Teacher and student evidence must have equal shape")
    reliability = (
        np.ones_like(expected)
        if evidence_reliability is None
        else np.asarray(evidence_reliability, dtype=float)
    )
    if reliability.shape != expected.shape:
        raise ValueError("Evidence reliability must match evidence shape")
    evidence_error = float(
        np.sum(np.abs(expected - predicted) * reliability)
        / max(float(np.sum(reliability)), 1e-8)
    )
    return 0.5 * mask_disagreement + 0.5 * evidence_error


def _unit_ranks(values: Sequence[float], higher_is_better: bool = True) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) == 1:
        return np.ones(1, dtype=float)
    ranks = rankdata(array, method="average") - 1
    ranks /= len(array) - 1
    return ranks if higher_is_better else 1 - ranks


def rank_train_candidates(
    candidates: Sequence[AcquisitionCandidate],
    weights: AcquisitionWeights = AcquisitionWeights(),
) -> tuple[list[RankedCandidate], Dict[str, int]]:
    """Rank only train candidates; other partitions are auditable exclusions."""

    excluded: Dict[str, int] = {}
    train: list[AcquisitionCandidate] = []
    for candidate in candidates:
        candidate.validate()
        if candidate.partition != DataPartition.TRAIN:
            excluded[candidate.partition.value] = (
                excluded.get(candidate.partition.value, 0) + 1
            )
            continue
        train.append(candidate)
    if not train:
        return [], excluded

    components = {
        "uncertainty": _unit_ranks(
            [item.student_uncertainty for item in train]
        ),
        "disagreement": _unit_ranks(
            [item.teacher_student_disagreement for item in train]
        ),
        "ood": _unit_ranks([item.ood_score for item in train]),
        "diversity": _unit_ranks(
            [item.diversity_score for item in train]
        ),
        "physics_violation": _unit_ranks(
            [item.physics_violation for item in train]
        ),
        "annotation_cost": _unit_ranks(
            [item.annotation_cost_seconds for item in train],
            higher_is_better=False,
        ),
    }
    weight_map = asdict(weights)
    denominator = max(float(sum(weight_map.values())), 1e-8)
    ranked = []
    for index, candidate in enumerate(train):
        component_values = {
            name: float(values[index]) for name, values in components.items()
        }
        score = sum(
            weight_map[name] * component_values[name]
            for name in component_values
        ) / denominator
        ranked.append(
            RankedCandidate(
                candidate_id=candidate.candidate_id,
                acquisition_score=float(score),
                annotation_cost_seconds=candidate.annotation_cost_seconds,
                component_ranks=component_values,
                metadata=candidate.metadata,
            )
        )
    ranked.sort(
        key=lambda item: (
            item.acquisition_score,
            -item.annotation_cost_seconds,
            item.candidate_id,
        ),
        reverse=True,
    )
    return ranked, excluded


def select_acquisition_batch(
    candidates: Sequence[AcquisitionCandidate],
    annotation_budget_seconds: float,
    *,
    maximum_candidates: int | None = None,
    weights: AcquisitionWeights = AcquisitionWeights(),
) -> AcquisitionPlan:
    """Greedy cost-constrained batch selection after partition-safe ranking."""

    if annotation_budget_seconds <= 0:
        raise ValueError("annotation_budget_seconds must be positive")
    ranked, excluded = rank_train_candidates(candidates, weights)
    selected = []
    spent = 0.0
    for candidate in ranked:
        if maximum_candidates is not None and len(selected) >= maximum_candidates:
            break
        if spent + candidate.annotation_cost_seconds > annotation_budget_seconds:
            continue
        selected.append(candidate)
        spent += candidate.annotation_cost_seconds
    return AcquisitionPlan(
        selected=tuple(selected),
        excluded_by_partition=excluded,
        annotation_budget_seconds=float(annotation_budget_seconds),
        expected_cost_seconds=float(spent),
        strategy=(
            "rank_aggregation(uncertainty,teacher_student_disagreement,"
            "ood,diversity,physics_violation,cost)"
        ),
    )
