"""Model lineage and non-regression release gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class MetricSnapshot:
    detection_score: float
    brier_score: float
    expected_calibration_error: float
    ood_auroc: float
    risk_at_target_coverage: float
    locked_external_test_hash: str
    locked_external_sample_count: int


@dataclass(frozen=True)
class ModelLineageRecord:
    model_version: str
    parent_version: str | None
    training_data_hash: str
    pseudo_teacher_version: str
    human_annotation_hash: str
    calibration_set_hash: str
    architecture_hash: str
    metrics_before: MetricSnapshot
    metrics_after: MetricSnapshot

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class ReleaseGatePolicy:
    minimum_detection_retained_fraction: float = 0.99
    maximum_brier_increase: float = 0.0
    maximum_ece_increase: float = 0.0
    maximum_risk_increase: float = 0.0
    minimum_ood_auroc_retained_fraction: float = 0.99
    require_locked_external_test: bool = True


@dataclass(frozen=True)
class ReleaseDecision:
    accepted_for_release: bool
    checks: Mapping[str, bool]
    reasons: tuple[str, ...]
    lineage_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted_for_release": self.accepted_for_release,
            "checks": dict(self.checks),
            "reasons": list(self.reasons),
            "lineage_hash": self.lineage_hash,
        }


def evaluate_release(
    lineage: ModelLineageRecord,
    policy: ReleaseGatePolicy = ReleaseGatePolicy(),
) -> ReleaseDecision:
    before = lineage.metrics_before
    after = lineage.metrics_after
    checks = {
        "locked_test_present": (
            after.locked_external_sample_count > 0
            if policy.require_locked_external_test
            else True
        ),
        "locked_test_unchanged": (
            bool(before.locked_external_test_hash)
            and before.locked_external_test_hash
            == after.locked_external_test_hash
        ),
        "detection_non_regression": (
            after.detection_score
            >= before.detection_score
            * policy.minimum_detection_retained_fraction
        ),
        "brier_non_regression": (
            after.brier_score
            <= before.brier_score + policy.maximum_brier_increase
        ),
        "ece_non_regression": (
            after.expected_calibration_error
            <= before.expected_calibration_error + policy.maximum_ece_increase
        ),
        "risk_non_regression": (
            after.risk_at_target_coverage
            <= before.risk_at_target_coverage + policy.maximum_risk_increase
        ),
        "ood_non_regression": (
            after.ood_auroc
            >= before.ood_auroc
            * policy.minimum_ood_auroc_retained_fraction
        ),
    }
    reasons = tuple(name for name, passed in checks.items() if not passed)
    return ReleaseDecision(
        accepted_for_release=not reasons,
        checks=checks,
        reasons=reasons,
        lineage_hash=lineage.content_hash(),
    )
