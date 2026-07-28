"""Auditable state machine for one self-improvement cycle."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class CycleStage(_StringEnum):
    INITIALIZED = "initialized"
    TEACHER_COMPLETED = "teacher_completed"
    ACQUISITION_FROZEN = "acquisition_frozen"
    EXPERT_REVIEW_COMPLETED = "expert_review_completed"
    STUDENT_TRAINED = "student_trained"
    CALIBRATED = "calibrated"
    EVALUATED = "evaluated"
    RELEASE_ACCEPTED = "release_accepted"
    RELEASE_REJECTED = "release_rejected"


_ALLOWED_NEXT = {
    CycleStage.INITIALIZED: {CycleStage.TEACHER_COMPLETED},
    CycleStage.TEACHER_COMPLETED: {CycleStage.ACQUISITION_FROZEN},
    CycleStage.ACQUISITION_FROZEN: {CycleStage.EXPERT_REVIEW_COMPLETED},
    CycleStage.EXPERT_REVIEW_COMPLETED: {CycleStage.STUDENT_TRAINED},
    CycleStage.STUDENT_TRAINED: {CycleStage.CALIBRATED},
    CycleStage.CALIBRATED: {CycleStage.EVALUATED},
    CycleStage.EVALUATED: {
        CycleStage.RELEASE_ACCEPTED,
        CycleStage.RELEASE_REJECTED,
    },
    CycleStage.RELEASE_ACCEPTED: set(),
    CycleStage.RELEASE_REJECTED: set(),
}

_REQUIRED_ARTIFACTS = {
    CycleStage.TEACHER_COMPLETED: {"candidate_table", "teacher_manifest"},
    CycleStage.ACQUISITION_FROZEN: {"acquisition_plan", "train_partition_hash"},
    CycleStage.EXPERT_REVIEW_COMPLETED: {
        "annotation_ledger",
        "review_time_report",
    },
    CycleStage.STUDENT_TRAINED: {"student_checkpoint", "training_manifest"},
    CycleStage.CALIBRATED: {"calibrator", "calibration_report"},
    CycleStage.EVALUATED: {"evaluation_report", "locked_test_hash"},
    CycleStage.RELEASE_ACCEPTED: {"release_decision", "model_lineage"},
    CycleStage.RELEASE_REJECTED: {"release_decision", "model_lineage"},
}


@dataclass(frozen=True)
class CycleEvent:
    sequence: int
    stage: CycleStage
    artifact_hashes: Mapping[str, str]
    metrics: Mapping[str, float]
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["stage"] = self.stage.value
        return payload


@dataclass
class SelfImprovementCycle:
    cycle_id: str
    registry_hash: str
    teacher_version: str
    parent_student_version: str | None
    locked_external_test_hash: str
    stage: CycleStage = CycleStage.INITIALIZED
    events: list[CycleEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.cycle_id or not self.registry_hash or not self.teacher_version:
            raise ValueError("Cycle id, registry hash, and teacher version are required")
        if not self.locked_external_test_hash:
            raise ValueError(
                "A frozen locked-test manifest hash is required, even when "
                "the manifest explicitly records zero assets"
            )

    def advance(
        self,
        stage: CycleStage,
        *,
        artifact_hashes: Mapping[str, str],
        metrics: Mapping[str, float] | None = None,
        notes: str = "",
    ) -> CycleEvent:
        if stage not in _ALLOWED_NEXT[self.stage]:
            raise ValueError(
                f"Invalid cycle transition {self.stage.value} -> {stage.value}"
            )
        missing = _REQUIRED_ARTIFACTS[stage] - set(artifact_hashes)
        if missing:
            raise ValueError(
                f"Stage {stage.value} is missing artifacts: "
                + ", ".join(sorted(missing))
            )
        if any(not value for value in artifact_hashes.values()):
            raise ValueError("Cycle artifact hashes cannot be empty")
        if stage == CycleStage.EVALUATED:
            if (
                artifact_hashes["locked_test_hash"]
                != self.locked_external_test_hash
            ):
                raise ValueError(
                    "Locked external test manifest changed during the cycle"
                )
        event = CycleEvent(
            sequence=len(self.events) + 1,
            stage=stage,
            artifact_hashes=dict(artifact_hashes),
            metrics=dict(metrics or {}),
            notes=notes,
        )
        self.events.append(event)
        self.stage = stage
        return event

    def manifest(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "registry_hash": self.registry_hash,
            "teacher_version": self.teacher_version,
            "parent_student_version": self.parent_student_version,
            "locked_external_test_hash": self.locked_external_test_hash,
            "stage": self.stage.value,
            "events": [event.to_dict() for event in self.events],
            "rules": {
                "gold_overwrite_allowed": False,
                "locked_test_active_learning_allowed": False,
                "teacher_updated_by_student": False,
                "release_requires_non_regression_gate": True,
            },
        }

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                self.manifest(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
