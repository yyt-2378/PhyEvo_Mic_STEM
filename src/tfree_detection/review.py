"""Low-cost expert review packets and two-reviewer consensus conversion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from .evidence import LabelTier
from .governance import (
    AnnotationRecord,
    AnnotationSource,
    CandidateRecord,
)


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class ReviewAction(_StringEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    CHANGE_TYPE = "change_type"
    MARK_UNKNOWN = "mark_unknown"
    MARK_ARTIFACT = "mark_artifact"
    ADJUST_POINT = "adjust_point"
    ADJUST_LINE = "adjust_line"
    ADJUST_CONTOUR = "adjust_contour"


_GEOMETRY_ACTIONS = {
    "point": (
        ReviewAction.ACCEPT,
        ReviewAction.REJECT,
        ReviewAction.CHANGE_TYPE,
        ReviewAction.MARK_UNKNOWN,
        ReviewAction.MARK_ARTIFACT,
        ReviewAction.ADJUST_POINT,
    ),
    "line": (
        ReviewAction.ACCEPT,
        ReviewAction.REJECT,
        ReviewAction.CHANGE_TYPE,
        ReviewAction.MARK_UNKNOWN,
        ReviewAction.MARK_ARTIFACT,
        ReviewAction.ADJUST_LINE,
    ),
    "region": (
        ReviewAction.ACCEPT,
        ReviewAction.REJECT,
        ReviewAction.CHANGE_TYPE,
        ReviewAction.MARK_UNKNOWN,
        ReviewAction.MARK_ARTIFACT,
        ReviewAction.ADJUST_CONTOUR,
    ),
    "interface": (
        ReviewAction.ACCEPT,
        ReviewAction.REJECT,
        ReviewAction.CHANGE_TYPE,
        ReviewAction.MARK_UNKNOWN,
        ReviewAction.MARK_ARTIFACT,
        ReviewAction.ADJUST_LINE,
        ReviewAction.ADJUST_CONTOUR,
    ),
}


@dataclass(frozen=True)
class ReviewCandidatePacket:
    packet_version: str
    candidate: Mapping[str, Any]
    allowed_actions: tuple[str, ...]
    image_artifact: str
    overlay_artifact: str
    evidence_artifact: str | None
    model_assistance_policy: str
    instructions: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["allowed_actions"] = list(self.allowed_actions)
        return payload


def make_review_packet(
    candidate: CandidateRecord,
    *,
    image_artifact: str,
    overlay_artifact: str,
    evidence_artifact: str | None = None,
    show_model_assistance: bool = True,
) -> ReviewCandidatePacket:
    actions = _GEOMETRY_ACTIONS.get(candidate.geometry)
    if actions is None:
        raise ValueError(
            f"Unsupported candidate geometry {candidate.geometry!r}"
        )
    candidate_payload = candidate.to_dict()
    if not show_model_assistance:
        candidate_payload = {
            key: candidate_payload[key]
            for key in (
                "candidate_id",
                "source_id",
                "dataset_id",
                "partition",
                "geometry",
                "bbox_xyxy",
                "centroid_xy",
                "area_pixels",
            )
        }
    return ReviewCandidatePacket(
        packet_version="1.0",
        candidate=candidate_payload,
        allowed_actions=tuple(action.value for action in actions),
        image_artifact=image_artifact,
        overlay_artifact=overlay_artifact,
        evidence_artifact=evidence_artifact,
        model_assistance_policy=(
            "assisted"
            if show_model_assistance
            else "blinded_model_output_hidden"
        ),
        instructions=(
            "Judge structural validity first; mark drift, defocus, saturation, "
            "vacuum, or scan jumps as artifact. Unknown is a valid response."
        ),
    )


@dataclass(frozen=True)
class ReviewResponse:
    response_id: str
    candidate_id: str
    reviewer_id: str
    action: ReviewAction
    defect_labels: tuple[str, ...]
    annotation_duration_seconds: float
    model_assistance_shown: bool
    geometry_payload: Mapping[str, Any] | None = None
    notes: str = ""

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ReviewResponse":
        labels = values.get("defect_labels", [])
        assistance = values["model_assistance_shown"]
        duration = values["annotation_duration_seconds"]
        if not isinstance(labels, list):
            raise TypeError("defect_labels must be a JSON array")
        if not isinstance(assistance, bool):
            raise TypeError("model_assistance_shown must be boolean")
        if isinstance(duration, bool) or not isinstance(
            duration, (int, float)
        ):
            raise TypeError("annotation_duration_seconds must be numeric")
        response = cls(
            response_id=str(values["response_id"]),
            candidate_id=str(values["candidate_id"]),
            reviewer_id=str(values["reviewer_id"]),
            action=ReviewAction(str(values["action"])),
            defect_labels=tuple(str(item) for item in labels),
            annotation_duration_seconds=float(duration),
            model_assistance_shown=assistance,
            geometry_payload=values.get("geometry_payload"),
            notes=str(values.get("notes", "")),
        )
        response.validate()
        return response

    def validate(self) -> None:
        if not self.response_id or not self.candidate_id or not self.reviewer_id:
            raise ValueError("Review response, candidate, and reviewer ids are required")
        if self.annotation_duration_seconds <= 0:
            raise ValueError("Annotation duration must be positive")
        if self.accepted is True and not self.defect_labels:
            raise ValueError("Accepted or adjusted candidates need defect labels")
        if self.action in {
            ReviewAction.ADJUST_POINT,
            ReviewAction.ADJUST_LINE,
            ReviewAction.ADJUST_CONTOUR,
        } and not self.geometry_payload:
            raise ValueError("Geometry adjustment requires geometry_payload")

    @property
    def accepted(self) -> bool | None:
        if self.action in {
            ReviewAction.ACCEPT,
            ReviewAction.CHANGE_TYPE,
            ReviewAction.ADJUST_POINT,
            ReviewAction.ADJUST_LINE,
            ReviewAction.ADJUST_CONTOUR,
        }:
            return True
        if self.action in {ReviewAction.REJECT, ReviewAction.MARK_ARTIFACT}:
            return False
        return None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["action"] = self.action.value
        payload["defect_labels"] = list(self.defect_labels)
        return payload


@dataclass(frozen=True)
class ConsensusOutcome:
    candidate_id: str
    status: str
    annotation: AnnotationRecord | None
    reviewer_count: int
    mean_annotation_duration_seconds: float
    assistance_conditions: tuple[str, ...]
    disagreement_reason: str | None = None


def consensus_outcomes(
    responses: Sequence[ReviewResponse],
    *,
    annotation_version: str,
) -> list[ConsensusOutcome]:
    """Convert independent reviews to Gold only when two reviewers agree."""

    grouped: Dict[str, list[ReviewResponse]] = {}
    response_ids: set[str] = set()
    reviewer_candidate_pairs: set[tuple[str, str]] = set()
    for response in responses:
        response.validate()
        if response.response_id in response_ids:
            raise ValueError(f"Duplicate response id {response.response_id!r}")
        response_ids.add(response.response_id)
        reviewer_candidate = (
            response.reviewer_id,
            response.candidate_id,
        )
        if reviewer_candidate in reviewer_candidate_pairs:
            raise ValueError(
                "A reviewer may submit at most one response per candidate"
            )
        reviewer_candidate_pairs.add(reviewer_candidate)
        grouped.setdefault(response.candidate_id, []).append(response)

    outcomes = []
    for candidate_id, candidate_responses in sorted(grouped.items()):
        reviewers = {response.reviewer_id for response in candidate_responses}
        conditions = tuple(
            sorted(
                {
                    "assisted" if response.model_assistance_shown else "blinded"
                    for response in candidate_responses
                }
            )
        )
        duration = sum(
            response.annotation_duration_seconds
            for response in candidate_responses
        ) / len(candidate_responses)
        if len(reviewers) < 2:
            outcomes.append(
                ConsensusOutcome(
                    candidate_id=candidate_id,
                    status=LabelTier.UNRESOLVED.value,
                    annotation=None,
                    reviewer_count=len(reviewers),
                    mean_annotation_duration_seconds=duration,
                    assistance_conditions=conditions,
                    disagreement_reason="fewer_than_two_independent_reviewers",
                )
            )
            continue
        accepted = {response.accepted for response in candidate_responses}
        label_sets = {
            tuple(sorted(response.defect_labels))
            for response in candidate_responses
            if response.accepted is True
        }
        if len(accepted) != 1 or None in accepted or len(label_sets) > 1:
            outcomes.append(
                ConsensusOutcome(
                    candidate_id=candidate_id,
                    status=LabelTier.UNRESOLVED.value,
                    annotation=None,
                    reviewer_count=len(reviewers),
                    mean_annotation_duration_seconds=duration,
                    assistance_conditions=conditions,
                    disagreement_reason="reviewer_label_or_acceptance_disagreement",
                )
            )
            continue
        agreed_acceptance = next(iter(accepted))
        labels = next(iter(label_sets)) if agreed_acceptance else ()
        digest_source = "|".join(
            sorted(response.response_id for response in candidate_responses)
        )
        digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
        annotation = AnnotationRecord(
            annotation_id=(
                f"consensus:{annotation_version}:{candidate_id}:"
                f"{digest}"
            ),
            candidate_id=candidate_id,
            label_tier=LabelTier.GOLD,
            source=AnnotationSource.EXPERT_CONSENSUS,
            annotation_version=annotation_version,
            defect_labels=labels,
            accepted=agreed_acceptance,
            reviewers=tuple(sorted(reviewers)),
            notes="Independent expert consensus",
        )
        annotation.validate()
        outcomes.append(
            ConsensusOutcome(
                candidate_id=candidate_id,
                status=LabelTier.GOLD.value,
                annotation=annotation,
                reviewer_count=len(reviewers),
                mean_annotation_duration_seconds=duration,
                assistance_conditions=conditions,
            )
        )
    return outcomes


def write_review_packets(
    packets: Iterable[ReviewCandidatePacket], path: str | Path
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite review packet file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(
            [packet.to_dict() for packet in packets],
            stream,
            ensure_ascii=False,
            indent=2,
        )
