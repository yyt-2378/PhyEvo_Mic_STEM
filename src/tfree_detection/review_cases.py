"""Group unlabeled image planes into non-redundant expert review cases."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .review_dataset import ReviewSample, validate_review_samples


REVIEW_CASE_FORMAT = "phyevo-mic-expert-review-case-v1"


@dataclass(frozen=True)
class ExpertReviewCase:
    case_id: str
    dataset_id: str
    review_split: str
    case_type: str
    sample_ids: tuple[str, ...]
    primary_sample_id: str
    acquisition_id: str
    frame_index: int | None
    geometry_hint: str
    annotation_target: str
    source_prior: str | None
    annotation_scope: str

    def validate(self) -> None:
        if not self.case_id or not self.sample_ids:
            raise ValueError("Review case id and sample ids are required")
        if self.primary_sample_id not in self.sample_ids:
            raise ValueError("Primary sample must belong to its review case")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Review case sample ids must be unique")
        if self.case_type not in {"single_plane", "multichannel_frame"}:
            raise ValueError("Unsupported review case type")
        if (
            self.case_type == "single_plane" and len(self.sample_ids) != 1
        ):
            raise ValueError("Single-plane cases contain exactly one sample")
        if (
            self.case_type == "multichannel_frame"
            and len(self.sample_ids) < 2
        ):
            raise ValueError("Multichannel cases require multiple samples")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["sample_ids"] = list(self.sample_ids)
        payload["case_format"] = REVIEW_CASE_FORMAT
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExpertReviewCase":
        case = cls(
            case_id=str(values["case_id"]),
            dataset_id=str(values["dataset_id"]),
            review_split=str(values["review_split"]),
            case_type=str(values["case_type"]),
            sample_ids=tuple(str(item) for item in values["sample_ids"]),
            primary_sample_id=str(values["primary_sample_id"]),
            acquisition_id=str(values["acquisition_id"]),
            frame_index=(
                None
                if values.get("frame_index") is None
                else int(values["frame_index"])
            ),
            geometry_hint=str(values["geometry_hint"]),
            annotation_target=str(values["annotation_target"]),
            source_prior=(
                None
                if values.get("source_prior") is None
                else str(values["source_prior"])
            ),
            annotation_scope=str(values["annotation_scope"]),
        )
        case.validate()
        return case


def _channel_rank(sample: ReviewSample) -> tuple[int, str]:
    channel = (sample.channel_name or "").upper()
    rank = 0 if "HAADF" in channel else 1 if channel == "BF" else 2
    return rank, sample.sample_id


def build_expert_review_cases(
    samples: Iterable[ReviewSample],
) -> list[ExpertReviewCase]:
    materialized = validate_review_samples(samples)
    grouped: dict[tuple[str, str, int | None], list[ReviewSample]] = (
        defaultdict(list)
    )
    singletons = []
    for sample in materialized:
        if sample.dataset_id == "al_sigma3_twin_boundaries":
            grouped[
                (sample.dataset_id, sample.acquisition_id, sample.frame_index)
            ].append(sample)
        else:
            singletons.append(sample)

    cases = []
    for sample in singletons:
        cases.append(
            ExpertReviewCase(
                case_id=f"case:{sample.sample_id}",
                dataset_id=sample.dataset_id,
                review_split=sample.review_split,
                case_type="single_plane",
                sample_ids=(sample.sample_id,),
                primary_sample_id=sample.sample_id,
                acquisition_id=sample.acquisition_id,
                frame_index=sample.frame_index,
                geometry_hint=sample.geometry_hint,
                annotation_target={
                    "point": "point_set",
                    "line": "line_trace",
                    "interface": "boundary_trace",
                    "region": "region_contour",
                }.get(sample.geometry_hint, "open_set_geometry"),
                source_prior=sample.source_prior,
                annotation_scope="single_plane",
            )
        )
    for (_, acquisition, frame_index), members in sorted(grouped.items()):
        splits = {item.review_split for item in members}
        if len(splits) != 1:
            raise ValueError("Multichannel frame crosses review splits")
        if len(members) != 5:
            raise ValueError(
                f"Expected five Sigma3 channels, found {len(members)}"
            )
        ordered = sorted(members, key=lambda item: item.sample_id)
        primary = min(members, key=_channel_rank)
        cases.append(
            ExpertReviewCase(
                case_id=f"case:{acquisition}:frame-{frame_index:03d}",
                dataset_id=primary.dataset_id,
                review_split=primary.review_split,
                case_type="multichannel_frame",
                sample_ids=tuple(item.sample_id for item in ordered),
                primary_sample_id=primary.sample_id,
                acquisition_id=acquisition,
                frame_index=frame_index,
                geometry_hint="interface",
                annotation_target="boundary_trace",
                source_prior="sigma3_twin_boundary_present",
                annotation_scope="shared_across_five_detector_channels",
            )
        )
    cases.sort(key=lambda item: item.case_id)
    seen_cases: set[str] = set()
    seen_samples: set[str] = set()
    for case in cases:
        case.validate()
        if case.case_id in seen_cases:
            raise ValueError(f"Duplicate review case: {case.case_id}")
        seen_cases.add(case.case_id)
        overlap = seen_samples.intersection(case.sample_ids)
        if overlap:
            raise ValueError(f"Samples occur in multiple cases: {overlap}")
        seen_samples.update(case.sample_ids)
    if seen_samples != {item.sample_id for item in materialized}:
        raise ValueError("Review cases do not cover the annotation queue")
    return cases


def summarize_review_cases(
    cases: Iterable[ExpertReviewCase],
) -> dict[str, Any]:
    materialized = list(cases)
    return {
        "case_format": REVIEW_CASE_FORMAT,
        "case_count": len(materialized),
        "image_plane_count": sum(len(item.sample_ids) for item in materialized),
        "dataset_case_counts": dict(
            sorted(Counter(item.dataset_id for item in materialized).items())
        ),
        "split_case_counts": dict(
            sorted(Counter(item.review_split for item in materialized).items())
        ),
        "case_type_counts": dict(
            sorted(Counter(item.case_type for item in materialized).items())
        ),
        "geometry_case_counts": dict(
            sorted(Counter(item.geometry_hint for item in materialized).items())
        ),
        "duplicate_annotation_planes_avoided": sum(
            len(item.sample_ids) - 1 for item in materialized
        ),
    }


def load_review_cases(path: str | Path) -> list[ExpertReviewCase]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("case_format") != REVIEW_CASE_FORMAT:
            raise ValueError("Unsupported expert review case format")
        cases.append(ExpertReviewCase.from_dict(payload))
    return cases
