"""Cross-artifact gates for the 154-case expert-review platform."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .review_cases import ExpertReviewCase
from .review_dataset import LabelKind, ReviewSample
from .review_prediction import ReviewPrediction


ALLOWED_EXPERT_LABEL_KINDS = {
    LabelKind.NONE.value,
    LabelKind.SOURCE_PRIOR_ONLY.value,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_review_platform(
    samples: Iterable[ReviewSample],
    cases: Iterable[ExpertReviewCase],
    predictions: Mapping[tuple[str, str], ReviewPrediction],
    *,
    require_v1_inventory: bool = True,
) -> dict[str, Any]:
    """Prove label isolation, split safety, case coverage and prediction display."""

    materialized_samples = list(samples)
    materialized_cases = list(cases)
    sample_by_id = {item.sample_id: item for item in materialized_samples}
    issues = []
    if len(sample_by_id) != len(materialized_samples):
        issues.append("duplicate_sample_ids")
    seen_samples = []
    for case in materialized_cases:
        try:
            case.validate()
        except ValueError as error:
            issues.append(f"invalid_case:{case.case_id}:{error}")
        seen_samples.extend(case.sample_ids)
        if case.review_split not in {"train", "validation"}:
            issues.append(f"forbidden_review_split:{case.case_id}")
        for sample_id in case.sample_ids:
            sample = sample_by_id.get(sample_id)
            if sample is None:
                issues.append(f"unknown_case_sample:{case.case_id}:{sample_id}")
                continue
            if sample.label_kind not in ALLOWED_EXPERT_LABEL_KINDS:
                issues.append(f"labeled_sample_in_expert_queue:{sample_id}")
            if sample.label_path is not None:
                issues.append(f"label_path_in_expert_queue:{sample_id}")
            if not sample.review_eligible:
                issues.append(f"ineligible_sample_in_expert_queue:{sample_id}")
            if sample.source_partition == "locked_external_test":
                issues.append(f"locked_test_in_expert_queue:{sample_id}")
    seen_counts = Counter(seen_samples)
    duplicate_case_samples = sorted(
        sample_id for sample_id, count in seen_counts.items() if count != 1
    )
    if duplicate_case_samples:
        issues.append("case_sample_coverage_is_not_one_to_one")
    missing_case_samples = sorted(set(sample_by_id) - set(seen_samples))
    if missing_case_samples:
        issues.append("expert_queue_samples_missing_from_cases")
    prediction_keys = {
        (sample.sample_id, sample.primary_task)
        for sample in materialized_samples
    }
    missing_predictions = sorted(prediction_keys - set(predictions))
    primary_case_prediction_keys = {
        (
            case.primary_sample_id,
            sample_by_id[case.primary_sample_id].primary_task,
        )
        for case in materialized_cases
        if case.primary_sample_id in sample_by_id
    }
    missing_primary_predictions = sorted(
        primary_case_prediction_keys - set(predictions)
    )
    if missing_predictions:
        issues.append("expert_plane_prediction_missing")
    if missing_primary_predictions:
        issues.append("expert_case_primary_prediction_missing")
    if require_v1_inventory:
        expected = {
            "sample_count": 354,
            "case_count": 154,
            "train_case_count": 131,
            "validation_case_count": 23,
            "al_multichannel_case_count": 50,
        }
        observed = {
            "sample_count": len(materialized_samples),
            "case_count": len(materialized_cases),
            "train_case_count": sum(
                item.review_split == "train" for item in materialized_cases
            ),
            "validation_case_count": sum(
                item.review_split == "validation"
                for item in materialized_cases
            ),
            "al_multichannel_case_count": sum(
                item.dataset_id == "al_sigma3_twin_boundaries"
                and item.case_type == "multichannel_frame"
                for item in materialized_cases
            ),
        }
        for key, value in expected.items():
            if observed[key] != value:
                issues.append(
                    f"v1_inventory_mismatch:{key}:expected={value}:"
                    f"observed={observed[key]}"
                )
    else:
        observed = {}
        expected = {}
    primary_predictions = [
        predictions[key]
        for key in primary_case_prediction_keys
        if key in predictions
    ]
    report = {
        "audit_format": "phyevo-mic-expert-review-platform-audit-v1",
        "passed": not issues,
        "issues": issues,
        "sample_count": len(materialized_samples),
        "case_count": len(materialized_cases),
        "split_case_counts": dict(
            sorted(Counter(item.review_split for item in materialized_cases).items())
        ),
        "geometry_case_counts": dict(
            sorted(Counter(item.geometry_hint for item in materialized_cases).items())
        ),
        "label_kind_counts": dict(
            sorted(Counter(item.label_kind for item in materialized_samples).items())
        ),
        "existing_pixel_label_count": sum(
            item.label_kind
            in {LabelKind.PIXEL_MASK.value, LabelKind.IMPLICIT_EMPTY_MASK.value}
            or item.label_path is not None
            for item in materialized_samples
        ),
        "locked_test_case_count": sum(
            item.review_split == "test" for item in materialized_cases
        ),
        "covered_sample_count": len(set(seen_samples)),
        "primary_prediction_count": len(primary_predictions),
        "all_plane_prediction_count": len(prediction_keys - set(missing_predictions)),
        "primary_prediction_status_counts": dict(
            sorted(Counter(item.status for item in primary_predictions).items())
        ),
        "missing_primary_prediction_count": len(missing_primary_predictions),
        "missing_plane_prediction_count": len(missing_predictions),
        "v1_expected_inventory": expected,
        "v1_observed_inventory": observed,
    }
    return report


def require_review_platform(report: Mapping[str, Any]) -> None:
    if not report.get("passed"):
        issues = ", ".join(str(item) for item in report.get("issues", []))
        raise RuntimeError(f"Expert review platform audit failed: {issues}")
