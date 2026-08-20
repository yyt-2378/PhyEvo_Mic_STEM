"""Compile current expert-review revisions into training-ready annotations."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.ndimage import binary_dilation
from skimage.draw import disk, line, polygon
from skimage.io import imsave

from .review_cases import ExpertReviewCase
from .review_dataset import ReviewSample
from .review_prediction import ReviewPrediction, load_prediction_mask
from .review_store import CaseReviewAction, ReviewRevision


REVIEW_EXPORT_FORMAT = "phyevo-mic-expert-annotation-export-v3"
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def review_usage_policy(
    sample: ReviewSample,
    *,
    compiled_mask_available: bool,
) -> dict[str, Any]:
    """Derive allowed uses without weakening the immutable source partition.

    ``review_split`` organizes the expert queue; it must never promote a
    development or validation asset into the student-training pool.  A single
    expert revision remains silver, so it is also ineligible for Gold-only
    calibration and locked evaluation.
    """

    training_eligible = bool(
        compiled_mask_available
        and sample.source_partition == "train"
        and sample.review_split == "train"
    )
    if training_eligible:
        allowed_uses = ["student_training", "expert_review_audit"]
        exclusion_reason = None
    else:
        allowed_uses = ["expert_review_audit"]
        exclusion_reason = (
            "no_trainable_geometry"
            if not compiled_mask_available
            else "source_partition_or_review_split_forbids_training"
        )
    return {
        "source_partition": sample.source_partition,
        "review_split": sample.review_split,
        "split_group_id": sample.split_group_id,
        "allowed_uses": allowed_uses,
        "training_eligible": training_eligible,
        "calibration_eligible": False,
        "locked_evaluation_eligible": False,
        "training_exclusion_reason": exclusion_reason,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_mask_name(revision: ReviewRevision) -> str:
    readable = _SAFE_STEM.sub("-", f"{revision.reviewer_id}-{revision.case_id}")
    readable = readable.strip("-.")[:100]
    suffix = hashlib.sha256(
        f"{revision.reviewer_id}|{revision.case_id}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{readable}-{suffix}.png"


def _validate_source_lineage(
    samples: Iterable[ReviewSample],
    *,
    source_review_samples: Iterable[ReviewSample],
    source_asset_records: Iterable[Mapping[str, Any]],
    source_registry_sha256: str,
) -> list[ReviewSample]:
    """Fail closed if a review queue changes immutable registry fields."""

    materialized = list(samples)
    source_by_id = {item.sample_id: item for item in source_review_samples}
    if len(source_by_id) == 0:
        raise ValueError("Source review manifest cannot be empty")
    asset_by_path: dict[str, Mapping[str, Any]] = {}
    registry_hashes = set()
    for record in source_asset_records:
        registry_hashes.add(str(record.get("registry_sha256", "")))
        asset = record.get("asset")
        if not isinstance(asset, Mapping):
            raise ValueError("Source asset record is missing its asset payload")
        relative_path = str(asset.get("relative_path", ""))
        if not relative_path or relative_path in asset_by_path:
            raise ValueError("Source asset paths must be present and unique")
        asset_by_path[relative_path] = asset
    if registry_hashes != {source_registry_sha256}:
        raise ValueError("Source asset manifest does not match the registry")
    field_pairs = (
        ("dataset_id", "dataset_id"),
        ("source_partition", "partition"),
        ("split_group_id", "split_group_id"),
        ("acquisition_id", "acquisition_id"),
        ("material", "material"),
    )
    for sample in materialized:
        source_sample = source_by_id.get(sample.sample_id)
        if source_sample is None or source_sample != sample:
            raise ValueError(
                "Review queue sample differs from its source review manifest"
            )
        asset = asset_by_path.get(sample.source_path)
        if asset is None:
            raise ValueError("Review sample is absent from the source registry")
        for sample_field, asset_field in field_pairs:
            if getattr(sample, sample_field) != str(asset.get(asset_field, "")):
                raise ValueError(
                    f"Review sample {sample.sample_id} changes {sample_field}"
                )
    return materialized


def rasterize_review_geometry(
    geometry: Mapping[str, Any],
    *,
    geometry_hint: str,
    image_shape: tuple[int, int],
    point_radius: int = 3,
    line_width: int = 3,
) -> np.ndarray:
    """Create a derived binary mask while retaining geometry as source truth."""

    if point_radius < 0 or line_width < 1:
        raise ValueError("Rasterization widths must be positive")
    mask = np.zeros(image_shape, dtype=bool)
    points = np.asarray(geometry["points_xy"], dtype=float)
    if geometry_hint == "point":
        for x, y in points:
            rows, columns = disk(
                (float(y), float(x)), point_radius, shape=image_shape
            )
            mask[rows, columns] = True
        return mask
    if geometry_hint in {"line", "interface"}:
        for first, second in zip(points[:-1], points[1:]):
            x0, y0 = np.round(first).astype(int)
            x1, y1 = np.round(second).astype(int)
            rows, columns = line(y0, x0, y1, x1)
            mask[rows, columns] = True
        if line_width > 1:
            mask = binary_dilation(mask, iterations=max(1, line_width // 2))
        return mask
    if geometry_hint == "region":
        rows, columns = polygon(points[:, 1], points[:, 0], shape=image_shape)
        mask[rows, columns] = True
        return mask
    raise ValueError(f"Unsupported geometry hint {geometry_hint!r}")


def review_training_mask(
    revision: ReviewRevision,
    *,
    prediction_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """Return a mask for trainable decisions, or ``None`` if unresolved."""

    payload = revision.submission
    image_shape = tuple(int(item) for item in payload["image_shape"])
    action = CaseReviewAction(revision.action)
    if action == CaseReviewAction.ACCEPT_PREDICTION:
        if prediction_mask is None:
            raise ValueError("Accepted predictions require their sealed mask")
        mask = np.asarray(prediction_mask, dtype=bool)
        if tuple(mask.shape) != image_shape:
            raise ValueError("Accepted prediction shape does not match review")
        return mask
    if action == CaseReviewAction.CORRECT_GEOMETRY:
        geometry = payload.get("geometry")
        if not isinstance(geometry, Mapping):
            raise ValueError("Corrected review is missing geometry")
        return rasterize_review_geometry(
            geometry,
            geometry_hint=str(payload["geometry_hint"]),
            image_shape=image_shape,
        )
    if action == CaseReviewAction.NO_DEFECT:
        return np.zeros(image_shape, dtype=bool)
    return None


def build_review_export(
    revisions: Iterable[ReviewRevision],
    *,
    cases: Iterable[ExpertReviewCase],
    samples: Iterable[ReviewSample],
    source_review_samples: Iterable[ReviewSample],
    source_asset_records: Iterable[Mapping[str, Any]],
    predictions: Mapping[tuple[str, str], ReviewPrediction],
    project_root: str | Path,
    output_directory: str | Path,
    case_manifest_sha256: str,
    review_queue_sha256: str,
    source_review_manifest_sha256: str,
    source_asset_manifest_sha256: str,
    source_registry_sha256: str,
) -> dict[str, Any]:
    """Write one immutable export from current review projections."""

    root = Path(project_root).resolve()
    provenance_digests = {
        "case_manifest_sha256": case_manifest_sha256,
        "review_queue_sha256": review_queue_sha256,
        "source_review_manifest_sha256": source_review_manifest_sha256,
        "source_asset_manifest_sha256": source_asset_manifest_sha256,
        "source_registry_sha256": source_registry_sha256,
    }
    for name, digest in provenance_digests.items():
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    output = Path(output_directory)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an expert review export")
    case_by_id = {item.case_id: item for item in cases}
    materialized_samples = _validate_source_lineage(
        samples,
        source_review_samples=source_review_samples,
        source_asset_records=source_asset_records,
        source_registry_sha256=source_registry_sha256,
    )
    sample_by_id = {item.sample_id: item for item in materialized_samples}
    materialized = sorted(
        revisions, key=lambda item: (item.reviewer_id, item.case_id)
    )
    output.mkdir(parents=True, exist_ok=False)
    mask_directory = output / "masks"
    mask_directory.mkdir()
    records = []
    actions = Counter()
    trainable = 0
    compiled_count = 0
    training_exclusions = Counter()
    source_partition_counts = Counter()
    for revision in materialized:
        payload = revision.submission
        case = case_by_id.get(revision.case_id)
        if case is None:
            raise ValueError(f"Review references unknown case {revision.case_id}")
        selected_sample_id = str(payload["selected_sample_id"])
        if selected_sample_id not in case.sample_ids:
            raise ValueError("Review sample does not belong to its case")
        if tuple(payload["applies_to_sample_ids"]) != case.sample_ids:
            raise ValueError("Review application scope differs from case manifest")
        if payload["review_split"] != case.review_split:
            raise ValueError("Review split differs from case manifest")
        if payload["case_manifest_sha256"] != case_manifest_sha256:
            raise ValueError("Review was created against a different case manifest")
        sample = sample_by_id[selected_sample_id]
        case_samples = []
        for sample_id in case.sample_ids:
            member = sample_by_id.get(sample_id)
            if member is None:
                raise ValueError(f"Case references unknown sample {sample_id}")
            case_samples.append(member)
        case_source_partitions = {
            item.source_partition for item in case_samples
        }
        split_groups = {item.split_group_id for item in case_samples}
        if len(case_source_partitions) != 1 or len(split_groups) != 1:
            raise ValueError(
                "A review case cannot cross source partitions or split groups"
            )
        if any(item.review_split != case.review_split for item in case_samples):
            raise ValueError("Case members disagree with the case review split")
        prediction = predictions.get((selected_sample_id, sample.primary_task))
        prediction_mask = None
        if revision.action == CaseReviewAction.ACCEPT_PREDICTION.value:
            if prediction is None:
                raise ValueError("Accepted review prediction is unavailable")
            if payload["prediction_sha256"] != prediction.mask_sha256:
                raise ValueError("Accepted prediction digest has changed")
            prediction_mask = load_prediction_mask(prediction, root)
        compiled = review_training_mask(
            revision, prediction_mask=prediction_mask
        )
        mask_path = None
        mask_digest = None
        if compiled is not None:
            compiled_count += 1
            filename = _safe_mask_name(revision)
            target = mask_directory / filename
            imsave(
                str(target), compiled.astype(np.uint8) * 255, check_contrast=False
            )
            mask_path = target.relative_to(root).as_posix()
            mask_digest = _sha256_file(target)
        usage = review_usage_policy(
            sample, compiled_mask_available=compiled is not None
        )
        if usage["training_eligible"]:
            trainable += 1
        elif usage["training_exclusion_reason"] is not None:
            training_exclusions[usage["training_exclusion_reason"]] += 1
        source_partition_counts[usage["source_partition"]] += 1
        actions[revision.action] += 1
        records.append(
            {
                "export_format": REVIEW_EXPORT_FORMAT,
                **revision.to_dict(),
                "label_tier": "expert_single_silver",
                **provenance_digests,
                **usage,
                "derived_mask_path": mask_path,
                "derived_mask_sha256": mask_digest,
                "derived_mask_semantics": (
                    None
                    if compiled is None
                    else "rasterized_from_expert_geometry_or_accepted_prediction"
                ),
            }
        )
    manifest = output / "annotations.jsonl"
    with manifest.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            stream.write("\n")
    summary = {
        "export_format": REVIEW_EXPORT_FORMAT,
        "current_review_count": len(records),
        "compiled_mask_count": compiled_count,
        "training_eligible_count": trainable,
        "training_excluded_count": len(records) - trainable,
        "training_exclusion_counts": dict(sorted(training_exclusions.items())),
        "source_partition_counts": dict(
            sorted(source_partition_counts.items())
        ),
        "action_counts": dict(sorted(actions.items())),
        "annotation_manifest": manifest.relative_to(root).as_posix(),
        "annotation_manifest_sha256": _sha256_file(manifest),
        "label_tier": "expert_single_silver",
        **provenance_digests,
        "source_geometry_is_authoritative": True,
        "derived_masks_are_raster_views": True,
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary
