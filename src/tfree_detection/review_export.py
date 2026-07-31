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


REVIEW_EXPORT_FORMAT = "phyevo-mic-expert-annotation-export-v1"
_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")


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
    predictions: Mapping[tuple[str, str], ReviewPrediction],
    project_root: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Write one immutable export from current review projections."""

    root = Path(project_root).resolve()
    output = Path(output_directory)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.relative_to(root)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an expert review export")
    case_by_id = {item.case_id: item for item in cases}
    sample_by_id = {item.sample_id: item for item in samples}
    materialized = sorted(
        revisions, key=lambda item: (item.reviewer_id, item.case_id)
    )
    output.mkdir(parents=True, exist_ok=False)
    mask_directory = output / "masks"
    mask_directory.mkdir()
    records = []
    actions = Counter()
    trainable = 0
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
        sample = sample_by_id[selected_sample_id]
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
            filename = _safe_mask_name(revision)
            target = mask_directory / filename
            imsave(
                str(target), compiled.astype(np.uint8) * 255, check_contrast=False
            )
            mask_path = target.relative_to(root).as_posix()
            mask_digest = _sha256_file(target)
            trainable += 1
        actions[revision.action] += 1
        records.append(
            {
                "export_format": REVIEW_EXPORT_FORMAT,
                **revision.to_dict(),
                "label_tier": "expert_single_silver",
                "training_eligible": compiled is not None,
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
        "training_eligible_count": trainable,
        "excluded_count": len(records) - trainable,
        "action_counts": dict(sorted(actions.items())),
        "annotation_manifest": manifest.relative_to(root).as_posix(),
        "annotation_manifest_sha256": _sha256_file(manifest),
        "label_tier": "expert_single_silver",
        "source_geometry_is_authoritative": True,
        "derived_masks_are_raster_views": True,
    }
    with (output / "summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary
