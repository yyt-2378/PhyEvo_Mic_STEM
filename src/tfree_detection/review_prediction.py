"""Compact prediction artifacts for the 2-D expert-review application."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.ndimage import binary_erosion
from skimage.io import imread, imsave

from .config import load_config
from .framework import TrainingFreeDetectionFramework
from .models import DetectionResult
from .review_dataset import ReviewSample, load_review_plane


PREDICTION_FORMAT = "phyevo-mic-2d-review-prediction-v1"
_SAFE_ID = re.compile(r"^[a-zA-Z0-9_.-]+$")
_TASK_CONFIG_FILENAMES = {
    "amorphous_region": "amorphous.yaml",
    "atomic_point_defect": "atomic_point_defect.yaml",
    "phase_boundary": "phase_boundary.yaml",
    "stacking_fault": "stacking_fault.yaml",
}


@dataclass(frozen=True)
class ReviewPrediction:
    prediction_format: str
    sample_id: str
    task: str
    sample_sha256: str
    config_path: str
    config_sha256: str
    pixel_sha256: str
    mask_path: str
    mask_sha256: str
    image_shape: tuple[int, int]
    detected_area_pixels: int
    detected_area_fraction: float
    candidates: tuple[Mapping[str, Any], ...]
    status: str = "completed"
    abstention_reason: str | None = None

    def validate(self) -> None:
        if self.prediction_format != PREDICTION_FORMAT:
            raise ValueError("Unsupported review prediction format")
        if not self.sample_id or not self.task:
            raise ValueError("Prediction sample_id and task are required")
        if len(self.image_shape) != 2 or min(self.image_shape) < 1:
            raise ValueError("Prediction image_shape must be positive 2-D")
        pixels = int(np.prod(self.image_shape))
        if not 0 <= self.detected_area_pixels <= pixels:
            raise ValueError("Prediction area is outside image bounds")
        if not 0.0 <= self.detected_area_fraction <= 1.0:
            raise ValueError("Prediction fraction must lie in [0, 1]")
        if self.status not in {"completed", "abstained"}:
            raise ValueError("Prediction status must be completed or abstained")
        if self.status == "completed" and self.abstention_reason is not None:
            raise ValueError("Completed predictions cannot have abstention reason")
        if self.status == "abstained":
            if not self.abstention_reason:
                raise ValueError("Abstained predictions require a reason")
            if self.detected_area_pixels or self.candidates:
                raise ValueError(
                    "Abstained predictions cannot contain positive geometry"
                )
        for digest in (
            self.sample_sha256,
            self.config_sha256,
            self.pixel_sha256,
            self.mask_sha256,
        ):
            if len(digest) != 64:
                raise ValueError("Prediction SHA-256 fields must be complete")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["image_shape"] = list(self.image_shape)
        payload["candidates"] = [dict(item) for item in self.candidates]
        return payload

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ReviewPrediction":
        record = cls(
            prediction_format=str(values["prediction_format"]),
            sample_id=str(values["sample_id"]),
            task=str(values["task"]),
            sample_sha256=str(values["sample_sha256"]),
            config_path=str(values["config_path"]),
            config_sha256=str(values["config_sha256"]),
            pixel_sha256=str(values["pixel_sha256"]),
            mask_path=str(values["mask_path"]),
            mask_sha256=str(values["mask_sha256"]),
            image_shape=tuple(int(item) for item in values["image_shape"]),
            detected_area_pixels=int(values["detected_area_pixels"]),
            detected_area_fraction=float(values["detected_area_fraction"]),
            candidates=tuple(dict(item) for item in values["candidates"]),
            status=str(values.get("status", "completed")),
            abstention_reason=(
                None
                if values.get("abstention_reason") is None
                else str(values["abstention_reason"])
            ),
        )
        record.validate()
        return record


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_sha256(sample: ReviewSample) -> str:
    return _sha256_bytes(
        json.dumps(
            sample.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _config_path(root: Path, task: str) -> Path:
    try:
        filename = _TASK_CONFIG_FILENAMES[task]
    except KeyError as error:
        raise ValueError(f"No review config is registered for task {task!r}") from error
    return root / "configs" / filename


def _candidate_payload(result: DetectionResult) -> tuple[dict[str, Any], ...]:
    payload = []
    for region in result.regions:
        payload.append(
            {
                "candidate_id": region.candidate_id,
                "bbox_xyxy": list(region.bbox_xyxy),
                "centroid_xy": list(region.centroid_xy),
                "area_pixels": int(region.area_pixels),
                "defect_type": region.defect_type,
                "resolved_label": region.resolved_label or None,
                "decision_state": region.decision_state,
                "reliability": float(region.reliability),
                "quality_flags": list(region.quality_flags),
            }
        )
    return tuple(payload)


def build_review_prediction(
    sample: ReviewSample,
    project_root: str | Path,
    output_root: str | Path,
    *,
    task: str | None = None,
) -> ReviewPrediction:
    """Run one deterministic task and create a compact, immutable artifact."""

    selected_task = sample.primary_task if task is None else task
    if selected_task not in sample.prediction_tasks:
        raise ValueError(
            f"Task {selected_task!r} is not declared for {sample.sample_id}"
        )
    if not _SAFE_ID.fullmatch(sample.sample_id):
        raise ValueError("sample_id is unsafe for an artifact path")
    root = Path(project_root).resolve()
    output = Path(output_root)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.relative_to(root)
    target = output / sample.sample_id / selected_task
    if target.exists():
        raise FileExistsError(
            f"Prediction target already exists: {target}"
        )

    config_path = _config_path(root, selected_task)
    config = load_config(config_path)
    image = load_review_plane(sample, root)
    result = TrainingFreeDetectionFramework(config).detect(image)
    mask = np.asarray(result.final_geometry_mask, dtype=bool)
    target.mkdir(parents=True, exist_ok=False)
    mask_path = target / "mask.png"
    imsave(
        str(mask_path),
        mask.astype(np.uint8) * 255,
        check_contrast=False,
    )
    provenance = result.metadata["input_provenance"]
    record = ReviewPrediction(
        prediction_format=PREDICTION_FORMAT,
        sample_id=sample.sample_id,
        task=selected_task,
        sample_sha256=sample_sha256(sample),
        config_path=config_path.relative_to(root).as_posix(),
        config_sha256=_sha256_file(config_path),
        pixel_sha256=str(provenance["pixel_sha256"]),
        mask_path=mask_path.relative_to(root).as_posix(),
        mask_sha256=_sha256_file(mask_path),
        image_shape=tuple(int(item) for item in result.image_shape),
        detected_area_pixels=int(mask.sum()),
        detected_area_fraction=float(mask.mean()),
        candidates=_candidate_payload(result),
    )
    record_path = target / "prediction.json"
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(record.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record


def build_abstention_prediction(
    sample: ReviewSample,
    project_root: str | Path,
    output_root: str | Path,
    *,
    task: str,
    reason: str,
) -> ReviewPrediction:
    """Record an explicit zero-geometry abstention after detector failure."""

    if task not in sample.prediction_tasks:
        raise ValueError(f"Task {task!r} is not declared for the sample")
    if not reason.strip():
        raise ValueError("Abstention reason cannot be empty")
    root = Path(project_root).resolve()
    output = Path(output_root)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.relative_to(root)
    target = output / sample.sample_id / task
    if target.exists():
        raise FileExistsError(
            f"Prediction target already exists: {target}"
        )
    image = np.ascontiguousarray(load_review_plane(sample, root))
    config_path = _config_path(root, task)
    pixel_digest = hashlib.sha256()
    pixel_digest.update(str(image.dtype).encode("utf-8"))
    pixel_digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    pixel_digest.update(image.view(np.uint8).tobytes())
    target.mkdir(parents=True, exist_ok=False)
    mask_path = target / "mask.png"
    imsave(
        str(mask_path),
        np.zeros(image.shape, dtype=np.uint8),
        check_contrast=False,
    )
    record = ReviewPrediction(
        prediction_format=PREDICTION_FORMAT,
        sample_id=sample.sample_id,
        task=task,
        sample_sha256=sample_sha256(sample),
        config_path=config_path.relative_to(root).as_posix(),
        config_sha256=_sha256_file(config_path),
        pixel_sha256=pixel_digest.hexdigest(),
        mask_path=mask_path.relative_to(root).as_posix(),
        mask_sha256=_sha256_file(mask_path),
        image_shape=tuple(int(item) for item in image.shape),
        detected_area_pixels=0,
        detected_area_fraction=0.0,
        candidates=(),
        status="abstained",
        abstention_reason=reason,
    )
    with (target / "prediction.json").open("x", encoding="utf-8") as stream:
        json.dump(record.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return record


def load_review_prediction(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    verify_mask: bool = True,
) -> ReviewPrediction:
    record_path = Path(path)
    record = ReviewPrediction.from_dict(
        json.loads(record_path.read_text(encoding="utf-8"))
    )
    if verify_mask:
        root = (
            record_path.parents[3]
            if project_root is None
            else Path(project_root).resolve()
        )
        mask = (root / record.mask_path).resolve()
        mask.relative_to(root)
        if not mask.is_file() or _sha256_file(mask) != record.mask_sha256:
            raise ValueError("Prediction mask is missing or checksum-mismatched")
    return record


def discover_review_predictions(
    output_root: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict[tuple[str, str], ReviewPrediction]:
    output = Path(output_root)
    records: dict[tuple[str, str], ReviewPrediction] = {}
    if not output.exists():
        return records
    for path in sorted(output.glob("*/*/prediction.json")):
        record = load_review_prediction(
            path, project_root=project_root, verify_mask=True
        )
        key = (record.sample_id, record.task)
        if key in records:
            raise ValueError(f"Duplicate review prediction: {key}")
        records[key] = record
    return records


def load_prediction_mask(
    prediction: ReviewPrediction,
    project_root: str | Path,
) -> np.ndarray:
    root = Path(project_root).resolve()
    path = (root / prediction.mask_path).resolve()
    path.relative_to(root)
    if _sha256_file(path) != prediction.mask_sha256:
        raise ValueError("Prediction mask checksum does not match its record")
    mask = np.asarray(imread(str(path)))
    if mask.ndim == 3:
        mask = np.any(mask != 0, axis=-1)
    else:
        mask = mask != 0
    if tuple(mask.shape) != prediction.image_shape:
        raise ValueError("Prediction mask shape does not match its record")
    return mask


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float64)
    low, high = np.percentile(values, (0.5, 99.5))
    scaled = np.clip((values - low) / max(high - low, 1e-12), 0, 1)
    return np.round(scaled * 255).astype(np.uint8)


def label_support(label: np.ndarray | None) -> np.ndarray | None:
    if label is None:
        return None
    values = np.asarray(label)
    if values.ndim == 3:
        return np.any(values != 0, axis=-1)
    return values != 0


def comparison_overlay(
    image: np.ndarray,
    *,
    label: np.ndarray | None,
    prediction_mask: np.ndarray | None,
) -> np.ndarray:
    """Overlay label boundary in green and prediction in orange/cyan."""

    raw = normalize_for_display(image)
    rgb = np.repeat(raw[..., None], 3, axis=-1).astype(np.float32)
    support = label_support(label)
    if prediction_mask is not None:
        prediction = np.asarray(prediction_mask, dtype=bool)
        if prediction.shape != raw.shape:
            raise ValueError("Prediction mask shape does not match image")
        orange = np.asarray([255, 132, 0], dtype=np.float32)
        rgb[prediction] = 0.58 * rgb[prediction] + 0.42 * orange
        boundary = prediction & ~binary_erosion(prediction)
        rgb[boundary] = np.asarray([0, 229, 255], dtype=np.float32)
    if support is not None:
        if support.shape != raw.shape:
            raise ValueError("Label shape does not match image")
        boundary = support & ~binary_erosion(support)
        rgb[boundary] = np.asarray([72, 255, 118], dtype=np.float32)
    return np.round(np.clip(rgb, 0, 255)).astype(np.uint8)


def write_prediction_index(
    predictions: Iterable[ReviewPrediction],
    output_path: str | Path,
) -> None:
    target = Path(output_path)
    if target.exists():
        raise FileExistsError("Refusing to overwrite prediction index")
    with target.open("x", encoding="utf-8") as stream:
        for record in sorted(
            predictions, key=lambda item: (item.sample_id, item.task)
        ):
            stream.write(
                json.dumps(record.to_dict(), ensure_ascii=False) + "\n"
            )
