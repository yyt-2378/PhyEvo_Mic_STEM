"""Training-free temporal support for separating persistent structure from scans."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import shift as shift_image
from skimage.registration import phase_cross_correlation

from .config import PreprocessConfig
from .models import DetectionResult
from .preprocessing import load_microscopy_stack, robust_normalize


TEMPORAL_SUPPORT_VERSION = "1.0"


@dataclass(frozen=True)
class TemporalSupportConfig:
    maximum_sampled_frames: int = 9
    registration_upsample_factor: int = 10
    maximum_drift_fraction: float = 0.20
    minimum_registered_frames: int = 3
    persistent_median_z_threshold: float = 1.5
    minimum_candidate_valid_fraction: float = 0.80

    def validate(self) -> None:
        if self.maximum_sampled_frames < 3:
            raise ValueError("maximum_sampled_frames must be at least 3")
        if self.registration_upsample_factor < 1:
            raise ValueError(
                "registration_upsample_factor must be positive"
            )
        if not 0 < self.maximum_drift_fraction < 0.5:
            raise ValueError("maximum_drift_fraction must be in (0, 0.5)")
        if self.minimum_registered_frames < 3:
            raise ValueError("minimum_registered_frames must be at least 3")
        if self.persistent_median_z_threshold <= 0:
            raise ValueError(
                "persistent_median_z_threshold must be positive"
            )
        if not 0 <= self.minimum_candidate_valid_fraction <= 1:
            raise ValueError(
                "minimum_candidate_valid_fraction must be in [0, 1]"
            )


@dataclass
class TemporalSupportReport:
    reference_frame_index: int
    sampled_frame_indices: list[int]
    accepted_frame_indices: list[int]
    registration_records: list[Dict[str, Any]]
    temporal_median_image: np.ndarray
    temporal_mad_map: np.ndarray
    temporal_transience_z: np.ndarray
    temporal_consistency_map: np.ndarray
    valid_mask: np.ndarray
    candidate_support: list[Dict[str, Any]]
    metadata: Dict[str, Any]

    def summary(self) -> Dict[str, Any]:
        persistent = sum(
            item["temporal_support_state"]
            == "temporally_persistent_appearance"
            for item in self.candidate_support
        )
        variable = sum(
            item["temporal_support_state"]
            == "temporally_variable_appearance"
            for item in self.candidate_support
        )
        return {
            "temporal_support_version": TEMPORAL_SUPPORT_VERSION,
            "reference_frame_index": self.reference_frame_index,
            "sampled_frame_indices": self.sampled_frame_indices,
            "accepted_frame_indices": self.accepted_frame_indices,
            "registration_records": self.registration_records,
            "candidate_count": len(self.candidate_support),
            "temporally_persistent_appearance_count": persistent,
            "temporally_variable_appearance_count": variable,
            "candidate_support": self.candidate_support,
            "metadata": self.metadata,
        }


def _sample_indices(
    frame_count: int, reference_index: int, maximum: int
) -> list[int]:
    count = min(frame_count, maximum)
    indices = {
        int(round(value))
        for value in np.linspace(0, frame_count - 1, count)
    }
    indices.add(reference_index)
    return sorted(indices)


def analyze_temporal_support_from_stack(
    stack: np.ndarray,
    candidate_result: DetectionResult,
    preprocess_config: PreprocessConfig,
    *,
    reference_frame_index: int,
    config: TemporalSupportConfig | None = None,
) -> TemporalSupportReport:
    """Register sampled frames and score candidate appearance persistence."""

    policy = config or TemporalSupportConfig()
    policy.validate()
    values = np.asarray(stack)
    if values.ndim != 3 or values.shape[0] < 3:
        raise ValueError("stack must have shape (frame, y, x) with >=3 frames")
    if not -values.shape[0] <= reference_frame_index < values.shape[0]:
        raise IndexError("reference_frame_index is outside the stack")
    reference_frame_index %= values.shape[0]
    if tuple(values.shape[1:]) != candidate_result.image_shape:
        raise ValueError(
            "Candidate result image shape does not match temporal stack"
        )
    reference, _ = robust_normalize(
        values[reference_frame_index], preprocess_config
    )
    if not np.allclose(
        reference, candidate_result.normalized_image, atol=2e-4
    ):
        raise ValueError(
            "Candidate result does not correspond to the declared reference "
            "frame and preprocessing configuration"
        )
    sampled = _sample_indices(
        values.shape[0],
        reference_frame_index,
        policy.maximum_sampled_frames,
    )
    registered = []
    accepted = []
    registration_records = []
    maximum_drift = (
        policy.maximum_drift_fraction * min(candidate_result.image_shape)
    )
    for index in sampled:
        frame, _ = robust_normalize(values[index], preprocess_config)
        if index == reference_frame_index:
            shift_yx = np.zeros(2, dtype=float)
            error = 0.0
            aligned = frame
        else:
            shift_yx, error, phase = phase_cross_correlation(
                reference,
                frame,
                upsample_factor=policy.registration_upsample_factor,
            )
            aligned = shift_image(
                frame,
                shift_yx,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
        drift = float(np.linalg.norm(shift_yx))
        keep = bool(np.isfinite(drift) and drift <= maximum_drift)
        registration_records.append(
            {
                "frame_index": index,
                "shift_yx_pixels": [
                    float(shift_yx[0]),
                    float(shift_yx[1]),
                ],
                "drift_pixels": drift,
                "registration_error": float(error),
                "accepted": keep,
            }
        )
        if keep:
            registered.append(aligned)
            accepted.append(index)
    if len(registered) < policy.minimum_registered_frames:
        raise ValueError(
            "Too few frames passed deterministic drift registration"
        )
    registered_stack = np.stack(registered)
    valid = np.all(np.isfinite(registered_stack), axis=0)
    median_image = np.nanmedian(registered_stack, axis=0)
    absolute_residual = np.abs(registered_stack - median_image)
    mad = np.nanmedian(absolute_residual, axis=0)
    reference_values = mad[valid]
    center = float(np.median(reference_values))
    scale = float(
        1.4826 * np.median(np.abs(reference_values - center))
    )
    if scale < 1e-8:
        scale = max(float(np.std(reference_values)), 1e-8)
    transience_z = np.zeros(candidate_result.image_shape, dtype=np.float32)
    transience_z[valid] = np.clip(
        (mad[valid] - center) / scale, 0, 12
    ).astype(np.float32)
    consistency = np.zeros(candidate_result.image_shape, dtype=np.float32)
    consistency[valid] = 1.0 / (1.0 + transience_z[valid])
    candidate_support = []
    for region in candidate_result.regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        local_valid = valid[y1:y2, x1:x2]
        local_z = transience_z[y1:y2, x1:x2][local_valid]
        local_consistency = consistency[y1:y2, x1:x2][local_valid]
        valid_fraction = float(np.mean(local_valid))
        median_z = (
            float(np.median(local_z)) if local_z.size else float("nan")
        )
        median_consistency = (
            float(np.median(local_consistency))
            if local_consistency.size
            else 0.0
        )
        persistent = (
            valid_fraction >= policy.minimum_candidate_valid_fraction
            and np.isfinite(median_z)
            and median_z <= policy.persistent_median_z_threshold
        )
        candidate_support.append(
            {
                "candidate_id": region.candidate_id
                or f"{candidate_result.task}:{region.region_id:04d}",
                "region_id": region.region_id,
                "centroid_xy": list(region.centroid_xy),
                "valid_fraction": valid_fraction,
                "median_temporal_transience_z": median_z,
                "p90_temporal_transience_z": (
                    float(np.percentile(local_z, 90))
                    if local_z.size
                    else float("nan")
                ),
                "median_temporal_consistency": median_consistency,
                "temporal_support_state": (
                    "temporally_persistent_appearance"
                    if persistent
                    else "temporally_variable_appearance"
                ),
                "hard_candidate_filter_applied": False,
            }
        )
    return TemporalSupportReport(
        reference_frame_index=reference_frame_index,
        sampled_frame_indices=sampled,
        accepted_frame_indices=accepted,
        registration_records=registration_records,
        temporal_median_image=median_image.astype(np.float32),
        temporal_mad_map=np.nan_to_num(mad).astype(np.float32),
        temporal_transience_z=transience_z,
        temporal_consistency_map=consistency,
        valid_mask=valid,
        candidate_support=candidate_support,
        metadata={
            "task": candidate_result.task,
            "image_shape": list(candidate_result.image_shape),
            "stack_shape": list(values.shape),
            "config": asdict(policy),
            "semantics": (
                "Auxiliary relative persistence evidence only; no candidate "
                "is deleted and no defect mechanism is confirmed."
            ),
            "temporal_mad_robust_center": center,
            "temporal_mad_robust_scale": scale,
            "valid_fraction": float(np.mean(valid)),
            "neural_training_used": False,
            "labels_read": False,
        },
    )


def analyze_temporal_support(
    source: str | Path,
    candidate_result: DetectionResult,
    preprocess_config: PreprocessConfig,
    *,
    reference_frame_index: int,
    dataset_path: str | None = None,
    config: TemporalSupportConfig | None = None,
) -> TemporalSupportReport:
    stack = load_microscopy_stack(source, dataset_path=dataset_path)
    report = analyze_temporal_support_from_stack(
        stack,
        candidate_result,
        preprocess_config,
        reference_frame_index=reference_frame_index,
        config=config,
    )
    report.metadata["source"] = str(Path(source).resolve())
    report.metadata["dataset_path"] = dataset_path
    return report


def save_temporal_support(
    report: TemporalSupportReport,
    candidate_result: DetectionResult,
    output: str | Path,
) -> Dict[str, str]:
    target = Path(output)
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite temporal support output: {target}"
        )
    target.mkdir(parents=True)
    summary_path = target / "temporal_support_summary.json"
    maps_path = target / "temporal_support_maps.npz"
    figure_path = target / "temporal_support_diagnostics.png"
    summary_path.write_text(
        json.dumps(report.summary(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        maps_path,
        temporal_median_image=report.temporal_median_image,
        temporal_mad_map=report.temporal_mad_map,
        temporal_transience_z=report.temporal_transience_z,
        temporal_consistency_map=report.temporal_consistency_map,
        valid_mask=report.valid_mask,
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    panels = (
        (candidate_result.normalized_image, "Reference frame", "gray"),
        (report.temporal_median_image, "Registered temporal median", "gray"),
        (report.temporal_mad_map, "Temporal MAD", "magma"),
        (candidate_result.score_map, "Candidate score", "magma"),
        (report.temporal_transience_z, "Temporal transience z", "inferno"),
        (report.temporal_consistency_map, "Temporal consistency", "viridis"),
    )
    for axis, (image, title, cmap) in zip(axes.ravel(), panels):
        axis.imshow(image, cmap=cmap)
        axis.set_title(title)
        axis.axis("off")
    for item in report.candidate_support:
        x, y = item["centroid_xy"]
        color = (
            "#00d084"
            if item["temporal_support_state"]
            == "temporally_persistent_appearance"
            else "#ff8c42"
        )
        axes[1, 2].plot(x, y, marker="o", ms=5, mfc="none", mec=color)
    figure.suptitle(
        "Training-free temporal support: green=persistent appearance, "
        "orange=variable appearance"
    )
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "summary": str(summary_path.resolve()),
        "maps": str(maps_path.resolve()),
        "diagnostics": str(figure_path.resolve()),
    }
