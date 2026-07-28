"""Training-free periodic counterfactual evidence for atomic motif anomalies.

For a crystalline image, translating by a direct-lattice vector should produce
the same local motif under nearly identical nuisance conditions. A defect breaks
that invariance. The detector compares each pixel with multiple translated
counterparts and keeps the minimum disagreement, so a candidate must disagree
with every available periodic counterpart rather than only one noisy neighbor.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.ndimage import gaussian_filter, shift
from skimage.draw import disk
from skimage.feature import peak_local_max

from .config import DetectorConfig
from .evidence import attach_evidence_protocol
from .features.lattice import (
    direct_basis_from_peaks,
    select_reciprocal_basis,
)
from .features.spectral import estimate_global_reciprocal_peaks
from .models import DetectionRegion, DetectionResult
from .preprocessing import load_grayscale, robust_normalize


@dataclass(frozen=True)
class PeriodicCounterfactualConfig:
    """Frozen numerical policy for the post-baseline motif branch."""

    image_smoothing_sigma: float = 1.0
    disagreement_smoothing_sigma: float = 1.5
    aggregation: str = "minimum"
    candidate_quantile: float = 0.997
    peak_min_distance_fraction: float = 0.50
    render_radius_fraction: float = 0.25
    border_translation_multiplier: float = 1.0

    def validate(self) -> None:
        if self.image_smoothing_sigma < 0:
            raise ValueError("image_smoothing_sigma cannot be negative")
        if self.disagreement_smoothing_sigma < 0:
            raise ValueError(
                "disagreement_smoothing_sigma cannot be negative"
            )
        if self.aggregation not in {"minimum", "median", "mean"}:
            raise ValueError(
                "aggregation must be minimum, median, or mean"
            )
        if not 0.5 < self.candidate_quantile < 1.0:
            raise ValueError("candidate_quantile must be between 0.5 and 1")
        if self.peak_min_distance_fraction <= 0:
            raise ValueError(
                "peak_min_distance_fraction must be positive"
            )
        if self.render_radius_fraction <= 0:
            raise ValueError("render_radius_fraction must be positive")

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


def _source_provenance(
    source: str | Path | np.ndarray,
    raw_image: np.ndarray,
    detector_config: DetectorConfig,
    motif_config: PeriodicCounterfactualConfig,
) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(raw_image)
    pixel_hasher = hashlib.sha256()
    pixel_hasher.update(str(contiguous.dtype).encode("utf-8"))
    pixel_hasher.update(
        np.asarray(contiguous.shape, dtype=np.int64).tobytes()
    )
    pixel_hasher.update(contiguous.view(np.uint8).tobytes())
    file_sha256 = None
    file_bytes = None
    if isinstance(source, (str, Path)):
        target = Path(source).resolve()
        source_name = str(target)
        file_hasher = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_hasher.update(block)
        file_sha256 = file_hasher.hexdigest()
        file_bytes = target.stat().st_size
    else:
        source_name = "<array>"
    combined_config = {
        "detector_config": detector_config.to_dict(),
        "periodic_counterfactual": asdict(motif_config),
    }
    config_hash = hashlib.sha256(
        json.dumps(
            combined_config, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    pixel_hash = pixel_hasher.hexdigest()
    return {
        "source": source_name,
        "input_provenance": {
            "source_image": source_name,
            "source_file_sha256": file_sha256,
            "source_file_bytes": file_bytes,
            "pixel_sha256": pixel_hash,
            "source_id": pixel_hash[:16],
            "raw_dtype": str(contiguous.dtype),
            "raw_shape": list(contiguous.shape),
            "config_sha256": config_hash,
        },
    }


def _translation_vectors(
    basis_xy_columns: np.ndarray,
) -> list[tuple[int, int]]:
    candidates_xy = (
        basis_xy_columns[:, 0],
        basis_xy_columns[:, 1],
        basis_xy_columns[:, 1] - basis_xy_columns[:, 0],
    )
    translations: list[tuple[int, int]] = []
    for vector_xy in candidates_xy:
        dx, dy = np.rint(vector_xy).astype(int)
        candidate = (int(dy), int(dx))
        if candidate == (0, 0):
            continue
        canonical = (
            candidate
            if candidate > (-candidate[0], -candidate[1])
            else (-candidate[0], -candidate[1])
        )
        if canonical not in translations:
            translations.append(canonical)
    if len(translations) < 2:
        raise ValueError(
            "Fewer than two distinct direct-lattice translations were inferred"
        )
    return translations


def _robust_positive_z(
    values: np.ndarray, valid_mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    reference = np.asarray(values[valid_mask], dtype=float)
    center = float(np.median(reference))
    scale = float(1.4826 * np.median(np.abs(reference - center)))
    if scale < 1e-8:
        scale = max(float(np.std(reference)), 1e-8)
    z = np.clip((values - center) / scale, 0, 12)
    return z.astype(np.float32), center, scale


def detect_periodic_counterfactual(
    source: str | Path | np.ndarray,
    detector_config: DetectorConfig,
    motif_config: PeriodicCounterfactualConfig | None = None,
) -> DetectionResult:
    """Detect rare motif violations without reading labels or exemplars."""

    policy = motif_config or PeriodicCounterfactualConfig()
    policy.validate()
    if isinstance(source, (str, Path)):
        raw_image = load_grayscale(source)
    else:
        raw_image = np.asarray(source)
        if raw_image.ndim != 2:
            raise ValueError("Array input must be a 2-D grayscale image")
    normalized, normalization_metadata = robust_normalize(
        raw_image, detector_config.preprocess
    )
    provenance = _source_provenance(
        source, raw_image, detector_config, policy
    )

    reciprocal_peaks = estimate_global_reciprocal_peaks(
        normalized, detector_config.spectral
    )
    selected = select_reciprocal_basis(reciprocal_peaks)
    basis = direct_basis_from_peaks(selected)
    lengths = np.linalg.norm(basis, axis=0)
    spacing = float(np.min(lengths))
    if (
        not np.all(np.isfinite(basis))
        or abs(float(np.linalg.det(basis))) < 1e-6
        or spacing < 2.0
        or float(np.max(lengths)) > min(normalized.shape) / 3
    ):
        raise ValueError("Inferred direct lattice is unusable")
    translations = _translation_vectors(basis)

    smoothed = gaussian_filter(
        normalized, policy.image_smoothing_sigma, mode="reflect"
    )
    disagreements = []
    for dy, dx in translations:
        for sign in (-1, 1):
            counterpart = shift(
                smoothed,
                (sign * dy, sign * dx),
                order=1,
                mode="reflect",
                prefilter=False,
            )
            disagreements.append(np.abs(smoothed - counterpart))
    stack = np.stack(disagreements, axis=0)
    if policy.aggregation == "minimum":
        raw_score = np.min(stack, axis=0)
    elif policy.aggregation == "median":
        raw_score = np.median(stack, axis=0)
    else:
        raw_score = np.mean(stack, axis=0)
    raw_score = gaussian_filter(
        raw_score,
        policy.disagreement_smoothing_sigma,
        mode="reflect",
    ).astype(np.float32)

    maximum_translation = max(
        max(abs(dy), abs(dx)) for dy, dx in translations
    )
    border = int(
        np.ceil(
            policy.border_translation_multiplier * maximum_translation
            + 3 * policy.disagreement_smoothing_sigma
        )
    )
    border = int(np.clip(border, 1, min(normalized.shape) // 3))
    valid = np.ones(normalized.shape, dtype=bool)
    valid[:border, :] = False
    valid[-border:, :] = False
    valid[:, :border] = False
    valid[:, -border:] = False
    robust_z, score_center, score_scale = _robust_positive_z(
        raw_score, valid
    )
    threshold = float(
        np.quantile(raw_score[valid], policy.candidate_quantile)
    )
    minimum_distance = max(
        2, int(round(policy.peak_min_distance_fraction * spacing))
    )
    coordinates_yx = peak_local_max(
        raw_score,
        min_distance=minimum_distance,
        threshold_abs=threshold,
        exclude_border=border,
    )

    shape = normalized.shape
    radius = max(2, int(round(policy.render_radius_fraction * spacing)))
    mask = np.zeros(shape, dtype=bool)
    rendered_z = np.zeros(shape, dtype=np.float32)
    regions = []
    for region_id, (y, x) in enumerate(coordinates_yx, start=1):
        rows, columns = disk((y, x), radius, shape=shape)
        mask[rows, columns] = True
        value = float(robust_z[y, x])
        rendered_z[rows, columns] = np.maximum(
            rendered_z[rows, columns], value
        )
        confidence = float(1.0 / (1.0 + np.exp(-(value - 3.0))))
        regions.append(
            DetectionRegion(
                region_id=region_id,
                bbox_xyxy=(
                    max(0, int(x) - radius),
                    max(0, int(y) - radius),
                    min(shape[1], int(x) + radius + 1),
                    min(shape[0], int(y) + radius + 1),
                ),
                centroid_xy=(float(x), float(y)),
                area_pixels=int(len(rows)),
                mean_score=value,
                max_score=value,
                confidence=confidence,
                confidence_level=(
                    "high" if value >= 5.0 else "medium"
                ),
                evidence_groups=["periodic_counterfactual"],
                defect_type="intensity_or_substitution_anomaly",
            )
        )
    probability = (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    (raw_score - threshold) / max(score_scale, 1e-8),
                    -20,
                    20,
                )
            )
        )
    ).astype(np.float32)
    reliability = valid.astype(np.float32)
    candidate_map = np.zeros(shape, dtype=np.float32)
    if len(coordinates_yx):
        candidate_map[
            coordinates_yx[:, 0], coordinates_yx[:, 1]
        ] = 1.0
    metadata = {
        **normalization_metadata,
        **provenance,
        "detector_branch": "periodic_counterfactual_motif_v1",
        "annotations_read": False,
        "reference_exemplar_used": False,
        "direct_basis_xy_columns": basis.tolist(),
        "lattice_spacing_pixels": spacing,
        "translation_vectors_yx": [list(item) for item in translations],
        "aggregation": policy.aggregation,
        "candidate_threshold_rule": {
            "kind": "within_image_quantile",
            "quantile": policy.candidate_quantile,
            "raw_threshold": threshold,
            "development_tuned": True,
        },
        "valid_border_margin_pixels": border,
        "raw_score_robust_center": score_center,
        "raw_score_robust_scale": score_scale,
        "periodic_counterfactual_config": asdict(policy),
        "periodic_counterfactual_config_sha256": policy.content_hash(),
    }
    result = DetectionResult(
        task="atomic_point_defect",
        image_shape=shape,
        normalized_image=normalized.astype(np.float32),
        score_map=rendered_z,
        probability_map=probability,
        mask=mask,
        regions=regions,
        feature_maps={
            "periodic_counterfactual_raw": raw_score,
            "periodic_counterfactual_robust_z": robust_z,
        },
        evidence_maps={
            "group_intensity": robust_z,
            "registration_reliability": reliability,
            "observed_atomic_peaks": candidate_map,
        },
        metadata=metadata,
    )
    protocol = detector_config.evidence_protocol
    if protocol.enabled:
        result = attach_evidence_protocol(
            result,
            reliable_threshold=protocol.reliable_threshold,
            invalid_threshold=protocol.invalid_threshold,
            minimum_confirmed_evidence_groups=(
                protocol.minimum_confirmed_evidence_groups
            ),
            detector_version="0.3.0-periodic-counterfactual",
        )
    return result
