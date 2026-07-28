"""Shared task-level robust statistics and result construction."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from scipy.special import expit
from skimage.measure import label, regionprops

from ..models import DetectionRegion


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if scale < 1e-8:
        q25, q75 = np.percentile(finite, [25, 75])
        scale = max(float((q75 - q25) / 1.349), 1e-8)
    return center, scale


def positive_robust_z(values: np.ndarray) -> tuple[np.ndarray, dict]:
    center, scale = robust_location_scale(values)
    score = np.clip((np.asarray(values) - center) / scale, 0, 12)
    return score.astype(np.float32), {
        "median": center,
        "robust_scale": scale,
    }


def probability_from_threshold(
    score: np.ndarray, threshold: float, scale: float | None = None
) -> np.ndarray:
    if scale is None:
        _, scale = robust_location_scale(score)
    return expit(
        (np.asarray(score, dtype=float) - threshold) / max(float(scale), 1e-8)
    ).astype(np.float32)


def regions_from_mask(
    mask: np.ndarray,
    score_map: np.ndarray,
    probability_map: np.ndarray,
    evidence_maps: Dict[str, np.ndarray],
    defect_type: str,
    support_threshold: float = 1.0,
    evidence_names: Iterable[str] | None = None,
) -> list[DetectionRegion]:
    names = list(evidence_names or evidence_maps)
    labels = label(mask, connectivity=2)
    regions: list[DetectionRegion] = []
    for region_id, region in enumerate(regionprops(labels), start=1):
        rows, columns = region.coords[:, 0], region.coords[:, 1]
        supported = []
        for name in names:
            values = evidence_maps.get(name)
            if values is None or values.shape != mask.shape:
                continue
            if float(np.percentile(values[rows, columns], 75)) >= support_threshold:
                supported.append(name.replace("group_", ""))
        min_row, min_column, max_row, max_column = region.bbox
        centroid_row, centroid_column = region.centroid
        regions.append(
            DetectionRegion(
                region_id=region_id,
                bbox_xyxy=(
                    int(min_column),
                    int(min_row),
                    int(max_column),
                    int(max_row),
                ),
                centroid_xy=(float(centroid_column), float(centroid_row)),
                area_pixels=int(region.area),
                mean_score=float(np.mean(score_map[rows, columns])),
                max_score=float(np.max(score_map[rows, columns])),
                confidence=float(np.mean(probability_map[rows, columns])),
                confidence_level=(
                    "high"
                    if len(supported) >= 2
                    else "medium"
                    if supported
                    else "review"
                ),
                evidence_groups=supported,
                defect_type=defect_type,
            )
        )
    return regions
