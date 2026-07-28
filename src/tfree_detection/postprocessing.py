"""Spatial projection, morphology, and interpretable region summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.special import expit
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)
from skimage.transform import resize

from .config import FusionConfig, PostprocessConfig
from .fusion import FusionResult
from .models import DetectionRegion


@dataclass
class PostprocessResult:
    score_map: np.ndarray
    probability_map: np.ndarray
    mask: np.ndarray
    group_maps: Dict[str, np.ndarray]
    support_count_map: np.ndarray
    regions: List[DetectionRegion]


def project_grid(
    values: np.ndarray, image_shape: Tuple[int, int], order: int = 1
) -> np.ndarray:
    return resize(
        values,
        image_shape,
        order=order,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    )


def postprocess_regions(
    fusion: FusionResult,
    image_shape: Tuple[int, int],
    fusion_config: FusionConfig,
    config: PostprocessConfig,
    patch_size: int,
) -> PostprocessResult:
    score_map = project_grid(fusion.score_grid, image_shape, order=1).astype(
        np.float32
    )
    group_maps = {
        name: project_grid(values, image_shape, order=1).astype(np.float32)
        for name, values in fusion.group_maps.items()
    }
    support_count = np.sum(
        np.stack(
            [
                values >= fusion_config.support_z
                for values in group_maps.values()
            ],
            axis=-1,
        ),
        axis=-1,
    ).astype(np.int16)
    mask = (score_map >= fusion.threshold) & (
        support_count >= fusion_config.minimum_evidence_groups
    )

    margin = (
        patch_size // 2 if config.border_margin is None else config.border_margin
    )
    margin = int(np.clip(margin, 0, max(min(image_shape) // 3, 0)))
    if margin:
        mask[:margin, :] = False
        mask[-margin:, :] = False
        mask[:, :margin] = False
        mask[:, -margin:] = False

    if config.morphology_radius > 0:
        footprint = disk(config.morphology_radius)
        mask = binary_closing(mask, footprint)
        mask = binary_opening(mask, footprint)
    if config.minimum_region_area > 0:
        mask = remove_small_objects(
            mask, min_size=config.minimum_region_area, connectivity=2
        )
    if config.minimum_hole_area > 0:
        mask = remove_small_holes(
            mask, area_threshold=config.minimum_hole_area, connectivity=2
        )
    if config.exclude_border_touching and np.any(mask):
        preliminary_labels = label(mask, connectivity=2)
        border_ids = set()
        inner_top = margin
        inner_bottom = image_shape[0] - margin - 1
        inner_left = margin
        inner_right = image_shape[1] - margin - 1
        if inner_top <= inner_bottom and inner_left <= inner_right:
            border_ids.update(np.unique(preliminary_labels[inner_top, :]).tolist())
            border_ids.update(np.unique(preliminary_labels[inner_bottom, :]).tolist())
            border_ids.update(np.unique(preliminary_labels[:, inner_left]).tolist())
            border_ids.update(np.unique(preliminary_labels[:, inner_right]).tolist())
        border_ids.discard(0)
        for border_id in border_ids:
            mask[preliminary_labels == border_id] = False

    score_scale = max(fusion.score_scale, 1e-8)
    probability_map = expit(
        (score_map.astype(np.float64) - fusion.threshold) / score_scale
    ).astype(np.float32)
    labels = label(mask, connectivity=2)
    regions: List[DetectionRegion] = []
    for region_id, region in enumerate(regionprops(labels), start=1):
        coordinates = region.coords
        rows, columns = coordinates[:, 0], coordinates[:, 1]
        region_scores = score_map[rows, columns]
        region_probabilities = probability_map[rows, columns]
        supported_groups = [
            name
            for name, values in group_maps.items()
            if float(np.percentile(values[rows, columns], 75))
            >= fusion_config.support_z
        ]
        if "spectrum" in supported_groups and "patterson" in supported_groups:
            confidence_level = "high"
        elif len(supported_groups) >= fusion_config.minimum_evidence_groups:
            confidence_level = "medium"
        else:
            confidence_level = "review"
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
                mean_score=float(np.mean(region_scores)),
                max_score=float(np.max(region_scores)),
                confidence=float(np.mean(region_probabilities)),
                confidence_level=confidence_level,
                evidence_groups=supported_groups,
            )
        )

    return PostprocessResult(
        score_map=score_map,
        probability_map=probability_map,
        mask=np.asarray(mask, dtype=bool),
        group_maps=group_maps,
        support_count_map=support_count,
        regions=regions,
    )
