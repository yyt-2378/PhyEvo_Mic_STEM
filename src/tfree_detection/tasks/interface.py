"""Training-free two-phase, twin, and grain-boundary detection."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    maximum_filter,
    median_filter,
    minimum_filter,
)
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, remove_small_objects
from skimage.transform import resize

from ..config import DetectorConfig
from ..context import DetectionContext
from ..models import DetectionResult
from .base import DetectionTask
from .common import probability_from_threshold, regions_from_mask


def _centers(length: int, stride: int, margin: int) -> np.ndarray:
    start = min(max(margin, 0), max(length - 1, 0))
    stop = max(start + 1, length - margin + 1)
    centers = np.arange(start, stop, stride, dtype=np.int32)
    if not len(centers):
        return np.asarray([length // 2], dtype=np.int32)
    return centers


def _local_reciprocal_signatures(
    image: np.ndarray,
    config: DetectorConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    boundary = config.phase_boundary
    size = boundary.patch_size
    centers_y = _centers(image.shape[0], boundary.stride, size // 2)
    centers_x = _centers(image.shape[1], boundary.stride, size // 2)
    frequencies = np.fft.fftshift(np.fft.fftfreq(size))
    fy, fx = np.meshgrid(frequencies, frequencies, indexing="ij")
    radius = np.hypot(fx, fy)
    angle = np.mod(np.arctan2(fy, fx), np.pi)
    radial_edges = np.linspace(
        config.spectral.min_frequency,
        config.spectral.max_frequency,
        boundary.radial_bins + 1,
    )
    orientation_edges = np.linspace(
        0, np.pi, boundary.orientation_bins + 1
    )
    masks = []
    for radial_index in range(boundary.radial_bins):
        radial = (
            (radius >= radial_edges[radial_index])
            & (radius < radial_edges[radial_index + 1])
        )
        for orientation_index in range(boundary.orientation_bins):
            masks.append(
                radial
                & (angle >= orientation_edges[orientation_index])
                & (angle < orientation_edges[orientation_index + 1])
            )
    window = np.outer(np.hanning(size), np.hanning(size))
    half = size // 2
    padded = np.pad(image, ((half, half), (half, half)), mode="reflect")
    signatures = np.empty(
        (len(centers_y), len(centers_x), len(masks)),
        dtype=np.float32,
    )
    local_contrast = np.empty(
        (len(centers_y), len(centers_x)), dtype=np.float32
    )
    for row, center_y in enumerate(centers_y):
        for column, center_x in enumerate(centers_x):
            patch = padded[
                center_y : center_y + size,
                center_x : center_x + size,
            ]
            local_contrast[row, column] = float(np.std(patch))
            standardized = (patch - np.mean(patch)) / (
                np.std(patch) + 1e-12
            )
            power = np.abs(
                np.fft.fftshift(np.fft.fft2(standardized * window))
            ) ** 2
            descriptor = np.asarray(
                [np.sum(power[mask]) for mask in masks], dtype=float
            )
            descriptor /= np.sum(descriptor) + 1e-12
            descriptor = np.sqrt(descriptor)
            descriptor /= np.linalg.norm(descriptor) + 1e-12
            signatures[row, column] = descriptor
    return signatures, centers_y, centers_x, local_contrast


def _two_robust_prototypes(
    signatures: np.ndarray,
    iterations: int,
    valid_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    full_flat = signatures.reshape(-1, signatures.shape[-1]).astype(float)
    if valid_grid is None:
        valid_flat = np.ones(len(full_flat), dtype=bool)
    else:
        valid_flat = np.asarray(valid_grid, dtype=bool).ravel()
    flat = full_flat[valid_flat]
    if len(flat) < 8:
        labels = np.full(len(full_flat), -1, dtype=np.int8)
        return labels, np.zeros((2, full_flat.shape[1])), np.zeros(len(full_flat)), 0.0
    median = np.median(flat, axis=0)
    scale = 1.4826 * np.median(np.abs(flat - median), axis=0)
    active = scale > 1e-5
    if int(np.sum(active)) < 2:
        labels = np.full(len(full_flat), -1, dtype=np.int8)
        labels[valid_flat] = 0
        return labels, np.zeros((2, flat.shape[1])), np.zeros(len(full_flat)), 0.0
    standardized = (flat[:, active] - median[active]) / scale[active]
    # Capping individual channels prevents a narrow mixed/interface patch
    # from becoming its own "phase" and forcing the two bulk phases together.
    standardized = np.clip(standardized, -1.5, 1.5)
    centered = standardized - np.mean(standardized, axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    projection = centered @ components[0]
    lower, upper = np.quantile(projection, [0.25, 0.75])
    centers = np.stack(
        [
            np.median(standardized[projection <= lower], axis=0),
            np.median(standardized[projection >= upper], axis=0),
        ]
    )
    labels = np.zeros(len(flat), dtype=np.int8)
    distances = np.empty((len(flat), 2), dtype=float)
    for _ in range(iterations):
        distances = np.linalg.norm(
            standardized[:, None, :] - centers[None, :, :], axis=2
        )
        updated_labels = np.argmin(distances, axis=1).astype(np.int8)
        if np.array_equal(updated_labels, labels):
            labels = updated_labels
            break
        labels = updated_labels
        for cluster in (0, 1):
            members = standardized[labels == cluster]
            if len(members):
                centers[cluster] = np.median(members, axis=0)
    distances = np.linalg.norm(
        standardized[:, None, :] - centers[None, :, :], axis=2
    )
    assigned = distances[np.arange(len(flat)), labels]
    separation = float(np.linalg.norm(centers[0] - centers[1]))
    within = float(np.median(assigned)) + 1e-12
    separation_ratio = separation / within
    margin = np.abs(distances[:, 0] - distances[:, 1]) / (
        np.sum(distances, axis=1) + 1e-12
    )
    full_centers = np.zeros((2, flat.shape[1]), dtype=float)
    full_centers[:, active] = centers
    full_labels = np.full(len(full_flat), -1, dtype=np.int8)
    full_margin = np.zeros(len(full_flat), dtype=float)
    full_labels[valid_flat] = labels
    full_margin[valid_flat] = margin
    return full_labels, full_centers, full_margin, separation_ratio


class PhaseBoundaryTask(DetectionTask):
    """Separate two reciprocal-texture domains and trace their interface."""

    name = "phase_boundary"

    def run(
        self,
        normalized_image: np.ndarray,
        config: DetectorConfig,
        input_metadata: Dict[str, Any],
        context: DetectionContext,
    ) -> DetectionResult:
        boundary_config = config.phase_boundary
        signatures, centers_y, centers_x, local_contrast = (
            _local_reciprocal_signatures(
            normalized_image, config
            )
        )
        contrast_threshold = float(threshold_otsu(local_contrast))
        high_contrast = local_contrast >= contrast_threshold
        low_fraction = float(np.mean(~high_contrast))
        lower_values = local_contrast[~high_contrast]
        upper_values = local_contrast[high_contrast]
        contrast_ratio = (
            float(np.median(upper_values))
            / max(float(np.median(lower_values)), 1e-12)
            if lower_values.size and upper_values.size
            else 1.0
        )
        specimen_mask_applied = bool(
            0.02 <= low_fraction <= 0.35 and contrast_ratio >= 1.8
        )
        crystalline_grid = (
            high_contrast
            if specimen_mask_applied
            else np.ones_like(high_contrast, dtype=bool)
        )
        labels, prototypes, margin, separation = _two_robust_prototypes(
            signatures,
            boundary_config.clustering_iterations,
            valid_grid=crystalline_grid,
        )
        label_grid = labels.reshape(signatures.shape[:2])
        smoothed_labels = median_filter(label_grid, size=3, mode="nearest")
        label_grid[crystalline_grid] = smoothed_labels[crystalline_grid]
        label_grid[~crystalline_grid] = -1
        fractions = [
            float(np.mean(label_grid[crystalline_grid] == cluster))
            for cluster in (0, 1)
        ]
        genuine_two_phase = (
            min(fractions) >= boundary_config.minimum_phase_fraction
            and separation >= boundary_config.minimum_signature_separation
        )

        image_shape = normalized_image.shape
        domain_label = resize(
            label_grid,
            image_shape,
            order=0,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.int8)
        crystalline_support = resize(
            crystalline_grid.astype(float),
            image_shape,
            order=0,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        ) >= 0.5
        margin_grid = margin.reshape(signatures.shape[:2])
        domain_margin = resize(
            margin_grid,
            image_shape,
            order=1,
            mode="edge",
            anti_aliasing=False,
            preserve_range=True,
        ).astype(np.float32)
        raw_boundary = (
            maximum_filter(
                (domain_label == 0).astype(np.uint8),
                size=3,
                mode="nearest",
            )
            > 0
        ) & (
            maximum_filter(
                (domain_label == 1).astype(np.uint8),
                size=3,
                mode="nearest",
            )
            > 0
        )
        if boundary_config.boundary_width > 0:
            raw_boundary = dilation(
                raw_boundary, disk(boundary_config.boundary_width)
            )

        valid = crystalline_support.copy()
        if specimen_mask_applied:
            specimen_interior = distance_transform_edt(
                crystalline_support
            ) > max(boundary_config.patch_size // 2, 1)
            valid &= specimen_interior
        border = min(
            boundary_config.patch_size // 2,
            min(image_shape) // 3,
        )
        if border:
            valid[:border, :] = False
            valid[-border:, :] = False
            valid[:, :border] = False
            valid[:, -border:] = False
        distance = distance_transform_edt(~raw_boundary)
        decay = max(boundary_config.boundary_width, 1)
        score = (
            separation * np.exp(-distance / decay)
            if genuine_two_phase
            else np.zeros(image_shape, dtype=float)
        )
        threshold = float(separation * np.exp(-1.15))
        mask = raw_boundary & valid if genuine_two_phase else np.zeros(
            image_shape, dtype=bool
        )
        if boundary_config.minimum_region_area > 0:
            mask = remove_small_objects(
                mask,
                min_size=boundary_config.minimum_region_area,
                connectivity=2,
            )
        if (
            boundary_config.minimum_boundary_length > 0
            and np.any(mask)
        ):
            components = label(mask, connectivity=2)
            for region in regionprops(components):
                if (
                    float(region.major_axis_length)
                    < boundary_config.minimum_boundary_length
                ):
                    mask[components == region.label] = False
        probability = probability_from_threshold(
            score, threshold, scale=max(0.20 * separation, 1e-3)
        )
        signature_support = (score / max(threshold, 1e-8)).astype(np.float32)
        evidence_maps = {
            "group_reciprocal_signature": signature_support,
            "domain_label": domain_label.astype(np.float32),
            "domain_membership_margin": domain_margin,
            "crystalline_support": crystalline_support.astype(np.float32),
            "local_contrast": resize(
                local_contrast,
                image_shape,
                order=1,
                mode="edge",
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.float32),
            "boundary_validity": valid.astype(np.float32),
        }
        regions = regions_from_mask(
            mask,
            np.asarray(score, dtype=np.float32),
            probability,
            evidence_maps,
            defect_type="two_phase_or_twin_boundary",
            support_threshold=1.0,
            evidence_names=["group_reciprocal_signature"],
        )
        metadata = {
            **input_metadata,
            "method": "local_polar_fourier_signature_two_prototype",
            "neural_training_used": False,
            "supervised_labels_used": False,
            "per_image_statistical_adaptation": True,
            "per_image_unsupervised_prototypes": True,
            "signature_grid_shape": list(signatures.shape[:2]),
            "signature_dimensions": int(signatures.shape[-1]),
            "domain_fractions": fractions,
            "specimen_mask_applied": specimen_mask_applied,
            "local_contrast_otsu_threshold": contrast_threshold,
            "low_contrast_fraction": low_fraction,
            "high_to_low_contrast_median_ratio": contrast_ratio,
            "prototype_separation_over_within_dispersion": separation,
            "minimum_required_separation": (
                boundary_config.minimum_signature_separation
            ),
            "genuine_two_phase": bool(genuine_two_phase),
            "valid_border_margin_pixels": border,
            "prototypes_standardized": prototypes.tolist(),
            "interpretation": (
                "Labels identify two reciprocal-texture domains and are "
                "per-image identifiers, not material chemistry classes."
            ),
            "context": {
                "skeleton_provided": context.skeleton is not None,
                "exemplar_provided": context.exemplar_image is not None,
            },
        }
        return DetectionResult(
            task=self.name,
            image_shape=image_shape,
            normalized_image=normalized_image,
            score_map=np.asarray(score, dtype=np.float32),
            probability_map=probability,
            mask=np.asarray(mask, dtype=bool),
            regions=regions,
            feature_maps={
                "domain_membership_margin": domain_margin,
            },
            evidence_maps=evidence_maps,
            metadata=metadata,
        )
