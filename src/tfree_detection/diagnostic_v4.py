"""Post-hoc v0.4 diagnostic candidates for broad external STEM defects.

This module is deliberately separate from the sealed v0.3 detector files.
It produces new, explicitly post-hoc masks without replacing the historical
Dennler prediction set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.ndimage import (
    binary_fill_holes,
    distance_transform_edt,
    gaussian_filter,
    maximum_filter,
)
from skimage.morphology import (
    binary_closing,
    binary_opening,
    dilation,
    disk,
    remove_small_objects,
)
from skimage.transform import resize

from .config import PreprocessConfig, SpectralConfig
from .features.lattice import (
    complex_demodulation,
    direct_basis_from_peaks,
    select_reciprocal_basis,
)
from .features.spectral import estimate_global_reciprocal_peaks
from .preprocessing import robust_normalize
from .tasks.interface import _two_robust_prototypes


@dataclass(frozen=True)
class PosthocV4Config:
    """Transparent thresholds for the v0.4 diagnostic candidate generator."""

    minimum_frequency: float = 0.025
    maximum_frequency: float = 0.40
    reciprocal_peak_count: int = 16
    minimum_basis_angle_degrees: float = 15.0
    minimum_lattice_spacing_pixels: float = 2.5
    maximum_lattice_spacing_pixels: float = 28.0
    coherence_threshold: float = 0.46
    carrier_reference_quantile: float = 0.80
    highpass_reference_quantile: float = 0.80
    minimum_coherence_region_cells: float = 5.0
    morphology_radius_cells: float = 0.45
    domain_sample_spacing_cells: float = 1.5
    minimum_domain_fraction: float = 0.025
    maximum_secondary_domain_fraction: float = 0.48
    minimum_domain_separation: float = 1.40
    minimum_domain_margin: float = 0.04
    minimum_boundary_deformation_z: float = 1.50
    minimum_boundary_support_fraction: float = 0.20
    minimum_domain_region_cells: float = 6.0
    boundary_width_cells: float = 0.45
    edge_reliability_floor: float = 0.25
    edge_reliability_ramp_cells: float = 4.0

    def validate(self) -> None:
        if not (
            0
            < self.minimum_frequency
            < self.maximum_frequency
            < 0.5
        ):
            raise ValueError("Diagnostic frequency bounds are invalid")
        if self.reciprocal_peak_count < 4:
            raise ValueError("At least four reciprocal peaks are required")
        if not 0 < self.coherence_threshold < 1:
            raise ValueError("coherence_threshold must be in (0, 1)")
        if not 0.5 < self.carrier_reference_quantile < 1:
            raise ValueError("carrier_reference_quantile must be in (0.5, 1)")
        if not 0.5 < self.highpass_reference_quantile < 1:
            raise ValueError("highpass_reference_quantile must be in (0.5, 1)")
        if not 0 <= self.edge_reliability_floor < 1:
            raise ValueError("edge_reliability_floor must be in [0, 1)")
        if not (
            0
            < self.minimum_domain_fraction
            < self.maximum_secondary_domain_fraction
            <= 0.5
        ):
            raise ValueError("Diagnostic domain fractions are invalid")
        if self.minimum_boundary_deformation_z < 0:
            raise ValueError("minimum_boundary_deformation_z cannot be negative")
        if not 0 <= self.minimum_boundary_support_fraction <= 1:
            raise ValueError(
                "minimum_boundary_support_fraction must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PosthocV4Result:
    """Masks and evidence maps kept separate by physical geometry."""

    normalized_image: np.ndarray
    coherence_loss_mask: np.ndarray
    secondary_symmetry_region_mask: np.ndarray
    structural_boundary_mask: np.ndarray
    any_defect_mask: np.ndarray
    reliability_map: np.ndarray
    evidence_maps: dict[str, np.ndarray]
    metadata: dict[str, Any]


@dataclass
class PosthocV4Composite:
    """Conservative primary masks plus explicitly exploratory augmentations."""

    coherence_loss_region: np.ndarray
    secondary_symmetry_region: np.ndarray
    structural_boundary: np.ndarray
    any_defect_region: np.ndarray
    domain_symmetry_region: np.ndarray
    augmented_secondary_symmetry_region: np.ndarray
    augmented_any_defect_region: np.ndarray


def compose_posthoc_v4(
    result: PosthocV4Result,
    historical_secondary_region: np.ndarray,
) -> PosthocV4Composite:
    """Keep an unstable domain proposal out of the conservative primary mask.

    The historical structure branch is read-only.  The new domain proposal
    remains available for ablation and visualization, but it does not enter
    the primary secondary-symmetry or any-defect outputs until it passes an
    independent normal-control gate.
    """

    historical = np.asarray(historical_secondary_region, dtype=bool)
    if historical.shape != result.coherence_loss_mask.shape:
        raise ValueError(
            "Historical secondary mask shape does not match v0.4 result"
        )
    domain = np.asarray(
        result.secondary_symmetry_region_mask,
        dtype=bool,
    )
    augmented_secondary = historical | domain
    return PosthocV4Composite(
        coherence_loss_region=np.asarray(
            result.coherence_loss_mask,
            dtype=bool,
        ),
        secondary_symmetry_region=historical.copy(),
        structural_boundary=np.asarray(
            result.structural_boundary_mask,
            dtype=bool,
        ),
        any_defect_region=(
            np.asarray(result.coherence_loss_mask, dtype=bool) | historical
        ),
        domain_symmetry_region=domain,
        augmented_secondary_symmetry_region=augmented_secondary,
        augmented_any_defect_region=(
            np.asarray(result.coherence_loss_mask, dtype=bool)
            | augmented_secondary
        ),
    )


def _spectral_config(config: PosthocV4Config) -> SpectralConfig:
    return SpectralConfig(
        patch_size=64,
        stride=12,
        min_frequency=config.minimum_frequency,
        max_frequency=config.maximum_frequency,
        global_peak_count=config.reciprocal_peak_count,
        global_peak_min_separation=0.008,
        global_peak_radius=0.012,
    )


def _lattice_model(
    image: np.ndarray,
    config: PosthocV4Config,
) -> tuple[list[tuple[float, float]], np.ndarray, float]:
    peaks = estimate_global_reciprocal_peaks(
        image,
        _spectral_config(config),
    )
    basis = select_reciprocal_basis(
        peaks,
        minimum_angle_degrees=config.minimum_basis_angle_degrees,
    )
    direct_basis = direct_basis_from_peaks(basis)
    lengths = np.linalg.norm(direct_basis, axis=0)
    finite = lengths[np.isfinite(lengths) & (lengths > 0)]
    if not finite.size:
        raise ValueError("Could not estimate a finite direct-lattice scale")
    spacing = float(
        np.clip(
            np.median(finite),
            config.minimum_lattice_spacing_pixels,
            config.maximum_lattice_spacing_pixels,
        )
    )
    return [tuple(values) for values in basis], direct_basis, spacing


def _edge_reliability(
    shape: tuple[int, int],
    spacing: float,
    config: PosthocV4Config,
) -> np.ndarray:
    height, width = shape
    yy, xx = np.mgrid[:height, :width]
    distance = np.minimum.reduce(
        (yy, xx, height - 1 - yy, width - 1 - xx)
    ).astype(float)
    ramp = max(config.edge_reliability_ramp_cells * spacing, 1.0)
    support = np.clip(distance / ramp, 0.0, 1.0)
    return (
        config.edge_reliability_floor
        + (1.0 - config.edge_reliability_floor) * support
    ).astype(np.float32)


def _carrier_and_deformation_maps(
    image: np.ndarray,
    basis: list[tuple[float, float]],
    spacing: float,
) -> tuple[np.ndarray, np.ndarray]:
    basis_radius = float(
        np.median([np.hypot(fy, fx) for fy, fx in basis])
    )
    frequency_sigma = float(np.clip(0.18 * basis_radius, 0.006, 0.028))
    derivative_sigma = float(np.clip(0.30 * spacing, 1.0, 6.0))
    qx, qy, amplitudes, _ = complex_demodulation(
        image,
        basis,
        frequency_sigma=frequency_sigma,
        derivative_sigma=derivative_sigma,
        padding=min(64, max(int(round(4 * spacing)), 16)),
    )
    carrier = np.sqrt(
        np.maximum(amplitudes[0], 0)
        * np.maximum(amplitudes[1], 0)
    )
    carrier = gaussian_filter(
        carrier,
        sigma=max(0.75 * spacing, 1.0),
        mode="reflect",
    )
    deformation = np.stack((qx[0], qy[0], qx[1], qy[1]), axis=-1)
    return carrier.astype(np.float32), deformation.astype(np.float32)


def _coherence_candidates(
    image: np.ndarray,
    carrier: np.ndarray,
    spacing: float,
    reliability: np.ndarray,
    config: PosthocV4Config,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    atomic_sigma = float(np.clip(0.28 * spacing, 0.8, 4.0))
    window_sigma = float(np.clip(1.25 * spacing, 2.0, 24.0))
    highpass = image - gaussian_filter(
        image,
        sigma=atomic_sigma,
        mode="reflect",
    )
    highpass_energy = np.sqrt(
        gaussian_filter(
            highpass**2,
            sigma=window_sigma,
            mode="reflect",
        )
        + 1e-12
    )
    carrier_reference = float(
        np.quantile(carrier, config.carrier_reference_quantile)
    )
    highpass_reference = float(
        np.quantile(
            highpass_energy,
            config.highpass_reference_quantile,
        )
    )
    carrier_ratio = carrier / max(carrier_reference, 1e-12)
    highpass_ratio = highpass_energy / max(highpass_reference, 1e-12)
    carrier_loss = np.clip(1.0 - carrier_ratio, 0.0, 1.0)
    highpass_loss = np.clip(1.0 - highpass_ratio, 0.0, 1.0)
    loss_score = gaussian_filter(
        0.68 * carrier_loss + 0.32 * highpass_loss,
        sigma=max(0.35 * spacing, 1.0),
        mode="reflect",
    )
    mask = loss_score >= config.coherence_threshold
    radius = max(
        1,
        int(round(config.morphology_radius_cells * spacing)),
    )
    mask = binary_closing(mask, disk(radius))
    mask = binary_opening(mask, disk(max(1, radius // 2)))
    minimum_area = max(
        16,
        int(
            round(
                config.minimum_coherence_region_cells
                * spacing**2
            )
        ),
    )
    mask = remove_small_objects(mask, min_size=minimum_area, connectivity=2)
    mask = binary_fill_holes(mask)
    maps = {
        "carrier_amplitude": carrier.astype(np.float32),
        "carrier_loss": carrier_loss.astype(np.float32),
        "highpass_energy": highpass_energy.astype(np.float32),
        "highpass_loss": highpass_loss.astype(np.float32),
        "coherence_loss_score": loss_score.astype(np.float32),
        "coherence_measurement_reliability": reliability.astype(np.float32),
    }
    metadata = {
        "carrier_reference": carrier_reference,
        "highpass_reference": highpass_reference,
        "threshold": config.coherence_threshold,
        "minimum_region_area_pixels": minimum_area,
        "border_touching_candidates_preserved": True,
    }
    return np.asarray(mask, dtype=bool), maps, metadata


def _domain_candidates(
    deformation: np.ndarray,
    carrier: np.ndarray,
    coherence_mask: np.ndarray,
    spacing: float,
    config: PosthocV4Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    step = max(
        3,
        int(round(config.domain_sample_spacing_cells * spacing)),
    )
    sampled = deformation[step // 2 :: step, step // 2 :: step]
    sampled_carrier = carrier[step // 2 :: step, step // 2 :: step]
    sampled_coherence = coherence_mask[
        step // 2 :: step,
        step // 2 :: step,
    ]
    carrier_floor = float(np.quantile(carrier, 0.25))
    valid = (
        np.all(np.isfinite(sampled), axis=-1)
        & (sampled_carrier >= carrier_floor)
        & ~sampled_coherence
    )
    labels, prototypes, margin, separation = _two_robust_prototypes(
        sampled,
        iterations=30,
        valid_grid=valid,
    )
    label_grid = labels.reshape(sampled.shape[:2])
    margin_grid = margin.reshape(sampled.shape[:2])
    valid_count = max(int(np.sum(valid)), 1)
    fractions = [
        float(np.sum((label_grid == cluster) & valid) / valid_count)
        for cluster in (0, 1)
    ]
    secondary_cluster = int(np.argmin(fractions))
    secondary_fraction = fractions[secondary_cluster]
    genuine_domains = bool(
        config.minimum_domain_fraction <= secondary_fraction
        <= config.maximum_secondary_domain_fraction
        and separation >= config.minimum_domain_separation
    )
    full_labels = resize(
        label_grid,
        carrier.shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.int8)
    full_margin = resize(
        margin_grid,
        carrier.shape,
        order=1,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32)
    full_valid = resize(
        valid.astype(float),
        carrier.shape,
        order=0,
        mode="edge",
        anti_aliasing=False,
        preserve_range=True,
    ) >= 0.5
    smoothed_deformation = np.stack(
        [
            gaussian_filter(
                deformation[..., index],
                sigma=max(0.35 * spacing, 1.0),
                mode="reflect",
            )
            for index in range(deformation.shape[-1])
        ],
        axis=-1,
    )
    deformation_gradient = np.zeros(carrier.shape, dtype=float)
    for index in range(smoothed_deformation.shape[-1]):
        gradient_y, gradient_x = np.gradient(
            smoothed_deformation[..., index]
        )
        deformation_gradient += gradient_x**2 + gradient_y**2
    deformation_gradient = np.sqrt(deformation_gradient)
    valid_gradients = deformation_gradient[full_valid]
    gradient_location = (
        float(np.median(valid_gradients))
        if valid_gradients.size
        else 0.0
    )
    gradient_scale = (
        float(
            1.4826
            * np.median(np.abs(valid_gradients - gradient_location))
        )
        if valid_gradients.size
        else 1.0
    )
    gradient_scale = max(gradient_scale, 1e-8)
    deformation_gradient_z = np.clip(
        (deformation_gradient - gradient_location) / gradient_scale,
        0.0,
        12.0,
    )
    first = maximum_filter(
        (full_labels == 0).astype(np.uint8),
        size=3,
        mode="nearest",
    )
    second = maximum_filter(
        (full_labels == 1).astype(np.uint8),
        size=3,
        mode="nearest",
    )
    raw_boundary = (first > 0) & (second > 0) & full_valid
    supported_boundary = raw_boundary & (
        deformation_gradient_z
        >= config.minimum_boundary_deformation_z
    )
    boundary_support_fraction = (
        float(np.sum(supported_boundary) / np.sum(raw_boundary))
        if np.any(raw_boundary)
        else 0.0
    )
    genuine_domains = bool(
        genuine_domains
        and boundary_support_fraction
        >= config.minimum_boundary_support_fraction
    )
    if genuine_domains:
        region = (
            (full_labels == secondary_cluster)
            & full_valid
            & (full_margin >= config.minimum_domain_margin)
            & ~coherence_mask
        )
        minimum_area = max(
            16,
            int(round(config.minimum_domain_region_cells * spacing**2)),
        )
        region = remove_small_objects(
            region,
            min_size=minimum_area,
            connectivity=2,
        )
        region = binary_closing(
            region,
            disk(max(1, int(round(0.35 * spacing)))),
        )
        boundary = supported_boundary
        boundary = dilation(
            boundary,
            disk(
                max(
                    1,
                    int(round(config.boundary_width_cells * spacing)),
                )
            ),
        )
        boundary &= ~coherence_mask
    else:
        region = np.zeros(carrier.shape, dtype=bool)
        boundary = np.zeros(carrier.shape, dtype=bool)
        minimum_area = max(
            16,
            int(round(config.minimum_domain_region_cells * spacing**2)),
        )
    maps = {
        "domain_label": full_labels.astype(np.float32),
        "domain_membership_margin": full_margin,
        "domain_validity": full_valid.astype(np.float32),
        "deformation_gradient": deformation_gradient.astype(np.float32),
        "deformation_gradient_z": deformation_gradient_z.astype(np.float32),
    }
    metadata = {
        "sample_step_pixels": step,
        "prototype_separation_over_within_dispersion": float(separation),
        "minimum_required_separation": config.minimum_domain_separation,
        "domain_fractions": fractions,
        "secondary_cluster": secondary_cluster,
        "secondary_fraction": secondary_fraction,
        "boundary_support_fraction": boundary_support_fraction,
        "minimum_boundary_support_fraction": (
            config.minimum_boundary_support_fraction
        ),
        "minimum_boundary_deformation_z": (
            config.minimum_boundary_deformation_z
        ),
        "genuine_domains": genuine_domains,
        "minimum_region_area_pixels": minimum_area,
        "prototypes_standardized": prototypes.tolist(),
    }
    return (
        np.asarray(region, dtype=bool),
        np.asarray(boundary, dtype=bool),
        maps,
        metadata,
    )


def analyze_posthoc_v4(
    image: np.ndarray,
    config: PosthocV4Config | None = None,
) -> PosthocV4Result:
    """Generate scale-aware region and boundary candidates without training."""

    active = config or PosthocV4Config()
    active.validate()
    values = np.asarray(image)
    if values.ndim != 2 or min(values.shape) < 64:
        raise ValueError("Post-hoc v0.4 analysis needs a 2-D image >= 64 px")
    if not np.all(np.isfinite(values)):
        raise ValueError("Post-hoc v0.4 image contains non-finite values")
    normalized, preprocessing = robust_normalize(
        values,
        PreprocessConfig(lower_percentile=0.5, upper_percentile=99.5),
    )
    basis, direct_basis, spacing = _lattice_model(normalized, active)
    reliability = _edge_reliability(normalized.shape, spacing, active)
    carrier, deformation = _carrier_and_deformation_maps(
        normalized,
        basis,
        spacing,
    )
    coherence, coherence_maps, coherence_metadata = (
        _coherence_candidates(
            normalized,
            carrier,
            spacing,
            reliability,
            active,
        )
    )
    domain_region, boundary, domain_maps, domain_metadata = (
        _domain_candidates(
            deformation,
            carrier,
            coherence,
            spacing,
            active,
        )
    )
    secondary = domain_region | boundary
    any_defect = coherence | secondary
    return PosthocV4Result(
        normalized_image=normalized.astype(np.float32),
        coherence_loss_mask=coherence,
        secondary_symmetry_region_mask=secondary,
        structural_boundary_mask=boundary,
        any_defect_mask=any_defect,
        reliability_map=reliability,
        evidence_maps={**coherence_maps, **domain_maps},
        metadata={
            "method": "posthoc-v0.4-scale-aware-physics-diagnostic",
            "scientific_status": "posthoc_diagnostic_not_locked",
            "neural_training_used": False,
            "supervised_labels_used": False,
            "legacy_locked_prediction_replaced": False,
            "lattice_spacing_pixels": spacing,
            "reciprocal_basis_fy_fx": [
                [float(fy), float(fx)] for fy, fx in basis
            ],
            "direct_basis_xy_columns": direct_basis.tolist(),
            "preprocessing": preprocessing,
            "coherence_loss": coherence_metadata,
            "domain_symmetry": domain_metadata,
            "config": active.to_dict(),
        },
    )
