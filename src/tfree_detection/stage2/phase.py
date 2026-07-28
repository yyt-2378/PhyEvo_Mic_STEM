"""Geometric phase analysis for deformation around amorphous candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.measure import label

from ..config import PhaseAnalysisConfig


Peak = Tuple[float, float]


@dataclass
class PhaseAnalysisResult:
    maps: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    region_metrics: List[Dict[str, Any]]


def _select_reciprocal_basis(
    peaks_fy_fx: Sequence[Peak], minimum_angle_degrees: float
) -> Tuple[Peak, Peak]:
    unique = [
        (float(fy), float(fx))
        for fy, fx in peaks_fy_fx
        if fy > 0 or (abs(fy) < 1e-12 and fx > 0)
    ]
    if len(unique) < 2:
        raise ValueError("At least two reciprocal-lattice directions are required")
    first = unique[0]
    first_xy = np.asarray([first[1], first[0]])
    minimum_sine = float(np.sin(np.deg2rad(minimum_angle_degrees)))
    for candidate in unique[1:]:
        candidate_xy = np.asarray([candidate[1], candidate[0]])
        sine = abs(float(np.linalg.det(np.stack([first_xy, candidate_xy]))))
        sine /= np.linalg.norm(first_xy) * np.linalg.norm(candidate_xy) + 1e-12
        if sine >= minimum_sine:
            # Peaks arrive in descending spectral strength. Prefer the first
            # usable direction instead of a weaker but more orthogonal peak;
            # weak carriers create noisy and inflated phase gradients.
            return first, candidate
    raise ValueError(
        "No sufficiently independent reciprocal-lattice peak pair was found"
    )


def _phase_derivatives(
    image: np.ndarray,
    peaks: Sequence[Peak],
    config: PhaseAnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    padding = min(
        config.fft_padding,
        max(min(image.shape) // 3, 0),
    )
    padded = np.pad(image, ((padding, padding), (padding, padding)), mode="reflect")
    height, width = padded.shape
    frequency_y = np.fft.fftshift(np.fft.fftfreq(height))[:, None]
    frequency_x = np.fft.fftshift(np.fft.fftfreq(width))[None, :]
    spectrum = np.fft.fftshift(np.fft.fft2(padded - np.mean(padded)))
    yy = np.arange(height)[:, None]
    xx = np.arange(width)[None, :]

    derivative_x = []
    derivative_y = []
    amplitude_maps = []
    for peak_y, peak_x in peaks:
        frequency_mask = np.exp(
            -0.5
            * (
                (frequency_x - peak_x) ** 2
                + (frequency_y - peak_y) ** 2
            )
            / config.frequency_filter_sigma**2
        )
        wave = np.fft.ifft2(np.fft.ifftshift(spectrum * frequency_mask))
        wave *= np.exp(-2j * np.pi * (peak_x * xx + peak_y * yy))
        amplitude_squared = np.abs(wave) ** 2
        denominator = gaussian_filter(
            amplitude_squared,
            config.derivative_smoothing_sigma,
            mode="nearest",
        ) + 1e-12
        derivative_x.append(
            gaussian_filter(
                np.imag(np.conj(wave) * np.gradient(wave, axis=1)),
                config.derivative_smoothing_sigma,
                mode="nearest",
            )
            / denominator
        )
        derivative_y.append(
            gaussian_filter(
                np.imag(np.conj(wave) * np.gradient(wave, axis=0)),
                config.derivative_smoothing_sigma,
                mode="nearest",
            )
            / denominator
        )
        amplitude_maps.append(np.sqrt(amplitude_squared))

    crop = (
        (slice(padding, -padding), slice(padding, -padding))
        if padding
        else (slice(None), slice(None))
    )
    qx = np.stack([values[crop] for values in derivative_x])
    qy = np.stack([values[crop] for values in derivative_y])

    coherence_components = []
    for amplitude in amplitude_maps:
        cropped = amplitude[crop]
        background = gaussian_filter(
            cropped,
            config.amplitude_background_sigma,
            mode="reflect",
        )
        relative = cropped / (background + 1e-12)
        margin = min(max(config.fft_padding, 16), min(image.shape) // 4)
        interior = (
            relative[margin:-margin, margin:-margin]
            if margin and min(image.shape) > 2 * margin
            else relative
        )
        relative /= np.median(interior) + 1e-12
        coherence_components.append(relative)
    coherence = np.sqrt(
        np.maximum(coherence_components[0], 0)
        * np.maximum(coherence_components[1], 0)
    )
    return qx, qy, np.clip(coherence, 0, 2)


def _distribution(values: np.ndarray) -> Dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "count": 0,
            "median": float("nan"),
            "p75": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
        }
    percentiles = np.percentile(finite, [50, 75, 90, 95])
    return {
        "count": int(finite.size),
        "median": float(percentiles[0]),
        "p75": float(percentiles[1]),
        "p90": float(percentiles[2]),
        "p95": float(percentiles[3]),
    }


def _signed_distribution(values: np.ndarray) -> Dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "count": 0,
            "median": float("nan"),
            "p10": float("nan"),
            "p90": float("nan"),
        }
    percentiles = np.percentile(finite, [10, 50, 90])
    return {
        "count": int(finite.size),
        "median": float(percentiles[1]),
        "p10": float(percentiles[0]),
        "p90": float(percentiles[2]),
    }


def _robust_z(value: float, reference: np.ndarray) -> float:
    finite = reference[np.isfinite(reference)]
    if not finite.size:
        return float("nan")
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    return float((value - center) / max(scale, 1e-8))


def analyze_lattice_deformation(
    image: np.ndarray,
    candidate_mask: np.ndarray,
    peaks_fy_fx: Sequence[Peak],
    config: PhaseAnalysisConfig,
) -> PhaseAnalysisResult:
    """Estimate strain relative to the dominant global lattice.

    The derivative of the complex demodulated phase is used directly. This
    avoids unstable 2-D phase unwrapping inside an amorphous core.
    """

    peak_1, peak_2 = _select_reciprocal_basis(
        peaks_fy_fx, config.minimum_basis_angle_degrees
    )
    reciprocal = np.asarray(
        [[peak_1[1], peak_1[0]], [peak_2[1], peak_2[0]]],
        dtype=np.float64,
    )
    inverse_reciprocal = np.linalg.inv(reciprocal)
    qx, qy, coherence = _phase_derivatives(
        image, [peak_1, peak_2], config
    )

    deformation_x = (
        -np.einsum("ij,jhw->ihw", inverse_reciprocal, qx) / (2 * np.pi)
    )
    deformation_y = (
        -np.einsum("ij,jhw->ihw", inverse_reciprocal, qy) / (2 * np.pi)
    )
    exx = deformation_x[0]
    duy_dx = deformation_x[1]
    dux_dy = deformation_y[0]
    eyy = deformation_y[1]
    exy = 0.5 * (dux_dy + duy_dx)
    rotation = 0.5 * (duy_dx - dux_dy)
    strain_magnitude = np.sqrt(exx**2 + eyy**2 + 2 * exy**2)
    mean_normal = 0.5 * (exx + eyy)
    maximum_shear = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    principal_tensile = mean_normal + maximum_shear
    principal_compressive = mean_normal - maximum_shear

    direct_basis = inverse_reciprocal
    basis_vectors = [direct_basis[:, index] for index in range(2)]
    relative_length_maps = []
    for vector in basis_vectors:
        unit = vector / (np.linalg.norm(vector) + 1e-12)
        relative_length_maps.append(
            unit[0] ** 2 * exx
            + 2 * unit[0] * unit[1] * exy
            + unit[1] ** 2 * eyy
        )

    spatial_support = int(
        np.ceil(3.0 / (2 * np.pi * config.frequency_filter_sigma))
    )
    valid_margin = min(
        max(config.fft_padding // 2, spatial_support),
        min(image.shape) // 3,
    )
    validity = np.ones(image.shape, dtype=bool)
    if valid_margin:
        validity[:valid_margin, :] = False
        validity[-valid_margin:, :] = False
        validity[:, :valid_margin] = False
        validity[:, -valid_margin:] = False
    maps = {
        "gpa_strain_magnitude": strain_magnitude.astype(np.float32),
        "gpa_lattice_coherence": coherence.astype(np.float32),
        "gpa_exx": exx.astype(np.float32),
        "gpa_eyy": eyy.astype(np.float32),
        "gpa_exy": exy.astype(np.float32),
        "gpa_rotation": rotation.astype(np.float32),
        "gpa_principal_tensile": principal_tensile.astype(np.float32),
        "gpa_principal_compressive": principal_compressive.astype(np.float32),
        "gpa_basis1_length_change": relative_length_maps[0].astype(np.float32),
        "gpa_basis2_length_change": relative_length_maps[1].astype(np.float32),
        "gpa_validity": validity.astype(np.float32),
    }

    reliable = (coherence >= config.minimum_coherence) & validity
    all_labels = label(candidate_mask, connectivity=2)
    region_metrics: List[Dict[str, Any]] = []
    for region_id in range(1, int(np.max(all_labels)) + 1):
        core = all_labels == region_id
        distance = distance_transform_edt(~core)
        transition = (
            (distance > config.transition_inner_radius)
            & (distance <= config.transition_outer_radius)
            & ~candidate_mask
        )
        reference = (
            (distance > config.reference_inner_radius)
            & (distance <= config.reference_outer_radius)
            & ~candidate_mask
        )
        core_valid = core & reliable
        transition_valid = transition & reliable
        reference_valid = reference & reliable
        reference_strain = strain_magnitude[reference_valid]
        transition_summary = _distribution(
            strain_magnitude[transition_valid]
        )
        reference_summary = _distribution(reference_strain)
        core_coherence = _distribution(coherence[core])
        reference_coherence = _distribution(coherence[reference])
        strain_z = _robust_z(
            float(transition_summary["median"]), reference_strain
        )
        p90_enrichment = float(transition_summary["p90"]) / max(
            float(reference_summary["p90"]), 1e-12
        )
        coherence_loss = 1.0 - float(core_coherence["median"]) / max(
            float(reference_coherence["median"]), 1e-12
        )
        if coherence_loss >= 0.20 and p90_enrichment >= 1.25:
            phase_support = "strong"
        elif coherence_loss >= 0.10 or p90_enrichment >= 1.10:
            phase_support = "moderate"
        else:
            phase_support = "weak"
        region_metrics.append(
            {
                "region_id": region_id,
                "phase_support": phase_support,
                "core": {
                    "strain_magnitude": _distribution(
                        strain_magnitude[core_valid]
                    ),
                    "coherence": core_coherence,
                },
                "transition": {
                    "strain_magnitude": transition_summary,
                    "coherence": _distribution(coherence[transition]),
                    "principal_tensile_percent": _signed_distribution(
                        100 * principal_tensile[transition_valid]
                    ),
                    "principal_compressive_percent": _signed_distribution(
                        100 * principal_compressive[transition_valid]
                    ),
                    "basis1_length_change_percent": _signed_distribution(
                        100 * relative_length_maps[0][transition_valid]
                    ),
                    "basis2_length_change_percent": _signed_distribution(
                        100 * relative_length_maps[1][transition_valid]
                    ),
                },
                "reference": {
                    "strain_magnitude": reference_summary,
                    "coherence": reference_coherence,
                },
                "transition_strain_robust_z": strain_z,
                "transition_p90_enrichment": p90_enrichment,
                "core_coherence_loss_fraction": coherence_loss,
            }
        )

    metadata = {
        "method": "complex_demodulation_phase_gradient",
        "reciprocal_basis_g1_g2_xy_cycles_per_pixel": reciprocal.tolist(),
        "direct_basis_vectors_a1_a2_xy_pixels": [
            vector.tolist() for vector in basis_vectors
        ],
        "direct_basis_lengths_pixels": [
            float(np.linalg.norm(vector)) for vector in basis_vectors
        ],
        "reciprocal_condition_number": float(np.linalg.cond(reciprocal)),
        "minimum_reliable_coherence": config.minimum_coherence,
        "valid_border_margin_pixels": valid_margin,
        "strain_reference": "relative to dominant global lattice",
    }
    return PhaseAnalysisResult(
        maps=maps,
        metadata=metadata,
        region_metrics=region_metrics,
    )
