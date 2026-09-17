"""Compare lattice deformation against one explicit reference image.

The ordinary GPA stage estimates local departures from each image's own
dominant carrier.  This module adds the missing cross-image contract: match the
moving reciprocal/direct basis to the reference basis, remove isotropic zoom
and rigid rotation as registration nuisance, and express the residual lattice
shape plus local GPA strain in the reference coordinate system.

Absolute translation and physical stress are intentionally not inferred.  The
former requires shared landmarks/overlap, while the latter requires calibrated
lengths and material elastic constants.
"""

from __future__ import annotations

import csv
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import logm, polar
from scipy.ndimage import distance_transform_edt
from scipy.stats import hypergeom
from skimage.transform import AffineTransform, warp

from .models import DetectionResult


_STRAIN_MAPS = ("gpa_exx", "gpa_eyy", "gpa_exy")
_RELIABILITY_MAPS = ("gpa_validity", "gpa_lattice_coherence")


@dataclass
class ReferenceDeformationResult:
    """One moving image expressed against a fixed lattice reference."""

    label: str
    summary: Dict[str, Any]
    maps_native: Dict[str, np.ndarray]
    maps_aligned: Dict[str, np.ndarray]
    aligned_image: np.ndarray
    aligned_reliability: np.ndarray


def _finite_or_none(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def _percentile_rank(sorted_values: np.ndarray, value: float) -> float:
    if not np.isfinite(value) or not len(sorted_values):
        return float("nan")
    return float(
        100.0
        * np.searchsorted(sorted_values, value, side="right")
        / len(sorted_values)
    )


def summarize_candidate_hotspot_overlap(
    comparison: ReferenceDeformationResult,
    candidate_points_xy: np.ndarray,
    *,
    candidate_ids: Sequence[str] | None = None,
    hotspot_quantile: float = 0.90,
    support_radius_pixels: float = 3.0,
) -> Dict[str, Any]:
    """Audit candidate locations against native-coordinate GPA hotspots.

    Candidate coordinates and ``maps_native`` share the moving image pixel
    frame.  GPA is deliberately applied *after* structure/contrast candidate
    selection: hotspot proximity is independent supporting evidence, never a
    gate that can create or remove a chemical candidate.
    """

    points = np.asarray(candidate_points_xy, dtype=float)
    if points.size == 0:
        points = np.empty((0, 2), dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("candidate_points_xy must have shape (n, 2)")
    if not 0.5 < hotspot_quantile < 1.0:
        raise ValueError("hotspot_quantile must lie strictly between 0.5 and 1")
    if support_radius_pixels < 0:
        raise ValueError("support_radius_pixels cannot be negative")
    if candidate_ids is None:
        ids = [f"candidate_{index + 1}" for index in range(len(points))]
    else:
        ids = [str(item) for item in candidate_ids]
        if len(ids) != len(points):
            raise ValueError("candidate_ids must match candidate_points_xy")

    maps = comparison.maps_native
    required = (
        "relative_distortion_magnitude",
        "stress_hotspot_proxy",
        "reliability",
    )
    missing = [name for name in required if name not in maps]
    if missing:
        raise ValueError(f"GPA comparison is missing native map(s): {missing}")
    distortion = np.asarray(maps[required[0]], dtype=float)
    stress_proxy = np.asarray(maps[required[1]], dtype=float)
    reliability = np.asarray(maps[required[2]], dtype=float)
    if distortion.shape != stress_proxy.shape or distortion.shape != reliability.shape:
        raise ValueError("Native GPA maps must have identical shapes")
    reliable = (
        (reliability >= 0.5)
        & np.isfinite(distortion)
        & np.isfinite(stress_proxy)
    )
    if not np.any(reliable):
        raise ValueError("GPA overlap audit has no reliable pixels")

    distortion_threshold = float(np.quantile(distortion[reliable], hotspot_quantile))
    stress_threshold = float(np.quantile(stress_proxy[reliable], hotspot_quantile))
    distortion_hotspot = reliable & (distortion >= distortion_threshold)
    stress_hotspot = reliable & (stress_proxy >= stress_threshold)
    distortion_distance = distance_transform_edt(~distortion_hotspot)
    stress_distance = distance_transform_edt(~stress_hotspot)
    sorted_distortion = np.sort(distortion[reliable])
    sorted_stress = np.sort(stress_proxy[reliable])
    height, width = distortion.shape
    radius = int(np.ceil(support_radius_pixels))

    records: list[Dict[str, Any]] = []
    for candidate_id, (x, y) in zip(ids, points):
        inside = bool(0 <= x < width and 0 <= y < height)
        if not inside:
            records.append(
                {
                    "candidate_id": candidate_id,
                    "x_pixels": float(x),
                    "y_pixels": float(y),
                    "inside_image": False,
                    "center_reliable": False,
                    "local_reliable_fraction": 0.0,
                    "distortion_center_percent": None,
                    "distortion_local_p90_percent": None,
                    "distortion_local_p90_percentile_rank": None,
                    "nearest_distortion_top_decile_pixels": None,
                    "within_distortion_hotspot_support_radius": False,
                    "stress_proxy_center_percent": None,
                    "stress_proxy_local_p90_percent": None,
                    "stress_proxy_local_p90_percentile_rank": None,
                    "nearest_stress_top_decile_pixels": None,
                    "within_stress_hotspot_support_radius": False,
                    "dual_hotspot_support": False,
                }
            )
            continue
        xi = int(np.clip(np.rint(x), 0, width - 1))
        yi = int(np.clip(np.rint(y), 0, height - 1))
        x0, x1 = max(0, xi - radius), min(width, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(height, yi + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        disk = (xx - x) ** 2 + (yy - y) ** 2 <= support_radius_pixels**2
        local_reliable = disk & reliable[y0:y1, x0:x1]
        local_distortion = distortion[y0:y1, x0:x1][local_reliable]
        local_stress = stress_proxy[y0:y1, x0:x1][local_reliable]
        distortion_p90 = (
            float(np.quantile(local_distortion, 0.90))
            if len(local_distortion)
            else float("nan")
        )
        stress_p90 = (
            float(np.quantile(local_stress, 0.90))
            if len(local_stress)
            else float("nan")
        )
        distortion_nearest = float(distortion_distance[yi, xi])
        stress_nearest = float(stress_distance[yi, xi])
        distortion_supported = bool(
            distortion_nearest <= support_radius_pixels
        )
        stress_supported = bool(stress_nearest <= support_radius_pixels)
        records.append(
            {
                "candidate_id": candidate_id,
                "x_pixels": float(x),
                "y_pixels": float(y),
                "inside_image": True,
                "center_reliable": bool(reliable[yi, xi]),
                "local_reliable_fraction": float(
                    np.sum(local_reliable) / max(int(np.sum(disk)), 1)
                ),
                "distortion_center_percent": _finite_or_none(
                    100.0 * distortion[yi, xi]
                ) if reliable[yi, xi] else None,
                "distortion_local_p90_percent": _finite_or_none(
                    100.0 * distortion_p90
                ),
                "distortion_local_p90_percentile_rank": _finite_or_none(
                    _percentile_rank(sorted_distortion, distortion_p90)
                ),
                "nearest_distortion_top_decile_pixels": distortion_nearest,
                "within_distortion_hotspot_support_radius": distortion_supported,
                "stress_proxy_center_percent": _finite_or_none(
                    100.0 * stress_proxy[yi, xi]
                ) if reliable[yi, xi] else None,
                "stress_proxy_local_p90_percent": _finite_or_none(
                    100.0 * stress_p90
                ),
                "stress_proxy_local_p90_percentile_rank": _finite_or_none(
                    _percentile_rank(sorted_stress, stress_p90)
                ),
                "nearest_stress_top_decile_pixels": stress_nearest,
                "within_stress_hotspot_support_radius": stress_supported,
                "dual_hotspot_support": bool(
                    distortion_supported and stress_supported
                ),
            }
        )

    return {
        "moving_label": comparison.label,
        "candidate_count": len(records),
        "hotspot_quantile": float(hotspot_quantile),
        "support_radius_pixels": float(support_radius_pixels),
        "distortion_hotspot_threshold_percent": 100.0 * distortion_threshold,
        "stress_proxy_hotspot_threshold_percent": 100.0 * stress_threshold,
        "candidate_count_with_distortion_hotspot_support": int(
            sum(item["within_distortion_hotspot_support_radius"] for item in records)
        ),
        "candidate_count_with_stress_hotspot_support": int(
            sum(item["within_stress_hotspot_support_radius"] for item in records)
        ),
        "candidate_count_with_dual_hotspot_support": int(
            sum(item["dual_hotspot_support"] for item in records)
        ),
        "candidates": records,
    }


def _phase_metadata(result: DetectionResult) -> Dict[str, Any]:
    try:
        phase = result.metadata["stage2"]["phase"]
    except (KeyError, TypeError) as error:
        raise ValueError(
            "Reference comparison requires a successful stage-2 GPA result"
        ) from error
    if not isinstance(phase, dict):
        raise ValueError("Stage-2 GPA metadata must be a dictionary")
    return phase


def _direct_basis(result: DetectionResult) -> np.ndarray:
    phase = _phase_metadata(result)
    try:
        vectors = np.asarray(
            phase["direct_basis_vectors_a1_a2_xy_pixels"], dtype=float
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("GPA direct-lattice basis is missing or invalid") from error
    if vectors.shape != (2, 2) or not np.all(np.isfinite(vectors)):
        raise ValueError("GPA direct-lattice basis must contain two finite 2-D vectors")
    basis = np.column_stack(vectors)
    if abs(float(np.linalg.det(basis))) < 1e-8:
        raise ValueError("GPA direct-lattice basis is singular")
    return basis


def _proper_signed_permutations() -> Iterable[np.ndarray]:
    for permutation in ((0, 1), (1, 0)):
        for signs in itertools.product((-1.0, 1.0), repeat=2):
            transform = np.zeros((2, 2), dtype=float)
            transform[permutation[0], 0] = signs[0]
            transform[permutation[1], 1] = signs[1]
            yield transform


def match_lattice_bases(
    reference_basis: np.ndarray,
    moving_basis: np.ndarray,
) -> Dict[str, Any]:
    """Match basis order/sign and estimate moving-to-reference similarity.

    Crystallographic basis sign and ordering are ambiguous.  All signed
    permutations are tested, but reflected image mappings are rejected.  If a
    180-degree lattice-equivalent solution exists, the smallest rigid rotation
    is selected for an interpretable visualization.
    """

    reference = np.asarray(reference_basis, dtype=float)
    moving = np.asarray(moving_basis, dtype=float)
    if reference.shape != (2, 2) or moving.shape != (2, 2):
        raise ValueError("Reference and moving bases must both have shape (2, 2)")
    candidates: list[Dict[str, Any]] = []
    for basis_transform in _proper_signed_permutations():
        matched = moving @ basis_transform
        full_mapping = reference @ np.linalg.inv(matched)
        determinant = float(np.linalg.det(full_mapping))
        if determinant <= 0:
            continue
        left, singular_values, right = np.linalg.svd(full_mapping)
        rotation = left @ right
        if float(np.linalg.det(rotation)) <= 0:
            continue
        scale = float(np.sqrt(determinant))
        residual = float(
            np.linalg.norm(full_mapping - scale * rotation)
            / (np.linalg.norm(full_mapping) + 1e-12)
        )
        angle = float(
            np.rad2deg(np.arctan2(rotation[1, 0], rotation[0, 0]))
        )
        candidates.append(
            {
                "basis_transform": basis_transform,
                "full_mapping": full_mapping,
                "rotation": rotation,
                "scale": scale,
                "relative_residual": residual,
                "rotation_degrees": angle,
                "singular_values": singular_values,
            }
        )
    if not candidates:
        raise ValueError("No proper lattice-basis correspondence was found")
    minimum_residual = min(item["relative_residual"] for item in candidates)
    near_optimal = [
        item
        for item in candidates
        if item["relative_residual"] <= minimum_residual + 0.02
    ]
    selected = min(
        near_optimal,
        key=lambda item: (
            abs(item["rotation_degrees"]),
            item["relative_residual"],
        ),
    )
    selected["similarity_mapping"] = (
        selected["scale"] * selected["rotation"]
    )
    return selected


def _reliable_mask(result: DetectionResult) -> np.ndarray:
    for name in (*_STRAIN_MAPS, *_RELIABILITY_MAPS):
        if name not in result.evidence_maps:
            raise ValueError(f"GPA result is missing evidence map {name!r}")
    phase = _phase_metadata(result)
    threshold = float(phase.get("minimum_reliable_coherence", 0.65))
    return (
        np.asarray(result.evidence_maps["gpa_validity"]) >= 0.5
    ) & (
        np.asarray(result.evidence_maps["gpa_lattice_coherence"])
        >= threshold
    )


def _centered_local_strain(
    result: DetectionResult, reliable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if not np.any(reliable):
        raise ValueError("GPA comparison has no reliable pixels")
    shape = result.image_shape
    strain = np.zeros(shape + (2, 2), dtype=np.float64)
    strain[..., 0, 0] = result.evidence_maps["gpa_exx"]
    strain[..., 1, 1] = result.evidence_maps["gpa_eyy"]
    strain[..., 0, 1] = result.evidence_maps["gpa_exy"]
    strain[..., 1, 0] = result.evidence_maps["gpa_exy"]
    component_center = np.median(strain[reliable], axis=0)
    return strain - component_center, component_center


def _derived_maps(strain: np.ndarray) -> Dict[str, np.ndarray]:
    exx = strain[..., 0, 0]
    eyy = strain[..., 1, 1]
    exy = 0.5 * (strain[..., 0, 1] + strain[..., 1, 0])
    mean_normal = 0.5 * (exx + eyy)
    maximum_shear = np.sqrt((0.5 * (exx - eyy)) ** 2 + exy**2)
    return {
        "relative_exx": exx.astype(np.float32),
        "relative_eyy": eyy.astype(np.float32),
        "relative_exy": exy.astype(np.float32),
        "relative_distortion_magnitude": np.sqrt(
            exx**2 + eyy**2 + 2.0 * exy**2
        ).astype(np.float32),
        "relative_principal_tensile": (
            mean_normal + maximum_shear
        ).astype(np.float32),
        "relative_principal_compressive": (
            mean_normal - maximum_shear
        ).astype(np.float32),
        "stress_hotspot_proxy": maximum_shear.astype(np.float32),
    }


def _percentile_summary(values: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    selected = np.asarray(values, dtype=float)[mask]
    if not selected.size:
        raise ValueError("Cannot summarize an empty reliable region")
    p10, median, p90, p95 = np.percentile(selected, (10, 50, 90, 95))
    return {
        "p10": float(p10),
        "median": float(median),
        "p90": float(p90),
        "p95": float(p95),
    }


def _centered_transform(
    linear: np.ndarray,
    moving_shape: tuple[int, int],
    reference_shape: tuple[int, int],
) -> AffineTransform:
    moving_center = np.asarray(
        [(moving_shape[1] - 1) / 2.0, (moving_shape[0] - 1) / 2.0]
    )
    reference_center = np.asarray(
        [(reference_shape[1] - 1) / 2.0, (reference_shape[0] - 1) / 2.0]
    )
    translation = reference_center - linear @ moving_center
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = linear
    matrix[:2, 2] = translation
    return AffineTransform(matrix=matrix)


def _warp_to_reference(
    values: np.ndarray,
    transform: AffineTransform,
    output_shape: tuple[int, int],
    *,
    order: int,
    cval: float,
) -> np.ndarray:
    return np.asarray(
        warp(
            values,
            inverse_map=transform.inverse,
            output_shape=output_shape,
            order=order,
            mode="constant",
            cval=cval,
            preserve_range=True,
        )
    )


def reference_baseline(result: DetectionResult, label: str) -> Dict[str, Any]:
    """Summarize the reference image's own centered GPA variation."""

    reliable = _reliable_mask(result)
    local, center = _centered_local_strain(result, reliable)
    maps = _derived_maps(local)
    return {
        "label": label,
        "image_shape": list(result.image_shape),
        "reliable_pixel_fraction": float(np.mean(reliable)),
        "gpa_component_center_removed": center.tolist(),
        "distortion_percent": {
            key: 100.0 * value
            for key, value in _percentile_summary(
                maps["relative_distortion_magnitude"], reliable
            ).items()
        },
        "maps": maps,
        "reliable_mask": reliable,
    }


def compare_to_reference(
    reference: DetectionResult,
    moving: DetectionResult,
    *,
    reference_label: str,
    moving_label: str,
) -> ReferenceDeformationResult:
    """Express a moving image's lattice deformation relative to a reference."""

    reference_basis = _direct_basis(reference)
    moving_basis = _direct_basis(moving)
    alignment = match_lattice_bases(reference_basis, moving_basis)
    similarity = alignment["similarity_mapping"]
    matched_basis = moving_basis @ alignment["basis_transform"]

    # The similarity is a registration nuisance.  What remains after applying
    # it is a scale-independent lattice-shape deformation relative to Y.
    residual_gradient = (
        similarity @ matched_basis @ np.linalg.inv(reference_basis)
    )
    _, residual_stretch = polar(residual_gradient)
    global_log_strain = np.asarray(logm(residual_stretch).real, dtype=float)
    principal_global = np.linalg.eigvalsh(global_log_strain)

    reliable = _reliable_mask(moving)
    local_strain, component_center = _centered_local_strain(moving, reliable)
    rotation = alignment["rotation"]
    local_in_reference_axes = np.einsum(
        "ij,...jk,lk->...il", rotation, local_strain, rotation
    )
    total_strain = local_in_reference_axes + global_log_strain
    maps_native = _derived_maps(total_strain)
    local_rotation = np.asarray(
        moving.evidence_maps.get(
            "gpa_rotation", np.zeros(moving.image_shape, dtype=float)
        ),
        dtype=float,
    )
    maps_native["relative_local_rotation"] = (
        local_rotation - float(np.median(local_rotation[reliable]))
    ).astype(np.float32)
    maps_native["reliability"] = reliable.astype(np.float32)

    transform = _centered_transform(
        similarity, moving.image_shape, reference.image_shape
    )
    aligned_image = _warp_to_reference(
        moving.normalized_image,
        transform,
        reference.image_shape,
        order=3,
        cval=float(np.median(moving.normalized_image)),
    ).astype(np.float32)
    aligned_reliability = _warp_to_reference(
        reliable.astype(float),
        transform,
        reference.image_shape,
        order=0,
        cval=0.0,
    ) >= 0.5
    maps_aligned: Dict[str, np.ndarray] = {}
    for name, values in maps_native.items():
        order = 0 if name == "reliability" else 1
        aligned = _warp_to_reference(
            np.asarray(values),
            transform,
            reference.image_shape,
            order=order,
            cval=0.0,
        ).astype(np.float32)
        if name != "reliability":
            aligned[~aligned_reliability] = np.nan
        maps_aligned[name] = aligned

    distributions = {
        name: {
            key: 100.0 * value
            for key, value in _percentile_summary(values, reliable).items()
        }
        for name, values in maps_native.items()
        if name not in {"reliability", "relative_local_rotation"}
    }
    rotation_distribution_degrees = {
        key: float(np.rad2deg(value))
        for key, value in _percentile_summary(
            maps_native["relative_local_rotation"], reliable
        ).items()
    }
    translation = transform.params[:2, 2]
    summary: Dict[str, Any] = {
        "reference_label": reference_label,
        "moving_label": moving_label,
        "moving_image_shape": list(moving.image_shape),
        "reference_image_shape": list(reference.image_shape),
        "method": (
            "global lattice-basis similarity registration plus centered local "
            "GPA strain, expressed in reference axes"
        ),
        "registration_nuisance": {
            "moving_to_reference_isotropic_pixel_scale": alignment["scale"],
            "moving_to_reference_rotation_degrees_modulo_lattice_symmetry": (
                alignment["rotation_degrees"]
            ),
            "center_alignment_translation_xy_pixels_visualization_only": (
                translation.tolist()
            ),
            "basis_fit_relative_residual": alignment["relative_residual"],
            "basis_sign_permutation": alignment["basis_transform"].tolist(),
            "absolute_translation_status": (
                "not identifiable without shared landmarks or a declared "
                "same-field overlap"
            ),
        },
        "global_scale_independent_lattice_shape": {
            "residual_deformation_gradient": residual_gradient.tolist(),
            "log_strain_tensor_percent": (100.0 * global_log_strain).tolist(),
            "principal_log_strain_percent": (
                100.0 * principal_global
            ).tolist(),
            "principal_distortion_span_percent": float(
                100.0 * (principal_global[1] - principal_global[0])
            ),
        },
        "local_gpa_component_center_removed": component_center.tolist(),
        "reliable_pixel_fraction_native": float(np.mean(reliable)),
        "reliable_pixel_fraction_aligned_canvas": float(
            np.mean(aligned_reliability)
        ),
        "relative_map_distributions_percent": distributions,
        "relative_local_rotation_distribution_degrees": (
            rotation_distribution_degrees
        ),
        "stress_status": (
            "physical stress not computed: elastic constants, thickness/model "
            "choice, and physical pixel calibration were not supplied"
        ),
        "stress_hotspot_proxy_definition": (
            "maximum-shear strain magnitude; dimensionless and not Pa/GPa"
        ),
        "absolute_isotropic_expansion_status": (
            "not identifiable because image pixel scale/FOV calibration differs"
        ),
    }
    return ReferenceDeformationResult(
        label=moving_label,
        summary=summary,
        maps_native=maps_native,
        maps_aligned=maps_aligned,
        aligned_image=aligned_image,
        aligned_reliability=aligned_reliability,
    )


def _masked(values: np.ndarray, mask: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.array(values, mask=(~mask) | ~np.isfinite(values))


def _save_target_figure(
    comparison: ReferenceDeformationResult,
    target: Path,
    dpi: int,
) -> None:
    mask = comparison.aligned_reliability
    fields = comparison.maps_aligned
    magnitude = 100.0 * fields["relative_distortion_magnitude"]
    signed = [
        100.0 * fields["relative_exx"],
        100.0 * fields["relative_eyy"],
        100.0 * fields["relative_exy"],
    ]
    magnitude_limit = float(np.nanpercentile(magnitude[mask], 95))
    signed_limit = max(
        float(np.nanpercentile(np.abs(values[mask]), 95)) for values in signed
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    axes[0, 0].imshow(comparison.aligned_image, cmap="gray")
    axes[0, 0].set_title(f"{comparison.label} aligned to reference axes")
    entries = [
        (axes[0, 1], magnitude, "Relative distortion (%)", "magma", 0, magnitude_limit),
        (axes[0, 2], 100.0 * fields["stress_hotspot_proxy"], "Stress-hotspot proxy (%)", "inferno", 0, magnitude_limit),
        (axes[1, 0], signed[0], "Relative exx (%)", "coolwarm", -signed_limit, signed_limit),
        (axes[1, 1], signed[1], "Relative eyy (%)", "coolwarm", -signed_limit, signed_limit),
        (axes[1, 2], signed[2], "Relative exy (%)", "coolwarm", -signed_limit, signed_limit),
    ]
    for axis, values, title, cmap, lower, upper in entries:
        image = axis.imshow(_masked(values, mask), cmap=cmap, vmin=lower, vmax=upper)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046)
    for axis in axes.flat:
        axis.set_axis_off()
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_reference_comparison(
    reference: DetectionResult,
    comparisons: list[ReferenceDeformationResult],
    *,
    reference_label: str,
    output_directory: str | Path,
    dpi: int = 160,
) -> Dict[str, str]:
    """Save an auditable cross-image comparison without overwriting outputs."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite comparison output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    baseline = reference_baseline(reference, reference_label)
    baseline_maps = baseline.pop("maps")
    baseline_mask = baseline.pop("reliable_mask")
    y_p90 = float(baseline["distortion_percent"]["p90"])
    summaries = []
    for comparison in comparisons:
        summary = comparison.summary
        native_magnitude = comparison.maps_native["relative_distortion_magnitude"]
        native_reliable = comparison.maps_native["reliability"] >= 0.5
        summary["comparison_to_reference_variation"] = {
            "reference_distortion_p90_percent": y_p90,
            "moving_fraction_above_reference_p90": float(
                np.mean(100.0 * native_magnitude[native_reliable] > y_p90)
            ),
        }
        summaries.append(summary)
        target_directory = output / comparison.label
        target_directory.mkdir(exist_ok=False)
        np.savez_compressed(
            target_directory / "relative_deformation_maps_native.npz",
            **comparison.maps_native,
        )
        np.savez_compressed(
            target_directory / "relative_deformation_maps_aligned_to_reference.npz",
            **comparison.maps_aligned,
            aligned_image=comparison.aligned_image,
            aligned_reliability=comparison.aligned_reliability.astype(np.float32),
        )
        plt.imsave(
            target_directory / "aligned_to_reference.png",
            comparison.aligned_image,
            cmap="gray",
        )
        _save_target_figure(
            comparison,
            target_directory / "relative_deformation.png",
            dpi,
        )

    payload = {
        "schema_version": "reference-deformation-v1",
        "reference": baseline,
        "comparisons": summaries,
        "interpretation_contract": {
            "valid": [
                "scale-independent global lattice-shape difference",
                "local relative strain/distortion in reference axes",
                "dimensionless stress-hotspot ranking",
            ],
            "not_valid_without_more_metadata": [
                "absolute translation/migration",
                "absolute isotropic lattice expansion",
                "stress in Pa or GPa",
                "pointwise subtraction between non-overlapping fields of view",
            ],
        },
    }
    summary_path = output / "reference_deformation_summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    csv_path = output / "reference_deformation_summary.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "moving_label",
                "scale_to_reference",
                "rotation_degrees",
                "global_principal_compressive_percent",
                "global_principal_tensile_percent",
                "global_distortion_span_percent",
                "relative_distortion_median_percent",
                "relative_distortion_p90_percent",
                "fraction_above_reference_p90",
                "reliable_pixel_fraction",
            ),
        )
        writer.writeheader()
        for summary in summaries:
            global_principal = summary["global_scale_independent_lattice_shape"][
                "principal_log_strain_percent"
            ]
            distortion = summary["relative_map_distributions_percent"][
                "relative_distortion_magnitude"
            ]
            writer.writerow(
                {
                    "moving_label": summary["moving_label"],
                    "scale_to_reference": summary["registration_nuisance"][
                        "moving_to_reference_isotropic_pixel_scale"
                    ],
                    "rotation_degrees": summary["registration_nuisance"][
                        "moving_to_reference_rotation_degrees_modulo_lattice_symmetry"
                    ],
                    "global_principal_compressive_percent": global_principal[0],
                    "global_principal_tensile_percent": global_principal[1],
                    "global_distortion_span_percent": summary[
                        "global_scale_independent_lattice_shape"
                    ]["principal_distortion_span_percent"],
                    "relative_distortion_median_percent": distortion["median"],
                    "relative_distortion_p90_percent": distortion["p90"],
                    "fraction_above_reference_p90": summary[
                        "comparison_to_reference_variation"
                    ]["moving_fraction_above_reference_p90"],
                    "reliable_pixel_fraction": summary[
                        "reliable_pixel_fraction_native"
                    ],
                }
            )

    reference_maps_path = output / "reference_centered_gpa_maps.npz"
    np.savez_compressed(
        reference_maps_path,
        **baseline_maps,
        reliability=baseline_mask.astype(np.float32),
    )

    rows = len(comparisons) + 1
    fig, axes = plt.subplots(rows, 4, figsize=(15, 4.1 * rows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    reference_magnitude = 100.0 * baseline_maps["relative_distortion_magnitude"]
    all_magnitudes = [reference_magnitude[baseline_mask]] + [
        100.0
        * item.maps_aligned["relative_distortion_magnitude"][
            item.aligned_reliability
        ]
        for item in comparisons
    ]
    magnitude_limit = float(np.percentile(np.concatenate(all_magnitudes), 95))
    signed_values = []
    for name in ("relative_principal_tensile", "relative_principal_compressive"):
        signed_values.append(100.0 * baseline_maps[name][baseline_mask])
        signed_values.extend(
            100.0 * item.maps_aligned[name][item.aligned_reliability]
            for item in comparisons
        )
    signed_limit = float(
        np.percentile(np.abs(np.concatenate(signed_values)), 95)
    )

    row_data = [
        (
            reference_label,
            reference.normalized_image,
            baseline_maps,
            baseline_mask,
            "reference texture/noise floor",
        )
    ] + [
        (
            item.label,
            item.aligned_image,
            item.maps_aligned,
            item.aligned_reliability,
            (
                "global principal: "
                f"{item.summary['global_scale_independent_lattice_shape']['principal_log_strain_percent'][0]:+.2f}% / "
                f"{item.summary['global_scale_independent_lattice_shape']['principal_log_strain_percent'][1]:+.2f}%"
            ),
        )
        for item in comparisons
    ]
    for row, (label, image, maps, mask, subtitle) in enumerate(row_data):
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title(f"{label}: Y-axis aligned\n{subtitle}")
        magnitude = 100.0 * maps["relative_distortion_magnitude"]
        tensile = 100.0 * maps["relative_principal_tensile"]
        compressive = 100.0 * maps["relative_principal_compressive"]
        entries = (
            (magnitude, "Relative distortion (%)", "magma", 0, magnitude_limit),
            (tensile, "Principal tensile (%)", "coolwarm", -signed_limit, signed_limit),
            (compressive, "Principal compressive (%)", "coolwarm", -signed_limit, signed_limit),
        )
        for column, (values, title, cmap, lower, upper) in enumerate(entries, start=1):
            shown = axes[row, column].imshow(
                _masked(values, mask), cmap=cmap, vmin=lower, vmax=upper
            )
            axes[row, column].set_title(title)
            fig.colorbar(shown, ax=axes[row, column], fraction=0.046)
        for axis in axes[row]:
            axis.set_axis_off()
    comparison_path = output / "reference_deformation_comparison.png"
    fig.savefig(comparison_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    notes_path = output / "RESULTS_ZH.md"
    lines = [
        f"# 以 {reference_label} 为模板的晶格相对变形结果",
        "",
        "本结果先匹配晶格基矢，并把不同放大倍数和刚体旋转作为配准量移除；随后将各图的局部 GPA 应变旋转到参考坐标，并叠加未被相似变换解释的晶格形状差异。",
        "",
        "可以解读：无量纲的相对拉伸/压缩、剪切、主应变和热点排序。",
        "",
        "不能解读：没有共同地标时的绝对平移；没有像素物理标定时的绝对等向晶格膨胀；没有弹性常数时的 Pa/GPa 应力。",
        "",
        f"数值总表：`{csv_path.name}`；完整机器可读记录：`{summary_path.name}`。",
        "",
    ]
    with notes_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))

    return {
        "summary_json": str(summary_path.resolve()),
        "summary_csv": str(csv_path.resolve()),
        "comparison_figure": str(comparison_path.resolve()),
        "results_notes": str(notes_path.resolve()),
        "reference_maps": str(reference_maps_path.resolve()),
    }


def save_candidate_hotspot_overlap(
    comparisons: Sequence[ReferenceDeformationResult],
    candidate_points_by_label: Mapping[str, np.ndarray],
    *,
    output_directory: str | Path,
    native_images_by_label: Mapping[str, np.ndarray] | None = None,
    candidate_ids_by_label: Mapping[str, Sequence[str]] | None = None,
    control_points_by_label: Mapping[str, np.ndarray] | None = None,
    control_ids_by_label: Mapping[str, Sequence[str]] | None = None,
    hotspot_quantile: float = 0.90,
    support_radius_pixels: float = 3.0,
    output_stem: str = "candidate_gpa_hotspot_overlap",
    dpi: int = 170,
) -> Dict[str, str]:
    """Save candidate-versus-GPA hotspot evidence in native image coordinates.

    The destination directory may already contain the ordinary reference GPA
    bundle, but none of the three files owned by this function may exist.
    """

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if not output_stem or Path(output_stem).name != output_stem:
        raise ValueError("output_stem must be one plain file stem")
    json_path = output / f"{output_stem}.json"
    csv_path = output / f"{output_stem}.csv"
    control_csv_path = output / f"{output_stem}_site_I_controls.csv"
    figure_path = output / f"{output_stem}.png"
    owned_paths = [json_path, csv_path, figure_path]
    if control_points_by_label is not None:
        owned_paths.append(control_csv_path)
    for path in owned_paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite GPA overlap output: {path}")

    audits: list[Dict[str, Any]] = []
    comparison_by_label = {item.label: item for item in comparisons}
    unknown = set(candidate_points_by_label) - set(comparison_by_label)
    if unknown:
        raise ValueError(f"Candidate labels have no GPA comparison: {sorted(unknown)}")
    for comparison in comparisons:
        points = np.asarray(
            candidate_points_by_label.get(comparison.label, np.empty((0, 2))),
            dtype=float,
        )
        ids = None
        if candidate_ids_by_label is not None:
            ids = candidate_ids_by_label.get(comparison.label)
        audit = summarize_candidate_hotspot_overlap(
            comparison,
            points,
            candidate_ids=ids,
            hotspot_quantile=hotspot_quantile,
            support_radius_pixels=support_radius_pixels,
        )
        if control_points_by_label is not None:
            controls = np.asarray(
                control_points_by_label.get(
                    comparison.label, np.empty((0, 2))
                ),
                dtype=float,
            )
            control_ids = None
            if control_ids_by_label is not None:
                control_ids = control_ids_by_label.get(comparison.label)
            control_audit = summarize_candidate_hotspot_overlap(
                comparison,
                controls,
                candidate_ids=control_ids,
                hotspot_quantile=hotspot_quantile,
                support_radius_pixels=support_radius_pixels,
            )
            population = int(control_audit["candidate_count"])
            population_dual = int(
                control_audit["candidate_count_with_dual_hotspot_support"]
            )
            sample = int(audit["candidate_count"])
            sample_dual = int(
                audit["candidate_count_with_dual_hotspot_support"]
            )
            control_fraction = (
                population_dual / population if population else float("nan")
            )
            candidate_fraction = (
                sample_dual / sample if sample else float("nan")
            )
            enrichment = (
                candidate_fraction / control_fraction
                if np.isfinite(candidate_fraction)
                and np.isfinite(control_fraction)
                and control_fraction > 0
                else float("nan")
            )
            p_value = (
                float(
                    hypergeom.sf(
                        sample_dual - 1,
                        population,
                        population_dual,
                        sample,
                    )
                )
                if 0 < sample <= population
                else float("nan")
            )
            audit["site_I_control_baseline"] = {
                "control_population_definition": (
                    "all model-valid template-empty site-I probes in the "
                    "reliable image interior, including candidates"
                ),
                "control_site_count": population,
                "control_count_with_distortion_hotspot_support": int(
                    control_audit[
                        "candidate_count_with_distortion_hotspot_support"
                    ]
                ),
                "control_count_with_stress_hotspot_support": int(
                    control_audit[
                        "candidate_count_with_stress_hotspot_support"
                    ]
                ),
                "control_count_with_dual_hotspot_support": population_dual,
                "control_dual_hotspot_fraction": _finite_or_none(
                    control_fraction
                ),
                "candidate_dual_hotspot_fraction": _finite_or_none(
                    candidate_fraction
                ),
                "candidate_to_control_dual_hotspot_enrichment_ratio": (
                    _finite_or_none(enrichment)
                ),
                "one_sided_hypergeometric_enrichment_p_value": (
                    _finite_or_none(p_value)
                ),
                "statistical_interpretation": (
                    "descriptive only; very small candidate counts and "
                    "periodic spatial correlation limit inference"
                ),
                "control_records": control_audit["candidates"],
            }
        audits.append(audit)

    payload = {
        "schema_version": "candidate-gpa-hotspot-overlap-v1",
        "method": (
            "Structure-first FAU site-I candidates are fixed before GPA. "
            "Their native pixel coordinates are audited against the top GPA "
            f"{100.0 * (1.0 - hotspot_quantile):.1f}% distortion and maximum-"
            "shear-strain proxy regions."
        ),
        "candidate_selection_independent_of_gpa": True,
        "gpa_role": "post-hoc spatial support only; never a candidate gate",
        "control_role": (
            "equivalent site-I controls quantify how often periodic GPA maps "
            "overlap ordinary template-empty positions"
            if control_points_by_label is not None
            else "not supplied"
        ),
        "stress_interpretation": (
            "dimensionless maximum-shear-strain hotspot proxy, not Pa/GPa"
        ),
        "coordinate_contract": (
            "candidate coordinates and maps_native use the same moving-image "
            "pixel frame; no cross-field pointwise subtraction is performed"
        ),
        "comparisons": audits,
    }
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    fieldnames = (
        "moving_label",
        "candidate_id",
        "x_pixels",
        "y_pixels",
        "inside_image",
        "center_reliable",
        "local_reliable_fraction",
        "distortion_center_percent",
        "distortion_local_p90_percent",
        "distortion_local_p90_percentile_rank",
        "nearest_distortion_top_decile_pixels",
        "within_distortion_hotspot_support_radius",
        "stress_proxy_center_percent",
        "stress_proxy_local_p90_percent",
        "stress_proxy_local_p90_percentile_rank",
        "nearest_stress_top_decile_pixels",
        "within_stress_hotspot_support_radius",
        "dual_hotspot_support",
    )
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            for record in audit["candidates"]:
                writer.writerow({"moving_label": audit["moving_label"], **record})

    if control_points_by_label is not None:
        with control_csv_path.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            for audit in audits:
                baseline = audit["site_I_control_baseline"]
                for record in baseline["control_records"]:
                    writer.writerow(
                        {"moving_label": audit["moving_label"], **record}
                    )

    rows = max(len(comparisons), 1)
    fig, axes = plt.subplots(
        rows,
        3,
        figsize=(14.2, 4.5 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    for row, (comparison, audit) in enumerate(zip(comparisons, audits)):
        label = comparison.label
        maps = comparison.maps_native
        reliable = np.asarray(maps["reliability"]) >= 0.5
        distortion = 100.0 * np.asarray(
            maps["relative_distortion_magnitude"], dtype=float
        )
        stress_proxy = 100.0 * np.asarray(
            maps["stress_hotspot_proxy"], dtype=float
        )
        if native_images_by_label is None or label not in native_images_by_label:
            texture = np.zeros_like(distortion)
        else:
            texture = np.asarray(native_images_by_label[label], dtype=float)
            if texture.shape != distortion.shape:
                raise ValueError(
                    f"Native image shape for {label} does not match GPA maps"
                )
        axes[row, 0].imshow(texture, cmap="gray")
        entries = (
            (
                axes[row, 1],
                distortion,
                audit["distortion_hotspot_threshold_percent"],
                "Relative distortion (%)",
                "magma",
            ),
            (
                axes[row, 2],
                stress_proxy,
                audit["stress_proxy_hotspot_threshold_percent"],
                "Maximum-shear proxy (%)",
                "inferno",
            ),
        )
        axes[row, 0].set_title(f"{label}: representative GPA plane")
        for axis, values, threshold, title, cmap in entries:
            upper = float(np.percentile(values[reliable], 97.5))
            shown = axis.imshow(
                _masked(values, reliable), cmap=cmap, vmin=0, vmax=max(upper, threshold)
            )
            axis.contour(
                reliable & (values >= threshold),
                levels=[0.5],
                colors=["#00FFFF"],
                linewidths=0.45,
            )
            axis.set_title(
                f"{label}: {title}\ncyan outline = top "
                f"{100.0 * (1.0 - hotspot_quantile):.0f}%"
            )
            fig.colorbar(shown, ax=axis, fraction=0.046)

        for record in audit["candidates"]:
            x = record["x_pixels"]
            y = record["y_pixels"]
            if record["dual_hotspot_support"]:
                color, marker = "#39FF14", "o"
            elif (
                record["within_distortion_hotspot_support_radius"]
                or record["within_stress_hotspot_support_radius"]
            ):
                color, marker = "#FFD400", "o"
            else:
                color, marker = "#00E5FF", "o"
            if not record["inside_image"]:
                color, marker = "#FF3B30", "x"
            for axis in axes[row]:
                axis.scatter(
                    [x],
                    [y],
                    s=105,
                    marker=marker,
                    facecolors="none" if marker == "o" else color,
                    edgecolors=color if marker == "o" else None,
                    c=None if marker == "o" else color,
                    linewidths=1.5,
                    zorder=6,
                )
                axis.text(
                    x + 4,
                    y - 4,
                    str(record["candidate_id"]),
                    color=color,
                    fontsize=6,
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1},
                    zorder=7,
                )
        baseline = audit.get("site_I_control_baseline")
        if baseline is not None:
            for axis in axes[row]:
                axis.scatter(
                    [item["x_pixels"] for item in baseline["control_records"]],
                    [item["y_pixels"] for item in baseline["control_records"]],
                    s=7,
                    marker=".",
                    c="#FFFFFF",
                    alpha=0.28,
                    linewidths=0,
                    zorder=4,
                )
            p_value = baseline[
                "one_sided_hypergeometric_enrichment_p_value"
            ]
            p_text = f"{p_value:.3f}" if p_value is not None else "n/a"
            axes[row, 0].set_title(
                f"{label}: representative GPA plane\n"
                f"candidate dual {audit['candidate_count_with_dual_hotspot_support']}/"
                f"{audit['candidate_count']}; all site-I dual "
                f"{baseline['control_count_with_dual_hotspot_support']}/"
                f"{baseline['control_site_count']}; enrichment p={p_text}"
            )
        for axis in axes[row]:
            axis.set_axis_off()
    if not comparisons:
        for axis in axes[0]:
            axis.text(0.5, 0.5, "No moving comparison", ha="center", va="center")
            axis.set_axis_off()
    fig.suptitle(
        "Structure-first site-I candidates vs independent GPA hotspots\n"
        "green = distortion + shear support; yellow = one; cyan = neither; "
        "GPA does not change candidate identity",
        fontsize=13,
    )
    fig.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    outputs = {
        "candidate_hotspot_summary_json": str(json_path.resolve()),
        "candidate_hotspot_table_csv": str(csv_path.resolve()),
        "candidate_hotspot_figure": str(figure_path.resolve()),
    }
    if control_points_by_label is not None:
        outputs["site_I_control_hotspot_table_csv"] = str(
            control_csv_path.resolve()
        )
    return outputs
