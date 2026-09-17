"""Species-resolved atomic-site inventory on a shared lattice canvas.

This module is intentionally separate from atomic *defect* detection. It
enumerates intensity maxima, assigns the ordered brightness classes supplied by
the experiment (rare-earth > Si > O), and maps all images to the reference
lattice orientation and pixel scale. Center alignment is visualization-only:
different fields of view do not acquire a fabricated absolute translation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    binary_erosion,
    gaussian_filter,
    map_coordinates,
)
from scipy.special import logsumexp
from skimage.feature import peak_local_max
from skimage.transform import AffineTransform, warp

from .config import DetectorConfig
from .features.lattice import (
    direct_basis_from_peaks,
    select_reciprocal_basis,
)
from .features.spectral import estimate_global_reciprocal_peaks
from .preprocessing import load_grayscale, robust_normalize
from .reference_deformation import match_lattice_bases


SPECIES_BRIGHTNESS_ORDER = ("O", "Si", "rare_earth")
SPECIES_COLORS = {
    "O": "#00D7FF",
    "Si": "#FFB000",
    "rare_earth": "#FF3CAC",
}


@dataclass(frozen=True)
class AtomicSiteInventoryConfig:
    """Numerical policy for aligned all-site detection and intensity typing."""

    peak_sigma: float = 0.75
    local_background_sigma: float = 3.0
    peak_min_distance_pixels: int = 2
    detection_noise_z: float = 0.50
    coverage_erosion_pixels: int = 4
    subpixel_window_radius: int = 2
    alignment_margin_pixels: int = 12
    minimum_sites_per_image: int = 60
    mixture_components: int = 3
    mixture_max_iterations: int = 300
    mixture_tolerance: float = 1e-7
    mixture_variance_floor_fraction: float = 0.002

    def validate(self) -> None:
        if self.peak_sigma <= 0:
            raise ValueError("peak_sigma must be positive")
        if self.local_background_sigma <= self.peak_sigma:
            raise ValueError(
                "local_background_sigma must exceed peak_sigma"
            )
        if self.peak_min_distance_pixels < 1:
            raise ValueError("peak_min_distance_pixels must be positive")
        if self.detection_noise_z < 0:
            raise ValueError("detection_noise_z cannot be negative")
        if self.coverage_erosion_pixels < 0:
            raise ValueError("coverage_erosion_pixels cannot be negative")
        if self.subpixel_window_radius < 1:
            raise ValueError("subpixel_window_radius must be positive")
        if self.alignment_margin_pixels < 0:
            raise ValueError("alignment_margin_pixels cannot be negative")
        if self.minimum_sites_per_image < 12:
            raise ValueError("minimum_sites_per_image must be at least 12")
        if self.mixture_components != len(SPECIES_BRIGHTNESS_ORDER):
            raise ValueError(
                "mixture_components must match the three declared species"
            )
        if self.mixture_max_iterations < 10:
            raise ValueError("mixture_max_iterations must be at least 10")
        if self.mixture_tolerance <= 0:
            raise ValueError("mixture_tolerance must be positive")
        if self.mixture_variance_floor_fraction <= 0:
            raise ValueError(
                "mixture_variance_floor_fraction must be positive"
            )


@dataclass
class OrderedIntensityMixture:
    """Ordered one-dimensional Gaussian mixture and site assignments."""

    means: np.ndarray
    standard_deviations: np.ndarray
    weights: np.ndarray
    responsibilities: np.ndarray
    labels: np.ndarray
    separation_sigma: np.ndarray
    log_likelihood: float


@dataclass
class AtomicSiteImageResult:
    """One image, aligned to the common lattice canvas, with all sites."""

    label: str
    source_path: Path
    source_sha256: str
    normalized_native_image: np.ndarray
    aligned_image: np.ndarray
    aligned_coverage: np.ndarray
    native_direct_basis: np.ndarray
    transform_native_to_aligned: AffineTransform
    points_aligned_xy: np.ndarray
    points_native_xy: np.ndarray
    species: np.ndarray
    local_contrast: np.ndarray
    normalized_peak_intensity: np.ndarray
    assignment_probability: np.ndarray
    mixture: OrderedIntensityMixture
    alignment: Dict[str, Any]


@dataclass
class AtomicSiteInventoryResult:
    """Images expressed in one reference-lattice coordinate system."""

    reference_label: str
    common_canvas_shape: tuple[int, int]
    images: list[AtomicSiteImageResult]
    config: AtomicSiteInventoryConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_log_density(
    values: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    return (
        np.log(weights + 1e-15)[None, :]
        - 0.5 * np.log(2.0 * np.pi * variances)[None, :]
        - 0.5
        * (values[:, None] - means[None, :]) ** 2
        / variances[None, :]
    )


def fit_ordered_intensity_mixture(
    values: np.ndarray,
    config: AtomicSiteInventoryConfig | None = None,
) -> OrderedIntensityMixture:
    """Fit three intensity populations and order them O, Si, rare-earth."""

    policy = config or AtomicSiteInventoryConfig()
    policy.validate()
    samples = np.asarray(values, dtype=float).reshape(-1)
    if len(samples) < policy.minimum_sites_per_image:
        raise ValueError(
            "Too few detected sites for a three-component intensity mixture"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError("Site intensities must all be finite")
    total_variance = max(float(np.var(samples)), 1e-8)
    variance_floor = max(
        policy.mixture_variance_floor_fraction * total_variance,
        1e-8,
    )
    starts = (
        (5.0, 45.0, 88.0),
        (10.0, 55.0, 92.0),
        (20.0, 60.0, 95.0),
        (25.0, 70.0, 97.0),
    )
    best: tuple[
        float, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ] | None = None
    for quantiles in starts:
        means = np.percentile(samples, quantiles)
        variances = np.full(
            policy.mixture_components, 0.25 * total_variance
        )
        weights = np.full(
            policy.mixture_components,
            1.0 / policy.mixture_components,
        )
        previous = -np.inf
        for _ in range(policy.mixture_max_iterations):
            log_density = _normal_log_density(
                samples, means, variances, weights
            )
            normalizer = logsumexp(log_density, axis=1)
            log_likelihood = float(np.sum(normalizer))
            responsibilities = np.exp(
                log_density - normalizer[:, None]
            )
            effective_count = (
                np.sum(responsibilities, axis=0) + 1e-12
            )
            weights = effective_count / len(samples)
            means = (
                np.sum(
                    responsibilities * samples[:, None], axis=0
                )
                / effective_count
            )
            variances = np.maximum(
                np.sum(
                    responsibilities
                    * (samples[:, None] - means[None, :]) ** 2,
                    axis=0,
                )
                / effective_count,
                variance_floor,
            )
            if abs(log_likelihood - previous) <= (
                policy.mixture_tolerance
                * max(1.0, abs(log_likelihood))
            ):
                break
            previous = log_likelihood
        candidate = (
            log_likelihood,
            weights.copy(),
            means.copy(),
            variances.copy(),
            responsibilities.copy(),
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("Intensity mixture did not produce a candidate")
    log_likelihood, weights, means, variances, responsibilities = best
    order = np.argsort(means)
    means = means[order]
    variances = variances[order]
    weights = weights[order]
    responsibilities = responsibilities[:, order]
    labels = np.argmax(responsibilities, axis=1)
    pooled = np.sqrt(
        0.5 * (variances[:-1] + variances[1:])
    )
    separation = np.diff(means) / np.maximum(pooled, 1e-12)
    return OrderedIntensityMixture(
        means=means,
        standard_deviations=np.sqrt(variances),
        weights=weights,
        responsibilities=responsibilities,
        labels=labels,
        separation_sigma=separation,
        log_likelihood=log_likelihood,
    )


def _sample(values: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    return map_coordinates(
        values,
        [points_xy[:, 1], points_xy[:, 0]],
        order=1,
        mode="nearest",
    )


def _refine_subpixel(
    response: np.ndarray,
    coordinates_yx: np.ndarray,
    threshold: float,
    radius: int,
) -> np.ndarray:
    height, width = response.shape
    refined = np.empty((len(coordinates_yx), 2), dtype=float)
    for index, (row, column) in enumerate(coordinates_yx):
        row_min = max(0, int(row) - radius)
        row_max = min(height, int(row) + radius + 1)
        column_min = max(0, int(column) - radius)
        column_max = min(width, int(column) + radius + 1)
        patch = response[row_min:row_max, column_min:column_max]
        weights = np.clip(patch - threshold, 0.0, None)
        if float(np.sum(weights)) <= 1e-12:
            refined[index] = (float(column), float(row))
            continue
        rows, columns = np.indices(patch.shape, dtype=float)
        total = float(np.sum(weights))
        refined[index, 0] = (
            float(np.sum(weights * columns)) / total + column_min
        )
        refined[index, 1] = (
            float(np.sum(weights * rows)) / total + row_min
        )
    return refined


def detect_species_sites(
    aligned_image: np.ndarray,
    coverage: np.ndarray,
    config: AtomicSiteInventoryConfig | None = None,
) -> Dict[str, Any]:
    """Detect all covered local maxima and assign ordered species classes."""

    policy = config or AtomicSiteInventoryConfig()
    policy.validate()
    image = np.asarray(aligned_image, dtype=float)
    valid = np.asarray(coverage, dtype=bool)
    if image.ndim != 2 or valid.shape != image.shape:
        raise ValueError(
            "Aligned image and coverage must be matching 2-D arrays"
        )
    if policy.coverage_erosion_pixels:
        valid = binary_erosion(
            valid,
            iterations=policy.coverage_erosion_pixels,
            border_value=0,
        )
    if not np.any(valid):
        raise ValueError("Aligned image has no valid interior coverage")
    smoothed = gaussian_filter(
        image, policy.peak_sigma, mode="reflect"
    )
    response = smoothed - gaussian_filter(
        image, policy.local_background_sigma, mode="reflect"
    )
    response_values = response[valid]
    center = float(np.median(response_values))
    scale = float(
        1.4826 * np.median(np.abs(response_values - center))
    )
    if scale <= 1e-8:
        scale = max(float(np.std(response_values)), 1e-8)
    threshold = center + policy.detection_noise_z * scale
    coordinates_yx = peak_local_max(
        response,
        min_distance=policy.peak_min_distance_pixels,
        threshold_abs=threshold,
        exclude_border=False,
        labels=valid.astype(np.uint8),
    )
    if len(coordinates_yx) < policy.minimum_sites_per_image:
        raise ValueError(
            "Too few atomic peaks were detected on the aligned image"
        )
    points_xy = _refine_subpixel(
        response,
        coordinates_yx,
        threshold,
        policy.subpixel_window_radius,
    )
    local_contrast = _sample(response, points_xy)
    normalized_intensity = _sample(smoothed, points_xy)
    mixture = fit_ordered_intensity_mixture(local_contrast, policy)
    species = np.asarray(
        [
            SPECIES_BRIGHTNESS_ORDER[int(index)]
            for index in mixture.labels
        ],
        dtype=object,
    )
    return {
        "points_xy": points_xy,
        "species": species,
        "local_contrast": local_contrast,
        "normalized_intensity": normalized_intensity,
        "assignment_probability": np.max(
            mixture.responsibilities, axis=1
        ),
        "mixture": mixture,
        "response_center": center,
        "response_scale": scale,
        "detection_threshold": threshold,
        "valid_coverage": valid,
    }


def _estimate_direct_basis(
    image: np.ndarray, detector_config: DetectorConfig
) -> np.ndarray:
    reciprocal = estimate_global_reciprocal_peaks(
        image, detector_config.spectral
    )
    return direct_basis_from_peaks(
        select_reciprocal_basis(reciprocal)
    )


def _transformed_half_extent(
    shape: tuple[int, int], linear: np.ndarray
) -> np.ndarray:
    center = np.asarray(
        [(shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0]
    )
    corners = np.asarray(
        [
            [0.0, 0.0],
            [shape[1] - 1.0, 0.0],
            [0.0, shape[0] - 1.0],
            [shape[1] - 1.0, shape[0] - 1.0],
        ]
    )
    transformed = (linear @ (corners - center).T).T
    return np.max(np.abs(transformed), axis=0)


def centered_similarity_transform(
    linear: np.ndarray,
    native_shape: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> AffineTransform:
    """Place a similarity-aligned image at the common canvas center."""

    native_center = np.asarray(
        [(native_shape[1] - 1) / 2.0, (native_shape[0] - 1) / 2.0]
    )
    canvas_center = np.asarray(
        [(canvas_shape[1] - 1) / 2.0, (canvas_shape[0] - 1) / 2.0]
    )
    matrix = np.eye(3, dtype=float)
    matrix[:2, :2] = linear
    matrix[:2, 2] = canvas_center - linear @ native_center
    return AffineTransform(matrix=matrix)


def _warp(
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


def analyze_atomic_site_inventory(
    reference_path: str | Path,
    moving_paths: Sequence[str | Path],
    detector_config: DetectorConfig,
    inventory_config: AtomicSiteInventoryConfig | None = None,
) -> AtomicSiteInventoryResult:
    """Align images to the reference lattice, then detect and type all sites."""

    policy = inventory_config or AtomicSiteInventoryConfig()
    policy.validate()
    sources = [Path(reference_path), *(Path(item) for item in moving_paths)]
    labels = [source.stem for source in sources]
    if len(set(labels)) != len(labels):
        raise ValueError("Input image stems must be unique labels")
    native_images: list[np.ndarray] = []
    bases: list[np.ndarray] = []
    hashes: list[str] = []
    for source in sources:
        raw = load_grayscale(source)
        normalized, _ = robust_normalize(
            raw, detector_config.preprocess
        )
        native_images.append(normalized)
        bases.append(_estimate_direct_basis(normalized, detector_config))
        hashes.append(_sha256_file(source.resolve()))

    reference_basis = bases[0]
    linear_mappings = [np.eye(2, dtype=float)]
    alignments: list[Dict[str, Any]] = [
        {
            "role": "reference",
            "scale_to_reference": 1.0,
            "rotation_degrees_modulo_lattice_symmetry": 0.0,
            "basis_fit_relative_residual": 0.0,
            "basis_sign_permutation": np.eye(2).tolist(),
            "absolute_translation_status": (
                "not inferred; canvas centers are aligned for visualization"
            ),
        }
    ]
    for basis in bases[1:]:
        match = match_lattice_bases(reference_basis, basis)
        linear_mappings.append(
            np.asarray(match["similarity_mapping"], dtype=float)
        )
        alignments.append(
            {
                "role": "moving",
                "scale_to_reference": float(match["scale"]),
                "rotation_degrees_modulo_lattice_symmetry": float(
                    match["rotation_degrees"]
                ),
                "basis_fit_relative_residual": float(
                    match["relative_residual"]
                ),
                "basis_sign_permutation": np.asarray(
                    match["basis_transform"]
                ).tolist(),
                "absolute_translation_status": (
                    "not inferred; canvas centers are aligned for visualization"
                ),
            }
        )
    half_extents = np.stack(
        [
            _transformed_half_extent(image.shape, linear)
            for image, linear in zip(native_images, linear_mappings)
        ]
    )
    maximum = np.max(half_extents, axis=0)
    canvas_width = int(
        np.ceil(2.0 * maximum[0] + 1.0)
        + 2 * policy.alignment_margin_pixels
    )
    canvas_height = int(
        np.ceil(2.0 * maximum[1] + 1.0)
        + 2 * policy.alignment_margin_pixels
    )
    canvas_shape = (canvas_height, canvas_width)

    results: list[AtomicSiteImageResult] = []
    for (
        label,
        source,
        source_hash,
        native,
        basis,
        linear,
        alignment,
    ) in zip(
        labels,
        sources,
        hashes,
        native_images,
        bases,
        linear_mappings,
        alignments,
    ):
        transform = centered_similarity_transform(
            linear, native.shape, canvas_shape
        )
        aligned = _warp(
            native,
            transform,
            canvas_shape,
            order=3,
            cval=float(np.median(native)),
        ).astype(np.float32)
        coverage = (
            _warp(
                np.ones(native.shape, dtype=float),
                transform,
                canvas_shape,
                order=0,
                cval=0.0,
            )
            >= 0.5
        )
        detected = detect_species_sites(aligned, coverage, policy)
        points_aligned = detected["points_xy"]
        points_native = np.asarray(
            transform.inverse(points_aligned), dtype=float
        )
        alignment[
            "center_alignment_translation_xy_pixels_visualization_only"
        ] = transform.params[:2, 2].tolist()
        alignment["native_direct_basis_vectors_xy"] = basis.T.tolist()
        results.append(
            AtomicSiteImageResult(
                label=label,
                source_path=source.resolve(),
                source_sha256=source_hash,
                normalized_native_image=native,
                aligned_image=aligned,
                aligned_coverage=coverage,
                native_direct_basis=basis,
                transform_native_to_aligned=transform,
                points_aligned_xy=points_aligned,
                points_native_xy=points_native,
                species=detected["species"],
                local_contrast=detected["local_contrast"],
                normalized_peak_intensity=detected[
                    "normalized_intensity"
                ],
                assignment_probability=detected[
                    "assignment_probability"
                ],
                mixture=detected["mixture"],
                alignment={
                    **alignment,
                    "response_center": detected["response_center"],
                    "response_robust_scale": detected["response_scale"],
                    "detection_threshold": detected[
                        "detection_threshold"
                    ],
                    "covered_canvas_fraction": float(
                        np.mean(coverage)
                    ),
                },
            )
        )
    return AtomicSiteInventoryResult(
        reference_label=labels[0],
        common_canvas_shape=canvas_shape,
        images=results,
        config=policy,
    )


def _mixture_summary(
    result: AtomicSiteImageResult,
) -> Dict[str, Any]:
    counts = {
        species: int(np.sum(result.species == species))
        for species in SPECIES_BRIGHTNESS_ORDER
    }
    low_probability = result.assignment_probability < 0.80
    return {
        "site_count": int(len(result.points_aligned_xy)),
        "counts_by_species_brightness_class": counts,
        "mixture": {
            "species_order_dim_to_bright": list(
                SPECIES_BRIGHTNESS_ORDER
            ),
            "local_contrast_means": result.mixture.means.tolist(),
            "local_contrast_standard_deviations": (
                result.mixture.standard_deviations.tolist()
            ),
            "component_weights": result.mixture.weights.tolist(),
            "adjacent_separation_sigma": (
                result.mixture.separation_sigma.tolist()
            ),
            "minimum_adjacent_separation_sigma": float(
                np.min(result.mixture.separation_sigma)
            ),
            "assignment_probability_median": float(
                np.median(result.assignment_probability)
            ),
            "fraction_assignment_probability_below_0_80": float(
                np.mean(low_probability)
            ),
        },
    }


def _scatter_sites(
    axis: Any,
    points: np.ndarray,
    species: np.ndarray,
    *,
    legend: bool,
) -> None:
    sizes = {"O": 5.0, "Si": 8.0, "rare_earth": 12.0}
    for name in SPECIES_BRIGHTNESS_ORDER:
        selected = species == name
        axis.scatter(
            points[selected, 0],
            points[selected, 1],
            s=sizes[name],
            c=SPECIES_COLORS[name],
            marker="o",
            linewidths=0,
            alpha=0.85,
            label=name,
        )
    if legend:
        axis.legend(
            loc="upper right",
            framealpha=0.75,
            fontsize=8,
            markerscale=1.8,
        )


def _save_figures(
    inventory: AtomicSiteInventoryResult,
    output: Path,
    dpi: int,
) -> Dict[str, str]:
    rows = len(inventory.images)
    overview, axes = plt.subplots(
        rows,
        2,
        figsize=(12, 5.2 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, result in enumerate(inventory.images):
        native_axis, aligned_axis = axes[row]
        native_axis.imshow(
            result.normalized_native_image, cmap="gray", vmin=0, vmax=1
        )
        _scatter_sites(
            native_axis,
            result.points_native_xy,
            result.species,
            legend=row == 0,
        )
        native_axis.set_title(
            f"{result.label}: native coordinates "
            f"({len(result.species)} sites)"
        )
        aligned_axis.imshow(
            result.aligned_image, cmap="gray", vmin=0, vmax=1
        )
        _scatter_sites(
            aligned_axis,
            result.points_aligned_xy,
            result.species,
            legend=False,
        )
        aligned_axis.set_title(
            f"{result.label}: aligned to "
            f"{inventory.reference_label} lattice axes"
        )
        native_axis.set_axis_off()
        aligned_axis.set_axis_off()
    overview_path = output / "atomic_sites_native_and_aligned.png"
    overview.savefig(overview_path, dpi=dpi, bbox_inches="tight")
    plt.close(overview)

    panels, axes = plt.subplots(
        rows,
        len(SPECIES_BRIGHTNESS_ORDER),
        figsize=(14, 4.6 * rows),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, result in enumerate(inventory.images):
        for column, species in enumerate(SPECIES_BRIGHTNESS_ORDER):
            axis = axes[row, column]
            axis.imshow(
                result.aligned_image, cmap="gray", vmin=0, vmax=1
            )
            selected = result.species == species
            axis.scatter(
                result.points_aligned_xy[selected, 0],
                result.points_aligned_xy[selected, 1],
                s={"O": 6, "Si": 10, "rare_earth": 15}[species],
                c=SPECIES_COLORS[species],
                linewidths=0,
                alpha=0.9,
            )
            axis.set_title(
                f"{result.label}: {species} "
                f"({int(np.sum(selected))})"
            )
            axis.set_axis_off()
    panels_path = output / "atomic_sites_by_species_aligned.png"
    panels.savefig(panels_path, dpi=dpi, bbox_inches="tight")
    plt.close(panels)

    histograms, axes = plt.subplots(
        1,
        rows,
        figsize=(5.1 * rows, 4.2),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, result in zip(axes, inventory.images):
        axis.hist(
            result.local_contrast,
            bins=80,
            color="#707070",
            alpha=0.75,
        )
        for species, center in zip(
            SPECIES_BRIGHTNESS_ORDER, result.mixture.means
        ):
            axis.axvline(
                center,
                color=SPECIES_COLORS[species],
                linewidth=2,
                label=f"{species}: {center:.3f}",
            )
        axis.set_title(
            f"{result.label}: local-contrast mixture"
        )
        axis.set_xlabel("Local peak contrast after background removal")
        axis.set_ylabel("Detected sites")
        axis.legend(fontsize=8)
    histograms_path = output / "species_intensity_mixtures.png"
    histograms.savefig(
        histograms_path, dpi=dpi, bbox_inches="tight"
    )
    plt.close(histograms)
    return {
        "overview_figure": str(overview_path.resolve()),
        "species_panels": str(panels_path.resolve()),
        "intensity_mixtures": str(histograms_path.resolve()),
    }


def save_atomic_site_inventory(
    inventory: AtomicSiteInventoryResult,
    output_directory: str | Path,
    *,
    dpi: int = 180,
) -> Dict[str, str]:
    """Save coordinates, audit metadata, and overlays without overwriting."""

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite atomic-site inventory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "atomic_sites.csv"
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        fieldnames = (
            "image",
            "site_id",
            "species_brightness_class",
            "aligned_x_pixels",
            "aligned_y_pixels",
            "native_x_pixels",
            "native_y_pixels",
            "local_contrast",
            "normalized_peak_intensity",
            "assignment_probability",
        )
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for image in inventory.images:
            for index in range(len(image.points_aligned_xy)):
                writer.writerow(
                    {
                        "image": image.label,
                        "site_id": index + 1,
                        "species_brightness_class": (
                            image.species[index]
                        ),
                        "aligned_x_pixels": image.points_aligned_xy[
                            index, 0
                        ],
                        "aligned_y_pixels": image.points_aligned_xy[
                            index, 1
                        ],
                        "native_x_pixels": image.points_native_xy[
                            index, 0
                        ],
                        "native_y_pixels": image.points_native_xy[
                            index, 1
                        ],
                        "local_contrast": image.local_contrast[index],
                        "normalized_peak_intensity": (
                            image.normalized_peak_intensity[index]
                        ),
                        "assignment_probability": (
                            image.assignment_probability[index]
                        ),
                    }
                )
    per_image_paths: Dict[str, str] = {}
    summaries = []
    for image in inventory.images:
        image_directory = output / image.label
        image_directory.mkdir(exist_ok=False)
        npz_path = image_directory / "atomic_sites.npz"
        np.savez_compressed(
            npz_path,
            points_aligned_xy=image.points_aligned_xy,
            points_native_xy=image.points_native_xy,
            species=image.species.astype(str),
            local_contrast=image.local_contrast,
            normalized_peak_intensity=image.normalized_peak_intensity,
            assignment_probability=image.assignment_probability,
            transform_native_to_aligned=(
                image.transform_native_to_aligned.params
            ),
            aligned_coverage=image.aligned_coverage.astype(np.uint8),
        )
        aligned_path = image_directory / "aligned_image.png"
        plt.imsave(
            aligned_path,
            image.aligned_image,
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        per_image_paths[f"{image.label}_sites_npz"] = str(
            npz_path.resolve()
        )
        per_image_paths[f"{image.label}_aligned_image"] = str(
            aligned_path.resolve()
        )
        summaries.append(
            {
                "label": image.label,
                "source_path": str(image.source_path),
                "source_sha256": image.source_sha256,
                "native_shape": list(
                    image.normalized_native_image.shape
                ),
                "alignment": image.alignment,
                **_mixture_summary(image),
            }
        )
    payload = {
        "schema_version": "atomic-site-inventory-v1",
        "reference_label": inventory.reference_label,
        "common_canvas_shape": list(inventory.common_canvas_shape),
        "method": (
            "global direct-lattice similarity alignment followed by local-"
            "contrast peak detection, subpixel centroid refinement, and an "
            "ordered three-component intensity mixture"
        ),
        "species_interpretation": {
            "declared_by_user_for_these_4dstem_pngs": True,
            "brightness_order_dim_to_bright": list(
                SPECIES_BRIGHTNESS_ORDER
            ),
            "mapping": {
                "lowest_local_contrast_component": "O",
                "middle_local_contrast_component": "Si",
                "highest_local_contrast_component": "rare_earth",
            },
            "boundary": (
                "brightness-derived species assignment; no independent EDS, "
                "EELS, multislice calibration, or expert point labels supplied"
            ),
        },
        "alignment_contract": {
            "valid": [
                "common lattice orientation",
                "common Y-referenced pixel scale",
                "within-image atomic-site coordinates",
            ],
            "not_valid": [
                "absolute translation between different fields of view",
                "one-to-one atom identity across Y, RA, and H2",
            ],
        },
        "config": asdict(inventory.config),
        "images": summaries,
    }
    summary_path = output / "atomic_site_inventory_summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    figures = _save_figures(inventory, output, dpi)
    notes_path = output / "RESULTS_ZH.md"
    lines = [
        "# Y、RA、H2 原子位点检测与元素亮度分型",
        "",
        "三图先按全局晶格基矢对齐到 Y 的方向和像素尺度，再在对齐图上检测局部峰并做亚像素位置细化。",
        "",
        "按实验者确认的 4D-STEM 亮度关系，将最低、中间和最高局部对比度簇分别记为 O、Si 和稀土原子柱。",
        "",
        "由于三图视野不同，画布中心对齐只用于比较方向和尺度，不能解释成相同原子的一一对应或绝对迁移。",
        "",
    ]
    for image in inventory.images:
        summary = _mixture_summary(image)
        counts = summary["counts_by_species_brightness_class"]
        lines.append(
            f"- {image.label}：总计 {summary['site_count']}；"
            f"O {counts['O']}，Si {counts['Si']}，"
            f"稀土 {counts['rare_earth']}；最小相邻簇分离度 "
            f"{summary['mixture']['minimum_adjacent_separation_sigma']:.2f}σ。"
        )
    lines.extend(
        [
            "",
            "逐点坐标和归属概率见 atomic_sites.csv；机器可读审计信息见 atomic_site_inventory_summary.json。",
            "",
            "注意：这里的元素名来自已声明的亮度顺序，不等同于 EDS/EELS 或已标定多重散射模型给出的独立化学鉴定。",
            "",
        ]
    )
    with notes_path.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return {
        "summary_json": str(summary_path.resolve()),
        "site_csv": str(csv_path.resolve()),
        "results_notes": str(notes_path.resolve()),
        **figures,
        **per_image_paths,
    }
