"""Training-free atomic-column point-defect detection."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, maximum_filter
from scipy.spatial import cKDTree
from skimage.draw import disk
from skimage.feature import peak_local_max

from ..config import DetectorConfig
from ..context import DetectionContext
from ..features.lattice import (
    direct_basis_from_peaks,
    select_reciprocal_basis,
)
from ..features.spectral import estimate_global_reciprocal_peaks
from ..models import DetectionRegion, DetectionResult
from .base import DetectionTask
from .common import probability_from_threshold, robust_location_scale


def _sample(values: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    if not len(points_xy):
        return np.empty(0, dtype=float)
    return map_coordinates(
        values,
        [points_xy[:, 1], points_xy[:, 0]],
        order=1,
        mode="nearest",
    )


def _enumerate_sites(
    shape: tuple[int, int],
    basis_xy_columns: np.ndarray,
    fractional_offset: np.ndarray,
    node_fractional: np.ndarray,
    node_names: list[str],
    node_intensities: np.ndarray,
    border: float,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    height, width = shape
    inverse = np.linalg.inv(basis_xy_columns)
    corners = np.asarray(
        [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]],
        dtype=float,
    )
    lattice_corners = (inverse @ corners.T).T
    lower = np.floor(np.min(lattice_corners, axis=0)).astype(int) - 2
    upper = np.ceil(np.max(lattice_corners, axis=0)).astype(int) + 2
    points = []
    names: list[str] = []
    intensities = []
    for first in range(int(lower[0]), int(upper[0]) + 1):
        for second in range(int(lower[1]), int(upper[1]) + 1):
            cell = np.asarray([first, second], dtype=float)
            for fractional, name, intensity in zip(
                node_fractional, node_names, node_intensities
            ):
                point = basis_xy_columns @ (
                    cell + fractional + fractional_offset
                )
                x, y = float(point[0]), float(point[1])
                if border <= x < width - border and border <= y < height - border:
                    points.append((x, y))
                    names.append(name)
                    intensities.append(float(intensity))
    return (
        np.asarray(points, dtype=float).reshape(-1, 2),
        names,
        np.asarray(intensities, dtype=float),
    )


def _choose_polarity(image: np.ndarray, requested: str) -> tuple[str, np.ndarray]:
    smoothed = gaussian_filter(image, 1.0, mode="reflect")
    if requested == "auto":
        median = float(np.median(smoothed))
        bright_tail = float(np.percentile(smoothed, 99) - median)
        dark_tail = float(median - np.percentile(smoothed, 1))
        requested = "bright" if bright_tail >= dark_tail else "dark"
    response = smoothed if requested == "bright" else 1.0 - smoothed
    return requested, response


def _infer_auto_sublattices(
    observed_xy: np.ndarray,
    basis_xy_columns: np.ndarray,
    image_shape: tuple[int, int],
    maximum_sublattices: int,
    histogram_bins: int,
) -> np.ndarray:
    """Infer stable fractional-coordinate modes on a periodic unit torus."""

    inverse = np.linalg.inv(basis_xy_columns)
    fractional = np.mod((inverse @ observed_xy.T).T, 1.0)
    cell_area = abs(float(np.linalg.det(basis_xy_columns)))
    approximate_cells = (
        image_shape[0] * image_shape[1] / max(cell_area, 1e-8)
    )
    count = int(
        np.clip(
            round(len(observed_xy) / max(approximate_cells, 1.0)),
            1,
            maximum_sublattices,
        )
    )
    histogram, _, _ = np.histogram2d(
        fractional[:, 1],
        fractional[:, 0],
        bins=histogram_bins,
        range=[[0, 1], [0, 1]],
    )
    smoothed = gaussian_filter(histogram, 1.0, mode="wrap")
    maxima = smoothed == maximum_filter(smoothed, size=5, mode="wrap")
    candidates_yx = np.argwhere(maxima)
    strengths = smoothed[candidates_yx[:, 0], candidates_yx[:, 1]]
    selected: list[np.ndarray] = []
    for candidate_index in np.argsort(strengths)[::-1]:
        row, column = candidates_yx[candidate_index]
        phase = np.asarray(
            [
                (column + 0.5) / histogram_bins,
                (row + 0.5) / histogram_bins,
            ],
            dtype=float,
        )
        if any(
            np.linalg.norm(
                np.minimum(np.abs(phase - old), 1.0 - np.abs(phase - old))
            )
            < 0.14
            for old in selected
        ):
            continue
        selected.append(phase)
        if len(selected) >= count:
            break
    if not selected:
        selected = [np.asarray([0.0, 0.0])]
    origin = selected[0]
    return np.mod(np.stack(selected) - origin, 1.0)


def _one_to_one_match(
    expected_xy: np.ndarray,
    observed_xy: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(observed_xy)
    distances, nearest_observed = tree.query(expected_xy, k=1)
    match = np.full(len(expected_xy), -1, dtype=int)
    used_observed: set[int] = set()
    for site_index in np.argsort(distances):
        observed_index = int(nearest_observed[site_index])
        if (
            distances[site_index] <= tolerance
            and observed_index not in used_observed
        ):
            match[site_index] = observed_index
            used_observed.add(observed_index)
    return match, distances


def _nonrigid_register(
    expected_xy: np.ndarray,
    observed_xy: np.ndarray,
    image_shape: tuple[int, int],
    tolerance: float,
    smoothing_sigma: float,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    adjusted = expected_xy.copy()
    field_x = np.zeros(image_shape, dtype=np.float32)
    field_y = np.zeros(image_shape, dtype=np.float32)
    for _ in range(iterations):
        match, distances = _one_to_one_match(
            adjusted, observed_xy, tolerance
        )
        usable = match >= 0
        if int(np.sum(usable)) < 8:
            break
        displacement = observed_xy[match[usable]] - adjusted[usable]
        rows = np.clip(
            np.rint(adjusted[usable, 1]).astype(int),
            0,
            image_shape[0] - 1,
        )
        columns = np.clip(
            np.rint(adjusted[usable, 0]).astype(int),
            0,
            image_shape[1] - 1,
        )
        weights = np.zeros(image_shape, dtype=np.float32)
        numerator_x = np.zeros(image_shape, dtype=np.float32)
        numerator_y = np.zeros(image_shape, dtype=np.float32)
        np.add.at(weights, (rows, columns), 1.0)
        np.add.at(numerator_x, (rows, columns), displacement[:, 0])
        np.add.at(numerator_y, (rows, columns), displacement[:, 1])
        denominator = gaussian_filter(
            weights, smoothing_sigma, mode="nearest"
        )
        field_x = gaussian_filter(
            numerator_x, smoothing_sigma, mode="nearest"
        ) / (denominator + 1e-12)
        field_y = gaussian_filter(
            numerator_y, smoothing_sigma, mode="nearest"
        ) / (denominator + 1e-12)
        correction = np.column_stack(
            [_sample(field_x, adjusted), _sample(field_y, adjusted)]
        )
        correction = np.clip(correction, -0.5 * tolerance, 0.5 * tolerance)
        adjusted += correction
    match, distances = _one_to_one_match(
        adjusted, observed_xy, tolerance
    )
    return adjusted, match, distances, field_x, field_y


class AtomicPointDefectTask(DetectionTask):
    """Find missing, displaced, extra, and intensity-anomalous columns."""

    name = "atomic_point_defect"

    def run(
        self,
        normalized_image: np.ndarray,
        config: DetectorConfig,
        input_metadata: Dict[str, Any],
        context: DetectionContext,
    ) -> DetectionResult:
        atomic = config.atomic
        polarity, response = _choose_polarity(
            normalized_image, atomic.polarity
        )
        response = gaussian_filter(
            response, atomic.peak_sigma, mode="reflect"
        )
        threshold_abs = max(
            float(np.quantile(response, atomic.peak_threshold_quantile)),
            float(
                np.median(response)
                + 0.20
                * (np.percentile(response, 99.5) - np.median(response))
            ),
        )
        coordinates_yx = peak_local_max(
            response,
            min_distance=atomic.peak_min_distance,
            threshold_abs=threshold_abs,
            exclude_border=False,
        )
        observed_xy = coordinates_yx[:, ::-1].astype(float)
        if len(observed_xy) < atomic.minimum_reference_sites:
            raise ValueError(
                "Too few atomic-column peaks were found; adjust atomic polarity, "
                "peak scale, or peak threshold"
            )

        if context.skeleton is not None:
            basis = np.asarray(context.skeleton.basis, dtype=float).T
            node_fractional = np.asarray(
                [
                    node.fractional_coordinate
                    for node in context.skeleton.nodes
                    if not node.optional
                ],
                dtype=float,
            )
            node_names = [
                node.site_id
                for node in context.skeleton.nodes
                if not node.optional
            ]
            node_intensities = np.asarray(
                [
                    node.expected_relative_intensity
                    if node.expected_relative_intensity is not None
                    else 1.0
                    for node in context.skeleton.nodes
                    if not node.optional
                ],
                dtype=float,
            )
            basis_source = "provided_skeleton"
            reciprocal_peaks: list[tuple[float, float]] = []
        else:
            reciprocal_peaks = estimate_global_reciprocal_peaks(
                normalized_image, config.spectral
            )
            selected = select_reciprocal_basis(reciprocal_peaks)
            basis = direct_basis_from_peaks(selected)
            node_fractional = _infer_auto_sublattices(
                observed_xy,
                basis,
                normalized_image.shape,
                atomic.max_auto_sublattices,
                atomic.fractional_histogram_bins,
            )
            node_names = [
                f"auto_sublattice_{index + 1}"
                for index in range(len(node_fractional))
            ]
            node_intensities = np.ones(len(node_fractional), dtype=float)
            basis_source = "fft_and_fractional_mode_estimate"

        lengths = np.linalg.norm(basis, axis=0)
        spacing = float(np.min(lengths))
        if (
            not np.all(np.isfinite(basis))
            or abs(float(np.linalg.det(basis))) < 1e-6
            or spacing < 2
            or float(np.max(lengths)) > max(normalized_image.shape)
        ):
            raise ValueError("The inferred direct lattice basis is not usable")

        observed_tree = cKDTree(observed_xy)
        best_alignment: tuple[float, np.ndarray, list[str], np.ndarray, np.ndarray] | None = None
        steps = atomic.offset_search_steps
        border = max(spacing * 0.55, 2.0)
        for first in np.arange(steps, dtype=float) / steps:
            for second in np.arange(steps, dtype=float) / steps:
                fractional_offset = np.asarray([first, second])
                sites, names, intensities = _enumerate_sites(
                    normalized_image.shape,
                    basis,
                    fractional_offset,
                    node_fractional,
                    node_names,
                    node_intensities,
                    border,
                )
                if not len(sites):
                    continue
                values = _sample(response, sites) / np.maximum(
                    intensities, 0.15
                )
                distances, _ = observed_tree.query(sites, k=1)
                objective = float(
                    np.median(values)
                    - 0.20 * np.median(distances) / max(spacing, 1e-8)
                )
                if best_alignment is None or objective > best_alignment[0]:
                    best_alignment = (
                        objective,
                        sites,
                        names,
                        intensities,
                        fractional_offset,
                    )
        if best_alignment is None:
            raise ValueError("Unable to align the expected lattice to the image")
        _, expected_xy, expected_names, expected_intensities, offset = (
            best_alignment
        )
        if len(expected_xy) < atomic.minimum_reference_sites:
            raise ValueError(
                "The aligned lattice contains too few reference sites"
            )

        tolerance = atomic.matching_tolerance_fraction * spacing
        (
            registered_expected_xy,
            match,
            distances,
            displacement_field_x,
            displacement_field_y,
        ) = _nonrigid_register(
            expected_xy,
            observed_xy,
            normalized_image.shape,
            tolerance,
            atomic.nonrigid_smoothing_cells * spacing,
            atomic.nonrigid_iterations,
        )

        site_response = _sample(response, registered_expected_xy) / np.maximum(
            expected_intensities, 0.15
        )
        matched = match >= 0
        reference = matched & (distances <= max(0.35 * tolerance, 1.0))
        if int(np.sum(reference)) < atomic.minimum_reference_sites:
            reference = matched
        if int(np.sum(reference)) < atomic.minimum_reference_sites:
            raise ValueError(
                "Too few one-to-one lattice matches remain for robust statistics"
            )

        global_amplitude_center, global_amplitude_scale = (
            robust_location_scale(site_response[reference])
        )
        amplitude_center_map = np.full(
            len(expected_xy), global_amplitude_center, dtype=float
        )
        amplitude_scale_map = np.full(
            len(expected_xy), global_amplitude_scale, dtype=float
        )
        sublattice_statistics = {}
        expected_name_array = np.asarray(expected_names)
        for name in sorted(set(expected_names)):
            members = expected_name_array == name
            local_reference = reference & members
            if int(np.sum(local_reference)) >= 8:
                center, scale = robust_location_scale(
                    site_response[local_reference]
                )
                amplitude_center_map[members] = center
                amplitude_scale_map[members] = scale
                sublattice_statistics[name] = {
                    "site_response_median": center,
                    "site_response_robust_scale": scale,
                    "reference_count": int(np.sum(local_reference)),
                }
        distance_center, distance_scale = robust_location_scale(
            distances[reference]
        )
        registration_total = np.zeros(
            normalized_image.shape, dtype=np.float32
        )
        registration_matched = np.zeros(
            normalized_image.shape, dtype=np.float32
        )
        registration_residual = np.zeros(
            normalized_image.shape, dtype=np.float32
        )
        registered_pixels = np.rint(
            registered_expected_xy[:, ::-1]
        ).astype(int)
        registered_pixels[:, 0] = np.clip(
            registered_pixels[:, 0], 0, normalized_image.shape[0] - 1
        )
        registered_pixels[:, 1] = np.clip(
            registered_pixels[:, 1], 0, normalized_image.shape[1] - 1
        )
        np.add.at(
            registration_total,
            (registered_pixels[:, 0], registered_pixels[:, 1]),
            1.0,
        )
        np.add.at(
            registration_matched,
            (
                registered_pixels[matched, 0],
                registered_pixels[matched, 1],
            ),
            1.0,
        )
        np.add.at(
            registration_residual,
            (
                registered_pixels[matched, 0],
                registered_pixels[matched, 1],
            ),
            distances[matched],
        )
        reliability_sigma = 2.0 * spacing
        local_total = gaussian_filter(
            registration_total, reliability_sigma, mode="nearest"
        )
        local_matched = gaussian_filter(
            registration_matched, reliability_sigma, mode="nearest"
        )
        local_residual_sum = gaussian_filter(
            registration_residual, reliability_sigma, mode="nearest"
        )
        local_match_fraction = local_matched / (local_total + 1e-12)
        local_residual = local_residual_sum / (local_matched + 1e-12)
        reliability_residual_limit = max(
            distance_center + 2.0 * distance_scale,
            0.55 * atomic.minimum_displacement_fraction * spacing,
        )
        registration_reliability = (
            (local_match_fraction >= 0.82)
            & (local_residual <= reliability_residual_limit)
        )
        reliability_border = max(int(np.ceil(spacing)), 2)
        registration_reliability[:reliability_border, :] = False
        registration_reliability[-reliability_border:, :] = False
        registration_reliability[:, :reliability_border] = False
        registration_reliability[:, -reliability_border:] = False
        reliable_sites = (
            _sample(
                registration_reliability.astype(float),
                registered_expected_xy,
            )
            >= 0.5
        )
        presence_z = np.clip(
            (amplitude_center_map - site_response) / amplitude_scale_map,
            0,
            12,
        )
        presence_z[~matched] = np.maximum(
            presence_z[~matched], atomic.site_anomaly_z + 1.0
        )
        intensity_z = np.clip(
            np.abs(
                (site_response - amplitude_center_map)
                / amplitude_scale_map
            ),
            0,
            12,
        )
        intensity_z[~matched] = 0
        displacement_z = np.clip(
            (distances - distance_center) / distance_scale, 0, 12
        )
        displacement_z[~matched] = 0
        physical_displacement = (
            distances >= atomic.minimum_displacement_fraction * spacing
        )
        scaled_displacement = displacement_z * (
            atomic.site_anomaly_z / max(atomic.displacement_z, 1e-8)
        )
        site_score = np.maximum.reduce(
            [presence_z, intensity_z, scaled_displacement]
        )

        registered_tree = cKDTree(registered_expected_xy)
        observed_to_expected, _ = registered_tree.query(observed_xy, k=1)
        reliable_observed = (
            _sample(
                registration_reliability.astype(float), observed_xy
            )
            >= 0.5
        )
        extra_indices = np.flatnonzero(
            (observed_to_expected > tolerance)
            & reliable_observed
            & (response[coordinates_yx[:, 0], coordinates_yx[:, 1]]
               >= max(threshold_abs, 0.45 * global_amplitude_center))
        )

        detections: list[dict[str, Any]] = []
        candidate_sites = (
            (
                (presence_z >= atomic.site_anomaly_z)
                | (intensity_z >= atomic.site_anomaly_z)
                | (
                    (displacement_z >= atomic.displacement_z)
                    & physical_displacement
                )
            )
            & reliable_sites
        )
        for site_index in np.flatnonzero(candidate_sites):
            evidence = []
            if presence_z[site_index] >= atomic.site_anomaly_z:
                evidence.append("presence")
            if (
                displacement_z[site_index] >= atomic.displacement_z
                and physical_displacement[site_index]
            ):
                evidence.append("displacement")
            if intensity_z[site_index] >= atomic.site_anomaly_z:
                evidence.append("intensity")
            if matched[site_index] and (
                displacement_z[site_index] >= atomic.displacement_z
                and physical_displacement[site_index]
            ):
                defect_type = "displaced_atom_column"
            elif presence_z[site_index] >= atomic.site_anomaly_z:
                defect_type = "vacancy_or_low_intensity_column"
            else:
                defect_type = "intensity_or_substitution_anomaly"
            detections.append(
                {
                    "xy": registered_expected_xy[site_index].tolist(),
                    "defect_type": defect_type,
                    "site": expected_names[site_index],
                    "score": float(site_score[site_index]),
                    "presence_z": float(presence_z[site_index]),
                    "intensity_z": float(intensity_z[site_index]),
                    "displacement_z": float(displacement_z[site_index]),
                    "evidence": evidence,
                }
            )
        for observed_index in extra_indices:
            detections.append(
                {
                    "xy": observed_xy[observed_index].tolist(),
                    "defect_type": "interstitial_or_extra_column",
                    "site": None,
                    "score": float(atomic.site_anomaly_z + 1.0),
                    "presence_z": 0.0,
                    "intensity_z": 0.0,
                    "displacement_z": 0.0,
                    "evidence": ["extra_peak"],
                }
            )

        shape = normalized_image.shape
        group_presence = np.zeros(shape, dtype=np.float32)
        group_intensity = np.zeros(shape, dtype=np.float32)
        group_displacement = np.zeros(shape, dtype=np.float32)
        group_extra_peak = np.zeros(shape, dtype=np.float32)
        score_map = np.zeros(shape, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        radius = max(2, int(round(atomic.render_radius_fraction * spacing)))
        regions: list[DetectionRegion] = []
        for region_id, detection in enumerate(detections, start=1):
            x, y = detection["xy"]
            rows, columns = disk((y, x), radius, shape=shape)
            value = float(detection["score"])
            score_map[rows, columns] = np.maximum(
                score_map[rows, columns], value
            )
            mask[rows, columns] = True
            for evidence in detection["evidence"]:
                target = {
                    "presence": group_presence,
                    "intensity": group_intensity,
                    "displacement": group_displacement,
                    "extra_peak": group_extra_peak,
                }[evidence]
                target[rows, columns] = np.maximum(target[rows, columns], value)
            x1, y1 = max(0, int(x) - radius), max(0, int(y) - radius)
            x2, y2 = min(shape[1], int(x) + radius + 1), min(
                shape[0], int(y) + radius + 1
            )
            regions.append(
                DetectionRegion(
                    region_id=region_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    centroid_xy=(float(x), float(y)),
                    area_pixels=int(len(rows)),
                    mean_score=value,
                    max_score=value,
                    confidence=float(
                        1 / (1 + np.exp(-(value - atomic.site_anomaly_z)))
                    ),
                    confidence_level=(
                        "high"
                        if value >= atomic.site_anomaly_z + 2
                        else "medium"
                    ),
                    evidence_groups=list(detection["evidence"]),
                    defect_type=str(detection["defect_type"]),
                )
            )

        probability = probability_from_threshold(
            score_map, atomic.site_anomaly_z, scale=1.0
        )
        expected_map = np.zeros(shape, dtype=np.float32)
        expected_pixels = np.rint(
            registered_expected_xy[:, ::-1]
        ).astype(int)
        expected_pixels[:, 0] = np.clip(
            expected_pixels[:, 0], 0, shape[0] - 1
        )
        expected_pixels[:, 1] = np.clip(
            expected_pixels[:, 1], 0, shape[1] - 1
        )
        expected_map[
            expected_pixels[:, 0], expected_pixels[:, 1]
        ] = 1.0
        observed_map = np.zeros(shape, dtype=np.float32)
        observed_map[coordinates_yx[:, 0], coordinates_yx[:, 1]] = 1.0
        evidence_maps = {
            "group_presence": group_presence,
            "group_intensity": group_intensity,
            "group_displacement": group_displacement,
            "group_extra_peak": group_extra_peak,
            "expected_lattice_sites": expected_map,
            "observed_atomic_peaks": observed_map,
            "registration_reliability": (
                registration_reliability.astype(np.float32)
            ),
            "registration_local_match_fraction": (
                local_match_fraction.astype(np.float32)
            ),
            "registration_local_residual": (
                local_residual.astype(np.float32)
            ),
        }
        metadata = {
            **input_metadata,
            "method": "lattice_registration_and_site_residuals",
            "neural_training_used": False,
            "supervised_labels_used": False,
            "per_image_statistical_adaptation": True,
            "polarity": polarity,
            "basis_source": basis_source,
            "direct_basis_vectors_xy_columns": basis.tolist(),
            "direct_basis_lengths_pixels": lengths.tolist(),
            "fractional_alignment_offset": offset.tolist(),
            "matching_tolerance_pixels": float(tolerance),
            "minimum_displacement_pixels": float(
                atomic.minimum_displacement_fraction * spacing
            ),
            "nonrigid_registration": {
                "iterations": atomic.nonrigid_iterations,
                "smoothing_sigma_pixels": (
                    atomic.nonrigid_smoothing_cells * spacing
                ),
                "reliability_smoothing_sigma_pixels": reliability_sigma,
                "reliability_residual_limit_pixels": (
                    reliability_residual_limit
                ),
                "reliable_image_fraction": float(
                    np.mean(registration_reliability)
                ),
            },
            "expected_site_count": int(len(expected_xy)),
            "inferred_sublattice_count": int(len(node_fractional)),
            "inferred_fractional_coordinates": node_fractional.tolist(),
            "observed_peak_count": int(len(observed_xy)),
            "matched_site_count": int(np.sum(matched)),
            "detections": detections,
            "reference_statistics": {
                "site_response_median": global_amplitude_center,
                "site_response_robust_scale": global_amplitude_scale,
                "registration_distance_median_pixels": distance_center,
                "registration_distance_robust_scale_pixels": distance_scale,
                "by_sublattice": sublattice_statistics,
            },
            "context": {
                "skeleton_provided": context.skeleton is not None,
                "skeleton_node_count": (
                    len(context.skeleton.nodes) if context.skeleton else 0
                ),
            },
        }
        return DetectionResult(
            task=self.name,
            image_shape=shape,
            normalized_image=normalized_image,
            score_map=score_map,
            probability_map=probability,
            mask=mask,
            regions=regions,
            feature_maps={
                "atomic_peak_response": response.astype(np.float32),
                "nonrigid_displacement_x": displacement_field_x,
                "nonrigid_displacement_y": displacement_field_y,
            },
            evidence_maps=evidence_maps,
            metadata=metadata,
        )
