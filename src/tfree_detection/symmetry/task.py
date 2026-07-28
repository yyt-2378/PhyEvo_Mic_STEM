"""Training-free Class1 analysis from reference-anchored symmetry relations."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import convolve, gaussian_filter
from scipy.special import expit
from skimage.measure import label
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    disk,
    remove_small_objects,
    skeletonize,
)
from skimage.segmentation import find_boundaries
from skimage.transform import resize

from ..config import DetectorConfig
from ..context import DetectionContext
from ..features.lattice import (
    complex_demodulation,
    direct_basis_from_peaks,
    select_reciprocal_basis,
)
from ..features.spectral import estimate_global_reciprocal_peaks
from ..models import DetectionRegion, DetectionResult
from ..tasks.base import DetectionTask
from ..tasks.common import regions_from_mask, robust_location_scale
from .group import GroupAlignment, align_group_motif, fold_periodic_motif
from .references import PatchObservation, select_reference_patches


_RELATION_CODES = {
    "identity": 0,
    "translation": 1,
    "rotation": 2,
    "mirror": 3,
    "sublattice_exchange": 4,
}


def _refine_reciprocal_peaks(
    image: np.ndarray,
    peaks_fy_fx: tuple[tuple[float, float], tuple[float, float]],
    *,
    iterations: int = 2,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Refine FFT-bin peaks by the robust residual carrier phase gradient."""

    refined = [tuple(item) for item in peaks_fy_fx]
    for _ in range(iterations):
        qx, qy, _, coherence = complex_demodulation(
            image,
            refined,
            frequency_sigma=0.012,
            derivative_sigma=2.0,
            padding=16,
        )
        valid = coherence >= np.median(coherence)
        updated = []
        for carrier, (peak_y, peak_x) in enumerate(refined):
            correction_y = float(np.median(qy[carrier][valid])) / (
                2 * np.pi
            )
            correction_x = float(np.median(qx[carrier][valid])) / (
                2 * np.pi
            )
            updated.append(
                (peak_y + correction_y, peak_x + correction_x)
            )
        refined = updated
    return refined[0], refined[1]


def _resize_grid(
    values: np.ndarray,
    shape: tuple[int, int],
    *,
    categorical: bool = False,
) -> np.ndarray:
    return resize(
        np.asarray(values),
        shape,
        order=0 if categorical else 1,
        mode="edge",
        preserve_range=True,
        anti_aliasing=False,
    )


def _positive_z(
    values: np.ndarray,
    reference_values: np.ndarray,
    *,
    scale_floor: float,
) -> tuple[np.ndarray, dict[str, float]]:
    center, scale = robust_location_scale(reference_values)
    scale = max(float(scale), float(scale_floor))
    score = np.clip((np.asarray(values) - center) / scale, 0, 12)
    return score.astype(np.float32), {
        "median": float(center),
        "robust_scale": float(scale),
    }


def _edge_support(
    center_yx: tuple[int, int],
    radius: int,
    shape: tuple[int, int],
) -> float:
    center_y, center_x = center_yx
    height, width = shape
    y1, y2 = max(center_y - radius, 0), min(center_y + radius + 1, height)
    x1, x2 = max(center_x - radius, 0), min(center_x + radius + 1, width)
    return float(
        ((y2 - y1) * (x2 - x1)) / max((2 * radius + 1) ** 2, 1)
    )


def _geometry_regions(
    mask: np.ndarray,
    score: np.ndarray,
    probability: np.ndarray,
    evidence_maps: dict[str, np.ndarray],
    *,
    start_id: int,
    defect_type: str,
    spatial_support: str,
    geometry: str,
    evidence_names: list[str],
) -> list[DetectionRegion]:
    regions = regions_from_mask(
        mask,
        score,
        probability,
        evidence_maps,
        defect_type=defect_type,
        support_threshold=1.0,
        evidence_names=evidence_names,
    )
    for offset, region in enumerate(regions):
        region.region_id = start_id + offset
        region.spatial_support = spatial_support
        region.provenance["geometry"] = geometry
    return regions


class SymmetryRelationTask(DetectionTask):
    """Detect Class1 group domains, interfaces, phase slips, and cores."""

    name = "symmetry_relation"

    def run(
        self,
        normalized_image: np.ndarray,
        config: DetectorConfig,
        input_metadata: Dict[str, Any],
        context: DetectionContext,
    ) -> DetectionResult:
        settings = config.symmetry_relation
        peaks = estimate_global_reciprocal_peaks(
            normalized_image, config.spectral
        )
        selected = _refine_reciprocal_peaks(
            normalized_image,
            select_reciprocal_basis(peaks),
        )
        direct_basis = direct_basis_from_peaks(selected)
        lattice_spacing = float(
            np.median(np.linalg.norm(direct_basis, axis=0))
        )
        lattice_spacing = float(np.clip(lattice_spacing, 3.0, min(normalized_image.shape) / 3))
        reciprocal = np.asarray(
            [
                [selected[0][1], selected[0][0]],
                [selected[1][1], selected[1][0]],
            ],
            dtype=float,
        )
        patch_radius = max(
            int(round(0.5 * settings.reference_patch_cells * lattice_spacing)),
            settings.motif_bins,
        )
        stride = max(
            int(round(settings.analysis_stride_cells * lattice_spacing)),
            4,
        )
        height, width = normalized_image.shape
        centers_y = np.arange(stride // 2, height, stride, dtype=int)
        centers_x = np.arange(stride // 2, width, stride, dtype=int)
        if len(centers_y) < 3 or len(centers_x) < 3:
            raise ValueError(
                "Image is too small for multi-reference symmetry analysis"
            )

        demodulation_sigma = float(
            np.clip(0.18 / lattice_spacing, 0.006, 0.025)
        )
        qx, qy, _, coherence = complex_demodulation(
            normalized_image,
            selected,
            demodulation_sigma,
            derivative_sigma=max(1.0, 0.12 * lattice_spacing),
            padding=max(16, patch_radius),
        )
        inverse_reciprocal = np.linalg.inv(reciprocal)
        deformation_x = (
            -np.einsum("ij,jhw->ihw", inverse_reciprocal, qx)
            / (2 * np.pi)
        )
        deformation_y = (
            -np.einsum("ij,jhw->ihw", inverse_reciprocal, qy)
            / (2 * np.pi)
        )
        deformation_fields = np.stack(
            (
                deformation_x[0],
                deformation_y[1],
                0.5 * (deformation_x[1] + deformation_y[0]),
                0.5 * (deformation_x[1] - deformation_y[0]),
            )
        )

        observations: list[PatchObservation] = []
        global_contrast = float(np.std(normalized_image)) + 1e-8
        for center_y in centers_y:
            for center_x in centers_x:
                center = (int(center_y), int(center_x))
                motif, coverage = fold_periodic_motif(
                    normalized_image,
                    reciprocal,
                    center,
                    patch_radius,
                    settings.motif_bins,
                )
                y1, y2 = max(center_y - patch_radius, 0), min(
                    center_y + patch_radius + 1, height
                )
                x1, x2 = max(center_x - patch_radius, 0), min(
                    center_x + patch_radius + 1, width
                )
                local_coherence = float(
                    np.median(coherence[y1:y2, x1:x2])
                )
                local_contrast = float(
                    np.std(normalized_image[y1:y2, x1:x2])
                    / global_contrast
                )
                support = _edge_support(center, patch_radius, (height, width))
                deformation = deformation_fields[:, center_y, center_x]
                quality = (
                    0.45 * np.clip(local_coherence / 1.2, 0, 1)
                    + 0.20 * coverage
                    + 0.20 * support
                    + 0.15 * np.clip(local_contrast, 0, 1)
                )
                observations.append(
                    PatchObservation(
                        center_yx=center,
                        motif=motif,
                        motif_coverage=coverage,
                        coherence=local_coherence,
                        deformation=np.asarray(deformation, dtype=float),
                        edge_support=support,
                        quality=float(quality),
                    )
                )

        selection = select_reference_patches(
            observations,
            maximum_references=settings.maximum_references,
            minimum_references=settings.minimum_references,
            quality_quantile=settings.reference_quality_quantile,
            candidate_limit=min(len(observations), 256),
        )
        references = [observations[index] for index in selection.indices]

        identity_residual = np.zeros(len(observations), dtype=float)
        aligned_residual = np.zeros(len(observations), dtype=float)
        relation_gain = np.zeros(len(observations), dtype=float)
        relation_code = np.zeros(len(observations), dtype=np.uint8)
        winning_reference = np.zeros(len(observations), dtype=np.int16)
        alignments: list[GroupAlignment] = []
        for observation_index, observation in enumerate(observations):
            candidates = [
                align_group_motif(
                    observation.motif,
                    reference.motif,
                    identity_shift_tolerance_bins=(
                        settings.identity_shift_tolerance_bins
                    ),
                )
                for reference in references
            ]
            identity_residual[observation_index] = min(
                item.identity_residual for item in candidates
            )
            best_index = int(
                np.argmin([item.aligned_residual for item in candidates])
            )
            best = candidates[best_index]
            alignments.append(best)
            aligned_residual[observation_index] = best.aligned_residual
            relation_gain[observation_index] = best.relation_gain
            relation_code[observation_index] = _RELATION_CODES.get(
                best.relation, 5
            )
            winning_reference[observation_index] = best_index

        reference_indices = np.asarray(selection.indices, dtype=int)
        motif_z, motif_statistics = _positive_z(
            identity_residual,
            identity_residual[reference_indices],
            scale_floor=0.035,
        )
        reference_deformation = np.stack(
            [item.deformation for item in references]
        )
        deformation_center = np.median(reference_deformation, axis=0)
        deformation_distance = np.linalg.norm(
            np.stack([item.deformation for item in observations])
            - deformation_center,
            axis=1,
        )
        orientation_z, orientation_statistics = _positive_z(
            deformation_distance,
            deformation_distance[reference_indices],
            scale_floor=0.012,
        )

        grid_shape = (len(centers_y), len(centers_x))
        identity_grid = identity_residual.reshape(grid_shape)
        aligned_grid = aligned_residual.reshape(grid_shape)
        gain_grid = relation_gain.reshape(grid_shape)
        motif_z_grid = motif_z.reshape(grid_shape)
        orientation_z_grid = orientation_z.reshape(grid_shape)
        relation_grid = relation_code.reshape(grid_shape)
        coherence_grid = np.asarray(
            [item.coherence for item in observations]
        ).reshape(grid_shape)
        edge_grid = np.asarray(
            [item.edge_support for item in observations]
        ).reshape(grid_shape)

        deformation_grid = np.stack(
            [item.deformation for item in observations]
        ).reshape(*grid_shape, 4)
        gradient_y = np.gradient(deformation_grid, axis=0)
        gradient_x = np.gradient(deformation_grid, axis=1)
        energy_y = np.linalg.norm(gradient_y, axis=-1)
        energy_x = np.linalg.norm(gradient_x, axis=-1)
        gradient_energy = np.hypot(energy_x, energy_y)
        gradient_z, artifact_statistics = _positive_z(
            gradient_energy,
            gradient_energy[
                np.unravel_index(reference_indices, grid_shape)
            ],
            scale_floor=0.01,
        )
        axis_dominance = np.abs(energy_x - energy_y) / (
            energy_x + energy_y + 1e-8
        )
        explained_relation = np.clip(
            gain_grid / max(settings.minimum_motif_effect, 1e-8),
            0,
            1,
        )
        artifact_grid = (
            gradient_z * axis_dominance * (1.0 - explained_relation)
        )

        relation_explained = (
            (relation_grid != _RELATION_CODES["identity"])
            & (gain_grid >= settings.minimum_motif_effect)
            & (aligned_grid <= settings.maximum_aligned_residual)
        )
        absolute_motif = identity_grid >= settings.minimum_motif_effect
        domain_grid_score = np.maximum(
            motif_z_grid,
            orientation_z_grid,
        )
        domain_grid_score = np.clip(
            domain_grid_score
            - settings.artifact_competition_weight * artifact_grid,
            0,
            12,
        )
        domain_grid = (
            (
                (relation_explained & absolute_motif)
                | (
                    (
                        orientation_z_grid
                        >= settings.orientation_score_threshold
                    )
                    & absolute_motif
                )
            )
            & (domain_grid_score >= settings.domain_score_threshold)
            & (coherence_grid >= settings.minimum_coherence)
        )
        domain_grid = remove_small_objects(
            domain_grid,
            min_size=settings.minimum_domain_cells,
            connectivity=2,
        )
        domain_grid = binary_closing(domain_grid, np.ones((3, 3), dtype=bool))

        domain_score = _resize_grid(domain_grid_score, (height, width))
        motif_evidence = _resize_grid(motif_z_grid, (height, width))
        orientation_evidence = _resize_grid(
            orientation_z_grid, (height, width)
        )
        artifact_evidence = _resize_grid(artifact_grid, (height, width))
        relation_label = _resize_grid(
            relation_grid, (height, width), categorical=True
        ).astype(np.uint8)
        domain_region = _resize_grid(
            domain_grid.astype(float),
            (height, width),
            categorical=True,
        ) >= 0.5
        domain_region = remove_small_objects(
            domain_region,
            min_size=max(
                settings.minimum_domain_cells * stride * stride // 2,
                8,
            ),
            connectivity=2,
        )

        geometry_width = max(
            1, int(round(settings.geometry_width_cells * lattice_spacing))
        )
        interface_line = find_boundaries(domain_region, mode="thick")
        interface_line = binary_dilation(
            interface_line, disk(max(1, geometry_width // 2))
        )

        phase_curvature = np.sqrt(
            sum(
                np.gradient(component, axis=axis) ** 2
                for component in [*qx, *qy]
                for axis in (0, 1)
            )
        )
        coherent_reference = (
            coherence >= max(
                settings.minimum_coherence,
                float(np.median(coherence)),
            )
        )
        phase_z, phase_statistics = _positive_z(
            phase_curvature,
            phase_curvature[coherent_reference],
            scale_floor=0.0075 / lattice_spacing,
        )
        artifact_full = artifact_evidence
        phase_score = np.clip(
            phase_z
            - settings.artifact_competition_weight * artifact_full,
            0,
            12,
        )
        phase_candidate = (
            (phase_score >= settings.phase_slip_score_threshold)
            & (coherence >= settings.minimum_coherence)
        )
        phase_candidate = binary_closing(
            phase_candidate, disk(max(1, geometry_width // 2))
        )
        phase_skeleton = skeletonize(phase_candidate)
        phase_skeleton = remove_small_objects(
            phase_skeleton,
            min_size=max(int(round(1.5 * lattice_spacing)), 8),
            connectivity=2,
        )
        phase_slip_line = binary_dilation(
            phase_skeleton, disk(max(1, geometry_width // 2))
        )
        boundary_support_radius = max(
            1,
            int(
                round(
                    settings.boundary_support_radius_cells
                    * lattice_spacing
                )
            ),
        )
        phase_neighborhood = binary_dilation(
            phase_skeleton,
            disk(boundary_support_radius),
        )
        supported_domain = np.zeros_like(domain_region, dtype=bool)
        domain_boundary_support: list[float] = []
        domain_labels = label(domain_region, connectivity=2)
        for component_id in range(1, int(domain_labels.max()) + 1):
            component = domain_labels == component_id
            boundary = find_boundaries(component, mode="inner")
            support_fraction = (
                float(np.mean(phase_neighborhood[boundary]))
                if np.any(boundary)
                else 0.0
            )
            domain_boundary_support.append(support_fraction)
            if (
                support_fraction
                >= settings.minimum_boundary_phase_support
            ):
                supported_domain |= component
        domain_region = supported_domain
        interface_line = binary_dilation(
            find_boundaries(domain_region, mode="thick"),
            disk(max(1, geometry_width // 2)),
        )
        near_domain_interface = phase_skeleton & binary_dilation(
            interface_line,
            disk(boundary_support_radius),
        )
        if np.any(near_domain_interface):
            interface_line = binary_dilation(
                near_domain_interface,
                disk(max(1, geometry_width // 2)),
            )

        curls = []
        for carrier in range(qx.shape[0]):
            curls.append(
                np.gradient(qy[carrier], axis=1)
                - np.gradient(qx[carrier], axis=0)
            )
        burgers_inconsistency = np.sqrt(
            np.sum(np.stack(curls) ** 2, axis=0)
        )
        burgers_z, burgers_statistics = _positive_z(
            burgers_inconsistency,
            burgers_inconsistency[coherent_reference],
            scale_floor=1e-5,
        )
        neighbor_count = convolve(
            phase_skeleton.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
            cval=0,
        ) - phase_skeleton.astype(np.uint8)
        endpoints = phase_skeleton & (neighbor_count <= 1)
        phase_component_labels = label(phase_skeleton, connectivity=2)
        supported_endpoints = np.zeros_like(endpoints)
        minimum_core_line_length = max(
            int(round(4.0 * lattice_spacing)),
            16,
        )
        phase_component_count = int(phase_component_labels.max())
        core_support_component_count = 0
        for component_id in range(1, phase_component_count + 1):
            component = phase_component_labels == component_id
            if int(np.sum(component)) < minimum_core_line_length:
                continue
            supported_endpoints |= endpoints & component
            core_support_component_count += 1
        endpoints = supported_endpoints
        endpoint_margin = max(1, int(round(lattice_spacing)))
        endpoints[:endpoint_margin, :] = False
        endpoints[-endpoint_margin:, :] = False
        endpoints[:, :endpoint_margin] = False
        endpoints[:, -endpoint_margin:] = False
        core_seeds = np.zeros_like(endpoints)
        core_search_radius = max(
            1, int(round(0.5 * lattice_spacing))
        )
        for endpoint_y, endpoint_x in np.argwhere(endpoints):
            y1, y2 = max(0, endpoint_y - core_search_radius), min(
                height, endpoint_y + core_search_radius + 1
            )
            x1, x2 = max(0, endpoint_x - core_search_radius), min(
                width, endpoint_x + core_search_radius + 1
            )
            local = burgers_z[y1:y2, x1:x2]
            flat_index = int(np.argmax(local))
            local_y, local_x = np.unravel_index(
                flat_index, local.shape
            )
            seed_y, seed_x = y1 + local_y, x1 + local_x
            if (
                burgers_z[seed_y, seed_x]
                >= settings.burgers_score_threshold
                and coherence[seed_y, seed_x]
                >= settings.minimum_coherence
            ):
                core_seeds[seed_y, seed_x] = True
        dislocation_core = binary_dilation(
            core_seeds,
            disk(max(1, int(round(0.20 * lattice_spacing)))),
        )

        reference_support_grid = np.zeros(grid_shape, dtype=float)
        for index in selection.indices:
            row, column = np.unravel_index(index, grid_shape)
            reference_support_grid[row, column] = 1.0
        reference_support = _resize_grid(
            gaussian_filter(reference_support_grid, 1.0),
            (height, width),
        )
        if np.max(reference_support) > 0:
            reference_support /= np.max(reference_support)
        edge_reliability = _resize_grid(edge_grid, (height, width))
        measurement_reliability = (
            np.clip(coherence, 0, 1)
            * (
                settings.edge_reliability_floor
                + (1 - settings.edge_reliability_floor) * edge_reliability
            )
        )

        composite_ratio = np.maximum.reduce(
            (
                domain_score / settings.domain_score_threshold,
                phase_score / settings.phase_slip_score_threshold,
                burgers_z / settings.burgers_score_threshold,
            )
        )
        score = (settings.domain_score_threshold * composite_ratio).astype(
            np.float32
        )
        probability = expit(
            (score - settings.domain_score_threshold) / 0.75
        ).astype(np.float32)
        mask = domain_region | phase_slip_line | dislocation_core

        evidence_maps = {
            "group_phase_motif_residual": motif_evidence.astype(np.float32),
            "group_orientation_strain": orientation_evidence.astype(np.float32),
            "group_displacement_discontinuity": phase_score.astype(np.float32),
            "group_burgers_loop_inconsistency": burgers_z.astype(np.float32),
            "group_scan_time_artifact": artifact_evidence.astype(np.float32),
            "lattice_coherence": coherence.astype(np.float32),
            "symmetry_relation_validity": measurement_reliability.astype(
                np.float32
            ),
            "reference_patch_support": reference_support.astype(np.float32),
            "relation_label": relation_label.astype(np.float32),
        }
        feature_maps = {
            **evidence_maps,
            "identity_residual": _resize_grid(
                identity_grid, (height, width)
            ).astype(np.float32),
            "aligned_group_residual": _resize_grid(
                aligned_grid, (height, width)
            ).astype(np.float32),
            "relation_gain": _resize_grid(
                gain_grid, (height, width)
            ).astype(np.float32),
            "symmetry_domain_region": domain_region.astype(np.float32),
            "interface_line": interface_line.astype(np.float32),
            "phase_slip_line": phase_slip_line.astype(np.float32),
            "dislocation_core_point": dislocation_core.astype(np.float32),
        }

        region_groups: list[DetectionRegion] = []
        geometry_specs = (
            (
                domain_region,
                "symmetry_related_domain",
                "region",
                "symmetry_domain_region",
                [
                    "group_phase_motif_residual",
                    "group_orientation_strain",
                ],
            ),
            (
                interface_line,
                "symmetry_domain_interface",
                "interface",
                "interface_line",
                [
                    "group_phase_motif_residual",
                    "group_orientation_strain",
                ],
            ),
            (
                phase_slip_line,
                "lattice_phase_slip_line",
                "line",
                "phase_slip_line",
                [
                    "group_displacement_discontinuity",
                    "group_phase_motif_residual",
                ],
            ),
            (
                dislocation_core,
                "dislocation_core_proxy",
                "point",
                "dislocation_core_point",
                [
                    "group_burgers_loop_inconsistency",
                    "group_displacement_discontinuity",
                ],
            ),
        )
        next_id = 1
        for (
            geometry_mask,
            defect_type,
            support,
            geometry,
            names,
        ) in geometry_specs:
            added = _geometry_regions(
                geometry_mask,
                score,
                probability,
                evidence_maps,
                start_id=next_id,
                defect_type=defect_type,
                spatial_support=support,
                geometry=geometry,
                evidence_names=names,
            )
            region_groups.extend(added)
            next_id += len(added)

        metadata = {
            **input_metadata,
            "method": "multi_reference_group_relation_five_teacher",
            "method_version": "class1-final-v1",
            "neural_training_used": False,
            "supervised_labels_used": False,
            "per_image_statistical_adaptation": True,
            "lattice_spacing_pixels": lattice_spacing,
            "selected_reciprocal_basis_fy_fx": [
                list(selected[0]),
                list(selected[1]),
            ],
            "reference_selection": {
                "selected_observation_indices": list(selection.indices),
                "selected_centers_yx": [
                    list(observations[index].center_yx)
                    for index in selection.indices
                ],
                "candidate_count": len(selection.candidate_indices),
                "identity_cluster_count": len(selection.cluster_indices),
                "cluster_support_fraction": (
                    selection.cluster_support_fraction
                ),
                "identity_radius": selection.identity_radius,
                "ambiguous_identity": selection.ambiguous_identity,
            },
            "domain_boundary_phase_support": domain_boundary_support,
            "teachers": {
                "group_phase_motif": motif_statistics,
                "orientation_strain": orientation_statistics,
                "displacement_discontinuity": phase_statistics,
                "burgers_loop_proxy": burgers_statistics,
                "scan_time_artifact": {
                    **artifact_statistics,
                    "temporal_evidence_available": False,
                },
            },
            "geometry_counts": {
                name: int(np.sum(feature_maps[name] > 0.5))
                for name in (
                    "symmetry_domain_region",
                    "interface_line",
                    "phase_slip_line",
                    "dislocation_core_point",
                )
            },
            "topology_counts": {
                "phase_skeleton_components": phase_component_count,
                "core_support_components": core_support_component_count,
                "supported_endpoints": int(np.sum(endpoints)),
                "core_seeds": int(np.sum(core_seeds)),
            },
            "no_defect_gate_passed": not bool(np.any(mask)),
            "interpretation_limits": [
                "Group relations are indexed in the estimated 2-D lattice basis.",
                "Burgers-loop output is a curl/topology proxy, not a confirmed Burgers vector.",
                "Single images provide scan-axis evidence but no direct temporal evidence.",
                "Edge candidates are retained with reduced measurement reliability.",
            ],
            "context": {
                "skeleton_provided": context.skeleton is not None,
                "exemplar_provided": context.exemplar_image is not None,
            },
        }
        return DetectionResult(
            task=self.name,
            image_shape=normalized_image.shape,
            normalized_image=normalized_image,
            score_map=score,
            probability_map=probability,
            mask=np.asarray(mask, dtype=bool),
            regions=region_groups,
            feature_maps=feature_maps,
            evidence_maps=evidence_maps,
            metadata=metadata,
        )
