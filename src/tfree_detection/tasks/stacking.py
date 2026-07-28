"""Training-free stacking-fault and dislocation-line detection."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import binary_closing, dilation, disk, remove_small_objects

from ..config import DetectorConfig
from ..context import DetectionContext
from ..features.lattice import complex_demodulation, select_reciprocal_basis
from ..features.spectral import estimate_global_reciprocal_peaks
from ..models import DetectionResult
from .base import DetectionTask
from .common import (
    positive_robust_z,
    probability_from_threshold,
    regions_from_mask,
)


class StackingFaultTask(DetectionTask):
    """Locate abrupt lattice phase slips and their terminating dislocations."""

    name = "stacking_fault"

    def run(
        self,
        normalized_image: np.ndarray,
        config: DetectorConfig,
        input_metadata: Dict[str, Any],
        context: DetectionContext,
    ) -> DetectionResult:
        line_config = config.line_defect
        spatial_support = int(
            np.ceil(
                3.0
                / (
                    2
                    * np.pi
                    * line_config.frequency_filter_sigma
                )
            )
        )
        requested_margin = max(
            line_config.valid_border_margin, spatial_support
        )
        all_peaks = estimate_global_reciprocal_peaks(
            normalized_image, config.spectral
        )
        selected = select_reciprocal_basis(all_peaks)
        qx, qy, amplitudes, coherence = complex_demodulation(
            normalized_image,
            selected,
            line_config.frequency_filter_sigma,
            line_config.derivative_smoothing_sigma,
            padding=max(requested_margin, 16),
        )

        reciprocal = np.asarray(
            [
                [selected[0][1], selected[0][0]],
                [selected[1][1], selected[1][0]],
            ],
            dtype=float,
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
        exx = deformation_x[0]
        eyy = deformation_y[1]
        exy = 0.5 * (deformation_x[1] + deformation_y[0])
        rotation = 0.5 * (deformation_x[1] - deformation_y[0])
        strain = np.sqrt(exx**2 + eyy**2 + 2 * exy**2)

        phase_jump = np.sqrt(
            np.sum(qx**2 + qy**2, axis=0)
        )
        phase_curvature = np.sqrt(
            sum(
                np.gradient(component, axis=axis) ** 2
                for component in [*qx, *qy]
                for axis in (0, 1)
            )
        )
        shear_rotation = np.sqrt(exy**2 + rotation**2)
        coherence_loss = np.clip(1.0 - coherence, 0, None)

        height, width = normalized_image.shape
        valid = np.ones((height, width), dtype=bool)
        margin = min(
            max(requested_margin, 0),
            min(height, width) // 3,
        )
        if margin:
            valid[:margin, :] = False
            valid[-margin:, :] = False
            valid[:, :margin] = False
            valid[:, -margin:] = False
        reference = valid & (coherence >= line_config.minimum_coherence)
        if int(np.sum(reference)) < 64:
            raise ValueError(
                "Too few coherent pixels remain for stacking-fault analysis"
            )

        def standardized(values: np.ndarray) -> tuple[np.ndarray, dict]:
            reference_score, statistics = positive_robust_z(values[reference])
            center = float(statistics["median"])
            scale = float(statistics["robust_scale"])
            full = np.clip((values - center) / scale, 0, 12)
            return full.astype(np.float32), statistics

        jump_z, jump_statistics = standardized(phase_jump)
        curvature_z, curvature_statistics = standardized(phase_curvature)
        shear_z, shear_statistics = standardized(shear_rotation)
        coherence_z, coherence_statistics = standardized(coherence_loss)
        stacked = np.stack(
            [jump_z, curvature_z, shear_z, coherence_z], axis=-1
        )
        strongest_two = np.partition(stacked, -2, axis=-1)[..., -2:]
        score = np.mean(strongest_two, axis=-1)
        if line_config.score_smoothing_sigma > 0:
            score = gaussian_filter(
                score,
                line_config.score_smoothing_sigma,
                mode="nearest",
            )
        threshold = max(
            float(np.quantile(score[valid], line_config.threshold_quantile)),
            float(line_config.threshold_mad),
        )
        mask = (score >= threshold) & valid
        if line_config.morphology_radius > 0:
            footprint = disk(line_config.morphology_radius)
            mask = binary_closing(mask, footprint)
            mask = dilation(mask, disk(max(1, line_config.morphology_radius // 2)))
        if line_config.minimum_region_area > 0:
            mask = remove_small_objects(
                mask,
                min_size=line_config.minimum_region_area,
                connectivity=2,
            )

        probability = probability_from_threshold(score, threshold)
        evidence_maps = {
            "group_phase_jump": jump_z,
            "group_phase_curvature": curvature_z,
            "group_shear_rotation": shear_z,
            "group_coherence_loss": coherence_z,
            "lattice_coherence": coherence.astype(np.float32),
            "strain_magnitude": strain.astype(np.float32),
            "phase_slip_validity": valid.astype(np.float32),
        }
        regions = regions_from_mask(
            mask,
            score,
            probability,
            evidence_maps,
            defect_type="stacking_fault_or_dislocation_line",
            support_threshold=1.5,
            evidence_names=[
                "group_phase_jump",
                "group_phase_curvature",
                "group_shear_rotation",
                "group_coherence_loss",
            ],
        )
        metadata = {
            **input_metadata,
            "method": "two_carrier_complex_phase_slip",
            "neural_training_used": False,
            "supervised_labels_used": False,
            "per_image_statistical_adaptation": True,
            "selected_reciprocal_basis_fy_fx": [
                list(selected[0]),
                list(selected[1]),
            ],
            "threshold": threshold,
            "valid_border_margin_pixels": margin,
            "minimum_reliable_coherence": line_config.minimum_coherence,
            "feature_statistics": {
                "phase_jump": jump_statistics,
                "phase_curvature": curvature_statistics,
                "shear_rotation": shear_statistics,
                "coherence_loss": coherence_statistics,
            },
            "interpretation": (
                "Candidates are abrupt phase-slip lines; endpoints with strong "
                "rotation are compatible with partial/full dislocation cores."
            ),
            "context": {
                "skeleton_provided": context.skeleton is not None,
                "exemplar_provided": context.exemplar_image is not None,
            },
        }
        return DetectionResult(
            task=self.name,
            image_shape=normalized_image.shape,
            normalized_image=normalized_image,
            score_map=np.asarray(score, dtype=np.float32),
            probability_map=probability,
            mask=np.asarray(mask, dtype=bool),
            regions=regions,
            feature_maps={
                "phase_jump": phase_jump.astype(np.float32),
                "phase_curvature": phase_curvature.astype(np.float32),
                "shear_rotation": shear_rotation.astype(np.float32),
            },
            evidence_maps=evidence_maps,
            metadata=metadata,
        )
