"""Training-free local amorphization detector."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..config import DetectorConfig
from ..context import DetectionContext
from ..features import SpectralFeatureExtractor
from ..fusion import EvidenceSpec, fuse_evidence
from ..models import DetectionResult
from ..postprocessing import postprocess_regions, project_grid
from .base import DetectionTask


class AmorphousRegionTask(DetectionTask):
    """Detect local loss of crystalline order using independent evidence."""

    name = "amorphous_region"

    evidence_specs: List[EvidenceSpec] = [
        EvidenceSpec("spectral_entropy", +1, 1.0, "spectrum"),
        EvidenceSpec("spectral_flatness", +1, 0.7, "spectrum"),
        EvidenceSpec("spectral_peak_concentration", -1, 1.0, "spectrum"),
        EvidenceSpec("radial_peak_concentration", -1, 0.8, "spectrum"),
        EvidenceSpec("global_lattice_power", -1, 1.1, "spectrum"),
        EvidenceSpec("patterson_order", -1, 1.2, "patterson"),
        EvidenceSpec("gradient_energy", -1, 0.7, "sharpness"),
    ]

    def run(
        self,
        normalized_image: np.ndarray,
        config: DetectorConfig,
        input_metadata: Dict[str, Any],
        context: DetectionContext,
    ) -> DetectionResult:
        feature_grid = SpectralFeatureExtractor(config.spectral).extract(
            normalized_image
        )
        fusion = fuse_evidence(
            feature_grid, self.evidence_specs, config.fusion
        )
        spatial = postprocess_regions(
            fusion=fusion,
            image_shape=normalized_image.shape,
            fusion_config=config.fusion,
            config=config.postprocess,
            patch_size=config.spectral.patch_size,
        )

        if config.output.save_feature_maps:
            feature_maps = {
                name: project_grid(values, normalized_image.shape, order=1).astype(
                    np.float32
                )
                for name, values in feature_grid.values.items()
            }
        else:
            feature_maps = {}
        evidence_maps = {
            f"group_{name}": values
            for name, values in spatial.group_maps.items()
        }
        evidence_maps["support_count"] = spatial.support_count_map.astype(
            np.float32
        )

        stage2_metadata: Dict[str, Any] = {"enabled": False}
        if config.stage2.enabled:
            stage2_metadata = {"enabled": True}
            stage2_regions: Dict[int, Dict[str, Any]] = {
                region.region_id: {"region_id": region.region_id}
                for region in spatial.regions
            }
            if config.stage2.phase.enabled:
                from ..stage2.phase import analyze_lattice_deformation

                peaks = feature_grid.metadata[
                    "global_reciprocal_peaks_fy_fx"
                ]
                phase = analyze_lattice_deformation(
                    normalized_image,
                    spatial.mask,
                    [tuple(peak) for peak in peaks],
                    config.stage2.phase,
                )
                evidence_maps.update(phase.maps)
                stage2_metadata["phase"] = phase.metadata
                for values in phase.region_metrics:
                    stage2_regions[values["region_id"]]["phase"] = values
            if config.stage2.dino.enabled:
                from ..stage2.dino import verify_regions_with_dino

                dino = verify_regions_with_dino(
                    normalized_image,
                    spatial.mask,
                    config.stage2.phase,
                    config.stage2.dino,
                    context.exemplar_image,
                )
                stage2_metadata["dino"] = dino.metadata
                for values in dino.region_metrics:
                    stage2_regions[values["region_id"]]["dino"] = values
            for values in stage2_regions.values():
                phase_support = values.get("phase", {}).get(
                    "phase_support", "not_run"
                )
                dino_support = values.get("dino", {}).get(
                    "dino_support", "not_run"
                )
                if phase_support == "strong" and dino_support in {
                    "strong",
                    "moderate",
                }:
                    combined = "strong"
                elif "strong" in {phase_support, dino_support} or (
                    phase_support == "moderate"
                    and dino_support == "moderate"
                ):
                    combined = "moderate"
                else:
                    combined = "weak"
                values["combined_support"] = combined
            stage2_metadata["regions"] = list(stage2_regions.values())

        metadata = {
            **input_metadata,
            "context": {
                "skeleton_provided": context.skeleton is not None,
                "skeleton_node_count": (
                    len(context.skeleton.nodes) if context.skeleton else 0
                ),
                "exemplar_provided": context.exemplar_image is not None,
            },
            "threshold": fusion.threshold,
            "score_scale": fusion.score_scale,
            "valid_border_margin_pixels": int(
                np.clip(
                    (
                        config.spectral.patch_size // 2
                        if config.postprocess.border_margin is None
                        else config.postprocess.border_margin
                    ),
                    0,
                    max(min(normalized_image.shape) // 3, 0),
                )
            ),
            "feature_statistics": fusion.statistics,
            "feature_grid": {
                "shape": list(feature_grid.shape),
                **feature_grid.metadata,
            },
            "decision_rule": {
                "minimum_evidence_groups": config.fusion.minimum_evidence_groups,
                "support_z": config.fusion.support_z,
                "groups": sorted(
                    {spec.group for spec in self.evidence_specs}
                ),
            },
            "stage2": stage2_metadata,
        }
        return DetectionResult(
            task=self.name,
            image_shape=tuple(normalized_image.shape),
            normalized_image=normalized_image,
            score_map=spatial.score_map,
            probability_map=spatial.probability_map,
            mask=spatial.mask,
            regions=spatial.regions,
            feature_maps=feature_maps,
            evidence_maps=evidence_maps,
            metadata=metadata,
        )
