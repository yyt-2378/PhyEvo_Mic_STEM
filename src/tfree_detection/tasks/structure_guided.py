"""Framework task for projected-template optional-site anomalies."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..config import DetectorConfig
from ..context import DetectionContext
from ..models import DetectionResult
from ..structure_guided.analyzer import (
    analysis_to_detection_result,
    analyze_projected_site_probes,
)
from ..structure_guided.contracts import StructureGuidedPolicy
from ..structure_guided.template import projected_probes_from_skeleton
from .base import DetectionTask


class StructureGuidedSiteTask(DetectionTask):
    """Detect an unexpected component at optional projected template sites."""

    name = "structure_guided_site"

    def run(
        self,
        normalized_image: np.ndarray,
        config: DetectorConfig,
        input_metadata: Dict[str, Any],
        context: DetectionContext,
    ) -> DetectionResult:
        if context.skeleton is None:
            raise ValueError(
                "structure_guided_site requires an IdealSkeleton or --skeleton JSON"
            )
        settings = config.structure_guided
        probes = projected_probes_from_skeleton(
            context.skeleton, tuple(normalized_image.shape)
        )
        analysis = analyze_projected_site_probes(
            normalized_image,
            probes,
            sample_role=settings.sample_role,
            policy=StructureGuidedPolicy(
                minimum_delta_bic=settings.minimum_delta_bic,
                profile_sigma_pixels=settings.profile_sigma_pixels,
                minimum_endpoint_separation_pixels=(
                    settings.minimum_endpoint_separation_pixels
                ),
                border_pixels=settings.border_pixels,
                candidate_radius_pixels=settings.candidate_radius_pixels,
            ),
        )
        return analysis_to_detection_result(
            analysis, normalized_image, input_metadata
        )
