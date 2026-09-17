"""Generic projected-site analysis and DetectionResult conversion."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from skimage.draw import disk

from ..models import DetectionRegion, DetectionResult
from .contracts import (
    ProjectedSiteProbe,
    SampleRole,
    StructureGuidedAnalysis,
    StructureGuidedPolicy,
)
from .model_selection import compare_optional_center_profile


def _inside_border(
    probe: ProjectedSiteProbe,
    shape: tuple[int, int],
    border: float,
) -> bool:
    height, width = shape
    points = np.vstack(
        (np.asarray(probe.center_xy), np.asarray(probe.endpoint_xy))
    )
    return bool(
        np.all(points[:, 0] >= border)
        and np.all(points[:, 0] < width - border)
        and np.all(points[:, 1] >= border)
        and np.all(points[:, 1] < height - border)
    )


def analyze_projected_site_probes(
    image_or_stack: np.ndarray,
    probes: Sequence[ProjectedSiteProbe],
    *,
    sample_role: SampleRole | str = SampleRole.TARGET_POSSIBLE,
    policy: StructureGuidedPolicy | None = None,
) -> StructureGuidedAnalysis:
    """Measure optional-center support and apply a declared sample role."""

    active_policy = policy or StructureGuidedPolicy()
    active_policy.validate()
    role = SampleRole.parse(sample_role)
    stack = np.asarray(image_or_stack, dtype=float)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError("image_or_stack must have shape (y, x) or (plane, y, x)")
    if not np.all(np.isfinite(stack)):
        raise ValueError("image_or_stack contains non-finite values")
    if not probes:
        raise ValueError("At least one ProjectedSiteProbe is required")
    for probe in probes:
        probe.validate()

    count = len(probes)
    best_plane_index = np.full(count, -1, dtype=int)
    delta_bic = np.full(count, float("-inf"), dtype=float)
    center_amplitude = np.zeros(count, dtype=float)
    model_valid = np.zeros(count, dtype=bool)
    interior = np.asarray(
        [
            _inside_border(
                probe, tuple(stack.shape[1:]), active_policy.border_pixels
            )
            for probe in probes
        ],
        dtype=bool,
    )

    # This role is declared before scoring. Its contract is stronger than
    # merely hiding candidates: do not retain a high-scoring reference pool
    # that downstream code could reinterpret as target-positive evidence.
    if role is SampleRole.TARGET_FREE_REFERENCE:
        return StructureGuidedAnalysis(
            probes=tuple(probes),
            sample_role=role,
            policy=active_policy,
            best_plane_index=best_plane_index,
            delta_bic=delta_bic,
            center_amplitude=center_amplitude,
            model_valid=model_valid,
            interior=interior,
            structural_positive=np.zeros(count, dtype=bool),
            candidate=np.zeros(count, dtype=bool),
            policy_suppressed=True,
        )

    for probe_index, probe in enumerate(probes):
        for plane_index, frame in enumerate(stack):
            comparison = compare_optional_center_profile(
                frame,
                probe.center_xy,
                probe.endpoint_xy,
                sigma_pixels=active_policy.profile_sigma_pixels,
                minimum_endpoint_separation_pixels=(
                    active_policy.minimum_endpoint_separation_pixels
                ),
            )
            if comparison.valid and comparison.delta_bic > delta_bic[probe_index]:
                best_plane_index[probe_index] = plane_index
                delta_bic[probe_index] = comparison.delta_bic
                center_amplitude[probe_index] = comparison.center_amplitude
                model_valid[probe_index] = True

    measured_positive = (
        model_valid
        & interior
        & (center_amplitude > 0)
        & (delta_bic >= active_policy.minimum_delta_bic)
    )
    policy_suppressed = False
    structural_positive = measured_positive.copy()
    candidate = measured_positive.copy()

    return StructureGuidedAnalysis(
        probes=tuple(probes),
        sample_role=role,
        policy=active_policy,
        best_plane_index=best_plane_index,
        delta_bic=delta_bic,
        center_amplitude=center_amplitude,
        model_valid=model_valid,
        interior=interior,
        structural_positive=structural_positive,
        candidate=candidate,
        policy_suppressed=policy_suppressed,
    )


def analysis_to_detection_result(
    analysis: StructureGuidedAnalysis,
    normalized_image: np.ndarray,
    input_metadata: dict[str, object] | None = None,
) -> DetectionResult:
    """Render interpreted probes into the framework's common result contract."""

    image = np.asarray(normalized_image, dtype=float)
    if image.ndim != 2:
        raise ValueError("normalized_image must be 2-D")
    height, width = image.shape
    score_map = np.zeros(image.shape, dtype=float)
    probability_map = np.zeros(image.shape, dtype=float)
    mask = np.zeros(image.shape, dtype=bool)
    model_gain_map = np.zeros(image.shape, dtype=float)
    center_component_map = np.zeros(image.shape, dtype=float)
    support_map = np.zeros(image.shape, dtype=float)
    reliability_map = np.zeros(image.shape, dtype=float)
    regions: list[DetectionRegion] = []
    radius = max(1, int(round(analysis.policy.candidate_radius_pixels)))

    # A declared target-free reference exposes no positive evidence map. Model
    # fitting remains an implementation detail, not a retained audit pool.
    expose_measurements = analysis.sample_role is SampleRole.TARGET_POSSIBLE
    for index, probe in enumerate(analysis.probes):
        x, y = probe.center_xy
        rr, cc = disk((y, x), radius, shape=image.shape)
        if analysis.model_valid[index] and analysis.interior[index]:
            support_map[rr, cc] = 1.0
            reliability_map[rr, cc] = 1.0
            if expose_measurements:
                gain = max(float(analysis.delta_bic[index]), 0.0)
                component = max(float(analysis.center_amplitude[index]), 0.0)
                model_gain_map[rr, cc] = np.maximum(model_gain_map[rr, cc], gain)
                center_component_map[rr, cc] = np.maximum(
                    center_component_map[rr, cc], component
                )
        if not analysis.candidate[index]:
            continue
        delta = float(analysis.delta_bic[index])
        excess = delta - analysis.policy.minimum_delta_bic
        probability = float(1.0 / (1.0 + np.exp(-excess)))
        score_map[rr, cc] = np.maximum(score_map[rr, cc], delta)
        probability_map[rr, cc] = np.maximum(probability_map[rr, cc], probability)
        mask[rr, cc] = True
        x1 = max(0, int(np.floor(x - radius)))
        y1 = max(0, int(np.floor(y - radius)))
        x2 = min(width, int(np.ceil(x + radius + 1)))
        y2 = min(height, int(np.ceil(y + radius + 1)))
        regions.append(
            DetectionRegion(
                region_id=len(regions) + 1,
                candidate_id=probe.probe_id,
                bbox_xyxy=(x1, y1, x2, y2),
                centroid_xy=(float(x), float(y)),
                area_pixels=int(len(rr)),
                mean_score=delta,
                max_score=delta,
                confidence=probability,
                confidence_level="high" if probability >= 0.8 else "medium",
                evidence_groups=[
                    "structure_model_selection",
                    "center_component",
                ],
                defect_type="unexpected_template_site_occupancy",
                spatial_support="projected_template_site",
                provenance={
                    "probe_family": probe.family,
                    "best_plane_index": int(analysis.best_plane_index[index]),
                },
            )
        )

    metadata = dict(input_metadata or {})
    metadata["structure_guided"] = analysis.summary()
    return DetectionResult(
        task="structure_guided_site",
        image_shape=image.shape,
        normalized_image=image,
        score_map=score_map,
        probability_map=probability_map,
        mask=mask,
        regions=regions,
        feature_maps={"projected_probe_support": support_map},
        evidence_maps={
            "group_structural_model_gain": model_gain_map,
            "group_center_component": center_component_map,
            "structure_probe_support": support_map,
            "structure_probe_reliability": reliability_map,
        },
        metadata=metadata,
    )
