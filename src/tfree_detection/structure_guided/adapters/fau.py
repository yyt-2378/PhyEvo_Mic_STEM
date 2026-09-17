"""Compatibility adapter from the frozen FAU v20 output to generic probes."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..analyzer import analyze_projected_site_probes
from ..contracts import (
    ProjectedSiteProbe,
    SampleRole,
    StructureGuidedAnalysis,
    StructureGuidedPolicy,
)


def sample_role_for_fau_label(
    label: str,
    *,
    reference_label: str,
    reference_is_target_free: bool,
) -> SampleRole:
    """Translate FAU sample semantics without leaking them into the core."""

    if reference_is_target_free and label == reference_label:
        return SampleRole.TARGET_FREE_REFERENCE
    return SampleRole.TARGET_POSSIBLE


def projected_probes_from_fau_site_i(evidence: Any) -> tuple[ProjectedSiteProbe, ...]:
    """Convert frozen ``SkeletonFirstSiteEvidence`` coordinates to probes."""

    centers = np.asarray(evidence.expected_xy, dtype=float)
    endpoints = np.asarray(evidence.oxygen_endpoint_xy, dtype=float)
    identifiers = np.asarray(evidence.canonical_site_id)
    valid = np.asarray(evidence.model_valid, dtype=bool)
    expected_shape = (len(centers),)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("FAU expected_xy must have shape (site, 2)")
    if endpoints.shape != (len(centers), 2, 2):
        raise ValueError("FAU oxygen_endpoint_xy must have shape (site, 2, 2)")
    if identifiers.shape != expected_shape or valid.shape != expected_shape:
        raise ValueError("FAU site arrays must have one entry per expected site")
    probes: list[ProjectedSiteProbe] = []
    for index in range(len(centers)):
        if not valid[index]:
            continue
        coordinates = np.vstack((centers[index], endpoints[index]))
        if not np.all(np.isfinite(coordinates)):
            continue
        probes.append(
            ProjectedSiteProbe(
                probe_id=str(identifiers[index]),
                family="optional_projected_site",
                center_xy=tuple(centers[index].tolist()),
                endpoint_xy=tuple(
                    tuple(point.tolist()) for point in endpoints[index]
                ),  # type: ignore[arg-type]
                metadata={
                    "adapter": "fau_site_i_v20",
                    "legacy_index": index,
                },
            )
        )
    if not probes:
        raise ValueError("FAU adapter produced no valid projected probes")
    return tuple(probes)


def analyze_fau_site_i_with_generic_engine(
    image_or_stack: np.ndarray,
    evidence: Any,
    *,
    label: str,
    reference_label: str,
    reference_is_target_free: bool,
    policy: StructureGuidedPolicy | None = None,
) -> StructureGuidedAnalysis:
    """Evaluate frozen FAU probe geometry through the generic engine."""

    return analyze_projected_site_probes(
        image_or_stack,
        projected_probes_from_fau_site_i(evidence),
        sample_role=sample_role_for_fau_label(
            label,
            reference_label=reference_label,
            reference_is_target_free=reference_is_target_free,
        ),
        policy=policy,
    )
