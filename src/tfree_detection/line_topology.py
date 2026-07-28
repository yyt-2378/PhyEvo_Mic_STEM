"""Deterministic topology descriptors for line and interface candidates."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.ndimage import convolve, label
from skimage.morphology import skeletonize

from .models import DetectionResult


LINE_TOPOLOGY_VERSION = "1.0"


def _skeleton_length(skeleton: np.ndarray) -> float:
    horizontal = np.sum(skeleton[:, :-1] & skeleton[:, 1:])
    vertical = np.sum(skeleton[:-1, :] & skeleton[1:, :])
    diagonal_first = np.sum(skeleton[:-1, :-1] & skeleton[1:, 1:])
    diagonal_second = np.sum(skeleton[1:, :-1] & skeleton[:-1, 1:])
    return float(
        horizontal
        + vertical
        + np.sqrt(2.0) * (diagonal_first + diagonal_second)
    )


def _geometry(skeleton: np.ndarray) -> Dict[str, Any]:
    coordinates_yx = np.argwhere(skeleton)
    if len(coordinates_yx) < 2:
        return {
            "skeleton_pixel_count": int(len(coordinates_yx)),
            "geodesic_edge_length_pixels": 0.0,
            "endpoint_count": 0,
            "junction_count": 0,
            "component_count": int(bool(len(coordinates_yx))),
            "linearity": 0.0,
            "major_axis_angle_degrees": None,
            "topology_state": "insufficient_line_support",
        }
    neighbors = convolve(
        skeleton.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    ) - skeleton.astype(np.uint8)
    endpoints = skeleton & (neighbors == 1)
    junctions = skeleton & (neighbors >= 3)
    _, component_count = label(skeleton, structure=np.ones((3, 3)))
    centered = coordinates_yx - np.mean(coordinates_yx, axis=0)
    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    major = max(float(eigenvalues[order[0]]), 1e-12)
    minor = max(float(eigenvalues[order[1]]), 0.0)
    linearity = float(np.clip(1.0 - minor / major, 0.0, 1.0))
    major_y, major_x = eigenvectors[:, order[0]]
    angle = float(np.degrees(np.arctan2(major_y, major_x)))
    endpoint_count = int(np.sum(endpoints))
    junction_count = int(np.sum(junctions))
    if component_count > 1:
        state = "fragmented_line_support"
    elif junction_count > 0:
        state = "branching_line_network"
    elif endpoint_count <= 2 and linearity >= 0.75:
        state = "simple_coherent_line"
    else:
        state = "curved_or_complex_line"
    return {
        "skeleton_pixel_count": int(len(coordinates_yx)),
        "geodesic_edge_length_pixels": _skeleton_length(skeleton),
        "endpoint_count": endpoint_count,
        "junction_count": junction_count,
        "component_count": int(component_count),
        "linearity": linearity,
        "major_axis_angle_degrees": angle,
        "topology_state": state,
    }


def attach_line_topology(result: DetectionResult) -> DetectionResult:
    """Add topology as diagnostic evidence without changing any mask."""

    if result.task not in {"stacking_fault", "phase_boundary"}:
        raise ValueError(
            "Line topology is only defined for stacking_fault or "
            "phase_boundary results"
        )
    skeleton_map = np.zeros(result.image_shape, dtype=np.float32)
    endpoint_map = np.zeros(result.image_shape, dtype=np.float32)
    junction_map = np.zeros(result.image_shape, dtype=np.float32)
    for region in result.regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        local_mask = result.mask[y1:y2, x1:x2]
        local_skeleton = skeletonize(local_mask)
        neighbors = convolve(
            local_skeleton.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            mode="constant",
            cval=0,
        ) - local_skeleton.astype(np.uint8)
        local_endpoints = local_skeleton & (neighbors == 1)
        local_junctions = local_skeleton & (neighbors >= 3)
        skeleton_map[y1:y2, x1:x2] = np.maximum(
            skeleton_map[y1:y2, x1:x2],
            local_skeleton.astype(np.float32),
        )
        endpoint_map[y1:y2, x1:x2] = np.maximum(
            endpoint_map[y1:y2, x1:x2],
            local_endpoints.astype(np.float32),
        )
        junction_map[y1:y2, x1:x2] = np.maximum(
            junction_map[y1:y2, x1:x2],
            local_junctions.astype(np.float32),
        )
        region.provenance = {
            **region.provenance,
            "line_topology": _geometry(local_skeleton),
            "line_topology_version": LINE_TOPOLOGY_VERSION,
            "mechanism_confirmation": False,
        }
    result.evidence_maps.update(
        {
            "topology_skeleton": skeleton_map,
            "topology_endpoints": endpoint_map,
            "topology_junctions": junction_map,
        }
    )
    result.feature_maps.update(
        {
            "topology_skeleton": skeleton_map,
            "topology_endpoints": endpoint_map,
            "topology_junctions": junction_map,
        }
    )
    result.metadata["line_topology"] = {
        "version": LINE_TOPOLOGY_VERSION,
        "candidate_count": len(result.regions),
        "semantics": (
            "Morphological line/interface support only; Burgers vector, "
            "stacking sequence, and phase identity remain unresolved."
        ),
        "mask_changed": False,
        "labels_read": False,
    }
    return result
