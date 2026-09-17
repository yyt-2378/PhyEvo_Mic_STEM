"""Convert a material-neutral IdealSkeleton into tiled projected-site probes."""

from __future__ import annotations

import numpy as np

from ..context import IdealSkeleton, SkeletonNode
from .contracts import ProjectedSiteProbe


def _optional_site_endpoints(
    skeleton: IdealSkeleton,
    optional_node: SkeletonNode,
) -> tuple[SkeletonNode, SkeletonNode]:
    by_id = {node.site_id: node for node in skeleton.nodes}
    neighbors: list[SkeletonNode] = []
    for edge in skeleton.edges:
        if edge.source == optional_node.site_id:
            neighbors.append(by_id[edge.target])
        elif edge.target == optional_node.site_id:
            neighbors.append(by_id[edge.source])
    unique = {node.site_id: node for node in neighbors}
    if len(unique) != 2 or any(node.optional for node in unique.values()):
        raise ValueError(
            f"Optional node {optional_node.site_id!r} must connect to exactly "
            "two non-optional endpoint nodes"
        )
    values = tuple(unique.values())
    return values[0], values[1]


def projected_probes_from_skeleton(
    skeleton: IdealSkeleton,
    image_shape: tuple[int, int],
) -> tuple[ProjectedSiteProbe, ...]:
    """Tile optional skeleton nodes over an image using an explicit pixel basis."""

    skeleton.validate()
    height, width = image_shape
    if height < 1 or width < 1:
        raise ValueError("image_shape must be positive")
    # IdealSkeleton stores the two lattice vectors as rows; pixel positions use
    # basis_columns @ fractional_coordinate, matching the atomic task contract.
    basis = np.asarray(skeleton.basis, dtype=float).T
    if not np.all(np.isfinite(basis)) or abs(float(np.linalg.det(basis))) < 1e-8:
        raise ValueError("Skeleton basis must be finite and non-singular")
    origin = np.asarray(skeleton.metadata.get("origin_xy", (0.0, 0.0)), dtype=float)
    if origin.shape != (2,) or not np.all(np.isfinite(origin)):
        raise ValueError("skeleton.metadata.origin_xy must contain two finite values")

    optional_nodes = [node for node in skeleton.nodes if node.optional]
    if not optional_nodes:
        raise ValueError(
            "structure_guided_site requires at least one optional skeleton node"
        )
    inverse = np.linalg.inv(basis)
    corners = np.asarray(
        ((0.0, 0.0), (width - 1.0, 0.0), (0.0, height - 1.0), (width - 1.0, height - 1.0))
    )
    fractional_corners = np.asarray([inverse @ (corner - origin) for corner in corners])
    lower = np.floor(np.min(fractional_corners, axis=0)).astype(int) - 2
    upper = np.ceil(np.max(fractional_corners, axis=0)).astype(int) + 2

    probes: list[ProjectedSiteProbe] = []
    for optional_node in optional_nodes:
        endpoint_nodes = _optional_site_endpoints(skeleton, optional_node)
        center_fractional = np.asarray(optional_node.fractional_coordinate, dtype=float)
        endpoint_deltas = []
        for endpoint_node in endpoint_nodes:
            delta = np.asarray(endpoint_node.fractional_coordinate, dtype=float) - center_fractional
            delta = delta - np.round(delta)
            endpoint_deltas.append(delta)
        for cell_i in range(int(lower[0]), int(upper[0]) + 1):
            for cell_j in range(int(lower[1]), int(upper[1]) + 1):
                cell = np.asarray((cell_i, cell_j), dtype=float)
                center = origin + basis @ (cell + center_fractional)
                if not (0 <= center[0] < width and 0 <= center[1] < height):
                    continue
                endpoints = tuple(
                    tuple((center + basis @ delta).tolist())
                    for delta in endpoint_deltas
                )
                probes.append(
                    ProjectedSiteProbe(
                        probe_id=f"{optional_node.site_id}@{cell_i},{cell_j}",
                        family=optional_node.sublattice,
                        center_xy=tuple(center.tolist()),
                        endpoint_xy=endpoints,  # type: ignore[arg-type]
                        metadata={
                            "template_site_id": optional_node.site_id,
                            "cell_index": [cell_i, cell_j],
                        },
                    )
                )
    if not probes:
        raise ValueError("Skeleton projection produced no probes inside the image")
    return tuple(probes)
