"""Optional structural priors carried through every detection task."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SkeletonNode:
    site_id: str
    fractional_coordinate: Tuple[float, float]
    sublattice: str
    expected_species: Optional[str] = None
    expected_relative_intensity: Optional[float] = None
    optional: bool = False


@dataclass
class SkeletonEdge:
    source: str
    target: str
    kind: str = "coordination"


@dataclass
class IdealSkeleton:
    basis: np.ndarray
    nodes: List[SkeletonNode]
    edges: List[SkeletonEdge] = field(default_factory=list)
    symmetries: List[str] = field(default_factory=lambda: ["identity"])
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if np.asarray(self.basis).shape != (2, 2):
            raise ValueError("Skeleton basis must have shape (2, 2)")
        identifiers = [node.site_id for node in self.nodes]
        if not identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError("Skeleton node site_id values must be non-empty and unique")
        known = set(identifiers)
        for edge in self.edges:
            if edge.source not in known or edge.target not in known:
                raise ValueError(
                    f"Skeleton edge {edge.source!r}->{edge.target!r} references an unknown node"
                )

    @classmethod
    def from_json(cls, path: str | Path) -> "IdealSkeleton":
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        nodes = [
            SkeletonNode(
                site_id=str(node["site_id"]),
                fractional_coordinate=tuple(node["fractional_coordinate"]),
                sublattice=str(node["sublattice"]),
                expected_species=node.get("expected_species"),
                expected_relative_intensity=node.get(
                    "expected_relative_intensity"
                ),
                optional=bool(node.get("optional", False)),
            )
            for node in payload["nodes"]
        ]
        edges = [
            SkeletonEdge(
                source=str(edge["source"]),
                target=str(edge["target"]),
                kind=str(edge.get("kind", "coordination")),
            )
            for edge in payload.get("edges", [])
        ]
        skeleton = cls(
            basis=np.asarray(payload["basis"], dtype=float),
            nodes=nodes,
            edges=edges,
            symmetries=list(payload.get("symmetries", ["identity"])),
            metadata=dict(payload.get("metadata", {})),
        )
        skeleton.validate()
        return skeleton


@dataclass
class DetectionContext:
    skeleton: Optional[IdealSkeleton] = None
    exemplar_image: Optional[np.ndarray] = None

