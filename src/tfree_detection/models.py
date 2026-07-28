"""Shared data contracts used by tasks, feature branches, and reporters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from .evidence import EvidenceBundle


@dataclass
class FeatureGrid:
    """Feature values sampled on a regular grid of patch centres."""

    values: Dict[str, np.ndarray]
    centers_y: np.ndarray
    centers_x: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        first = next(iter(self.values.values()))
        return tuple(first.shape)  # type: ignore[return-value]

    def validate(self) -> None:
        expected = (len(self.centers_y), len(self.centers_x))
        if not self.values:
            raise ValueError("FeatureGrid must contain at least one feature")
        for name, value in self.values.items():
            if value.shape != expected:
                raise ValueError(
                    f"Feature {name!r} has shape {value.shape}; expected {expected}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Feature {name!r} contains non-finite values")


@dataclass
class DetectionRegion:
    region_id: int
    bbox_xyxy: Tuple[int, int, int, int]
    centroid_xy: Tuple[float, float]
    area_pixels: int
    mean_score: float
    max_score: float
    confidence: float
    confidence_level: str
    evidence_groups: List[str]
    defect_type: str = "local_amorphous_region"
    candidate_id: str = ""
    spatial_support: str = ""
    violation_types: List[str] = field(default_factory=list)
    reliability: float = 0.0
    decision_state: str = "uncertain_candidate"
    label_tier: str = "unresolved"
    evidence_vector: Dict[str, Dict[str, float]] = field(default_factory=dict)
    quality_flags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    resolved_family: str = ""
    resolved_label: str = ""
    epistemic_state: str = "unresolved"
    hypothesis_ranking: List[Dict[str, Any]] = field(default_factory=list)
    missing_auxiliary_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DetectionResult:
    task: str
    image_shape: Tuple[int, int]
    normalized_image: np.ndarray
    score_map: np.ndarray
    probability_map: np.ndarray
    mask: np.ndarray
    regions: List[DetectionRegion]
    feature_maps: Dict[str, np.ndarray]
    evidence_maps: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_bundle: "EvidenceBundle | None" = None
    reliability_map: np.ndarray | None = None
    decision_map: np.ndarray | None = None
    hypothesis_maps: Dict[str, np.ndarray] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        payload = {
            "task": self.task,
            "image_shape": list(self.image_shape),
            "detected_region_count": len(self.regions),
            "detected_area_pixels": int(self.mask.sum()),
            "detected_area_fraction": float(self.mask.mean()),
            "regions": [region.to_dict() for region in self.regions],
            "metadata": self.metadata,
        }
        if self.evidence_bundle is not None:
            payload["evidence_bundle"] = self.evidence_bundle.protocol_dict()
        return payload
