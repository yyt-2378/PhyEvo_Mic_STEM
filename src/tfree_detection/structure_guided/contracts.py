"""Typed contracts for structure-guided projected-site analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class SampleRole(str, Enum):
    """Interpretation role assigned before looking at candidate scores."""

    TARGET_POSSIBLE = "target_possible"
    TARGET_FREE_REFERENCE = "target_free_reference"

    @classmethod
    def parse(cls, value: "SampleRole | str") -> "SampleRole":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as error:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown sample role {value!r}; choose one of {choices}"
            ) from error


@dataclass(frozen=True)
class ProjectedSiteProbe:
    """One expected empty/optional site bracketed by two template columns."""

    probe_id: str
    family: str
    center_xy: tuple[float, float]
    endpoint_xy: tuple[tuple[float, float], tuple[float, float]]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.probe_id:
            raise ValueError("ProjectedSiteProbe.probe_id cannot be empty")
        center = np.asarray(self.center_xy, dtype=float)
        endpoints = np.asarray(self.endpoint_xy, dtype=float)
        if center.shape != (2,):
            raise ValueError("ProjectedSiteProbe.center_xy must have shape (2,)")
        if endpoints.shape != (2, 2):
            raise ValueError(
                "ProjectedSiteProbe.endpoint_xy must have shape (2, 2)"
            )
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(endpoints)):
            raise ValueError("ProjectedSiteProbe coordinates must be finite")
        if float(np.linalg.norm(endpoints[1] - endpoints[0])) <= 0:
            raise ValueError("ProjectedSiteProbe endpoints must be distinct")


@dataclass(frozen=True)
class StructureGuidedPolicy:
    """Material-neutral model-selection and rendering policy."""

    minimum_delta_bic: float = 6.0
    profile_sigma_pixels: float = 1.25
    minimum_endpoint_separation_pixels: float = 4.0
    border_pixels: float = 4.0
    candidate_radius_pixels: float = 2.5

    def validate(self) -> None:
        if self.minimum_delta_bic < 0:
            raise ValueError("minimum_delta_bic cannot be negative")
        if self.profile_sigma_pixels <= 0:
            raise ValueError("profile_sigma_pixels must be positive")
        if self.minimum_endpoint_separation_pixels <= 0:
            raise ValueError(
                "minimum_endpoint_separation_pixels must be positive"
            )
        if self.border_pixels < 0:
            raise ValueError("border_pixels cannot be negative")
        if self.candidate_radius_pixels <= 0:
            raise ValueError("candidate_radius_pixels must be positive")


@dataclass(frozen=True)
class StructureGuidedAnalysis:
    """Measurements and policy-controlled interpretation for projected probes."""

    probes: Sequence[ProjectedSiteProbe]
    sample_role: SampleRole
    policy: StructureGuidedPolicy
    best_plane_index: np.ndarray
    delta_bic: np.ndarray
    center_amplitude: np.ndarray
    model_valid: np.ndarray
    interior: np.ndarray
    structural_positive: np.ndarray
    candidate: np.ndarray
    policy_suppressed: bool

    def __post_init__(self) -> None:
        expected = (len(self.probes),)
        arrays = {
            "best_plane_index": self.best_plane_index,
            "delta_bic": self.delta_bic,
            "center_amplitude": self.center_amplitude,
            "model_valid": self.model_valid,
            "interior": self.interior,
            "structural_positive": self.structural_positive,
            "candidate": self.candidate,
        }
        for name, value in arrays.items():
            if np.asarray(value).shape != expected:
                raise ValueError(
                    f"{name} has shape {np.asarray(value).shape}; expected {expected}"
                )
        if self.sample_role is SampleRole.TARGET_FREE_REFERENCE:
            if np.any(self.structural_positive) or np.any(self.candidate):
                raise ValueError(
                    "target_free_reference must expose zero structural positives "
                    "and zero candidates"
                )

    def summary(self) -> dict[str, Any]:
        """Return interpretation metadata without a false-positive audit pool."""

        payload: dict[str, Any] = {
            "sample_role": self.sample_role.value,
            "probe_count": len(self.probes),
            "model_valid_count": int(np.count_nonzero(self.model_valid)),
            "candidate_count": int(np.count_nonzero(self.candidate)),
            "minimum_delta_bic": float(self.policy.minimum_delta_bic),
            "target_absence_enforced": bool(self.policy_suppressed),
        }
        return payload
