"""Unified, reliability-aware structural evidence protocol.

The task-specific detectors remain the numerical source of the first protocol
version.  This module gives their outputs shared scientific semantics without
silently changing masks or scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt

if TYPE_CHECKING:
    from .models import DetectionResult


EVIDENCE_PROTOCOL_VERSION = "2.0"
PLATFORM_VERSION = "0.4.0"


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class EvidenceScale(_StringEnum):
    NODE = "node"
    EDGE = "edge"
    REGION = "region"
    FIELD = "field"


class EvidenceRole(_StringEnum):
    ANOMALY = "anomaly"
    STRUCTURE = "structure"
    CONTEXT = "context"
    QUALITY_CONTROL = "quality_control"


class EvidenceDirection(_StringEnum):
    HIGHER_IS_MORE_ABNORMAL = "higher_is_more_abnormal"
    LOWER_IS_MORE_ABNORMAL = "lower_is_more_abnormal"
    ABSOLUTE_IS_MORE_ABNORMAL = "absolute_is_more_abnormal"
    DIAGNOSTIC_ONLY = "diagnostic_only"


class ViolationType(_StringEnum):
    OCCUPANCY = "occupancy"
    CHEMICAL_CONTRAST = "chemical-contrast"
    INTENSITY = "intensity"
    DISPLACEMENT = "displacement"
    SHAPE = "shape"
    PHASE_SLIP = "phase-slip"
    STRAIN = "strain"
    ROTATION = "rotation"
    COHERENCE_LOSS = "coherence-loss"
    PERIODICITY_LOSS = "periodicity-loss"
    SHORT_RANGE_ORDER_CHANGE = "short-range-order-change"
    DOMAIN_CHANGE = "domain-change"
    DIFFRACTION_SIGNATURE_CHANGE = "diffraction-signature-change"
    SURFACE_MORPHOLOGY = "surface-morphology"
    TEMPORAL_CHANGE = "temporal-change"
    ACQUISITION_DEGRADATION = "acquisition-degradation"
    SPECIMEN_SUPPORT = "specimen-support"


class SpatialSupport(_StringEnum):
    POINT = "point"
    LINE = "line"
    REGION = "region"
    INTERFACE = "interface"
    FIELD = "field"


class LabelTier(_StringEnum):
    GOLD = "gold"
    SILVER = "silver"
    UNRESOLVED = "unresolved"


class DecisionState(_StringEnum):
    BACKGROUND = "background"
    CONFIRMED_CANDIDATE = "confirmed_candidate"
    UNCERTAIN_CANDIDATE = "uncertain_candidate"
    OUT_OF_DOMAIN = "out_of_domain"
    INVALID_MEASUREMENT = "invalid_measurement"
    ARTIFACT = "artifact"
    UNRESOLVED = "unresolved"


DECISION_STATE_CODES: Dict[str, int] = {
    DecisionState.BACKGROUND.value: 0,
    DecisionState.CONFIRMED_CANDIDATE.value: 1,
    DecisionState.UNCERTAIN_CANDIDATE.value: 2,
    DecisionState.OUT_OF_DOMAIN.value: 3,
    DecisionState.INVALID_MEASUREMENT.value: 4,
    DecisionState.ARTIFACT.value: 5,
    DecisionState.UNRESOLVED.value: 6,
}


@dataclass(frozen=True)
class EvidenceDescriptor:
    """Scientific meaning of one spatial evidence field."""

    name: str
    scale: EvidenceScale
    violation_type: ViolationType
    role: EvidenceRole
    direction: EvidenceDirection
    units: str = "dimensionless"
    description: str = ""

    def to_dict(self) -> Dict[str, str]:
        return {
            "name": self.name,
            "scale": self.scale.value,
            "violation_type": self.violation_type.value,
            "role": self.role.value,
            "direction": self.direction.value,
            "units": self.units,
            "description": self.description,
        }


@dataclass
class EvidenceField:
    """Evidence values plus field-specific reliability and validity."""

    descriptor: EvidenceDescriptor
    values: np.ndarray
    reliability: np.ndarray
    valid_mask: np.ndarray
    source_key: str

    def validate(self, expected_shape: tuple[int, int]) -> None:
        for label, values in (
            ("values", self.values),
            ("reliability", self.reliability),
            ("valid_mask", self.valid_mask),
        ):
            if values.shape != expected_shape:
                raise ValueError(
                    f"Evidence {self.descriptor.name!r} {label} has shape "
                    f"{values.shape}; expected {expected_shape}"
                )
        if not np.all(np.isfinite(self.values[self.valid_mask])):
            raise ValueError(
                f"Evidence {self.descriptor.name!r} has non-finite valid values"
            )
        if not np.all(np.isfinite(self.reliability)):
            raise ValueError(
                f"Evidence {self.descriptor.name!r} reliability is non-finite"
            )
        if np.any(self.reliability < 0) or np.any(self.reliability > 1):
            raise ValueError(
                f"Evidence {self.descriptor.name!r} reliability must be in [0, 1]"
            )

    def protocol_dict(self) -> Dict[str, Any]:
        valid = self.valid_mask
        reliable_values = self.reliability[valid]
        return {
            **self.descriptor.to_dict(),
            "source_key": self.source_key,
            "shape": list(self.values.shape),
            "valid_fraction": float(np.mean(valid)),
            "mean_reliability_in_valid_area": (
                float(np.mean(reliable_values)) if reliable_values.size else 0.0
            ),
        }


@dataclass
class ReliabilityBundle:
    """Task-level measurement reliability, validity, and optional OOD state."""

    reliability_map: np.ndarray
    valid_mask: np.ndarray
    ood_map: np.ndarray
    state_map: np.ndarray
    reliable_threshold: float
    invalid_threshold: float
    sources: list[str] = field(default_factory=list)

    def validate(self, expected_shape: tuple[int, int]) -> None:
        for label, values in (
            ("reliability_map", self.reliability_map),
            ("valid_mask", self.valid_mask),
            ("ood_map", self.ood_map),
            ("state_map", self.state_map),
        ):
            if values.shape != expected_shape:
                raise ValueError(
                    f"Reliability {label} has shape {values.shape}; "
                    f"expected {expected_shape}"
                )
        if not np.all(np.isfinite(self.reliability_map)):
            raise ValueError("Reliability map contains non-finite values")
        if not np.all(np.isfinite(self.ood_map)):
            raise ValueError("OOD map contains non-finite values")
        if np.any(self.reliability_map < 0) or np.any(
            self.reliability_map > 1
        ):
            raise ValueError("Reliability map must be in [0, 1]")
        if np.any(self.ood_map < 0) or np.any(self.ood_map > 1):
            raise ValueError("OOD map must be in [0, 1]")
        if not 0 <= self.invalid_threshold < self.reliable_threshold <= 1:
            raise ValueError(
                "Reliability thresholds must satisfy "
                "0 <= invalid < reliable <= 1"
            )

    def summary(self) -> Dict[str, Any]:
        counts = {
            "reliable": int(np.sum(self.state_map == 0)),
            "uncertain": int(np.sum(self.state_map == 1)),
            "invalid": int(np.sum(self.state_map == 2)),
            "out_of_domain": int(np.sum(self.state_map == 3)),
        }
        return {
            "reliable_threshold": self.reliable_threshold,
            "invalid_threshold": self.invalid_threshold,
            "mean_reliability": float(np.mean(self.reliability_map)),
            "valid_fraction": float(np.mean(self.valid_mask)),
            "ood_fraction": float(np.mean(self.ood_map >= 0.5)),
            "state_counts": counts,
            "sources": list(self.sources),
        }


@dataclass(frozen=True)
class DefectHypothesis:
    """Sparse and auditable evidence definition for a defect hypothesis."""

    name: str
    violation_types: tuple[ViolationType, ...]
    spatial_support: SpatialSupport
    evidence_weights: Mapping[str, float]
    minimum_supported_fields: int = 1
    description: str = ""
    family: str = "open_set_resolution"
    resolution_kind: str = "physical"
    required_auxiliary_evidence: tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "violation_types": [item.value for item in self.violation_types],
            "spatial_support": self.spatial_support.value,
            "evidence_weights": dict(self.evidence_weights),
            "minimum_supported_fields": self.minimum_supported_fields,
            "description": self.description,
            "family": self.family,
            "resolution_kind": self.resolution_kind,
            "required_auxiliary_evidence": list(
                self.required_auxiliary_evidence
            ),
        }


@dataclass
class HypothesisScore:
    hypothesis: DefectHypothesis
    energy_map: np.ndarray
    support_count_map: np.ndarray
    available_fields: list[str]

    def summary(self) -> Dict[str, Any]:
        supported = self.support_count_map >= self.hypothesis.minimum_supported_fields
        return {
            **self.hypothesis.to_dict(),
            "available_fields": list(self.available_fields),
            "supported_fraction": float(np.mean(supported)),
            "energy_p95": float(np.percentile(self.energy_map, 95)),
            "energy_max": float(np.max(self.energy_map)),
        }


@dataclass
class EvidenceBundle:
    """Node-edge-region evidence graph for one detection result."""

    task: str
    image_shape: tuple[int, int]
    fields: Dict[str, EvidenceField]
    reliability: ReliabilityBundle
    hypotheses: Dict[str, HypothesisScore] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    protocol_version: str = EVIDENCE_PROTOCOL_VERSION

    def validate(self) -> None:
        if not self.fields:
            raise ValueError("EvidenceBundle must contain at least one field")
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("EvidenceBundle image_shape must be a positive 2-D shape")
        for name, evidence in self.fields.items():
            if name != evidence.descriptor.name:
                raise ValueError(
                    f"Evidence key {name!r} differs from descriptor name "
                    f"{evidence.descriptor.name!r}"
                )
            evidence.validate(self.image_shape)
        self.reliability.validate(self.image_shape)
        for name, score in self.hypotheses.items():
            if score.energy_map.shape != self.image_shape:
                raise ValueError(
                    f"Hypothesis {name!r} energy shape is invalid"
                )
            if score.support_count_map.shape != self.image_shape:
                raise ValueError(
                    f"Hypothesis {name!r} support shape is invalid"
                )
            if not np.all(np.isfinite(score.energy_map)):
                raise ValueError(
                    f"Hypothesis {name!r} energy contains non-finite values"
                )

    def protocol_dict(self) -> Dict[str, Any]:
        scales = {
            scale.value: sorted(
                name
                for name, evidence in self.fields.items()
                if evidence.descriptor.scale == scale
            )
            for scale in EvidenceScale
        }
        return {
            "protocol_version": self.protocol_version,
            "task": self.task,
            "image_shape": list(self.image_shape),
            "scientific_representation": "structural_evidence_graph",
            "fields": {
                name: evidence.protocol_dict()
                for name, evidence in sorted(self.fields.items())
            },
            "fields_by_scale": scales,
            "reliability": self.reliability.summary(),
            "hypotheses": {
                name: score.summary()
                for name, score in sorted(self.hypotheses.items())
            },
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class _LegacyMapping:
    source_key: str
    descriptor: EvidenceDescriptor


def _descriptor(
    name: str,
    scale: EvidenceScale,
    violation: ViolationType,
    role: EvidenceRole,
    direction: EvidenceDirection,
    description: str,
    units: str = "dimensionless",
) -> EvidenceDescriptor:
    return EvidenceDescriptor(
        name=name,
        scale=scale,
        violation_type=violation,
        role=role,
        direction=direction,
        units=units,
        description=description,
    )


_MAPPINGS: Dict[str, tuple[_LegacyMapping, ...]] = {
    "amorphous_region": (
        _LegacyMapping(
            "group_spectrum",
            _descriptor(
                "region.fourier_periodicity_loss",
                EvidenceScale.REGION,
                ViolationType.PERIODICITY_LOSS,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Combined local Fourier entropy, flatness, and peak-loss evidence.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_patterson",
            _descriptor(
                "region.patterson_order_loss",
                EvidenceScale.REGION,
                ViolationType.PERIODICITY_LOSS,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Loss of repeated Patterson autocorrelation order.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_sharpness",
            _descriptor(
                "region.sharpness_loss",
                EvidenceScale.REGION,
                ViolationType.COHERENCE_LOSS,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Loss of local high-frequency/gradient sharpness.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "support_count",
            _descriptor(
                "region.spatial_sampling_support",
                EvidenceScale.REGION,
                ViolationType.SPECIMEN_SUPPORT,
                EvidenceRole.QUALITY_CONTROL,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Number of independent anomaly evidence groups above support threshold.",
                "evidence_groups",
            ),
        ),
        _LegacyMapping(
            "gpa_strain_magnitude",
            _descriptor(
                "edge.gpa_strain_magnitude",
                EvidenceScale.EDGE,
                ViolationType.STRAIN,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "GPA deformation magnitude around the candidate.",
                "relative_strain",
            ),
        ),
        _LegacyMapping(
            "gpa_lattice_coherence",
            _descriptor(
                "edge.gpa_lattice_coherence",
                EvidenceScale.EDGE,
                ViolationType.COHERENCE_LOSS,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.LOWER_IS_MORE_ABNORMAL,
                "Reliability and crystalline coherence of the GPA carrier.",
            ),
        ),
        *tuple(
            _LegacyMapping(
                source_key,
                _descriptor(
                    canonical,
                    EvidenceScale.EDGE,
                    violation,
                    EvidenceRole.ANOMALY,
                    EvidenceDirection.ABSOLUTE_IS_MORE_ABNORMAL,
                    description,
                    units,
                ),
            )
            for source_key, canonical, violation, description, units in (
                (
                    "gpa_exx",
                    "edge.gpa_exx",
                    ViolationType.STRAIN,
                    "GPA normal strain along x.",
                    "relative_strain",
                ),
                (
                    "gpa_eyy",
                    "edge.gpa_eyy",
                    ViolationType.STRAIN,
                    "GPA normal strain along y.",
                    "relative_strain",
                ),
                (
                    "gpa_exy",
                    "edge.gpa_exy",
                    ViolationType.STRAIN,
                    "GPA shear strain.",
                    "relative_strain",
                ),
                (
                    "gpa_rotation",
                    "edge.gpa_rotation",
                    ViolationType.ROTATION,
                    "GPA local lattice rotation.",
                    "radians",
                ),
                (
                    "gpa_principal_tensile",
                    "edge.gpa_principal_tensile",
                    ViolationType.STRAIN,
                    "Principal tensile strain.",
                    "relative_strain",
                ),
                (
                    "gpa_principal_compressive",
                    "edge.gpa_principal_compressive",
                    ViolationType.STRAIN,
                    "Principal compressive strain.",
                    "relative_strain",
                ),
                (
                    "gpa_basis1_length_change",
                    "edge.gpa_basis1_length_change",
                    ViolationType.STRAIN,
                    "Relative length change of direct-lattice basis 1.",
                    "relative_change",
                ),
                (
                    "gpa_basis2_length_change",
                    "edge.gpa_basis2_length_change",
                    ViolationType.STRAIN,
                    "Relative length change of direct-lattice basis 2.",
                    "relative_change",
                ),
            )
        ),
    ),
    "atomic_point_defect": (
        _LegacyMapping(
            "group_presence",
            _descriptor(
                "node.presence_residual",
                EvidenceScale.NODE,
                ViolationType.OCCUPANCY,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Missing or weak atom-column response at an expected site.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_intensity",
            _descriptor(
                "node.intensity_residual",
                EvidenceScale.NODE,
                ViolationType.INTENSITY,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Sublattice-conditioned atom-column intensity residual.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_displacement",
            _descriptor(
                "node.displacement_residual",
                EvidenceScale.NODE,
                ViolationType.DISPLACEMENT,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Residual displacement after smooth non-rigid registration.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_extra_peak",
            _descriptor(
                "node.extra_peak_residual",
                EvidenceScale.NODE,
                ViolationType.OCCUPANCY,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Observed atomic peak without a matched expected site.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "expected_lattice_sites",
            _descriptor(
                "node.expected_lattice_sites",
                EvidenceScale.NODE,
                ViolationType.OCCUPANCY,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Rendered sites from the registered reference lattice.",
                "binary",
            ),
        ),
        _LegacyMapping(
            "observed_atomic_peaks",
            _descriptor(
                "node.observed_atomic_peaks",
                EvidenceScale.NODE,
                ViolationType.OCCUPANCY,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Observed atom-column peak locations.",
                "binary",
            ),
        ),
        _LegacyMapping(
            "registration_local_match_fraction",
            _descriptor(
                "node.registration_match_fraction",
                EvidenceScale.NODE,
                ViolationType.DISPLACEMENT,
                EvidenceRole.QUALITY_CONTROL,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Local expected-to-observed lattice match fraction.",
                "fraction",
            ),
        ),
        _LegacyMapping(
            "registration_local_residual",
            _descriptor(
                "node.registration_residual",
                EvidenceScale.NODE,
                ViolationType.DISPLACEMENT,
                EvidenceRole.QUALITY_CONTROL,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Local registration residual used for reliability gating.",
                "pixels",
            ),
        ),
    ),
    "stacking_fault": (
        _LegacyMapping(
            "group_phase_jump",
            _descriptor(
                "edge.phase_jump",
                EvidenceScale.EDGE,
                ViolationType.PHASE_SLIP,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Abrupt lattice phase-gradient evidence.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_phase_curvature",
            _descriptor(
                "edge.phase_curvature",
                EvidenceScale.EDGE,
                ViolationType.PHASE_SLIP,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Curvature of local lattice phase gradients.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_shear_rotation",
            _descriptor(
                "edge.shear_rotation",
                EvidenceScale.EDGE,
                ViolationType.ROTATION,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Combined shear and rotation anomaly.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_coherence_loss",
            _descriptor(
                "edge.coherence_loss",
                EvidenceScale.EDGE,
                ViolationType.COHERENCE_LOSS,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Loss of complex-demodulation lattice coherence.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "lattice_coherence",
            _descriptor(
                "edge.lattice_coherence",
                EvidenceScale.EDGE,
                ViolationType.COHERENCE_LOSS,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.LOWER_IS_MORE_ABNORMAL,
                "Local coherence of the two reciprocal-lattice carriers.",
            ),
        ),
        _LegacyMapping(
            "strain_magnitude",
            _descriptor(
                "edge.strain_magnitude",
                EvidenceScale.EDGE,
                ViolationType.STRAIN,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Strain magnitude derived from phase gradients.",
                "relative_strain",
            ),
        ),
        _LegacyMapping(
            "topology_skeleton",
            _descriptor(
                "edge.topology_skeleton",
                EvidenceScale.EDGE,
                ViolationType.PHASE_SLIP,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Skeleton support used to describe line-candidate topology.",
                "binary",
            ),
        ),
    ),
    "phase_boundary": (
        _LegacyMapping(
            "group_reciprocal_signature",
            _descriptor(
                "edge.domain_signature_transition",
                EvidenceScale.EDGE,
                ViolationType.DOMAIN_CHANGE,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Transition between two local reciprocal-texture prototypes.",
                "threshold_ratio",
            ),
        ),
        _LegacyMapping(
            "domain_label",
            _descriptor(
                "region.domain_label",
                EvidenceScale.REGION,
                ViolationType.DOMAIN_CHANGE,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Per-image reciprocal-texture domain identifier.",
                "categorical",
            ),
        ),
        _LegacyMapping(
            "domain_membership_margin",
            _descriptor(
                "region.domain_membership_margin",
                EvidenceScale.REGION,
                ViolationType.DOMAIN_CHANGE,
                EvidenceRole.QUALITY_CONTROL,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Distance margin to the alternative domain prototype.",
            ),
        ),
        _LegacyMapping(
            "crystalline_support",
            _descriptor(
                "region.crystalline_support",
                EvidenceScale.REGION,
                ViolationType.SPECIMEN_SUPPORT,
                EvidenceRole.QUALITY_CONTROL,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Support that the pixel belongs to a crystalline specimen.",
                "fraction",
            ),
        ),
        _LegacyMapping(
            "local_contrast",
            _descriptor(
                "region.local_contrast",
                EvidenceScale.REGION,
                ViolationType.SPECIMEN_SUPPORT,
                EvidenceRole.CONTEXT,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Local contrast used to distinguish specimen from vacuum.",
            ),
        ),
        _LegacyMapping(
            "topology_skeleton",
            _descriptor(
                "edge.interface_topology_skeleton",
                EvidenceScale.EDGE,
                ViolationType.DOMAIN_CHANGE,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Skeleton support used to describe interface topology.",
                "binary",
            ),
        ),
    ),
    "symmetry_relation": (
        _LegacyMapping(
            "group_phase_motif_residual",
            _descriptor(
                "region.group_phase_motif_residual",
                EvidenceScale.REGION,
                ViolationType.DOMAIN_CHANGE,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Identity residual explained by a non-identity lattice-group action.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_orientation_strain",
            _descriptor(
                "edge.group_orientation_strain",
                EvidenceScale.EDGE,
                ViolationType.STRAIN,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Local orientation/strain departure from the reference patch ensemble.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_displacement_discontinuity",
            _descriptor(
                "edge.displacement_discontinuity",
                EvidenceScale.EDGE,
                ViolationType.PHASE_SLIP,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Abrupt curvature of the reference-corrected carrier displacement.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_burgers_loop_inconsistency",
            _descriptor(
                "node.burgers_loop_inconsistency",
                EvidenceScale.NODE,
                ViolationType.DISPLACEMENT,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Curl/topology proxy for non-closing lattice displacement loops.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "group_scan_time_artifact",
            _descriptor(
                "field.scan_time_artifact",
                EvidenceScale.FIELD,
                ViolationType.ACQUISITION_DEGRADATION,
                EvidenceRole.ANOMALY,
                EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
                "Scan-axis-dominant deformation unexplained by a lattice-group relation.",
                "robust_z",
            ),
        ),
        _LegacyMapping(
            "lattice_coherence",
            _descriptor(
                "field.lattice_coherence",
                EvidenceScale.FIELD,
                ViolationType.COHERENCE_LOSS,
                EvidenceRole.STRUCTURE,
                EvidenceDirection.LOWER_IS_MORE_ABNORMAL,
                "Two-carrier lattice coherence used for measurement reliability.",
            ),
        ),
        _LegacyMapping(
            "symmetry_relation_validity",
            _descriptor(
                "field.symmetry_relation_validity",
                EvidenceScale.FIELD,
                ViolationType.SPECIMEN_SUPPORT,
                EvidenceRole.QUALITY_CONTROL,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Combined coherence and edge-context measurement reliability.",
            ),
        ),
        _LegacyMapping(
            "reference_patch_support",
            _descriptor(
                "field.reference_patch_support",
                EvidenceScale.FIELD,
                ViolationType.SPECIMEN_SUPPORT,
                EvidenceRole.QUALITY_CONTROL,
                EvidenceDirection.DIAGNOSTIC_ONLY,
                "Spatial support of automatically selected normal references.",
            ),
        ),
    ),
}

_MAPPINGS["structure_guided_site"] = (
    _LegacyMapping(
        "group_structural_model_gain",
        _descriptor(
            "node.template_model_gain",
            EvidenceScale.NODE,
            ViolationType.OCCUPANCY,
            EvidenceRole.ANOMALY,
            EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
            "BIC gain of an endpoint-plus-center model over the template-only model.",
            "delta_bic",
        ),
    ),
    _LegacyMapping(
        "group_center_component",
        _descriptor(
            "node.extra_center_component",
            EvidenceScale.NODE,
            ViolationType.CHEMICAL_CONTRAST,
            EvidenceRole.ANOMALY,
            EvidenceDirection.HIGHER_IS_MORE_ABNORMAL,
            "Positive fitted component at an optional projected template site.",
            "robust_amplitude",
        ),
    ),
    _LegacyMapping(
        "structure_probe_support",
        _descriptor(
            "node.projected_template_support",
            EvidenceScale.NODE,
            ViolationType.OCCUPANCY,
            EvidenceRole.STRUCTURE,
            EvidenceDirection.DIAGNOSTIC_ONLY,
            "Sites where the projected structural probe is geometrically valid.",
            "binary",
        ),
    ),
)


DEFAULT_HYPOTHESES: Dict[str, DefectHypothesis] = {
    "amorphous_region": DefectHypothesis(
        name="local_amorphous_region",
        violation_types=(
            ViolationType.PERIODICITY_LOSS,
            ViolationType.COHERENCE_LOSS,
        ),
        spatial_support=SpatialSupport.REGION,
        evidence_weights={
            "region.fourier_periodicity_loss": 1.0,
            "region.patterson_order_loss": 1.2,
            "region.sharpness_loss": 0.7,
        },
        minimum_supported_fields=2,
        description="Regional failure of periodic and coherent lattice order.",
        family="order_and_coherence_degradation",
    ),
    "atomic_point_defect": DefectHypothesis(
        name="atomic_point_defect",
        violation_types=(
            ViolationType.OCCUPANCY,
            ViolationType.INTENSITY,
            ViolationType.DISPLACEMENT,
        ),
        spatial_support=SpatialSupport.POINT,
        evidence_weights={
            "node.presence_residual": 1.0,
            "node.intensity_residual": 0.8,
            "node.displacement_residual": 1.0,
            "node.extra_peak_residual": 1.0,
        },
        minimum_supported_fields=1,
        description="Point-supported violation of an expected atom-column site.",
        family="atomic_site_anomaly",
        resolution_kind="fallback",
    ),
    "stacking_fault": DefectHypothesis(
        name="stacking_fault_or_dislocation_line",
        violation_types=(
            ViolationType.PHASE_SLIP,
            ViolationType.ROTATION,
            ViolationType.COHERENCE_LOSS,
        ),
        spatial_support=SpatialSupport.LINE,
        evidence_weights={
            "edge.phase_jump": 1.0,
            "edge.phase_curvature": 0.8,
            "edge.shear_rotation": 1.0,
            "edge.coherence_loss": 0.7,
        },
        minimum_supported_fields=2,
        description="Line-supported phase slip with deformation evidence.",
        family="lattice_deformation_and_topology",
        resolution_kind="fallback",
    ),
    "phase_boundary": DefectHypothesis(
        name="two_phase_or_twin_boundary",
        violation_types=(ViolationType.DOMAIN_CHANGE,),
        spatial_support=SpatialSupport.INTERFACE,
        evidence_weights={"edge.domain_signature_transition": 1.0},
        minimum_supported_fields=1,
        description="Interface-supported change between reciprocal-texture domains.",
        family="domain_symmetry_and_interface",
        resolution_kind="fallback",
    ),
    "symmetry_relation": DefectHypothesis(
        name="symmetry_relation_anomaly",
        violation_types=(
            ViolationType.DOMAIN_CHANGE,
            ViolationType.STRAIN,
            ViolationType.PHASE_SLIP,
            ViolationType.DISPLACEMENT,
        ),
        spatial_support=SpatialSupport.REGION,
        evidence_weights={
            "region.group_phase_motif_residual": 1.0,
            "edge.group_orientation_strain": 0.8,
            "edge.displacement_discontinuity": 1.0,
            "node.burgers_loop_inconsistency": 0.8,
        },
        minimum_supported_fields=1,
        description="Group-related domain, interface, phase slip, or core proxy.",
        family="domain_symmetry_and_interface",
        resolution_kind="fallback",
    ),
}

DEFAULT_HYPOTHESES["structure_guided_site"] = DefectHypothesis(
    name="unexpected_template_site_occupancy",
    violation_types=(
        ViolationType.OCCUPANCY,
        ViolationType.CHEMICAL_CONTRAST,
    ),
    spatial_support=SpatialSupport.POINT,
    evidence_weights={
        "node.template_model_gain": 1.0,
        "node.extra_center_component": 0.7,
    },
    minimum_supported_fields=1,
    description=(
        "An optional or nominally empty projected template site is better "
        "explained by an added center component."
    ),
    family="projected_structure_site_anomaly",
)


OPEN_WORLD_HYPOTHESES: Dict[str, tuple[DefectHypothesis, ...]] = {
    "amorphous_region": (
        DEFAULT_HYPOTHESES["amorphous_region"],
        DefectHypothesis(
            name="partial_disordering",
            violation_types=(
                ViolationType.PERIODICITY_LOSS,
                ViolationType.SHORT_RANGE_ORDER_CHANGE,
            ),
            spatial_support=SpatialSupport.REGION,
            evidence_weights={
                "region.fourier_periodicity_loss": 0.8,
                "region.patterson_order_loss": 1.0,
            },
            minimum_supported_fields=1,
            description=(
                "Regional order loss that may retain partial lattice coherence."
            ),
            family="order_and_coherence_degradation",
        ),
        DefectHypothesis(
            name="local_strain_or_deformation_region",
            violation_types=(
                ViolationType.STRAIN,
                ViolationType.ROTATION,
                ViolationType.DISPLACEMENT,
            ),
            spatial_support=SpatialSupport.REGION,
            evidence_weights={
                "edge.gpa_strain_magnitude": 1.0,
                "edge.gpa_exx": 0.5,
                "edge.gpa_eyy": 0.5,
                "edge.gpa_exy": 0.7,
                "edge.gpa_rotation": 0.7,
            },
            minimum_supported_fields=2,
            description=(
                "A deformed crystalline neighborhood rather than complete "
                "loss of atomic order."
            ),
            family="lattice_deformation_and_topology",
        ),
        DefectHypothesis(
            name="defocus_or_blur",
            violation_types=(
                ViolationType.ACQUISITION_DEGRADATION,
                ViolationType.COHERENCE_LOSS,
            ),
            spatial_support=SpatialSupport.REGION,
            evidence_weights={"region.sharpness_loss": 1.0},
            minimum_supported_fields=1,
            description=(
                "Sharpness loss without sufficient independent order-loss "
                "evidence; treated as an acquisition-artifact hypothesis."
            ),
            family="acquisition_artifact_and_quality",
            resolution_kind="artifact",
            required_auxiliary_evidence=(
                "through_focus_or_repeat_scan",
                "reciprocal_space_consistency",
            ),
        ),
    ),
    "atomic_point_defect": (
        DefectHypothesis(
            name="vacancy_like",
            violation_types=(
                ViolationType.OCCUPANCY,
                ViolationType.INTENSITY,
            ),
            spatial_support=SpatialSupport.POINT,
            evidence_weights={
                "node.presence_residual": 1.0,
                "node.intensity_residual": 0.6,
            },
            minimum_supported_fields=1,
            description="Missing or weak expected atom-column response.",
            family="atomic_site_anomaly",
            required_auxiliary_evidence=(
                "thickness_or_forward_model",
                "chemical_identity_if_available",
            ),
        ),
        DefectHypothesis(
            name="substitution_or_intensity_anomaly",
            violation_types=(
                ViolationType.CHEMICAL_CONTRAST,
                ViolationType.INTENSITY,
            ),
            spatial_support=SpatialSupport.POINT,
            evidence_weights={"node.intensity_residual": 1.0},
            minimum_supported_fields=1,
            description=(
                "Sublattice-conditioned intensity anomaly; substitution is "
                "only one possible cause."
            ),
            family="atomic_site_anomaly",
            required_auxiliary_evidence=(
                "spectroscopy_or_validated_forward_model",
            ),
        ),
        DefectHypothesis(
            name="interstitial_or_adatom_like",
            violation_types=(ViolationType.OCCUPANCY,),
            spatial_support=SpatialSupport.POINT,
            evidence_weights={"node.extra_peak_residual": 1.0},
            minimum_supported_fields=1,
            description="Observed atomic peak without a matched expected site.",
            family="atomic_site_anomaly",
        ),
        DefectHypothesis(
            name="displaced_atom_column",
            violation_types=(
                ViolationType.DISPLACEMENT,
                ViolationType.STRAIN,
            ),
            spatial_support=SpatialSupport.POINT,
            evidence_weights={"node.displacement_residual": 1.0},
            minimum_supported_fields=1,
            description=(
                "Local displacement remaining after smooth non-rigid "
                "registration."
            ),
            family="atomic_site_anomaly",
        ),
        DEFAULT_HYPOTHESES["atomic_point_defect"],
    ),
    "stacking_fault": (
        DefectHypothesis(
            name="stacking_fault",
            violation_types=(
                ViolationType.PHASE_SLIP,
                ViolationType.COHERENCE_LOSS,
            ),
            spatial_support=SpatialSupport.LINE,
            evidence_weights={
                "edge.phase_jump": 1.0,
                "edge.phase_curvature": 0.8,
                "edge.coherence_loss": 0.3,
            },
            minimum_supported_fields=2,
            description="Line-supported phase discontinuity.",
            family="lattice_deformation_and_topology",
        ),
        DefectHypothesis(
            name="dislocation_core_or_line",
            violation_types=(
                ViolationType.PHASE_SLIP,
                ViolationType.ROTATION,
                ViolationType.STRAIN,
            ),
            spatial_support=SpatialSupport.LINE,
            evidence_weights={
                "edge.phase_jump": 0.7,
                "edge.shear_rotation": 1.0,
                "edge.strain_magnitude": 0.8,
            },
            minimum_supported_fields=2,
            description=(
                "Phase slip accompanied by localized rotation or strain."
            ),
            family="lattice_deformation_and_topology",
            required_auxiliary_evidence=("burgers_circuit_or_atom_graph",),
        ),
        DefectHypothesis(
            name="lattice_rotation_or_bending",
            violation_types=(
                ViolationType.ROTATION,
                ViolationType.STRAIN,
            ),
            spatial_support=SpatialSupport.LINE,
            evidence_weights={
                "edge.shear_rotation": 1.0,
                "edge.strain_magnitude": 0.8,
            },
            minimum_supported_fields=1,
            description=(
                "Rotation/strain line without decisive stacking sequence "
                "evidence."
            ),
            family="lattice_deformation_and_topology",
        ),
        DefectHypothesis(
            name="scan_or_coherence_artifact",
            violation_types=(
                ViolationType.ACQUISITION_DEGRADATION,
                ViolationType.COHERENCE_LOSS,
            ),
            spatial_support=SpatialSupport.LINE,
            evidence_weights={
                "edge.coherence_loss": 1.0,
                "edge.lattice_coherence": 0.7,
            },
            minimum_supported_fields=2,
            description=(
                "Line-like coherence loss lacking independent phase-slip "
                "support."
            ),
            family="acquisition_artifact_and_quality",
            resolution_kind="artifact",
            required_auxiliary_evidence=(
                "orthogonal_scan_or_repeat_frame",
            ),
        ),
        DEFAULT_HYPOTHESES["stacking_fault"],
    ),
    "phase_boundary": (
        DefectHypothesis(
            name="heterophase_interface",
            violation_types=(
                ViolationType.DOMAIN_CHANGE,
                ViolationType.DIFFRACTION_SIGNATURE_CHANGE,
            ),
            spatial_support=SpatialSupport.INTERFACE,
            evidence_weights={"edge.domain_signature_transition": 1.0},
            minimum_supported_fields=1,
            description="Boundary between distinct local reciprocal signatures.",
            family="domain_symmetry_and_interface",
            required_auxiliary_evidence=(
                "phase_identity_or_multimodal_confirmation",
            ),
        ),
        DefectHypothesis(
            name="twin_or_orientation_domain_boundary",
            violation_types=(
                ViolationType.DOMAIN_CHANGE,
                ViolationType.ROTATION,
            ),
            spatial_support=SpatialSupport.INTERFACE,
            evidence_weights={"edge.domain_signature_transition": 1.0},
            minimum_supported_fields=1,
            description=(
                "Twin or orientation-domain interpretation of a reciprocal "
                "signature boundary."
            ),
            family="domain_symmetry_and_interface",
            required_auxiliary_evidence=(
                "symmetry_relation_or_orientation_indexing",
            ),
        ),
        DEFAULT_HYPOTHESES["phase_boundary"],
    ),
    "symmetry_relation": (
        DefectHypothesis(
            name="translation_or_sublattice_domain",
            violation_types=(ViolationType.DOMAIN_CHANGE,),
            spatial_support=SpatialSupport.REGION,
            evidence_weights={
                "region.group_phase_motif_residual": 1.0,
            },
            description="Region related to the reference lattice by translation.",
            family="domain_symmetry_and_interface",
        ),
        DefectHypothesis(
            name="orientation_twin_or_mirror_domain",
            violation_types=(
                ViolationType.DOMAIN_CHANGE,
                ViolationType.ROTATION,
                ViolationType.STRAIN,
            ),
            spatial_support=SpatialSupport.REGION,
            evidence_weights={
                "region.group_phase_motif_residual": 0.8,
                "edge.group_orientation_strain": 1.0,
            },
            minimum_supported_fields=2,
            description="Rotation, mirror, twin, or strained symmetry domain.",
            family="domain_symmetry_and_interface",
        ),
        DefectHypothesis(
            name="lattice_phase_slip",
            violation_types=(ViolationType.PHASE_SLIP,),
            spatial_support=SpatialSupport.LINE,
            evidence_weights={
                "edge.displacement_discontinuity": 1.0,
                "region.group_phase_motif_residual": 0.5,
            },
            description="Line-supported discontinuity in lattice displacement.",
            family="lattice_deformation_and_topology",
        ),
        DefectHypothesis(
            name="dislocation_core_proxy",
            violation_types=(
                ViolationType.DISPLACEMENT,
                ViolationType.PHASE_SLIP,
            ),
            spatial_support=SpatialSupport.POINT,
            evidence_weights={
                "node.burgers_loop_inconsistency": 1.0,
                "edge.displacement_discontinuity": 0.7,
            },
            minimum_supported_fields=2,
            description="Endpoint-localized non-closing displacement-loop proxy.",
            family="lattice_deformation_and_topology",
            required_auxiliary_evidence=("atom_graph_burgers_circuit",),
        ),
        DefectHypothesis(
            name="scan_or_time_artifact",
            violation_types=(ViolationType.ACQUISITION_DEGRADATION,),
            spatial_support=SpatialSupport.FIELD,
            evidence_weights={"field.scan_time_artifact": 1.0},
            description="Scan-axis deformation not explained by a group relation.",
            family="acquisition_artifact_and_quality",
            resolution_kind="artifact",
            required_auxiliary_evidence=("repeat_or_orthogonal_scan",),
        ),
        DEFAULT_HYPOTHESES["symmetry_relation"],
    ),
}

OPEN_WORLD_HYPOTHESES["structure_guided_site"] = (
    DEFAULT_HYPOTHESES["structure_guided_site"],
)


def hypotheses_for_task(task: str) -> tuple[DefectHypothesis, ...]:
    try:
        return OPEN_WORLD_HYPOTHESES[task]
    except KeyError as error:
        raise ValueError(f"No hypotheses registered for task {task!r}") from error


def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)


def _task_reliability(
    task: str,
    evidence_maps: Mapping[str, np.ndarray],
    shape: tuple[int, int],
    reliable_threshold: float,
    invalid_threshold: float,
    metadata: Mapping[str, Any] | None = None,
) -> ReliabilityBundle:
    sources: list[str] = []
    explicit_valid = np.ones(shape, dtype=bool)
    reliability = np.ones(shape, dtype=np.float32)

    if task == "amorphous_region":
        margin = int((metadata or {}).get("valid_border_margin_pixels", 0))
        margin = int(np.clip(margin, 0, min(shape) // 3))
        if margin:
            explicit_valid[:margin, :] = False
            explicit_valid[-margin:, :] = False
            explicit_valid[:, :margin] = False
            explicit_valid[:, -margin:] = False
            distance = distance_transform_edt(explicit_valid)
            ramp_width = max(min(margin / 2.0, 32.0), 1.0)
            reliability = _clip01(distance / ramp_width)
            reliability *= explicit_valid
            sources.append("fourier_valid_border_and_window_context")
        else:
            sources.append("full_image_window_context")
    elif task == "atomic_point_defect":
        registration = evidence_maps.get("registration_reliability")
        if registration is not None:
            reliability = _clip01(registration)
            explicit_valid = reliability > 0
            sources.append("registration_reliability")
    elif task == "structure_guided_site":
        probe_reliability = evidence_maps.get("structure_probe_reliability")
        if probe_reliability is not None:
            reliability = _clip01(probe_reliability)
            explicit_valid = reliability > 0
            sources.append("structure_probe_reliability")
    elif task == "stacking_fault":
        validity = evidence_maps.get("phase_slip_validity")
        coherence = evidence_maps.get("lattice_coherence")
        if validity is not None:
            explicit_valid = validity >= 0.5
            sources.append("phase_slip_validity")
        if coherence is not None:
            reliability = _clip01(coherence)
            sources.append("lattice_coherence")
        reliability = reliability * explicit_valid
    elif task == "phase_boundary":
        validity = evidence_maps.get("boundary_validity")
        crystalline = evidence_maps.get("crystalline_support")
        if validity is not None:
            explicit_valid = validity >= 0.5
            sources.append("boundary_validity")
        if crystalline is not None:
            reliability = _clip01(crystalline)
            sources.append("crystalline_support")
        reliability = reliability * explicit_valid
    elif task == "symmetry_relation":
        validity = evidence_maps.get("symmetry_relation_validity")
        coherence = evidence_maps.get("lattice_coherence")
        if validity is not None:
            reliability = _clip01(validity)
            explicit_valid = reliability > 0
            sources.append("symmetry_relation_validity")
        if coherence is not None:
            reliability *= np.sqrt(_clip01(coherence))
            sources.append("lattice_coherence")

    ood = np.zeros(shape, dtype=np.float32)
    state = np.ones(shape, dtype=np.uint8)
    state[reliability >= reliable_threshold] = 0
    state[(~explicit_valid) | (reliability < invalid_threshold)] = 2
    state[ood >= 0.5] = 3
    return ReliabilityBundle(
        reliability_map=reliability.astype(np.float32),
        valid_mask=explicit_valid,
        ood_map=ood,
        state_map=state,
        reliable_threshold=reliable_threshold,
        invalid_threshold=invalid_threshold,
        sources=sources or ["implicit_full_image_support"],
    )


def _field_reliability(
    task: str,
    source_key: str,
    evidence_maps: Mapping[str, np.ndarray],
    task_reliability: ReliabilityBundle,
) -> tuple[np.ndarray, np.ndarray]:
    reliability = task_reliability.reliability_map.copy()
    valid = task_reliability.valid_mask.copy()
    if source_key.startswith("gpa_"):
        validity = evidence_maps.get("gpa_validity")
        coherence = evidence_maps.get("gpa_lattice_coherence")
        if validity is not None:
            valid &= validity >= 0.5
        if coherence is not None:
            reliability = reliability * _clip01(coherence)
        reliability *= valid
    return reliability.astype(np.float32), valid


def _robust_anomaly(values: np.ndarray, field: EvidenceField) -> np.ndarray:
    direction = field.descriptor.direction
    if direction == EvidenceDirection.DIAGNOSTIC_ONLY:
        return np.zeros_like(values, dtype=np.float32)
    transformed = np.asarray(values, dtype=np.float32)
    if direction == EvidenceDirection.LOWER_IS_MORE_ABNORMAL:
        transformed = -transformed
    elif direction == EvidenceDirection.ABSOLUTE_IS_MORE_ABNORMAL:
        transformed = np.abs(transformed)

    reference = transformed[field.valid_mask]
    if not reference.size:
        return np.zeros_like(transformed, dtype=np.float32)
    if float(np.min(reference)) >= 0 and float(np.percentile(reference, 95)) > 1:
        return np.clip(transformed, 0, 12).astype(np.float32)
    center = float(np.median(reference))
    scale = float(1.4826 * np.median(np.abs(reference - center)))
    if scale < 1e-8:
        scale = max(float(np.std(reference)), 1e-8)
    return np.clip((transformed - center) / scale, 0, 12).astype(np.float32)


def score_hypothesis(
    bundle: EvidenceBundle, hypothesis: DefectHypothesis
) -> HypothesisScore:
    """Compute transparent robust evidence energy without learned weights."""

    numerator = np.zeros(bundle.image_shape, dtype=np.float32)
    denominator = np.zeros(bundle.image_shape, dtype=np.float32)
    support_count = np.zeros(bundle.image_shape, dtype=np.uint8)
    available: list[str] = []
    for field_name, weight in hypothesis.evidence_weights.items():
        field = bundle.fields.get(field_name)
        if field is None or weight <= 0:
            continue
        transformed = np.log1p(_robust_anomaly(field.values, field))
        weighted_reliability = float(weight) * field.reliability
        numerator += transformed * weighted_reliability
        denominator += weighted_reliability
        support_count += (
            field.valid_mask & (field.reliability >= bundle.reliability.invalid_threshold)
        ).astype(np.uint8)
        available.append(field_name)
    energy = numerator / np.maximum(denominator, 1e-8)
    energy[support_count < hypothesis.minimum_supported_fields] = 0
    return HypothesisScore(
        hypothesis=hypothesis,
        energy_map=energy.astype(np.float32),
        support_count_map=support_count,
        available_fields=available,
    )


def build_evidence_bundle(
    result: "DetectionResult",
    *,
    reliable_threshold: float = 0.60,
    invalid_threshold: float = 0.20,
    detector_version: str = "0.2.0",
) -> EvidenceBundle:
    """Translate legacy task maps into the shared node-edge-region protocol."""

    mappings = _MAPPINGS.get(result.task)
    if mappings is None:
        raise ValueError(f"No evidence ontology mapping for task {result.task!r}")
    reliability = _task_reliability(
        result.task,
        result.evidence_maps,
        result.image_shape,
        reliable_threshold,
        invalid_threshold,
        result.metadata,
    )
    fields: Dict[str, EvidenceField] = {}
    for mapping in mappings:
        values = result.evidence_maps.get(mapping.source_key)
        if values is None:
            continue
        field_reliability, valid = _field_reliability(
            result.task, mapping.source_key, result.evidence_maps, reliability
        )
        fields[mapping.descriptor.name] = EvidenceField(
            descriptor=mapping.descriptor,
            values=np.asarray(values, dtype=np.float32),
            reliability=field_reliability,
            valid_mask=valid,
            source_key=mapping.source_key,
        )

    source_name = str(result.metadata.get("source", "<unknown>"))
    input_provenance = dict(result.metadata.get("input_provenance", {}))
    bundle = EvidenceBundle(
        task=result.task,
        image_shape=result.image_shape,
        fields=fields,
        reliability=reliability,
        provenance={
            "source_kind": "physics_pseudo",
            "source_image": source_name,
            "source_id": input_provenance.get("source_id", "unknown"),
            "source_file_sha256": input_provenance.get("source_file_sha256"),
            "pixel_sha256": input_provenance.get("pixel_sha256"),
            "config_sha256": input_provenance.get("config_sha256"),
            "detector_version": detector_version,
            "platform_version": PLATFORM_VERSION,
            "task_detector": result.task,
            "teacher_independent_of_student": True,
            "gold_label_created": False,
        },
    )
    for hypothesis in hypotheses_for_task(result.task):
        if hypothesis.name in bundle.hypotheses:
            raise ValueError(
                f"Duplicate hypothesis name for {result.task}: "
                f"{hypothesis.name}"
            )
        bundle.hypotheses[hypothesis.name] = score_hypothesis(
            bundle, hypothesis
        )
    bundle.validate()
    return bundle


def _region_pixels(
    result: "DetectionResult", bbox_xyxy: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = (int(value) for value in bbox_xyxy)
    local = result.mask[y1:y2, x1:x2]
    rows, columns = np.nonzero(local)
    if not len(rows):
        rows, columns = np.indices((max(y2 - y1, 1), max(x2 - x1, 1)))
        rows, columns = rows.ravel(), columns.ravel()
    return rows + y1, columns + x1


def _region_violations(defect_type: str) -> list[str]:
    if defect_type == "local_amorphous_region":
        return [
            ViolationType.PERIODICITY_LOSS.value,
            ViolationType.COHERENCE_LOSS.value,
        ]
    if defect_type == "displaced_atom_column":
        return [ViolationType.DISPLACEMENT.value]
    if defect_type == "intensity_or_substitution_anomaly":
        return [ViolationType.INTENSITY.value]
    if defect_type in {
        "vacancy_or_low_intensity_column",
        "interstitial_or_extra_column",
    }:
        return [ViolationType.OCCUPANCY.value]
    if defect_type == "unexpected_template_site_occupancy":
        return [
            ViolationType.OCCUPANCY.value,
            ViolationType.CHEMICAL_CONTRAST.value,
        ]
    if defect_type == "stacking_fault_or_dislocation_line":
        return [
            ViolationType.PHASE_SLIP.value,
            ViolationType.ROTATION.value,
            ViolationType.COHERENCE_LOSS.value,
        ]
    if defect_type == "two_phase_or_twin_boundary":
        return [ViolationType.DOMAIN_CHANGE.value]
    if defect_type == "symmetry_related_domain":
        return [ViolationType.DOMAIN_CHANGE.value]
    if defect_type == "symmetry_domain_interface":
        return [
            ViolationType.DOMAIN_CHANGE.value,
            ViolationType.STRAIN.value,
        ]
    if defect_type == "lattice_phase_slip_line":
        return [ViolationType.PHASE_SLIP.value]
    if defect_type == "dislocation_core_proxy":
        return [
            ViolationType.DISPLACEMENT.value,
            ViolationType.PHASE_SLIP.value,
        ]
    return []


def _decision_map(
    result: "DetectionResult",
    reliability: ReliabilityBundle,
    minimum_confirmed_evidence_groups: int,
) -> np.ndarray:
    decision = np.full(
        result.image_shape,
        DECISION_STATE_CODES[DecisionState.BACKGROUND.value],
        dtype=np.uint8,
    )
    invalid = (~reliability.valid_mask) | (
        reliability.reliability_map < reliability.invalid_threshold
    )
    decision[invalid] = DECISION_STATE_CODES[
        DecisionState.INVALID_MEASUREMENT.value
    ]
    decision[reliability.ood_map >= 0.5] = DECISION_STATE_CODES[
        DecisionState.OUT_OF_DOMAIN.value
    ]
    candidate_reliable = (
        result.mask
        & ~invalid
        & (reliability.reliability_map >= reliability.reliable_threshold)
    )
    decision[result.mask & ~candidate_reliable & ~invalid] = (
        DECISION_STATE_CODES[DecisionState.UNCERTAIN_CANDIDATE.value]
    )
    decision[candidate_reliable] = DECISION_STATE_CODES[
        DecisionState.CONFIRMED_CANDIDATE.value
    ]

    if minimum_confirmed_evidence_groups > 1:
        # Region-level evidence independence is not naturally a per-pixel map.
        # Downgrade regions that fail the explicit group-count requirement.
        for region in result.regions:
            if len(region.evidence_groups) >= minimum_confirmed_evidence_groups:
                continue
            rows, columns = _region_pixels(result, region.bbox_xyxy)
            selected = (
                decision[rows, columns]
                == DECISION_STATE_CODES[
                    DecisionState.CONFIRMED_CANDIDATE.value
                ]
            )
            decision[rows[selected], columns[selected]] = DECISION_STATE_CODES[
                DecisionState.UNCERTAIN_CANDIDATE.value
            ]
    return decision


def attach_evidence_protocol(
    result: "DetectionResult",
    *,
    reliable_threshold: float = 0.60,
    invalid_threshold: float = 0.20,
    minimum_confirmed_evidence_groups: int = 1,
    detector_version: str = "0.2.0",
) -> "DetectionResult":
    """Attach shared evidence, provenance, reliability, and abstention semantics."""

    if result.evidence_bundle is not None:
        return result
    bundle = build_evidence_bundle(
        result,
        reliable_threshold=reliable_threshold,
        invalid_threshold=invalid_threshold,
        detector_version=detector_version,
    )
    decision = _decision_map(
        result, bundle.reliability, minimum_confirmed_evidence_groups
    )
    result.evidence_bundle = bundle
    result.reliability_map = bundle.reliability.reliability_map
    result.decision_map = decision
    result.hypothesis_maps = {
        name: score.energy_map for name, score in bundle.hypotheses.items()
    }

    hypothesis = DEFAULT_HYPOTHESES[result.task]
    for region in result.regions:
        rows, columns = _region_pixels(result, region.bbox_xyxy)
        region.reliability = float(
            np.mean(bundle.reliability.reliability_map[rows, columns])
        )
        codes = decision[rows, columns]
        if np.any(
            codes
            == DECISION_STATE_CODES[DecisionState.OUT_OF_DOMAIN.value]
        ):
            region.decision_state = DecisionState.OUT_OF_DOMAIN.value
        elif np.any(
            codes
            == DECISION_STATE_CODES[DecisionState.CONFIRMED_CANDIDATE.value]
        ):
            region.decision_state = DecisionState.CONFIRMED_CANDIDATE.value
        elif np.all(
            codes
            == DECISION_STATE_CODES[DecisionState.INVALID_MEASUREMENT.value]
        ):
            region.decision_state = DecisionState.INVALID_MEASUREMENT.value
        else:
            region.decision_state = DecisionState.UNCERTAIN_CANDIDATE.value
        region.label_tier = (
            LabelTier.SILVER.value
            if region.decision_state == DecisionState.CONFIRMED_CANDIDATE.value
            else LabelTier.UNRESOLVED.value
        )
        if not region.spatial_support:
            region.spatial_support = hypothesis.spatial_support.value
        region.violation_types = _region_violations(region.defect_type)
        source_id = str(bundle.provenance.get("source_id", "unknown"))
        region.candidate_id = (
            f"{source_id}:{result.task}:{region.region_id:04d}"
        )
        region.quality_flags = ["uncalibrated_confidence"]
        if region.decision_state == DecisionState.INVALID_MEASUREMENT.value:
            region.quality_flags.append("invalid_measurement")
        if region.decision_state == DecisionState.OUT_OF_DOMAIN.value:
            region.quality_flags.append("out_of_domain")
        region.evidence_vector = {}
        for name, evidence in bundle.fields.items():
            values = evidence.values[rows, columns]
            field_reliability = evidence.reliability[rows, columns]
            field_validity = evidence.valid_mask[rows, columns]
            region.evidence_vector[name] = {
                "mean": float(np.mean(values)),
                "max": float(np.max(values)),
                "mean_reliability": float(np.mean(field_reliability)),
                "valid_fraction": float(np.mean(field_validity)),
            }
        region.provenance = {
            **region.provenance,
            **bundle.provenance,
            "candidate_id": region.candidate_id,
            "label_tier": region.label_tier,
        }

    unique, counts = np.unique(decision, return_counts=True)
    inverse_codes = {value: key for key, value in DECISION_STATE_CODES.items()}
    decision_counts = {
        inverse_codes[int(code)]: int(count)
        for code, count in zip(unique, counts)
    }
    result.metadata["evidence_protocol"] = {
        "version": EVIDENCE_PROTOCOL_VERSION,
        "representation": "node-edge-region_structural_evidence_graph",
        "ontology_axes": [
            "violation_type",
            "spatial_support",
            "hypothesis_family",
            "epistemic_state",
        ],
        "field_count": len(bundle.fields),
        "legacy_hypothesis": hypothesis.to_dict(),
        "hypotheses": [
            item.to_dict() for item in hypotheses_for_task(result.task)
        ],
    }
    result.metadata["decision_protocol"] = {
        "state_codes": DECISION_STATE_CODES,
        "state_counts": decision_counts,
        "automatic_label_policy": {
            "physics_teacher_can_create": [
                LabelTier.SILVER.value,
                LabelTier.UNRESOLVED.value,
            ],
            "physics_teacher_cannot_create": LabelTier.GOLD.value,
        },
        "confidence_semantics": (
            "Legacy confidence is an uncalibrated within-image ranking score. "
            "It must not be interpreted as probability of ground-truth correctness."
        ),
        "confirmed_candidate_semantics": (
            "Reliable and internally supported physics-teacher candidate; "
            "not human gold and not a calibrated probability."
        ),
    }
    result.metadata["provenance"] = bundle.provenance
    return result


def ontology_manifest() -> Dict[str, Any]:
    """Return the machine-readable default ontology without array payloads."""

    return {
        "protocol_version": EVIDENCE_PROTOCOL_VERSION,
        "axes": {
            "violation_type": [item.value for item in ViolationType],
            "spatial_support": [item.value for item in SpatialSupport],
        },
        "hypotheses": {
            task: [hypothesis.to_dict() for hypothesis in hypotheses]
            for task, hypotheses in OPEN_WORLD_HYPOTHESES.items()
        },
        "legacy_mappings": {
            task: {
                mapping.source_key: mapping.descriptor.to_dict()
                for mapping in mappings
            }
            for task, mappings in _MAPPINGS.items()
        },
    }
