"""Evidence-preserving multi-head student and physics-teacher targets.

PyTorch is optional.  Target construction remains available without it; model
construction raises a clear error when the optional dependency is absent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from .evidence import (
    DECISION_STATE_CODES,
    DEFAULT_HYPOTHESES,
    DecisionState,
    EvidenceDirection,
    EvidenceRole,
    SpatialSupport,
    ViolationType,
)

if TYPE_CHECKING:
    from .digital_twin import CounterfactualTwinPair
    from .models import DetectionResult

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised only in minimal installs
    torch = None
    Tensor = Any
    nn = None
    F = None


GEOMETRY_CHANNELS = (
    "background",
    SpatialSupport.POINT.value,
    SpatialSupport.LINE.value,
    SpatialSupport.REGION.value,
    SpatialSupport.INTERFACE.value,
)

VIOLATION_CHANNELS = (
    ViolationType.OCCUPANCY.value,
    ViolationType.INTENSITY.value,
    ViolationType.DISPLACEMENT.value,
    ViolationType.SHAPE.value,
    ViolationType.PHASE_SLIP.value,
    ViolationType.STRAIN.value,
    ViolationType.ROTATION.value,
    ViolationType.COHERENCE_LOSS.value,
    ViolationType.PERIODICITY_LOSS.value,
    ViolationType.DOMAIN_CHANGE.value,
)

DEFAULT_STUDENT_EVIDENCE = (
    "node.presence_residual",
    "node.intensity_residual",
    "node.displacement_residual",
    "node.extra_peak_residual",
    "edge.gpa_strain_magnitude",
    "edge.gpa_exx",
    "edge.gpa_eyy",
    "edge.gpa_exy",
    "edge.gpa_rotation",
    "edge.gpa_principal_tensile",
    "edge.gpa_principal_compressive",
    "edge.gpa_basis1_length_change",
    "edge.gpa_basis2_length_change",
    "edge.phase_jump",
    "edge.phase_curvature",
    "edge.shear_rotation",
    "edge.coherence_loss",
    "edge.strain_magnitude",
    "edge.domain_signature_transition",
    "region.fourier_periodicity_loss",
    "region.patterson_order_loss",
    "region.sharpness_loss",
)


@dataclass
class StudentTarget:
    """Dense supervision distilled from one physics-teacher result."""

    image: np.ndarray
    geometry_labels: np.ndarray
    violation_targets: np.ndarray
    evidence_targets: np.ndarray
    evidence_weights: np.ndarray
    teacher_soft_mask: np.ndarray
    reliability_target: np.ndarray
    valid_mask: np.ndarray
    supervision_weight: np.ndarray
    ood_target: np.ndarray
    ood_available: np.ndarray
    evidence_names: tuple[str, ...]
    evidence_statistics: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )
    provenance: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        shape = self.image.shape
        if self.image.ndim != 2:
            raise ValueError("Student image must be 2-D")
        for name, values in (
            ("geometry_labels", self.geometry_labels),
            ("teacher_soft_mask", self.teacher_soft_mask),
            ("reliability_target", self.reliability_target),
            ("valid_mask", self.valid_mask),
            ("supervision_weight", self.supervision_weight),
            ("ood_target", self.ood_target),
            ("ood_available", self.ood_available),
        ):
            if values.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        expected_channels = len(self.evidence_names)
        if self.evidence_targets.shape != (expected_channels, *shape):
            raise ValueError("Student evidence target shape is invalid")
        if self.evidence_weights.shape != self.evidence_targets.shape:
            raise ValueError("Student evidence weights must match targets")
        if self.violation_targets.shape != (
            len(VIOLATION_CHANNELS),
            *shape,
        ):
            raise ValueError("Student violation target shape is invalid")
        if not np.all(np.isfinite(self.image)):
            raise ValueError("Student image contains non-finite values")
        for values in (
            self.teacher_soft_mask,
            self.reliability_target,
            self.valid_mask,
            self.supervision_weight,
            self.ood_target,
            self.ood_available,
            self.evidence_weights,
        ):
            if np.any(values < 0) or np.any(values > 1):
                raise ValueError("Student weights and probability targets must be in [0, 1]")


def _normalize_evidence(
    values: np.ndarray,
    valid: np.ndarray,
    direction: EvidenceDirection,
) -> tuple[np.ndarray, Dict[str, float]]:
    finite = np.asarray(values, dtype=np.float32)[valid]
    if not finite.size:
        return np.zeros_like(values, dtype=np.float32), {
            "center": 0.0,
            "scale": 1.0,
            "clip": 8.0,
        }
    center = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - center)))
    if scale < 1e-8:
        scale = max(float(np.std(finite)), 1e-8)
    normalized = (np.asarray(values, dtype=np.float32) - center) / scale
    if direction == EvidenceDirection.ABSOLUTE_IS_MORE_ABNORMAL:
        # Preserve the sign for physical strain components; the descriptor
        # direction still records that magnitude is anomalous.
        pass
    normalized = np.clip(normalized, -8, 8) / 8.0
    normalized[~valid] = 0
    return normalized.astype(np.float32), {
        "center": center,
        "scale": scale,
        "clip": 8.0,
    }


def student_target_from_result(
    result: "DetectionResult",
    evidence_names: Sequence[str] = DEFAULT_STUDENT_EVIDENCE,
) -> StudentTarget:
    """Convert a unified physics-teacher result into soft dense supervision."""

    bundle = result.evidence_bundle
    if bundle is None or result.reliability_map is None or result.decision_map is None:
        raise ValueError("A unified EvidenceBundle is required for student targets")
    height, width = result.image_shape
    geometry = np.zeros((height, width), dtype=np.int64)
    legacy_name = DEFAULT_HYPOTHESES[result.task].name
    legacy_score = bundle.hypotheses[legacy_name]
    support = legacy_score.hypothesis.spatial_support
    geometry_index = GEOMETRY_CHANNELS.index(support.value)
    geometry[result.mask] = geometry_index

    violations = np.zeros(
        (len(VIOLATION_CHANNELS), height, width), dtype=np.float32
    )
    hypothesis = legacy_score.hypothesis
    for violation in hypothesis.violation_types:
        if violation.value in VIOLATION_CHANNELS:
            violations[VIOLATION_CHANNELS.index(violation.value), result.mask] = 1

    evidence_targets = np.zeros(
        (len(evidence_names), height, width), dtype=np.float32
    )
    evidence_weights = np.zeros_like(evidence_targets)
    statistics: Dict[str, Dict[str, float]] = {}
    for index, name in enumerate(evidence_names):
        field = bundle.fields.get(name)
        if field is None or field.descriptor.role != EvidenceRole.ANOMALY:
            continue
        normalized, summary = _normalize_evidence(
            field.values, field.valid_mask, field.descriptor.direction
        )
        evidence_targets[index] = normalized
        evidence_weights[index] = (
            field.reliability * field.valid_mask
        ).astype(np.float32)
        statistics[name] = summary

    reliability = result.reliability_map.astype(np.float32)
    valid = bundle.reliability.valid_mask.astype(np.float32)
    confirmed_code = DECISION_STATE_CODES[
        DecisionState.CONFIRMED_CANDIDATE.value
    ]
    uncertain_code = DECISION_STATE_CODES[
        DecisionState.UNCERTAIN_CANDIDATE.value
    ]
    supervision = 0.5 * reliability * (~result.mask)
    supervision += (
        reliability * (result.decision_map == confirmed_code)
    )
    supervision += (
        0.25 * reliability * (result.decision_map == uncertain_code)
    )
    supervision *= valid

    legacy_probability = np.clip(result.probability_map, 0, 1)
    # Preserve the teacher's accepted support while retaining a soft ranking
    # within candidate and background regions.  The raw per-image sigmoid is
    # not calibrated and can be diffuse over large valid backgrounds.
    teacher_soft_mask = np.where(
        result.mask,
        0.5 + 0.5 * legacy_probability,
        0.1 * legacy_probability,
    )
    target = StudentTarget(
        image=result.normalized_image.astype(np.float32),
        geometry_labels=geometry,
        violation_targets=violations,
        evidence_targets=evidence_targets,
        evidence_weights=evidence_weights,
        teacher_soft_mask=teacher_soft_mask.astype(np.float32),
        reliability_target=reliability,
        valid_mask=valid,
        supervision_weight=supervision.astype(np.float32),
        ood_target=bundle.reliability.ood_map.astype(np.float32),
        # The v1 physics teacher has no learned domain model.  Zeros are not
        # negative OOD labels; availability therefore remains zero.
        ood_available=np.zeros((height, width), dtype=np.float32),
        evidence_names=tuple(evidence_names),
        evidence_statistics=statistics,
        provenance={
            **bundle.provenance,
            "target_kind": "evidence_preserving_distillation",
            "teacher_soft_mask_semantics": (
                "accepted-support-preserving uncalibrated teacher softness"
            ),
            "high_reliability_normal_weight": 0.5,
            "unresolved_candidate_weight": 0.25,
        },
    )
    target.validate()
    return target


_TWIN_TASK_SEMANTICS = {
    "amorphous_region": (
        SpatialSupport.REGION,
        (
            ViolationType.PERIODICITY_LOSS,
            ViolationType.COHERENCE_LOSS,
        ),
        "region.fourier_periodicity_loss",
    ),
    "atomic_point_defect": (
        SpatialSupport.POINT,
        (ViolationType.OCCUPANCY, ViolationType.INTENSITY),
        "node.presence_residual",
    ),
    "stacking_fault": (
        SpatialSupport.LINE,
        (ViolationType.PHASE_SLIP, ViolationType.ROTATION),
        "edge.phase_jump",
    ),
    "phase_boundary": (
        SpatialSupport.INTERFACE,
        (ViolationType.DOMAIN_CHANGE,),
        "edge.domain_signature_transition",
    ),
}


def student_targets_from_counterfactual_twin(
    pair: "CounterfactualTwinPair",
    evidence_names: Sequence[str] = DEFAULT_STUDENT_EVIDENCE,
) -> tuple[StudentTarget, StudentTarget]:
    """Create paired synthetic-Gold normal/defect targets with shared nuisance."""

    pair.validate()
    support, violations, primary_evidence = _TWIN_TASK_SEMANTICS[pair.task]
    height, width = pair.normal_image.shape

    def base_target(image: np.ndarray) -> Dict[str, Any]:
        return {
            "image": image.astype(np.float32) / 65535.0,
            "geometry_labels": np.zeros((height, width), dtype=np.int64),
            "violation_targets": np.zeros(
                (len(VIOLATION_CHANNELS), height, width), dtype=np.float32
            ),
            "evidence_targets": np.zeros(
                (len(evidence_names), height, width), dtype=np.float32
            ),
            "evidence_weights": np.zeros(
                (len(evidence_names), height, width), dtype=np.float32
            ),
            "teacher_soft_mask": np.zeros(
                (height, width), dtype=np.float32
            ),
            "reliability_target": np.ones(
                (height, width), dtype=np.float32
            ),
            "valid_mask": np.ones((height, width), dtype=np.float32),
            "supervision_weight": np.ones(
                (height, width), dtype=np.float32
            ),
            "ood_target": np.zeros((height, width), dtype=np.float32),
            "ood_available": np.zeros((height, width), dtype=np.float32),
            "evidence_names": tuple(evidence_names),
            "evidence_statistics": {},
        }

    common_provenance = {
        **pair.manifest(),
        "target_kind": "counterfactual_synthetic_gold",
        "experimental_accuracy_claim_allowed": False,
    }
    normal_values = base_target(pair.normal_image)
    normal = StudentTarget(
        **normal_values,
        provenance={
            **common_provenance,
            "counterfactual_member": "normal",
        },
    )

    defect_values = base_target(pair.defect_image)
    mask = pair.defect_mask
    defect_values["geometry_labels"][mask] = GEOMETRY_CHANNELS.index(
        support.value
    )
    for violation in violations:
        defect_values["violation_targets"][
            VIOLATION_CHANNELS.index(violation.value), mask
        ] = 1
    soft_mask = np.clip(
        gaussian_filter(mask.astype(np.float32), 0.8), 0, 1
    )
    defect_values["teacher_soft_mask"] = soft_mask.astype(np.float32)
    if primary_evidence in evidence_names:
        index = tuple(evidence_names).index(primary_evidence)
        delta = np.abs(pair.structural_delta)
        delta /= max(float(np.percentile(delta[mask], 95)), 1e-8)
        defect_values["evidence_targets"][index] = np.clip(
            delta, 0, 1
        )
        defect_values["evidence_weights"][index] = mask.astype(np.float32)
        defect_values["evidence_statistics"][primary_evidence] = {
            "center": 0.0,
            "scale": max(
                float(np.percentile(np.abs(pair.structural_delta)[mask], 95)),
                1e-8,
            ),
            "clip": 1.0,
        }
    defect = StudentTarget(
        **defect_values,
        provenance={
            **common_provenance,
            "counterfactual_member": "defect",
        },
    )
    normal.validate()
    defect.validate()
    return normal, defect


def stack_student_targets(
    targets: Sequence[StudentTarget],
    device: str | Any = "cpu",
) -> Dict[str, Tensor]:
    if torch is None:
        raise ImportError("PyTorch is required for student training")
    if not targets:
        raise ValueError("At least one StudentTarget is required")
    for target in targets:
        target.validate()
    shapes = {target.image.shape for target in targets}
    names = {target.evidence_names for target in targets}
    if len(shapes) != 1 or len(names) != 1:
        raise ValueError("Batched student targets need equal shapes and channels")

    def tensor(values: np.ndarray, dtype: Any = torch.float32) -> Tensor:
        return torch.as_tensor(values, dtype=dtype, device=device)

    return {
        "image": tensor(np.stack([target.image for target in targets])[:, None]),
        "geometry_labels": tensor(
            np.stack([target.geometry_labels for target in targets]),
            dtype=torch.long,
        ),
        "violation_targets": tensor(
            np.stack([target.violation_targets for target in targets])
        ),
        "evidence_targets": tensor(
            np.stack([target.evidence_targets for target in targets])
        ),
        "evidence_weights": tensor(
            np.stack([target.evidence_weights for target in targets])
        ),
        "teacher_soft_mask": tensor(
            np.stack([target.teacher_soft_mask for target in targets])[:, None]
        ),
        "reliability_target": tensor(
            np.stack([target.reliability_target for target in targets])[:, None]
        ),
        "valid_mask": tensor(
            np.stack([target.valid_mask for target in targets])[:, None]
        ),
        "supervision_weight": tensor(
            np.stack([target.supervision_weight for target in targets])[:, None]
        ),
        "ood_target": tensor(
            np.stack([target.ood_target for target in targets])[:, None]
        ),
        "ood_available": tensor(
            np.stack([target.ood_available for target in targets])[:, None]
        ),
    }


@dataclass
class StudentOutput:
    geometry_logits: Tensor
    violation_logits: Tensor
    evidence_prediction: Tensor
    reliability_logits: Tensor
    ood_logits: Tensor
    embedding: Tensor

    @property
    def candidate_logits(self) -> Tensor:
        if torch is None:
            raise ImportError("PyTorch is required")
        return torch.logsumexp(self.geometry_logits[:, 1:], dim=1, keepdim=True) - (
            self.geometry_logits[:, :1]
        )


if nn is not None:

    class _ConvBlock(nn.Module):
        def __init__(self, input_channels: int, output_channels: int):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.GroupNorm(
                    num_groups=max(1, min(4, output_channels)),
                    num_channels=output_channels,
                ),
                nn.GELU(),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.GroupNorm(
                    num_groups=max(1, min(4, output_channels)),
                    num_channels=output_channels,
                ),
                nn.GELU(),
            )

        def forward(self, values: Tensor) -> Tensor:
            return self.layers(values)


    class EvidencePreservingStudent(nn.Module):
        """Small U-Net-like shared encoder with auditable multi-task heads."""

        def __init__(
            self,
            evidence_channels: int = len(DEFAULT_STUDENT_EVIDENCE),
            base_channels: int = 16,
        ):
            super().__init__()
            if evidence_channels < 1 or base_channels < 4:
                raise ValueError("Student channel counts are too small")
            self.evidence_channels = evidence_channels
            self.encoder1 = _ConvBlock(1, base_channels)
            self.encoder2 = _ConvBlock(base_channels, 2 * base_channels)
            self.bottleneck = _ConvBlock(2 * base_channels, 4 * base_channels)
            self.decoder2 = _ConvBlock(6 * base_channels, 2 * base_channels)
            self.decoder1 = _ConvBlock(3 * base_channels, base_channels)
            self.evidence_fusion = nn.Sequential(
                nn.Conv2d(
                    base_channels
                    + len(VIOLATION_CHANNELS)
                    + evidence_channels
                    + 1,
                    base_channels,
                    1,
                ),
                nn.GELU(),
            )
            self.geometry_head = nn.Conv2d(
                base_channels, len(GEOMETRY_CHANNELS), 1
            )
            self.violation_head = nn.Conv2d(
                base_channels, len(VIOLATION_CHANNELS), 1
            )
            self.evidence_head = nn.Conv2d(
                base_channels, evidence_channels, 1
            )
            self.reliability_ood_head = nn.Conv2d(base_channels, 2, 1)

        def forward(self, image: Tensor) -> StudentOutput:
            first = self.encoder1(image)
            second = self.encoder2(F.max_pool2d(first, 2))
            bottleneck = self.bottleneck(F.max_pool2d(second, 2))
            up_second = F.interpolate(
                bottleneck,
                size=second.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded_second = self.decoder2(
                torch.cat([up_second, second], dim=1)
            )
            up_first = F.interpolate(
                decoded_second,
                size=first.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded_first = self.decoder1(
                torch.cat([up_first, first], dim=1)
            )
            reliability_ood = self.reliability_ood_head(decoded_first)
            violation_logits = self.violation_head(decoded_first)
            evidence_prediction = torch.tanh(
                self.evidence_head(decoded_first)
            )
            fused = decoded_first + self.evidence_fusion(
                torch.cat(
                    [
                        decoded_first,
                        torch.sigmoid(violation_logits),
                        evidence_prediction,
                        torch.sigmoid(reliability_ood[:, :1]),
                    ],
                    dim=1,
                )
            )
            return StudentOutput(
                geometry_logits=self.geometry_head(fused),
                violation_logits=violation_logits,
                evidence_prediction=evidence_prediction,
                reliability_logits=reliability_ood[:, :1],
                ood_logits=reliability_ood[:, 1:2],
                embedding=torch.mean(bottleneck, dim=(-2, -1)),
            )

else:

    class EvidencePreservingStudent:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any):
            raise ImportError(
                "EvidencePreservingStudent requires the optional PyTorch dependency"
            )


@dataclass(frozen=True)
class StudentLossWeights:
    geometry: float = 1.0
    soft_pseudo_mask: float = 1.0
    violation: float = 0.8
    evidence: float = 1.0
    reliability: float = 0.5
    ood: float = 0.5


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-8)


def evidence_preserving_loss(
    output: StudentOutput,
    target: Mapping[str, Tensor],
    weights: StudentLossWeights = StudentLossWeights(),
) -> tuple[Tensor, Dict[str, Tensor]]:
    """Physics-aware objective with separate, inspectable loss components."""

    if torch is None:
        raise ImportError("PyTorch is required for student training")
    supervision = target["supervision_weight"]
    valid = target["valid_mask"]
    geometry_values = F.cross_entropy(
        output.geometry_logits,
        target["geometry_labels"],
        reduction="none",
    )
    geometry = _weighted_mean(geometry_values, supervision[:, 0])
    soft_mask = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output.candidate_logits,
            target["teacher_soft_mask"],
            reduction="none",
        ),
        supervision,
    )
    candidate_weight = supervision * target["teacher_soft_mask"]
    violation = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output.violation_logits,
            target["violation_targets"],
            reduction="none",
        ),
        candidate_weight,
    )
    evidence = _weighted_mean(
        F.smooth_l1_loss(
            output.evidence_prediction,
            target["evidence_targets"],
            reduction="none",
        ),
        target["evidence_weights"],
    )
    reliability = _weighted_mean(
        F.binary_cross_entropy_with_logits(
            output.reliability_logits,
            target["reliability_target"],
            reduction="none",
        ),
        torch.ones_like(valid),
    )
    if float(torch.sum(target["ood_available"]).detach().cpu()) > 0:
        ood = _weighted_mean(
            F.binary_cross_entropy_with_logits(
                output.ood_logits,
                target["ood_target"],
                reduction="none",
            ),
            target["ood_available"],
        )
    else:
        ood = torch.sum(output.ood_logits) * 0
    components = {
        "geometry": geometry,
        "soft_pseudo_mask": soft_mask,
        "violation": violation,
        "evidence": evidence,
        "reliability": reliability,
        "ood": ood,
    }
    total = (
        weights.geometry * geometry
        + weights.soft_pseudo_mask * soft_mask
        + weights.violation * violation
        + weights.evidence * evidence
        + weights.reliability * reliability
        + weights.ood * ood
    )
    return total, components


def hard_pseudolabel_baseline_loss(
    output: StudentOutput,
    hard_mask: Tensor,
    supervision_weight: Tensor | None = None,
) -> Tensor:
    """Ablation baseline that learns only a thresholded teacher mask."""

    if torch is None:
        raise ImportError("PyTorch is required for student training")
    if hard_mask.ndim == 3:
        hard_mask = hard_mask[:, None]
    values = F.binary_cross_entropy_with_logits(
        output.candidate_logits, hard_mask.float(), reduction="none"
    )
    if supervision_weight is None:
        return torch.mean(values)
    return _weighted_mean(values, supervision_weight)
