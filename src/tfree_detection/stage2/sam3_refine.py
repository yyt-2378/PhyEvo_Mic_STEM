"""Physics-prompted SAM 3 boundary proposals with deterministic fallback.

This module deliberately has no dependency on SAM 3 or PyTorch.  The main
training-free environment owns physical candidate discovery, prompt creation,
proposal validation, and the final accept/reject decision.  A separate Python
3.12 worker only turns those prompts into mask proposals.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
)
from skimage.measure import label

from ..config import SAM3RefinementConfig
from ..models import DetectionResult


SAM3_PROTOCOL_VERSION = "tfree-sam3-boundary-v2"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class SAM3Prompt:
    """One deterministic perturbation of one physical candidate prompt."""

    candidate_id: str
    component_label: int
    ensemble_index: int
    box_xyxy: Tuple[int, int, int, int]
    positive_points_xy: Tuple[Tuple[int, int], ...]
    negative_points_xy: Tuple[Tuple[int, int], ...]
    use_coarse_mask_prompt: bool
    prompt_role: str = "candidate"
    geometry_kind: str = "region"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["box_xyxy"] = list(self.box_xyxy)
        payload["positive_points_xy"] = [
            list(point) for point in self.positive_points_xy
        ]
        payload["negative_points_xy"] = [
            list(point) for point in self.negative_points_xy
        ]
        return payload


@dataclass
class PhysicsPromptPlan:
    """Physics-owned prompt material and gate reference regions."""

    prompts: List[SAM3Prompt]
    physics_core: np.ndarray
    negative_reference: np.ndarray
    component_labels: np.ndarray
    padding_pixels: int
    padding_source: str
    geometry_kind: str = "region"
    domain_labels: np.ndarray | None = None
    valid_mask: np.ndarray | None = None


@dataclass
class SAM3Proposal:
    """One full-image-sized mask for one candidate and one perturbation."""

    proposal_id: str
    mask: np.ndarray
    score: float
    ensemble_index: int
    candidate_id: str = "candidate-0001"
    component_label: int = 1
    prompt_role: str = "candidate"
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateBoundaryRefinement:
    """Candidate-local consensus, gate decision, and deterministic fallback."""

    candidate_id: str
    component_label: int
    coarse_mask: np.ndarray
    proposals: List[SAM3Proposal]
    inclusion_probability: np.ndarray
    refined_mask: np.ndarray
    stable_core: np.ndarray
    possible_extent: np.ndarray
    boundary_uncertainty: np.ndarray
    final_mask: np.ndarray
    physics_core: np.ndarray
    negative_reference: np.ndarray
    accepted: bool
    status: str
    rejection_reasons: List[str]
    metrics: Dict[str, Any]
    prompts: List[SAM3Prompt]

    def to_summary(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "component_label": self.component_label,
            "status": self.status,
            "accepted": self.accepted,
            "final_geometry_source": (
                "sam3_consensus" if self.accepted else "physics_coarse_mask"
            ),
            "proposal_count": len(self.proposals),
            "coarse_area_pixels": int(self.coarse_mask.sum()),
            "refined_area_pixels": int(self.refined_mask.sum()),
            "final_area_pixels": int(self.final_mask.sum()),
            "rejection_reasons": list(self.rejection_reasons),
            "metrics": self.metrics,
        }


@dataclass
class BoundaryRefinementResult:
    """Auditable result of proposal consensus and physical guardrails."""

    coarse_mask: np.ndarray
    proposals: List[SAM3Proposal]
    inclusion_probability: np.ndarray
    refined_mask: np.ndarray
    stable_core: np.ndarray
    possible_extent: np.ndarray
    boundary_uncertainty: np.ndarray
    final_mask: np.ndarray
    physics_core: np.ndarray
    negative_reference: np.ndarray
    accepted: bool
    status: str
    rejection_reasons: List[str]
    metrics: Dict[str, Any]
    prompts: List[SAM3Prompt]
    candidate_results: List[CandidateBoundaryRefinement] = field(
        default_factory=list
    )
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_summary(self) -> Dict[str, Any]:
        worker = self.provenance.get("worker_response", {})
        return {
            "protocol_version": SAM3_PROTOCOL_VERSION,
            "status": self.status,
            "accepted": self.accepted,
            "policy": (
                "A: SAM refines geometry only; preserve >=95% physics core "
                "or fall back to coarse mask"
            ),
            "scientific_candidate_mask_preserved": True,
            "final_geometry_source": (
                "mixed_sam3_and_physics_coarse"
                if self.status == "partially_accepted"
                else "sam3_consensus"
                if self.accepted
                else "physics_coarse_mask"
            ),
            "proposal_count": len(self.proposals),
            "coarse_area_pixels": int(self.coarse_mask.sum()),
            "refined_area_pixels": int(self.refined_mask.sum()),
            "final_area_pixels": int(self.final_mask.sum()),
            "stable_core_area_pixels": int(self.stable_core.sum()),
            "possible_extent_area_pixels": int(self.possible_extent.sum()),
            "rejection_reasons": list(self.rejection_reasons),
            "metrics": self.metrics,
            "prompt_count": len(self.prompts),
            "prompt_candidate_count": len(
                {prompt.candidate_id for prompt in self.prompts}
            ),
            "prompt_ensemble_count": len(
                {prompt.ensemble_index for prompt in self.prompts}
            ),
            "accepted_candidate_count": sum(
                item.accepted for item in self.candidate_results
            ),
            "fallback_candidate_count": sum(
                not item.accepted for item in self.candidate_results
            ),
            "candidate_results": [
                item.to_summary() for item in self.candidate_results
            ],
            "runtime_and_memory": {
                "worker_runtime_seconds": worker.get("runtime_seconds"),
                "subprocess_runtime_seconds": self.provenance.get(
                    "subprocess_runtime_seconds"
                ),
                "peak_process_rss_bytes": worker.get(
                    "peak_process_rss_bytes"
                ),
                "peak_device_memory_bytes": worker.get(
                    "peak_device_memory_bytes"
                ),
            },
            "expert_correction": {
                "status": "not_measured_during_inference",
                "measurement_path": (
                    "assisted/blinded review responses -> "
                    "evaluation.expert_correction_metrics"
                ),
            },
            "provenance": self.provenance,
        }


def _array_sha256(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    hasher = hashlib.sha256()
    hasher.update(str(contiguous.dtype).encode("utf-8"))
    hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    hasher.update(contiguous.view(np.uint8).tobytes())
    return hasher.hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _write_array_exclusive(path: Path, values: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def _write_json_exclusive(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


def _safe_exchange_directory(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(
            "SAM3 exchange_directory must remain inside the project root: "
            f"{PROJECT_ROOT}"
        )
    return resolved


def _padding_from_metadata(
    metadata: Dict[str, Any], config: SAM3RefinementConfig
) -> Tuple[int, str]:
    stage2 = metadata.get("stage2", {})
    phase = stage2.get("phase", {}) if isinstance(stage2, dict) else {}
    lengths = phase.get("direct_basis_lengths_pixels", [])
    finite = [
        float(value)
        for value in lengths
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if finite and config.box_padding_lattice_cells > 0:
        padding = int(
            round(np.median(finite) * config.box_padding_lattice_cells)
        )
        return max(padding, 1), "gpa_direct_basis"
    return int(config.box_padding_pixels), "configured_pixels"


def _spaced_points(
    candidate_mask: np.ndarray,
    quality: np.ndarray,
    count: int,
    offset: int,
) -> Tuple[Tuple[int, int], ...]:
    coordinates = np.argwhere(candidate_mask)
    if not len(coordinates):
        return ()
    values = quality[candidate_mask]
    quality_threshold = float(np.quantile(values, 0.70))
    eligible = coordinates[values >= quality_threshold]
    eligible_values = values[values >= quality_threshold]
    if len(eligible) > 4096:
        sampled = np.linspace(
            0, len(eligible) - 1, 4096, dtype=int
        )
        eligible = eligible[sampled]
        eligible_values = eligible_values[sampled]
    pool = eligible
    order = np.argsort(-eligible_values, kind="stable")
    start = int(order[offset % min(len(order), max(count * 8, 1))])
    chosen = [pool[start]]
    available = np.ones(len(pool), dtype=bool)
    available[start] = False
    while len(chosen) < min(count, len(pool)):
        remaining = np.flatnonzero(available)
        distances = np.full(len(remaining), np.inf, dtype=float)
        for point in chosen:
            squared = np.sum((pool[remaining] - point) ** 2, axis=1)
            distances = np.minimum(distances, squared)
        best = remaining[int(np.argmax(distances))]
        chosen.append(pool[best])
        available[best] = False
    return tuple((int(column), int(row)) for row, column in chosen)


def _expanded_box(
    component: np.ndarray,
    padding: int,
    ensemble_index: int,
    ensemble_size: int,
) -> Tuple[int, int, int, int]:
    coordinates = np.argwhere(component)
    y1, x1 = coordinates.min(axis=0)
    y2, x2 = coordinates.max(axis=0) + 1
    if ensemble_size > 1:
        relative = ensemble_index / (ensemble_size - 1) - 0.5
    else:
        relative = 0.0
    varied_padding = max(0, int(round(padding * (1.0 + 0.3 * relative))))
    height, width = component.shape
    return (
        max(0, int(x1) - varied_padding),
        max(0, int(y1) - varied_padding),
        min(width, int(x2) + varied_padding),
        min(height, int(y2) + varied_padding),
    )


def build_physics_prompt_plan(
    coarse_mask: np.ndarray,
    score_map: np.ndarray,
    metadata: Dict[str, Any],
    config: SAM3RefinementConfig,
) -> PhysicsPromptPlan:
    """Generate deterministic points, boxes, mask prompts, and gate zones."""

    coarse = np.asarray(coarse_mask, dtype=bool)
    score = np.asarray(score_map, dtype=float)
    if coarse.shape != score.shape:
        raise ValueError("coarse_mask and score_map must have identical shapes")
    if not np.any(coarse):
        raise ValueError("SAM3 refinement requires at least one coarse candidate")
    finite_score = np.where(np.isfinite(score), score, -np.inf)
    component_labels = label(coarse, connectivity=2)
    physics_core = np.zeros_like(coarse)
    padding, padding_source = _padding_from_metadata(metadata, config)

    for component_label in range(1, int(component_labels.max()) + 1):
        component = component_labels == component_label
        component_scores = finite_score[component]
        threshold = float(
            np.quantile(component_scores, config.core_score_quantile)
        )
        score_core = component & (finite_score >= threshold)
        distance = distance_transform_edt(component)
        if float(distance.max()) >= 3.0:
            interior_values = distance[component]
            interior_threshold = max(
                1.0, float(np.quantile(interior_values, 0.25))
            )
            interior_core = score_core & (distance >= interior_threshold)
            if np.any(interior_core):
                score_core = interior_core
        if not np.any(score_core):
            maximum = np.argmax(np.where(component, finite_score, -np.inf))
            score_core.flat[int(maximum)] = True
        physics_core |= score_core

    outside_distance = distance_transform_edt(~coarse)
    inner = max(2, padding // 3)
    outer = max(inner + 1, padding * 2)
    negative_ring = (
        (~coarse)
        & (outside_distance >= inner)
        & (outside_distance <= outer)
        & np.isfinite(score)
    )
    if np.any(negative_ring):
        low_score = float(np.quantile(score[negative_ring], 0.50))
        negative_reference = negative_ring & (score <= low_score)
    else:
        negative_reference = (~coarse) & (outside_distance >= 1)
    if not np.any(negative_reference):
        negative_reference = ~coarse

    prompts: List[SAM3Prompt] = []
    for component_label in range(1, int(component_labels.max()) + 1):
        component = component_labels == component_label
        component_core = physics_core & component
        component_distance = distance_transform_edt(component)
        positive_quality = finite_score.copy()
        finite_values = positive_quality[np.isfinite(positive_quality)]
        floor = float(finite_values.min()) if finite_values.size else 0.0
        positive_quality[~np.isfinite(positive_quality)] = floor
        positive_quality += component_distance / (
            float(component_distance.max()) + 1e-12
        )
        base_box = _expanded_box(
            component, padding, 0, config.ensemble_size
        )
        x1, y1, x2, y2 = base_box
        local_negative = np.zeros_like(coarse)
        local_negative[y1:y2, x1:x2] = True
        local_negative &= negative_reference
        if not np.any(local_negative):
            local_negative = negative_reference
        finite_negative_quality = -score[np.isfinite(score)]
        negative_floor = (
            float(np.min(finite_negative_quality))
            if finite_negative_quality.size
            else 0.0
        )
        negative_quality = np.where(
            np.isfinite(score), -score, negative_floor
        )
        for ensemble_index in range(config.ensemble_size):
            prompts.append(
                SAM3Prompt(
                    candidate_id=f"candidate-{component_label:04d}",
                    component_label=component_label,
                    ensemble_index=ensemble_index,
                    box_xyxy=_expanded_box(
                        component,
                        padding,
                        ensemble_index,
                        config.ensemble_size,
                    ),
                    positive_points_xy=_spaced_points(
                        component_core,
                        positive_quality,
                        config.positive_points_per_candidate,
                        ensemble_index + config.random_seed,
                    ),
                    negative_points_xy=_spaced_points(
                        local_negative,
                        negative_quality,
                        config.negative_points_per_candidate,
                        ensemble_index * 3 + config.random_seed,
                    ),
                    use_coarse_mask_prompt=config.use_coarse_mask_prompt,
                )
            )
    return PhysicsPromptPlan(
        prompts=prompts,
        physics_core=physics_core,
        negative_reference=negative_reference,
        component_labels=component_labels.astype(np.int32),
        padding_pixels=padding,
        padding_source=padding_source,
    )


def build_interface_prompt_plan(
    coarse_mask: np.ndarray,
    score_map: np.ndarray,
    domain_labels: np.ndarray,
    valid_mask: np.ndarray,
    metadata: Dict[str, Any],
    config: SAM3RefinementConfig,
) -> PhysicsPromptPlan:
    """Prompt both domains; SAM never fills the interface band directly."""

    base = build_physics_prompt_plan(
        coarse_mask, score_map, metadata, config
    )
    domains = np.asarray(domain_labels)
    valid = np.asarray(valid_mask, dtype=bool)
    if domains.shape != base.component_labels.shape or valid.shape != domains.shape:
        raise ValueError(
            "Phase-domain labels, validity, and coarse interface must align"
        )
    if not np.any((domains == 0) & valid) or not np.any(
        (domains == 1) & valid
    ):
        raise ValueError("Paired-domain prompting requires two valid domains")

    prompts: List[SAM3Prompt] = []
    for component_label in range(
        1, int(base.component_labels.max()) + 1
    ):
        component = base.component_labels == component_label
        candidate_id = f"candidate-{component_label:04d}"
        for ensemble_index in range(config.ensemble_size):
            box = _expanded_box(
                component,
                base.padding_pixels,
                ensemble_index,
                config.ensemble_size,
            )
            x1, y1, x2, y2 = box
            local = np.zeros(component.shape, dtype=bool)
            local[y1:y2, x1:x2] = True
            for domain_value, role, opposite_value in (
                (0, "domain_a", 1),
                (1, "domain_b", 0),
            ):
                positive_seed = (
                    (domains == domain_value) & valid & local
                )
                negative_seed = (
                    (domains == opposite_value) & valid & local
                )
                if not np.any(positive_seed) or not np.any(negative_seed):
                    raise ValueError(
                        f"Interface candidate {candidate_id} lacks both "
                        f"domains inside prompt box {box}"
                    )
                positive_quality = distance_transform_edt(positive_seed)
                negative_quality = distance_transform_edt(negative_seed)
                prompts.append(
                    SAM3Prompt(
                        candidate_id=candidate_id,
                        component_label=component_label,
                        ensemble_index=ensemble_index,
                        box_xyxy=box,
                        positive_points_xy=_spaced_points(
                            positive_seed,
                            positive_quality,
                            config.positive_points_per_candidate,
                            ensemble_index
                            + domain_value * config.ensemble_size
                            + config.random_seed,
                        ),
                        negative_points_xy=_spaced_points(
                            negative_seed,
                            negative_quality,
                            config.negative_points_per_candidate,
                            ensemble_index * 3
                            + domain_value * config.ensemble_size
                            + config.random_seed,
                        ),
                        use_coarse_mask_prompt=(
                            config.use_coarse_mask_prompt
                        ),
                        prompt_role=role,
                        geometry_kind="paired_domains",
                    )
                )
    return PhysicsPromptPlan(
        prompts=prompts,
        physics_core=base.physics_core,
        negative_reference=base.negative_reference,
        component_labels=base.component_labels,
        padding_pixels=base.padding_pixels,
        padding_source=base.padding_source,
        geometry_kind="paired_domains",
        domain_labels=domains.astype(np.int8),
        valid_mask=valid.copy(),
    )


def _mean_iou(masks: Sequence[np.ndarray], reference: np.ndarray) -> float:
    if not masks:
        return 0.0
    values = []
    for mask in masks:
        union = np.logical_or(mask, reference).sum()
        intersection = np.logical_and(mask, reference).sum()
        values.append(float(intersection / union) if union else 1.0)
    return float(np.mean(values))


def _has_component_merge(
    coarse_mask: np.ndarray, refined_mask: np.ndarray
) -> bool:
    coarse_labels = label(coarse_mask, connectivity=2)
    if int(coarse_labels.max()) <= 1:
        return False
    refined_labels = label(refined_mask, connectivity=2)
    for refined_label in range(1, int(refined_labels.max()) + 1):
        overlapped = np.unique(
            coarse_labels[refined_labels == refined_label]
        )
        if np.count_nonzero(overlapped) > 1:
            return True
    return False


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=bool)
    return values & ~binary_erosion(values)


def _boundary_evidence_alignment(
    mask: np.ndarray, score_map: np.ndarray
) -> float:
    boundary = _mask_boundary(mask)
    finite = np.asarray(score_map, dtype=float)
    finite_values = finite[np.isfinite(finite)]
    if not np.any(boundary) or not finite_values.size:
        return float("nan")
    fill = float(np.median(finite_values))
    safe = np.where(np.isfinite(finite), finite, fill)
    gradient_y, gradient_x = np.gradient(safe)
    gradient = np.hypot(gradient_y, gradient_x)
    scale = float(np.quantile(gradient, 0.95))
    if scale <= 1e-12:
        return float("nan")
    return float(np.mean(np.clip(gradient[boundary] / scale, 0, 1)))


def evaluate_sam3_proposals(
    coarse_mask: np.ndarray,
    score_map: np.ndarray,
    physics_core: np.ndarray,
    negative_reference: np.ndarray,
    proposals: Sequence[SAM3Proposal],
    prompts: Sequence[SAM3Prompt],
    config: SAM3RefinementConfig,
    *,
    provenance: Dict[str, Any] | None = None,
) -> BoundaryRefinementResult:
    """Form prompt consensus, apply Policy A gates, and choose final geometry."""

    coarse = np.asarray(coarse_mask, dtype=bool)
    score = np.asarray(score_map, dtype=float)
    core = np.asarray(physics_core, dtype=bool)
    negative = np.asarray(negative_reference, dtype=bool)
    if not proposals:
        return fallback_boundary_refinement(
            coarse,
            core,
            negative,
            prompts,
            "no_valid_sam3_proposals",
            provenance=provenance,
        )
    masks = []
    for proposal in proposals:
        mask = np.asarray(proposal.mask, dtype=bool)
        if mask.shape != coarse.shape:
            raise ValueError(
                f"Proposal {proposal.proposal_id} shape {mask.shape} "
                f"does not match {coarse.shape}"
            )
        masks.append(mask)
    inclusion = np.mean(np.stack(masks, axis=0), axis=0).astype(np.float32)
    refined = inclusion >= config.selection_threshold
    stable = inclusion >= config.stable_core_threshold
    possible = inclusion >= config.possible_extent_threshold
    uncertainty_band = possible & ~stable
    uncertainty = np.where(
        uncertainty_band,
        1.0 - np.abs(2.0 * inclusion - 1.0),
        0.0,
    ).astype(np.float32)

    core_count = int(core.sum())
    core_recall = (
        float(np.logical_and(refined, core).sum() / core_count)
        if core_count
        else 1.0
    )
    negative_count = int(negative.sum())
    negative_overlap = int(np.logical_and(refined, negative).sum())
    coarse_area = int(coarse.sum())
    refined_area = int(refined.sum())
    negative_leakage = (
        float(negative_overlap / refined_area)
        if refined_area
        else 0.0
    )
    area_ratio = float(refined_area / coarse_area) if coarse_area else 0.0
    inside_values = score[refined & np.isfinite(score)]
    outside_values = score[negative & np.isfinite(score)]
    physics_contrast = (
        float(np.mean(inside_values) - np.mean(outside_values))
        if inside_values.size and outside_values.size
        else float("nan")
    )
    prompt_stability = _mean_iou(masks, refined)
    component_merge = _has_component_merge(coarse, refined)
    boundary_alignment = _boundary_evidence_alignment(refined, score)
    coarse_component_count = int(label(coarse, connectivity=2).max())
    refined_component_count = int(label(refined, connectivity=2).max())
    topology_changed = (
        coarse_component_count != refined_component_count
    )

    reasons: List[str] = []
    if core_recall < config.minimum_seed_core_recall:
        reasons.append(
            "seed_core_recall_below_"
            f"{config.minimum_seed_core_recall:.3f}"
        )
    if negative_leakage > config.maximum_negative_leakage:
        reasons.append(
            "negative_leakage_above_"
            f"{config.maximum_negative_leakage:.3f}"
        )
    if area_ratio < config.minimum_area_ratio:
        reasons.append(
            f"area_ratio_below_{config.minimum_area_ratio:.3f}"
        )
    if area_ratio > config.maximum_area_ratio:
        reasons.append(
            f"area_ratio_above_{config.maximum_area_ratio:.3f}"
        )
    if (
        math.isfinite(physics_contrast)
        and physics_contrast < config.minimum_physics_contrast
    ):
        reasons.append(
            "physics_contrast_below_"
            f"{config.minimum_physics_contrast:.3f}"
        )
    if prompt_stability < config.minimum_prompt_stability:
        reasons.append(
            "prompt_stability_below_"
            f"{config.minimum_prompt_stability:.3f}"
        )
    if (
        math.isfinite(boundary_alignment)
        and boundary_alignment
        < config.minimum_boundary_evidence_alignment
    ):
        reasons.append(
            "boundary_evidence_alignment_below_"
            f"{config.minimum_boundary_evidence_alignment:.3f}"
        )
    if component_merge and not config.allow_component_merging:
        reasons.append("unjustified_component_merge")
    if topology_changed and config.preserve_candidate_topology:
        reasons.append("candidate_topology_changed")

    accepted = not reasons
    final_mask = refined.copy() if accepted else coarse.copy()
    metrics = {
        "seed_core_recall": core_recall,
        "minimum_seed_core_recall": config.minimum_seed_core_recall,
        "negative_leakage": negative_leakage,
        "negative_reference_coverage": (
            float(negative_overlap / negative_count)
            if negative_count
            else 0.0
        ),
        "maximum_negative_leakage": config.maximum_negative_leakage,
        "area_ratio": area_ratio,
        "area_ratio_range": [
            config.minimum_area_ratio,
            config.maximum_area_ratio,
        ],
        "physics_score_contrast": (
            physics_contrast
            if math.isfinite(physics_contrast)
            else None
        ),
        "minimum_physics_contrast": config.minimum_physics_contrast,
        "prompt_stability_mean_iou": prompt_stability,
        "minimum_prompt_stability": config.minimum_prompt_stability,
        "boundary_evidence_alignment": (
            boundary_alignment
            if math.isfinite(boundary_alignment)
            else None
        ),
        "minimum_boundary_evidence_alignment": (
            config.minimum_boundary_evidence_alignment
        ),
        "component_merge_detected": component_merge,
        "coarse_component_count": coarse_component_count,
        "refined_component_count": refined_component_count,
        "candidate_topology_changed": topology_changed,
        "proposal_scores": [float(item.score) for item in proposals],
    }
    return BoundaryRefinementResult(
        coarse_mask=coarse.copy(),
        proposals=list(proposals),
        inclusion_probability=inclusion,
        refined_mask=refined,
        stable_core=stable,
        possible_extent=possible,
        boundary_uncertainty=uncertainty,
        final_mask=final_mask,
        physics_core=core.copy(),
        negative_reference=negative.copy(),
        accepted=accepted,
        status="accepted" if accepted else "rejected_fallback",
        rejection_reasons=reasons,
        metrics=metrics,
        prompts=list(prompts),
        provenance=provenance or {},
    )


def _candidate_prompt_groups(
    plan: PhysicsPromptPlan,
) -> Dict[str, Tuple[int, List[SAM3Prompt]]]:
    grouped: Dict[str, Tuple[int, List[SAM3Prompt]]] = {}
    component_owners: Dict[int, str] = {}
    for prompt in plan.prompts:
        old_owner = component_owners.setdefault(
            prompt.component_label, prompt.candidate_id
        )
        if old_owner != prompt.candidate_id:
            raise ValueError(
                f"Component {prompt.component_label} is owned by multiple "
                "SAM3 candidate ids"
            )
        if prompt.candidate_id not in grouped:
            grouped[prompt.candidate_id] = (prompt.component_label, [])
        component_label, prompts = grouped[prompt.candidate_id]
        if component_label != prompt.component_label:
            raise ValueError(
                f"SAM3 candidate {prompt.candidate_id!r} spans multiple "
                "physical components"
            )
        prompts.append(prompt)
    return grouped


def _candidate_negative_reference(
    shape: Tuple[int, int],
    prompts: Sequence[SAM3Prompt],
    negative_reference: np.ndarray,
) -> np.ndarray:
    local_zone = np.zeros(shape, dtype=bool)
    for prompt in prompts:
        x1, y1, x2, y2 = prompt.box_xyxy
        local_zone[y1:y2, x1:x2] = True
    local_negative = (
        np.asarray(negative_reference, dtype=bool) & local_zone
    )
    if np.any(local_negative):
        return local_negative
    return np.asarray(negative_reference, dtype=bool).copy()


def _shared_interface_from_domains(
    domain_a: np.ndarray,
    domain_b: np.ndarray,
    roi: np.ndarray,
    valid_mask: np.ndarray,
    *,
    maximum_gap_pixels: int,
    half_width_pixels: int,
) -> np.ndarray:
    """Derive a shared line/band from two region masks."""

    valid_roi = np.asarray(roi, dtype=bool) & np.asarray(
        valid_mask, dtype=bool
    )
    a = np.asarray(domain_a, dtype=bool) & valid_roi
    b = np.asarray(domain_b, dtype=bool) & valid_roi
    a_only = a & ~b
    b_only = b & ~a
    if not np.any(a_only) or not np.any(b_only):
        return np.zeros(a.shape, dtype=bool)
    distance_to_a = distance_transform_edt(~a_only)
    distance_to_b = distance_transform_edt(~b_only)
    near_both = (
        (distance_to_a <= maximum_gap_pixels)
        & (distance_to_b <= maximum_gap_pixels)
        & valid_roi
    )
    seam = near_both & (
        np.abs(distance_to_a - distance_to_b) <= 1.5
    )
    if not np.any(seam):
        boundary_a = a_only & ~binary_erosion(a_only)
        boundary_b = b_only & ~binary_erosion(b_only)
        seam = (
            (
                boundary_a
                & (distance_transform_edt(~boundary_b) <= maximum_gap_pixels)
            )
            | (
                boundary_b
                & (distance_transform_edt(~boundary_a) <= maximum_gap_pixels)
            )
        ) & valid_roi
    if half_width_pixels:
        seam = binary_dilation(
            seam, iterations=int(half_width_pixels)
        )
    return seam & valid_roi


def interface_proposals_from_paired_domains(
    plan: PhysicsPromptPlan,
    proposals: Sequence[SAM3Proposal],
    config: SAM3RefinementConfig,
) -> List[SAM3Proposal]:
    """Convert domain-A/domain-B proposals into candidate interface bands."""

    if plan.geometry_kind != "paired_domains":
        return list(proposals)
    valid = (
        np.asarray(plan.valid_mask, dtype=bool)
        if plan.valid_mask is not None
        else np.ones(plan.component_labels.shape, dtype=bool)
    )
    prompts_by_key = {
        (
            prompt.candidate_id,
            prompt.ensemble_index,
            prompt.prompt_role,
        ): prompt
        for prompt in plan.prompts
    }
    proposals_by_key = {
        (
            proposal.candidate_id,
            proposal.ensemble_index,
            proposal.prompt_role,
        ): proposal
        for proposal in proposals
    }
    candidate_ensembles = sorted(
        {
            (prompt.candidate_id, prompt.ensemble_index)
            for prompt in plan.prompts
        }
    )
    interface_proposals: List[SAM3Proposal] = []
    for candidate_id, ensemble_index in candidate_ensembles:
        key_a = (candidate_id, ensemble_index, "domain_a")
        key_b = (candidate_id, ensemble_index, "domain_b")
        proposal_a = proposals_by_key.get(key_a)
        proposal_b = proposals_by_key.get(key_b)
        prompt_a = prompts_by_key.get(key_a)
        prompt_b = prompts_by_key.get(key_b)
        if (
            proposal_a is None
            or proposal_b is None
            or prompt_a is None
            or prompt_b is None
        ):
            continue
        roi = np.zeros(plan.component_labels.shape, dtype=bool)
        for prompt in (prompt_a, prompt_b):
            x1, y1, x2, y2 = prompt.box_xyxy
            roi[y1:y2, x1:x2] = True
        interface_mask = _shared_interface_from_domains(
            proposal_a.mask,
            proposal_b.mask,
            roi,
            valid,
            maximum_gap_pixels=config.interface_max_gap_pixels,
            half_width_pixels=config.interface_half_width_pixels,
        )
        interface_proposals.append(
            SAM3Proposal(
                proposal_id=(
                    f"interface:{candidate_id}:{ensemble_index:03d}"
                ),
                mask=interface_mask,
                score=float(0.5 * (proposal_a.score + proposal_b.score)),
                ensemble_index=ensemble_index,
                candidate_id=candidate_id,
                component_label=prompt_a.component_label,
                prompt_role="interface",
                provenance={
                    "geometry_adapter": "paired_domains_to_shared_interface",
                    "domain_a_proposal_id": proposal_a.proposal_id,
                    "domain_b_proposal_id": proposal_b.proposal_id,
                },
            )
        )
    return interface_proposals


def evaluate_candidate_sam3_proposals(
    coarse_mask: np.ndarray,
    score_map: np.ndarray,
    plan: PhysicsPromptPlan,
    proposals: Sequence[SAM3Proposal],
    config: SAM3RefinementConfig,
    *,
    provenance: Dict[str, Any] | None = None,
) -> BoundaryRefinementResult:
    """Gate each physical candidate independently, then compose the image.

    A failed candidate contributes its own untouched coarse component.  It
    never replaces a neighboring candidate that passed the SAM3 guardrails.
    """

    coarse = np.asarray(coarse_mask, dtype=bool)
    score = np.asarray(score_map, dtype=float)
    if coarse.shape != score.shape:
        raise ValueError("coarse_mask and score_map must have identical shapes")
    if plan.component_labels.shape != coarse.shape:
        raise ValueError("Prompt-plan component labels do not match coarse mask")
    prompt_groups = _candidate_prompt_groups(plan)
    if not prompt_groups:
        return fallback_boundary_refinement(
            coarse,
            plan.physics_core,
            plan.negative_reference,
            plan.prompts,
            "no_candidate_prompt_groups",
            provenance=provenance,
        )

    unknown_candidates = sorted(
        {
            proposal.candidate_id
            for proposal in proposals
            if proposal.candidate_id not in prompt_groups
        }
    )
    if unknown_candidates:
        raise ValueError(
            "SAM3 proposals contain unknown candidate ids: "
            + ", ".join(unknown_candidates)
        )

    candidate_results: List[CandidateBoundaryRefinement] = []
    for candidate_id, (component_label, candidate_prompts) in sorted(
        prompt_groups.items(), key=lambda item: item[1][0]
    ):
        candidate_coarse = plan.component_labels == component_label
        if not np.any(candidate_coarse):
            raise ValueError(
                f"Prompt component {component_label} has no coarse pixels"
            )
        candidate_core = (
            np.asarray(plan.physics_core, dtype=bool) & candidate_coarse
        )
        candidate_negative = _candidate_negative_reference(
            coarse.shape,
            candidate_prompts,
            plan.negative_reference,
        )
        candidate_proposals = [
            proposal
            for proposal in proposals
            if proposal.candidate_id == candidate_id
        ]
        if any(
            proposal.component_label != component_label
            for proposal in candidate_proposals
        ):
            raise ValueError(
                f"SAM3 proposal component label mismatch for {candidate_id}"
            )
        expected_indices = {
            prompt.ensemble_index for prompt in candidate_prompts
        }
        observed_indices = [
            proposal.ensemble_index for proposal in candidate_proposals
        ]
        duplicate_indices = len(observed_indices) != len(
            set(observed_indices)
        )
        incomplete_ensemble = set(observed_indices) != expected_indices
        if duplicate_indices or incomplete_ensemble:
            local = fallback_boundary_refinement(
                candidate_coarse,
                candidate_core,
                candidate_negative,
                candidate_prompts,
                "candidate_proposal_ensemble_incomplete",
                status="rejected_fallback",
            )
        else:
            local = evaluate_sam3_proposals(
                candidate_coarse,
                score,
                candidate_core,
                candidate_negative,
                candidate_proposals,
                candidate_prompts,
                config,
            )

        forbidden_neighbors = coarse & ~candidate_coarse
        neighbor_overlap = int(
            np.logical_and(local.refined_mask, forbidden_neighbors).sum()
        )
        local.metrics["neighbor_candidate_overlap_pixels"] = neighbor_overlap
        if (
            neighbor_overlap
            and not config.allow_component_merging
            and "unjustified_neighbor_candidate_overlap"
            not in local.rejection_reasons
        ):
            local.rejection_reasons.append(
                "unjustified_neighbor_candidate_overlap"
            )
            local.accepted = False
            local.status = "rejected_fallback"
            local.final_mask = candidate_coarse.copy()

        candidate_results.append(
            CandidateBoundaryRefinement(
                candidate_id=candidate_id,
                component_label=component_label,
                coarse_mask=candidate_coarse.copy(),
                proposals=list(candidate_proposals),
                inclusion_probability=local.inclusion_probability.copy(),
                refined_mask=local.refined_mask.copy(),
                stable_core=local.stable_core.copy(),
                possible_extent=local.possible_extent.copy(),
                boundary_uncertainty=local.boundary_uncertainty.copy(),
                final_mask=local.final_mask.copy(),
                physics_core=candidate_core.copy(),
                negative_reference=candidate_negative.copy(),
                accepted=local.accepted,
                status=local.status,
                rejection_reasons=list(local.rejection_reasons),
                metrics=dict(local.metrics),
                prompts=list(candidate_prompts),
            )
        )

    if not config.allow_component_merging:
        collision_pixels: Dict[str, int] = {
            item.candidate_id: 0 for item in candidate_results
        }
        for left_index, left in enumerate(candidate_results):
            if not left.accepted:
                continue
            for right in candidate_results[left_index + 1 :]:
                if not right.accepted:
                    continue
                overlap = int(
                    np.logical_and(
                        left.final_mask, right.final_mask
                    ).sum()
                )
                if overlap:
                    collision_pixels[left.candidate_id] += overlap
                    collision_pixels[right.candidate_id] += overlap
        for candidate in candidate_results:
            overlap = collision_pixels[candidate.candidate_id]
            candidate.metrics[
                "composed_candidate_overlap_pixels"
            ] = overlap
            if overlap:
                candidate.accepted = False
                candidate.status = "rejected_fallback"
                candidate.rejection_reasons.append(
                    "composed_candidate_overlap"
                )
                candidate.final_mask = candidate.coarse_mask.copy()

    inclusion = np.zeros(coarse.shape, dtype=np.float32)
    refined = np.zeros(coarse.shape, dtype=bool)
    stable = np.zeros(coarse.shape, dtype=bool)
    possible = np.zeros(coarse.shape, dtype=bool)
    uncertainty = np.zeros(coarse.shape, dtype=np.float32)
    final_mask = np.zeros(coarse.shape, dtype=bool)
    for candidate in candidate_results:
        inclusion = np.maximum(
            inclusion, candidate.inclusion_probability
        )
        refined |= candidate.refined_mask
        stable |= candidate.stable_core
        possible |= candidate.possible_extent
        uncertainty = np.maximum(
            uncertainty, candidate.boundary_uncertainty
        )
        final_mask |= candidate.final_mask

    accepted_count = sum(item.accepted for item in candidate_results)
    if accepted_count == len(candidate_results):
        status = "accepted"
    elif accepted_count:
        status = "partially_accepted"
    else:
        status = "rejected_fallback"
    rejection_reasons = [
        f"{item.candidate_id}:{reason}"
        for item in candidate_results
        for reason in item.rejection_reasons
    ]
    metrics = {
        "candidate_count": len(candidate_results),
        "accepted_candidate_count": accepted_count,
        "fallback_candidate_count": len(candidate_results) - accepted_count,
        "candidate_local_fallback": True,
        "fallback_equality_by_candidate": {
            item.candidate_id: bool(
                np.array_equal(item.final_mask, item.coarse_mask)
            )
            for item in candidate_results
            if not item.accepted
        },
        "final_selection_valid_by_candidate": {
            item.candidate_id: bool(
                item.accepted
                or np.array_equal(item.final_mask, item.coarse_mask)
            )
            for item in candidate_results
        },
    }
    result_provenance = dict(provenance or {})
    result_provenance["fallback_scope"] = "candidate"
    return BoundaryRefinementResult(
        coarse_mask=coarse.copy(),
        proposals=list(proposals),
        inclusion_probability=inclusion,
        refined_mask=refined,
        stable_core=stable,
        possible_extent=possible,
        boundary_uncertainty=uncertainty,
        final_mask=final_mask,
        physics_core=np.asarray(plan.physics_core, dtype=bool).copy(),
        negative_reference=np.asarray(
            plan.negative_reference, dtype=bool
        ).copy(),
        accepted=accepted_count == len(candidate_results),
        status=status,
        rejection_reasons=rejection_reasons,
        metrics=metrics,
        prompts=list(plan.prompts),
        candidate_results=candidate_results,
        provenance=result_provenance,
    )


def evaluate_refinement_proposals(
    coarse_mask: np.ndarray,
    score_map: np.ndarray,
    plan: PhysicsPromptPlan,
    proposals: Sequence[SAM3Proposal],
    config: SAM3RefinementConfig,
    *,
    provenance: Dict[str, Any] | None = None,
) -> BoundaryRefinementResult:
    """Dispatch region or paired-domain proposals to candidate-local gates."""

    adapted = interface_proposals_from_paired_domains(
        plan, proposals, config
    )
    result_provenance = dict(provenance or {})
    result_provenance.update(
        {
            "geometry_kind": plan.geometry_kind,
            "raw_worker_proposal_count": len(proposals),
            "gated_geometry_proposal_count": len(adapted),
        }
    )
    return evaluate_candidate_sam3_proposals(
        coarse_mask,
        score_map,
        plan,
        adapted,
        config,
        provenance=result_provenance,
    )


def fallback_boundary_refinement(
    coarse_mask: np.ndarray,
    physics_core: np.ndarray,
    negative_reference: np.ndarray,
    prompts: Sequence[SAM3Prompt],
    reason: str,
    *,
    status: str = "fallback",
    provenance: Dict[str, Any] | None = None,
) -> BoundaryRefinementResult:
    """Build a deterministic coarse-mask fallback without hiding the reason."""

    coarse = np.asarray(coarse_mask, dtype=bool)
    final_mask = coarse.copy()
    zeros_float = np.zeros(coarse.shape, dtype=np.float32)
    zeros_bool = np.zeros(coarse.shape, dtype=bool)
    return BoundaryRefinementResult(
        coarse_mask=coarse.copy(),
        proposals=[],
        inclusion_probability=zeros_float,
        refined_mask=zeros_bool,
        stable_core=zeros_bool.copy(),
        possible_extent=zeros_bool.copy(),
        boundary_uncertainty=zeros_float.copy(),
        final_mask=final_mask,
        physics_core=np.asarray(physics_core, dtype=bool).copy(),
        negative_reference=np.asarray(
            negative_reference, dtype=bool
        ).copy(),
        accepted=False,
        status=status,
        rejection_reasons=[reason],
        metrics={
            "seed_core_recall": None,
            "fallback_equality_verified": bool(
                np.array_equal(final_mask, coarse)
            ),
        },
        prompts=list(prompts),
        provenance=provenance or {},
    )


def _request_payload(
    image: np.ndarray,
    coarse_mask: np.ndarray,
    score_map: np.ndarray,
    plan: PhysicsPromptPlan,
    config: SAM3RefinementConfig,
) -> Dict[str, Any]:
    arrays = {
        "image": {
            "file": "image.npy",
            "sha256": _array_sha256(image),
            "dtype": str(image.dtype),
            "shape": list(image.shape),
        },
        "coarse_mask": {
            "file": "coarse_mask.npy",
            "sha256": _array_sha256(coarse_mask),
            "dtype": str(coarse_mask.dtype),
            "shape": list(coarse_mask.shape),
        },
        "score_map": {
            "file": "score_map.npy",
            "sha256": _array_sha256(score_map),
            "dtype": str(score_map.dtype),
            "shape": list(score_map.shape),
        },
        "physics_core": {
            "file": "physics_core.npy",
            "sha256": _array_sha256(plan.physics_core),
            "dtype": str(plan.physics_core.dtype),
            "shape": list(plan.physics_core.shape),
        },
        "negative_reference": {
            "file": "negative_reference.npy",
            "sha256": _array_sha256(plan.negative_reference),
            "dtype": str(plan.negative_reference.dtype),
            "shape": list(plan.negative_reference.shape),
        },
        "component_labels": {
            "file": "component_labels.npy",
            "sha256": _array_sha256(plan.component_labels),
            "dtype": str(plan.component_labels.dtype),
            "shape": list(plan.component_labels.shape),
        },
    }
    if plan.domain_labels is not None:
        arrays["domain_labels"] = {
            "file": "domain_labels.npy",
            "sha256": _array_sha256(plan.domain_labels),
            "dtype": str(plan.domain_labels.dtype),
            "shape": list(plan.domain_labels.shape),
        }
    if plan.valid_mask is not None:
        arrays["valid_mask"] = {
            "file": "valid_mask.npy",
            "sha256": _array_sha256(plan.valid_mask),
            "dtype": str(plan.valid_mask.dtype),
            "shape": list(plan.valid_mask.shape),
        }
    worker_identity = {
        "backend": config.backend,
        "checkpoint_path": config.checkpoint_path,
        "model_variant": config.model_variant,
        "device": config.device,
    }
    identity = {
        "protocol_version": SAM3_PROTOCOL_VERSION,
        "arrays": arrays,
        "prompts": [prompt.to_dict() for prompt in plan.prompts],
        "ensemble_size": config.ensemble_size,
        "use_coarse_mask_prompt": config.use_coarse_mask_prompt,
        "model_variant": config.model_variant,
        "geometry_kind": plan.geometry_kind,
        "worker": worker_identity,
    }
    request_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:24]
    return {
        **identity,
        "request_id": request_id,
        "response_file": "sam_response.json",
        "physics_prompt_metadata": {
            "padding_pixels": plan.padding_pixels,
            "padding_source": plan.padding_source,
            "core_score_quantile": config.core_score_quantile,
        },
    }


def prepare_sam3_exchange(
    exchange_directory: str | Path,
    image: np.ndarray,
    coarse_mask: np.ndarray,
    score_map: np.ndarray,
    plan: PhysicsPromptPlan,
    config: SAM3RefinementConfig,
    *,
    reuse_existing: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Write or validate the versioned file protocol for an isolated worker."""

    exchange = _safe_exchange_directory(exchange_directory)
    exchange.mkdir(parents=True, exist_ok=True)
    image_values = np.asarray(image, dtype=np.float32)
    coarse_values = np.asarray(coarse_mask, dtype=bool)
    score_values = np.asarray(score_map, dtype=np.float32)
    request = _request_payload(
        image_values, coarse_values, score_values, plan, config
    )
    request_path = exchange / "sam_request.json"
    if reuse_existing:
        if not request_path.is_file():
            raise FileNotFoundError(
                f"Existing SAM3 request not found: {request_path}"
            )
        with request_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing.get("request_id") != request["request_id"]:
            raise ValueError(
                "Existing SAM3 request does not match the current image, "
                "coarse mask, score map, prompts, and model policy"
            )
        return request_path, existing

    arrays = {
        "image.npy": image_values,
        "coarse_mask.npy": coarse_values,
        "score_map.npy": score_values,
        "physics_core.npy": plan.physics_core.astype(bool),
        "negative_reference.npy": plan.negative_reference.astype(bool),
        "component_labels.npy": plan.component_labels.astype(np.int32),
    }
    if plan.domain_labels is not None:
        arrays["domain_labels.npy"] = plan.domain_labels.astype(np.int8)
    if plan.valid_mask is not None:
        arrays["valid_mask.npy"] = plan.valid_mask.astype(bool)
    for filename, values in arrays.items():
        _write_array_exclusive(exchange / filename, values)
    _write_json_exclusive(request_path, request)
    return request_path, request


def _resolve_protocol_file(exchange: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError("SAM3 protocol paths must be relative")
    resolved = (exchange / path).resolve()
    if not resolved.is_relative_to(exchange.resolve()):
        raise ValueError("SAM3 protocol path traversal was rejected")
    return resolved


def load_sam3_response(
    request_path: str | Path,
    request: Dict[str, Any],
) -> Tuple[List[SAM3Proposal], Dict[str, Any]]:
    """Load and verify every proposal file from the isolated process."""

    request_file = Path(request_path).resolve()
    exchange = request_file.parent
    response_path = _resolve_protocol_file(
        exchange, str(request["response_file"])
    )
    with response_path.open("r", encoding="utf-8") as stream:
        response = json.load(stream)
    if response.get("protocol_version") != SAM3_PROTOCOL_VERSION:
        raise ValueError("SAM3 response protocol version mismatch")
    if response.get("request_id") != request.get("request_id"):
        raise ValueError("SAM3 response request_id mismatch")
    expected_shape = tuple(request["arrays"]["coarse_mask"]["shape"])
    records = response.get("proposals", [])
    request_prompts = request.get("prompts", [])
    expected_pairs = {
        (
            str(item["candidate_id"]),
            int(item["ensemble_index"]),
            str(item.get("prompt_role", "candidate")),
        )
        for item in request_prompts
    }
    if len(expected_pairs) != len(request_prompts):
        raise ValueError(
            "SAM3 request contains duplicate candidate/ensemble prompts"
        )
    expected_count = len(expected_pairs)
    if len(records) != expected_count:
        raise ValueError(
            f"SAM3 response contains {len(records)} proposals; "
            f"expected {expected_count}"
        )
    observed_pairs = [
        (
            str(item.get("candidate_id", "")),
            int(item["ensemble_index"]),
            str(item.get("prompt_role", "candidate")),
        )
        for item in records
    ]
    if len(set(observed_pairs)) != len(observed_pairs):
        raise ValueError(
            "SAM3 response candidate/ensemble pairs must be unique"
        )
    if set(observed_pairs) != expected_pairs:
        raise ValueError(
            "SAM3 response candidate/ensemble pairs do not match the request"
        )
    expected_components = {
        (
            str(item["candidate_id"]),
            int(item["ensemble_index"]),
            str(item.get("prompt_role", "candidate")),
        ): int(item["component_label"])
        for item in request_prompts
    }
    proposals: List[SAM3Proposal] = []
    seen_paths = set()
    for index, record in enumerate(records):
        path = _resolve_protocol_file(exchange, str(record["mask_file"]))
        if path in seen_paths:
            raise ValueError("SAM3 response reuses a proposal mask file")
        seen_paths.add(path)
        if not path.is_file():
            raise FileNotFoundError(f"SAM3 proposal file missing: {path}")
        if _file_sha256(path) != record.get("sha256"):
            raise ValueError(f"SAM3 proposal checksum mismatch: {path.name}")
        mask = np.load(path, allow_pickle=False)
        if mask.shape != expected_shape:
            raise ValueError(
                f"SAM3 proposal {path.name} has shape {mask.shape}; "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"SAM3 proposal {path.name} is non-finite")
        score = float(record.get("score", float("nan")))
        if not math.isfinite(score):
            raise ValueError(f"SAM3 proposal {path.name} has invalid score")
        candidate_id = str(record.get("candidate_id", ""))
        ensemble_index = int(record.get("ensemble_index", index))
        prompt_role = str(record.get("prompt_role", "candidate"))
        component_label = int(record.get("component_label", 0))
        expected_component = expected_components[
            (candidate_id, ensemble_index, prompt_role)
        ]
        if component_label != expected_component:
            raise ValueError(
                f"SAM3 proposal {path.name} component label mismatch"
            )
        proposals.append(
            SAM3Proposal(
                proposal_id=str(
                    record.get("proposal_id", f"proposal-{index:03d}")
                ),
                mask=np.asarray(mask, dtype=bool),
                score=score,
                ensemble_index=ensemble_index,
                candidate_id=candidate_id,
                component_label=component_label,
                prompt_role=prompt_role,
                provenance={
                    "mask_file": str(path),
                    "sha256": record["sha256"],
                    "worker": response.get("worker", {}),
                },
            )
        )
    if not proposals:
        raise ValueError("SAM3 response contains no proposals")
    return proposals, response


def run_isolated_sam3_worker(
    request_path: str | Path,
    config: SAM3RefinementConfig,
) -> Dict[str, Any]:
    """Invoke the project-pinned worker without a shell."""

    if not config.worker_python:
        raise ValueError(
            "stage2.sam3.worker_python must point to the isolated Python "
            "3.12 SAM3 environment"
        )
    worker_python = Path(config.worker_python)
    if not worker_python.is_absolute():
        worker_python = PROJECT_ROOT / worker_python
    worker_script = Path(config.worker_script)
    if not worker_script.is_absolute():
        worker_script = PROJECT_ROOT / worker_script
    worker_python = worker_python.resolve()
    worker_script = worker_script.resolve()
    if not worker_python.is_file():
        raise FileNotFoundError(f"SAM3 worker Python not found: {worker_python}")
    if (
        not worker_script.is_file()
        or not worker_script.is_relative_to(PROJECT_ROOT)
    ):
        raise FileNotFoundError(
            "SAM3 worker script must be an existing project file: "
            f"{worker_script}"
        )
    started_at = time.perf_counter()
    completed = subprocess.run(
        [
            str(worker_python),
            str(worker_script),
            "--request",
            str(Path(request_path).resolve()),
        ],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
        timeout=config.timeout_seconds,
    )
    exchange = Path(request_path).resolve().parent
    with (exchange / "sam3_worker_stdout.log").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(completed.stdout)
    with (exchange / "sam3_worker_stderr.log").open(
        "x", encoding="utf-8"
    ) as stream:
        stream.write(completed.stderr)
    if completed.returncode:
        raise RuntimeError(
            f"SAM3 worker exited with code {completed.returncode}; "
            "inspect sam3_worker_stderr.log"
        )
    return {
        "worker_python": str(worker_python),
        "worker_script": str(worker_script),
        "returncode": completed.returncode,
        "subprocess_runtime_seconds": float(
            time.perf_counter() - started_at
        ),
    }


def _attach_refinement(
    result: DetectionResult, refinement: BoundaryRefinementResult
) -> DetectionResult:
    result.boundary_refinement = refinement
    result.metadata["sam3_boundary_refinement"] = refinement.to_summary()
    return result


def refine_detection_result_with_sam3(
    result: DetectionResult,
    config: SAM3RefinementConfig,
    exchange_directory: str | Path | None,
) -> DetectionResult:
    """Prepare, run/load, gate, and attach a SAM3 boundary proposal."""

    if not config.enabled:
        return result
    coarse = np.asarray(result.mask, dtype=bool)
    empty = np.zeros(coarse.shape, dtype=bool)
    if result.task not in config.supported_tasks:
        return _attach_refinement(
            result,
            fallback_boundary_refinement(
                coarse,
                empty,
                ~coarse,
                [],
                f"unsupported_task:{result.task}",
                status="skipped",
            ),
        )
    if not np.any(coarse):
        return _attach_refinement(
            result,
            fallback_boundary_refinement(
                coarse,
                empty,
                ~coarse,
                [],
                "no_coarse_candidates",
                status="skipped",
            ),
        )
    try:
        if result.task == "phase_boundary":
            plan = build_interface_prompt_plan(
                coarse,
                result.score_map,
                result.evidence_maps["domain_label"],
                result.evidence_maps["boundary_validity"] >= 0.5,
                result.metadata,
                config,
            )
        else:
            plan = build_physics_prompt_plan(
                coarse, result.score_map, result.metadata, config
            )
    except Exception as error:
        return _attach_refinement(
            result,
            fallback_boundary_refinement(
                coarse,
                empty,
                ~coarse,
                [],
                f"prompt_generation_failed:{type(error).__name__}:{error}",
            ),
        )
    selected_exchange = (
        config.exchange_directory
        if config.exchange_directory is not None
        else exchange_directory
    )
    if selected_exchange is None:
        return _attach_refinement(
            result,
            fallback_boundary_refinement(
                coarse,
                plan.physics_core,
                plan.negative_reference,
                plan.prompts,
                "exchange_directory_not_provided",
            ),
        )
    provenance: Dict[str, Any] = {
        "backend": config.backend,
        "model_variant": config.model_variant,
        "inference_mode": config.inference_mode,
    }
    try:
        request_path, request = prepare_sam3_exchange(
            selected_exchange,
            result.normalized_image,
            coarse,
            result.score_map,
            plan,
            config,
            reuse_existing=config.inference_mode == "response_only",
        )
        provenance.update(
            {
                "request_id": request["request_id"],
                "request_file": str(request_path),
                "exchange_directory": str(request_path.parent),
            }
        )
        if config.inference_mode == "prepare_only":
            return _attach_refinement(
                result,
                fallback_boundary_refinement(
                    coarse,
                    plan.physics_core,
                    plan.negative_reference,
                    plan.prompts,
                    "sam3_response_pending",
                    status="proposal_pending",
                    provenance=provenance,
                ),
            )
        if config.inference_mode == "isolated_process":
            provenance.update(run_isolated_sam3_worker(request_path, config))
        proposals, response = load_sam3_response(request_path, request)
        provenance["worker_response"] = response.get("worker", {})
        refinement = evaluate_refinement_proposals(
            coarse,
            result.score_map,
            plan,
            proposals,
            config,
            provenance=provenance,
        )
        return _attach_refinement(result, refinement)
    except Exception as error:
        provenance["error_type"] = type(error).__name__
        return _attach_refinement(
            result,
            fallback_boundary_refinement(
                coarse,
                plan.physics_core,
                plan.negative_reference,
                plan.prompts,
                f"sam3_inference_failed:{type(error).__name__}:{error}",
                provenance=provenance,
            ),
        )


__all__ = [
    "BoundaryRefinementResult",
    "CandidateBoundaryRefinement",
    "PhysicsPromptPlan",
    "SAM3Proposal",
    "SAM3Prompt",
    "SAM3_PROTOCOL_VERSION",
    "build_physics_prompt_plan",
    "build_interface_prompt_plan",
    "evaluate_candidate_sam3_proposals",
    "evaluate_refinement_proposals",
    "evaluate_sam3_proposals",
    "fallback_boundary_refinement",
    "load_sam3_response",
    "interface_proposals_from_paired_domains",
    "prepare_sam3_exchange",
    "refine_detection_result_with_sam3",
    "run_isolated_sam3_worker",
]
