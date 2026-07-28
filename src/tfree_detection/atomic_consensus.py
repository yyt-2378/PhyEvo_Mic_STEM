"""Unified two-physics-teacher consensus for atomic-site anomalies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from skimage.draw import disk

from .config import DetectorConfig
from .evidence import (
    DECISION_STATE_CODES,
    DecisionState,
    LabelTier,
    attach_evidence_protocol,
)
from .framework import TrainingFreeDetectionFramework
from .models import DetectionRegion, DetectionResult
from .periodic_counterfactual import (
    PeriodicCounterfactualConfig,
    detect_periodic_counterfactual,
)
from .preprocessing import load_grayscale
from .site_consensus import fuse_site_consensus


ATOMIC_CONSENSUS_VERSION = "1.0"


@dataclass(frozen=True)
class AtomicConsensusConfig:
    """Frozen fusion policy; no labels are read during inference."""

    matching_tolerance_fraction: float = 0.35
    render_radius_fraction: float = 0.25
    retain_single_teacher_candidates: bool = True

    def validate(self) -> None:
        if self.matching_tolerance_fraction <= 0:
            raise ValueError(
                "matching_tolerance_fraction must be positive"
            )
        if self.render_radius_fraction <= 0:
            raise ValueError("render_radius_fraction must be positive")

    def content_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(
                asdict(self), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()


def _input_provenance(
    source: str | Path | np.ndarray,
    image: np.ndarray,
    *,
    frame_index: int | None,
    projection: str | None,
    dataset_path: str | None,
) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(image)
    pixel_digest = hashlib.sha256()
    pixel_digest.update(str(contiguous.dtype).encode("utf-8"))
    pixel_digest.update(
        np.asarray(contiguous.shape, dtype=np.int64).tobytes()
    )
    pixel_digest.update(contiguous.view(np.uint8).tobytes())
    file_sha256 = None
    file_bytes = None
    source_name = "<array>"
    if isinstance(source, (str, Path)):
        target = Path(source).resolve()
        source_name = str(target)
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        file_sha256 = digest.hexdigest()
        file_bytes = target.stat().st_size
    pixel_sha256 = pixel_digest.hexdigest()
    return {
        "source": source_name,
        "input_provenance": {
            "source_image": source_name,
            "source_file_sha256": file_sha256,
            "source_file_bytes": file_bytes,
            "pixel_sha256": pixel_sha256,
            "source_id": pixel_sha256[:16],
            "raw_dtype": str(contiguous.dtype),
            "raw_shape": list(contiguous.shape),
            "stack_selection": {
                "frame_index": frame_index,
                "projection": projection,
                "dataset_path": dataset_path,
            },
        },
    }


def _spacing(result: DetectionResult) -> float:
    if "lattice_spacing_pixels" in result.metadata:
        return float(result.metadata["lattice_spacing_pixels"])
    return float(
        np.min(result.metadata["direct_basis_lengths_pixels"])
    )


def _region_points(result: DetectionResult) -> np.ndarray:
    return np.asarray(
        [region.centroid_xy for region in result.regions],
        dtype=float,
    ).reshape(-1, 2)


def _merge_evidence_maps(
    primary: DetectionResult, secondary: DetectionResult
) -> Dict[str, np.ndarray]:
    shape = primary.image_shape
    merged = {
        key: np.asarray(values).copy()
        for key, values in primary.evidence_maps.items()
    }
    periodic_intensity = secondary.evidence_maps.get("group_intensity")
    if periodic_intensity is not None:
        legacy_intensity = merged.get(
            "group_intensity", np.zeros(shape, dtype=np.float32)
        )
        merged["group_intensity"] = np.maximum(
            legacy_intensity, periodic_intensity
        ).astype(np.float32)
    periodic_peaks = secondary.evidence_maps.get("observed_atomic_peaks")
    if periodic_peaks is not None:
        legacy_peaks = merged.get(
            "observed_atomic_peaks", np.zeros(shape, dtype=np.float32)
        )
        merged["observed_atomic_peaks"] = np.maximum(
            legacy_peaks, periodic_peaks
        ).astype(np.float32)
    primary_reliability = merged.get("registration_reliability")
    secondary_reliability = secondary.evidence_maps.get(
        "registration_reliability"
    )
    if primary_reliability is not None and secondary_reliability is not None:
        merged["registration_reliability"] = np.minimum(
            primary_reliability, secondary_reliability
        ).astype(np.float32)
    elif secondary_reliability is not None:
        merged["registration_reliability"] = np.asarray(
            secondary_reliability, dtype=np.float32
        )
    return merged


def _candidate_region(
    *,
    region_id: int,
    point_xy: np.ndarray,
    radius: int,
    shape: tuple[int, int],
    score: float,
    confidence: float,
    branch_state: str,
    evidence_groups: list[str],
    defect_type: str,
    provenance: Dict[str, Any],
) -> DetectionRegion:
    x, y = (float(point_xy[0]), float(point_xy[1]))
    rows, columns = disk((y, x), radius, shape=shape)
    return DetectionRegion(
        region_id=region_id,
        bbox_xyxy=(
            max(0, int(round(x)) - radius),
            max(0, int(round(y)) - radius),
            min(shape[1], int(round(x)) + radius + 1),
            min(shape[0], int(round(y)) + radius + 1),
        ),
        centroid_xy=(x, y),
        area_pixels=int(len(rows)),
        mean_score=float(score),
        max_score=float(score),
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        confidence_level="high" if branch_state == "consensus" else "medium",
        evidence_groups=evidence_groups,
        defect_type=defect_type,
        provenance={
            **provenance,
            "teacher_consensus_state": branch_state,
        },
    )


def detect_atomic_consensus(
    source: str | Path | np.ndarray,
    detector_config: DetectorConfig,
    motif_config: PeriodicCounterfactualConfig | None = None,
    consensus_config: AtomicConsensusConfig | None = None,
    *,
    frame_index: int | None = None,
    projection: str | None = None,
    dataset_path: str | None = None,
) -> DetectionResult:
    """Run lattice-residual and periodic-counterfactual teachers together."""

    if detector_config.task != "atomic_point_defect":
        raise ValueError(
            "Atomic consensus requires task='atomic_point_defect'"
        )
    policy = consensus_config or AtomicConsensusConfig()
    motif_policy = motif_config or PeriodicCounterfactualConfig()
    policy.validate()
    motif_policy.validate()
    if isinstance(source, (str, Path)):
        raw = load_grayscale(
            source,
            frame_index=frame_index,
            projection=projection,
            dataset_path=dataset_path,
        )
    else:
        raw = np.asarray(source)
        if raw.ndim != 2:
            raise ValueError("Array input must be a 2-D grayscale image")

    primary = TrainingFreeDetectionFramework(detector_config).detect(raw)
    secondary = detect_periodic_counterfactual(
        raw, detector_config, motif_policy
    )
    primary_spacing = _spacing(primary)
    secondary_spacing = _spacing(secondary)
    tolerance = policy.matching_tolerance_fraction * min(
        primary_spacing, secondary_spacing
    )
    fusion = fuse_site_consensus(
        _region_points(primary),
        _region_points(secondary),
        tolerance_pixels=tolerance,
    )
    radius = max(
        2,
        int(
            round(
                policy.render_radius_fraction
                * min(primary_spacing, secondary_spacing)
            )
        ),
    )
    regions: list[DetectionRegion] = []
    primary_regions = primary.regions
    secondary_regions = secondary.regions
    for match_index, (primary_index, secondary_index) in enumerate(
        zip(fusion.primary_indices, fusion.secondary_indices), start=1
    ):
        first = primary_regions[int(primary_index)]
        second = secondary_regions[int(secondary_index)]
        regions.append(
            _candidate_region(
                region_id=match_index,
                point_xy=fusion.consensus_xy[match_index - 1],
                radius=radius,
                shape=primary.image_shape,
                score=min(first.mean_score, second.mean_score),
                confidence=float(
                    np.sqrt(first.confidence * second.confidence)
                ),
                branch_state="consensus",
                evidence_groups=[
                    "legacy_lattice_residual",
                    "periodic_counterfactual",
                ],
                defect_type=first.defect_type,
                provenance={
                    "legacy_region_id": first.region_id,
                    "periodic_region_id": second.region_id,
                    "agreement_distance_pixels": float(
                        fusion.agreement_distances[match_index - 1]
                    ),
                    "legacy_defect_type": first.defect_type,
                    "periodic_defect_type": second.defect_type,
                },
            )
        )
    if policy.retain_single_teacher_candidates:
        matched_primary = set(fusion.primary_indices.tolist())
        matched_secondary = set(fusion.secondary_indices.tolist())
        for index, source_region in enumerate(primary_regions):
            if index in matched_primary:
                continue
            regions.append(
                _candidate_region(
                    region_id=len(regions) + 1,
                    point_xy=np.asarray(source_region.centroid_xy),
                    radius=radius,
                    shape=primary.image_shape,
                    score=source_region.mean_score,
                    confidence=source_region.confidence,
                    branch_state="legacy_only",
                    evidence_groups=["legacy_lattice_residual"],
                    defect_type=source_region.defect_type,
                    provenance={
                        "legacy_region_id": source_region.region_id,
                        "legacy_defect_type": source_region.defect_type,
                    },
                )
            )
        for index, source_region in enumerate(secondary_regions):
            if index in matched_secondary:
                continue
            regions.append(
                _candidate_region(
                    region_id=len(regions) + 1,
                    point_xy=np.asarray(source_region.centroid_xy),
                    radius=radius,
                    shape=primary.image_shape,
                    score=source_region.mean_score,
                    confidence=source_region.confidence,
                    branch_state="periodic_only",
                    evidence_groups=["periodic_counterfactual"],
                    defect_type=source_region.defect_type,
                    provenance={
                        "periodic_region_id": source_region.region_id,
                        "periodic_defect_type": source_region.defect_type,
                    },
                )
            )

    shape = primary.image_shape
    mask = np.zeros(shape, dtype=bool)
    score_map = np.zeros(shape, dtype=np.float32)
    probability_map = np.zeros(shape, dtype=np.float32)
    agreement_map = np.zeros(shape, dtype=np.float32)
    legacy_only_map = np.zeros(shape, dtype=np.float32)
    periodic_only_map = np.zeros(shape, dtype=np.float32)
    for region in regions:
        x, y = region.centroid_xy
        rows, columns = disk((y, x), radius, shape=shape)
        mask[rows, columns] = True
        score_map[rows, columns] = np.maximum(
            score_map[rows, columns], region.mean_score
        )
        probability_map[rows, columns] = np.maximum(
            probability_map[rows, columns], region.confidence
        )
        state = region.provenance["teacher_consensus_state"]
        target = {
            "consensus": agreement_map,
            "legacy_only": legacy_only_map,
            "periodic_only": periodic_only_map,
        }[state]
        target[rows, columns] = 1.0

    provenance = _input_provenance(
        source,
        raw,
        frame_index=frame_index,
        projection=projection,
        dataset_path=dataset_path,
    )
    metadata = {
        **provenance,
        "method": "two_physics_teacher_atomic_consensus",
        "atomic_consensus_version": ATOMIC_CONSENSUS_VERSION,
        "neural_training_used": False,
        "supervised_labels_used": False,
        "annotations_read": False,
        "consensus_config": asdict(policy),
        "consensus_config_sha256": policy.content_hash(),
        "periodic_counterfactual_config": asdict(motif_policy),
        "periodic_counterfactual_config_sha256": (
            motif_policy.content_hash()
        ),
        "matching_tolerance_pixels": tolerance,
        "render_radius_pixels": radius,
        "legacy_lattice_spacing_pixels": primary_spacing,
        "periodic_lattice_spacing_pixels": secondary_spacing,
        "consensus_summary": fusion.to_summary(),
        "branch_candidate_counts": {
            "legacy_lattice_residual": len(primary.regions),
            "periodic_counterfactual": len(secondary.regions),
            "combined_retained": len(regions),
        },
        "single_teacher_policy": (
            "retain_as_unresolved"
            if policy.retain_single_teacher_candidates
            else "omit_from_combined_result"
        ),
    }
    evidence_maps = _merge_evidence_maps(primary, secondary)
    evidence_maps.update(
        {
            "teacher_agreement": agreement_map,
            "legacy_teacher_only": legacy_only_map,
            "periodic_teacher_only": periodic_only_map,
        }
    )
    result = DetectionResult(
        task="atomic_point_defect",
        image_shape=shape,
        normalized_image=primary.normalized_image,
        score_map=score_map,
        probability_map=probability_map,
        mask=mask,
        regions=regions,
        feature_maps={
            "legacy_score": primary.score_map,
            "periodic_score": secondary.score_map,
            "teacher_agreement": agreement_map,
            "legacy_teacher_only": legacy_only_map,
            "periodic_teacher_only": periodic_only_map,
        },
        evidence_maps=evidence_maps,
        metadata=metadata,
    )
    protocol = detector_config.evidence_protocol
    if protocol.enabled:
        fusion_provenance = [
            dict(region.provenance) for region in result.regions
        ]
        result = attach_evidence_protocol(
            result,
            reliable_threshold=protocol.reliable_threshold,
            invalid_threshold=protocol.invalid_threshold,
            minimum_confirmed_evidence_groups=(
                protocol.minimum_confirmed_evidence_groups
            ),
            detector_version="0.4.0-atomic-consensus",
        )
        assert result.decision_map is not None
        for region, branch_provenance in zip(
            result.regions, fusion_provenance
        ):
            region.provenance = {
                **region.provenance,
                **branch_provenance,
            }
            x, y = region.centroid_xy
            rows, columns = disk((y, x), radius, shape=shape)
            state = region.provenance["teacher_consensus_state"]
            if state == "consensus":
                region.decision_state = (
                    DecisionState.CONFIRMED_CANDIDATE.value
                )
                region.label_tier = LabelTier.SILVER.value
                region.epistemic_state = "probable"
                region.resolved_family = "atomic_site_anomaly"
                region.resolved_label = "multi_teacher_atomic_site_anomaly"
                region.quality_flags.append(
                    "independent_physics_teacher_agreement"
                )
                result.decision_map[rows, columns] = DECISION_STATE_CODES[
                    DecisionState.CONFIRMED_CANDIDATE.value
                ]
            else:
                region.decision_state = (
                    DecisionState.UNCERTAIN_CANDIDATE.value
                )
                region.label_tier = LabelTier.UNRESOLVED.value
                region.epistemic_state = "unresolved"
                region.resolved_family = "atomic_site_anomaly"
                region.resolved_label = (
                    "single_teacher_atomic_site_anomaly"
                )
                region.quality_flags.append("teacher_disagreement")
                only_unset = (
                    result.decision_map[rows, columns]
                    != DECISION_STATE_CODES[
                        DecisionState.CONFIRMED_CANDIDATE.value
                    ]
                )
                result.decision_map[
                    rows[only_unset], columns[only_unset]
                ] = DECISION_STATE_CODES[
                    DecisionState.UNCERTAIN_CANDIDATE.value
                ]
    return result
