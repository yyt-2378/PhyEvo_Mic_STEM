"""Open-world, multi-hypothesis resolution over a shared EvidenceBundle."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, Sequence

import numpy as np

from .evidence import (
    DECISION_STATE_CODES,
    DecisionState,
    EvidenceBundle,
    LabelTier,
)
from .models import DetectionRegion, DetectionResult


RESOLVER_VERSION = "1.0"


@dataclass(frozen=True)
class ResolverPolicy:
    minimum_hypothesis_score: float = 0.15
    minimum_score_margin: float = 0.05
    minimum_region_support_fraction: float = 0.20
    maximum_ranked_hypotheses: int = 5

    def validate(self) -> None:
        if self.minimum_hypothesis_score < 0:
            raise ValueError("minimum_hypothesis_score cannot be negative")
        if self.minimum_score_margin < 0:
            raise ValueError("minimum_score_margin cannot be negative")
        if not 0 <= self.minimum_region_support_fraction <= 1:
            raise ValueError(
                "minimum_region_support_fraction must be in [0, 1]"
            )
        if self.maximum_ranked_hypotheses < 1:
            raise ValueError("maximum_ranked_hypotheses must be positive")


@dataclass(frozen=True)
class RankedHypothesis:
    name: str
    family: str
    resolution_kind: str
    score: float
    mean_energy: float
    p90_energy: float
    support_fraction: float
    evidence_multiplicity_factor: float
    available_fields: tuple[str, ...]
    required_auxiliary_evidence: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["available_fields"] = list(self.available_fields)
        payload["required_auxiliary_evidence"] = list(
            self.required_auxiliary_evidence
        )
        return payload


@dataclass(frozen=True)
class HypothesisResolution:
    candidate_id: str
    family: str
    label: str
    epistemic_state: str
    primary_score: float
    score_margin: float
    ranking: tuple[RankedHypothesis, ...]
    missing_auxiliary_evidence: tuple[str, ...]
    rationale: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolver_version": RESOLVER_VERSION,
            "candidate_id": self.candidate_id,
            "family": self.family,
            "label": self.label,
            "epistemic_state": self.epistemic_state,
            "primary_score": self.primary_score,
            "score_margin": self.score_margin,
            "ranking": [item.to_dict() for item in self.ranking],
            "missing_auxiliary_evidence": list(
                self.missing_auxiliary_evidence
            ),
            "rationale": list(self.rationale),
        }


def _region_pixels(
    result: DetectionResult, region: DetectionRegion
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = (int(value) for value in region.bbox_xyxy)
    local = result.mask[y1:y2, x1:x2]
    rows, columns = np.nonzero(local)
    if rows.size:
        return rows + y1, columns + x1
    height = max(y2 - y1, 1)
    width = max(x2 - x1, 1)
    rows, columns = np.indices((height, width))
    return rows.ravel() + y1, columns.ravel() + x1


def _rank_region(
    bundle: EvidenceBundle,
    rows: np.ndarray,
    columns: np.ndarray,
    policy: ResolverPolicy,
) -> list[RankedHypothesis]:
    ranking: list[RankedHypothesis] = []
    for score in bundle.hypotheses.values():
        if not score.available_fields:
            continue
        values = score.energy_map[rows, columns]
        support = (
            score.support_count_map[rows, columns]
            >= score.hypothesis.minimum_supported_fields
        )
        support_fraction = float(np.mean(support)) if support.size else 0.0
        if support_fraction < policy.minimum_region_support_fraction:
            continue
        if not values.size:
            continue
        mean_energy = float(np.mean(values))
        p90_energy = float(np.percentile(values, 90))
        multiplicity_factor = 1.0 + 0.35 * max(
            score.hypothesis.minimum_supported_fields - 1,
            0,
        )
        regional_score = (
            0.60 * mean_energy + 0.40 * p90_energy
        ) * multiplicity_factor
        ranking.append(
            RankedHypothesis(
                name=score.hypothesis.name,
                family=score.hypothesis.family,
                resolution_kind=score.hypothesis.resolution_kind,
                score=regional_score,
                mean_energy=mean_energy,
                p90_energy=p90_energy,
                support_fraction=support_fraction,
                evidence_multiplicity_factor=multiplicity_factor,
                available_fields=tuple(score.available_fields),
                required_auxiliary_evidence=(
                    score.hypothesis.required_auxiliary_evidence
                ),
            )
        )
    ranking.sort(
        key=lambda item: (
            item.score,
            item.support_fraction,
            len(item.available_fields),
            item.name,
        ),
        reverse=True,
    )
    return ranking


_UNRESOLVED_LABELS = {
    "atomic_site_anomaly": "atomic_site_subtype_unresolved",
    "lattice_deformation_and_topology": "line_or_deformation_subtype_unresolved",
    "domain_symmetry_and_interface": "interface_subtype_unresolved",
    "order_and_coherence_degradation": "order_degradation_subtype_unresolved",
    "acquisition_artifact_and_quality": "artifact_subtype_unresolved",
}


def _unresolved_label(ranking: Sequence[RankedHypothesis]) -> tuple[str, str]:
    competitive = [
        item
        for item in ranking
        if item.resolution_kind != "fallback"
    ][:2]
    families = {item.family for item in competitive}
    if len(families) == 1:
        family = next(iter(families))
        return family, _UNRESOLVED_LABELS.get(
            family, "family_subtype_unresolved"
        )
    return "open_set_resolution", "unknown_structural_anomaly"


def resolve_region(
    result: DetectionResult,
    bundle: EvidenceBundle,
    region: DetectionRegion,
    policy: ResolverPolicy,
) -> HypothesisResolution:
    rows, columns = _region_pixels(result, region)
    ranking = _rank_region(bundle, rows, columns, policy)
    limited = tuple(ranking[: policy.maximum_ranked_hypotheses])
    rationale: list[str] = []

    if region.decision_state == DecisionState.INVALID_MEASUREMENT.value:
        return HypothesisResolution(
            candidate_id=region.candidate_id,
            family="open_set_resolution",
            label="invalid_measurement",
            epistemic_state="invalid_measurement",
            primary_score=limited[0].score if limited else 0.0,
            score_margin=0.0,
            ranking=limited,
            missing_auxiliary_evidence=(),
            rationale=("Measurement validity overrides physical diagnosis.",),
        )
    if region.decision_state == DecisionState.OUT_OF_DOMAIN.value:
        return HypothesisResolution(
            candidate_id=region.candidate_id,
            family="open_set_resolution",
            label="out_of_distribution_sample",
            epistemic_state="out_of_domain",
            primary_score=limited[0].score if limited else 0.0,
            score_margin=0.0,
            ranking=limited,
            missing_auxiliary_evidence=(),
            rationale=("OOD state overrides in-domain physical diagnosis.",),
        )

    eligible = [
        item
        for item in ranking
        if item.score >= policy.minimum_hypothesis_score
    ]
    physical = [
        item
        for item in eligible
        if item.resolution_kind == "physical"
    ]
    artifacts = [
        item
        for item in eligible
        if item.resolution_kind == "artifact"
    ]
    fallbacks = [
        item
        for item in eligible
        if item.resolution_kind == "fallback"
    ]

    if artifacts and (
        not physical
        or artifacts[0].score
        >= physical[0].score + policy.minimum_score_margin
    ):
        top = artifacts[0]
        competing = physical[0].score if physical else 0.0
        margin = top.score - competing
        rationale.append(
            "Artifact evidence exceeds every supported physical hypothesis."
        )
        return HypothesisResolution(
            candidate_id=region.candidate_id,
            family=top.family,
            label=top.name,
            epistemic_state="artifact",
            primary_score=top.score,
            score_margin=margin,
            ranking=limited,
            missing_auxiliary_evidence=top.required_auxiliary_evidence,
            rationale=tuple(rationale),
        )

    if not physical:
        family, label = _unresolved_label(ranking)
        reason = (
            "Only a broad fallback hypothesis is supported."
            if fallbacks
            else "No hypothesis exceeds the transparent score threshold."
        )
        return HypothesisResolution(
            candidate_id=region.candidate_id,
            family=family,
            label=label,
            epistemic_state="unresolved",
            primary_score=limited[0].score if limited else 0.0,
            score_margin=0.0,
            ranking=limited,
            missing_auxiliary_evidence=(),
            rationale=(reason,),
        )

    top = physical[0]
    second = physical[1] if len(physical) > 1 else None
    margin = top.score - (second.score if second is not None else 0.0)
    if (
        second is not None
        and margin < policy.minimum_score_margin
    ):
        family, label = _unresolved_label(physical)
        missing = tuple(
            sorted(
                set(top.required_auxiliary_evidence)
                | set(second.required_auxiliary_evidence)
            )
        )
        return HypothesisResolution(
            candidate_id=region.candidate_id,
            family=family,
            label=label,
            epistemic_state="unresolved",
            primary_score=top.score,
            score_margin=margin,
            ranking=limited,
            missing_auxiliary_evidence=missing,
            rationale=(
                "Top physical hypotheses are not separated by the required "
                "margin.",
            ),
        )

    if region.decision_state != DecisionState.CONFIRMED_CANDIDATE.value:
        return HypothesisResolution(
            candidate_id=region.candidate_id,
            family=top.family,
            label=top.name,
            epistemic_state="unresolved",
            primary_score=top.score,
            score_margin=margin,
            ranking=limited,
            missing_auxiliary_evidence=top.required_auxiliary_evidence,
            rationale=(
                "Candidate measurement reliability is insufficient for a "
                "probable diagnosis.",
            ),
        )

    rationale.append(
        "One supported physical hypothesis exceeds competing explanations."
    )
    if top.required_auxiliary_evidence:
        rationale.append(
            "Causal confirmation still requires the listed auxiliary evidence."
        )
    return HypothesisResolution(
        candidate_id=region.candidate_id,
        family=top.family,
        label=top.name,
        epistemic_state="probable",
        primary_score=top.score,
        score_margin=margin,
        ranking=limited,
        missing_auxiliary_evidence=top.required_auxiliary_evidence,
        rationale=tuple(rationale),
    )


def attach_hypothesis_resolutions(
    result: DetectionResult, policy: ResolverPolicy | None = None
) -> DetectionResult:
    """Resolve every candidate while preserving detector masks and scores."""

    if result.evidence_bundle is None:
        raise ValueError("Multi-hypothesis resolution requires EvidenceBundle")
    active_policy = policy or ResolverPolicy()
    active_policy.validate()
    if result.metadata.get("resolver", {}).get("version") == RESOLVER_VERSION:
        return result

    resolutions: list[HypothesisResolution] = []
    for region in result.regions:
        resolution = resolve_region(
            result,
            result.evidence_bundle,
            region,
            active_policy,
        )
        resolutions.append(resolution)
        region.resolved_family = resolution.family
        region.resolved_label = resolution.label
        region.epistemic_state = resolution.epistemic_state
        region.hypothesis_ranking = [
            item.to_dict() for item in resolution.ranking
        ]
        region.missing_auxiliary_evidence = list(
            resolution.missing_auxiliary_evidence
        )
        region.provenance = {
            **region.provenance,
            "resolver_version": RESOLVER_VERSION,
            "resolved_family": resolution.family,
            "resolved_label": resolution.label,
            "epistemic_state": resolution.epistemic_state,
        }
        rows, columns = _region_pixels(result, region)
        if resolution.epistemic_state == "artifact":
            region.decision_state = DecisionState.ARTIFACT.value
            region.label_tier = LabelTier.UNRESOLVED.value
            if "artifact" not in region.quality_flags:
                region.quality_flags.append("artifact")
            if result.decision_map is not None:
                mutable = ~np.isin(
                    result.decision_map[rows, columns],
                    (
                        DECISION_STATE_CODES[
                            DecisionState.INVALID_MEASUREMENT.value
                        ],
                        DECISION_STATE_CODES[
                            DecisionState.OUT_OF_DOMAIN.value
                        ],
                    ),
                )
                result.decision_map[
                    rows[mutable], columns[mutable]
                ] = DECISION_STATE_CODES[DecisionState.ARTIFACT.value]
        elif resolution.epistemic_state == "unresolved":
            region.decision_state = DecisionState.UNRESOLVED.value
            region.label_tier = LabelTier.UNRESOLVED.value
            if "multi_hypothesis_unresolved" not in region.quality_flags:
                region.quality_flags.append("multi_hypothesis_unresolved")
            if result.decision_map is not None:
                mutable = ~np.isin(
                    result.decision_map[rows, columns],
                    (
                        DECISION_STATE_CODES[
                            DecisionState.INVALID_MEASUREMENT.value
                        ],
                        DECISION_STATE_CODES[
                            DecisionState.OUT_OF_DOMAIN.value
                        ],
                    ),
                )
                result.decision_map[
                    rows[mutable], columns[mutable]
                ] = DECISION_STATE_CODES[DecisionState.UNRESOLVED.value]

    state_counts = Counter(item.epistemic_state for item in resolutions)
    result.metadata["resolver"] = {
        "version": RESOLVER_VERSION,
        "policy": asdict(active_policy),
        "candidate_count": len(resolutions),
        "epistemic_state_counts": dict(sorted(state_counts.items())),
        "task_branch_semantics": (
            "The task branch generates candidate geometry and evidence fields; "
            "it does not directly assign the resolved physical subtype."
        ),
        "resolutions": [item.to_dict() for item in resolutions],
    }
    return result
