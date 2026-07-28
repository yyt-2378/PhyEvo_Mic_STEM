"""Provenance-preserving adapters for heterogeneous public dataset labels.

External labels are never overwritten or silently promoted to Gold.  Each
adapter keeps the source payload, assigns a platform observation/diagnostic
label, records plausible alternatives, and states whether the result is safe
for hard-label training.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from fnmatch import fnmatch
from typing import Any, Dict, Iterable, Mapping, Protocol

from .evidence import LabelTier


LABEL_ADAPTER_PROTOCOL_VERSION = "1.0"


class _StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class SemanticRole(_StringEnum):
    DEFECT_CANDIDATE = "defect_candidate"
    STRUCTURE_REFERENCE = "structure_reference"
    NORMAL_CONTROL = "normal_control"
    ARTIFACT = "artifact"
    UNKNOWN = "unknown"


class EpistemicState(_StringEnum):
    CONFIRMED = "confirmed"
    PROBABLE = "probable"
    COMPOSITE = "composite"
    UNRESOLVED = "unresolved"
    ARTIFACT = "artifact"
    OUT_OF_DOMAIN = "out_of_domain"
    INVALID_MEASUREMENT = "invalid_measurement"


@dataclass(frozen=True)
class ExternalLabel:
    """One immutable external annotation before ontology translation."""

    dataset_id: str
    asset_id: str
    annotation_id: str
    source_label: str
    spatial_support: str
    annotation_qualification: str
    source_payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ExternalLabel":
        return cls(
            dataset_id=str(values["dataset_id"]),
            asset_id=str(values["asset_id"]),
            annotation_id=str(values["annotation_id"]),
            source_label=str(values["source_label"]),
            spatial_support=str(values.get("spatial_support", "region")),
            annotation_qualification=str(
                values.get("annotation_qualification", "none")
            ),
            source_payload=dict(values.get("source_payload", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "source_payload": dict(self.source_payload),
        }


@dataclass(frozen=True)
class AdaptedLabel:
    """Auditable ontology view of one external annotation."""

    adapter_id: str
    adapter_version: str
    dataset_id: str
    asset_id: str
    annotation_id: str
    source_label: str
    source_payload: Mapping[str, Any]
    semantic_role: SemanticRole
    ontology_family: str
    ontology_label: str
    alternative_hypotheses: tuple[str, ...]
    spatial_support: str
    epistemic_state: EpistemicState
    label_tier: LabelTier
    annotation_qualification: str
    eligible_for_hard_label_training: bool
    requires_auxiliary_evidence: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def validate(self) -> None:
        required = (
            self.adapter_id,
            self.adapter_version,
            self.dataset_id,
            self.asset_id,
            self.annotation_id,
            self.source_label,
            self.ontology_family,
            self.ontology_label,
            self.spatial_support,
            self.annotation_qualification,
        )
        if not all(required):
            raise ValueError("Adapted label fields cannot be empty")
        if self.epistemic_state in {
            EpistemicState.COMPOSITE,
            EpistemicState.UNRESOLVED,
            EpistemicState.OUT_OF_DOMAIN,
            EpistemicState.INVALID_MEASUREMENT,
        } and self.eligible_for_hard_label_training:
            raise ValueError(
                "Composite, unresolved, OOD, or invalid labels cannot be hard "
                "training targets"
            )
        if (
            self.label_tier == LabelTier.GOLD
            and self.annotation_qualification
            not in {"expert_consensus", "simulation_truth"}
        ):
            raise ValueError(
                "Gold requires expert consensus or verifiable simulation truth"
            )
        if (
            self.semantic_role == SemanticRole.ARTIFACT
            and self.epistemic_state != EpistemicState.ARTIFACT
        ):
            raise ValueError("Artifact semantic role requires artifact state")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return {
            "adapter_protocol_version": LABEL_ADAPTER_PROTOCOL_VERSION,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "dataset_id": self.dataset_id,
            "asset_id": self.asset_id,
            "annotation_id": self.annotation_id,
            "original": {
                "source_label": self.source_label,
                "source_payload": dict(self.source_payload),
                "annotation_qualification": self.annotation_qualification,
            },
            "adapted": {
                "semantic_role": self.semantic_role.value,
                "ontology_family": self.ontology_family,
                "ontology_label": self.ontology_label,
                "alternative_hypotheses": list(
                    self.alternative_hypotheses
                ),
                "spatial_support": self.spatial_support,
                "epistemic_state": self.epistemic_state.value,
                "label_tier": self.label_tier.value,
                "eligible_for_hard_label_training": (
                    self.eligible_for_hard_label_training
                ),
                "requires_auxiliary_evidence": list(
                    self.requires_auxiliary_evidence
                ),
                "notes": list(self.notes),
            },
        }


class DatasetLabelAdapter(Protocol):
    adapter_id: str
    adapter_version: str
    dataset_patterns: tuple[str, ...]

    def adapt(self, label: ExternalLabel) -> AdaptedLabel:
        ...

    def manifest(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class _Rule:
    source_patterns: tuple[str, ...]
    semantic_role: SemanticRole
    family: str
    label: str
    state: EpistemicState
    tier: LabelTier
    alternatives: tuple[str, ...] = ()
    auxiliary: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    hard_label_allowed: bool = False
    qualification_override: str | None = None


class RuleBasedDatasetAdapter:
    """Ordered transparent mapping rules for one dataset family."""

    def __init__(
        self,
        *,
        adapter_id: str,
        dataset_patterns: Iterable[str],
        rules: Iterable[_Rule],
        fallback: _Rule,
        adapter_version: str = "1.0",
    ) -> None:
        self.adapter_id = adapter_id
        self.adapter_version = adapter_version
        self.dataset_patterns = tuple(dataset_patterns)
        self._rules = tuple(rules)
        self._fallback = fallback

    def matches_dataset(self, dataset_id: str) -> bool:
        return any(
            fnmatch(dataset_id.lower(), pattern.lower())
            for pattern in self.dataset_patterns
        )

    def _rule_for(self, source_label: str) -> _Rule:
        normalized = source_label.strip().lower()
        for rule in self._rules:
            if any(
                fnmatch(normalized, pattern.lower())
                for pattern in rule.source_patterns
            ):
                return rule
        return self._fallback

    def adapt(self, label: ExternalLabel) -> AdaptedLabel:
        if not self.matches_dataset(label.dataset_id):
            raise ValueError(
                f"Adapter {self.adapter_id!r} does not accept dataset "
                f"{label.dataset_id!r}"
            )
        rule = self._rule_for(label.source_label)
        qualification = (
            rule.qualification_override
            or label.annotation_qualification
        )
        adapted = AdaptedLabel(
            adapter_id=self.adapter_id,
            adapter_version=self.adapter_version,
            dataset_id=label.dataset_id,
            asset_id=label.asset_id,
            annotation_id=label.annotation_id,
            source_label=label.source_label,
            source_payload=dict(label.source_payload),
            semantic_role=rule.semantic_role,
            ontology_family=rule.family,
            ontology_label=rule.label,
            alternative_hypotheses=rule.alternatives,
            spatial_support=label.spatial_support,
            epistemic_state=rule.state,
            label_tier=rule.tier,
            annotation_qualification=qualification,
            eligible_for_hard_label_training=rule.hard_label_allowed,
            requires_auxiliary_evidence=rule.auxiliary,
            notes=rule.notes,
        )
        adapted.validate()
        return adapted

    def manifest(self) -> Mapping[str, Any]:
        def serialize(rule: _Rule) -> Dict[str, Any]:
            return {
                "source_patterns": list(rule.source_patterns),
                "semantic_role": rule.semantic_role.value,
                "ontology_family": rule.family,
                "ontology_label": rule.label,
                "epistemic_state": rule.state.value,
                "label_tier": rule.tier.value,
                "alternative_hypotheses": list(rule.alternatives),
                "requires_auxiliary_evidence": list(rule.auxiliary),
                "hard_label_allowed": rule.hard_label_allowed,
                "notes": list(rule.notes),
            }

        return {
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "dataset_patterns": list(self.dataset_patterns),
            "rules": [serialize(rule) for rule in self._rules],
            "fallback": serialize(self._fallback),
        }


class DatasetLabelAdapterRegistry:
    def __init__(self, adapters: Iterable[RuleBasedDatasetAdapter] = ()) -> None:
        self._adapters: Dict[str, RuleBasedDatasetAdapter] = {}
        for adapter in adapters:
            self.register(adapter)

    def register(self, adapter: RuleBasedDatasetAdapter) -> None:
        if adapter.adapter_id in self._adapters:
            raise ValueError(f"Duplicate adapter id: {adapter.adapter_id}")
        self._adapters[adapter.adapter_id] = adapter

    def adapter_for(self, dataset_id: str) -> RuleBasedDatasetAdapter:
        matches = [
            adapter
            for adapter in self._adapters.values()
            if adapter.matches_dataset(dataset_id)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one label adapter for {dataset_id!r}, found "
                f"{len(matches)}"
            )
        return matches[0]

    def adapt(self, label: ExternalLabel) -> AdaptedLabel:
        return self.adapter_for(label.dataset_id).adapt(label)

    def manifest(self) -> Dict[str, Any]:
        return {
            "adapter_protocol_version": LABEL_ADAPTER_PROTOCOL_VERSION,
            "adapters": {
                adapter_id: adapter.manifest()
                for adapter_id, adapter in sorted(self._adapters.items())
            },
        }


_UNRESOLVED = _Rule(
    source_patterns=("*",),
    semantic_role=SemanticRole.UNKNOWN,
    family="open_set_resolution",
    label="unknown_structural_anomaly",
    state=EpistemicState.UNRESOLVED,
    tier=LabelTier.UNRESOLVED,
    auxiliary=("expert_review", "source_label_semantics"),
    notes=("No source-specific rule matched; original label is preserved.",),
)


def default_label_adapter_registry() -> DatasetLabelAdapterRegistry:
    """Return built-in adapters for all currently catalogued label sources."""

    emseek = RuleBasedDatasetAdapter(
        adapter_id="emseek-v1",
        dataset_patterns=("emseek_*", "EMSeek"),
        rules=(
            _Rule(
                source_patterns=("wse2*", "*atomic*defect*", "*point*mask*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="atomic_site_anomaly",
                label="atomic_column_intensity_anomaly",
                state=EpistemicState.PROBABLE,
                tier=LabelTier.SILVER,
                alternatives=(
                    "vacancy_like",
                    "substitution_like",
                    "displaced_atom_column",
                ),
                auxiliary=(
                    "source_palette_semantics",
                    "thickness_or_forward_model",
                ),
                notes=(
                    "The public mask localizes an atomic-site anomaly but does "
                    "not by itself confirm vacancy species.",
                ),
            ),
            _Rule(
                source_patterns=("atom_column*", "*column*mask*"),
                semantic_role=SemanticRole.STRUCTURE_REFERENCE,
                family="atomic_site_anomaly",
                label="atomic_column_reference",
                state=EpistemicState.PROBABLE,
                tier=LabelTier.SILVER,
                notes=("Localization reference, not a physical defect label.",),
            ),
            _Rule(
                source_patterns=("fecral*", "*irradiation*", "*damage*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="order_and_coherence_degradation",
                label="beam_damage_consistent_region",
                state=EpistemicState.COMPOSITE,
                tier=LabelTier.UNRESOLVED,
                alternatives=(
                    "irradiation_damage",
                    "local_amorphous_region",
                    "acquisition_artifact",
                ),
                auxiliary=("dose_metadata", "temporal_sequence"),
            ),
            _Rule(
                source_patterns=(
                    "nanoparticle*",
                    "au@*",
                    "cdse@*",
                    "fe3o4*",
                    "pd@*",
                ),
                semantic_role=SemanticRole.STRUCTURE_REFERENCE,
                family="particle_surface_and_morphology",
                label="particle_region",
                state=EpistemicState.PROBABLE,
                tier=LabelTier.SILVER,
                alternatives=("particle_support_interface", "particle_facet"),
                notes=("Particle mask is not automatically a defect mask.",),
            ),
        ),
        fallback=_UNRESOLVED,
    )
    guo = RuleBasedDatasetAdapter(
        adapter_id="guo-2021-v1",
        dataset_patterns=("guo_2021_*",),
        rules=(
            _Rule(
                source_patterns=("*partial*dislocation*", "*al*dislocation*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="lattice_deformation_and_topology",
                label="dislocation_or_stacking_fault_unresolved",
                state=EpistemicState.COMPOSITE,
                tier=LabelTier.UNRESOLVED,
                alternatives=("dislocation_core_or_line", "stacking_fault"),
                auxiliary=("expert_trace", "burgers_circuit_or_phase_analysis"),
            ),
            _Rule(
                source_patterns=("*zro2*twin*", "*twin*boundary*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="domain_symmetry_and_interface",
                label="twin_boundary",
                state=EpistemicState.PROBABLE,
                tier=LabelTier.SILVER,
                alternatives=("orientation_domain_wall",),
                auxiliary=("expert_boundary_trace",),
            ),
            _Rule(
                source_patterns=("*mowte2*", "*large*field*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="open_set_resolution",
                label="mixed_atomic_and_line_anomaly",
                state=EpistemicState.COMPOSITE,
                tier=LabelTier.UNRESOLVED,
                alternatives=("atomic_site_anomaly", "stacking_fault"),
                auxiliary=("candidate_level_annotation",),
            ),
            _Rule(
                source_patterns=("*superscan*", "*movie*", "*sequence*"),
                semantic_role=SemanticRole.UNKNOWN,
                family="open_set_resolution",
                label="unlabeled_temporal_sequence",
                state=EpistemicState.UNRESOLVED,
                tier=LabelTier.UNRESOLVED,
                alternatives=(
                    "temporal_defect_evolution",
                    "scan_artifact",
                ),
                auxiliary=("frame_level_expert_review", "acquisition_metadata"),
            ),
        ),
        fallback=_UNRESOLVED,
    )
    sigma3 = RuleBasedDatasetAdapter(
        adapter_id="al-sigma3-v1",
        dataset_patterns=("al_sigma3_*",),
        rules=(
            _Rule(
                source_patterns=("*ori*", "*stem*", "*experimental*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="domain_symmetry_and_interface",
                label="twin_boundary",
                state=EpistemicState.PROBABLE,
                tier=LabelTier.SILVER,
                alternatives=("grain_boundary", "interface_disconnection"),
                auxiliary=("expert_boundary_trace",),
            ),
            _Rule(
                source_patterns=("*ebsd*",),
                semantic_role=SemanticRole.STRUCTURE_REFERENCE,
                family="domain_symmetry_and_interface",
                label="orientation_evidence",
                state=EpistemicState.PROBABLE,
                tier=LabelTier.SILVER,
                notes=("EBSD supplies context and is not a STEM pixel mask.",),
            ),
            _Rule(
                source_patterns=("*simulation*", "*digital*twin*"),
                semantic_role=SemanticRole.STRUCTURE_REFERENCE,
                family="domain_symmetry_and_interface",
                label="twin_boundary_simulation_truth",
                state=EpistemicState.CONFIRMED,
                tier=LabelTier.GOLD,
                hard_label_allowed=True,
                qualification_override="simulation_truth",
                notes=(
                    "Simulation Gold must be reported separately from "
                    "experimental Gold.",
                ),
            ),
        ),
        fallback=_UNRESOLVED,
    )
    dennler = RuleBasedDatasetAdapter(
        adapter_id="dennler-2021-v1",
        dataset_patterns=("dennler_2021_*",),
        rules=(
            _Rule(
                source_patterns=("0", "class_0", "*non*defective*", "*normal*"),
                semantic_role=SemanticRole.NORMAL_CONTROL,
                family="open_set_resolution",
                label="normal_crystalline_control",
                state=EpistemicState.PROBABLE,
                tier=LabelTier.SILVER,
            ),
            _Rule(
                source_patterns=("1", "class_1", "*secondary*symmetry*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="domain_symmetry_and_interface",
                label="secondary_symmetry_region",
                state=EpistemicState.COMPOSITE,
                tier=LabelTier.UNRESOLVED,
                alternatives=(
                    "stacking_fault",
                    "orientation_domain_wall",
                    "heterophase_interface",
                ),
                auxiliary=("candidate_level_expert_subtype",),
            ),
            _Rule(
                source_patterns=(
                    "2",
                    "class_2",
                    "*blurred*",
                    "*amorphous*beam*",
                ),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="open_set_resolution",
                label="coherence_loss_composite",
                state=EpistemicState.COMPOSITE,
                tier=LabelTier.UNRESOLVED,
                alternatives=(
                    "local_amorphous_region",
                    "defocus_or_blur",
                    "beam_damage_consistent_region",
                ),
                auxiliary=("dose_or_time_series", "reciprocal_space_evidence"),
            ),
        ),
        fallback=_UNRESOLVED,
    )
    stemgym = RuleBasedDatasetAdapter(
        adapter_id="stemgym-v1",
        dataset_patterns=("stemgym*",),
        rules=(
            _Rule(
                source_patterns=("*vacancy*",),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="atomic_site_anomaly",
                label="vacancy_like",
                state=EpistemicState.CONFIRMED,
                tier=LabelTier.GOLD,
                hard_label_allowed=True,
                qualification_override="simulation_truth",
            ),
            _Rule(
                source_patterns=("*substitution*",),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="atomic_site_anomaly",
                label="substitution_like",
                state=EpistemicState.CONFIRMED,
                tier=LabelTier.GOLD,
                hard_label_allowed=True,
                qualification_override="simulation_truth",
            ),
            _Rule(
                source_patterns=("*grain*boundary*",),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="domain_symmetry_and_interface",
                label="grain_boundary",
                state=EpistemicState.CONFIRMED,
                tier=LabelTier.GOLD,
                hard_label_allowed=True,
                qualification_override="simulation_truth",
            ),
            _Rule(
                source_patterns=("*phase*map*", "*phase*boundary*"),
                semantic_role=SemanticRole.DEFECT_CANDIDATE,
                family="domain_symmetry_and_interface",
                label="heterophase_interface",
                state=EpistemicState.CONFIRMED,
                tier=LabelTier.GOLD,
                hard_label_allowed=True,
                qualification_override="simulation_truth",
            ),
        ),
        fallback=_UNRESOLVED,
    )
    return DatasetLabelAdapterRegistry(
        (emseek, guo, sigma3, dennler, stemgym)
    )


def adapter_manifest() -> Dict[str, Any]:
    return default_label_adapter_registry().manifest()
