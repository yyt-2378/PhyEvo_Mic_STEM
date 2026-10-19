"""Evidence-bounded capability releases, observable routing, and append-only rollback.

This is controlled system improvement, not an assertion of autonomous RSI.
Development evidence may justify a shadow experiment, never production promotion.
Scores are compared at an identical frozen budget; unlabelled pixels are not
negative truth. The original physics output remains available in every mode.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping


OBSERVABLE_FEATURES = frozenset({
    "periodicity", "physics_valid_fraction", "snr_estimate", "contamination_fraction",
    "physics_area_fraction", "teacher_empty", "laplacian_noise_proxy",
    "fft_top20_power_fraction", "robust_contrast_p95_p05",
})
FRACTION_FEATURES = frozenset({"periodicity", "physics_valid_fraction", "contamination_fraction",
                               "physics_area_fraction", "fft_top20_power_fraction"})
NONNEGATIVE_FEATURES = frozenset({"laplacian_noise_proxy", "robust_contrast_p95_p05"})
GEOMETRIES = frozenset({"point", "interface", "region", "line"})
BRANCHES = frozenset({"student", "hybrid"})
PARTIAL_LABEL_METRICS = frozenset({"marked_point_recall", "annotated_band_coverage"})
SAFETY_METRICS = frozenset({"false_positive_rate", "calibration_error", "ood_risk"})


def content_hash(value: Any) -> str:
    """Canonical, finite-only JSON digest used for protocols and decisions."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _inside(root: Path, path: str | Path) -> Path:
    root = root.resolve(strict=True)
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(root) or candidate == root:
        raise ValueError("RSI artifact must be a file or dedicated directory inside project_root")
    return candidate


@dataclass(frozen=True)
class ArtifactRef:
    role: str
    path: str
    sha256: str

    def verify(self, project_root: Path) -> bool:
        if not _digest(self.sha256) or Path(self.path).is_absolute():
            return False
        try:
            path = _inside(project_root, self.path)
            if not path.is_file():
                return False
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest() == self.sha256
        except (OSError, ValueError):
            return False


@dataclass(frozen=True)
class FeatureRange:
    name: str
    minimum: float
    maximum: float

    def __post_init__(self):
        if self.name not in OBSERVABLE_FEATURES:
            raise ValueError("Scope can only use predeclared image/physics observables, not labels or test IDs")
        if not _finite(self.minimum) or not _finite(self.maximum) or self.minimum > self.maximum:
            raise ValueError("Feature range must have finite ordered bounds")
        if self.name in FRACTION_FEATURES and not 0 <= self.minimum <= self.maximum <= 1:
            raise ValueError("Fractional observable ranges must be within [0, 1]")
        if self.name in NONNEGATIVE_FEATURES and self.minimum < 0:
            raise ValueError("Proxy range cannot be negative")
        if self.name == "teacher_empty" and (self.minimum not in (0, 1) or self.maximum not in (0, 1)):
            raise ValueError("teacher_empty uses 0/1, not a confidence score")


@dataclass(frozen=True)
class ObservableContext:
    task: str
    geometry: str
    periodicity: float | None = None
    physics_valid_fraction: float | None = None
    snr_estimate: float | None = None
    contamination_fraction: float | None = None
    physics_area_fraction: float | None = None
    teacher_empty: int | bool | None = None
    laplacian_noise_proxy: float | None = None
    fft_top20_power_fraction: float | None = None
    robust_contrast_p95_p05: float | None = None

    def valid(self) -> bool:
        if not self.task or self.geometry not in GEOMETRIES:
            return False
        for key in OBSERVABLE_FEATURES:
            value = getattr(self, key)
            if value is None:
                continue
            if key == "teacher_empty":
                if not isinstance(value, (int, bool)) or value not in (0, 1):
                    return False
            elif (not _finite(value) or key in FRACTION_FEATURES and not 0 <= value <= 1
                  or key in NONNEGATIVE_FEATURES and value < 0):
                return False
        return True


@dataclass(frozen=True)
class CapabilityScope:
    scope_id: str
    tasks: tuple[str, ...]
    geometry: str
    ranges: tuple[FeatureRange, ...] = ()

    def __post_init__(self):
        if not self.scope_id or not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("A scope needs an id and nonempty unique tasks")
        if self.geometry not in GEOMETRIES:
            raise ValueError("Unknown geometry")
        if len({r.name for r in self.ranges}) != len(self.ranges):
            raise ValueError("Duplicate observable range")

    def matches(self, context: ObservableContext) -> bool:
        if not context.valid() or context.task not in self.tasks or context.geometry != self.geometry:
            return False
        return all(getattr(context, r.name) is not None
                   and r.minimum <= getattr(context, r.name) <= r.maximum for r in self.ranges)


@dataclass(frozen=True)
class PairedMetric:
    """A paired comparison; intervals concern independent acquisition groups.

    Positive normalized delta always means improvement. Seed variation must not
    be supplied as an acquisition-group confidence interval.
    """
    name: str
    reference: float
    candidate: float
    higher_is_better: bool
    reference_budget_sha256: str
    candidate_budget_sha256: str
    delta_lower_bound: float | None = None
    delta_upper_bound: float | None = None
    interval_unit: str | None = None

    def valid(self) -> bool:
        if not self.name or not all(_finite(v) for v in (self.reference, self.candidate)):
            return False
        if not isinstance(self.higher_is_better, bool):
            return False
        if (self.name in SAFETY_METRICS and self.higher_is_better
                or self.name in PARTIAL_LABEL_METRICS and not self.higher_is_better):
            return False
        if not (_digest(self.reference_budget_sha256)
                and self.reference_budget_sha256 == self.candidate_budget_sha256):
            return False
        bounds = (self.delta_lower_bound, self.delta_upper_bound)
        if bounds == (None, None):
            return True
        return (all(_finite(v) for v in bounds) and bounds[0] <= bounds[1]
                and bounds[0] <= self.delta <= bounds[1])

    @property
    def delta(self) -> float:
        return (self.candidate - self.reference) * (1 if self.higher_is_better else -1)


@dataclass(frozen=True)
class RetentionCheck:
    scope_id: str
    reference_policy_sha256: str
    candidate_policy_sha256: str
    metrics: tuple[PairedMetric, ...]
    independent_group_ids: tuple[str, ...]


@dataclass(frozen=True)
class CapabilityCard:
    version: str
    scope: CapabilityScope
    candidate_branch: str
    evidence_kind: str
    artifacts: tuple[ArtifactRef, ...]
    metrics: tuple[PairedMetric, ...]
    evaluation_group_ids: tuple[str, ...]
    training_group_ids: tuple[str, ...]
    frozen_before_evaluation: bool = False
    chosen_using_evaluation_labels: bool = False
    exhaustive_negative_labels: bool = False
    calibrated: bool = False
    protected_scope_ids: tuple[str, ...] = ()
    retention: tuple[RetentionCheck, ...] = ()
    caveats: tuple[str, ...] = ()
    selection_method: str = "fixed"
    selection_group_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        return content_hash(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CapabilityCard":
        fields = dict(value)
        scope = dict(fields.pop("scope"))
        scope["tasks"] = tuple(scope["tasks"])
        scope["ranges"] = tuple(FeatureRange(**r) for r in scope.get("ranges", ()))
        fields["scope"] = CapabilityScope(**scope)
        fields["artifacts"] = tuple(ArtifactRef(**a) for a in fields["artifacts"])
        fields["metrics"] = tuple(PairedMetric(**m) for m in fields["metrics"])
        fields["retention"] = tuple(RetentionCheck(
            scope_id=r["scope_id"], reference_policy_sha256=r["reference_policy_sha256"],
            candidate_policy_sha256=r["candidate_policy_sha256"],
            metrics=tuple(PairedMetric(**m) for m in r["metrics"]),
            independent_group_ids=tuple(r["independent_group_ids"])) for r in fields.get("retention", ()))
        for name in ("evaluation_group_ids", "training_group_ids", "protected_scope_ids", "caveats", "selection_group_ids"):
            fields[name] = tuple(fields.get(name, ()))
        return cls(**fields)


@dataclass(frozen=True)
class GatePolicy:
    minimum_independent_groups: int = 3
    minimum_improvement: float = 0.0
    maximum_retention_loss: float = 0.0
    required_artifact_roles: tuple[str, ...] = (
        "checkpoint", "evidence", "protocol", "locked_test_manifest", "calibration", "router",
    )

    def __post_init__(self):
        if not isinstance(self.minimum_independent_groups, int) or self.minimum_independent_groups < 2:
            raise ValueError("Production gate needs at least two independent groups")
        if any(not _finite(v) or v < 0 for v in (self.minimum_improvement, self.maximum_retention_loss)):
            raise ValueError("Gate tolerances must be finite and nonnegative")
        required = {"checkpoint", "evidence", "protocol", "locked_test_manifest", "calibration", "router"}
        if not required.issubset(self.required_artifact_roles):
            raise ValueError("Required production evidence cannot be disabled")


@dataclass(frozen=True)
class GateDecision:
    card_sha256: str
    production_approved: bool
    research_comparison_valid: bool
    checks: tuple[tuple[str, bool], ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "checks": dict(self.checks)}


def evaluate_capability(card: CapabilityCard, policy: GatePolicy = GatePolicy()) -> GateDecision:
    """Fail closed on missing evidence; this does not verify file bytes (route/ledger do)."""
    metrics = card.metrics
    eval_groups = set(card.evaluation_group_ids)
    group_valid = (bool(eval_groups) and all(isinstance(g, str) and g for g in eval_groups)
                   and len(eval_groups) == len(card.evaluation_group_ids)
                   and all(isinstance(g, str) and g for g in card.training_group_ids))
    names = {m.name for m in metrics}
    paired = bool(metrics) and len(names) == len(metrics) and all(m.valid() for m in metrics)
    positives_valid = card.exhaustive_negative_labels or names.issubset(PARTIAL_LABEL_METRICS)
    interval_valid = paired and all(m.delta_lower_bound is not None
                                   and m.interval_unit == "independent_acquisition_group" for m in metrics)
    efficacy = [m for m in metrics if m.name not in SAFETY_METRICS]
    artifact_roles = [a.role for a in card.artifacts]
    artifact_valid = (len(set(artifact_roles)) == len(artifact_roles)
                      and all(_digest(a.sha256) and a.path and not Path(a.path).is_absolute()
                              and ".." not in Path(a.path).parts for a in card.artifacts))
    retention_ids = [r.scope_id for r in card.retention]
    retention_complete = (set(retention_ids) == set(card.protected_scope_ids)
                          and len(set(retention_ids)) == len(retention_ids))
    retention_ok = retention_complete and all(
        _digest(r.reference_policy_sha256) and _digest(r.candidate_policy_sha256)
        and r.candidate_policy_sha256 in {a.sha256 for a in card.artifacts if a.role == "router"}
        and len(set(r.independent_group_ids)) >= policy.minimum_independent_groups
        and len(set(r.independent_group_ids)) == len(r.independent_group_ids)
        and "" not in r.independent_group_ids
        and not (set(r.independent_group_ids) & set(card.training_group_ids))
        and bool(r.metrics) and all(m.valid() and m.interval_unit == "independent_acquisition_group"
                                   and m.delta_lower_bound is not None
                                   and m.delta_lower_bound >= -policy.maximum_retention_loss for m in r.metrics)
        for r in card.retention)
    checks = {
        "known_branch": card.candidate_branch in BRANCHES,
        "known_evidence_kind": card.evidence_kind in {"development", "locked_external"},
        "version_present": bool(card.version),
        "strict_boolean_evidence_flags": all(isinstance(getattr(card, key), bool) for key in (
            "frozen_before_evaluation", "chosen_using_evaluation_labels", "exhaustive_negative_labels", "calibrated")),
        "paired_same_budget_finite_metrics": paired,
        "label_semantics_supported": positives_valid,
        "group_ids_valid": group_valid,
        "no_training_group_overlap": not (eval_groups & set(card.training_group_ids)),
        "artifacts_well_formed": artifact_valid,
        "locked_external_evidence": card.evidence_kind == "locked_external",
        "protocol_and_router_frozen_before_evaluation": card.frozen_before_evaluation,
        "not_test_oracle_selection": not card.chosen_using_evaluation_labels,
        "selection_protocol_valid": card.selection_method in {"fixed", "nested_group_cv"}
            and (card.selection_method == "fixed" or (len(set(card.selection_group_ids)) >= 2
                 and "" not in card.selection_group_ids
                 and len(set(card.selection_group_ids)) == len(card.selection_group_ids)
                 and not (set(card.selection_group_ids) & eval_groups))),
        "independent_groups_sufficient": len(eval_groups) >= policy.minimum_independent_groups,
        "complete_negative_evidence": card.exhaustive_negative_labels,
        "independent_calibration": card.calibrated,
        "required_artifacts_present": set(policy.required_artifact_roles).issubset(artifact_roles),
        "group_intervals_not_seed_intervals": interval_valid,
        "positive_lower_bound_improvement": interval_valid and bool(efficacy)
            and all(m.delta_lower_bound >= 0 for m in efficacy)
            and any(m.delta_lower_bound > policy.minimum_improvement for m in efficacy),
        "safety_non_regression": interval_valid and SAFETY_METRICS.issubset(names)
            and all(m.delta_lower_bound >= 0 for m in metrics if m.name in SAFETY_METRICS),
        "all_protected_capabilities_replayed": retention_complete,
        "protected_capabilities_non_regression": retention_ok,
    }
    research_keys = ("known_branch", "known_evidence_kind", "version_present", "strict_boolean_evidence_flags",
                     "paired_same_budget_finite_metrics",
                     "label_semantics_supported", "group_ids_valid", "no_training_group_overlap", "artifacts_well_formed",
                     "selection_protocol_valid")
    warnings = ["No claim of autonomous recursive self-improvement or universal non-regression"]
    if not card.exhaustive_negative_labels:
        warnings.append("Positive/unknown annotations: no complete precision, F1, specificity or presence/absence claim")
    if not interval_valid or len(eval_groups) < policy.minimum_independent_groups:
        warnings.append("Insufficient independent acquisition-group evidence for a significant improvement claim")
    if card.chosen_using_evaluation_labels:
        warnings.append("Retrospective best-by-test/oracle is a non-deployable upper bound only")
    reasons = tuple(k for k, passed in checks.items() if not passed)
    # Invalid metrics (NaN) must reject safely and still produce a serializable receipt.
    try:
        card_hash = card.content_hash()
    except (TypeError, ValueError):
        card_hash = "invalid_noncanonical_card"
    return GateDecision(card_hash, not reasons, all(checks[k] for k in research_keys),
                        tuple(checks.items()), reasons, tuple(warnings))


@dataclass(frozen=True)
class RoutingDecision:
    primary_branch: str
    shadow_branch: str | None
    card_sha256: str | None
    reason: str
    teacher_preserved: bool = True
    deployment_mode: str = "production"


def evidence_contract(card: CapabilityCard) -> dict[str, Any]:
    """Expected content of the metrics evidence receipt (also accepts extra keys).

    A report writer should build this from actual paired results, then hash that
    receipt as the card's evidence artifact. Source prediction references and
    recomputation receipts may be added; this contract is not an attestation that
    an experiment took place and does not replace independent verification.
    """
    value = card.to_dict()
    keys = ("evidence_kind", "metrics", "evaluation_group_ids", "training_group_ids",
            "exhaustive_negative_labels", "calibrated", "protected_scope_ids", "retention",
            "selection_method", "selection_group_ids")
    return {"format": "phyevo-capability-evidence-v1", **{key: value[key] for key in keys}}


def verify_artifacts(card: CapabilityCard, project_root: Path) -> bool:
    if not card.artifacts or not all(a.verify(project_root) for a in card.artifacts):
        return False
    evidence = [a for a in card.artifacts if a.role == "evidence"]
    if len(evidence) != 1:
        return False
    try:
        actual = json.loads(_inside(project_root, evidence[0].path).read_text(encoding="utf-8"))
        expected = evidence_contract(card)
        return content_hash({key: actual[key] for key in expected}) == content_hash(expected)
    except (OSError, ValueError, TypeError, KeyError):
        return False


def route(context: ObservableContext, cards: tuple[CapabilityCard, ...], *, project_root: Path,
          policy: GatePolicy = GatePolicy(), allow_experimental_shadow: bool = False,
          mode: str = "production") -> RoutingDecision:
    """Deterministic inference-time routing; never accepts ground truth or test scores."""
    if mode not in {"production", "research"}:
        raise ValueError("Routing mode must be production or explicitly research")
    if not context.valid():
        return RoutingDecision("physics", None, None, "invalid_or_nonfinite_observables")
    matched = [card for card in cards if card.scope.matches(context)]
    if len(matched) != 1:
        return RoutingDecision("physics", None, None, "out_of_scope" if not matched else "ambiguous_overlapping_scopes")
    card = matched[0]
    decision = evaluate_capability(card, policy)
    if not verify_artifacts(card, project_root):
        return RoutingDecision("physics", None, decision.card_sha256, "artifact_integrity_or_availability_failed")
    if decision.production_approved:
        return RoutingDecision(card.candidate_branch, None, decision.card_sha256, "validated_local_capability")
    research_ready = (decision.research_comparison_valid and card.frozen_before_evaluation
                      and not card.chosen_using_evaluation_labels
                      and {"checkpoint", "evidence", "protocol", "router"}.issubset(a.role for a in card.artifacts))
    if mode == "research" and research_ready:
        return RoutingDecision(card.candidate_branch, None, decision.card_sha256,
                               "explicit_research_primary_not_production", deployment_mode="research")
    if allow_experimental_shadow and research_ready:
        return RoutingDecision("physics", card.candidate_branch, decision.card_sha256, "explicit_development_shadow_only")
    return RoutingDecision("physics", None, decision.card_sha256, "production_evidence_insufficient")


def run_guarded(decision: RoutingDecision, *, physics: Callable[[], Any],
                student: Callable[[], Any] | None = None,
                hybrid: Callable[[Any, Any], Any] | None = None,
                validate_candidate: Callable[[Any], bool] | None = None) -> dict[str, Any]:
    """Keep physics results even if optional neural/hybrid execution fails.

    Callers obtain the decision from ``route`` and implement hybrid fusion at
    the frozen, tested candidate budget. A candidate validator can check shapes,
    finite scores, coordinate systems and budget compliance before selection.
    Student/hybrid outputs are separate; this helper never mutates physics masks.
    """
    teacher = physics()  # A physics failure is not silently relabelled as safe.
    output = {"physics": teacher, "selected": teacher, "student": None,
              "routing": asdict(decision), "status": "physics_retained"}
    branch = decision.shadow_branch or (decision.primary_branch if decision.primary_branch != "physics" else None)
    if branch is None:
        return output
    try:
        if student is None:
            raise ValueError("Student callable unavailable")
        learned = student()
        output["student"] = learned
        if branch == "hybrid":
            if hybrid is None:
                raise ValueError("Hybrid callable unavailable")
            candidate = hybrid(copy.deepcopy(teacher), learned)
        else:
            candidate = learned
        if candidate is None or (validate_candidate is not None and not validate_candidate(candidate)):
            raise ValueError("Candidate failed output validation")
        if decision.primary_branch != "physics":
            output["selected"] = candidate
            output["status"] = ("experimental_research_primary_physics_preserved" if decision.deployment_mode == "research"
                                else "validated_local_capability")
        else:
            output["shadow"] = candidate
            output["status"] = "experimental_shadow_physics_retained"
    except Exception as error:
        output["status"] = "optional_branch_failed_physics_retained"
        output["failure_type"] = type(error).__name__
    return output


class CapabilityLedger:
    """One exclusive-create JSON file per event, hash chained, never rewritten.

    This detects accidental edits/reordering, not a malicious rewrite of the
    entire chain. Concurrent appenders compete for the same sequence filename;
    the loser receives FileExistsError and must reload/retry explicitly.
    """
    def __init__(self, project_root: str | Path, directory: str | Path,
                 policy: GatePolicy = GatePolicy()):
        self.root = Path(project_root).resolve(strict=True)
        self.directory = _inside(self.root, directory)
        self.policy = policy

    def events(self) -> list[dict[str, Any]]:
        directory = _inside(self.root, self.directory)
        if not directory.exists():
            return []
        files = sorted(directory.glob("event-*.json"))
        events = []
        previous = None
        for sequence, path in enumerate(files, 1):
            if path.name != f"event-{sequence:06d}.json":
                raise ValueError("Ledger sequence missing or reordered")
            path = _inside(self.root, path)
            value = json.loads(path.read_text(encoding="utf-8"))
            payload = {k: v for k, v in value.items() if k != "event_sha256"}
            if (payload.get("sequence") != sequence or payload.get("previous_sha256") != previous
                    or value.get("event_sha256") != content_hash(payload)):
                raise ValueError("Ledger hash-chain integrity failed")
            if payload.get("kind") not in {"capability_evaluated", "rollback"}:
                raise ValueError("Unknown ledger event kind")
            events.append(value)
            previous = value["event_sha256"]
        return events

    def _append(self, fields: dict[str, Any]) -> dict[str, Any]:
        events = self.events()
        payload = {"format": "phyevo-capability-event-v1", "sequence": len(events) + 1,
                   "previous_sha256": events[-1]["event_sha256"] if events else None, **fields}
        value = {**payload, "event_sha256": content_hash(payload)}
        directory = _inside(self.root, self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = _inside(self.root, directory / f"event-{len(events) + 1:06d}.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        return value

    def record(self, card: CapabilityCard) -> dict[str, Any]:
        decision = evaluate_capability(card, self.policy)
        verified = verify_artifacts(card, self.root)
        active = self.active_cards()
        old_scopes = {c.scope.scope_id for c in active}
        replay_bound = old_scopes.issubset(card.protected_scope_ids)
        references = {r.scope_id: r.reference_policy_sha256 for r in card.retention}
        replay_bound = replay_bound and all(references.get(c.scope.scope_id) in
                                           {a.sha256 for a in c.artifacts if a.role == "router"} for c in active)
        return self._append({"kind": "capability_evaluated", "scope_id": card.scope.scope_id,
                             "card": card.to_dict(), "card_sha256": card.content_hash(),
                             "gate_policy": asdict(self.policy), "decision": decision.to_dict(),
                             "artifact_bytes_verified": verified,
                             "previous_capabilities_replay_bound": replay_bound,
                             "activated": decision.production_approved and verified and replay_bound})

    def active_cards(self) -> tuple[CapabilityCard, ...]:
        active: dict[str, CapabilityCard] = {}
        approved: dict[str, CapabilityCard] = {}
        for event in self.events():
            scope = event["scope_id"]
            if event["kind"] == "capability_evaluated" and event["activated"]:
                card = CapabilityCard.from_dict(event["card"])
                if card.content_hash() != event["card_sha256"] or card.scope.scope_id != scope:
                    raise ValueError("Ledger card digest mismatch")
                # Never trust a stored 'activated' boolean without rechecking.
                if not evaluate_capability(card, self.policy).production_approved:
                    continue
                approved[card.content_hash()] = card
                active[scope] = card
            elif event["kind"] == "rollback":
                target = event["target_card_sha256"]
                if target is None:
                    active.pop(scope, None)
                elif target in approved and approved[target].scope.scope_id == scope:
                    active[scope] = approved[target]
                else:
                    raise ValueError("Rollback target was never approved in this scope")
        return tuple(active.values())

    def rollback(self, scope_id: str, *, target_card_sha256: str | None = None,
                 reason: str) -> dict[str, Any]:
        if not scope_id or not reason.strip():
            raise ValueError("Rollback requires a scope and an audit reason")
        events = self.events()
        if target_card_sha256 is not None:
            matching = [e for e in events if e["kind"] == "capability_evaluated" and e["activated"]
                        and e["scope_id"] == scope_id and e["card_sha256"] == target_card_sha256]
            if not matching:
                raise ValueError("Rollback target was never approved in this scope")
            card = CapabilityCard.from_dict(matching[-1]["card"])
            if not evaluate_capability(card, self.policy).production_approved or not verify_artifacts(card, self.root):
                raise ValueError("Rollback target no longer passes current integrity/evidence gate")
        return self._append({"kind": "rollback", "scope_id": scope_id,
                             "target_card_sha256": target_card_sha256, "reason": reason})
