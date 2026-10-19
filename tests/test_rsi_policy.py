"""Synthetic governance tests; these are not scientific evidence of improvement."""
from dataclasses import asdict, replace
import hashlib
import json

import pytest

from tfree_detection.rsi_policy import (
    ArtifactRef, CapabilityCard, CapabilityLedger, CapabilityScope, FeatureRange,
    GatePolicy, ObservableContext, PairedMetric, RetentionCheck, RoutingDecision,
    content_hash, evaluate_capability, evidence_contract, route, run_guarded, verify_artifacts,
)


BUDGET = "b" * 64


def metric(name="marked_point_recall", *, delta=0.1, lower=0.05, unit="independent_acquisition_group"):
    high = name not in {"false_positive_rate", "calibration_error", "ood_risk"}
    return PairedMetric(name, 0.4, 0.4 + (delta if high else -delta), high, BUDGET, BUDGET,
                        lower, max(delta, lower) + 0.1, unit)


def seal(root, name, value):
    path = root / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def card(root, *, production=True, version="v1", scope_id="point_periodic", **updates):
    artifacts = []
    for role in ("checkpoint", "protocol", "router", "locked_test_manifest", "calibration"):
        name = f"{version}-{role}.json"
        artifacts.append(ArtifactRef(role, name, seal(root, name, {"synthetic": role, "version": version})))
    scope = CapabilityScope(scope_id, ("atomic_point_defect",), "point", (FeatureRange("periodicity", 0.6, 1.0),))
    value = CapabilityCard(
        version, scope, "hybrid", "locked_external" if production else "development", tuple(artifacts),
        (metric(), *(metric(name) for name in ("false_positive_rate", "calibration_error", "ood_risk")))
        if production else (metric(),),
        ("held-out-a", "held-out-b", "held-out-c") if production else ("held-out-a",), ("train-a",),
        frozen_before_evaluation=True, exhaustive_negative_labels=production, calibrated=production,
    )
    value = replace(value, **updates)
    name = f"{version}-evidence.json"
    artifact = ArtifactRef("evidence", name, seal(root, name, evidence_contract(value)))
    return replace(value, artifacts=value.artifacts + (artifact,))


CONTEXT = ObservableContext("atomic_point_defect", "point", periodicity=0.8)


def test_locked_complete_paired_evidence_passes_and_is_locally_routed(tmp_path):
    value = card(tmp_path)
    decision = evaluate_capability(value)
    assert decision.production_approved
    assert decision.research_comparison_valid
    assert verify_artifacts(value, tmp_path)
    selection = route(CONTEXT, (value,), project_root=tmp_path)
    assert selection.primary_branch == "hybrid"
    assert selection.teacher_preserved
    assert CapabilityCard.from_dict(value.to_dict()) == value
    assert len(decision.card_sha256) == 64


def test_current_positive_only_development_data_never_promotes(tmp_path):
    value = card(tmp_path, production=False)
    decision = evaluate_capability(value)
    assert decision.research_comparison_valid
    assert not decision.production_approved
    assert {"locked_external_evidence", "complete_negative_evidence", "independent_groups_sufficient"}.issubset(decision.reasons)
    assert any("precision" in warning for warning in decision.warnings)
    assert route(CONTEXT, (value,), project_root=tmp_path).primary_branch == "physics"
    shadow = route(CONTEXT, (value,), project_root=tmp_path, allow_experimental_shadow=True)
    assert shadow.primary_branch == "physics" and shadow.shadow_branch == "hybrid"
    research = route(CONTEXT, (value,), project_root=tmp_path, mode="research")
    assert research.primary_branch == "hybrid" and research.deployment_mode == "research"


@pytest.mark.parametrize("change,reason", [
    ({"chosen_using_evaluation_labels": True}, "not_test_oracle_selection"),
    ({"frozen_before_evaluation": False}, "protocol_and_router_frozen_before_evaluation"),
    ({"training_group_ids": ("held-out-a",)}, "no_training_group_overlap"),
    ({"calibrated": False}, "independent_calibration"),
    ({"evaluation_group_ids": ("held-out-a",) * 3}, "group_ids_valid"),
    ({"candidate_branch": "uncontrolled_autonomous"}, "known_branch"),
    ({"protected_scope_ids": ("forgotten_old_skill",)}, "all_protected_capabilities_replayed"),
])
def test_missing_or_invalid_evidence_fails_closed(tmp_path, change, reason):
    value = card(tmp_path, **change)
    decision = evaluate_capability(value)
    assert not decision.production_approved
    assert reason in decision.reasons
    assert route(CONTEXT, (value,), project_root=tmp_path).primary_branch == "physics"


def test_oracle_cannot_be_research_primary_or_shadow(tmp_path):
    value = card(tmp_path, production=False, chosen_using_evaluation_labels=True)
    assert route(CONTEXT, (value,), project_root=tmp_path, mode="research").primary_branch == "physics"
    assert route(CONTEXT, (value,), project_root=tmp_path, allow_experimental_shadow=True).shadow_branch is None
    assert "non-deployable" in evaluate_capability(value).warnings[-1]


@pytest.mark.parametrize("name", ["case_id", "expert_recall", "held_out_group", "ground_truth", "student_confidence"])
def test_router_cannot_use_labels_ids_or_uncalibrated_scores(name):
    with pytest.raises(ValueError, match="observables"):
        FeatureRange(name, 0, 1)


@pytest.mark.parametrize("context", [
    ObservableContext("atomic_point_defect", "point"),
    ObservableContext("atomic_point_defect", "point", periodicity=0.3),
    ObservableContext("amorphous_region", "region", periodicity=0.8),
    ObservableContext("atomic_point_defect", "point", periodicity=float("nan")),
    ObservableContext("atomic_point_defect", "point", periodicity=1.1),
])
def test_missing_features_out_of_scope_nan_and_wrong_task_retain_physics(tmp_path, context):
    value = card(tmp_path)
    assert route(context, (value,), project_root=tmp_path).primary_branch == "physics"


def test_overlapping_capabilities_do_not_silently_prioritize(tmp_path):
    first, second = card(tmp_path), card(tmp_path, version="v2", scope_id="different_name_same_scope")
    result = route(CONTEXT, (first, second), project_root=tmp_path)
    assert result.primary_branch == "physics"
    assert result.reason == "ambiguous_overlapping_scopes"


@pytest.mark.parametrize("bad_metric", [
    replace(metric(), candidate=float("nan")),
    replace(metric(), candidate=float("inf")),
    replace(metric(), candidate_budget_sha256="c" * 64),
    replace(metric(), interval_unit="random_seed"),
    replace(metric(), delta_lower_bound=-0.1),
    replace(metric(), delta_lower_bound=0.5, delta_upper_bound=0.6),
    replace(metric("false_positive_rate"), higher_is_better=True),
])
def test_metric_direction_intervals_budget_and_finiteness(tmp_path, bad_metric):
    value = card(tmp_path)
    value = replace(value, metrics=(bad_metric,) + value.metrics[1:])
    assert not evaluate_capability(value).production_approved


def test_complete_precision_is_not_invented_from_positive_only_labels(tmp_path):
    value = card(tmp_path, production=False, metrics=(replace(metric(), name="precision"),))
    assert not evaluate_capability(value).research_comparison_valid
    assert "label_semantics_supported" in evaluate_capability(value).reasons


def test_metric_receipt_is_bound_to_actual_card_claims(tmp_path):
    value = card(tmp_path)
    changed = replace(value, metrics=(replace(value.metrics[0], reference=0.45),) + value.metrics[1:])
    assert all(a.verify(tmp_path) for a in changed.artifacts)
    assert not verify_artifacts(changed, tmp_path)
    assert route(CONTEXT, (changed,), project_root=tmp_path).primary_branch == "physics"


def test_wrong_hash_missing_file_and_escape_are_rejected(tmp_path):
    value = card(tmp_path)
    wrong = replace(value.artifacts[0], sha256="a" * 64)
    missing = replace(value.artifacts[0], path="absent.json")
    escape = replace(value.artifacts[0], path="../outside.json")
    absolute = replace(value.artifacts[0], path=str(tmp_path / value.artifacts[0].path))
    for artifact in (wrong, missing, escape, absolute):
        changed = replace(value, artifacts=(artifact,) + value.artifacts[1:])
        assert route(CONTEXT, (changed,), project_root=tmp_path).primary_branch == "physics"


def test_exception_fallback_preserves_physics_and_does_not_hide_physics_failure():
    teacher = {"mask": [1, 0, 0]}
    decision = RoutingDecision("student", None, "a" * 64, "test")

    def fail():
        raise RuntimeError("GPU failure")

    result = run_guarded(decision, physics=lambda: teacher, student=fail)
    assert result["physics"] is teacher and result["selected"] is teacher
    assert result["status"] == "optional_branch_failed_physics_retained"
    with pytest.raises(RuntimeError):
        run_guarded(decision, physics=fail, student=lambda: 1)


def test_explicit_research_inference_runs_real_branch_without_mutating_physics():
    teacher = {"mask": [1, 0, 0]}
    decision = RoutingDecision("hybrid", None, "a" * 64, "test", deployment_mode="research")

    def hybrid(physical, learned):
        physical["mask"][1] = learned
        return physical

    result = run_guarded(decision, physics=lambda: teacher, student=lambda: 1, hybrid=hybrid)
    assert result["selected"]["mask"] == [1, 1, 0]
    assert teacher["mask"] == [1, 0, 0]
    assert result["status"] == "experimental_research_primary_physics_preserved"


def test_shadow_and_missing_hybrid_never_replace_teacher():
    teacher = [1]
    decision = RoutingDecision("physics", "student", "a" * 64, "test")
    result = run_guarded(decision, physics=lambda: teacher, student=lambda: [2])
    assert result["selected"] is teacher and result["shadow"] == [2]
    failure = run_guarded(replace(decision, shadow_branch="hybrid"), physics=lambda: teacher, student=lambda: [2])
    assert failure["selected"] is teacher and failure["failure_type"] == "ValueError"


def test_ledger_rejection_and_rollback_are_append_only(tmp_path):
    ledger = CapabilityLedger(tmp_path, "ledger")
    value = card(tmp_path)
    first = ledger.record(value)
    first_bytes = (tmp_path / "ledger/event-000001.json").read_bytes()
    assert first["activated"]
    assert ledger.active_cards() == (value,)
    rejection = ledger.record(card(tmp_path, production=False, version="v2"))
    assert not rejection["activated"]
    assert ledger.active_cards() == (value,)
    rollback = ledger.rollback(value.scope.scope_id, reason="safe fallback")
    assert rollback["target_card_sha256"] is None
    assert ledger.active_cards() == ()
    ledger.rollback(value.scope.scope_id, target_card_sha256=value.content_hash(), reason="restore verified scope")
    assert ledger.active_cards() == (value,)
    assert (tmp_path / "ledger/event-000001.json").read_bytes() == first_bytes
    assert len(ledger.events()) == 4


def test_ledger_cannot_replace_old_skill_without_replay(tmp_path):
    ledger = CapabilityLedger(tmp_path, "ledger")
    first = card(tmp_path)
    assert ledger.record(first)["activated"]
    second = card(tmp_path, version="v2")
    assert evaluate_capability(second).production_approved
    event = ledger.record(second)
    assert not event["activated"]
    assert not event["previous_capabilities_replay_bound"]
    assert ledger.active_cards() == (first,)


def test_retention_must_cover_groups_and_use_same_candidate_router(tmp_path):
    base = card(tmp_path)
    router_hash = next(a.sha256 for a in base.artifacts if a.role == "router")
    check = RetentionCheck("old", "f" * 64, router_hash, (metric(),), ("g1", "g2", "g3"))
    candidate = replace(base, protected_scope_ids=("old",), retention=(check,))
    assert evaluate_capability(candidate).production_approved
    for changed in (replace(check, candidate_policy_sha256="e" * 64),
                    replace(check, independent_group_ids=("g1",)),
                    replace(check, independent_group_ids=("g1", "g2", "train-a")),
                    replace(check, metrics=(replace(metric(), delta_lower_bound=-0.2),))):
        assert not evaluate_capability(replace(candidate, retention=(changed,))).production_approved


def test_unknown_and_cross_scope_rollback_refused(tmp_path):
    ledger = CapabilityLedger(tmp_path, "ledger")
    value = card(tmp_path)
    ledger.record(value)
    with pytest.raises(ValueError, match="never approved"):
        ledger.rollback(value.scope.scope_id, target_card_sha256="f" * 64, reason="unknown")
    with pytest.raises(ValueError, match="never approved"):
        ledger.rollback("wrong-scope", target_card_sha256=value.content_hash(), reason="cross-scope")
    assert len(ledger.events()) == 1


def test_ledger_detects_tamper_without_overwriting_evidence(tmp_path):
    # A deliberately malformed fixture is created initially, not by altering a
    # successful run's ledger or real project data.
    directory = tmp_path / "broken-ledger"
    directory.mkdir()
    fixture = {"sequence": 1, "previous_sha256": None, "kind": "rollback", "event_sha256": "0" * 64}
    seal(directory, "event-000001.json", fixture)
    ledger = CapabilityLedger(tmp_path, directory)
    with pytest.raises(ValueError, match="integrity"):
        ledger.events()
    with pytest.raises(ValueError, match="integrity"):
        ledger.rollback("any", reason="must not append on corrupted history")
    assert len(list(directory.iterdir())) == 1


def test_policy_cannot_disable_core_safety_or_accept_nan():
    with pytest.raises(ValueError):
        GatePolicy(minimum_independent_groups=1)
    with pytest.raises(ValueError):
        GatePolicy(required_artifact_roles=("checkpoint",))
    with pytest.raises(ValueError):
        GatePolicy(maximum_retention_loss=float("nan"))
    with pytest.raises(ValueError):
        content_hash({"nan": float("nan")})


def test_external_ledger_location_refused(tmp_path):
    with pytest.raises(ValueError, match="inside"):
        CapabilityLedger(tmp_path, tmp_path.parent / "not-this-project")


def test_invalid_mode_is_not_a_silent_production_override(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        route(CONTEXT, (), project_root=tmp_path, mode="force")


def test_actual_image_proxies_are_named_honestly_and_empty_teacher_is_observable(tmp_path):
    scope = CapabilityScope("teacher_empty_point", ("atomic_point_defect",), "point",
                            (FeatureRange("teacher_empty", 1, 1),))
    value = card(tmp_path, production=False, scope=scope)
    observed = ObservableContext("atomic_point_defect", "point", teacher_empty=True,
                                 physics_area_fraction=0, laplacian_noise_proxy=3.4,
                                 fft_top20_power_fraction=0.7, robust_contrast_p95_p05=24)
    assert observed.snr_estimate is None and observed.physics_valid_fraction is None
    assert observed.valid()
    assert route(observed, (value,), project_root=tmp_path, mode="research").primary_branch == "hybrid"
    assert route(replace(observed, teacher_empty=False), (value,), project_root=tmp_path, mode="research").primary_branch == "physics"
    assert not replace(observed, teacher_empty=0.7).valid()
    assert not replace(observed, laplacian_noise_proxy=-1).valid()


@pytest.mark.parametrize("inner_groups,valid", [
    (("inner-a", "inner-b"), True),
    (("inner-a",), False),
    (("inner-a", "held-out-a"), False),
    (("inner-a", "inner-a"), False),
])
def test_nested_selection_requires_two_inner_groups_and_never_outer_labels(tmp_path, inner_groups, valid):
    value = card(tmp_path, production=False, selection_method="nested_group_cv", selection_group_ids=inner_groups)
    assert evaluate_capability(value).research_comparison_valid is valid
    assert (route(CONTEXT, (value,), project_root=tmp_path, mode="research").primary_branch == "hybrid") is valid


@pytest.mark.parametrize("flag", ["calibrated", "exhaustive_negative_labels", "frozen_before_evaluation"])
def test_string_boolean_claims_cannot_enable_production(tmp_path, flag):
    value = card(tmp_path, **{flag: "false"})
    assert not evaluate_capability(value).production_approved
    assert not evaluate_capability(value).research_comparison_valid


def test_invalid_candidate_and_failed_output_validation_fall_back():
    decision = RoutingDecision("student", None, "a" * 64, "test")
    for candidate in (None, [float("nan")]):
        result = run_guarded(decision, physics=lambda: [1], student=lambda: candidate,
                             validate_candidate=lambda value: False)
        assert result["selected"] == [1]
        assert result["status"] == "optional_branch_failed_physics_retained"
