"""Runtime contract tests; synthetic fixtures are not performance evidence."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from tfree_detection import rsi_runtime as runtime
from tfree_detection.expert_supervision import sha256_file
from tfree_detection.models import DetectionRegion, DetectionResult
from tfree_detection.rsi_evaluation import EvaluationProtocol


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream)
    return path


def bundle(root, *, branch="student_expert", outer="acquisition-a", checkpoint_outer=None,
           support=("point", "interface"), metadata_update=None):
    router = {"format": "phyevo-nested-observable-router-v1", "outer_group_excluded": outer,
              "selection_uses_outer_labels": False, "inference_features": ["geometry", "teacher_empty"],
              "production_approved": False,
              "scopes": {f"{geometry}::{condition}": {"selected_branch": branch, "seeds": [23]}
                         for geometry in ("point", "interface") for condition in ("teacher_empty", "teacher_nonempty")}}
    training = {"seed": 23, "steps": 1200, "learning_manifest_sha256": "a" * 64,
                "folds": [{"held_out_group": outer, "train_case_numbers": [1, 2]}]}
    files = {"router": put(root / "router.json", router),
             "evaluation_protocol": put(root / "evaluation.json", {"protocols": {"main": asdict(EvaluationProtocol())}}),
             "training_protocol": put(root / "training.json", training)}
    directory = root / "checkpoint"
    directory.mkdir()
    with (directory / "state.npz").open("xb") as stream:
        np.savez(stream, stub=np.zeros(1))
    metadata = {"protocol_sha256": sha256_file(files["training_protocol"]),
                "held_out_group": checkpoint_outer if checkpoint_outer is not None else outer,
                "train_groups": ["acquisition-b"], "variant": "teacher_plus_expert", "seed": 23,
                "learning_manifest_sha256": "a" * 64, "step": 1200, "train_case_numbers": [1, 2],
                "state_sha256": sha256_file(directory / "state.npz"), "geometry_support": list(support),
                "unvalidated_geometries": ["region", "line"]}
    metadata.update(metadata_update or {})
    files["checkpoint"] = put(directory / "checkpoint.json", metadata)
    binding = put(root / "binding.json", {"format": "phyevo-rsi-runtime-binding-v1", "artifacts": [
        {"role": role, "path": str(path.relative_to(root)), "sha256": sha256_file(path)} for role, path in files.items()]})
    return binding, metadata


def physics(image, task="atomic_point_defect", *, nonempty=False):
    mask = np.zeros(image.shape, bool)
    regions = []
    if nonempty:
        mask[4:7, 4:7] = True
        regions = [DetectionRegion(1, (4, 4, 7, 7), (5, 5), 9, 1, 1, 0.8, "high", [], reliability=0.8)]
    mask.setflags(write=False)
    return DetectionResult(task, image.shape, image * 0 + 0.123, image.copy(), image.copy(), mask,
        regions, {}, {"original_physics": np.ones(image.shape)},
        {"input_provenance": {"pixel_sha256": runtime.pixel_sha256(image)}})


class Framework:
    def __init__(self, result):
        self.result = result
    def detect(self, source, **kwargs):
        return self.result


def install_student(monkeypatch, metadata, calls=None):
    monkeypatch.setattr(runtime, "restore_checkpoint", lambda *args, **kwargs: (object(), None, None, metadata))
    def score(model, image, geometry, **options):
        assert runtime.torch.are_deterministic_algorithms_enabled()
        if calls is not None:
            calls.append((image.copy(), geometry, options))
        return np.random.default_rng(42).random(image.shape).astype(np.float32)
    monkeypatch.setattr(runtime, "predict_scores", score)


def test_default_native_does_not_load_any_optional_artifact(tmp_path, monkeypatch):
    image = np.ones((16, 16))
    teacher = physics(image)
    monkeypatch.setattr(runtime, "load_binding", lambda *a: pytest.fail("Should not load a student by default"))
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding="not-a-file")
    assert result.primary is teacher
    assert result.metadata["status"] == "physics_default"
    assert not result.candidates


@pytest.mark.parametrize("branch", ["physics_budgeted", "student_expert", "hybrid_fixed_budget"])
@pytest.mark.parametrize("task", ["atomic_point_defect", "phase_boundary"])
def test_explicit_research_is_bound_budgeted_and_preserves_physics(tmp_path, monkeypatch, branch, task):
    binding, metadata = bundle(tmp_path, branch=branch)
    image = np.random.default_rng(2).integers(0, 65000, (32, 48), dtype=np.uint16)
    teacher = physics(image, task, nonempty=True)
    before = teacher.summary()
    calls = []
    install_student(monkeypatch, metadata, calls)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path,
                               binding=binding, experimental_research=True)
    assert result.primary_branch == branch
    assert result.physics is teacher and teacher.summary() == before
    assert teacher.mask.flags.writeable is False
    assert result.metadata["production_approved"] is False
    assert np.array_equal(calls[0][0], image)
    assert calls[0][2]["normalized"] is False
    assert result.metadata["raw_pixel_sha256"] == runtime.pixel_sha256(image)
    assert "student_teacher_only" not in result.candidates
    for name in ("student_expert", "hybrid_fixed_budget", "physics_budgeted"):
        prediction = result.candidates[name]
        assert (len(prediction.points_xy) if task == "atomic_point_defect" else prediction.mask.sum()) <= prediction.budget
    assert result.metadata["coordinate_space"] == "source_pixels_xy"


def test_shadow_never_switches_native_primary(tmp_path, monkeypatch):
    binding, metadata = bundle(tmp_path)
    install_student(monkeypatch, metadata)
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding, shadow=True)
    assert result.primary is teacher
    assert result.metadata["status"] == "physics_with_shadow"
    assert result.candidates["student_expert"].points_xy.shape[0] > 0


@pytest.mark.parametrize("updates,reason", [
    ({"held_out_group": None}, "outer-group"),
    ({"train_groups": ["acquisition-a"]}, "outer-group"),
    ({"variant": "teacher_only"}, "outer-group"),
    ({"seed": 41}, "seed"),
    ({"learning_manifest_sha256": "b" * 64}, "manifest"),
    ({"train_case_numbers": [3]}, "inventory"),
    ({"protocol_sha256": "c" * 64}, "protocol"),
    ({"state_sha256": "d" * 64}, "state"),
])
def test_lineage_substitutions_fail_closed(tmp_path, updates, reason):
    binding, _ = bundle(tmp_path, metadata_update=updates)
    with pytest.raises(ValueError, match=reason):
        runtime.load_binding(tmp_path, binding)
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding, experimental_research=True)
    assert result.primary is teacher
    assert reason in result.metadata["reason"]


def test_changed_raw_pixels_do_not_silently_route(tmp_path, monkeypatch):
    binding, metadata = bundle(tmp_path)
    install_student(monkeypatch, metadata)
    image = np.ones((16, 16))
    teacher = physics(image)
    image[0, 0] = 0
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding, experimental_research=True)
    assert result.primary is teacher
    assert "pixel hash" in result.metadata["reason"]
    assert not result.candidates


@pytest.mark.parametrize("task,support,reason", [
    ("symmetry_relation", ("point", "interface"), "unsupported_geometry"),
    ("stacking_fault", ("point", "interface"), "unsupported_geometry"),
    ("amorphous_region", ("point", "region"), "unvalidated_geometry"),
])
def test_unsupported_or_unvalidated_geometry_falls_back_before_model_load(tmp_path, monkeypatch, task, support, reason):
    binding, _ = bundle(tmp_path, support=support)
    monkeypatch.setattr(runtime, "restore_checkpoint", lambda *a, **k: pytest.fail("Unsupported geometry must not load model"))
    image = np.ones((16, 16))
    result = runtime.detect_rsi(Framework(physics(image, task)), image, project_root=tmp_path,
                               binding=binding, experimental_research=True)
    assert result.primary_branch == "physics_native" and result.metadata["reason"] == reason


def test_neural_failure_retains_physics_but_physics_error_propagates(tmp_path, monkeypatch):
    binding, _ = bundle(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("injected device failure")
    monkeypatch.setattr(runtime, "restore_checkpoint", fail)
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding, experimental_research=True)
    assert result.primary is teacher
    assert result.metadata["failure_type"] == "RuntimeError"
    framework = Framework(teacher)
    framework.detect = fail
    with pytest.raises(RuntimeError, match="device failure"):
        runtime.detect_rsi(framework, image, project_root=tmp_path, binding=binding, experimental_research=True)


def test_deterministic_setting_restored_after_inference_error(monkeypatch):
    previous = runtime.torch.are_deterministic_algorithms_enabled()
    runtime.torch.use_deterministic_algorithms(False)
    def fail(*args, **kwargs):
        assert runtime.torch.are_deterministic_algorithms_enabled()
        raise RuntimeError("inference failure")
    monkeypatch.setattr(runtime, "predict_scores", fail)
    try:
        with pytest.raises(RuntimeError):
            runtime._predict_reproducibly(None, np.ones((16, 16)), "point", "cpu")
        assert not runtime.torch.are_deterministic_algorithms_enabled()
    finally:
        runtime.torch.use_deterministic_algorithms(previous)


def test_external_source_and_symlink_escape_rejected(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    outside = put(tmp_path / "outside.json", {})
    (root / "link.json").symlink_to(outside)
    for path in (outside, root / "link.json", root):
        with pytest.raises(ValueError, match="real child"):
            runtime.inside(root, path)


def test_nonfinite_student_scores_fail_closed(tmp_path, monkeypatch):
    binding, metadata = bundle(tmp_path)
    install_student(monkeypatch, metadata)
    monkeypatch.setattr(runtime, "predict_scores", lambda m, image, g, **kw: image * np.nan)
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding, experimental_research=True)
    assert result.primary is teacher and result.metadata["status"] == "physics_fallback"


def test_save_exclusive_preserves_original_summary_and_budget_arrays(tmp_path, monkeypatch):
    from tfree_detection import reporting
    binding, metadata = bundle(tmp_path)
    install_student(monkeypatch, metadata)
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding, experimental_research=True)
    def save(original, config, path):
        assert original is teacher
        put(path / "summary.json", original.summary())
    monkeypatch.setattr(reporting, "save_result", save)
    output = tmp_path / "output"
    receipt = runtime.save_rsi_detection(result, None, output, project_root=tmp_path)
    assert receipt["primary_branch"] == "student_expert"
    for artifact in receipt["candidate_artifacts"].values():
        assert sha256_file(tmp_path / artifact["path"]) == artifact["sha256"]
    with pytest.raises(FileExistsError, match="overwrite"):
        runtime.save_rsi_detection(result, None, output, project_root=tmp_path)


def test_cli_refuses_existing_output_before_running_physics(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts/run_rsi_detection.py"
    spec = importlib.util.spec_from_file_location("rsi_cli_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.main(["--project-root", str(tmp_path), "--input", "absent.png", "--config", "absent.yaml",
                     "--output", str(output)])


def make_card(root, binding, *, production):
    from dataclasses import replace
    from tfree_detection.rsi_policy import ArtifactRef, CapabilityCard, CapabilityScope, PairedMetric, evidence_contract
    frozen = runtime.load_binding(root, binding)
    refs = [ArtifactRef(role, frozen.artifacts[source].path, frozen.artifacts[source].sha256)
            for role, source in (("checkpoint", "checkpoint"), ("router", "router"), ("protocol", "evaluation_protocol"))]
    for role in ("locked_test_manifest", "calibration"):
        path = put(root / f"{role}.json", {"synthetic_gate_fixture": True})
        refs.append(ArtifactRef(role, path.name, sha256_file(path)))
    budget_sha = frozen.artifacts["evaluation_protocol"].sha256
    metrics = [PairedMetric("marked_point_recall", 0.1, 0.2, True, budget_sha, budget_sha,
                           0.05, 0.15, "independent_acquisition_group")]
    if production:
        metrics += [PairedMetric(name, 0.2, 0.1, False, budget_sha, budget_sha,
                                 0.05, 0.15, "independent_acquisition_group")
                    for name in ("false_positive_rate", "calibration_error", "ood_risk")]
    card = CapabilityCard("synthetic-runtime-v1", CapabilityScope("point", ("atomic_point_defect",), "point"),
        "student", "locked_external" if production else "development", tuple(refs), tuple(metrics),
        ("test-a", "test-b", "test-c"), ("train-z",), frozen_before_evaluation=True,
        exhaustive_negative_labels=production, calibrated=production)
    path = put(root / "evidence.json", evidence_contract(card))
    card = replace(card, artifacts=card.artifacts + (ArtifactRef("evidence", path.name, sha256_file(path)),))
    return put(root / "card.json", card.to_dict())


@pytest.mark.parametrize("production", [False, True])
def test_production_requires_strict_card_not_a_research_claim(tmp_path, monkeypatch, production):
    binding, metadata = bundle(tmp_path)
    card = make_card(tmp_path, binding, production=production)
    ledger = runtime.CapabilityLedger(tmp_path, "activation-ledger")
    recorded = runtime.CapabilityCard.from_dict(json.loads(card.read_text()))
    assert ledger.record(recorded)["activated"] is production
    install_student(monkeypatch, metadata)
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding,
                               cards=(card,), production_ledger=ledger.directory)
    assert result.metadata["production_approved"] is production
    assert result.primary_branch == ("student_expert" if production else "physics_native")
    if not production:
        assert "not active" in result.metadata["reason"]
        assert not result.candidates
    else:
        assert result.metadata["production_card_sha256"] == recorded.content_hash()
        assert result.metadata["production_ledger_head_sha256"] == ledger.events()[-1]["event_sha256"]


def test_missing_card_does_not_load_student_or_change_primary(tmp_path, monkeypatch):
    binding, _ = bundle(tmp_path)
    monkeypatch.setattr(runtime, "restore_checkpoint", lambda *a, **k: pytest.fail("Unbound card must not load model"))
    image = np.ones((16, 16))
    teacher = physics(image)
    directory = tmp_path / "empty-ledger"
    directory.mkdir()
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path, binding=binding,
                               cards=("missing.json",), production_ledger=directory)
    assert result.primary is teacher
    assert result.metadata["failure_type"] == "FileNotFoundError"


def test_valid_production_card_without_ledger_cannot_run(tmp_path, monkeypatch):
    binding, _ = bundle(tmp_path)
    card_path = make_card(tmp_path, binding, production=True)
    monkeypatch.setattr(runtime, "restore_checkpoint", lambda *a, **k: pytest.fail("No activation authority"))
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path,
                               binding=binding, cards=(card_path,))
    assert result.primary is teacher and not result.metadata["production_approved"]
    assert "require an activation ledger" in result.metadata["reason"]


def test_rolled_back_card_cannot_be_resurrected_by_direct_path(tmp_path, monkeypatch):
    binding, _ = bundle(tmp_path)
    path = make_card(tmp_path, binding, production=True)
    card = runtime.CapabilityCard.from_dict(json.loads(path.read_text()))
    ledger = runtime.CapabilityLedger(tmp_path, "activation-ledger")
    assert ledger.record(card)["activated"]
    ledger.rollback(card.scope.scope_id, reason="Synthetic operational rollback")
    assert not ledger.active_cards()
    monkeypatch.setattr(runtime, "restore_checkpoint", lambda *a, **k: pytest.fail("Revoked model must not load"))
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path,
                               binding=binding, cards=(path,), production_ledger=ledger.directory)
    assert result.primary is teacher and not result.metadata["production_approved"]
    assert "not active" in result.metadata["reason"]


def test_card_passing_own_gate_cannot_bypass_failed_activation(tmp_path, monkeypatch):
    from dataclasses import replace
    from tfree_detection.rsi_policy import evaluate_capability
    binding, _ = bundle(tmp_path)
    path = make_card(tmp_path, binding, production=True)
    first = runtime.CapabilityCard.from_dict(json.loads(path.read_text()))
    ledger = runtime.CapabilityLedger(tmp_path, "activation-ledger")
    assert ledger.record(first)["activated"]
    second = replace(first, version="synthetic-v2-without-old-capability-replay")
    assert evaluate_capability(second).production_approved
    event = ledger.record(second)
    assert not event["activated"] and not event["previous_capabilities_replay_bound"]
    second_path = put(tmp_path / "unactivated-card.json", second.to_dict())
    monkeypatch.setattr(runtime, "restore_checkpoint", lambda *a, **k: pytest.fail("Unactivated model must not load"))
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path,
                               binding=binding, cards=(second_path,), production_ledger=ledger.directory)
    assert result.primary is teacher and "not active" in result.metadata["reason"]


def test_ledger_can_supply_only_its_currently_active_cards(tmp_path, monkeypatch):
    binding, metadata = bundle(tmp_path)
    path = make_card(tmp_path, binding, production=True)
    card = runtime.CapabilityCard.from_dict(json.loads(path.read_text()))
    ledger = runtime.CapabilityLedger(tmp_path, "activation-ledger")
    ledger.record(card)
    install_student(monkeypatch, metadata)
    image = np.ones((16, 16))
    result = runtime.detect_rsi(Framework(physics(image)), image, project_root=tmp_path,
                               binding=binding, production_ledger=ledger.directory)
    assert result.primary_branch == "student_expert" and result.metadata["production_approved"]
    assert result.metadata["production_card_sha256"] == card.content_hash()


def test_rollback_during_inference_invalidates_optional_primary(tmp_path, monkeypatch):
    binding, metadata = bundle(tmp_path)
    path = make_card(tmp_path, binding, production=True)
    card = runtime.CapabilityCard.from_dict(json.loads(path.read_text()))
    ledger = runtime.CapabilityLedger(tmp_path, "activation-ledger")
    ledger.record(card)
    install_student(monkeypatch, metadata)
    def score_and_revoke(model, image, geometry, **options):
        ledger.rollback(card.scope.scope_id, reason="Synthetic revocation while inference was running")
        return np.ones(image.shape, dtype=np.float32)
    monkeypatch.setattr(runtime, "predict_scores", score_and_revoke)
    image = np.ones((16, 16))
    teacher = physics(image)
    result = runtime.detect_rsi(Framework(teacher), image, project_root=tmp_path,
                               binding=binding, cards=(path,), production_ledger=ledger.directory)
    assert result.primary is teacher and not result.metadata["production_approved"]
    assert "changed during inference" in result.metadata["reason"]
    assert not result.candidates
