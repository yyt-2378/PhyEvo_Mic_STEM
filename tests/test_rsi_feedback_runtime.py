from dataclasses import asdict
import json

import numpy as np
import pytest

from test_rsi_runtime import Framework, physics, put
from tfree_detection import rsi_feedback_runtime as runtime
from tfree_detection.expert_supervision import normalize_image, sha256_file
from tfree_detection.rsi_evaluation import EvaluationProtocol
from tfree_detection.rsi_feedback import FEEDBACK_BRANCH


def bundle(root, *, selector=False):
    fold = {"fold_id": "outer-00", "stage": "outer", "excluded_groups": ["heldout"],
            "train_case_numbers": [1,2], "validation_case_numbers": [99]}
    training = put(root/"training.json", {"format": "phyevo-rsi-refinement-development-v1",
        "learning_manifest_sha256": "abc", "seeds": [23,41], "policies": {"control": {}}, "checkpoints": [400,800], "folds": [fold]})
    checkpoint = root/"step-000400"
    checkpoint.mkdir()
    with (checkpoint/"state.npz").open("xb") as stream:
        np.savez(stream, placeholder=np.zeros(1))
    metadata = {"protocol_sha256": sha256_file(training), "learning_manifest_sha256": "abc", "fold": fold,
                "seed": 23, "policy": "control", "step": 400, "train_groups": ["a"], "train_case_numbers": [1,2],
                "geometry_support": ["point", "interface"], "state_sha256": sha256_file(checkpoint/"state.npz")}
    put(checkpoint/"checkpoint.json", metadata)
    with (checkpoint/"training_embedding_bank.npz").open("xb") as stream:
        np.savez(stream, embeddings=np.array([[1.,0],[.99,.01]]))
    put(checkpoint/"training_embedding_bank.json", {"sha256": sha256_file(checkpoint/"training_embedding_bank.npz"),
        "protocol_sha256": sha256_file(training), "checkpoint_sha256": sha256_file(checkpoint/"checkpoint.json"),
        "selection_uses_validation": False, "cases": [{"case_number": n,"group": "a","geometry": "point"} for n in (1,2)]})
    extra = {}
    if selector:
        chosen = {"policy": "control", "step": 400, "branch": FEEDBACK_BRANCH, "embedding_veto": False}
        selection = put(root/"selector.json", {"format": "phyevo-refinement-feedback-selector-v1", "selection_uses_outer_labels": False,
            "outer_group_excluded": "heldout", "production_approved": False, "scopes": {
                "point::teacher_nonempty": {"selected": chosen,"inner_groups":["a","b"]},
                "point::teacher_empty": {"selected": {**chosen,"step":800},"inner_groups":["a","b"]}}})
        analysis = put(root/"analysis.json", {"training_protocol_sha256": sha256_file(training), "evaluation": asdict(EvaluationProtocol())})
        extra = {"selector": selection, "analysis_protocol": analysis}
    runtime.create_feedback_binding(root, checkpoint, training, root/"binding.json", **extra)
    return root/"binding.json", metadata


def install(monkeypatch, meta, calls):
    monkeypatch.setattr(runtime,"restore_checkpoint",lambda *a,**k: (object(),None,None,meta))
    def predict(model,image,geometry,**kwargs):
        assert runtime.torch.are_deterministic_algorithms_enabled()
        calls.append(image.copy())
        return np.random.default_rng(17).random(image.shape).astype(np.float32), np.array([1.,0])
    monkeypatch.setattr(runtime,"predict_with_embedding",predict)


def test_default_is_physics_only_and_does_not_read_binding(tmp_path, monkeypatch):
    image = np.ones((32,32))
    native = physics(image)
    monkeypatch.setattr(runtime,"load_feedback_binding",lambda *a: pytest.fail("optional model read"))
    result = runtime.detect_with_feedback(Framework(native),image,project_root=tmp_path,binding="missing")
    assert result.primary is native


@pytest.mark.parametrize("research", [False,True])
def test_feedback_is_integrated_with_native_physics_and_source_pixels(tmp_path,monkeypatch,research):
    binding, meta = bundle(tmp_path,selector=research)
    calls = []
    install(monkeypatch,meta,calls)
    image = np.random.default_rng(9).integers(0,65000,(32,48),dtype=np.uint16)
    native = physics(image,nonempty=True)
    before = native.summary()
    result = runtime.detect_with_feedback(Framework(native),image,project_root=tmp_path,binding=binding,
                                          shadow=not research,experimental_research=research)
    assert result.metadata["status"] != "physics_fallback", result.metadata
    assert result.physics is native and before == native.summary()
    assert result.primary_branch == (FEEDBACK_BRANCH if research else "physics_native")
    np.testing.assert_array_equal(calls[0],normalize_image(image))
    assert result.metadata["embedding_context"]["neighbors"][0]["case_number"] == 1
    assert result.metadata["production_approved"] is False


def test_changed_bound_reference_falls_back_to_original_physics(tmp_path):
    binding, _ = bundle(tmp_path)
    # A newly bound hash cannot silently make an invalid bank a training bank.
    payload = json.loads(binding.read_text())
    meta_path = tmp_path/payload["artifacts"]["embedding_metadata"]["path"]
    changed = json.loads(meta_path.read_text())
    changed["cases"][0]["case_number"] = 99
    substitute = put(tmp_path/"tampered_bank.json",changed)
    payload["artifacts"]["embedding_metadata"] = {"path": str(substitute.relative_to(tmp_path)), "sha256": sha256_file(substitute)}
    new_binding = put(tmp_path/"tampered_binding.json",payload)
    with pytest.raises(ValueError,match="training set"):
        runtime.load_feedback_binding(tmp_path,new_binding)
    image = np.ones((32,32))
    native = physics(image)
    result = runtime.detect_with_feedback(Framework(native),image,project_root=tmp_path,binding=new_binding,shadow=True)
    assert result.primary is native and result.metadata["status"] == "physics_fallback"


def test_research_cannot_use_unselected_shadow_checkpoint(tmp_path):
    binding, _ = bundle(tmp_path)
    image = np.ones((32,32))
    native = physics(image)
    result = runtime.detect_with_feedback(Framework(native),image,project_root=tmp_path,binding=binding,experimental_research=True)
    assert result.primary is native and result.metadata["reason"] == "No frozen inner selector; shadow only"


def second_binding(root, original, *, seed=23):
    """Synthetic second checkpoint in the same frozen model family."""
    payload=json.loads(original.read_text())
    paths={k:root/v["path"] for k,v in payload["artifacts"].items()}
    directory=root/"step-000800"
    directory.mkdir()
    with (directory/"state.npz").open("xb") as stream:
        np.savez(stream,placeholder=np.ones(1))
    metadata=json.loads(paths["checkpoint"].read_text())
    metadata.update(step=800,seed=seed,state_sha256=sha256_file(directory/"state.npz"))
    put(directory/"checkpoint.json",metadata)
    with (directory/"training_embedding_bank.npz").open("xb") as stream:
        np.savez(stream,embeddings=np.array([[1.,0],[.99,.01]]))
    bank=json.loads(paths["embedding_metadata"].read_text())
    bank.update(sha256=sha256_file(directory/"training_embedding_bank.npz"),
                checkpoint_sha256=sha256_file(directory/"checkpoint.json"))
    put(directory/"training_embedding_bank.json",bank)
    output=root/"binding-800.json"
    runtime.create_feedback_binding(root,directory,paths["training_protocol"],output,
        selector=paths["selector"],analysis_protocol=paths["analysis_protocol"])
    return output,metadata


@pytest.mark.parametrize("empty,step",[(False,400),(True,800)])
def test_registry_selects_checkpoint_from_native_observables_only(tmp_path,monkeypatch,empty,step):
    first,meta400=bundle(tmp_path,selector=True)
    second,meta800=second_binding(tmp_path,first)
    registry=tmp_path/"registry.json"
    runtime.create_feedback_registry(tmp_path,tmp_path/"selector.json",
        {"control::400":first,"control::800":second},"control::400",registry)
    calls=[]
    install(monkeypatch,meta800 if empty else meta400,calls)
    image=np.random.default_rng(9).integers(0,65000,(32,48),dtype=np.uint16)
    native=physics(image,nonempty=not empty)
    result=runtime.detect_with_feedback(Framework(native),image,project_root=tmp_path,binding=registry,
                                       experimental_research=True)
    assert result.metadata["status"]=="research_feedback",result.metadata
    assert result.metadata["step"]==step
    assert result.metadata["registry_sha256"]==sha256_file(registry)
    assert result.primary_branch==FEEDBACK_BRANCH
    assert result.physics is native
    # A scope with no selected student uses the fixed shadow model, not a test-case lookup.
    path,digest=runtime.resolve_context_binding(tmp_path,registry,geometry="interface",teacher_empty=False)
    assert path==first and digest==sha256_file(registry)


def test_registry_rejects_incomplete_model_inventory_and_mixed_seeds(tmp_path):
    first,_=bundle(tmp_path,selector=True)
    with pytest.raises(ValueError,match="lacks a selected model"):
        runtime.create_feedback_registry(tmp_path,tmp_path/"selector.json",
            {"control::400":first},"control::400",tmp_path/"incomplete.json")
    second,_=second_binding(tmp_path,first,seed=41)
    with pytest.raises(ValueError,match="mixes fixed seeds"):
        runtime.create_feedback_registry(tmp_path,tmp_path/"selector.json",
            {"control::400":first,"control::800":second},"control::400",tmp_path/"mixed.json")


def test_changed_registry_member_falls_back_without_loading_student(tmp_path,monkeypatch):
    first,_=bundle(tmp_path,selector=True)
    second,_=second_binding(tmp_path,first)
    registry=tmp_path/"registry.json"
    value=runtime.create_feedback_registry(tmp_path,tmp_path/"selector.json",
        {"control::400":first,"control::800":second},"control::400",registry)
    value["bindings"]["control::400"]["sha256"]="0"*64
    altered=put(tmp_path/"changed-registry.json",value)
    monkeypatch.setattr(runtime,"restore_checkpoint",lambda *a,**k:pytest.fail("invalid member reached student"))
    image=np.ones((32,32))
    native=physics(image,nonempty=True)
    result=runtime.detect_with_feedback(Framework(native),image,project_root=tmp_path,binding=altered,shadow=True)
    assert result.primary is native and result.metadata["status"]=="physics_fallback"
    assert result.metadata["reason"]=="Registry binding changed"


def test_actual_selected_metrics_are_recorded_without_production_activation(tmp_path,monkeypatch):
    from scripts import analyze_rsi_refinement as analysis_script
    from scripts import run_rsi_feedback_detection as runner
    from tfree_detection.rsi_policy import CapabilityLedger
    first,meta=bundle(tmp_path,selector=True)
    second,_=second_binding(tmp_path,first)
    bindings=tmp_path/"bindings"
    registry=bindings/"outer-00/seed-23/registry.json"
    registry.parent.mkdir(parents=True)
    runtime.create_feedback_registry(tmp_path,tmp_path/"selector.json",
        {"control::400":first,"control::800":second},"control::400",registry)
    selected=json.loads((tmp_path/"selector.json").read_text())["scopes"]["point::teacher_nonempty"]["selected"]
    analysis=tmp_path/"analysis-final"
    rows=[{"main_comparison_eligible":True,"group":"heldout","seed":23,"geometry":"point",
        "teacher_empty":False,"case_number":99,"selected_configuration":selected,"selected_branch":FEEDBACK_BRANCH,
        "checkpoint":"step-000400","methods":{"physics_budgeted":{"positive_coverage":.1},"nested_feedback":{"positive_coverage":.3}}}]
    put(analysis/"selected_outer_cases.json",rows)
    protocol={"evaluation":asdict(EvaluationProtocol())}
    put(analysis/"analysis_protocol.json",protocol)
    put(analysis/"result_seal.json",{"files":{name:sha256_file(analysis/name) for name in ("selected_outer_cases.json","analysis_protocol.json")}})
    monkeypatch.setattr(runner,"ROOT",tmp_path)
    monkeypatch.setattr(analysis_script,"load_analysis",lambda path:(analysis,protocol,tmp_path,{"folds":[meta["fold"]]},tmp_path))
    put(tmp_path/"src/tfree_detection/rsi_feedback_runtime.py",{"synthetic_source_snapshot":True})
    summary=runner.record_evidence(analysis,bindings,tmp_path/"evidence")
    assert summary["events"]==1 and summary["production_activated"]==0
    event=CapabilityLedger(tmp_path,tmp_path/"evidence/events").events()[0]
    assert not event["activated"] and event["artifact_bytes_verified"]
    assert event["decision"]["research_comparison_valid"]
    assert summary["decisions"][0]["candidate"]==.3
    with pytest.raises(FileExistsError):runner.record_evidence(analysis,bindings,tmp_path/"evidence")
