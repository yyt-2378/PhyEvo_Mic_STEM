"""Opt-in refined-student feedback connected to the original detection framework.

Default remains native physics. Shadow exposes proposal reranking, rescue and
training-reference neighbors. Research routing additionally requires a frozen
inner-selected policy matching this exact checkpoint; no production promotion.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from .expert_runtime import TASK_GEOMETRY
from .expert_student import restore_checkpoint, torch
from .expert_supervision import GEOMETRY_INDEX, normalize_image, sha256_file, write_new_json
from .preprocessing import load_grayscale
from .rsi_evaluation import EvaluationProtocol
from .rsi_feedback import BRANCHES, embedding_reference_context, feedback_predictions, selected_configuration
from .rsi_refinement import predict_with_embedding
from .rsi_runtime import RSIDetection, WARNINGS, inside, pixel_sha256, save_rsi_detection


def _json(path):
    return json.loads(path.read_text())


def create_feedback_binding(root, checkpoint, training_protocol, output, *, selector=None, analysis_protocol=None):
    root = Path(root).resolve(strict=True)
    checkpoint = inside(root, checkpoint)
    refs = {"checkpoint": checkpoint/"checkpoint.json", "training_protocol": inside(root, training_protocol),
            "embedding_bank": checkpoint/"training_embedding_bank.npz",
            "embedding_metadata": checkpoint/"training_embedding_bank.json"}
    if (selector is None) != (analysis_protocol is None):
        raise ValueError("Research routing requires both selector and analysis protocol")
    if selector is not None:
        refs.update(selector=inside(root, selector), analysis_protocol=inside(root, analysis_protocol))
    value = {"format": "phyevo-refined-feedback-binding-v1", "production_approved": False,
             "evaluation": asdict(EvaluationProtocol()), "artifacts": {
                 role: {"path": str(inside(root,path).relative_to(root)), "sha256": sha256_file(path)} for role,path in refs.items()}}
    # Validate before writing, including train/holdout and reference-bank lineage.
    _validate_binding(root, value)
    write_new_json(inside(root, output), value)
    return value


def _validate_binding(root, value):
    if value.get("format") != "phyevo-refined-feedback-binding-v1" or value.get("production_approved") is not False:
        raise ValueError("Refined feedback is an explicit research binding, not production approval")
    refs = value["artifacts"]
    required = {"checkpoint", "training_protocol", "embedding_bank", "embedding_metadata"}
    if not required <= set(refs) or ("selector" in refs) != ("analysis_protocol" in refs):
        raise ValueError("Incomplete refined binding")
    paths = {role: inside(root, ref["path"]) for role,ref in refs.items()}
    for role,path in paths.items():
        if sha256_file(path) != refs[role]["sha256"]:
            raise ValueError(f"Changed bound artifact: {role}")
    meta, training = _json(paths["checkpoint"]), _json(paths["training_protocol"])
    if training.get("format") != "phyevo-rsi-refinement-development-v1":
        raise ValueError("Wrong training protocol")
    fold = meta.get("fold", {})
    if (meta["protocol_sha256"] != refs["training_protocol"]["sha256"] or
            meta.get("learning_manifest_sha256") != training.get("learning_manifest_sha256") or
            meta.get("seed") not in training["seeds"] or meta.get("policy") not in training["policies"] or
            meta.get("step") not in training["checkpoints"] or fold not in training["folds"] or
            fold.get("stage") != "outer" or len(fold.get("excluded_groups", [])) != 1 or
            set(fold["excluded_groups"]) & set(meta["train_groups"]) or
            meta["train_case_numbers"] != fold["train_case_numbers"] or
            set(fold["train_case_numbers"]) & set(fold["validation_case_numbers"])):
        raise ValueError("Refined checkpoint/protocol/outer-holdout mismatch")
    state = inside(root, paths["checkpoint"].parent/"state.npz")
    if sha256_file(state) != meta["state_sha256"]:
        raise ValueError("Refined checkpoint state changed")
    bank_meta = _json(paths["embedding_metadata"])
    if (bank_meta.get("sha256") != refs["embedding_bank"]["sha256"] or
            bank_meta.get("checkpoint_sha256") != refs["checkpoint"]["sha256"] or
            bank_meta.get("protocol_sha256") != refs["training_protocol"]["sha256"] or
            bank_meta.get("selection_uses_validation") is not False or
            not {r["case_number"] for r in bank_meta["cases"]} <= set(meta["train_case_numbers"]) or
            not {r["group"] for r in bank_meta["cases"]} <= set(meta["train_groups"])):
        raise ValueError("Embedding reference bank is not bound to this training set")
    selector = None
    if "selector" in paths:
        selector, analysis = _json(paths["selector"]), _json(paths["analysis_protocol"])
        if (selector.get("format") != "phyevo-refinement-feedback-selector-v1" or
                selector.get("selection_uses_outer_labels") is not False or
                selector.get("outer_group_excluded") != fold["excluded_groups"][0] or
                selector.get("production_approved") is not False or
                analysis.get("training_protocol_sha256") != refs["training_protocol"]["sha256"] or
                value["evaluation"] != analysis.get("evaluation")):
            raise ValueError("Selector is not bound to this held-out learner/evaluation")
    return paths, meta, bank_meta, selector, EvaluationProtocol(**value["evaluation"])


def load_feedback_binding(root, path):
    root = Path(root).resolve(strict=True)
    path = inside(root, path)
    result = _validate_binding(root, _json(path))
    return (*result, sha256_file(path))


def create_feedback_registry(root, selector, bindings, default_shadow, output):
    """Bind one outer learner family to observable-scope model choices.

    Model keys are policy::step, never image/case/acquisition identifiers. The
    caller's fixed seed and outer exclusion are checked across every member.
    """
    root=Path(root).resolve(strict=True)
    selector=inside(root,selector)
    expected_hash=sha256_file(selector)
    frozen=_json(selector)
    registered={}
    seeds=set()
    for key,binding in bindings.items():
        path=inside(root,binding)
        _,meta,_,child_selector,_,_=load_feedback_binding(root,path)
        value=_json(path)
        if (key!=f"{meta['policy']}::{meta['step']}" or child_selector!=frozen or
                value["artifacts"].get("selector",{}).get("sha256")!=expected_hash):
            raise ValueError("Registry member differs from frozen selector or model key")
        registered[key]={"path":str(path.relative_to(root)),"sha256":sha256_file(path)}
        seeds.add(meta["seed"])
    needed={f"{scope['selected']['policy']}::{scope['selected']['step']}" for scope in frozen["scopes"].values() if scope.get("selected")}
    if len(seeds)!=1 or default_shadow not in registered or not needed<=set(registered):
        raise ValueError("Registry lacks a selected model or mixes fixed seeds")
    value={"format":"phyevo-refined-feedback-registry-v1","production_approved":False,
           "selector":{"path":str(selector.relative_to(root)),"sha256":expected_hash},
           "bindings":registered,"default_shadow":default_shadow,"seed":next(iter(seeds)),
           "inference_features":["geometry","teacher_empty"],
           "outer_group_excluded":frozen["outer_group_excluded"]}
    write_new_json(inside(root,output),value)
    return value


def resolve_context_binding(root, path, *, geometry, teacher_empty):
    """Choose a bound model from native task/empty status, never expert labels."""
    path=inside(root,path)
    registry=_json(path)
    if registry.get("format")=="phyevo-refined-feedback-binding-v1":
        return path,None
    if (registry.get("format")!="phyevo-refined-feedback-registry-v1" or
            registry.get("production_approved") is not False or
            registry.get("inference_features")!=["geometry","teacher_empty"]):
        raise ValueError("Unknown feedback registry or inference features")
    ref=registry["selector"]
    selector_path=inside(root,ref["path"])
    if sha256_file(selector_path)!=ref["sha256"]:raise ValueError("Registry selector changed")
    selector=_json(selector_path)
    if selector.get("outer_group_excluded")!=registry["outer_group_excluded"]:
        raise ValueError("Registry and selector outer exclusion differ")
    choice=selected_configuration(selector,geometry=geometry,teacher_empty=teacher_empty)
    key=f"{choice['policy']}::{choice['step']}" if choice else registry["default_shadow"]
    selected=registry["bindings"].get(key)
    if selected is None:raise ValueError("Selected model absent from feedback registry")
    selected_path=inside(root,selected["path"])
    if sha256_file(selected_path)!=selected["sha256"]:raise ValueError("Registry binding changed")
    value=_json(selected_path)
    _,meta,_,child_selector,_,_=load_feedback_binding(root,selected_path)
    if (key!=f"{meta['policy']}::{meta['step']}" or meta["seed"]!=registry["seed"] or
            child_selector!=selector or value["artifacts"].get("selector",{}).get("sha256")!=ref["sha256"]):
        raise ValueError("Registry resolved a different model family")
    return selected_path,sha256_file(path)


def detect_with_feedback(framework, source, *, project_root, binding=None,
                         shadow=False, experimental_research=False, student_device="cpu", **detect_options):
    root = Path(project_root).resolve(strict=True)
    if isinstance(source, (str, Path)):
        source = inside(root, source)
    for name in ("skeleton", "exemplar", "sam3_exchange_directory"):
        if isinstance(detect_options.get(name), (str,Path)):
            detect_options[name] = inside(root, detect_options[name])
    physics = framework.detect(source, **detect_options)
    result = RSIDetection(physics, metadata={"format": "phyevo-refined-feedback-detection-v1",
        "status": "physics_default", "reason": "no_explicit_optional_feedback", "production_approved": False,
        "physics_preserved": True, "coordinate_space": "source_pixels_xy", "warnings": WARNINGS.copy()})
    if not (shadow or experimental_research):
        return result
    try:
        if binding is None:
            raise ValueError("A hash-bound refinement checkpoint is required")
        geometry = TASK_GEOMETRY.get(physics.task)
        if geometry not in ("point","interface"):
            raise ValueError("Unsupported or unvalidated refinement geometry")
        resolved,registry_hash=resolve_context_binding(root,binding,geometry=geometry,teacher_empty=not bool(physics.mask.any()))
        paths, metadata, bank_meta, selector, protocol, binding_hash = load_feedback_binding(root, resolved)
        if geometry not in ("point", "interface") or geometry not in metadata["geometry_support"]:
            raise ValueError("Unsupported or unvalidated refinement geometry")
        if experimental_research and selector is None:
            raise ValueError("No frozen inner selector; shadow only")
        image = load_grayscale(source, frame_index=detect_options.get("frame_index"),
                               projection=detect_options.get("projection"), dataset_path=detect_options.get("dataset_path")) if isinstance(source,Path) else np.asarray(source)
        if (pixel_sha256(image) != physics.metadata["input_provenance"]["pixel_sha256"] or
                image.shape != tuple(physics.image_shape) or image.shape != physics.mask.shape):
            raise ValueError("Raw pixel provenance differs between physics and student")
        model, _, _, restored = restore_checkpoint(paths["checkpoint"].parent, device=student_device)
        if restored != metadata:
            raise ValueError("Checkpoint changed after binding validation")
        previous, warn = torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            score, embedding = predict_with_embedding(model, normalize_image(image), GEOMETRY_INDEX[geometry], device=student_device)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn)
        with np.load(paths["embedding_bank"], allow_pickle=False) as archive:
            bank = archive["embeddings"]
        context = embedding_reference_context(embedding, bank, bank_meta["cases"], geometry)
        predictions = feedback_predictions(image, physics.mask, [r.to_dict() for r in physics.regions], score, geometry, protocol=protocol)
        primary, reason = "physics_native", "explicit_shadow_only"
        chosen = selected_configuration(selector, geometry=geometry, teacher_empty=not bool(physics.mask.any())) if selector else None
        if experimental_research:
            primary, reason = "physics_budgeted", "inner_selector_retains_physics"
            if chosen:
                if chosen["policy"] != metadata["policy"] or chosen["step"] != metadata["step"]:
                    raise ValueError("Bound checkpoint is not selected for this observable scope")
                if chosen["branch"] not in BRANCHES:
                    raise ValueError("Unknown selected feedback branch")
                if chosen["embedding_veto"] and context["novelty_veto"]:
                    reason = "experimental_embedding_veto_retains_physics"
                else:
                    primary, reason = chosen["branch"], "inner_selected_refinement_not_production"
        result.candidates, result.student_score, result.primary_branch = predictions, score, primary
        result.metadata.update(status="research_feedback" if experimental_research else "physics_with_feedback_shadow",
            reason=reason, geometry=geometry, selected_configuration=chosen, embedding_context=context,
            query_embedding=embedding.tolist(), binding_sha256=binding_hash,
            registry_sha256=registry_hash, resolved_binding=str(resolved.relative_to(root)),
            checkpoint=str(paths["checkpoint"].parent.relative_to(root)), policy=metadata["policy"], step=metadata["step"],
            seed=metadata["seed"], raw_pixel_sha256=pixel_sha256(image), protocol=asdict(protocol),
            student_normalization="raw_plane_once_by_expert_supervision.normalize_image",
            learned_feedback_scope="candidate_priority_and_rescue_only_not_physical_measurements",
            inference_features=["geometry", "teacher_empty", "optional_training_reference_distance"])
    except Exception as error:
        result.primary_branch = "physics_native"
        result.metadata.update(status="physics_fallback", reason=str(error), failure_type=type(error).__name__)
    return result


def save_feedback_detection(result, config, output, *, project_root):
    summary = save_rsi_detection(result, config, output, project_root=project_root)
    if not result.candidates:
        return summary
    import matplotlib.pyplot as plt
    from .rsi_feedback import FEEDBACK_BRANCH
    output = inside(Path(project_root), output)
    names = ["physics_budgeted", "student_expert", "hybrid_fixed_budget", FEEDBACK_BRANCH]
    titles = ["Physics at fixed budget", "Student only", "Fixed physics + student", "Student reranks physics + rescue"]
    fig, axes = plt.subplots(1,4,figsize=(16,4), constrained_layout=True)
    for ax, name, title in zip(axes, names, titles):
        ax.imshow(result.physics.normalized_image, cmap="gray", vmin=0, vmax=1)
        prediction = result.candidates[name]
        if prediction.points_xy is not None:
            xy = prediction.points_xy
            if len(xy):
                ax.scatter(xy[:,0],xy[:,1],s=15,facecolors="none",edgecolors="#e0ac40",linewidths=.6)
            quantity=f"{len(xy)} candidate points"
        else:
            if prediction.mask.any() and not prediction.mask.all():
                ax.contour(prediction.mask.astype(float), levels=[.5], colors=["#e0ac40"], linewidths=.5)
            quantity=f"{prediction.mask.mean():.1%} image area"
        ax.set_title(title+"\n"+quantity, fontsize=10)
        ax.set_axis_off()
    fig.suptitle("Research proposals only | physical evidence unchanged | empty prediction is not a defect-free verdict", fontsize=11)
    path = output/"feedback_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    write_new_json(output/"feedback_artifact.json", {"rsi_detection_sha256": sha256_file(output/"rsi_detection.json"),
        "figure": {"path": str(path.relative_to(Path(project_root).resolve())), "sha256": sha256_file(path)}})
    return summary
