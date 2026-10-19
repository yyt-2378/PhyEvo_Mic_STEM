"""Opt-in controlled-RSI inference with immutable physics and bound model lineage.

Nested development routing is not production approval. The original detector is
always run and retained; learned coordinates/masks are a separate, budgeted
candidate stream, never physical evidence or chemical identities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .expert_runtime import TASK_GEOMETRY
from .expert_student import predict_scores, restore_checkpoint, torch
from .expert_supervision import GEOMETRY_INDEX, normalize_image, sha256_file, write_new_json
from .models import DetectionResult
from .preprocessing import load_grayscale
from .rsi_evaluation import EvaluationProtocol, Prediction, observable_features, predict_methods
from .rsi_policy import ArtifactRef, CapabilityCard, CapabilityLedger, ObservableContext, route
from .rsi_router import select_branch


REQUIRED_ROLES = frozenset({"router", "evaluation_protocol", "checkpoint", "training_protocol"})
WARNINGS = ["Research candidates are not confirmed defects or chemical identities",
            "Student scores are uncalibrated; no OOD or universal non-regression guarantee",
            "Fixed-budget candidates may exist on a normal image",
            "Physics evidence belongs to the unchanged physics result, not the student"]


def inside(root: Path, value: str | Path) -> Path:
    root = root.resolve(strict=True)
    path = (root / value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("RSI path must be a real child of project_root")
    return path


def pixel_sha256(image: np.ndarray) -> str:
    raw = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(raw.dtype).encode())
    digest.update(np.asarray(raw.shape, dtype=np.int64).tobytes())
    digest.update(raw.view(np.uint8).tobytes())
    return digest.hexdigest()


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


@dataclass(frozen=True)
class RuntimeBinding:
    checkpoint: Path
    checkpoint_metadata: Mapping
    router: Mapping
    protocol: EvaluationProtocol
    artifacts: Mapping[str, ArtifactRef]
    manifest_sha256: str


def load_binding(project_root: str | Path, path: str | Path) -> RuntimeBinding:
    """Verify all bytes and cross-bind a router to its exact outer-fold learner.

    SHA receipts detect accidental substitutions, not hostile rewriting of an
    entire receipt chain. A bundle is a research deployment specification, not
    independent certification that its development evidence is sufficient.
    """
    root = Path(project_root).resolve(strict=True)
    path = inside(root, path)
    value = _read(path)
    if value.get("format") != "phyevo-rsi-runtime-binding-v1":
        raise ValueError("Unknown runtime binding format")
    refs = [ArtifactRef(**item) for item in value["artifacts"]]
    artifacts = {ref.role: ref for ref in refs}
    if len(refs) != len(artifacts) or not REQUIRED_ROLES.issubset(artifacts):
        raise ValueError("Missing or duplicate runtime lineage artifacts")
    if not all(ref.verify(root) for ref in refs):
        raise ValueError("Runtime artifact checksum or availability failed")
    router = _read(inside(root, artifacts["router"].path))
    if (router.get("format") != "phyevo-nested-observable-router-v1"
            or router.get("selection_uses_outer_labels") is not False
            or router.get("inference_features") != ["geometry", "teacher_empty"]
            or not router.get("outer_group_excluded")):
        raise ValueError("Router is not an outer-label-independent frozen nested rule")
    evaluation = _read(inside(root, artifacts["evaluation_protocol"].path))
    evaluation = evaluation.get("protocols", {}).get("main", evaluation)
    protocol = EvaluationProtocol(**evaluation)
    metadata_path = inside(root, artifacts["checkpoint"].path)
    if metadata_path.name != "checkpoint.json":
        raise ValueError("Checkpoint artifact must identify checkpoint.json")
    metadata = _read(metadata_path)
    training = _read(inside(root, artifacts["training_protocol"].path))
    if metadata.get("protocol_sha256") != artifacts["training_protocol"].sha256:
        raise ValueError("Checkpoint training protocol lineage mismatch")
    outer = router["outer_group_excluded"]
    if (metadata.get("held_out_group") != outer or outer in metadata.get("train_groups", [])
            or metadata.get("variant") != "teacher_plus_expert"):
        raise ValueError("Router/checkpoint outer-group mismatch; all-label substitution is forbidden")
    router_seeds = {seed for scope in router.get("scopes", {}).values() for seed in scope.get("seeds", [])}
    if metadata.get("seed") not in router_seeds:
        raise ValueError("Checkpoint seed was not represented in the frozen router evidence")
    if (metadata.get("learning_manifest_sha256") != training.get("learning_manifest_sha256")
            or metadata.get("seed") != training.get("seed")
            or metadata.get("step") != training.get("steps")):
        raise ValueError("Checkpoint training manifest, seed or completed step mismatch")
    folds = [fold for fold in training.get("folds", []) if fold.get("held_out_group") == outer]
    if len(folds) != 1 or metadata.get("train_case_numbers") != folds[0].get("train_case_numbers"):
        raise ValueError("Outer-fold training inventory mismatch")
    state = inside(root, metadata_path.parent / "state.npz")
    if sha256_file(state) != metadata.get("state_sha256"):
        raise ValueError("Checkpoint state checksum mismatch")
    return RuntimeBinding(metadata_path.parent, metadata, router, protocol, artifacts, sha256_file(path))


@dataclass
class RSIDetection:
    physics: DetectionResult
    primary_branch: str = "physics_native"
    candidates: dict[str, Prediction] = field(default_factory=dict)
    student_score: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def primary(self) -> DetectionResult | Prediction:
        return self.physics if self.primary_branch == "physics_native" else self.candidates[self.primary_branch]


def _predict_reproducibly(model, image, geometry, device):
    """Match the training/evaluation producer, restoring caller-global settings.

    Runtime entry points are intended for serial inference in a process. CUDA
    callers must set CUBLAS_WORKSPACE_CONFIG before initializing the device;
    the CLI does this. This setting is not a guarantee across library versions.
    """
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        return predict_scores(model, image, GEOMETRY_INDEX[geometry], device=device,
                              normalized=False, tile_size=256, overlap=64)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


def _active_production_cards(root: Path, ledger_path, requested_cards=()):
    """A valid card is not an activation: bind inference to current ledger state.

    Reading an archived card directly must not resurrect a rolled-back version
    or bypass a ledger rejection caused by missing old-capability replay.
    Explicit research/shadow inference does not assert production activation.
    """
    if ledger_path is None:
        raise ValueError("Production capability cards require an activation ledger")
    path = inside(root, ledger_path)
    if not path.is_dir():
        raise ValueError("Production activation ledger directory is missing")
    ledger = CapabilityLedger(root, path)
    before = ledger.events()
    active = ledger.active_cards()
    after = ledger.events()
    if before != after:
        raise ValueError("Production ledger changed while resolving active cards")
    by_hash = {card.content_hash(): card for card in active}
    if requested_cards:
        selected = tuple(CapabilityCard.from_dict(_read(inside(root, value))) for value in requested_cards)
        if any(card.content_hash() not in by_hash for card in selected):
            raise ValueError("Requested production card is not active in the activation ledger")
    else:
        selected = active
    head = after[-1]["event_sha256"] if after else None
    return selected, ledger, head


def detect_rsi(framework, source, *, project_root, binding=None, cards=(),
               production_ledger=None, experimental_research=False, shadow=False,
               student_device="cpu", **detect_options) -> RSIDetection:
    """Run physics first; optional branch failures fall back without suppressing it.

    Neither this function nor the router accepts case IDs, acquisition IDs or
    expert geometry. The held-out group in checkpoint provenance validates a
    training artifact; it is not an inference-time feature or a test oracle.
    """
    root = Path(project_root).resolve(strict=True)
    if isinstance(source, (str, Path)):
        source = inside(root, source)
    for name in ("skeleton", "exemplar", "sam3_exchange_directory"):
        if isinstance(detect_options.get(name), (str, Path)):
            detect_options[name] = inside(root, detect_options[name])
    physics = framework.detect(source, **detect_options)  # Never swallow physics errors.
    result = RSIDetection(physics, metadata={
        "format": "phyevo-controlled-rsi-detection-v1", "status": "physics_default",
        "reason": "no_explicit_research_shadow_or_approved_card", "physics_preserved": True,
        "experimental_research": bool(experimental_research), "production_approved": False,
        "coordinate_space": "source_pixels_xy", "warnings": WARNINGS.copy(),
    })
    if not (experimental_research or shadow or cards or production_ledger):
        return result
    try:
        if binding is None:
            raise ValueError("A frozen runtime binding is required for optional inference")
        frozen = load_binding(root, binding)
        geometry = TASK_GEOMETRY.get(physics.task)
        if geometry not in frozen.checkpoint_metadata.get("geometry_support", []):
            result.metadata.update(status="physics_fallback", reason="unsupported_geometry")
            return result
        if geometry in frozen.checkpoint_metadata.get("unvalidated_geometries", ["line", "region"]):
            result.metadata.update(status="physics_fallback", reason="unvalidated_geometry")
            return result
        image = (load_grayscale(source, frame_index=detect_options.get("frame_index"),
                                projection=detect_options.get("projection"), dataset_path=detect_options.get("dataset_path"))
                 if isinstance(source, (str, Path)) else np.asarray(source))
        digest = pixel_sha256(image)
        if digest != physics.metadata["input_provenance"]["pixel_sha256"]:
            raise ValueError("Raw input pixel hash differs between physics and student")
        if image.shape != tuple(physics.image_shape) or physics.mask.shape != image.shape:
            raise ValueError("Physics image geometry differs from raw source pixels")
        normalized = normalize_image(image)
        features = observable_features(normalized, physics.mask)
        context = ObservableContext(task=physics.task, geometry=geometry,
            teacher_empty=not bool(physics.mask.any()), **{name: features[name] for name in (
                "physics_area_fraction", "laplacian_noise_proxy", "fft_top20_power_fraction", "robust_contrast_p95_p05")})
        selected = "physics_native"
        reason = "explicit_shadow_only"
        activation_ledger = None
        activation_head = None
        if experimental_research:
            selected = select_branch(frozen.router, geometry=geometry, teacher_empty=not bool(physics.mask.any()))
            reason = "frozen_nested_research_router_not_production"
        elif cards or production_ledger:
            loaded_cards, activation_ledger, activation_head = _active_production_cards(
                root, production_ledger, cards)
            result.metadata.update(production_ledger=str(activation_ledger.directory.relative_to(root)),
                                   production_ledger_head_sha256=activation_head)
            decision = route(context, loaded_cards, project_root=root)
            reason = decision.reason
            if decision.primary_branch != "physics":
                matching = [c for c in loaded_cards if c.content_hash() == decision.card_sha256]
                if len(matching) != 1:
                    raise ValueError("Production capability card is not uniquely bound")
                bound = {ref.role: ref for ref in matching[0].artifacts}
                for card_role, binding_role in (("checkpoint", "checkpoint"), ("router", "router"),
                                                ("protocol", "evaluation_protocol")):
                    if bound[card_role].sha256 != frozen.artifacts[binding_role].sha256:
                        raise ValueError("Production card does not authorize these runtime artifacts")
                selected = {"student": "student_expert", "hybrid": "hybrid_fixed_budget"}[decision.primary_branch]
                result.metadata["production_approved"] = True
                result.metadata["production_card_sha256"] = decision.card_sha256
            elif not shadow:
                result.metadata.update(reason=reason, status="physics_production_gate_retained")
                return result
        model, _, _, metadata = restore_checkpoint(frozen.checkpoint, device=student_device)
        if metadata != frozen.checkpoint_metadata:
            raise ValueError("Checkpoint changed after binding verification")
        # The original raw plane is passed explicitly. The learning normalizer
        # runs exactly once inside predict_scores; physics normalization may
        # differ and must never be fed into the student as though it were raw.
        score = _predict_reproducibly(model, image, geometry, student_device)
        predictions = predict_methods(image, physics.mask, [region.to_dict() for region in physics.regions],
                                      score, score, geometry, protocol=frozen.protocol)
        # There is only one learned model at runtime: do not publish a duplicate
        # score as a fictitious "teacher-only student" ablation.
        predictions.pop("student_teacher_only")
        if activation_ledger is not None and result.metadata["production_approved"]:
            # A revocation during a long GPU inference must not be ignored.
            events_now = activation_ledger.events()
            head_now = events_now[-1]["event_sha256"] if events_now else None
            if head_now != activation_head:
                raise ValueError("Production activation ledger changed during inference; retain physics")
        result.candidates = predictions
        result.student_score = score
        result.primary_branch = selected
        result.metadata.update(status="research_routed" if experimental_research else (
            "production_card_routed" if result.metadata["production_approved"] else "physics_with_shadow"),
            reason=reason, geometry=geometry, raw_pixel_sha256=digest, image_shape=list(image.shape),
            binding_sha256=frozen.manifest_sha256, checkpoint_sha256=frozen.artifacts["checkpoint"].sha256,
            lineage_artifacts={role: asdict(ref) for role, ref in frozen.artifacts.items()},
            checkpoint=str(frozen.checkpoint.relative_to(root)),
            held_out_training_group=frozen.checkpoint_metadata["held_out_group"],
            protocol=asdict(frozen.protocol), inference_features=["geometry", "teacher_empty"],
            student_normalization="raw_plane_once_by_expert_supervision.normalize_image",
            deterministic_algorithms=True, student_device=student_device,
            interface_semantics="affected_band_not_exact_centerline" if geometry == "interface" else None)
    except Exception as error:
        result.primary_branch = "physics_native"
        result.metadata.update(status="physics_fallback", reason=str(error), failure_type=type(error).__name__,
                               production_approved=False)
    return result


def save_rsi_detection(result: RSIDetection, config, output, *, project_root) -> dict:
    """Exclusive-create output; original reporting remains in its own directory."""
    root = Path(project_root).resolve(strict=True)
    output = inside(root, output)
    if output.exists():
        raise FileExistsError("Refusing to overwrite an RSI detection")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    from .reporting import save_result

    save_result(result.physics, config, output / "physics_original")
    artifacts = {}
    summaries = {}
    for name, prediction in result.candidates.items():
        array_path = inside(root, output / f"{name}.npz")
        payload = {"points_xy": prediction.points_xy} if prediction.points_xy is not None else {"mask": prediction.mask}
        with array_path.open("xb") as stream:
            np.savez_compressed(stream, **payload)
        artifacts[name] = {"path": str(array_path.relative_to(root)), "sha256": sha256_file(array_path)}
        summaries[name] = {"geometry": prediction.geometry, "budget": prediction.budget,
            "candidate_count": len(prediction.points_xy) if prediction.points_xy is not None else None,
            "area_pixels": int(prediction.mask.sum()) if prediction.mask is not None else None,
            "metadata": prediction.metadata}
    if result.student_score is not None:
        path = output / "student_score.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, score=result.student_score)
        artifacts["student_score"] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
    if result.candidates:
        path = output / "candidate_comparison.png"
        _save_comparison(result, path)
        artifacts["candidate_comparison"] = {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
    summary = {**result.metadata, "primary_branch": result.primary_branch, "candidates": summaries,
               "candidate_artifacts": artifacts, "physics_summary": "physics_original/summary.json"}
    write_new_json(output / "rsi_detection.json", summary)
    return summary


def _save_comparison(result: RSIDetection, path: Path) -> None:
    """Source-pixel overlays; budgeted geometry is not relabelled as physics truth."""
    import matplotlib.pyplot as plt

    names = ("physics_native", "physics_budgeted", "student_expert", "hybrid_fixed_budget")
    labels = ("Original physics: unchanged", "Physics: comparison budget only",
              "Expert student: research candidates", "Fixed-budget hybrid: research candidates")
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    for axis, name, label in zip(axes.ravel(), names, labels):
        axis.imshow(result.physics.normalized_image, cmap="gray", vmin=0, vmax=1)
        prediction = result.candidates[name]
        if prediction.points_xy is not None:
            if len(prediction.points_xy):
                axis.scatter(*prediction.points_xy.T, s=35, facecolors="none", edgecolors="#f2bd50", linewidths=0.8)
            quantity = f"{len(prediction.points_xy)} points"
        else:
            mask = prediction.mask
            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = (0.98, 0.70, 0.15, 0.48)
            axis.imshow(overlay)
            quantity = f"{int(mask.sum())} pixels ({mask.mean():.1%} area)"
        suffix = "native support" if prediction.budget is None else f"budget <= {prediction.budget}"
        axis.set_title(f"{label}\n{quantity}; {suffix}", fontsize=11)
        axis.set_axis_off()
    figure.suptitle(f"RSI inference | primary: {result.primary_branch}\n"
                    "Candidates are not confirmed defects or chemical identities; physics evidence stays separate",
                    fontsize=12)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=150)
    plt.close(figure)
