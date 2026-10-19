"""Expert adaptation of the existing student, without inventing dense negatives.

Only the geometry heads are supervised in this round. Untrained auxiliary heads
are zeroed/frozen and never exposed as evidence, reliability or OOD estimates.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

from .expert_supervision import contained_path, normalize_image, sha256_file, write_new_json
from .student import EvidencePreservingStudent, torch, F


CHECKPOINT_FORMAT = "phyevo-expert-student-checkpoint-v1"


def make_expert_student(base_channels: int = 8):
    if torch is None:
        raise ImportError("Expert learning requires the existing optional torch dependency")
    model = EvidencePreservingStudent(evidence_channels=1, base_channels=base_channels)
    for head in (model.violation_head, model.evidence_head, model.reliability_ood_head):
        for parameter in head.parameters():
            torch.nn.init.zeros_(parameter)
            parameter.requires_grad_(False)
    return model


def geometry_logits(output, geometry_index):
    indices = geometry_index[:, None, None, None].expand(-1, 1, *output.geometry_logits.shape[-2:])
    return torch.gather(output.geometry_logits, 1, indices)[:, 0] - output.geometry_logits[:, 0]


def expert_adaptation_loss(output, batch, *, use_expert: bool = True):
    """Human loss is independent of teacher validity; unlabeled != human-negative."""
    logits = geometry_logits(output, batch["geometry_index"])
    teacher = batch["teacher"].float()
    # In the ablation, no expert masks or their exclusion halo are consulted.
    teacher_weight = (batch["teacher_weight"] if use_expert else
                      batch["teacher_available"][:, None, None].expand_as(teacher))
    teacher_weight = teacher_weight * batch.get("valid_pixels", torch.ones_like(teacher))
    values = F.binary_cross_entropy_with_logits(logits, teacher, reduction="none")

    def mean(values, weight):
        axes = tuple(range(1, values.ndim))
        count = weight.sum(dim=axes)
        per_image = (values * weight).sum(dim=axes) / count.clamp(min=1)
        available = (count > 0).to(values.dtype)
        # A wide interface ROI must not drown out a few explicitly marked atoms.
        return (per_image * available).sum() / available.sum().clamp(min=1)

    teacher_positive = mean(values, teacher_weight * teacher)
    teacher_negative = mean(values, teacher_weight * (1 - teacher))
    teacher_loss = teacher_positive + 0.25 * teacher_negative
    human_values = F.binary_cross_entropy_with_logits(logits, batch["human"].float(), reduction="none")
    human = mean(human_values, batch["human_weight"]) if use_expert else logits.sum() * 0
    total = teacher_loss + 2.0 * human
    return total, {"teacher_positive": teacher_positive, "teacher_background_prior": teacher_negative,
                   "human": human, "total": total}


def sample_batch(entries, arrays, rng, *, batch_size: int, patch_size: int, device: str):
    """Identical, annotation-independent patch schedule for both ablations."""
    if patch_size < 8 or batch_size < 1:
        raise ValueError("Invalid patch/batch size")
    geometries_available = sorted({entry["geometry_index"] for entry in entries})
    groups = sorted({entry["group"] for entry in entries})
    group_entries = {g: [e for e in entries if e["group"] == g] for g in groups}
    geometry_groups = {geometry: sorted({e["group"] for e in entries if e["geometry_index"] == geometry})
                       for geometry in geometries_available}
    result = {key: [] for key in ("image", "teacher", "human", "human_weight", "teacher_weight")}
    valid_pixels = []
    geometries, availability = [], []
    for _ in range(batch_size):
        geometry = geometries_available[int(rng.integers(len(geometries_available)))]
        eligible_groups = geometry_groups[geometry]
        group = eligible_groups[int(rng.integers(len(eligible_groups)))]
        candidates = [e for e in group_entries[group] if e["geometry_index"] == geometry]
        entry = candidates[int(rng.integers(len(candidates)))]
        plane = arrays[entry["case_number"]]
        h, w = plane["image"].shape
        y = int(rng.integers(max(1, h - patch_size + 1)))
        x = int(rng.integers(max(1, w - patch_size + 1)))
        # Border coverage must not depend on knowing where held-out defects are.
        if rng.random() < 0.4:
            edge = int(rng.integers(4))
            if edge < 2:
                y = 0 if edge == 0 else max(0, h - patch_size)
            else:
                x = 0 if edge == 2 else max(0, w - patch_size)
        rotation = int(rng.integers(4))
        flip = bool(rng.integers(2))
        valid = np.zeros((patch_size, patch_size), np.float32)
        valid[:min(patch_size, h - y), :min(patch_size, w - x)] = 1
        valid = np.rot90(valid, rotation)
        valid_pixels.append(valid[:, ::-1] if flip else valid)
        for key in result:
            values = plane[key][y:y + patch_size, x:x + patch_size]
            padding = ((0, patch_size - values.shape[0]), (0, patch_size - values.shape[1]))
            # Padded pixels never acquire human/teacher supervision.
            values = np.pad(values, padding, mode="reflect" if key == "image" else "constant")
            values = np.rot90(values, rotation)
            if flip:
                values = values[:, ::-1]
            result[key].append(np.asarray(values, dtype=np.float32))
        geometries.append(entry["geometry_index"])
        availability.append(float(entry["teacher_positive_pixels"] > 0 and entry["action"] != "reject_prediction"))
    batch = {key: torch.as_tensor(np.stack(values), device=device) for key, values in result.items()}
    batch["image"] = batch["image"][:, None]
    batch["geometry_index"] = torch.tensor(geometries, dtype=torch.long, device=device)
    batch["teacher_available"] = torch.tensor(availability, dtype=torch.float32, device=device)
    batch["valid_pixels"] = torch.as_tensor(np.stack(valid_pixels), device=device)
    return batch


def save_checkpoint(directory: Path, model, optimizer, rng, metadata: dict) -> None:
    """Pickle-free weights AND optimizer/RNG state; every checkpoint is new."""
    if directory.exists():
        raise FileExistsError("Refusing to overwrite a checkpoint")
    directory.mkdir(parents=True)
    values = {"model:" + name: tensor.detach().cpu().numpy() for name, tensor in model.state_dict().items()}
    values["rng:torch"] = torch.get_rng_state().numpy()
    optimizer_state = optimizer.state_dict()
    scalar_state = {}
    for key, state in optimizer_state["state"].items():
        scalar_state[str(key)] = {}
        for name, value in state.items():
            if torch.is_tensor(value):
                values[f"optimizer:{key}:{name}"] = value.cpu().numpy()
            else:
                scalar_state[str(key)][name] = value
    if torch.cuda.is_available():
        for index, state in enumerate(torch.cuda.get_rng_state_all()):
            values[f"rng:cuda:{index}"] = state.cpu().numpy()
    with (directory / "state.npz").open("xb") as stream:
        np.savez_compressed(stream, **values)
    write_new_json(directory / "checkpoint.json", {
        **metadata, "format": CHECKPOINT_FORMAT, "state_sha256": sha256_file(directory / "state.npz"),
        "numpy_rng_state": rng.bit_generator.state, "optimizer_scalar_state": scalar_state,
        "optimizer_param_groups": optimizer_state["param_groups"],
        "supervised_heads": ["geometry"], "auxiliary_heads_validated": False,
        "calibrated": False, "production_release_eligible": False,
    })


def restore_checkpoint(directory: Path, *, device: str = "cpu", resume: bool = False):
    directory = directory.resolve()
    meta = json.loads((directory / "checkpoint.json").read_text())
    if meta.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Unsupported expert checkpoint")
    if sha256_file(directory / "state.npz") != meta["state_sha256"]:
        raise ValueError("Expert checkpoint checksum mismatch")
    model = make_expert_student(int(meta["base_channels"])).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=float(meta["learning_rate"]))
    rng = np.random.default_rng()
    with np.load(directory / "state.npz", allow_pickle=False) as archive:
        state = {key.removeprefix("model:"): torch.from_numpy(archive[key].copy())
                 for key in archive.files if key.startswith("model:")}
        model.load_state_dict(state)
        if resume:
            optimizer_state = {int(key): dict(value) for key, value in meta["optimizer_scalar_state"].items()}
            for key in archive.files:
                if key.startswith("optimizer:"):
                    _, index, name = key.split(":", 2)
                    optimizer_state[int(index)][name] = torch.from_numpy(archive[key].copy())
            optimizer.load_state_dict({"state": optimizer_state, "param_groups": meta["optimizer_param_groups"]})
            torch.set_rng_state(torch.from_numpy(archive["rng:torch"].copy()))
            if device.startswith("cuda"):
                states = [torch.from_numpy(archive[f"rng:cuda:{i}"].copy()) for i in range(torch.cuda.device_count())]
                torch.cuda.set_rng_state_all(states)
            rng.bit_generator.state = meta["numpy_rng_state"]
    return model, optimizer, rng, meta


def predict_scores(model, image: np.ndarray, geometry_index: int, *,
                   device: str = "cpu", tile_size: int = 256, overlap: int = 64,
                   normalized: bool = False) -> np.ndarray:
    """Source-pixel overlap/add inference; no resizing/coordinate remapping."""
    if not 0 < overlap < tile_size or tile_size < 8:
        raise ValueError("Require 0 < overlap < tile size")
    if geometry_index not in {1, 2, 3, 4}:
        raise ValueError("Invalid geometry index")
    image = np.asarray(image, dtype=np.float32) if normalized else normalize_image(image)
    if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
        raise ValueError("Expected finite 2-D image")
    h, w = image.shape
    score, weights = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)

    def starts(length):
        return sorted(set(range(0, max(1, length - tile_size + 1), tile_size - overlap)) |
                      {max(0, length - tile_size)})

    model.eval()
    with torch.no_grad():
        for y in starts(h):
            for x in starts(w):
                tile = image[y:y + tile_size, x:x + tile_size]
                th, tw = tile.shape
                pad = ((0, max(0, 8 - th)), (0, max(0, 8 - tw)))
                tile = np.pad(tile, pad, mode="reflect")
                tensor = torch.as_tensor(np.ascontiguousarray(tile)[None, None], device=device)
                output = model(tensor)
                logits = output.geometry_logits[:, geometry_index] - output.geometry_logits[:, 0]
                values = torch.sigmoid(logits)[0, :th, :tw].cpu().numpy()
                window = np.maximum(np.outer(np.hanning(th), np.hanning(tw)), 0.05).astype(np.float32)
                score[y:y + th, x:x + tw] += values * window
                weights[y:y + th, x:x + tw] += window
    return score / weights


def point_candidates(score: np.ndarray, budget: int, *, threshold: float = 0.0) -> np.ndarray:
    if budget < 1:
        return np.empty((0, 2), dtype=float)
    yx = peak_local_max(score, min_distance=4, threshold_abs=threshold,
                        exclude_border=False, num_peaks=budget)
    return yx[:, ::-1].astype(float)


def matched_points(predicted_xy: np.ndarray, expert_xy: np.ndarray, tolerance: float = 6.0) -> int:
    predicted_xy, expert_xy = np.asarray(predicted_xy), np.asarray(expert_xy)
    if not len(predicted_xy) or not len(expert_xy):
        return 0
    distance = cdist(predicted_xy, expert_xy)
    # A valid match always takes precedence over improving subpixel distance.
    cost = np.where(distance <= tolerance, distance / max(tolerance, 1e-8), 1e6)
    rr, cc = linear_sum_assignment(cost)
    return int((distance[rr, cc] <= tolerance).sum())


def evaluate_case(entry: dict, plane: dict, score: np.ndarray) -> dict:
    human, teacher = plane["human"].astype(bool), plane["teacher"].astype(bool)
    result = {"case_number": entry["case_number"], "geometry": entry["geometry"], "group": entry["group"],
              "action": entry["action"], "semantic_kind": entry["semantic_kind"],
              "score_mean": float(score.mean()), "fraction_above_0_5": float((score >= 0.5).mean()),
              "teacher_area_fraction": float(teacher.mean()), "precision": None,
              "negative_label_completeness_verified": False}
    if entry["geometry"] == "point" and entry["action"] == "correct_geometry":
        points = np.asarray(entry["points_xy"])
        xy = np.rint(points).astype(int)
        result["annotated_points"] = len(points)
        result["teacher_marked_point_coverage"] = float(teacher[xy[:, 1], xy[:, 0]].mean())
        result["point_recall_at_budget"] = {}
        yy, xx = np.indices(score.shape)
        border_distance = np.minimum.reduce([xx, yy, score.shape[1] - 1 - xx, score.shape[0] - 1 - yy])
        random = np.random.default_rng(1000 + entry["case_number"])
        null_scores = {
            "uniform_spatial_null": gaussian_filter(random.random(score.shape), 1),
            "border_only_null": np.exp(-border_distance / 24.0) + random.random(score.shape) * 0.05,
        }
        result["null_controls"] = {name: {} for name in null_scores}
        for density in (20, 50, 100):
            budget = max(1, round(density * score.size / (512 * 512)))
            candidates = point_candidates(score, budget)
            matches = matched_points(candidates, points)
            result["point_recall_at_budget"][str(density)] = {
                "budget_per_512_square": density, "candidate_count": len(candidates),
                "matches": matches, "recall_of_marked_points": matches / len(points)}
            for name, null_score in null_scores.items():
                null_candidates = point_candidates(null_score, budget)
                result["null_controls"][name][str(density)] = {
                    "matches": matched_points(null_candidates, points),
                    "candidate_count": len(null_candidates), "annotated_points": len(points),
                }
    if human.any():
        # Threshold ties are resolved deterministically, never using human labels.
        budget = max(1, math.ceil(0.1 * score.size))
        order = np.argsort(-score.ravel(), kind="stable")[:budget]
        selected = np.zeros(score.shape, bool)
        selected.ravel()[order] = True
        result["human_positive_coverage_at_10pct_area"] = float((selected & human).sum() / human.sum())
        result["teacher_human_positive_coverage"] = float((teacher & human).sum() / human.sum())
        result["budget_area_fraction"] = float(selected.mean())
    return result
