"""Auditable training ablations; frozen v1 learner and expert labels stay intact.

The area penalty is an explicit experimental occupancy prior, NOT a negative
expert label. Weak bands are positive bags rather than exact boundary masks.
Neither this loss nor neural embeddings are calibrated physical evidence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math

import numpy as np

from .expert_student import expert_adaptation_loss, geometry_logits, torch, F


@dataclass(frozen=True)
class RefinementPolicy:
    name: str
    source_power: float = 0.0
    positive_crop_probability: float = 0.0
    photometric: bool = False
    weak_band_fraction: float = 1.0
    occupancy_weight: float = 0.0
    point_occupancy: float = 0.03
    area_occupancy: float = 0.15

    def __post_init__(self):
        if not self.name or not 0 <= self.source_power <= 1:
            raise ValueError("Invalid source weighting")
        if not 0 <= self.positive_crop_probability <= 1:
            raise ValueError("Invalid crop probability")
        if not 0 < self.weak_band_fraction <= 1 or not 0 <= self.occupancy_weight <= 10:
            raise ValueError("Invalid weak supervision")
        if not 0 < self.point_occupancy < 1 or not 0 < self.area_occupancy < 1:
            raise ValueError("Invalid experimental occupancy prior")


POLICIES = {
    "control": RefinementPolicy("control"),
    "source_sqrt": RefinementPolicy("source_sqrt", source_power=0.5),
    "positive_crops": RefinementPolicy("positive_crops", source_power=0.5,
                                        positive_crop_probability=0.5),
    "photometric": RefinementPolicy("photometric", source_power=0.5,
                                     positive_crop_probability=0.5, photometric=True),
    "weak_bag": RefinementPolicy("weak_bag", source_power=0.5,
                                  positive_crop_probability=0.5, photometric=True,
                                  weak_band_fraction=0.25),
    "weak_roi": RefinementPolicy("weak_roi", source_power=0.5,
                                  positive_crop_probability=0.5, photometric=True,
                                  weak_band_fraction=0.25, occupancy_weight=2.0),
}


def source_probabilities(entries, geometry, power):
    counts = Counter(e["group"] for e in entries if e["geometry_index"] == geometry)
    if not counts or not 0 <= power <= 1:
        raise ValueError("No valid source for geometry")
    groups = sorted(counts)
    weights = np.asarray([counts[g] ** power for g in groups], dtype=float)
    return groups, weights / weights.sum()


def photometric_augmentation(image, rng):
    """Mild intensity/noise perturbation only; no elastic/scale/label changes."""
    gain, gamma = rng.uniform(0.9, 1.1, size=2)
    offset = rng.uniform(-0.03, 0.03)
    sigma = rng.uniform(0, 0.015)
    result = np.clip(image, 0, 1) ** gamma * gain + offset
    result += rng.normal(0, sigma, size=image.shape)
    return np.clip(result, 0, 1).astype(np.float32)


class RefinedSampler:
    """Training-only sampler, with checkpointable per-source/case accounting.

    Control mode preserves the original random draw sequence exactly.
    Photometry has its own step-indexed RNG and cannot change crop coordinates.
    """

    def __init__(self, entries, arrays, policy, *, excluded_groups=()):
        self.entries, self.arrays, self.policy = list(entries), arrays, policy
        if not entries or any(e["group"] in excluded_groups for e in entries):
            raise ValueError("Empty training set or held-out source in sampler")
        self.geometries = sorted({e["geometry_index"] for e in entries})
        self.sources = {g: source_probabilities(entries, g, policy.source_power) for g in self.geometries}
        self.cases = {(g, source): [e for e in entries if e["geometry_index"] == g and e["group"] == source]
                      for g in self.geometries for source in self.sources[g][0]}
        self.positive_yx = {e["case_number"]: np.argwhere(arrays[e["case_number"]]["human"] > 0)
                            for e in entries}
        self.audit = {"patches": 0, "positive_patches": 0, "guided_patches": 0,
                      "by_case": {}, "by_source": {}}

    def probabilities(self):
        return {str(g): {source: float(p) for source, p in zip(*self.sources[g])}
                for g in self.geometries}

    def sample(self, rng, *, batch_size, patch_size, device, augmentation_seed):
        if batch_size < 1 or patch_size < 8:
            raise ValueError("Invalid patch/batch size")
        result = {k: [] for k in ("image", "teacher", "human", "human_weight", "teacher_weight")}
        valids, geometries, available, bands, records = [], [], [], [], []
        aug_rng = np.random.default_rng(augmentation_seed)
        for _ in range(batch_size):
            geometry = self.geometries[int(rng.integers(len(self.geometries)))]
            groups, probabilities = self.sources[geometry]
            index = (int(rng.integers(len(groups))) if self.policy.source_power == 0 else
                     int(rng.choice(len(groups), p=probabilities)))
            source = groups[index]
            candidates = self.cases[(geometry, source)]
            entry = candidates[int(rng.integers(len(candidates)))]
            number = entry["case_number"]
            plane = self.arrays[number]
            h, w = plane["image"].shape
            y, x = int(rng.integers(max(1, h - patch_size + 1))), int(rng.integers(max(1, w - patch_size + 1)))
            if rng.random() < 0.4:
                edge = int(rng.integers(4))
                if edge < 2:
                    y = 0 if edge == 0 else max(0, h - patch_size)
                else:
                    x = 0 if edge == 2 else max(0, w - patch_size)
            guided = False
            if self.policy.positive_crop_probability > 0:
                guide = rng.random() < self.policy.positive_crop_probability
                points = self.positive_yx[number]
                if guide and len(points):
                    py, px = points[int(rng.integers(len(points)))]
                    # Random offset, not a fixed center shortcut.
                    y = int(np.clip(py - rng.integers(patch_size), 0, max(0, h - patch_size)))
                    x = int(np.clip(px - rng.integers(patch_size), 0, max(0, w - patch_size)))
                    guided = True
            rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
            valid = np.zeros((patch_size, patch_size), np.float32)
            valid[:min(patch_size, h-y), :min(patch_size, w-x)] = 1
            valid = np.rot90(valid, rotation)
            valids.append(valid[:, ::-1] if flip else valid)
            for key in result:
                value = plane[key][y:y+patch_size, x:x+patch_size]
                padding = ((0, patch_size-value.shape[0]), (0, patch_size-value.shape[1]))
                value = np.pad(value, padding, mode="reflect" if key == "image" else "constant")
                value = np.rot90(value, rotation)
                if flip:
                    value = value[:, ::-1]
                value = np.asarray(value, dtype=np.float32)
                if key == "image" and self.policy.photometric:
                    value = photometric_augmentation(value, aug_rng)
                result[key].append(value)
            positive = bool(result["human_weight"][-1].any())
            self.audit["patches"] += 1
            self.audit["positive_patches"] += int(positive)
            self.audit["guided_patches"] += int(guided)
            for key, label in (("by_case", str(number)), ("by_source", source)):
                self.audit[key][label] = self.audit[key].get(label, 0) + 1
            records.append({"case_number": number, "source": source, "yx": [y,x],
                            "rotation": rotation, "flip": flip, "guided": guided,
                            "has_human_supervision": positive})
            geometries.append(geometry)
            bands.append(entry.get("semantic_kind") == "interface_band_roi_not_centerline")
            available.append(float(entry["teacher_positive_pixels"] > 0 and entry["action"] != "reject_prediction"))
        batch = {k: torch.as_tensor(np.stack(v), device=device) for k,v in result.items()}
        batch["image"] = batch["image"][:,None]
        batch["geometry_index"] = torch.tensor(geometries, dtype=torch.long, device=device)
        batch["teacher_available"] = torch.tensor(available, dtype=torch.float32, device=device)
        batch["valid_pixels"] = torch.as_tensor(np.stack(valids), device=device)
        batch["weak_band"] = torch.tensor(bands, dtype=torch.bool, device=device)
        return batch, records


def refinement_loss(output, batch, policy):
    if policy.weak_band_fraction == 1 and policy.occupancy_weight == 0:
        return expert_adaptation_loss(output, batch, use_expert=True)
    _, parts = expert_adaptation_loss(output, batch, use_expert=True)
    logits = geometry_logits(output, batch["geometry_index"])
    values = []
    for i in range(len(logits)):
        known = batch["human_weight"][i] > 0
        if not bool(known.any()):
            continue
        z = logits[i][known]
        target = batch["human"][i][known].float()
        if bool(batch["weak_band"][i]) and bool((target == 1).all()):
            z = torch.topk(z, max(1, math.ceil(z.numel()*policy.weak_band_fraction))).values
            values.append(F.softplus(-z).mean())
        else:
            weight = batch["human_weight"][i][known]
            values.append((F.binary_cross_entropy_with_logits(z,target,reduction="none")*weight).sum()/weight.sum())
    human = torch.stack(values).mean() if values else logits.sum()*0
    valid = batch["valid_pixels"]
    occupancy = (torch.sigmoid(logits)*valid).sum((1,2))/valid.sum((1,2)).clamp(min=1)
    target = torch.where(batch["geometry_index"] == 1, policy.point_occupancy, policy.area_occupancy)
    prior = torch.relu(occupancy-target).square().mean()
    total = parts["teacher_positive"] + .25*parts["teacher_background_prior"] + 2*human + policy.occupancy_weight*prior
    return total, {**parts, "human": human, "occupancy_prior": prior,
                   "mean_occupancy": occupancy.mean(), "total": total}


def predict_with_embedding(model, image, geometry_index, *, device="cpu", tile_size=256, overlap=64):
    """Same pixel-preserving tiling as v1; additionally expose normalized features.

    Embedding is a tile-area-weighted encoder feature, not chemical identity,
    calibrated uncertainty or physical strain. No annotation is an input.
    """
    image = np.asarray(image, np.float32)
    if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).all():
        raise ValueError("Invalid normalized input image")
    if geometry_index not in (1,2,3,4) or tile_size < 8 or not 0 < overlap < tile_size:
        raise ValueError("Invalid geometry/tiling")
    h,w = image.shape
    score, weight = np.zeros((h,w),np.float32), np.zeros((h,w),np.float32)
    embeddings, areas = [], []
    def starts(length):
        return sorted(set(range(0,max(1,length-tile_size+1),tile_size-overlap))|{max(0,length-tile_size)})
    model.eval()
    with torch.no_grad():
        for y in starts(h):
            for x in starts(w):
                tile = image[y:y+tile_size,x:x+tile_size]
                th,tw = tile.shape
                tile = np.pad(tile,((0,max(0,8-th)),(0,max(0,8-tw))),mode="reflect")
                tensor = torch.as_tensor(np.ascontiguousarray(tile)[None,None],device=device)
                out = model(tensor)
                z = out.geometry_logits[:,geometry_index]-out.geometry_logits[:,0]
                value = torch.sigmoid(z)[0,:th,:tw].cpu().numpy()
                window = np.maximum(np.outer(np.hanning(th),np.hanning(tw)),.05).astype(np.float32)
                score[y:y+th,x:x+tw] += value*window
                weight[y:y+th,x:x+tw] += window
                embeddings.append(out.embedding[0].cpu().numpy())
                areas.append(th*tw)
    feature = np.average(np.stack(embeddings),axis=0,weights=areas).astype(np.float32)
    feature /= max(float(np.linalg.norm(feature)),1e-12)
    return score/weight, feature


def policy_dict(name):
    return asdict(POLICIES[name])
