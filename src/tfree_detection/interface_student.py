"""Training-only physical ranking within weak expert interface regions.

The expert ROI is never overwritten. Outside pixels remain human-unknown;
only explicitly qualified low-variation reference patches receive a weak
physical prior. Neither references nor geometric proxies establish absence
of defects. Selection/quality guards consume no held-out annotations.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from .expert_student import F, geometry_logits, torch
from .rsi_refinement import RefinedSampler


def physical_training_targets(plane, physical):
    human = np.asarray(plane["human"], bool)
    valid = np.asarray(physical["valid"], bool)
    variation = np.asarray(physical["variation"], np.float32)
    if human.shape != valid.shape or variation.shape != human.shape:
        raise ValueError("Physical target shape mismatch")
    if not np.isfinite(variation).all():
        raise ValueError("Nonfinite physical variation")
    positive = human & valid
    high, low, reference = (np.zeros_like(human) for _ in range(3))
    if positive.any():
        lo, hi = np.quantile(variation[positive], [.2, .6])
        # Tie/flat evidence cannot invent a within-band ranking.
        high = positive & (variation >= hi)
        if hi > lo + 1e-8:
            low = positive & (variation <= lo)
        unknown_far = valid & ~human & (distance_transform_edt(~human) >= 12)
        if unknown_far.any():
            threshold = np.quantile(variation[unknown_far], .25)
            reference = unknown_far & (variation <= threshold)
    return {"physical_positive": high.astype(np.float32),
            "physical_low_in_roi": low.astype(np.float32),
            "physical_reference_prior": reference.astype(np.float32)}


class InterfaceSampler(RefinedSampler):
    """Same crop/augmentation schedule for control and physical-rank ablation."""

    def __init__(self, entries, arrays, policy, physical, *, excluded_groups=()):
        super().__init__(entries, arrays, policy, excluded_groups=excluded_groups)
        if any(e["geometry"] != "interface" for e in entries):
            raise ValueError("Interface specialist cannot update point/region models")
        self.targets = {e["case_number"]: physical_training_targets(
            arrays[e["case_number"]], physical[e["case_number"]]) for e in entries}

    def sample(self, rng, *, batch_size, patch_size, device, augmentation_seed):
        batch, records = super().sample(rng, batch_size=batch_size, patch_size=patch_size,
                                        device=device, augmentation_seed=augmentation_seed)
        keys = ("physical_positive", "physical_low_in_roi", "physical_reference_prior")
        transformed = {k: [] for k in keys}
        for record in records:
            y, x = record["yx"]
            for key in keys:
                value = self.targets[record["case_number"]][key][y:y+patch_size, x:x+patch_size]
                value = np.pad(value, ((0, patch_size-value.shape[0]), (0, patch_size-value.shape[1])))
                value = np.rot90(value, record["rotation"])
                if record["flip"]:
                    value = value[:, ::-1]
                transformed[key].append(value)
        for key, values in transformed.items():
            batch[key] = torch.as_tensor(np.stack(values), device=device)
        return batch, records


def _masked_mean(value, mask):
    count = mask.sum((1, 2))
    per = (value * mask).sum((1, 2)) / count.clamp(min=1)
    return (per * (count > 0)).sum() / (count > 0).sum().clamp(min=1)


def physical_ranking_loss(output, batch, scaled_output=None):
    z = geometry_logits(output, batch["geometry_index"])
    valid = batch["valid_pixels"]
    high = batch["physical_positive"] * valid
    low = batch["physical_low_in_roi"] * valid
    ref = batch["physical_reference_prior"] * valid
    # Human labels outside the original ROI are neither created nor consulted.
    positive = _masked_mean(F.softplus(-z), high)
    reference_prior = _masked_mean(F.softplus(z), ref)
    ranks, inside = [], []
    for i in range(len(z)):
        if bool(high[i].any()):
            zh = z[i][high[i] > 0].mean()
            if bool(ref[i].any()):
                ranks.append(F.softplus(1 + z[i][ref[i] > 0].mean() - zh))
            if bool(low[i].any()):
                inside.append(F.softplus(.25 + z[i][low[i] > 0].mean() - zh))
    zero = z.sum() * 0
    rank = torch.stack(ranks).mean() if ranks else zero
    in_roi = torch.stack(inside).mean() if inside else zero
    p = torch.sigmoid(z)
    occupancy = (p*valid).sum((1, 2)) / valid.sum((1, 2)).clamp(min=1)
    occupancy_prior = torch.relu(occupancy-.15).square().mean()
    # Local continuity only within the weak ROI, not across the entire image.
    roi = batch["human"] * valid
    continuity = ((p[:, 1:]-p[:, :-1]).square()*roi[:, 1:]*roi[:, :-1]).mean()
    continuity += ((p[:, :, 1:]-p[:, :, :-1]).square()*roi[:, :, 1:]*roi[:, :, :-1]).mean()
    scale = zero
    if scaled_output is not None:
        coarse = torch.sigmoid(geometry_logits(scaled_output, batch["geometry_index"]))
        coarse = F.interpolate(coarse[:, None], size=z.shape[-2:], mode="bilinear",
                               align_corners=False)[:, 0]
        scale = _masked_mean((p-coarse).square(), valid)
    total = (2*positive + rank + .25*in_roi + .5*reference_prior +
             4*occupancy_prior + .2*continuity + .2*scale)
    return total, {"positive": positive, "relative_ranking": rank,
                   "within_roi_ranking": in_roi, "qualified_reference_prior": reference_prior,
                   "occupancy_prior": occupancy_prior, "continuity": continuity,
                   "scale_consistency": scale, "total": total}


def score_health(score):
    score = np.asarray(score)
    if score.ndim != 2 or not score.size or not np.isfinite(score).all():
        return {"eligible": False, "reason": "invalid_score"}
    occupied = float((score > .5).mean())
    dynamic = float(np.quantile(score, .95)-np.quantile(score, .05))
    eligible = occupied < .98 and dynamic >= .01
    return {"eligible": eligible, "reason": "passed" if eligible else "all_high_or_flat",
            "fraction_above_half": occupied, "p95_minus_p05": dynamic,
            "mean": float(score.mean()), "calibrated": False}
