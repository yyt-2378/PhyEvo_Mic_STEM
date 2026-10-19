"""Evidence-driven spatial-program critic, learner, and fixed promotion gate.

This is bounded algorithm synthesis, not an LLM or unrestricted self-rewriting
agent. Only development tensors reach the critic. Audit labels never enter its
API. The immutable governor is deliberately outside the mutable program grammar.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F

from .autonomous_program import CHANNELS, evaluate, identity, make_pool, normalize, top_mask, validate


@dataclass
class Case:
    number: int
    group: str
    fields: dict
    base: torch.Tensor
    allowed: torch.Tensor
    human: torch.Tensor
    gates: torch.Tensor
    small_human: torch.Tensor
    feature_cache: dict


def prepare_case(number, group, image, physical, student, base_score, allowed, human,
                 *, physical_route, band, device="cpu", resolution=384):
    def tensor(v):
        return torch.as_tensor(np.asarray(v).copy(), dtype=torch.float32, device=device)
    base = tensor(base_score)
    height, width = base.shape
    ratio = min(1., resolution / max(height, width))
    shape = (max(4, round(height * ratio)), max(4, round(width * ratio)))

    def small(v):
        return F.interpolate(tensor(v)[None, None], size=shape, mode="area")[0, 0]
    raw = {"base": base_score, "physics": physical["score"], "student": student,
           "variation": physical["variation"], "image": image, "valid": physical["valid"].astype(np.float32)}
    fields = {name: normalize(small(value)) for name, value in raw.items()}
    # Preserve every original ranking/tie at V0. Float32 normalization or
    # clipping can merge near-equal values at the fixed-budget cutoff.
    precise = base.double()
    precise = (precise - precise.mean()) / precise.std(correction=0).clamp(min=1e-8)
    fields["base"] = F.interpolate(precise[None, None], size=shape, mode="area")[0, 0].float()
    return Case(number, group, fields, precise, tensor(allowed).bool(), tensor(human).bool(),
                tensor([1., float(physical_route), float(band)]), small(human.astype(np.float32)), {})


def feature(case, program, *, full=False):
    key = identity(program)
    if key not in case.feature_cache:
        with torch.no_grad():
            value = normalize(evaluate(program, case.fields))
        if not torch.isfinite(value).all():
            raise ValueError("Non-finite synthesized feature")
        case.feature_cache[key] = value
    value = case.feature_cache[key]
    if full:
        return F.interpolate(value[None, None], size=case.base.shape, mode="bilinear", align_corners=False)[0, 0]
    return value


def scores(case, stages, *, full=False):
    value = case.base if full else case.fields["base"]
    for stage in stages:
        weights = torch.as_tensor(stage["weights"], device=value.device, dtype=case.gates.dtype)
        if weights.shape != (len(stage["programs"]), 3) or not torch.isfinite(weights).all():
            raise ValueError("Invalid synthesized readout parameters")
        for row, program in zip(weights, stage["programs"], strict=True):
            value = value + torch.dot(row, case.gates) * feature(case, program, full=full)
    return value


def case_coverage(case, stages):
    score = scores(case, stages, full=True)
    if stages and (not torch.isfinite(score).all() or float(score.std(correction=0)) < 1e-5):
        raise ValueError("Nonfinite or flat candidate ranking")
    mask = top_mask(score, case.allowed)
    value = int((mask & case.human).sum()) / max(1, int(case.human.sum()))
    return value, mask


def diagnose(cases, parents):
    """Measured failure evidence, not prewritten ORI or ZrO2 diagnoses."""
    rows = []
    for case, parent in zip(cases, parents, strict=True):
        value = scores(case, parent)
        allowed = F.interpolate(case.allowed.float()[None, None], size=value.shape, mode="nearest")[0, 0].bool()
        predicted = top_mask(value, allowed)
        total = case.small_human.sum().clamp(min=1)
        coverage = float((predicted * case.small_human).sum() / total)
        mass_outside = float((case.small_human * ~allowed).sum() / total)
        rows.append({"case_number": case.number, "development_group": case.group,
                     "coarse_positive_coverage": coverage, "positive_mass_outside_fixed_support": mass_outside,
                     "selected_fraction": float(predicted.float().mean())})
    groups = sorted({r["development_group"] for r in rows})
    by_group = {g: float(np.mean([r["coarse_positive_coverage"] for r in rows if r["development_group"] == g])) for g in groups}
    return {"kind": "unselected_known_positive_mass_under_fixed_area_cap", "rows": rows,
            "source_macro_coverage": float(np.mean(list(by_group.values()))), "groups": by_group,
            "unknown_pixels_are_negative_labels": False,
            "falsifier": "No source-macro validation gain, any source-mean regression, or seed regression beyond frozen tolerance",
            "scope": "development replay; not a new blinded test"}


def propose(cases, parents, history, rng, *, arm, count=6, pool_size=32):
    if arm not in {"feedback", "open_loop", "parameter_only"}:
        raise ValueError("Unknown experiment arm")
    evidence = diagnose(cases, parents)
    past = [p for item in history if item["accepted"] for p in item["programs"]] if arm == "feedback" else []
    seen = [identity(p) for item in history for p in item["programs"]]
    pool = make_pool(rng, past, size=pool_size, seen=seen)
    associations = []
    # Equal-source association to unexplained positive mass. Used ONLY for the
    # feedback arm; both controls receive the same pool construction budget.
    for node in pool:
        by_group = {}
        for case, parent in zip(cases, parents, strict=True):
            value = scores(case, parent)
            threshold = torch.quantile(value.flatten(), .9)
            missed = case.small_human * torch.sigmoid((threshold - value) / .5)
            f = feature(case, node)
            association = float((f * missed).sum() / missed.sum().clamp(min=1) - f.mean())
            by_group.setdefault(case.group, []).append(association)
        association = float(np.mean([np.mean(v) for v in by_group.values()]))
        associations.append({"program": node, "residual_association": association,
                             "priority": abs(association) / math.sqrt(validate(node))})
    if arm == "feedback":
        selected = sorted(associations, key=lambda r: (-r["priority"], identity(r["program"])))[:count]
    else:
        selected = [associations[int(i)] for i in rng.permutation(len(pool))[:count]]
    candidates = []
    for record in selected:
        programs = ([{"op": "field", "name": n} for n in ("physics", "student", "variation", "image")]
                    if arm == "parameter_only" else [record["program"]])
        candidates.append({"programs": programs,
            "learning_rate": float(np.exp(rng.uniform(np.log(.005), np.log(.05)))),
            "retention": float(np.exp(rng.uniform(np.log(.01), np.log(.3)))),
            "temperature": float(np.exp(rng.uniform(np.log(.3), np.log(1.5)))),
            "hard_positive_weight": float(1 + 3 * (1 - evidence["source_macro_coverage"])) if arm == "feedback" else 1.,
            "hypothesis": {"mechanism": ("fixed-basis readout improved by training hyperparameters" if arm == "parameter_only"
                                           else "spatial feature enriches currently missed expert-positive mass"),
                           "measured_development_association": record["residual_association"],
                           "association_used_for_selection": arm == "feedback",
                           "expected_effect": "positive coverage increases without more candidate area",
                           "falsifier": evidence["falsifier"]},
            "uses_parent_feedback": arm == "feedback", "unknown_as_negative_labels": False})
    return candidates, {"evidence": evidence, "pool": associations, "arm": arm,
                        "prior_trials": len(history), "generated_pool_size": len(pool)}


def fit_candidate(training, parents, candidate, *, seed, steps=120):
    if not training or len(training) != len(parents):
        raise ValueError("Training inputs missing")
    device = training[0].base.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    programs = candidate["programs"]
    weights = torch.nn.Parameter(torch.zeros((len(programs), 3), device=device))
    optimizer = torch.optim.Adam([weights], lr=candidate["learning_rate"])
    cached = []
    for case, parent in zip(training, parents, strict=True):
        field = torch.stack([feature(case, node).flatten() for node in programs], dim=1)
        old = scores(case, parent).detach().flatten()
        positive = case.small_human.flatten()
        indices = torch.nonzero(positive > .05, as_tuple=True)[0]
        if not indices.numel():
            raise ValueError("No expert-positive training support")
        density = positive[indices]
        missed = torch.sigmoid((torch.quantile(old, .9) - old[indices]) / .5)
        probabilities = (density * (1 + candidate["hard_positive_weight"] * missed)).cpu().numpy()
        probabilities /= probabilities.sum()
        cached.append((case, field, old, indices, probabilities))
    groups = sorted({c.group for c in training})
    events = []
    for step in range(steps):
        # One case per source per step: source count is not pixel/frame count.
        losses = []
        for group in groups:
            options = [c for c in cached if c[0].group == group]
            case, field, old, positive, probabilities = options[int(rng.integers(len(options)))]
            uniform = torch.as_tensor(rng.integers(0, len(old), 4096), device=device)
            positions = rng.choice(len(positive), 512, replace=True, p=probabilities)
            pos = positive[torch.as_tensor(positions, device=device)]
            coefficients = weights @ case.gates
            delta_uniform = field[uniform] @ coefficients
            z_all = (old[uniform] + delta_uniform) / candidate["temperature"]
            z_pos = (old[pos] + field[pos] @ coefficients) / candidate["temperature"]
            # Positive-versus-population density objective. Population pixels
            # remain UNLABELED; there is no BCE target=0 for expert-unknowns.
            density_loss = torch.logsumexp(z_all, 0) - math.log(len(uniform)) - z_pos.mean()
            retain = delta_uniform.square().mean()
            losses.append(density_loss + candidate["retention"] * retain)
        loss = torch.stack(losses).mean() + .001 * weights.square().sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("Candidate training became nonfinite")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([weights], 5.)
        optimizer.step()
        with torch.no_grad():
            weights.clamp_(-2., 2.)
        if step in {0, steps - 1}:
            events.append({"step": step + 1, "objective": float(loss.detach())})
    return {"programs": programs, "weights": weights.detach().cpu().tolist(),
            "candidate": candidate, "training_seed": seed, "steps": steps,
            "train_groups": groups, "train_cases": sorted(c.number for c in training),
            "events": events, "unknown_as_negative_labels": False}


def summarize(rows):
    groups = sorted({r["group"] for r in rows})
    return {g: {"mean": float(np.mean([r["coverage"] for r in rows if r["group"] == g])),
                "seeds": {str(seed): float(np.mean([r["coverage"] for r in rows if r["group"] == g and r["seed"] == seed]))
                          for seed in sorted({r["seed"] for r in rows if r["group"] == g})}}
            for g in groups}


def promotion(candidate_rows, parent_rows, baseline_rows, *, minimum_gain=.002,
              source_regression=1e-8, seed_regression=.01):
    def keys(rows):
        return [(r["group"], r["case_number"], r["seed"]) for r in rows]
    expected = keys(baseline_rows)
    if len(expected) != len(set(expected)) or set(keys(parent_rows)) != set(expected) or set(keys(candidate_rows)) != set(expected):
        raise ValueError("Promotion requires exactly matched, nonduplicate evaluation rows")
    if len(keys(parent_rows)) != len(expected) or len(keys(candidate_rows)) != len(expected):
        raise ValueError("Duplicate candidate or parent rows")
    if not all(np.isfinite(r["coverage"]) and 0 <= r["coverage"] <= 1 and r["selected_pixels"] <= r["budget"]
               for rows in (candidate_rows, parent_rows, baseline_rows) for r in rows):
        return {"accepted": False, "reason": "invalid_metric_or_area_budget", "macro_gain": None}
    new, old, base = map(summarize, (candidate_rows, parent_rows, baseline_rows))
    gain = float(np.mean([new[g]["mean"] - old[g]["mean"] for g in new]))
    regressions = [g for g in new if new[g]["mean"] < max(old[g]["mean"], base[g]["mean"]) - source_regression]
    seed_failures = [(g, s) for g in new for s in new[g]["seeds"]
                     if new[g]["seeds"][s] < max(old[g]["seeds"][s], base[g]["seeds"][s]) - seed_regression]
    accepted = gain >= minimum_gain and not regressions and not seed_failures
    return {"accepted": accepted, "reason": "passed" if accepted else "gain_or_nonregression_failed",
            "macro_gain": gain, "source_regressions": regressions, "seed_regressions": seed_failures,
            "candidate": new, "parent": old, "baseline": base,
            "minimum_gain": minimum_gain, "source_regression_tolerance": source_regression,
            "seed_regression_tolerance": seed_regression, "statistical_significance_claimed": False}
