"""Frozen label-side evaluator; never imported by a candidate process."""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max


def evaluate(sample, prediction, protocol):
    image = sample["image"]
    score = np.asarray(prediction["score"])
    support = np.asarray(prediction["support"])
    if score.shape != image.shape or support.shape != image.shape or support.dtype != bool:
        raise ValueError("Shape/support contract violated")
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite scores")
    # 'Valid' means physically analyzable, not 'the old teacher detected it'.
    valid = np.asarray(sample["valid"], bool)
    if np.any(support & ~valid):
        raise ValueError("Prediction includes physically invalid pixels")
    if sample["geometry"] == "point":
        budget = max(1, round(protocol.point_budget_per_512 * image.size / 512**2))
        eligible = np.where(support, score, -np.inf)
        yx = peak_local_max(eligible, min_distance=4, exclude_border=False,
                            num_peaks=budget, threshold_abs=0)
        xy = yx[:, ::-1].astype(float)
        truth = np.asarray(sample["points_xy"], float).reshape(-1, 2)
        matched = 0
        if len(xy) and len(truth):
            distances = cdist(truth, xy)
            # Maximize the number of in-tolerance matches first, then distance.
            cost = np.where(distances <= protocol.point_tolerance_px,
                            distances / max(protocol.point_tolerance_px, 1),
                            (min(len(truth), len(xy)) + 1) * 2)
            rows, cols = linear_sum_assignment(cost)
            matched = int((distances[rows, cols] <= protocol.point_tolerance_px).sum())
        denominator, selected = len(truth), len(xy)
        metric = "known_point_coverage_at_fixed_count"
    elif sample["geometry"] in {"interface", "region", "line"}:
        budget = max(1, math.ceil(protocol.area_fraction * image.size))
        indices = np.flatnonzero(support & (score > 0))
        order = np.argsort(-score.ravel()[indices], kind="stable")
        chosen = indices[order[:budget]]
        truth = np.asarray(sample["human"], bool)
        denominator, matched, selected = int(truth.sum()), int(truth.ravel()[chosen].sum()), len(chosen)
        metric = "known_positive_area_coverage_at_fixed_area"
    else:
        raise ValueError("Unknown task geometry")
    if denominator == 0:
        raise ValueError("No confirmed positive target for coverage evaluation")
    return {"coverage": matched / denominator, "matched": matched, "denominator": denominator,
            "selected": selected, "budget": budget, "metric": metric,
            "precision_measured": False, "unknown_as_negative": False}


def summarize(rows):
    if not rows:
        raise ValueError("No evaluation rows")
    # Never average point coverage and area coverage into an 'overall accuracy'.
    groups = {}
    for row in rows:
        key = (row["group"], row["metric"])
        groups.setdefault(key, {}).setdefault(row["seed"], []).append(row["coverage"])
    return {group: {"metric": metric,
                    "mean": float(np.mean([np.mean(v) for v in seeds.values()])),
                    "seeds": {str(k): float(np.mean(v)) for k, v in seeds.items()}}
            for (group, metric), seeds in groups.items()}


def compare(candidate, incumbent, baseline, protocol):
    def keyed(rows):
        result = {(r["group"], r["case"], r["seed"]): r for r in rows}
        if len(result) != len(rows):
            raise ValueError("Duplicate evaluation rows")
        return result
    ca, inc, base = map(keyed, (candidate, incumbent, baseline))
    if not ca or ca.keys() != inc.keys() or ca.keys() != base.keys():
        raise ValueError("Unmatched evaluation cohorts")
    for key in ca:
        for row in (ca[key], inc[key], base[key]):
            if (not 0 <= row["coverage"] <= 1 or row["selected"] > row["budget"] or
                    row["metric"] != ca[key]["metric"] or row["budget"] != ca[key]["budget"]):
                raise ValueError("Invalid or unmatched metric/budget")
    new, old, original = map(summarize, (candidate, incumbent, baseline))
    regressions = [g for g in new if new[g]["mean"] <
                   max(old[g]["mean"], original[g]["mean"]) - protocol.maximum_group_regression]
    gains = {}
    for metric in {v["metric"] for v in new.values()}:
        gains[metric] = float(np.mean([new[g]["mean"] - old[g]["mean"]
                                      for g in new if new[g]["metric"] == metric]))
    accepted = bool(not regressions and all(v >= -protocol.maximum_group_regression for v in gains.values())
                    and any(v >= protocol.minimum_gain for v in gains.values()))
    return {"accepted": accepted, "task_gains": gains, "regressions": regressions,
            "reason": "validation_improvement" if accepted else "insufficient_gain_or_regression",
            "statistical_significance_claimed": False}
