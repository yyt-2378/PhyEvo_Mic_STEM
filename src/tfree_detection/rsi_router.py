"""Small developmental router fitted only to doubly held-out acquisition scores.

The router may consume inner validation labels; inference sees only geometry
and whether original physics produced any support. Its evaluation acquisition
must have been excluded from every learner that supplied its fitting evidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np


BRANCHES = ("physics_budgeted", "hybrid_fixed_budget", "student_expert")


@dataclass(frozen=True)
class RouterProtocol:
    minimum_inner_groups: int = 2
    minimum_mean_gain: float = 0.01
    maximum_group_loss: float = 0.01

    def __post_init__(self):
        if self.minimum_inner_groups < 2:
            raise ValueError("At least two independent inner acquisitions are required")
        if not all(math.isfinite(v) and v >= 0 for v in (self.minimum_mean_gain, self.maximum_group_loss)):
            raise ValueError("Router limits must be finite and nonnegative")


def scope_key(geometry: str, teacher_empty: bool) -> str:
    if geometry not in {"point", "interface", "line", "region"} or type(teacher_empty) is not bool:
        raise ValueError("Router requires geometry and an observed Boolean teacher-empty flag")
    return geometry + ("::teacher_empty" if teacher_empty else "::teacher_nonempty")


def select_branch(router: Mapping, *, geometry: str, teacher_empty: bool) -> str:
    """Never receives image IDs, acquisition IDs, annotation coordinates or labels."""
    key = scope_key(geometry, teacher_empty)
    selected = router.get("scopes", {}).get(key, {}).get("selected_branch", "physics_budgeted")
    if selected not in BRANCHES:
        raise ValueError("Unknown or malformed router branch")
    return selected


def fit_router(records: Sequence[Mapping], *, outer_group: str,
               protocol: RouterProtocol | None = None) -> dict:
    protocol = protocol or RouterProtocol()
    scopes = defaultdict(list)
    seen = set()
    for record in records:
        if record["outer_group"] != outer_group or record["group"] == outer_group:
            raise ValueError("Outer evaluation labels cannot fit their own router")
        train_groups = set(record["model_train_groups"])
        if outer_group in train_groups or record["group"] in train_groups:
            raise ValueError("Nested student was exposed to outer or inner validation acquisition")
        identity = (record["seed"], record["group"], record["case_number"])
        if identity in seen:
            raise ValueError("Duplicate model-case evidence would overweight an acquisition")
        seen.add(identity)
        if record["main_comparison_eligible"]:
            scopes[scope_key(record["geometry"], record["teacher_empty"])].append(record)
    decisions = {}
    for key in [scope_key(g, e) for g in ("point", "interface") for e in (False, True)]:
        scoped = scopes.get(key, [])
        grouped = defaultdict(list)
        for r in scoped:
            grouped[(r["group"], r["seed"])].append(r)
        group_metrics = defaultdict(lambda: defaultdict(list))
        for (group, seed), cases in sorted(grouped.items()):
            for branch in BRANCHES:
                if branch != "physics_budgeted" and any(c["geometry"] not in c["model_geometry_support"] for c in cases):
                    continue
                values = [c["methods"][branch]["positive_coverage"] for c in cases]
                if any(v is None or not math.isfinite(v) for v in values):
                    raise ValueError("Nonfinite or absent inner evaluation metric")
                group_metrics[group][branch].append(float(np.mean(values)))
        expected_seeds = sorted({r["seed"] for r in scoped})
        summary = {g: {b: float(np.mean(v)) for b, v in branches.items()
                       if len(v) == len(expected_seeds)} for g, branches in group_metrics.items()}
        candidates = {}
        for branch in BRANCHES[1:]:
            complete = bool(summary) and all(branch in v and "physics_budgeted" in v for v in summary.values())
            deltas = [v[branch] - v["physics_budgeted"] for v in summary.values()] if complete else []
            mean = float(np.mean(deltas)) if deltas else None
            minimum = min(deltas) if deltas else None
            eligible = (len(deltas) >= protocol.minimum_inner_groups
                        and mean >= protocol.minimum_mean_gain
                        and minimum >= -protocol.maximum_group_loss)
            candidates[branch] = {"eligible": eligible, "mean_group_gain": mean,
                                  "worst_group_gain": minimum, "complete_group_count": len(deltas)}
        eligible = [b for b in BRANCHES[1:] if candidates[b]["eligible"]]
        selected = max(eligible, key=lambda b: (candidates[b]["mean_group_gain"], -BRANCHES.index(b))) if eligible else "physics_budgeted"
        decisions[key] = {
            "selected_branch": selected, "inner_group_metrics": summary,
            "independent_inner_group_count": len(summary), "seeds": expected_seeds,
            "candidates": candidates,
            "reason": "validated_inner_gain_at_fixed_budget" if eligible else "insufficient_groups_or_inner_regression",
        }
    return {"format": "phyevo-nested-observable-router-v1", "outer_group_excluded": outer_group,
            "protocol": asdict(protocol), "scopes": decisions,
            "inference_features": ["geometry", "teacher_empty"],
            "selection_uses_outer_labels": False, "production_approved": False,
            "evidence_kind": "nested_development_not_locked_external_test"}
