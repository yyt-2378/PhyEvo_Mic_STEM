"""Student-to-physics feedback without treating learned scores as measurements.

The student can reorder original proposals and add a budgeted rescue stream.
Embedding neighbors retrieve training references and support an experimental
novelty veto, not chemical identification or calibrated out-of-distribution
probability. All native physical evidence remains independently available.
"""
from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from .rsi_evaluation import (
    EvaluationProtocol, Prediction, _append_unique, _finite_plane, _xy,
    physics_points, predict_methods, student_mask, student_points,
)


FEEDBACK_BRANCH = "student_reranked_physics_plus_rescue"
BRANCHES = ("physics_budgeted", "hybrid_fixed_budget", FEEDBACK_BRANCH, "student_expert")


def feedback_predictions(image, teacher_mask, candidates, score, geometry, *, protocol=None):
    """No expert annotations, case identity or acquisition identity is accepted."""
    protocol = protocol or EvaluationProtocol()
    result = predict_methods(image, teacher_mask, candidates, score, score, geometry, protocol=protocol)
    result.pop("student_teacher_only")
    score = _finite_plane(score, "student score", np.asarray(image).shape)
    metadata = {"selection_uses_expert_labels": False, "calibrated": False, "budget_matched": True,
                "physical_evidence_modified": False, "student_scores_are_physical_evidence": False,
                "selection": "student_reranks_original_physics_quota_then_budgeted_student_rescue",
                "teacher_fraction": protocol.hybrid_teacher_fraction}
    if geometry == "point":
        budget = protocol.point_budget(score.shape)
        native = physics_points(candidates, score.shape)
        if len(native):
            xy = np.rint(native).astype(int)
            # Input order breaks score ties and already preserves physics ranking.
            order = np.argsort(-score[xy[:, 1], xy[:, 0]], kind="stable")
            ranked = native[order]
        else:
            ranked = native
        selected = []
        _append_unique(selected, ranked, int(math.floor(budget*protocol.hybrid_teacher_fraction)), protocol.point_min_distance_px)
        reserved = len(selected)
        _append_unique(selected, student_points(score, budget*4, protocol.point_min_distance_px), budget, protocol.point_min_distance_px)
        _append_unique(selected, ranked, budget, protocol.point_min_distance_px)
        result[FEEDBACK_BRANCH] = Prediction(geometry, points_xy=_xy(selected), budget=budget,
                                            metadata={**metadata, "reserved_physics_candidates": reserved})
    else:
        budget = protocol.area_budget(score.shape)
        teacher = np.asarray(teacher_mask, dtype=bool)
        # Student learns WHERE to spend a physics-support quota, not new physics.
        chosen = student_mask(score, int(math.floor(budget*protocol.hybrid_teacher_fraction)), exclude=~teacher)
        reserved = int(chosen.sum())
        chosen |= student_mask(score, budget-reserved, exclude=chosen)
        result[FEEDBACK_BRANCH] = Prediction(geometry, mask=chosen, budget=budget,
                                            metadata={**metadata, "reserved_physics_pixels": reserved})
    return result


def embedding_reference_context(query, bank, records, geometry, *, top_k=3):
    """Cosine neighbors from training-only, geometry-compatible reference bank.

    Threshold = 95th percentile of leave-one-reference-out nearest distances.
    Repeated frames are NOT independent calibration data. A single reference
    cannot define the threshold and returns 'insufficient_reference_diversity'.
    Close distance does not establish correct predictions or material identity.
    """
    query, bank = np.asarray(query, float), np.asarray(bank, float)
    if (query.ndim != 1 or bank.ndim != 2 or bank.shape != (len(records), len(query)) or
            not np.isfinite(query).all() or not np.isfinite(bank).all() or top_k < 1):
        raise ValueError("Invalid embedding/reference shape or value")
    norms = np.linalg.norm(bank, axis=1)
    qnorm = float(np.linalg.norm(query))
    if qnorm < 1e-12 or (norms < 1e-12).any():
        raise ValueError("Zero embeddings cannot define cosine references")
    indices = [i for i,r in enumerate(records) if r["geometry"] == geometry]
    common = {"calibrated_ood": False, "chemical_identity_inferred": False,
              "reference_count": len(indices), "reference_acquisition_count": len({records[i]["group"] for i in indices}),
              "threshold_uses_validation_labels": False}
    if not indices:
        return {**common, "nearest_distance": None, "threshold": None, "novelty_veto": True,
                "reason": "no_geometry_reference", "neighbors": []}
    references = bank[indices]/norms[indices, None]
    distances = np.clip(1-references@(query/qnorm), 0, 2)
    order = np.argsort(distances, kind="stable")[:top_k]
    neighbors = [{"case_number": records[indices[i]]["case_number"], "group": records[indices[i]]["group"],
                  "cosine_distance": float(distances[i])} for i in order]
    threshold = None
    reason = "insufficient_reference_diversity"
    if len(indices) >= 2:
        matrix = np.clip(1-references@references.T, 0, 2)
        np.fill_diagonal(matrix, np.inf)
        threshold = max(float(np.quantile(matrix.min(axis=1), .95)), 1e-6)
        reason = "experimental_reference_distance_exceeded" if distances.min() > threshold else "within_training_reference_distance_not_a_correctness_guarantee"
    return {**common, "nearest_distance": float(distances.min()), "threshold": threshold,
            "novelty_veto": threshold is None or bool(distances.min() > threshold), "reason": reason,
            "neighbors": neighbors}


def scope_key(geometry, teacher_empty):
    if geometry not in {"point", "interface", "line", "region"} or type(teacher_empty) is not bool:
        raise ValueError("Invalid observable feedback scope")
    return geometry + ("::teacher_empty" if teacher_empty else "::teacher_nonempty")


def fit_feedback_selector(records, *, outer_group, minimum_groups=2, minimum_gain=.01, maximum_group_loss=.01):
    """Select policy/step/branch/veto strictly inside doubly-held-out evidence.

    Scores are first averaged over cases, then seeds, then acquisitions. Labels
    select a development configuration, never enter runtime branch features.
    A single surviving point acquisition cannot justify automatic point routing.
    """
    if minimum_groups < 2 or minimum_gain < 0 or maximum_group_loss < 0:
        raise ValueError("Unsafe selector evidence thresholds")
    scoped, seen = defaultdict(list), set()
    for row in records:
        if (row["outer_group"] != outer_group or row["group"] == outer_group or
                outer_group in row["model_train_groups"] or row["group"] in row["model_train_groups"]):
            raise ValueError("Outer/inner validation leakage into feedback selection")
        identity = row["policy"], row["step"], row["seed"], row["group"], row["case_number"]
        if identity in seen:
            raise ValueError("Duplicate evidence would overweight an acquisition")
        seen.add(identity)
        if row["main_comparison_eligible"]:
            scoped[scope_key(row["geometry"], row["teacher_empty"])].append(row)
    scopes = {}
    for geometry in ("point", "interface"):
        for empty in (False, True):
            key = scope_key(geometry, empty)
            rows = scoped.get(key, [])
            candidates = []
            for policy, step in sorted({(r["policy"], r["step"]) for r in rows}):
                subset = [r for r in rows if r["policy"] == policy and r["step"] == step]
                for branch in BRANCHES[1:]:
                    for use_veto in (False, True):
                        grouped = defaultdict(list)
                        unsupported = False
                        for row in subset:
                            if geometry not in row["model_geometry_support"]:
                                unsupported = True
                            base = row["methods"]["physics_budgeted"]["positive_coverage"]
                            method = "physics_budgeted" if use_veto and row["embedding_context"]["novelty_veto"] else branch
                            value = row["methods"][method]["positive_coverage"]
                            grouped[(row["group"], row["seed"])].append((value, base))
                        per_group = defaultdict(list)
                        for (group, seed), values in grouped.items():
                            per_group[group].append(np.mean(np.asarray(values), axis=0))
                        gains = {g: float(np.mean(v, axis=0)[0]-np.mean(v, axis=0)[1]) for g,v in per_group.items()}
                        mean = float(np.mean(list(gains.values()))) if gains else None
                        approved = (not unsupported and len(gains) >= minimum_groups and mean >= minimum_gain and
                                    min(gains.values()) >= -maximum_group_loss)
                        candidates.append({"policy": policy, "step": step, "branch": branch, "embedding_veto": use_veto,
                                           "group_gains": gains, "mean_gain": mean, "eligible": approved,
                                           "unsupported_geometry": unsupported})
            accepted = [c for c in candidates if c["eligible"]]
            # Prefer less complex/no-veto and earlier checkpoint on exact ties.
            best = sorted(accepted, key=lambda c: (-c["mean_gain"], c["embedding_veto"], c["step"],
                                                   BRANCHES.index(c["branch"]), c["policy"]))[0] if accepted else None
            scopes[key] = {"selected": best, "fallback": "physics_budgeted", "candidates": candidates,
                           "inner_groups": sorted({r["group"] for r in rows}),
                           "reason": "inner_validated_development_candidate" if best else "insufficient_or_nonretained_inner_evidence"}
    return {"format": "phyevo-refinement-feedback-selector-v1", "outer_group_excluded": outer_group,
            "selection_uses_outer_labels": False, "inference_features": ["geometry", "teacher_empty", "optional_training_reference_distance"],
            "minimum_groups": minimum_groups, "minimum_gain": minimum_gain, "maximum_group_loss": maximum_group_loss,
            "production_approved": False, "scopes": scopes}


def selected_configuration(selector, *, geometry, teacher_empty):
    return selector.get("scopes", {}).get(scope_key(geometry, teacher_empty), {}).get("selected")
