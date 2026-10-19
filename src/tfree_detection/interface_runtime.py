"""Geometry-aware interface candidates with explicit, quota-free branch gates.

Reliability here means an auditable consistency test, not calibrated defect
probability. The output is a review proposal under a fixed area cap. Sparse
expert positives cannot certify precision or normal-image specificity.
"""
from __future__ import annotations

from copy import deepcopy
import math

import numpy as np

from .config import DetectorConfig
from .context import DetectionContext
from .interface_geometry import ranked_support
from .interface_student import score_health
from .tasks.interface import PhaseBoundaryTask


def physical_consistency(image, config, *, area_fraction=.1):
    if not 0 < area_fraction <= 1:
        raise ValueError("Invalid area fraction")
    base_config=deepcopy(config)
    base_config.phase_boundary.geometry_interpretation="auto"
    base=PhaseBoundaryTask().run(image,base_config,{},DetectionContext())
    valid=base.evidence_maps["boundary_validity"]>.5
    budget=math.ceil(area_fraction*image.size)
    mask=ranked_support(base.score_map,budget,valid=valid)
    kind=base.metadata["geometry_interpretation"]["hypothesis"]
    views=[]
    # Geometry-derived, fixed relative window perturbation; no labels/names.
    for factor in (.75,1.25):
        alternate=deepcopy(base_config)
        alternate.phase_boundary.patch_size=max(16,2*round(config.phase_boundary.patch_size*factor/2))
        view=PhaseBoundaryTask().run(image,alternate,{},DetectionContext())
        other=ranked_support(view.score_map,budget,valid=view.evidence_maps["boundary_validity"]>.5)
        view_kind=view.metadata["geometry_interpretation"]["hypothesis"]
        overlap=float((mask&other).sum()/max(1,max(mask.sum(),other.sum())))
        views.append({"patch_size":alternate.phase_boundary.patch_size,
                      "hypothesis":view_kind,"top_budget_overlap":overlap,
                      "agree":view_kind==kind and overlap>=.5})
    stable=kind!="unresolved" and all(v["agree"] for v in views)
    return base,{"eligible":bool(stable),"hypothesis":kind,"views":views,
                 "threshold":.5,"required_agreeing_views":2,"uses_labels":False,
                 "uses_source_name":False,"calibrated":False}


def choose_interface(physical_score, physical_valid, physical_guard,
                     student_score=None, *, student_admitted=False, budget):
    """Choose one qualified branch, never reserve pixels for an unreliable one.

Student admission is supplied by a frozen validation receipt, NOT by current
expert labels. No filename, material or acquisition ID is an argument.
"""
    physics=ranked_support(physical_score,budget,valid=physical_valid)
    health=score_health(student_score) if student_score is not None else {"eligible":False,"reason":"absent"}
    if physical_guard.get("eligible",False):
        return physics,{"branch":"physics_geometry","reason":"two_scale_physical_consistency",
                        "student_health":health,"fixed_quota":False}
    if student_admitted and health["eligible"]:
        return ranked_support(student_score,budget),{"branch":"student","reason":"validation_admitted_and_output_healthy",
                                                    "student_health":health,"fixed_quota":False}
    # Preserve uncertain physical hypotheses for audit, but do not force-fill a
    # budget or present an unqualified model as a reliable joint improvement.
    return physics,{"branch":"physics_unqualified_review","reason":"no_qualified_branch",
                    "student_health":health,"fixed_quota":False,"needs_expert_review":True}
