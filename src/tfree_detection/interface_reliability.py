"""Observable geometric support for the boundary (not band) hypothesis.

A long, narrow connected interface supported over many analysis windows is
distinct from scattered texture transitions. These are explicit development
guardrails, NOT learned/calibrated probabilities or material-name routing.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import label,maximum_filter
from skimage.measure import regionprops


def long_boundary_support(labels,valid,patch_size):
    labels=np.asarray(labels);valid=np.asarray(valid,bool)
    if labels.shape!=valid.shape or labels.ndim!=2 or patch_size<1:
        raise ValueError("Invalid geometry inputs")
    boundary=(maximum_filter((labels==0).astype(np.uint8),size=3)>0)&(
        maximum_filter((labels==1).astype(np.uint8),size=3)>0)&valid
    props=sorted(regionprops(label(boundary,np.ones((3,3)))[0]),key=lambda r:r.area,reverse=True)
    if not props:return {"eligible":False,"reason":"no_connected_boundary"}
    main=props[0]
    aspect=float(main.major_axis_length/max(main.minor_axis_length,1.))
    windows=float(main.major_axis_length/patch_size)
    coherent=float(main.area/max(1,boundary.sum()))
    eligible=aspect>=12 and windows>=8 and coherent>=.8
    return {"eligible":bool(eligible),"aspect_ratio":aspect,"span_in_patch_windows":windows,
            "largest_component_share":coherent,
            "thresholds":{"minimum_aspect":12,"minimum_window_span":8,"minimum_coherence":.8},
            "calibrated":False}


def qualify_boundary_or_band(physical,strict_guard,*,patch_size=64):
    """Keep strict band test; corroborate long boundaries with one non-abstaining scale.

An unresolved alternate scale is an abstention, not evidence of the opposite
hypothesis. A conflicting resolved band is never accepted as corroboration.
"""
    result=dict(strict_guard)
    long=long_boundary_support(physical["labels"],physical["valid"],patch_size)
    result["long_boundary_support"]=long
    kind=strict_guard["hypothesis"]
    corroboration=any(v["hypothesis"]=="two_domain_boundary" and v["top_budget_overlap"]>=.5
                      for v in strict_guard["views"])
    conflict=any(v["hypothesis"]=="enclosed_texture_band" for v in strict_guard["views"])
    if kind=="two_domain_boundary" and long["eligible"] and corroboration and not conflict:
        result.update(eligible=True,qualification="long_boundary_with_scale_corroboration")
    else:
        result["qualification"]="strict_two_scale_consistency" if result["eligible"] else "insufficient_geometric_support"
    return result
