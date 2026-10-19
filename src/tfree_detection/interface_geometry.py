"""Interpret reciprocal-texture topology, without names or expert labels.

Two separated, substantial flanks of the SAME texture are evidence for an
A|B|A band hypothesis, not proof of a chemical phase or of a defect. Other
topologies retain the original two-domain boundary hypothesis. Thresholds
are geometric guardrails, not learned/calibrated confidence probabilities.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, label


def interpret_domain_geometry(domain_labels, valid, *, separated: bool):
    labels = np.asarray(domain_labels)
    valid = np.asarray(valid, bool)
    if labels.ndim != 2 or labels.shape != valid.shape or not labels.size:
        raise ValueError("Expected same-shaped nonempty domain and validity planes")
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Unknown reciprocal-texture label")
    score = np.zeros(labels.shape, np.float32)
    receipt = {"hypothesis": "two_domain_boundary" if separated else "unresolved",
               "uses_source_name": False, "uses_expert_labels": False,
               "calibrated": False, "band_checks": []}
    if not separated:
        return score, receipt
    total = labels.size
    for value in (0, 1):
        band = labels == value
        fraction = float(band.mean())
        if not .08 <= fraction <= .45:
            continue
        components, _ = label(band, structure=np.ones((3, 3)))
        counts = np.bincount(components.ravel()); counts[0] = 0
        dominant = components == int(np.argmax(counts))
        coherence = float(dominant.sum() / max(1, band.sum()))
        # Invalid/vacuum pixels are NOT counted as matching crystalline flanks.
        flanks, _ = label(labels == 1-value, structure=np.ones((3, 3)))
        sizes = np.sort(np.bincount(flanks.ravel())[1:])[::-1]
        second = int(sizes[1]) if len(sizes) >= 2 else 0
        paired = float(sizes[:2].sum() / max(1, (labels == 1-value).sum()))
        yx = np.argwhere(dominant)
        eig = np.linalg.eigvalsh(np.cov(yx.T)) if len(yx) >= 3 else np.ones(2)
        elongation = float(np.sqrt(eig[-1] / max(eig[0], 1e-8)))
        valid_fraction = float(valid[dominant].mean()) if dominant.any() else 0.
        passed = (coherence >= .90 and second / total >= .08 and
                  paired >= .90 and elongation >= 2 and valid_fraction >= .40)
        receipt["band_checks"].append({"label": value, "area_fraction": fraction,
            "component_coherence": coherence, "second_flank_fraction": second/total,
            "two_flank_share": paired, "elongation": elongation,
            "valid_fraction": valid_fraction, "passed": bool(passed)})
        if passed:
            score = distance_transform_edt(dominant).astype(np.float32)
            score[~valid] = 0
            receipt.update(hypothesis="enclosed_texture_band", band_label=value,
                           interpretation="band interior; similar projected texture on both flanks")
            break
    return score, receipt


def ranked_support(score, budget, *, valid=None):
    """Stable top-k positive evidence only; never fill invalid/zero-score pixels."""
    score = np.asarray(score)
    if score.ndim != 2 or not np.isfinite(score).all() or budget < 0:
        raise ValueError("Invalid score or budget")
    available = score > 0
    if valid is not None:
        if np.shape(valid) != score.shape:
            raise ValueError("Validity shape mismatch")
        available &= np.asarray(valid, bool)
    indices = np.flatnonzero(available)
    order = np.argsort(-score.ravel()[indices], kind="stable")
    mask = np.zeros(score.shape, bool)
    mask.ravel()[indices[order[:int(budget)]]] = True
    return mask
