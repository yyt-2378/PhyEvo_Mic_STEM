"""Label-free diagnostics for degenerate fields and geometric instability.

These functions describe evidence; they do not repair a candidate, pick a
branch, change the frozen evaluator, or grant system admission. Ordinal ranks
below deliberately reproduce the legacy tie-breaking operation as a probe,
not as a recommended normalization. No coordinates or input arrays are returned.
"""
from __future__ import annotations

import math

import numpy as np

from .active_diagnostics import selected_mask


TRANSFORMS = ('rot90', 'rot180', 'rot270', 'flip_lr', 'flip_ud', 'transpose')


def transform_array(value, transform, *, inverse=False):
    """Exact pixel permutations only: no resampling or changed intensity."""
    value = np.asarray(value)
    if value.ndim != 2 or transform not in TRANSFORMS:
        raise ValueError('An aligned 2D array and a supported transform are required')
    if transform.startswith('rot'):
        k = int(transform[3:]) // 90
        return np.rot90(value, -k if inverse else k)
    if transform == 'flip_lr':
        return np.fliplr(value)
    if transform == 'flip_ud':
        return np.flipud(value)
    return value.T


def _field(values, mask):
    values, mask = np.asarray(values, dtype=np.float64), np.asarray(mask)
    if values.ndim != 2 or values.size == 0 or mask.shape != values.shape or mask.dtype != bool:
        raise ValueError('Aligned nonempty 2D values and a boolean mask are required')
    return values, mask & np.isfinite(values)


def field_information(values, mask):
    """Report exact ties and numerical spread within the declared mask only.

    A constant field is not evidence for a defect, even if every positive
    receives a high right-inclusive percentile. Low *nonzero* spread is reported
    numerically, without inventing a universal domain-specific usefulness gate.
    """
    values, finite = _field(values, mask)
    masked = int(np.asarray(mask).sum())
    sample = values[finite]
    record = {'masked_pixels': masked, 'finite_pixels': int(sample.size),
              'nonfinite_pixels_in_mask': masked - int(sample.size),
              'available': bool(sample.size)}
    if not sample.size:
        return dict(record, constant=None, distinct_values=0,
                    largest_equal_value_fraction=None, tied_pixel_fraction=None,
                    minimum=None, maximum=None, std=None, q10_q50_q90=None)
    unique, counts = np.unique(sample, return_counts=True)
    return dict(record, constant=bool(unique.size == 1), distinct_values=int(unique.size),
                largest_equal_value_fraction=float(counts.max() / sample.size),
                tied_pixel_fraction=float(counts[counts > 1].sum() / sample.size),
                minimum=float(unique[0]), maximum=float(unique[-1]),
                std=float(np.std(sample)), q10_q50_q90=np.quantile(sample, [.1, .5, .9]).tolist())


def _legacy_ordinal_rank(values, mask):
    ids = np.flatnonzero(mask.ravel())
    order = np.argsort(values.ravel()[ids], kind='stable')
    ranked = np.zeros(values.shape, dtype=np.float64)
    if ids.size:
        ranked.ravel()[ids[order]] = (np.arange(ids.size) + .5) / ids.size
    return ranked


def ordinal_tie_probe(values, mask, transform='rot90'):
    """How much arbitrary equal-value ordering changes under a permutation.

    This identifies susceptibility of a *specific operation*. It does not prove
    that a whole candidate uses it or that it causes all of its observed loss.
    """
    values, finite = _field(values, mask)
    original = _legacy_ordinal_rank(values, finite)
    moved = _legacy_ordinal_rank(transform_array(values, transform),
                                 transform_array(finite, transform))
    delta = np.abs(original - transform_array(moved, transform, inverse=True))[finite]
    return {'transform': transform, 'finite_pixels': int(delta.size),
            'mean_absolute_rank_drift': float(delta.mean()) if delta.size else None,
            'maximum_absolute_rank_drift': float(delta.max()) if delta.size else None,
            'changed_rank_fraction': float(np.mean(delta > 0)) if delta.size else None,
            'operation': 'stable ordinal rank; ties ordered by flattened pixel index',
            'whole_program_causality_proven': False}


def _prediction(score, support, fraction):
    score, support = np.asarray(score), np.asarray(support)
    if (score.ndim != 2 or score.size == 0 or support.shape != score.shape
            or support.dtype != bool or not np.isfinite(score).all()):
        raise ValueError('Aligned finite scores and boolean support are required')
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Area fraction must be finite and in (0, 1]')
    return score, support


def cutoff_ties(score, support, fraction=.10):
    """Ambiguity at the exact frozen area-budget cutoff, not an alternative metric."""
    score, support = _prediction(score, support, fraction)
    budget = max(1, math.ceil(fraction * score.size))
    values = score[support & (score > 0)]
    selected = min(budget, values.size)
    record = {'budget': budget, 'eligible_positive_pixels': int(values.size),
              'selected_pixels': int(selected), 'cutoff_score': None,
              'pixels_tied_at_cutoff': 0, 'selected_pixels_at_cutoff': 0,
              'cutoff_splits_equal_values': False,
              'selected_fraction_from_split_tie': 0.}
    if not selected:
        return record
    cutoff = np.partition(values, values.size - selected)[values.size - selected]
    tied = int(np.count_nonzero(values == cutoff))
    chosen_ties = int(selected - np.count_nonzero(values > cutoff))
    split = 0 < chosen_ties < tied
    return dict(record, cutoff_score=float(cutoff), pixels_tied_at_cutoff=tied,
                selected_pixels_at_cutoff=chosen_ties, cutoff_splits_equal_values=split,
                selected_fraction_from_split_tie=float(chosen_ties / selected) if split else 0.)


def prediction_stability(original_score, original_support, transformed_score,
                         transformed_support, transform='rot90', fraction=.10,
                         *, original_branch=None, transformed_branch=None):
    """Compare actual saved executions after undoing the input transform.

    Selection occurs in each execution's own coordinates *before* alignment,
    preserving the frozen evaluator's stable tie rule. This distinguishes an
    equivariant score map from a non-equivariant selected region. Labels are
    deliberately not accepted. Supplied branch IDs are compared, not returned.
    """
    a, sa = _prediction(original_score, original_support, fraction)
    moved, moved_support = _prediction(transformed_score, transformed_support, fraction)
    b = transform_array(moved, transform, inverse=True)
    sb = transform_array(moved_support, transform, inverse=True)
    if b.shape != a.shape:
        raise ValueError('Transformed prediction does not align with the original shape')
    ca = selected_mask(a, sa, fraction)
    cb = transform_array(selected_mask(moved, moved_support, fraction), transform, inverse=True)
    intersection, union = int((ca & cb).sum()), int((ca | cb).sum())
    common_support = sa & sb
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))[common_support]
    branch_known = original_branch is not None and transformed_branch is not None
    return {'transform': transform, 'image_pixels': int(a.size),
            'same_selected_branch': original_branch == transformed_branch if branch_known else None,
            'support_changed_fraction_of_image': float(np.mean(sa != sb)),
            'common_support_pixels': int(common_support.sum()),
            'mean_absolute_score_drift_on_common_support': float(delta.mean()) if delta.size else None,
            'maximum_absolute_score_drift_on_common_support': float(delta.max()) if delta.size else None,
            'original_selected_pixels': int(ca.sum()), 'transformed_selected_pixels': int(cb.sum()),
            'selection_jaccard': intersection / union if union else None,
            'selection_changed_fraction_of_image': float(np.mean(ca != cb)),
            'original_selection_retained_fraction': intersection / int(ca.sum()) if ca.any() else None,
            'original_cutoff': cutoff_ties(a, sa, fraction),
            'transformed_cutoff': cutoff_ties(moved, moved_support, fraction),
            'labels_used': False, 'accuracy_measured': False}
