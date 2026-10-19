"""Synthetic evidence distinguishes tied scores, routing and actual selection."""
import json

import numpy as np
import pytest

from .contracts import Protocol
from .evaluation import evaluate
from .stability_diagnostics import (TRANSFORMS, cutoff_ties, field_information,
    ordinal_tie_probe, prediction_stability, transform_array)


def test_constant_field_is_explicitly_uninformative_not_a_high_percentile_signal():
    values = np.ones((8, 10))
    mask = np.ones(values.shape, bool)
    result = field_information(values, mask)
    assert result['constant'] and result['std'] == 0
    assert result['largest_equal_value_fraction'] == result['tied_pixel_fraction'] == 1.
    drift = ordinal_tie_probe(values, mask)
    assert drift['mean_absolute_rank_drift'] > .3
    assert drift['maximum_absolute_rank_drift'] > .8
    assert not drift['whole_program_causality_proven']


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_distinct_values_have_no_ordinal_rank_drift(transform):
    value = np.arange(63).reshape(7, 9).astype(float)
    valid = value % 3 != 0
    result = ordinal_tie_probe(value, valid, transform)
    assert result['mean_absolute_rank_drift'] == 0
    assert result['changed_rank_fraction'] == 0
    assert np.array_equal(value, transform_array(transform_array(value, transform), transform, inverse=True))


def test_mask_and_nonfinite_values_are_accounted_for_without_fabricated_evidence():
    value = np.array([[1., np.nan], [np.inf, 9.]])
    mask = np.array([[True, True], [True, False]])
    result = field_information(value, mask)
    assert result['finite_pixels'] == 1 and result['nonfinite_pixels_in_mask'] == 2
    assert result['constant'] and result['minimum'] == result['maximum'] == 1
    empty = field_information(value, np.zeros(value.shape, bool))
    assert not empty['available'] and empty['constant'] is None
    assert empty['std'] is None and empty['largest_equal_value_fraction'] is None
    probe = ordinal_tie_probe(value, np.zeros(value.shape, bool))
    assert probe['mean_absolute_rank_drift'] is None
    json.dumps([result, empty, probe], allow_nan=False)


def test_cutoff_uses_entire_image_budget_and_reports_only_crossing_ties():
    score = np.array([[9., 8., 8., 8.], [1., 0., -1., 0.]])
    support = np.ones(score.shape, bool)
    result = cutoff_ties(score, support, .25)
    assert result['budget'] == result['selected_pixels'] == 2
    assert result['pixels_tied_at_cutoff'] == 3 and result['selected_pixels_at_cutoff'] == 1
    assert result['cutoff_splits_equal_values'] and result['selected_fraction_from_split_tie'] == .5
    # A tie is harmless for the cutoff if all its members fit in the budget.
    assert not cutoff_ties(score, support, .5)['cutoff_splits_equal_values']
    assert cutoff_ties(score, support, 1.)['selected_pixels'] == 5
    assert cutoff_ties(-np.ones((2, 3)), np.ones((2, 3), bool))['cutoff_score'] is None


def test_invariant_constant_scores_can_still_have_unstable_selected_pixels():
    score = np.ones((6, 8)); support = np.ones(score.shape, bool)
    result = prediction_stability(score, support, np.rot90(score), np.rot90(support),
                                  original_branch='b0', transformed_branch='b0')
    assert result['same_selected_branch'] is True
    assert result['mean_absolute_score_drift_on_common_support'] == 0
    assert result['selection_changed_fraction_of_image'] > 0
    assert result['selection_jaccard'] < 1
    assert result['original_cutoff']['cutoff_splits_equal_values']
    assert not result['labels_used'] and not result['accuracy_measured']


@pytest.mark.parametrize('transform', TRANSFORMS)
def test_actual_equivariant_prediction_and_fixed_budget_evaluator_agree(transform):
    score = np.arange(48).reshape(6, 8).astype(float)
    support = score % 3 != 0
    moved, moved_support = transform_array(score, transform), transform_array(support, transform)
    result = prediction_stability(score, support, moved, moved_support, transform)
    assert result['selection_jaccard'] == 1 and result['original_selection_retained_fraction'] == 1
    assert result['support_changed_fraction_of_image'] == 0
    protocol = Protocol(('A',), ('B',), ('C',), area_fraction=.1)
    truth = {'image': score, 'valid': np.ones(score.shape, bool), 'human': support, 'geometry': 'interface'}
    measured = evaluate(truth, {'score': score, 'support': support}, protocol)
    assert result['original_selected_pixels'] == measured['selected']
    assert result['original_cutoff']['budget'] == measured['budget']


def test_empty_support_and_changed_branch_are_not_reported_as_perfect_retention():
    score = np.zeros((5, 7)); support = np.zeros(score.shape, bool)
    result = prediction_stability(score, support, score, support, 'flip_lr',
                                  original_branch='old', transformed_branch='new')
    assert result['same_selected_branch'] is False
    assert result['selection_jaccard'] is None and result['original_selection_retained_fraction'] is None
    assert result['mean_absolute_score_drift_on_common_support'] is None
    json.dumps(result, allow_nan=False)


def test_mismatched_shapes_masks_nonfinite_scores_and_invalid_budget_are_rejected():
    score = np.ones((5, 7)); support = np.ones(score.shape, bool)
    with pytest.raises(ValueError, match='boolean'):
        field_information(score, support.astype(int))
    with pytest.raises(ValueError, match='align'):
        prediction_stability(score, support, score, support)  # rot90 changes shape
    with pytest.raises(ValueError, match='finite scores'):
        cutoff_ties(score * np.nan, support)
    for fraction in (0., -1., float('nan'), float('inf'), 1.01):
        with pytest.raises(ValueError, match='Area fraction'):
            cutoff_ties(score, support, fraction)
    with pytest.raises(ValueError, match='supported transform'):
        ordinal_tie_probe(score, support, 'warp')
