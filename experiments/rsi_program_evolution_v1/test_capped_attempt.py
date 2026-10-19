"""Limited reservation never changes actual write limits or scientific gates."""
from copy import deepcopy
import json

import pytest

from .capped_attempt_continuation import assess, verify_prior_refusals


def inputs():
    policy = {'max_prediction_pool_bytes': 256*1024**2, 'max_growth_bytes': 1024**3}
    plan = {'input_pool_fits': True, 'prediction_pool_fits_expected_replay': False,
            'planned_total_new_bytes': 110*1024**2, 'planned_prediction_bytes': 80*1024**2,
            'missing_predictions': [{'branch': 'b49', 'split': 'development'}],
            'training_jobs': 0, 'checkpoint_bytes_bound': 0}
    measured = {'reason': None, 'growth_bytes': 420*1024**2}
    return plan, measured, 202*1024**2, 51, policy


def test_partial_reservation_is_never_reported_as_strict_fit():
    args = inputs(); before = deepcopy(args)
    result = assess(*args, allow_uncertain=True)
    assert result['bounded_attempt_allowed'] and result['uncertain_prediction_reservation']
    assert not result['strict_preflight_fits'] and not result['completion_space_guaranteed']
    assert args == before and result['unchanged_actual_storage_caps']


def test_uncertain_attempt_requires_explicit_opt_in():
    result = assess(*inputs())
    assert not result['bounded_attempt_allowed']


@pytest.mark.parametrize('change', ['input_limit', 'growth_limit', 'disk_reserve',
                                   'total_growth', 'training', 'checkpoint', 'no_unknowns'])
def test_other_failures_cannot_be_overridden(change):
    plan, measured, used, cohort, policy = inputs()
    if change == 'input_limit': plan['input_pool_fits'] = False
    if change in ('growth_limit', 'disk_reserve'): measured['reason'] = change
    if change == 'total_growth': plan['planned_total_new_bytes'] = policy['max_growth_bytes']
    if change == 'training': plan['training_jobs'] = 1
    if change == 'checkpoint': plan['checkpoint_bytes_bound'] = 1
    if change == 'no_unknowns': plan['missing_predictions'] = []
    assert not assess(plan, measured, used, cohort, policy, allow_uncertain=True)['bounded_attempt_allowed']


def test_remaining_worker_metadata_space_is_required():
    plan, measured, _, cohort, policy = inputs()
    used = policy['max_prediction_pool_bytes'] - (65536 + cohort*16384)
    assert not assess(plan, measured, used, cohort, policy, allow_uncertain=True)['bounded_attempt_allowed']


def test_strict_fit_remains_a_separate_case():
    plan, measured, _, cohort, policy = inputs()
    plan['prediction_pool_fits_expected_replay'] = True
    result = assess(plan, measured, 0, cohort, policy)
    assert result['strict_preflight_fits'] and result['bounded_attempt_allowed']
    assert not result['uncertain_prediction_reservation']


@pytest.mark.parametrize('changed', ['fits', 'scientific_round_started', 'archive_tail_unchanged', 'raw_proposal_sha256'])
def test_started_or_changed_prior_attempt_cannot_be_replayed(tmp_path, changed):
    folder = tmp_path/'storage_runtime_input_reuse_v1'; folder.mkdir()
    path = folder/'full-s41-r05-volume.json'
    record = {'fits': False, 'scientific_round_started': False,
              'archive_tail_unchanged': 'tail', 'raw_proposal_sha256': 'raw'}
    record[changed] = True if changed in ('fits', 'scientific_round_started') else 'changed'
    with path.open('x') as stream: json.dump(record, stream)
    with pytest.raises(ValueError, match='may have started or changed'):
        verify_prior_refusals(tmp_path, 'full-s41-r05', 'tail', 'raw')
