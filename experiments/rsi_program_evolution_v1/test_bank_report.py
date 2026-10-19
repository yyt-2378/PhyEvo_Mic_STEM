"""Success requires added capacity, not only routing or relative stress survival."""
from copy import deepcopy

from .bank_report import paired_decision


def evidence():
    summary = lambda value: {g: {'mean': value} for g in ('A', 'B', 'C')}
    stress = lambda value: {'stress-' + g: {'mean': value} for g in ('A', 'B', 'C')}
    full = {'seed': 23, 'prepared': {'summary': summary(.5)}, 'development': {'summary': summary(.7)},
        'rounds': [{'status': 'accepted'}, {'status': 'accepted'}, {'status': 'retained_for_research'}],
        'stress_audit': {'frozen': {'receipt': {'summary': stress(.5)}},
                         'selected': {'receipt': {'summary': stress(.7)}}}}
    route = {'development': {'summary': summary(.55)}}
    config = {'evaluation': {'validation_groups': ['B', 'C'], 'maximum_group_regression': .002}}
    required = {'minimum_full_over_frozen_nontraining_source_mean_gain': .002,
        'minimum_full_over_route_only_nontraining_source_mean_gain': .002,
        'minimum_positive_selected_new_capacity_gain_over_old_library_ceiling': .002,
        'minimum_accepted_updates_per_successful_full_trajectory': 2}
    novelty = {'mean_selected_new_gain_over_old_casewise_ceiling': {'A': .1, 'B': .1, 'C': .1}}
    return full, route, config, required, novelty


def test_measured_creation_adoption_and_repeated_inheritance_can_pass():
    result = paired_decision(*evidence())
    assert result['passed'] and result['accepted_updates'] == 2
    assert result['inherited_updates'] == 2 and result['stress_vs_own_original_retained']


def test_both_systems_failing_under_rotation_is_not_absolute_retention():
    values = evidence(); full = values[0]
    for group in full['stress_audit']['frozen']['receipt']['summary']:
        full['stress_audit']['frozen']['receipt']['summary'][group]['mean'] = .3
        full['stress_audit']['selected']['receipt']['summary'][group]['mean'] = .4
    result = paired_decision(*values)
    assert result['stress_vs_frozen_retained']
    assert not result['stress_vs_own_original_retained'] and not result['passed']


def test_routing_gain_without_new_capacity_does_not_complete_the_goal():
    values = evidence()
    values[-1]['mean_selected_new_gain_over_old_casewise_ceiling'] = {'A': 0., 'B': 0., 'C': 0.}
    result = paired_decision(*values)
    assert result['full_over_frozen_gain'] > 0 and not result['passed']


def test_one_update_or_training_source_only_improvement_is_insufficient():
    values = evidence()
    values[0]['rounds'] = [{'status': 'accepted'}, {'status': 'retained_for_research'}]
    assert not paired_decision(*values)['passed']
    values = evidence()
    values[0]['development']['summary'] = deepcopy(values[0]['prepared']['summary'])
    values[0]['development']['summary']['A']['mean'] = .99
    result = paired_decision(*values)
    assert result['full_over_frozen_gain'] == 0. and not result['passed']
