"""Capability creation is separate from routing and training-set improvement."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from .archive import Archive
from .contracts import Protocol
from .expansion_state import CompactSystemExecutor, array_digest, expansion_admission, library_gaps, selected_novel_gain, transformed_audit
from .expansion_study import choose_reference, parse_proposal, program_identity
from .system_state import RAW_PIPELINE
from .system_study import checkpoint
from .test_system_state import fixture, prepare_initial, proposal


def rows(a=.2, b=.3, branch='b0'):
    return [{'case': g, 'group': g, 'seed': 23, 'coverage': v, 'metric': 'area',
             'budget': 10, 'selected': 10, 'denominator': 10, 'branch': branch}
            for g, v in [('A', a), ('B', b)]]


def test_diagnosis_separates_wrong_choice_from_missing_capacity():
    result = library_gaps(rows(), {'b0': rows(), 'b1': rows(.8, .35)})
    assert result[0]['routing_gap'] == pytest.approx(.6)
    assert result[0]['remaining_library_deficit'] == pytest.approx(.2)
    assert result[1]['routing_gap'] == pytest.approx(.05)
    assert result[1]['remaining_library_deficit'] == pytest.approx(.65)
    assert all(r['oracle_is_diagnostic_only'] for r in result)


def test_library_ceiling_does_not_allow_unmatched_candidates_or_budgets():
    with pytest.raises(ValueError, match='cohorts'):
        library_gaps(rows(), {'b0': rows()[:1]})
    invalid = rows(); invalid[0]['budget'] = 11
    with pytest.raises(ValueError, match='budget'):
        library_gaps(rows(), {'b0': invalid})
    with pytest.raises(ValueError, match='Unique'):
        library_gaps(rows() + rows()[:1], {'b0': rows()})


def test_budget_limited_misses_are_not_mislabeled_as_fixable_capability_gap():
    current = rows(.05, .06); candidate = rows(.08, .09)
    for row in current + candidate:
        row['denominator'] = 100
    result = library_gaps(current, {'b0': current, 'b1': candidate})
    assert result[0]['remaining_library_deficit'] == pytest.approx(.02)
    assert result[0]['unavoidable_budget_limited_miss_fraction'] == pytest.approx(.9)
    candidate[0]['coverage'] = .11
    with pytest.raises(ValueError, match='upper bound'):
        library_gaps(current, {'b1': candidate})


def test_new_name_without_additional_capacity_is_not_a_novel_gain():
    frozen = {'b0': rows(.6, .5), 'b1': rows(.5, .8)}
    result = selected_novel_gain(rows(.59, .79, 'b2'), frozen, {'b0', 'b1'})
    assert all(v < 0 for v in result['mean_selected_new_gain_over_old_casewise_ceiling'].values())
    better = selected_novel_gain(rows(.65, .85, 'b2'), frozen, {'b0', 'b1'})
    assert better['mean_selected_new_gain_over_old_casewise_ceiling']['B'] == pytest.approx(.05)
    existing = selected_novel_gain(rows(.65, .85, 'b1'), frozen, {'b0', 'b1'})
    assert existing['mean_selected_new_gain_over_old_casewise_ceiling'] == {'A': 0., 'B': 0.}


def test_training_replay_gain_alone_cannot_admit_expansion():
    p = Protocol(('A',), ('B',), ('C',), seeds=(23,), maximum_group_regression=.002)
    old = rows(.5, .5)
    result = expansion_admission(rows(.8, .5), old, old, p, {'A': .5, 'B': .5})
    assert not result['accepted'] and result['weight_fit_excluded_development_gain'] == 0.
    assert expansion_admission(rows(.5, .6), old, old, p, {'A': .5, 'B': .5})['accepted']
    assert not expansion_admission(rows(.4, .9), old, old, p, {'A': .5, 'B': .5})['accepted']


def envelope(**changes):
    return {'research_decision': {'diagnosis': 'routing_gap', 'operation_target': 'adoption',
                'evidence': 'An existing branch covers more development positives',
                'alternatives': 'A new branch may be unnecessary', 'new_capacity_test': 'Compare against frozen-library ceiling'},
            'system': asdict(proposal(**changes))}


def test_research_controls_restrict_operations_not_outcomes():
    value = envelope()
    parse_proposal(value, 'route', 'a'*64, {'b0': {}, 'b1': {}})
    b = {'name': 'b2', 'pipeline_code': RAW_PIPELINE, 'training_steps': 0, 'parent_branch': None, 'replaces': None}
    value = envelope(operation='add_branch', new_branch=b, active_branches=['b0', 'b1', 'b2'])
    value['research_decision']['operation_target'] = 'analysis'
    parse_proposal(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})
    with pytest.raises(ValueError, match='Routing-only'):
        parse_proposal(value, 'route', 'a'*64, {'b0': {}, 'b1': {}})
    value['research_decision']['operation_target'] = 'student'
    with pytest.raises(ValueError, match='actually train'):
        parse_proposal(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})


def test_shared_budget_counts_router_and_student_training():
    code = 'def fit(samples,context):\n return {}\ndef predict(sample,state,context):\n return {}\n'
    b = {'name': 'b2', 'pipeline_code': code, 'training_steps': 3000, 'parent_branch': None, 'replaces': None}
    value = envelope(operation='add_branch', new_branch=b, active_branches=['b0', 'b1', 'b2'],
                     router_code='def route(s,b,x,c):\n return {}\ndef fit(s,c):\n return {}\n', router_training_steps=2000)
    with pytest.raises(ValueError, match='total optimization'):
        parse_proposal(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})


def test_program_identity_ignores_formatting_not_actual_mechanism():
    assert program_identity('def f(x):\n return x+1\n') == program_identity('def f(x):\n    return x + 1 # same\n')
    assert program_identity('def f(x):\n return x+1\n') != program_identity('def f(x):\n return x+2\n')


def test_freeze_best_complete_system_not_highest_single_source():
    summary = lambda a, b: {'A': {'mean': a}, 'B': {'mean': b}}
    def item(key, a, b, c):
        return {'trajectory': key, 'prepared': {'baseline_summary': summary(.2, .2)},
            'selected_development': {'summary': summary(a, b)},
            'audit': {'old_system': {'candidate': key+'old', 'receipt': {'rows': [], 'summary': {'C': {'mean': .2}}}},
                      'selected_system': {'candidate': key+'new', 'receipt': {'rows': [], 'summary': {'C': {'mean': c}}}}}}
    best, compared = choose_reference({'trajectories': [item('one', .95, .95, 0.), item('two', .7, .7, .7)]})
    assert best['trajectory'] == 'two' and best['role'] == 'selected_system'
    assert len(compared) == 4


def test_compact_staging_shares_bytes_without_exposing_labels(tmp_path):
    archive, original = fixture(tmp_path/'original')
    new_archive = Archive(tmp_path/'new/archive', archive.protocol, create=True)
    runner = CompactSystemExecutor(original.workspace, new_archive, original.sandbox, original.samples, tmp_path/'pool')
    try:
        runner.stage(tmp_path/'first', [original.samples[0]])
        runner.stage(tmp_path/'second', [original.samples[0]])
        a = tmp_path/'first/sample-0000.npz'; b = tmp_path/'second/sample-0000.npz'
        assert a.stat().st_ino == b.stat().st_ino and a.stat().st_nlink == 3
        with np.load(a, allow_pickle=False) as arrays:
            assert 'human' not in arrays and 'known' not in arrays
        runner.stage(tmp_path/'fit', [original.samples[0]], mode='fit', steps=1)
        with np.load(tmp_path/'fit/sample-0000.npz', allow_pickle=False) as arrays:
            assert 'human' in arrays and 'known' in arrays
        assert a.stat().st_ino != (tmp_path/'fit/sample-0000.npz').stat().st_ino
        with pytest.raises(ValueError, match='nontraining'):
            runner.stage(tmp_path/'invalid_fit', [original.samples[1]], mode='fit')
    finally:
        archive.close(); new_archive.close()


def test_imported_complete_state_replays_in_real_sandbox(tmp_path):
    archive, original = fixture(tmp_path/'original')
    new_archive = Archive(tmp_path/'new/archive', archive.protocol, create=True)
    runner = CompactSystemExecutor(original.workspace, new_archive, original.sandbox, original.samples, tmp_path/'pool')
    try:
        identity, expected = prepare_initial(original)
        old_checkpoint = checkpoint(archive, original)
        imported = runner.import_system(original, identity)
        trial, measured = runner.execute(imported)
        assert 'rows' in measured, measured
        assert measured['summary'] == expected['summary']
        new_archive.select(imported, True, 'verified frozen import', trial)
        assert checkpoint(archive, original) == old_checkpoint
        _, replay = runner.execute(imported, replay=True)
        assert replay['summary'] == measured['summary']
        for a, b in zip(expected['rows'], measured['rows']):
            assert a['branch'] == b['branch'] and a['coverage'] == b['coverage']
    finally:
        archive.close(); new_archive.close()


def test_stress_preserves_alignment_but_is_not_a_new_original_source(tmp_path):
    archive, runner = fixture(tmp_path)
    try:
        samples = transformed_audit(runner.samples)
        assert samples[0].group == 'stress-A'
        assert np.array_equal(samples[0].human, np.rot90(runner.samples[0].human))
        assert np.array_equal(samples[0].fields['base'], np.rot90(runner.samples[0].fields['base']))
        assert array_digest(samples[0].fields) != array_digest(runner.samples[0].fields)
    finally:
        archive.close()
