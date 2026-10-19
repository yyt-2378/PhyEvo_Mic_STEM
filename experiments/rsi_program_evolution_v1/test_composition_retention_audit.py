import pytest
import numpy as np
from types import SimpleNamespace

from . import composition_study as study
from .composition_retention_audit import domain_field_provenance, trace_calls, trajectory_requirements
from .contracts import file_hash, write_new


CRITERIA = {'accepted_updates_per_successful_trajectory': 2,
            'minimum_nontraining_gain_over_frozen_and_route': .002,
            'minimum_new_capacity_over_old_casewise_ceiling': .002}


def row(number, identity, calls=None, accepted=True):
    return {'round': number, 'identity': identity, 'accepted': accepted,
            'ancestor_execution_calls': calls or {},
            'nontraining_gain_over_frozen': .01,
            'nontraining_gain_over_route': .01,
            'nontraining_gain_over_old_ceiling': .01}


def test_accepted_labels_without_actual_later_calls_do_not_suffice():
    result = trajectory_requirements([row(1, 'a'), row(2, 'b')], CRITERIA)
    assert result['accepted_updates'] == 2
    assert not result['qualifies_for_final_audit']


def test_rejected_descendant_can_witness_real_use_but_is_not_an_upgrade():
    rows = [row(1, 'a'), row(2, 'b', {'a': 51}), row(3, 'c', {'a': 51, 'b': 51}, False)]
    result = trajectory_requirements(rows, CRITERIA)
    assert result['accepted_updates'] == 2
    assert result['qualifies_for_final_audit']
    assert result['latest_accepted'] == 'b'
    rows[1]['nontraining_gain_over_old_ceiling'] = .001
    assert not trajectory_requirements(rows, CRITERIA)['qualifies_for_final_audit']


def test_one_improvement_is_not_repeatable_improvement():
    assert not trajectory_requirements([row(1, 'a'), row(2, 'b', {'a': 51}, False)], CRITERIA)['qualifies_for_final_audit']
    assert not trajectory_requirements([], CRITERIA)['qualifies_for_final_audit']


def test_trace_requires_matching_node_and_counts_real_ancestor_calls(tmp_path):
    output = tmp_path / 'original_output'
    output.mkdir()
    write_new(output / 'composition-trace-0000.json', {'node': 'child', 'calls': [
        {'call': 'parent', 'node': 'ancestor'}, {'call': 'parent', 'node': 'frozen'}]})
    result = {'identity': 'child', 'original': {'rows': [{}]}}
    assert trace_calls(tmp_path, result, 'original') == {'ancestor': 1}
    with pytest.raises(ValueError, match='another candidate'):
        trace_calls(tmp_path, dict(result, identity='wrong'), 'original')
    with pytest.raises(FileNotFoundError):
        trace_calls(tmp_path, {'identity': 'child', 'original': {'rows': [{}, {}]}}, 'original')


def test_domain_provenance_requires_bound_physical_array_not_arbitrary_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    physical = tmp_path / 'outputs/interface_improvement_20261015_v1'
    (physical / 'physical').mkdir(parents=True)
    hashes = {}
    for name in ('scripts/run_interface_improvement.py', 'src/tfree_detection/tasks/interface.py'):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        write_new(p, {'synthetic_hash_fixture': True})
        hashes[name] = file_hash(p)
    write_new(physical / 'physical_audit.json', {'source_hashes': hashes})
    labels = np.array([[0, 0], [1, 1]], dtype=np.int8)
    p = physical / 'physical/case-001.npz'
    with p.open('xb') as stream:
        np.savez_compressed(stream, labels=labels)
    fields = tmp_path / 'fields'
    fields.mkdir()
    write_new(fields / 'dataset.json', {'samples': [
        {'key': 'case-001', 'physical_sha256': file_hash(p)}]})
    sample = SimpleNamespace(key='case-001', fields={'domain_labels': labels.copy()})
    assert domain_field_provenance({'fields': 'fields'}, [sample])['cases_checked'] == 1
    sample.fields['domain_labels'][0, 0] = 1
    with pytest.raises(ValueError, match='does not reproduce'):
        domain_field_provenance({'fields': 'fields'}, [sample])
