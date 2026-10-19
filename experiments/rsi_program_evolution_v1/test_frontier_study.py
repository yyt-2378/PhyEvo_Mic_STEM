"""Regression checks for the new experiment's interpretation, not synthetic gains."""
import pytest
from .frontier_study import frontier_gain, mean_coverage, ancestry_order, TRAJECTORIES


def test_recovering_a_better_old_candidate_is_not_new_frontier_gain():
    result=frontier_gain(.38,.35,.39,.40)
    assert result['relative_to_raw_start']==pytest.approx(.03)
    assert result['relative_to_frozen_library_best']==pytest.approx(-.01)
    assert result['relative_to_all_pre_round_candidates']==pytest.approx(-.02)


def test_gain_is_not_clipped_to_disguise_regression():
    assert frontier_gain(.20,.35,.39,.40)['relative_to_raw_start']==pytest.approx(-.15)


def test_mixed_geometry_metrics_are_not_pooled_as_accuracy():
    with pytest.raises(ValueError,match='different task metrics'):
        mean_coverage({'summary':{'A':{'metric':'point','mean':.8},'B':{'metric':'area','mean':.2}}})


def test_three_preregistered_source_rotations_not_three_starts_on_one_source():
    assert TRAJECTORIES=={'t01':24,'t02':6,'t03':12}


def test_json_key_sorting_does_not_import_descendants_before_parents():
    values={'a-child':{'parents':['z-parent']},'z-parent':{'parents':[]}}
    assert [k for k,v in ancestry_order(values)]==['z-parent','a-child']


def test_missing_parent_is_not_silently_reparented():
    with pytest.raises(ValueError,match='Missing parent'):
        ancestry_order({'a':{'parents':['unknown']}})


def test_cyclic_ancestry_is_rejected():
    with pytest.raises(ValueError,match='cyclic'):
        ancestry_order({'a':{'parents':['b']},'b':{'parents':['a']}})


def test_report_escapes_model_supplied_text():
    from .frontier_report import table
    result=table(['<heading>'],[['<script>bad</script>']])
    assert '<script>' not in result
    assert '&lt;script&gt;' in result and '&lt;heading&gt;' in result


def test_report_refuses_changed_frozen_runner(tmp_path):
    from .contracts import write_new
    from .frontier_report import verify_evidence
    write_new(tmp_path/'pilot_protocol.json',{'study_runner_sha256':'0'*64})
    with pytest.raises(ValueError,match='Frozen pilot runner'):
        verify_evidence(tmp_path)
