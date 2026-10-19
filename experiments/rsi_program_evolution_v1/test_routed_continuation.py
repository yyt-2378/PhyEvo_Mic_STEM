"""Selection estimates do not change scientific decisions or hard storage caps."""
from dataclasses import replace
import json

from . import routed_continuation as routed
from .field_storage import allocated_bytes
from .test_bounded_continuation import actual_runner
from .test_storage_continuation import small_policy
from .test_system_state import prepare_initial, proposal


def test_planning_is_read_only_and_keeps_unknown_branch_and_checkpoint_reserves(tmp_path):
    archive, runner = actual_runner(tmp_path)
    try:
        initial, _ = prepare_initial(runner)
        branch = {'name': 'b2', 'pipeline_code': 'def predict(s,x,c): return {}',
                  'training_steps': 2, 'parent_branch': None, 'replaces': None}
        p = proposal(initial, operation='add_branch', new_branch=branch, active_branches=['b0','b1','b2'])
        library = runner.frozen_state(initial)['library']
        tail, size = archive.verify(), allocated_bytes(tmp_path / 'pool')
        plan = routed.estimate(runner, p, library, small_policy())
        assert plan['checkpoint_bytes_bound'] == 256 * 1024**2
        assert plan['planned_prediction_bytes'] >= sum(s.fields['image'].size*5+16384 for s in runner.cohorts('development'))
        assert plan['prediction_bytes_bound'] > plan['planned_prediction_bytes']
        assert not plan['label_or_score_based_route_prediction']
        assert archive.verify() == tail and allocated_bytes(tmp_path/'pool') == size
    finally:
        archive.close()


def test_selected_existing_predictions_are_recomputed_but_add_no_payload(tmp_path):
    archive, runner = actual_runner(tmp_path)
    try:
        initial, _ = prepare_initial(runner)
        child = runner.register_system(proposal(initial), runner.frozen_state(initial))
        _, result = runner.execute(child)
        assert 'rows' in result, result
        trace = result['execution']['prediction_storage']['sharing']
        assert trace['computed_predictions'] == 2
        assert trace['new_prediction_payload_bytes'] == 0
        assert trace['inference_skipped'] is False
    finally:
        archive.close()


def test_mutating_router_is_still_rejected_by_original_numeric_contract(tmp_path):
    archive, runner = actual_runner(tmp_path)
    try:
        initial, _ = prepare_initial(runner)
        code = "def route(s,b,x,c):\n b['b0']['score'][:]=0\n return {'branch':'b0','reason':'mutation'}\n"
        child = runner.register_system(proposal(initial, router_code=code), runner.frozen_state(initial))
        _, result = runner.execute(child)
        assert 'changed a capability output' in result['error']
        assert archive.incumbent() == initial
    finally:
        archive.close()


def test_unknown_predictions_do_not_receive_zero_storage_reservation(tmp_path):
    archive, runner = actual_runner(tmp_path)
    try:
        initial, _ = prepare_initial(runner)
        branch = {'name': 'b2', 'pipeline_code': 'def predict(s,x,c): return {}',
                  'training_steps': 0, 'parent_branch': None, 'replaces': None}
        p = proposal(initial, operation='add_branch', new_branch=branch, active_branches=['b0','b1','b2'])
        policy = small_policy() | {'max_prediction_pool_bytes': allocated_bytes(runner.sandbox.prediction_pool.root) + 1}
        plan = routed.estimate(runner, p, runner.frozen_state(initial)['library'], policy)
        assert plan['planned_prediction_bytes'] > 0
        assert not plan['prediction_pool_fits_expected_replay']
        assert plan['original_worst_case_bounds_preserved_in_record']
    finally:
        archive.close()
