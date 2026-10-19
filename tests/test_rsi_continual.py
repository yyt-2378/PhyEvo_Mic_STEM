from copy import deepcopy
import json

import numpy as np
import pytest

from tfree_detection.expert_student import make_expert_student, restore_checkpoint, save_checkpoint, torch
from tfree_detection.rsi_continual import (
    FeedbackSampler, error_state, make_partitions, materialize, retention_gate, validate_partitions,
)
from tfree_detection.rsi_refinement import POLICIES, refinement_loss


def fixtures():
    entries, arrays, records = [], {}, {}
    for n in range(1, 11):
        shape = (512, 512)
        human = np.zeros(shape, np.float32)
        human[79:82, 79:82] = 1
        human[429:432, 429:432] = 1
        e = {'case_number': n, 'group': 'single' if n == 1 else ('outer' if n == 2 else 'sequence'),
             'geometry': 'point', 'geometry_index': 1, 'shape': list(shape),
             'action': 'correct_geometry', 'human_positive_pixels': 18, 'human_known_pixels': 18,
             'teacher_positive_pixels': 18, 'points_xy': [[80., 80.], [430., 430.]]}
        entries.append(e)
        arrays[n] = {'image': np.random.default_rng(n).random(shape).astype(np.float32),
                     'human': human, 'human_weight': human.copy(), 'teacher': human.copy(),
                     'teacher_weight': np.ones(shape, np.float32)}
        records[n] = {'prediction': {'candidates': [{'centroid_xy': [80., 80.], 'reliability': .9},
                                                  {'centroid_xy': [430., 430.], 'reliability': .8}]}}
    return entries, arrays, records


def test_order_split_no_outer_or_control_overlap_and_guard():
    entries, _, _ = fixtures()
    part = make_partitions(entries, 'outer')
    assert part['outer'] == [{'case_number': 2, 'rows': [0, 512]}]
    assert part['train'][0] == {'case_number': 3, 'rows': [0, 512]}  # sorted group order
    a = next(x for x in part['train'] if x['case_number'] == 1)
    b = next(x for x in part['control'] if x['case_number'] == 1)
    assert b['rows'][0]-a['rows'][1] == 128
    assert part['unused_gap_cases'] == [7, 8]
    assert not part['control_is_independent_acquisition']
    bad = deepcopy(part); bad['train'].append(part['outer'][0])
    with pytest.raises(ValueError, match='Outer acquisition'):
        validate_partitions(entries, bad)
    bad = deepcopy(part)
    next(x for x in bad['control'] if x['case_number'] == 1)['rows'][0] = 0
    with pytest.raises(ValueError, match='overlap'):
        validate_partitions(entries, bad)


def test_crop_coordinates_and_no_input_mutation():
    entries, arrays, records = fixtures()
    before = arrays[1]['human'].copy()
    selected, planes, candidates = materialize(entries, arrays, records, [{'case_number': 1, 'rows': [320, 512]}])
    assert selected[0]['points_xy'] == [[430., 110.]]
    assert candidates[1][0]['centroid_xy'] == [430., 110.]
    assert planes[1]['image'].shape == (192, 512)
    assert selected[0]['human_positive_pixels'] == 9
    planes[1]['human'][:] = 0
    np.testing.assert_array_equal(before, arrays[1]['human'])
    assert entries[0]['points_xy'] == [[80., 80.], [430., 430.]]
    assert records[1]['prediction']['candidates'][1]['centroid_xy'] == [430., 430.]


def test_train_side_error_feedback_and_forgetting_are_bounded():
    entries, arrays, _ = fixtures()
    scores = np.zeros((512, 512), np.float32)
    scores[80, 80] = 1
    state, hard = error_state(entries[0], arrays[1], scores, previous_best=1.)
    assert state['coverage'] == .5
    assert state['forgetting'] == .5
    assert state['best_coverage'] == 1
    assert 1 <= state['weight'] <= 5
    assert hard[430, 430] and not hard[80, 80]
    assert not np.any(hard & (arrays[1]['human'] == 0))


@pytest.mark.skipif(torch is None, reason='optional torch unavailable')
def test_feedback_changes_next_sampling_but_first_round_is_matched():
    entries, arrays, _ = fixtures()
    a, b = FeedbackSampler(entries, arrays, enabled=False), FeedbackSampler(entries, arrays, enabled=True)
    def sample(s):
        return s.sample(np.random.default_rng(11), batch_size=32, patch_size=32,
                        device='cpu', augmentation_seed=13)
    aa, ar = sample(a); bb, br = sample(b)
    assert ar == br
    for k in aa:
        assert torch.equal(aa[k], bb[k])
    state = {str(e['case_number']): {'weight': 5 if e['case_number'] == 1 else 1} for e in entries}
    hard = {n: a['human'] > 0 for n, a in arrays.items()}
    active = FeedbackSampler(entries, arrays, state=state, hard_masks=hard)
    inactive = FeedbackSampler(entries, arrays, state=state, hard_masks=hard, enabled=False)
    assert sample(active)[1] != ar
    assert sample(inactive)[1] == ar
    assert set(active.audit['by_case']) <= {str(e['case_number']) for e in entries}
    for _, probabilities in active.sources.values():
        assert (probabilities > 0).all()
        assert probabilities.sum() == pytest.approx(1)
    with pytest.raises(ValueError, match='Non-training'):
        FeedbackSampler(entries, arrays, state={**state, '999': {'weight': 1}}, hard_masks=hard)
    hard[1] = np.ones((512, 512), bool)
    with pytest.raises(ValueError, match='subset'):
        FeedbackSampler(entries, arrays, state=state, hard_masks=hard)


def control_rows(student=(.4, .4), hybrid=(.35, .4)):
    return [{'geometry': 'point', 'group': group, 'main_comparison_eligible': True,
             'methods': {k: {'positive_coverage': v} for k, v in
                         [('physics_budgeted', .1), ('student_expert', student[i]),
                          ('student_reranked_physics_plus_rescue', hybrid[i])]}}
            for i, group in enumerate(('a', 'b'))]


def test_gate_accepts_then_retains_old_checkpoint_on_regression():
    previous, decisions = retention_gate(control_rows(), {}, 1)
    assert previous['point']['round'] == 1 and decisions[0]['accepted']
    result, decisions = retention_gate(control_rows((.6, .2), (.3, .3)), previous, 2)
    assert result == previous and not decisions[0]['accepted']
    result, decisions = retention_gate(control_rows((.5, .5)), previous, 3)
    assert result['point']['round'] == 3 and decisions[0]['accepted']
    # No outer outcomes/case names are accepted by the gate API.
    with pytest.raises(TypeError):
        retention_gate(control_rows(), previous, 4, outer_results=[1.])


def test_gate_rejects_control_inventory_change():
    incumbent, _ = retention_gate(control_rows(), {}, 1)
    with pytest.raises(ValueError, match='inventory'):
        retention_gate(control_rows()[:1], incumbent, 2)


@pytest.mark.skipif(torch is None, reason='optional torch unavailable')
def test_feedback_checkpoint_resume_next_update_exact(tmp_path):
    torch.set_num_threads(1)
    entries, arrays, _ = fixtures()
    entries, arrays = entries[:1], {1: arrays[1]}
    state = {'1': {'weight': 3}}
    hard = {1: arrays[1]['human'] > 0}
    sampler = FeedbackSampler(entries, arrays, state=state, hard_masks=hard)
    torch.manual_seed(23)
    model = make_expert_student(4)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=.001)
    rng = np.random.default_rng(23)
    def step(model, opt, rng):
        model.train()
        data, records = sampler.sample(rng, batch_size=2, patch_size=32, device='cpu', augmentation_seed=99)
        opt.zero_grad(set_to_none=True)
        loss, _ = refinement_loss(model(data['image']), data, POLICIES['photometric'])
        loss.backward(); opt.step()
        return records
    step(model, opt, rng)
    save_checkpoint(tmp_path/'checkpoint', model, opt, rng, {'base_channels': 4, 'learning_rate': .001,
                                                           'parent_feedback_sha256': 'test-parent'})
    expected = step(model, opt, rng)
    restored, opt2, rng2, meta = restore_checkpoint(tmp_path/'checkpoint', resume=True)
    assert meta['parent_feedback_sha256'] == 'test-parent'
    assert step(restored, opt2, rng2) == expected
    for k, v in model.state_dict().items():
        assert torch.equal(v, restored.state_dict()[k]), k
    with pytest.raises(FileExistsError):
        save_checkpoint(tmp_path/'checkpoint', model, opt, rng, {})


def test_sealed_artifact_detects_changed_input(tmp_path, monkeypatch):
    from scripts import run_rsi_continual as runner
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    original = tmp_path/'input.json'; original.write_text('{"v":1}')
    target = tmp_path/'result.json'
    runner.save_sealed(target, {'value': 2}, [original])
    assert runner.verified(target) == {'value': 2}
    original.write_text('{"v":3}')
    with pytest.raises(ValueError, match='Changed input'):
        runner.verified(target)
