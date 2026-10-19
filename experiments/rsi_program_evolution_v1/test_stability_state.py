"""Actual sandbox probes do not train, admit, or replace validation evidence."""
from copy import deepcopy

import numpy as np
import pytest

from .contracts import file_hash
from .expansion_state import CompactSystemExecutor, array_digest
from .stability_state import StabilitySystemExecutor, stability_admission
from .system_study import latest
from .test_system_state import fixture, prepare_initial, proposal, rows


def gate_inputs():
    original = [dict(r, denominator=10) for r in rows(.7, .6)]
    moved = [dict(r, case='rot90-' + r['case']) for r in original]
    probe = {'transform': 'rot90', 'rows': moved, 'actual_execution': True,
             'execution': {'status': 'ok'}, 'router_training': None}
    return ({'accepted': True}, original, {'rot90': probe}, ('rot90',), .002,
            {'rot90': {'A': .5, 'B': .5}})


def test_stability_conjoins_not_replaces_gain_and_history_retention():
    args = gate_inputs()
    assert stability_admission(*args)['accepted']
    args[0]['accepted'] = False
    assert not stability_admission(*args)['accepted']
    args = gate_inputs(); args[2]['rot90']['rows'][0]['coverage'] = .65
    result = stability_admission(*args)
    assert not result['accepted'] and result['stability_regressions'][0]['group'] == 'A'
    args = gate_inputs(); args[-1]['rot90']['B'] = .65
    assert not stability_admission(*args)['accepted']


def test_missing_failed_or_refitted_probe_cannot_pass():
    args = gate_inputs(); args[2].clear()
    with pytest.raises(ValueError, match='Every declared'):
        stability_admission(*args)
    args = gate_inputs(); args[2]['rot90'] = {'error': 'timeout', 'actual_execution': True}
    result = stability_admission(*args)
    assert not result['accepted'] and result['failed_executions'] == ['rot90']
    args = gate_inputs(); args[2]['rot90']['router_training'] = {'steps': 1}
    with pytest.raises(ValueError, match='without refitting'):
        stability_admission(*args)


def test_cohort_and_budget_mismatch_is_not_tolerated():
    for field, value in (('budget', 1000), ('denominator', 200), ('metric', 'accuracy'), ('case', 'wrong')):
        args = gate_inputs(); args[2]['rot90']['rows'][0][field] = value
        with pytest.raises(ValueError):
            stability_admission(*args)


def test_actual_frozen_stability_runs_are_separate_from_fit_and_validation(tmp_path):
    archive, old = fixture(tmp_path)
    runner = StabilitySystemExecutor(old.workspace, archive, old.sandbox, old.samples,
                                     tmp_path / 'shared_inputs', ('rot90', 'flip_lr'))
    try:
        initial, _ = prepare_initial(runner)
        original = latest(archive, initial)
        before = deepcopy(runner.frozen_state(initial))
        trial, receipt = runner.probe_system(initial, 'rot90')
        assert 'rows' in receipt, receipt
        assert receipt['router_training'] is None and receipt['router_checkpoint'] == before['router_checkpoint']
        assert receipt['execution']['isolation'].startswith('bubblewrap')
        assert all(r['case'].startswith('rot90-') for r in receipt['rows'])
        assert {r['group'] for r in receipt['rows']} == {'A', 'B'}  # C audit is excluded
        assert archive.incumbent() == initial and runner.frozen_state(initial) == before
        assert latest(archive, initial) == original
        assert runner.probe_system(initial, 'rot90') == (trial, receipt)
        assert len(archive.events('development_stability_measured')) == 1
        with pytest.raises(ValueError, match='not declared'):
            runner.probe_system(initial, 'rot180')
        # Training membership did not grow to include transformed cases.
        assert [s.key for s in runner.samples if runner.protocol.split(s.group) == 'train'] == ['A']
        # Frozen input fields really were transformed, rather than copying old scores.
        _, spec = runner.system(initial)
        cap = runner.checked_ref(spec['library']['b0']).parent
        with np.load(cap / 'stability-rot90/input/sample-0000.npz', allow_pickle=False) as z:
            assert np.array_equal(z['base'], np.rot90(runner.samples[0].fields['base']))
            assert 'human' not in z and 'known' not in z
        archive.verify()
    finally:
        archive.close()


def test_trained_router_state_is_inherited_without_probe_refitting(tmp_path):
    archive, old = fixture(tmp_path)
    runner = StabilitySystemExecutor(old.workspace, archive, old.sandbox, old.samples,
                                     tmp_path / 'shared_inputs', ('rot90',))
    try:
        initial, _ = prepare_initial(runner)
        code = '''import numpy as np
def fit(samples,context):
    assert len(samples)==1 and float(samples[0]['image'].mean()) < .5
    assert 'human' in samples[0] and 'known' in samples[0]
    steps=context['training_steps']
    w=np.array([1.],dtype=np.float64); start=w.copy(); losses=[]
    for _ in range(steps):
        losses.append(float(w[0]**2)); w-=.1*2*w
    return {'parameter':w,'audit_initial_parameters':start,'audit_final_parameters':w.copy(),
            'audit_loss_trace':np.array(losses),'audit_steps':np.array([steps])}
def route(sample,branches,state,context):
    assert 'human' not in sample and 'known' not in sample
    assert np.allclose(state['parameter'],[.64])
    return {'branch':'b0','reason':'synthetic frozen learned parameter'}
'''
        child = runner.register_system(proposal(initial, router_code=code, router_training_steps=2),
                                       runner.frozen_state(initial))
        _, measured = runner.execute(child)
        assert 'rows' in measured, measured
        before = runner.frozen_state(child)
        state_path = runner.checked_ref(before['router_checkpoint'])
        before_hash = file_hash(state_path)
        _, probed = runner.probe_system(child, 'rot90')
        assert 'rows' in probed, probed
        assert probed['router_training'] is None
        assert probed['router_checkpoint'] == before['router_checkpoint']
        assert runner.frozen_state(child) == before and file_hash(state_path) == before_hash
        assert archive.incumbent() == initial  # Neither fit nor stability execution admits.
    finally:
        archive.close()


def test_prior_input_pool_reuses_bytes_without_modifying_source_or_skipping_execution(tmp_path):
    archive, old = fixture(tmp_path)
    source = CompactSystemExecutor(old.workspace, archive, old.sandbox, old.samples, tmp_path / 'old_pool')
    runner = StabilitySystemExecutor(old.workspace, archive, old.sandbox, old.samples,
        tmp_path / 'new_pool', ('rot90',), readonly_input_pools=(source.pool,))
    try:
        sample = old.samples[0]
        source.stage(tmp_path / 'old_inputs', [sample])
        before = {p.name: file_hash(p) for p in source.pool.iterdir() if p.is_file()}
        runner.stage(tmp_path / 'new_inputs', [sample])
        old_bundle = tmp_path / 'old_inputs/sample-0000.npz'
        new_bundle = tmp_path / 'new_inputs/sample-0000.npz'
        assert old_bundle.stat().st_ino == new_bundle.stat().st_ino
        assert file_hash(old_bundle) == file_hash(new_bundle)
        assert {p.name: file_hash(p) for p in source.pool.iterdir() if p.is_file()} == before
        key = array_digest(sample.fields)
        from .system_state import read
        assert not read(runner.pool / (key + '.reuse.json'))['predictions_reused']
        with np.load(new_bundle, allow_pickle=False) as z:
            assert 'human' not in z and 'known' not in z
        initial, receipt = prepare_initial(runner)
        assert receipt['actual_execution'] and receipt['execution']['status'] == 'ok'
        assert receipt['execution']['isolation'].startswith('bubblewrap')
        _, probe = runner.probe_system(initial, 'rot90')
        assert probe['actual_execution'] and probe['execution']['status'] == 'ok'
        assert {p.name: file_hash(p) for p in source.pool.iterdir() if p.is_file()} == before
    finally:
        archive.close()
