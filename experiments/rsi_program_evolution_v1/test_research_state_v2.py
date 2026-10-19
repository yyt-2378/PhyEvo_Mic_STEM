"""Research accumulation is not deployment; a diagnosis is not an operation."""
from dataclasses import asdict

import numpy as np
import pytest

from .contracts import file_hash
from .research_state_v2 import MASK_FEATURE_CONTRACT, ResearchBank, mask_observables, parse_action, response_schema
from .system_state import RAW_PIPELINE
from .test_system_state import fixture, prepare_initial, proposal


def answer(action='route_existing', **kwargs):
    system = asdict(proposal(**kwargs))
    del system['operation']
    return {'diagnosis': {'kind': 'capability_gap', 'evidence': 'Old-library deficit',
            'alternatives': 'Routing can still recover some already measured capability',
            'new_capacity_test': 'Compare new analysis to the frozen library under the same budget'},
        'action': action, 'system': system}


def test_capability_diagnosis_does_not_invalidate_a_routing_control():
    value = answer(active_branches=['b0', 'b2'])
    parsed, diagnosis = parse_action(value, 'route', 'a'*64, {'b0': {}, 'b1': {}, 'b2': {}})
    assert diagnosis['kind'] == 'capability_gap'
    assert parsed.operation == 'route_only' and parsed.active_branches == ['b0', 'b2']
    assert response_schema('route')['properties']['action']['enum'] == ['route_existing']
    assert 'operation' not in response_schema('full')['properties']['system']['properties']
    # No repair of a second, inconsistent operation/target field.
    value['system']['operation'] = 'route_only'
    with pytest.raises(ValueError, match='action alone'):
        parse_action(value, 'route', 'a'*64, {'b0': {}, 'b1': {}, 'b2': {}})


def test_student_is_optional_but_when_selected_must_actually_train():
    branch = {'name': 'b2', 'pipeline_code': RAW_PIPELINE, 'training_steps': 0,
              'parent_branch': None, 'replaces': None}
    value = answer('create_analysis', new_branch=branch, active_branches=['b0', 'b1', 'b2'])
    parse_action(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})
    with pytest.raises(ValueError, match='unavailable'):
        parse_action(value, 'route', 'a'*64, {'b0': {}, 'b1': {}})
    value['action'] = 'train_student'
    with pytest.raises(ValueError, match='must train'):
        parse_action(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})
    branch = dict(branch, pipeline_code=RAW_PIPELINE + 'def fit(samples,context):\n return {}\n', training_steps=32)
    value = answer('train_student', new_branch=branch, active_branches=['b0', 'b1', 'b2'])
    parse_action(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})
    value['action'] = 'create_analysis'
    with pytest.raises(ValueError, match='deterministic'):
        parse_action(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})


def test_new_alias_and_joint_training_budget_are_unambiguous():
    code = RAW_PIPELINE + 'def fit(samples,context):\n return {}\n'
    branch = {'name': 'b2', 'pipeline_code': code, 'training_steps': 3000,
              'parent_branch': None, 'replaces': None}
    router = 'def route(s,b,x,c):\n return {}\ndef fit(s,c):\n return {}\n'
    value = answer('train_student', new_branch=branch, active_branches=['b0', 'b1', 'b2'],
                   router_code=router, router_training_steps=2000)
    with pytest.raises(ValueError, match='Combined'):
        parse_action(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})
    value = answer('create_analysis', new_branch=dict(branch, name='b7', training_steps=0),
                   active_branches=['b0', 'b1', 'b7'])
    with pytest.raises(ValueError, match='reserved bank alias'):
        parse_action(value, 'full', 'a'*64, {'b0': {}, 'b1': {}})


def test_mask_statistics_distinguish_global_and_conditional_fractions():
    valid = np.ones((4, 4), bool)
    support = np.zeros((4, 4), bool)
    support[0] = True
    fields = {'valid': valid, 'support': support, 'physics_valid': support.copy()}
    values = mask_observables(fields)
    assert values['physical_valid_fraction_of_image'] == .25
    assert values['physical_valid_fraction_within_valid'] == .25
    assert values['physical_valid_fraction_within_support'] == 1.
    assert set(values) == set(MASK_FEATURE_CONTRACT)
    # A global threshold of 0.66 would select the opposite regime if the
    # generated program silently replaced its denominator with support size.
    assert (values['physical_valid_fraction_of_image'] < .66) != (
        values['physical_valid_fraction_within_support'] < .66)


def test_empty_condition_is_unavailable_and_mask_alignment_is_checked():
    empty = np.zeros((4, 4), bool)
    values = mask_observables({'valid': empty, 'support': empty, 'physics_valid': empty})
    assert values['physical_valid_fraction_within_valid'] is None
    assert values['physical_valid_fraction_within_support'] is None
    assert values['physical_valid_fraction_of_image'] == 0.
    with pytest.raises(ValueError, match='Aligned'):
        mask_observables({'valid': empty, 'support': empty[:, :2], 'physics_valid': empty})


def test_rejected_capability_is_reusable_without_changing_the_incumbent(tmp_path):
    archive, runner = fixture(tmp_path)
    try:
        initial, _ = prepare_initial(runner)
        bank = ResearchBank(runner)
        assert bank.initialize() == ['b0', 'b1']
        before = bank.checkpoint()
        code = "def predict(sample,state,context):\n    return {'score':-sample['base'],'support':sample['valid']}\n"
        branch = {'name': 'b2', 'pipeline_code': code, 'training_steps': 0,
                  'parent_branch': None, 'replaces': None}
        child = runner.register_system(proposal(initial, operation='add_branch', new_branch=branch,
            active_branches=['b0', 'b1', 'b2']), bank.inherited_manifest())
        trial, measured = runner.execute(child)
        assert 'rows' in measured, measured
        archive.select(child, False, 'synthetic rejection of whole system', trial)
        assert bank.retain_measured(child) == ['b2']
        assert bank.retain_measured(child) == []
        after = bank.checkpoint()
        assert archive.incumbent() == initial
        assert before['system']['executable_system'] == after['system']['executable_system']
        assert before['research_checkpoint'] != after['research_checkpoint']
        assert bank.catalog()['b2']['pipeline_code'] == code
        assert not bank.catalog()['b2']['active_in_incumbent']
        # A later proposal explicitly adopts the retained branch; no code rewriting.
        router = "def route(s,b,x,c):\n return {'branch':'b2','reason':'synthetic reuse test'}\n"
        value = answer(parent=initial, active_branches=['b0', 'b2'], router_code=router)
        parsed, _ = parse_action(value, 'full', initial, bank.inherited_manifest()['library'])
        adopted = runner.register_system(parsed, bank.inherited_manifest())
        _, result = runner.execute(adopted)
        assert 'rows' in result and set(result['choices']) == {'b2'}, result
        assert archive.incumbent() == initial  # Execution still does not grant admission.
        original = runner.predict_capability(bank.entries()['b2']['binding'], 'development')
        for key, ref in result['predictions'].items():
            with np.load(runner.checked_ref(ref), allow_pickle=False) as actual:
                with np.load(runner.checked_ref(original['predictions'][key]), allow_pickle=False) as expected:
                    assert np.array_equal(actual['score'], expected['score'])
        archive.verify()
    finally:
        archive.close()


def test_bank_rejects_modified_code_and_never_silently_reinitializes(tmp_path):
    archive, runner = fixture(tmp_path)
    try:
        prepare_initial(runner)
        bank = ResearchBank(runner)
        bank.initialize()
        with pytest.raises(FileExistsError):
            bank.initialize()
        entry = bank.entries()['b0']
        from .system_state import read
        cap = read(runner.checked_ref(entry['binding']))
        program = runner.checked_ref(cap['program'])
        old_hash = file_hash(program)
        with program.open('a') as stream:
            stream.write('\n# synthetic corruption\n')
        assert file_hash(program) != old_hash
        with pytest.raises(ValueError, match='changed'):
            bank.checkpoint()
    finally:
        archive.close()


def test_unadmitted_student_weights_can_seed_a_later_learning_action(tmp_path):
    archive, runner = fixture(tmp_path)
    try:
        initial, _ = prepare_initial(runner)
        bank = ResearchBank(runner)
        bank.initialize()
        code = '''def fit(samples,context):
    import numpy as np
    import torch
    assert len(samples)==1 and 'human' in samples[0]
    value=context.get('parent_state',{}).get('w',np.zeros(1,dtype=np.float32))
    p=torch.nn.Parameter(torch.tensor(value,dtype=torch.float32))
    before=p.detach().numpy().copy()
    optimizer=torch.optim.SGD([p],lr=.1)
    losses=[]
    for _ in range(context['training_steps']):
        optimizer.zero_grad(); loss=(p-1).square().sum();loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    after=p.detach().numpy().copy()
    return {'w':after,'audit_initial_parameters':before,'audit_final_parameters':after,
        'audit_loss_trace':np.asarray(losses),'audit_steps':np.asarray([context['training_steps']])}
def predict(sample,state,context):
    assert 'human' not in sample and 'known' not in sample
    return {'score':sample['base']*state['w'][0],'support':sample['valid']}
'''
        def learned(name, parent_branch):
            branch = {'name': name, 'pipeline_code': code, 'training_steps': 2,
                      'parent_branch': parent_branch, 'replaces': None}
            raw = answer('train_student', parent=initial, new_branch=branch,
                active_branches=['b0', name],
                router_code=f"def route(s,b,x,c):\n return {{'branch':'{name}','reason':'synthetic learned reuse'}}\n")
            manifest = bank.inherited_manifest()
            parsed, _ = parse_action(raw, 'full', initial, manifest['library'])
            candidate = runner.register_system(parsed, manifest)
            trial, result = runner.execute(candidate)
            assert 'rows' in result, result
            archive.select(candidate, False, 'synthetic whole-system rejection', trial)
            bank.retain_measured(candidate)
            from .system_state import read
            cap = read(runner.checked_ref(bank.entries()[name]['binding']))
            assert cap['training']['trace']['steps'] == 2 and cap['weight_training_groups'] == ['A']
            return cap
        first = learned('b2', None)
        weights = runner.checked_ref(first['checkpoint'])
        before_hash = file_hash(weights)
        second = learned('b3', 'b2')
        with np.load(weights, allow_pickle=False) as a:
            with np.load(runner.checked_ref(second['checkpoint']), allow_pickle=False) as b:
                assert np.array_equal(a['audit_final_parameters'], b['audit_initial_parameters'])
                assert not np.array_equal(a['w'], b['w'])
        assert file_hash(weights) == before_hash
        assert archive.incumbent() == initial
    finally:
        archive.close()


def test_failed_prediction_does_not_become_a_reusable_bank_entry(tmp_path):
    archive, runner = fixture(tmp_path)
    try:
        initial, _ = prepare_initial(runner)
        bank = ResearchBank(runner)
        bank.initialize()
        before = bank.checkpoint()
        branch = {'name': 'b2', 'pipeline_code': 'def predict(s,x,c):\n raise RuntimeError("synthetic failure")\n',
                  'training_steps': 0, 'parent_branch': None, 'replaces': None}
        child = runner.register_system(proposal(initial, operation='add_branch', new_branch=branch,
            active_branches=['b0', 'b1', 'b2']), bank.inherited_manifest())
        _, result = runner.execute(child)
        assert 'synthetic failure' in result['error']
        assert bank.retain_measured(child) == []
        assert bank.checkpoint() == before and 'b2' not in bank.catalog()
        assert archive.incumbent() == initial
    finally:
        archive.close()
