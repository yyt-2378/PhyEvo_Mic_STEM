"""Synthetic tests for whole-system inheritance, adoption and retained ability."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from .archive import Archive
from .contracts import Protocol
from .executor import Sample
from .runtime_compat import ConfiguredSandbox
from .system_state import RAW_PIPELINE, SystemExecutor, SystemProposal, joint_admission
from .system_study import checkpoint
from .test_core import ROOT


def test_system_request_serializer_is_bound():
    from . import system_study
    assert system_study.canonical({'z': 1, 'a': 2}) == '{"a":2,"z":1}'


def proposal(parent='a'*64, **changes):
    p = SystemProposal(parent, 'route_only', 'Observable disagreement', 'Image evidence distinguishes useful branches',
        'Select existing branches using input evidence', 'Whole-system score regresses', ['b0','b1'],
        "def route(sample,branches,state,context):\n    assert 'human' not in sample and 'known' not in sample\n    return {'branch':'b1' if sample['image'].mean()>.5 else 'b0','reason':'observed contrast'}\n",
        0, True, None)
    return replace(p, **changes)


def test_system_proposal_cannot_drop_base_or_invent_parent():
    library={'b0':{},'b1':{}}
    proposal().validate('a'*64,library)
    for p in (proposal(parent='b'*64),proposal(active_branches=['b1']),
              proposal(active_branches=['b0','b1','b1']),proposal(active_branches=['b0','b7'])):
        with pytest.raises(ValueError):p.validate('a'*64,library)


def test_new_algorithms_must_be_declared_and_optional_replacement_explicit():
    b={'name':'b2','pipeline_code':RAW_PIPELINE,'training_steps':0,'parent_branch':None,'replaces':None}
    proposal(operation='add_branch',new_branch=b,active_branches=['b0','b1','b2']).validate('a'*64,{'b0':{},'b1':{}})
    with pytest.raises(ValueError,match='conceal'):
        proposal(new_branch=b).validate('a'*64,{'b0':{},'b1':{}})
    with pytest.raises(ValueError,match='protected base'):
        proposal(operation='replace_branch',new_branch=dict(b,replaces='b0'),active_branches=['b0','b2']).validate('a'*64,{'b0':{},'b1':{}})


def rows(a,b):
    return [{'group':g,'case':g,'seed':23,'coverage':v,'selected':10,'budget':10,
             'metric':'known_positive_area_coverage_at_fixed_area'} for g,v in [('A',a),('B',b)]]


def test_average_improvement_cannot_purchase_lost_old_capability():
    p=Protocol(('A',),('B',),('C',),seeds=(23,),minimum_gain=.002,maximum_group_regression=.002)
    result=joint_admission(rows(.9,.49),rows(.5,.5),rows(.5,.5),p,{'A':.5,'B':.5})
    assert not result['accepted'] and result['retained_capability_regressions']==['B']
    assert joint_admission(rows(.6,.5),rows(.5,.5),rows(.5,.5),p,{'A':.5,'B':.5})['accepted']


def test_history_watermark_prevents_cumulative_small_losses():
    p=Protocol(('A',),('B',),('C',),seeds=(23,),maximum_group_regression=.002)
    result=joint_admission(rows(.6,.498),rows(.5,.499),rows(.5,.498),p,{'A':.5,'B':.501})
    assert not result['accepted']


def test_whole_system_needs_multiple_development_sources():
    p=Protocol(('A',),('B',),('C',),seeds=(23,))
    with pytest.raises(ValueError,match='two development'):
        joint_admission(rows(.6,.6)[:1],rows(.5,.5)[:1],rows(.5,.5)[:1],p,{'A':.5})


def fixture(tmp_path):
    p=Protocol(('A',),('B',),('C',),seeds=(23,))
    a=Archive(tmp_path/'archive',p,create=True)
    samples=[]
    for group,value in [('A',.1),('B',.9),('C',.3)]:
        image=np.full((8,8),value,np.float32); human=np.zeros((8,8),bool);human[2:4,2:4]=True
        fields={'image':image,'base':np.arange(64,dtype=np.float32).reshape(8,8),
                'valid':np.ones((8,8),bool),'support':np.ones((8,8),bool)}
        samples.append(Sample(group,group,'interface',fields,human,human.astype(float),np.empty((0,2))))
    runner=SystemExecutor(ROOT,a,ConfiguredSandbox(ROOT,ROOT/'.venvs/sam2',timeout=45),samples)
    return a,runner


def prepare_initial(runner):
    b0=runner.capability({'pipeline_code':RAW_PIPELINE,'training_steps':0,'parent_state':None})
    code="def predict(sample,state,context):\n    return {'score':sample['image'],'support':sample['valid']}\n"
    b1=runner.capability({'pipeline_code':code,'training_steps':0,'parent_state':None})
    identity=runner.register_initial({'b0':b0,'b1':b1})
    trial,result=runner.execute(identity)
    assert 'rows' in result,result
    runner.archive.select(identity,True,'initial',trial)
    return identity,result


def test_real_system_routes_without_labels_and_replays_reversed_order(tmp_path):
    a,r=fixture(tmp_path)
    try:
        initial,_=prepare_initial(r)
        child=r.register_system(proposal(initial),r.frozen_state(initial))
        trial,result=r.execute(child)
        assert 'rows' in result,result
        assert {v['case']:v['branch'] for v in result['rows']}=={'A':'b0','B':'b1'}
        _,replay=r.execute(child,replay=True)
        assert [v['case'] for v in replay['rows']]==['B','A']
        assert {v['case']:v['branch'] for v in replay['rows']}=={'A':'b0','B':'b1'}
        a.select(child,False,'synthetic rejection',trial)
        assert a.incumbent()==initial
        before=checkpoint(a,r)
        a.remember(child,trial,'Rejected but learned from the outcome','development')
        after=checkpoint(a,r)
        assert before['executable_system']==after['executable_system']
        assert before['research_checkpoint']!=after['research_checkpoint']
    finally:a.close()


def test_router_cannot_mutate_candidate_arrays(tmp_path):
    a,r=fixture(tmp_path)
    try:
        initial,_=prepare_initial(r)
        code="def route(sample,branches,state,context):\n    branches['b0']['score'][:]=0\n    return {'branch':'b0','reason':'mutation'}\n"
        child=r.register_system(proposal(initial,router_code=code),r.frozen_state(initial))
        _,result=r.execute(child)
        assert 'changed a capability output' in result['error']
        assert a.incumbent()==initial
    finally:a.close()


def test_audit_is_inaccessible_until_sealed(tmp_path):
    a,r=fixture(tmp_path)
    try:
        initial,_=prepare_initial(r)
        with pytest.raises(ValueError,match='sealed'):
            r.execute(initial,split='audit')
        a.event('search_sealed',checkpoint(a,r))
        _,result=r.execute(initial,split='audit')
        assert result['summary'].keys()=={'C'}
    finally:a.close()


def test_replacement_keeps_old_capability_bytes_and_binding(tmp_path):
    a,r=fixture(tmp_path)
    try:
        initial,_=prepare_initial(r); old=r.frozen_state(initial)
        b={'name':'b2','pipeline_code':RAW_PIPELINE,'training_steps':0,'parent_branch':None,'replaces':'b1'}
        child=r.register_system(proposal(initial,operation='replace_branch',new_branch=b,active_branches=['b0','b2'],
            router_code="def route(sample,branches,state,context):\n    return {'branch':'b2','reason':'replacement'}\n"),old)
        _,spec=r.system(child)
        assert set(spec['library'])=={'b0','b1','b2'}
        assert spec['library']['b1']==old['library']['b1']
        assert spec['active_branches']==['b0','b2']
    finally:a.close()


def test_numeric_student_checkpoint_is_retained_without_refitting(tmp_path):
    a,r=fixture(tmp_path)
    try:
        initial,_=prepare_initial(r)
        code='''def fit(samples,context):
    import numpy as np
    import torch
    assert len(samples)==1 and 'human' in samples[0]
    p=torch.nn.Parameter(torch.zeros(1))
    initial=p.detach().numpy().copy()
    optimizer=torch.optim.SGD([p],lr=.1)
    losses=[]
    for _ in range(context['training_steps']):
        optimizer.zero_grad(); loss=(p-1).square().sum();loss.backward();optimizer.step()
        losses.append(float(loss.detach()))
    final=p.detach().numpy().copy()
    return {'w':final,'audit_initial_parameters':initial,'audit_final_parameters':final,
            'audit_loss_trace':np.asarray(losses),'audit_steps':np.asarray([context['training_steps']])}
def predict(sample,state,context):
    assert 'human' not in sample
    return {'score':sample['base']+state['w'][0],'support':sample['valid']}
'''
        b={'name':'b2','pipeline_code':code,'training_steps':2,'parent_branch':None,'replaces':None}
        p=proposal(initial,operation='add_branch',new_branch=b,active_branches=['b0','b1','b2'],
            router_code="def route(sample,branches,state,context):\n    return {'branch':'b2','reason':'trained branch'}\n")
        child=r.register_system(p,r.frozen_state(initial))
        trial,result=r.execute(child);assert 'rows' in result,result
        a.select(child,True,'synthetic acceptance',trial)
        state=r.frozen_state(child)
        cap_path=r.checked_ref(state['library']['b2'])
        import json
        cap=json.loads(cap_path.read_text());weight=r.checked_ref(cap['checkpoint']);before=weight.read_bytes()
        assert cap['training']['trace']['steps']==2 and cap['weight_training_groups']==['A']
        descendant=r.register_system(proposal(child,active_branches=['b0','b1','b2']),state)
        _,next_state=r.system(descendant)
        assert next_state['library']['b2']==state['library']['b2']
        assert weight.read_bytes()==before
    finally:a.close()


def test_registry_detects_router_change(tmp_path):
    a,r=fixture(tmp_path)
    try:
        initial,_=prepare_initial(r)
        folder,_=r.system(initial)
        with (folder/'router.py').open('a') as stream:stream.write('\n# deliberate synthetic tamper\n')
        with pytest.raises(ValueError,match='router changed'):r.system(initial)
    finally:a.close()
