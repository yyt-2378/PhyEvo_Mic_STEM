"""Synthetic actual-sandbox tests; never open real images or labels."""
import json
from pathlib import Path

import numpy as np
import pytest

from .composition_study import ENTRY, ROOT, domain_signal_profile, error_signal_profiles, stage, verified_result
from .composition_study import training_supervision_summary, fitted_state_summary
from .contracts import Protocol, write_new
from .executor import Sample
from .retained_portfolio import WRAPPER
from .runtime_compat import ConfiguredSandbox


def put(path, text):
    with path.open('x') as stream:
        stream.write(text)


def fixture(tmp_path, nodes, current, states=None):
    program=tmp_path/'program';program.mkdir()
    put(program/'pipeline.py',ENTRY)
    put(program/'capabilities.py',Path(__file__).with_name('composition_runtime.py').read_text())
    put(program/'retained.py',WRAPPER)
    put(program/'b0.py',"import numpy as np\ndef predict(s,w,c):\n return {'score':s['image'], 'support':s['valid'].astype(bool)}\n")
    put(program/'router.py',"def route(s,b,w,c):\n return {'branch':'b0','reason':'synthetic'}\n")
    write_new(program/'portfolio.json',{'active_branches':['b0'],
        'branches':{'b0':{'program':'b0.py','checkpoint':None}},'router_checkpoint':None})
    table={}
    for name,parent,code,arm in nodes:
        put(program/(name+'.py'),code)
        checkpoint=None
        if states and name in states:
            checkpoint=name+'.npz'
            with (program/checkpoint).open('xb') as stream:
                np.savez_compressed(stream,**states[name])
        table[name]={'parent':parent,'program':name+'.py','checkpoint':checkpoint,'arm':arm}
    write_new(program/'composition.json',{'current':current,'nodes':table})
    return program


def run(tmp_path, program, images):
    inputs=tmp_path/'input';inputs.mkdir()
    records=[]
    for i,image in enumerate(images):
        name=f'sample-{i:04d}.npz'
        with (inputs/name).open('xb') as stream:
            np.savez_compressed(stream,image=image,valid=np.ones(image.shape,bool))
        records.append({'file':name})
    write_new(inputs/'request.json',{'mode':'predict','seed':41,'samples':records,
              'state_file':False,'training_steps':0,'device':'cpu'})
    output=tmp_path/'output'
    receipt=ConfiguredSandbox(ROOT,ROOT/'.venvs/sam2',timeout=60).run(program,inputs,output)
    return output,receipt


def test_recursive_parent_is_scoped_to_executing_node(tmp_path):
    code="import capabilities\ndef predict(s,w,c):\n p=capabilities.parent(s,c)\n return {'score':p['score']+AMOUNT,'support':p['support']}\n"
    program=fixture(tmp_path,[('a','frozen',code.replace('AMOUNT','1'),'full'),
                              ('b','a',code.replace('AMOUNT','2'),'full')],'b')
    image=np.arange(35,dtype=np.float32).reshape(5,7)
    output,receipt=run(tmp_path,program,[image,np.rot90(image)])
    assert receipt['status']=='ok',(output/'stderr.log').read_text()
    for i,expected in enumerate([image+3,np.rot90(image)+3]):
        with np.load(output/f'prediction-{i:04d}.npz') as z:
            assert np.array_equal(z['score'],expected)
        trace=json.loads((output/f'composition-trace-{i:04d}.json').read_text())['calls']
        assert trace==[{'call':'parent','node':'a'},{'call':'parent','node':'frozen'},
                       {'call':'frozen_best','branch':'b0'}]


def test_route_control_cannot_replace_branch_output(tmp_path):
    code="def route(s,w,c):\n return {'branch':'b0','reason':'unchanged','score':s['image']*0}\n"
    program=fixture(tmp_path,[('a','frozen',code,'route')],'a')
    image=np.arange(12,dtype=np.float32).reshape(3,4)
    output,receipt=run(tmp_path,program,[image])
    assert receipt['status']=='ok',(output/'stderr.log').read_text()
    with np.load(output/'prediction-0000.npz') as z:
        assert np.array_equal(z['score'],image)


def test_cyclic_parent_fails_in_actual_worker(tmp_path):
    code="import capabilities\ndef predict(s,w,c):\n return capabilities.parent(s,c)\n"
    program=fixture(tmp_path,[('a','a',code,'full')],'a')
    output,receipt=run(tmp_path,program,[np.ones((3,4),np.float32)])
    assert receipt['status']=='failed'
    assert 'Cyclic or excessive' in (output/'stderr.log').read_text()
    assert not (output/'prediction-0000.npz').exists()


def test_supervision_cannot_be_passed_to_parent(tmp_path):
    code="import capabilities\ndef predict(s,w,c):\n return capabilities.parent(dict(s,human=s['valid']),c)\n"
    program=fixture(tmp_path,[('a','frozen',code,'full')],'a')
    output,receipt=run(tmp_path,program,[np.ones((3,4),np.float32)])
    assert receipt['status']=='failed'
    assert 'Strip training supervision' in (output/'stderr.log').read_text()


def test_inherited_numeric_state_belongs_to_parent_not_child(tmp_path):
    parent="import capabilities\ndef predict(s,w,c):\n p=capabilities.parent(s,c)\n return {'score':p['score']+w['delta'],'support':p['support']}\n"
    child="import capabilities\ndef predict(s,w,c):\n assert not w\n return capabilities.parent(s,c)\n"
    program=fixture(tmp_path,[('a','frozen',parent,'full'),('b','a',child,'full')],
                    'b',{'a':{'delta':np.array([4.5])}})
    image=np.ones((3,4),np.float32)
    output,receipt=run(tmp_path,program,[image])
    assert receipt['status']=='ok',(output/'stderr.log').read_text()
    with np.load(output/'prediction-0000.npz') as z:
        assert np.array_equal(z['score'],image+4.5)


def test_weight_fit_excluded_source_never_reaches_field_writer(tmp_path):
    protocol=Protocol(('A',),('B',),('C',),seeds=(41,))
    sample=Sample('withheld','B','interface',{'image':np.ones((3,4)),
                  'valid':np.ones((3,4),bool)},np.ones((3,4),bool),
                  np.ones((3,4)),np.empty((0,2)))
    class ForbiddenPool:
        def stage(self,*args):
            raise AssertionError('Must reject before writing labels')
    with pytest.raises(ValueError,match='Weight-fit excluded'):
        stage(tmp_path/'input',[sample],ForbiddenPool(),protocol,fit=True,steps=1)
    assert not list((tmp_path/'input').iterdir())


def test_result_seal_is_checked_before_reusing_a_completed_attempt(tmp_path):
    write_new(tmp_path/'result.json',{'decision':{'accepted':True}})
    write_new(tmp_path/'result_binding.json',{'result.json':'0'*64})
    with pytest.raises(ValueError,match='evidence changed'):
        verified_result(tmp_path)


def test_error_profiles_are_positive_only_tie_aware_and_do_not_mutate_inputs():
    image=np.arange(12,dtype=np.float32).reshape(3,4)
    valid=np.ones(image.shape,bool)
    initial=np.zeros(image.shape,bool);initial[:,:2]=True
    human=np.zeros(image.shape,bool);human.ravel()[[0,11]]=True
    selected=np.zeros(image.shape,bool);selected.ravel()[[0,5]]=True
    fields={'image':image,'student':image/11,'base':np.zeros_like(image),
            'valid':valid,'support':initial,'physics_valid':valid.copy()}
    original={k:v.copy() for k,v in fields.items()}
    result=error_signal_profiles(fields,human,selected)
    assert result['unknown_pixels_are_not_negatives']
    regions=result['regions']
    assert regions['selected_known_positive']['positive_pixels']==1
    missed=regions['missed_known_positive_outside_initial_support_physics_valid']
    assert missed['positive_pixels']==1
    assert missed['fields']['student']['valid_midrank_q10_q50_q90']==[1.,1.,1.]
    assert missed['fields']['base']['valid_midrank_q10_q50_q90']==[.5,.5,.5]
    assert regions['missed_known_positive_inside_initial_support']['positive_pixels']==0
    assert all(np.array_equal(fields[k],v) for k,v in original.items())


def test_domain_availability_distinguishes_constant_support_from_two_domain_image():
    labels=np.array([[0,0,1,1],[0,0,1,1]],dtype=np.float32)
    valid=np.ones(labels.shape,bool)
    fields={'domain_labels':labels,'valid':valid,'support':labels==1,'physics_valid':valid.copy()}
    original={k:v.copy() for k,v in fields.items()}
    result=domain_signal_profile(fields)
    assert result['observable_only']
    assert result['regions']['support']['distinct_values']==1
    assert result['regions']['support']['internal_domain_transitions']==0
    assert result['regions']['valid']['distinct_values']==2
    assert result['regions']['valid']['internal_domain_transitions']==2
    assert all(np.array_equal(fields[k],v) for k,v in original.items())


def test_training_availability_does_not_turn_unknown_into_negative_or_read_excluded_labels():
    from types import SimpleNamespace
    protocol=Protocol(('A',),('B',),('C',))
    human=np.array([[True,False,False,True]])
    known=np.array([[1.,.5,0.,0.]])
    sample=SimpleNamespace(group='A',human=human,known=known,
                           fields={'valid':np.ones(human.shape,bool)})
    excluded=SimpleNamespace(group='B')  # Accessing its labels would fail.
    result=training_supervision_summary([sample,excluded],protocol)
    row=result['sources']['A']
    assert row=={'cases':1,'valid_pixels':4,'known_positive_pixels':1,
                 'known_negative_pixels':1,'unknown_pixels':2,
                 'positive_without_known_weight_pixels':1}
    assert np.array_equal(known,[[1.,.5,0.,0.]])
    assert result['unknown_pixels_are_not_negatives']
    assert result['fit_sources_only']


def test_fitted_metadata_is_bound_and_does_not_expose_full_weights(tmp_path,monkeypatch):
    from . import composition_study as study
    from .contracts import file_hash
    monkeypatch.setattr(study,'ROOT',tmp_path)
    path=tmp_path/'state.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream,weights=np.arange(50),negative_count=np.array([0]),
                            has_negative=np.array([0]),audit_steps=np.array([48]))
    state={'path':'state.npz','sha256':file_hash(path)}
    result=fitted_state_summary({'state':state})
    assert result['state_shapes']['weights']==[50]
    assert 'weights' not in result['candidate_reported_scalar_metadata']
    assert result['candidate_reported_scalar_metadata']['negative_count']==0
    assert result['metadata_is_not_independent_proof_of_iterations_or_training_gain']
    with pytest.raises(ValueError,match='state changed'):
        fitted_state_summary({'state':dict(state,sha256='0'*64)})


def test_fit_decisions_expose_zero_quota_and_small_route_feedback_not_weights(tmp_path,monkeypatch):
    from . import composition_study as study
    from .contracts import file_hash
    monkeypatch.setattr(study,'ROOT',tmp_path)
    path=tmp_path/'state.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream,full_quota_fraction=np.array([0.]),
            fit_mean_gain=np.array([0.]),fit_samples_used=np.array([20]),
            route_index=np.array([2,-1,0]),route_gain=np.array([.01,0.,.003]),
            weights=np.arange(4),table_parent_variation=np.ones((32,32)))
    result=fitted_state_summary({'state':{'path':'state.npz','sha256':file_hash(path)}})
    assert result['candidate_reported_training_decisions']=={
        'full_quota_fraction':[0.], 'fit_mean_gain':[0.], 'fit_samples_used':[20],
        'route_index':[2,-1,0], 'route_gain':[.01,0.,.003]}
    assert result['metadata_is_not_independent_proof_of_iterations_or_training_gain']
    assert 'weights' not in result['candidate_reported_training_decisions']


def test_fit_decision_feedback_omits_nonfinite_or_oversized_arrays(tmp_path,monkeypatch):
    from . import composition_study as study
    from .contracts import file_hash
    monkeypatch.setattr(study,'ROOT',tmp_path)
    path=tmp_path/'state.npz'
    with path.open('xb') as stream:
        np.savez_compressed(stream,fit_mean_gain=np.array([np.nan]),
            route_gain=np.arange(9),route_index=np.array(['not-a-number']))
    result=fitted_state_summary({'state':{'path':'state.npz','sha256':file_hash(path)}})
    assert result['candidate_reported_training_decisions']=={}
