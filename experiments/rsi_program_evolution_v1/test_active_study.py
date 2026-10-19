"""Focused tests of autonomous eligibility and anonymous diagnostic semantics."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest

from .active_diagnostics import available_actions, boundary_diagnostic, diagnose, probe_proposals, selected_mask, validate_decision
from .active_study import stop_status
from .contracts import Protocol
from .test_core import proposal, protocol


def test_crossover_and_generation_are_available_after_stagnation():
    for count in (0,2,5,11):
        assert available_actions(count)['allowed']==['generate','tune','evolve','repair']


def test_free_choice_is_not_forced_to_use_incumbent_or_train():
    p=proposal();validate_decision(p,[],protocol())
    parent='a'*64
    validate_decision(replace(p,action='tune',parents=(parent,),target='learning',training_steps=64),[parent],protocol())
    with pytest.raises(ValueError,match='visible parents'):
        validate_decision(replace(p,action='tune',parents=(parent,)),[],protocol())
    with pytest.raises(ValueError,match='declared as learning'):
        validate_decision(replace(p,training_steps=2),[],protocol())


def test_probe_library_is_ten_distinct_explicit_references():
    probes=probe_proposals()
    assert len(probes)==len({p.identity for _,p in probes})==10
    assert all(p.validate(protocol()) and p.training_steps==0 for _,p in probes)


def test_area_budget_is_for_whole_image_not_supported_area():
    score=np.arange(100).reshape(10,10)+1
    support=np.zeros((10,10),bool);support[:2]=True
    selected=selected_mask(score,support,.1)
    assert selected.sum()==10
    assert not np.any(selected & ~support)


def test_diagnostics_never_read_audit_arrays_or_leak_coordinates():
    shape=(10,10);human=np.zeros(shape,bool);human[3:5]=True
    fields={k:np.arange(100,dtype=float).reshape(shape) for k in ('image','base','physics','student','variation')}
    fields.update(valid=np.ones(shape,bool),support=np.ones(shape,bool))
    train=SimpleNamespace(group='A',key='raw-private-case',human=human,known=human.astype(float),fields=fields)
    val=SimpleNamespace(group='B',key='private-case2',human=human,known=human.astype(float),fields=fields)
    audit=SimpleNamespace(group='C') # Any access beyond its split would fail.
    result=diagnose([train,val,audit],protocol(),{})
    assert result['train']['planes']==1
    assert result['train']['mean_per_plane']['explicit_negative_pixels']==0
    assert 'raw-private-case' not in str(result) and 'private-case2' not in str(result)
    def keys(value):
        if not isinstance(value,dict):return set()
        return set(value).union(*(keys(v) for v in value.values()))
    assert not {'coordinates','points_xy','human','known','key','group'} & keys(result)
    assert 'audit' not in result


def test_stop_rule_uses_stagnation_not_best_outcome_peeking():
    c={'maximum_rounds_per_episode':12,'minimum_rounds_per_episode':6,'stop_after_stagnant_rounds':4}
    events=[{'body':{'status':'retained_for_research'}} for _ in range(6)]
    assert not stop_status(events[:3],c)['finished']
    assert stop_status(events,c)['finished']
    events[-1]['body']['status']='accepted'
    assert not stop_status(events,c)['finished']
    assert stop_status(events+events,c)['finished']


def test_boundary_diagnostic_distinguishes_absent_and_present_evidence():
    human=np.zeros((12,12),bool);human[:,6]=True
    valid=np.ones_like(human)
    absent=boundary_diagnostic({'valid':valid},human)
    assert absent['domain_boundaries_present']==0
    assert absent['positive_mass_in_distance_bins_0_2_4_8_16_32_inf_px']==[0.,0.,0.,0.,0.,1.]
    domains=np.ones_like(human,dtype=int);domains[:,6:]=2
    present=boundary_diagnostic({'valid':valid,'domain_labels':domains},human)
    assert present['domain_boundaries_present']==1
    assert present['positive_mass_in_distance_bins_0_2_4_8_16_32_inf_px'][0]==1
    assert sum(present['valid_area_in_same_distance_bins'])==pytest.approx(1.)


def test_background_domain_zero_is_not_a_grain_boundary():
    domains=np.zeros((8,8),int);domains[:,4:]=1
    human=np.ones((8,8),bool)
    result=boundary_diagnostic({'valid':human,'domain_labels':domains},human)
    assert result['domain_boundaries_present']==0


def test_library_configuration_adds_only_sandbox_local_environment(tmp_path):
    from .runtime_compat import ConfiguredSandbox
    from .test_core import ROOT, sandbox_fixture
    inputs,program,original=sandbox_fixture(tmp_path)
    adapter=ConfiguredSandbox(ROOT,ROOT/'.venvs/sam2',timeout=30)
    worker=ROOT/'experiments/rsi_program_evolution_v1/worker.py'
    old=original.command(program,inputs,tmp_path,worker)
    new=adapter.command(program,inputs,tmp_path,worker)
    assert new[:-13]==old[:-4]
    assert new[-4:]==old[-4:]
    assert new[-13:-4]==['--setenv','XDG_CONFIG_HOME','/work/config',
                          '--setenv','MPLCONFIGDIR','/work/config/matplotlib',
                          '--setenv','MPLBACKEND','Agg']


def test_compatibility_replay_requires_sealed_results_before_writing(tmp_path):
    from .runtime_compat import replay_failed_import
    output=tmp_path/'replay'
    with pytest.raises(ValueError,match='Complete and seal'):
        replay_failed_import(tmp_path/'study','t02',7,output)
    assert not output.exists()


def test_configured_sandbox_retains_real_os_isolation(tmp_path):
    import json
    from .runtime_compat import ConfiguredSandbox
    from .test_core import ROOT, sandbox_fixture
    from .contracts import write_new
    inputs,program,_=sandbox_fixture(tmp_path)
    write_new(inputs/'request.json',{'mode':'probe','production_path':str(ROOT/'src')})
    output=tmp_path/'execution'
    receipt=ConfiguredSandbox(ROOT,ROOT/'.venvs/sam2',timeout=45).run(program,inputs,output)
    assert receipt['status']=='ok',(output/'stderr.log').read_text()
    result=json.loads((output/'result.json').read_text())
    assert all(result[k] for k in ('credentials_absent','host_home_absent','production_absent',
                                  'input_readonly','network_blocked'))
    assert receipt['runtime_configuration']['host_mounts_or_permissions_extended'] is False


def test_watershed_import_and_inference_in_configured_sandbox(tmp_path):
    from .runtime_compat import ConfiguredSandbox
    from .test_core import ROOT, sandbox_fixture
    from .contracts import write_new
    inputs,program,_=sandbox_fixture(tmp_path)
    code=('def predict(sample,state,context):\n'
          '    import numpy as np\n'
          '    from skimage.segmentation import watershed\n'
          '    import matplotlib\n'
          '    assert matplotlib.get_configdir()=="/work/config/matplotlib"\n'
          '    markers=np.zeros(sample["image"].shape,np.int32)\n'
          '    markers[1,1]=1;markers[-2,-2]=2\n'
          '    labels=watershed(sample["image"],markers,mask=sample["valid"])\n'
          '    return {"score":labels.astype(float),"support":sample["valid"]}\n')
    with (program/'pipeline.py').open('x') as f:f.write(code)
    with (inputs/'sample.npz').open('xb') as f:
        np.savez(f,image=np.arange(64).reshape(8,8).astype(float),valid=np.ones((8,8),bool))
    write_new(inputs/'request.json',{'mode':'predict','seed':23,'samples':[{'file':'sample.npz'}]})
    output=tmp_path/'execution'
    receipt=ConfiguredSandbox(ROOT,ROOT/'.venvs/sam2',timeout=45).run(program,inputs,output)
    assert receipt['status']=='ok',(output/'stderr.log').read_text()
    with np.load(output/'prediction-0000.npz',allow_pickle=False) as z:
        assert z['score'].shape==(8,8) and set(np.unique(z['score']))=={1.,2.}
