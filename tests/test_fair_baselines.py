from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tfree_detection.fair_baselines import make_folds,reference_bank,shared_loss,UNet,make_model,logits_for,predict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_source_split_keeps_frames_together_and_excludes_accepted():
    entries=[]
    for i,g in enumerate(['a','a','b','b'],1):
        entries.append(dict(case_number=i,group=g,source_file_sha256=g,geometry='point',action='correct_geometry'))
    entries.append(dict(case_number=5,group='a',source_file_sha256='a',geometry='point',action='accept_prediction'))
    folds=make_folds(entries)
    assert len(folds)==2
    assert folds[0]['test_case_numbers']==[1,2]
    assert all(5 not in f['train_case_numbers']+f['test_case_numbers'] for f in folds)
    entries[2]['source_file_sha256']='a'
    with pytest.raises(ValueError,match='leakage'):make_folds(entries)

def test_al_material_holdout_excludes_both_orientations():
    entries=[dict(case_number=i,group=g,source_file_sha256=g,geometry='interface',action='correct_geometry')
             for i,g in enumerate(['sigma3-orientation-i','sigma3-orientation-ii','zro2'],1)]
    fold=make_folds(entries)[-1]
    assert fold['fold_id']=='material-Al-held-out'
    assert fold['train_case_numbers']==[3] and fold['test_case_numbers']==[1,2]

def test_reference_rejects_test_source_and_never_uses_test_pixels():
    entry=dict(case_number=1,group='train',geometry_index=1)
    plane={'image':np.ones((16,16),np.float32),'human':np.eye(16,dtype=np.float32)}
    bank,audit=reference_bank([entry],{1:plane},excluded_groups=['test'],size=8)
    assert bank[1].shape==(1,8,8) and audit[0]['group']=='train'
    with pytest.raises(ValueError,match='Held-out'):reference_bank([entry],{1:plane},excluded_groups=['train'])

def test_expert_only_loss_does_not_read_teacher_or_call_unknown_negative():
    z=torch.full((1,8,8),-8.,requires_grad=True)
    human=torch.zeros_like(z); human.data[0,2,2]=1
    batch={'human':human,'human_weight':human.clone(),'valid_pixels':torch.ones_like(z),'geometry_index':torch.tensor([1])}
    loss,parts=shared_loss(z,batch)
    loss.backward()
    assert torch.count_nonzero(z.grad)==1 and z.grad[0,2,2]<0
    assert float(parts['occupancy_prior'])==0 and float(parts['teacher'])==0
    assert 'teacher' not in batch

@pytest.mark.parametrize('method',['phyevo_teacher_expert','same_student_expert_only','unet_expert_only','emseek_refunet_expert_only'])
def test_complete_models_forward_backward_and_source_pixel_output(method):
    if method=='emseek_refunet_expert_only' and not (ROOT/'vendor/fair_baselines_20261014/emseek/emseek/models/unetpp_ref.py').is_file():
        pytest.skip('Run scripts/prepare_fair_baselines.py to fetch pinned official network')
    torch.set_num_threads(1);torch.manual_seed(23)
    model=make_model(method,ROOT)
    image=torch.rand(2,1,64,64);geo=torch.tensor([1,4]);ref=torch.rand(2,1,64,64)
    z,_=logits_for(model,method,image,geo,ref)
    assert z.shape==(2,64,64) and torch.isfinite(z).all()
    z.square().mean().backward()
    assert any(p.grad is not None for p in model.parameters())
    bank={1:np.ones((1,64,64),np.float32)}
    out=predict(model,method,np.ones((39,47),np.float32),1,bank,device='cpu',tile_size=64,overlap=16)
    assert out.shape==(39,47) and np.isfinite(out).all()

def test_ablation_initial_parameters_identical():
    torch.manual_seed(41);a=make_model('phyevo_teacher_expert',ROOT)
    torch.manual_seed(41);b=make_model('same_student_expert_only',ROOT)
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())

def test_summary_keeps_geometry_and_source_separate():
    from scripts.analyze_fair_baselines import summarize
    rows=[]
    for seed in (23,41,67):
        for number in (1,2):
            rows.append({'main_comparison_eligible':True,'fold':'x','group':'g','geometry':'point','step':800,
                         'budget_name':'main','seed':seed,'methods':{'m':{'annotated_positive_count':number*2,
                         'matched_positive_count':number,'positive_coverage':.5}}})
    seeds,summary=summarize(rows)
    assert len(seeds)==3 and len(summary)==1
    assert summary[0]['micro_coverage_mean']==.5 and summary[0]['positive_count']==6
    assert summary[0]['seed_count']==3 and summary[0]['case_count']==2

def test_audit_accepts_only_known_shallow_counter_copy_not_prediction_change():
    from copy import deepcopy
    from scripts.analyze_fair_baselines import verify_checkpoint_copy
    a={'step':400,'score':'abc','sampler_audit':{'patches':2,'by_case':{'1':2},'by_source':{'x':2}}}
    final={'step':800,'score':'def','sampler_audit':{'patches':4,'by_case':{'1':4},'by_source':{'x':4}}}
    b=deepcopy(a);b['sampler_audit']['by_case']={'1':4};b['sampler_audit']['by_source']={'x':4}
    assert verify_checkpoint_copy(a,b,final)
    b['score']='modified'
    with pytest.raises(ValueError,match='identity'):verify_checkpoint_copy(a,b,final)
