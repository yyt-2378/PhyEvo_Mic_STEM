from types import SimpleNamespace

import numpy as np
import pytest

from tfree_detection.interface_retention import RetentionSampler,freeze_feature_extractor,retention_loss
from tfree_detection.interface_reliability import long_boundary_support,qualify_boundary_or_band
from tfree_detection.interface_runtime import choose_interface
from tfree_detection.rsi_refinement import RefinementPolicy

torch=pytest.importorskip("torch")


def test_expert_positive_outside_physical_validity_is_not_erased():
    plane={"image":np.zeros((32,32),np.float32),"human":np.ones((32,32),bool),
           "human_weight":np.ones((32,32),np.float32),"teacher":np.zeros((32,32),bool),
           "teacher_weight":np.zeros((32,32),np.float32)}
    physical={"valid":np.zeros((32,32),bool),"variation":np.zeros((32,32),np.float32)}
    entry={"case_number":1,"group":"a","geometry_index":4,"geometry":"interface"}
    sampler=RetentionSampler([entry],{1:plane},RefinementPolicy("x"),{1:physical})
    assert sampler.targets[1]["physical_positive"].all()
    assert not sampler.targets[1]["physical_reference_prior"].any()


def test_feature_extractor_is_frozen_and_retention_ignores_uniform_logit_offset():
    from tfree_detection.expert_student import make_expert_student
    model=freeze_feature_extractor(make_expert_student())
    assert all(p.requires_grad==n.startswith("geometry_head.") for n,p in model.named_parameters())
    z=torch.randn(1,5,16,16);zz=z.clone();zz[:,4]+=3
    batch={k:torch.ones(1,16,16) for k in ["valid_pixels","human","physical_positive"]}
    batch.update({k:torch.zeros(1,16,16) for k in ["physical_low_in_roi","physical_reference_prior"]})
    batch["geometry_index"]=torch.tensor([4])
    _,parts=retention_loss(SimpleNamespace(geometry_logits=zz),SimpleNamespace(geometry_logits=z),batch)
    assert parts["centered_logit_retention"]<1e-10


def test_long_boundary_support_is_rotation_invariant():
    labels=np.zeros((256,256),np.int8);labels[:,120:]=1
    for k in range(4):
        a=np.rot90(labels,k)
        assert long_boundary_support(a,np.ones_like(a,bool),32)["eligible"]
    labels=np.zeros((256,256),np.int8);labels[80:160,80:160]=1
    assert not long_boundary_support(labels,np.ones_like(labels,bool),32)["eligible"]


def test_quota_free_selection_never_forces_unreliable_physics_into_student():
    phys=np.zeros((32,32));phys[:3]=1
    student=np.full((32,32),.1);student[20:23]=.9
    mask,why=choose_interface(phys,np.ones_like(phys,bool),{"eligible":False},student,
                              student_admitted=True,budget=50)
    assert why["branch"]=="student" and not mask[:3].any() and mask.sum()==50
    mask,why=choose_interface(phys,np.ones_like(phys,bool),{"eligible":True},student,
                              student_admitted=True,budget=50)
    assert why["branch"]=="physics_geometry" and not mask[20:23].any()
    _,why=choose_interface(phys,np.ones_like(phys,bool),{"eligible":False},np.full((32,32),.99),
                           student_admitted=True,budget=50)
    assert why["branch"]=="physics_unqualified_review"


def test_conflicting_scale_never_qualifies_long_boundary():
    labels=np.zeros((512,512),np.int8);labels[:,256:]=1
    physical={"labels":labels,"valid":np.ones_like(labels,bool)}
    strict={"eligible":False,"hypothesis":"two_domain_boundary","views":[
        {"hypothesis":"enclosed_texture_band","top_budget_overlap":.9},
        {"hypothesis":"two_domain_boundary","top_budget_overlap":.8}]}
    assert not qualify_boundary_or_band(physical,strict)["eligible"]
