from types import SimpleNamespace

import numpy as np
import pytest

from tfree_detection.interface_student import (InterfaceSampler, physical_ranking_loss,
    physical_training_targets, score_health)
from tfree_detection.rsi_refinement import RefinementPolicy

torch = pytest.importorskip("torch")


def example():
    human=np.zeros((64,64),bool);human[:,24:40]=True
    y,x=np.indices(human.shape)
    plane={"image":np.cos(x/3).astype(np.float32),"human":human,
           "human_weight":human.astype(np.float32),"teacher":np.zeros_like(human),
           "teacher_weight":np.zeros_like(human,dtype=np.float32)}
    physical={"valid":np.ones_like(human),"variation":np.exp(-((x-31)/8)**2).astype(np.float32)}
    return plane,physical


def test_original_unknown_labels_unchanged_and_only_qualified_references_used():
    plane,physical=example()
    old={k:v.copy() for k,v in plane.items()}
    targets=physical_training_targets(plane,physical)
    assert all(np.array_equal(plane[k],old[k]) for k in old)
    assert not targets["physical_positive"][~plane["human"]].any()
    assert not targets["physical_reference_prior"][plane["human"]].any()
    assert not targets["physical_reference_prior"][:,15:49].any()
    assert targets["physical_reference_prior"].any()


def test_all_high_solution_costs_more_than_localized_solution():
    plane,physical=example(); targets=physical_training_targets(plane,physical)
    batch={k:torch.tensor(v[None],dtype=torch.float32) for k,v in {**plane,**targets}.items()}
    batch["geometry_index"]=torch.tensor([4]);batch["valid_pixels"]=torch.ones(1,64,64)
    z=torch.zeros(1,5,64,64);z[:,4]=8
    bad,_=physical_ranking_loss(SimpleNamespace(geometry_logits=z),batch)
    z=z.clone();z[:,4]=-3;z[:,4, :,24:40]=3
    good,_=physical_ranking_loss(SimpleNamespace(geometry_logits=z),batch)
    assert good < bad


def test_relative_ranking_has_gradient_and_scale_is_consistent():
    plane,physical=example();targets=physical_training_targets(plane,physical)
    batch={k:torch.tensor(v[None],dtype=torch.float32) for k,v in {**plane,**targets}.items()}
    batch["geometry_index"]=torch.tensor([4]);batch["valid_pixels"]=torch.ones(1,64,64)
    z=torch.zeros(1,5,64,64,requires_grad=True)
    loss,parts=physical_ranking_loss(SimpleNamespace(geometry_logits=z),batch,
                                   SimpleNamespace(geometry_logits=z[:,:,::2,::2]))
    loss.backward()
    assert torch.isfinite(z.grad).all()
    assert parts["scale_consistency"] == 0
    assert z.grad[:,4][batch["physical_positive"]>0].mean() < 0


def test_health_guard_rejects_saturated_and_flat_checkpoints():
    assert not score_health(np.full((32,32),.9))["eligible"]
    assert not score_health(np.full((32,32),.2))["eligible"]
    p=np.full((32,32),.1);p[:,10:13]=.9
    assert score_health(p)["eligible"]


def test_sampler_preserves_identical_control_schedule_and_annotation_geometry():
    plane,physical=example()
    entry={"case_number":1,"group":"train","geometry":"interface","geometry_index":4,
        "semantic_kind":"interface_band_roi_not_centerline","teacher_positive_pixels":0,"action":"correct_geometry"}
    policy=RefinementPolicy("matched",positive_crop_probability=.75,photometric=True)
    a=InterfaceSampler([entry],{1:plane},policy,{1:physical},excluded_groups=["test"])
    b=InterfaceSampler([entry],{1:plane},policy,{1:physical},excluded_groups=["test"])
    aa,ar=a.sample(np.random.default_rng(1),batch_size=4,patch_size=32,device="cpu",augmentation_seed=3)
    bb,br=b.sample(np.random.default_rng(1),batch_size=4,patch_size=32,device="cpu",augmentation_seed=3)
    assert ar==br
    assert all(torch.equal(aa[k],bb[k]) for k in aa)
    assert not aa["physical_positive"][aa["human"]==0].any()
    with pytest.raises(ValueError,match="held-out"):
        InterfaceSampler([entry],{1:plane},policy,{1:physical},excluded_groups=["train"])
