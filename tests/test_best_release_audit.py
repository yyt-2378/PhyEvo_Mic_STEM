import copy
import numpy as np
import pytest
from scripts.freeze_best_release import audit_split,pixel_duplicates,CHOICES,COMBINATION_BRANCHES

def fixture():
    entries={1:{'group':'a','source_file_sha256':'aaa'},2:{'group':'b','source_file_sha256':'bbb'}}
    fold={'train_case_numbers':[1],'validation_case_numbers':[2],'excluded_groups':['b']}
    meta={'train_case_numbers':[1],'sampler_audit':{'by_case':{'1':4},'by_source':{'a':4},'patches':4}}
    bank={'selection_uses_validation':False,'cases':[{'case_number':1}]}
    return entries,fold,meta,bank

def test_split_checks_actual_sampler_and_reference_inventory():
    a=fixture();assert audit_split(*a)['source_file_disjoint']
    a[2]['sampler_audit']['by_case']={'2':4}
    with pytest.raises(ValueError,match='Observed sampler'):audit_split(*a)

def test_same_image_under_different_groups_is_not_independent():
    a=fixture();a[0][2]['source_file_sha256']='aaa'
    with pytest.raises(ValueError,match='Source-file'):audit_split(*a)

def test_test_reference_rejected():
    a=fixture();a[3]['cases']=[{'case_number':2}]
    with pytest.raises(ValueError,match='Reference bank'):audit_split(*a)

def test_sampler_counter_drift_rejected():
    a=fixture();a[2]['sampler_audit']['patches']=5
    with pytest.raises(ValueError,match='Sampler count'):audit_split(*a)

def test_d4_pixel_duplicates_detected_cross_source_only():
    a=np.arange(16,dtype=np.float32).reshape(4,4)
    entries={1:{'group':'a'},2:{'group':'b'}}
    planes={1:{'image':a},2:{'image':np.rot90(a)[:,::-1]}}
    assert pixel_duplicates(planes,entries)==[[1,2]]
    entries[2]['group']='a';assert pixel_duplicates(planes,entries)==[]

def test_best_collection_does_not_fake_one_checkpoint_configuration():
    assert len(CHOICES)==5
    assert len({(v[0],v[1]) for v in CHOICES.values()})==5
    assert CHOICES['guo-superscan-single-acquisition'][2:]==(28.82,22.06)

def test_reported_combination_column_preserves_its_actual_branch():
    assert COMBINATION_BRANCHES['guo-superscan-single-acquisition']=='student_reranked_physics_plus_rescue'
    assert COMBINATION_BRANCHES['guo_2021_mowte2_acquisition']=='student_reranked_physics_plus_rescue'
    assert COMBINATION_BRANCHES['sigma3-orientation-i']=='hybrid_fixed_budget'
