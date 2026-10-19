import numpy as np
import pytest
from tfree_detection.fair_synthetic import make_training_texture,require_synthetic_training

@pytest.mark.parametrize('index,classes',[(0,{0}),(18,{0,2}),(28,{0,1})])
def test_generator_deterministic_without_any_experimental_input(index,classes):
    a,t,m=make_training_texture(index,size=128);b,u,n=make_training_texture(index,size=128)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(t,u)
    assert set(np.unique(t))==classes and m==n and m['experimental_inputs']==[]

def test_locked_real_dataset_cannot_enter_training_or_be_silently_relabelled():
    from scripts.run_fair_dense import training_inputs
    with pytest.raises(PermissionError,match='locked'):training_inputs({'rows':[{'split':'train'}]})
    valid={'training_source_kind':'procedural_synthetic_only','training_rows':[{'source_kind':'procedural_synthetic'} for _ in range(37)],'rows':[{'split':'test'}]}
    require_synthetic_training(valid)
    valid['rows'][0]['split']='train'
    with pytest.raises(PermissionError):require_synthetic_training(valid)

def test_disqualified_run_refused_before_checkpoint_or_score_access(tmp_path):
    from scripts.run_fair_dense import checked
    (tmp_path/'DISQUALIFIED.json').write_text('{}')
    with pytest.raises(PermissionError,match='Quarantined'):checked(tmp_path)
