import numpy as np
import pytest

from scripts.verify_rsi_refinement import compare_archives


def archive(path,values):
    with path.open("xb") as stream: np.savez(stream,**values)
    return path


def test_verifier_compares_tensor_values_not_archive_labels(tmp_path):
    a=archive(tmp_path/"a.npz",{"model:x":np.zeros(2),"optimizer:x":np.ones(1)})
    b=archive(tmp_path/"b.npz",{"model:x":np.zeros(2),"optimizer:x":np.zeros(1)})
    assert compare_archives(a,b)["all_arrays_equal"]
    assert not compare_archives(a,b,"optimizer:")["all_arrays_equal"]


def test_verifier_catches_tensor_shape_and_missing_inventory(tmp_path):
    a=archive(tmp_path/"a.npz",{"model:x":np.zeros(2)})
    b=archive(tmp_path/"b.npz",{"model:x":np.zeros((1,2))})
    c=archive(tmp_path/"c.npz",{"model:y":np.zeros(2)})
    assert compare_archives(a,b)["different_arrays"]==["model:x"]
    with pytest.raises(ValueError,match="inventory"):compare_archives(a,c)
