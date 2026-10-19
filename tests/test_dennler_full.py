import ast
from pathlib import Path
import numpy as np
import pytest
from scipy import ndimage
import torch
from tfree_detection.dennler_full import energy_tensor,edge_factors,best_reference,patch_classes,DennlerVGG16

ROOT=Path(__file__).resolve().parents[1]

def upstream_function(filename,name,scope):
    path=ROOT/'vendor/fair_baselines_20261014/dennler/src'/filename
    if not path.exists():pytest.skip('Pinned upstream audit sources not downloaded')
    tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    return scope[name]

def test_batched_energy_matches_complete_original_filter_and_even_blur():
    torch.set_num_threads(1)
    rng=np.random.default_rng(8);patch=rng.normal(size=(32,32));filters=rng.normal(size=(3,7,7))
    actual=energy_tensor(torch.tensor(patch[None,None]),torch.tensor(filters[:,None])).numpy()[0]
    expected=np.stack([ndimage.convolve(ndimage.convolve(patch,k,mode='mirror')**2,np.ones((16,16)),mode='constant') for k in filters])
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-8)

def test_edge_factors_exactly_match_released_loop():
    fn=upstream_function('utils/eigenfiltering.py','handle_edges',{})
    image=np.ones((48,64))
    expected=fn(image.copy(),edgesize=4,patchsize=16)
    np.testing.assert_array_equal(edge_factors(image.shape,edge=4,patch=16),expected)

@pytest.mark.parametrize('seed',[3,8,14])
def test_reference_graph_matches_original_with_compatibility_alias(seed,monkeypatch):
    import networkx as nx
    if not hasattr(nx.Graph,'node'):monkeypatch.setattr(nx.Graph,'node',property(lambda s:s._node),raising=False)
    graph=upstream_function('utils/helpers.py','get_graph',{'np':np})
    choose=upstream_function('utils/helpers.py','get_best_node_from_Kneighbors',{})
    values=np.random.default_rng(seed).integers(0,2,(8,8))
    normal,defect,_=graph(values,connectivity=4)
    selected=normal if len(normal) else defect
    node,value=choose(selected,k=10,connectivity=4)
    if value==0:
        normal,defect,_=graph(values,connectivity=8)
        node,_=choose(normal if len(normal) else defect,k=10,connectivity=8)
    assert best_reference(values)==node

def test_published_patch_class_target_and_full_vgg_forward():
    labels=np.zeros((3,128,128),np.uint8);labels[1,:64]=1;labels[2]=1
    assert patch_classes(labels).tolist()==[1,0,1]
    torch.set_num_threads(1);model=DennlerVGG16()
    logits=model(torch.zeros((2,1,128,128)))
    assert logits.shape==(2,2) and torch.isfinite(logits).all()

def test_complete_eigenfilter_score_matches_scalar_mahalanobis_at_interior_pixels():
    from scipy.spatial.distance import mahalanobis
    from tfree_detection.dennler_full import eigenfilter_score
    rng=np.random.default_rng(52);reference=rng.normal(size=(128,128));image=rng.normal(size=(128,128))
    samples=np.lib.stride_tricks.sliding_window_view(reference,(7,7)).reshape(-1,49)
    w,v=np.linalg.eig(np.cov(samples.T));filters=v[w.argsort()[::-1]].reshape(49,7,7)
    def energy(a):return np.stack([ndimage.convolve(ndimage.convolve(a,k,mode='mirror')**2,np.ones((16,16)),mode='constant') for k in filters])
    a,b=energy(image),energy(reference);inv=np.linalg.inv(np.cov(b.reshape(49,-1).T,rowvar=False))
    torch.set_num_threads(1);actual=eigenfilter_score(image,reference,device='cpu')
    factors=edge_factors(image.shape)
    for y,x in [(32,32),(42,70),(96,95)]:
        expected=mahalanobis(a[:,y,x],b[:,y,x],inv)*factors[y,x]
        assert actual[y,x]==pytest.approx(expected,rel=1e-8,abs=1e-8)
    assert not actual[:16].any() and np.isfinite(actual).all()
