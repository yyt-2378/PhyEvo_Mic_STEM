import numpy as np
import pytest
from PIL import Image
from scripts.dennler_author_weights import layers_from_config
from scripts.run_author_pretrained import label_points
from scripts.fetch_author_pretrained import RemoteZip,checked,ROOT

def test_nested_h5_architecture_order_preserved():
    layers=[{'class_name':'Conv2D','config':{'name':'a'}},{'class_name':'Dense','config':{'name':'b'}}]
    config={'class_name':'Model','config':{'layers':[{'class_name':'InputLayer','config':{}},
        {'class_name':'Model','config':{'layers':layers}}]}}
    assert layers_from_config(config)==layers

def test_author_label_centroids_keep_distinct_diagonal_components(tmp_path):
    image=np.zeros((16,16,3),np.uint8);image[3:5,7:9,0]=255;image[5,9,2]=255
    path=tmp_path/'labels.png';Image.fromarray(image).save(path)
    np.testing.assert_array_equal(label_points(path),[[7.5,3.5],[9,5]])

def test_remote_archive_inventory_seek_and_bounds():
    z=RemoteZip('https://example.invalid/archive.zip',128)
    assert z.seek(-22,2)==106 and z.tell()==106
    assert z.seek(128)==128 and z.read(1)==b''
    with pytest.raises(ValueError,match='Negative'):z.seek(-1)

def test_download_path_cannot_escape_project():
    with pytest.raises(ValueError):checked(ROOT.parent/'unexpected-file')
    with pytest.raises(ValueError):checked(ROOT)

def test_native_postprocessing_matches_published_functions():
    import ast
    import cv2
    from types import SimpleNamespace
    from tfree_detection.dennler_full import postprocess,localvariance
    def load(relative,name,scope):
        path=ROOT/'vendor/fair_baselines_20261014/dennler/src'/relative
        node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
        return scope[name]
    gaussian=load('utils/preprocessing.py','gaussian_filtering',{'cv2':cv2})
    cut=load('utils/preprocessing.py','cut_intensity',{})
    normalize=load('utils/preprocessing.py','normalize_intensity',{'np':np})
    scope={'np':np,'cv2':cv2,'gaussian_filtering':gaussian,'cut_intensity':cut,'normalize_intensity':normalize,
           'helpers':SimpleNamespace(localvariance_filter=localvariance)}
    symmetry=load('postprocessing.py','binarization_symmetry',scope)
    variance=load('postprocessing.py','binarization_localvariance',scope)
    combine=load('postprocessing.py','combine_symmetry_localvariance',scope)
    rng=np.random.default_rng(33);image=rng.normal(127,30,(256,256));reference=image[:128,:128]
    score=np.maximum(0,rng.normal(50,40,image.shape));score[100:180,120:190]+=400
    expected=combine(symmetry(score),variance(localvariance(image),reference))
    expected[expected==1.05]=2
    np.testing.assert_array_equal(postprocess(score,image,reference),expected.astype(np.uint8))
