"""Strict read-only Keras-H5 to Torch VGG16 parameter adapter.

Loads every convolution/dense parameter without training, partial-key fallbacks,
or random weight substitution. Keras parity is a separate checked precondition.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import hashlib
import h5py
import numpy as np
import torch
from torch import nn
from tfree_detection.dennler_full import DennlerVGG16
from tfree_detection.expert_supervision import contained_path,sha256_file,write_new_json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'vendor/author_pretrained_20261014_v1/dennler'

def layers_from_config(config):
    layers=[]
    def walk(obj):
        if isinstance(obj,dict):
            if obj.get('class_name') in ('Conv2D','Dense'):layers.append(obj)
            elif 'config' in obj and isinstance(obj['config'],dict) and 'layers' in obj['config']:
                for v in obj['config']['layers']:walk(v)
    walk(config)
    return layers

def load_author_model(path,device='cpu'):
    path=contained_path(ROOT,path)
    model=DennlerVGG16();dest=[m for m in model.modules() if isinstance(m,(nn.Conv2d,nn.Linear))]
    with h5py.File(path,'r') as f:
        config=json.loads(f.attrs['model_config']);layers=layers_from_config(config)
        tensors={}
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset):tensors[name]=obj[()]
        f['model_weights'].visititems(visit)
    if len(layers)!=17 or len(dest)!=len(layers):raise ValueError('Released model differs from complete VGG architecture')
    used=set();mapping=[]
    with torch.no_grad():
        for layer,target in zip(layers,dest):
            cfg=layer['config'];name=cfg['name']
            candidates={k:v for k,v in tensors.items() if name in Path(k).parts}
            kernel=[k for k in candidates if k.endswith(('kernel:0','kernel'))];bias=[k for k in candidates if k.endswith(('bias:0','bias'))]
            if len(kernel)!=1 or len(bias)!=1:raise ValueError(f'Ambiguous/missing released weights for {name}')
            k,b=kernel[0],bias[0];w=tensors[k];w=w.transpose(3,2,0,1) if isinstance(target,nn.Conv2d) else w.T
            if w.shape!=tuple(target.weight.shape) or tensors[b].shape!=tuple(target.bias.shape):raise ValueError(f'Weight shape mismatch: {name}')
            if not np.isfinite(w).all() or not np.isfinite(tensors[b]).all():raise ValueError('Nonfinite author weights')
            target.weight.copy_(torch.from_numpy(w.copy()));target.bias.copy_(torch.from_numpy(tensors[b].copy()))
            used.update([k,b]);mapping.append({'layer':name,'class':layer['class_name'],'torch_shape':list(w.shape),'kernel_sha256':hashlib.sha256(tensors[k].tobytes()).hexdigest()})
    if used!=set(tensors):raise ValueError(f'Unconsumed model tensors: {set(tensors)-used}')
    return model.to(device).eval(),{'source_sha256':sha256_file(path),'parameter_tensors':len(used),'mapping':mapping}

def parity(path,output):
    # Caller MUST set KERAS_HOME inside this project before importing Keras.
    import os
    if os.environ.get('KERAS_BACKEND')!='torch':raise ValueError('Use the installed Torch backend explicitly')
    contained_path(ROOT,os.environ['KERAS_HOME'])
    import keras
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    # NHWC (Keras) versus NCHW (port) can choose different oneDNN kernels.
    # Use the same non-oneDNN convolution backend for this CPU parity audit.
    torch.backends.mkldnn.enabled=False
    port,receipt=load_author_model(path)
    with h5py.File(path,'r') as f:
        cfg=json.loads(f.attrs['model_config'])
        def audit(obj):
            if isinstance(obj,dict):
                if obj.get('class_name') in ['Lambda','TFOpLambda']:raise ValueError('Executable custom model layer refused')
                for v in obj.values():audit(v)
            elif isinstance(obj,list):
                for v in obj:audit(v)
        audit(cfg)
    original=keras.models.load_model(path,compile=False,safe_mode=True)
    rng=np.random.default_rng(20261014);yy,xx=np.indices((128,128))
    patches=np.stack([np.zeros((128,128)),np.full((128,128),127),np.full((128,128),254),
        127+100*np.sin(xx*.5)*np.cos(yy*.3),*rng.uniform(0,254,(4,128,128))]).astype(np.float32)
    reference_input=keras.applications.vgg16.preprocess_input(np.repeat(patches[...,None],3,axis=-1).copy())
    # Compare both classifier probabilities and all weighted-layer activations;
    # saturated probabilities alone could hide an incorrect conversion.
    weighted=[l for l in original.layers if l.__class__.__name__ in ('Conv2D','Dense')]
    if len(weighted)!=17:raise ValueError('Unexpected Keras nested model; inspect before parity')
    outputs=keras.Model(original.input,[l.output for l in weighted])
    actual=[];handles=[]
    # Torch modules have separate ReLU; hook after each activation (last is logits).
    for layer in [m for m in port.features if isinstance(m,nn.ReLU)]+[port.classifier[i] for i in (1,4,7,9)]:
        handles.append(layer.register_forward_hook(lambda m,i,o:actual.append(o.detach().cpu().numpy().copy())))
    with torch.inference_mode():
        logits=port(torch.from_numpy(patches[:,None]));prob=torch.softmax(logits,1).cpu().numpy()
        expected=[v.detach().cpu().numpy() for v in outputs([reference_input],training=False)]
    for h in handles:h.remove()
    details=[]
    for i,(a,b) in enumerate(zip(actual,expected)):
        if a.ndim==4:a=a.transpose(0,2,3,1)
        if i==16:a=prob
        delta=float(np.max(np.abs(a-b)));scale=max(float(np.max(np.abs(b))),1.0)
        # Float32 convolution reductions with distinct tensor strides can leave
        # milliscale errors near ReLU zero even when activations are thousands.
        # Check scale-normalized error, then verify the entire model in float64
        # below rather than treating cancellation at zero as a weight mismatch.
        if not np.isfinite(a).all() or delta/scale>5e-6:raise ValueError('Float32 activation parity failed')
        details.append({'layer':weighted[i].name,'max_abs_difference':delta,'max_error_over_layer_scale':delta/scale})
    np.testing.assert_allclose(prob,expected[-1],rtol=1e-5,atol=1e-6)
    original64=keras.models.clone_model(original,input_tensors=[keras.Input(shape=(128,128,3),dtype='float64')],
        clone_function=lambda layer:layer.__class__.from_config({**layer.get_config(),'dtype':'float64'}))
    original64.dtype_policy=keras.dtype_policies.DTypePolicy('float64')
    original64.set_weights(original.get_weights());port.double();actual.clear()
    weighted64=[l for l in original64.layers if l.__class__.__name__ in ('Conv2D','Dense')]
    outputs64=keras.Model(original64.input,[l.output for l in weighted64],dtype='float64');handles=[]
    for layer in [m for m in port.features if isinstance(m,nn.ReLU)]+[port.classifier[i] for i in (1,4,7,9)]:
        handles.append(layer.register_forward_hook(lambda m,i,o:actual.append(o.detach().cpu().numpy().copy())))
    with torch.inference_mode():
        logits64=port(torch.from_numpy(patches[:4,None].astype(np.float64)));prob64=torch.softmax(logits64,1).cpu().numpy()
        x64=keras.applications.vgg16.preprocess_input(np.repeat(patches[:4,...,None].astype(np.float64),3,axis=-1))
        expected64=[v.detach().cpu().numpy() for v in outputs64([x64],training=False)]
    for h in handles:h.remove()
    checks64=[]
    for i,(a,b) in enumerate(zip(actual,expected64)):
        if a.ndim==4:a=a.transpose(0,2,3,1)
        if i==16:a=prob64
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-9)
        checks64.append({'layer':weighted64[i].name,'max_abs_difference':float(np.max(np.abs(a-b)))})
    write_new_json(contained_path(ROOT,output),{**receipt,'keras':keras.__version__,'backend':'torch','torch':torch.__version__,
        'synthetic_patches':len(patches),'layer_checks':details,'probability_max_abs_difference':float(np.max(np.abs(prob-expected[-1]))),
        'float64_patches':4,'float64_layer_checks':checks64,
        'all_checks_passed':True,'training_performed':False,'adapter_sha256':sha256_file(Path(__file__))})
    print('AUTHOR H5 PARITY PASSED',receipt['parameter_tensors'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    parity(contained_path(ROOT,a.model),a.output)
