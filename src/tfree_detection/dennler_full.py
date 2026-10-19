"""Complete VGG16 -> graph reference -> eigenfilter/variance segmentation.

Numerical port of MIT-licensed nkdnnlr/TEMDefectClassification at
72d37c3ca5a84cfa39337c7243a80e3ffcc0f648. Preserve the released eigenvector-row
ordering (including its row/column convention), graph scoring, edge factors and
postprocessing. Tensor batching replaces pixel loops, not a shorter algorithm.
No pretrained weights are loaded; author-model reproduction is a separate track.
"""
from __future__ import annotations
import math
import cv2
import networkx as nx
import numpy as np
from scipy import ndimage
from skimage.filters import gaussian
import torch
from torch import nn
from torch.nn import functional as F

class DennlerVGG16(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[];incoming=3
        for out in [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M']:
            if out=='M':layers.append(nn.MaxPool2d(2,2))
            else:layers.extend([nn.Conv2d(incoming,out,3,padding=1),nn.ReLU()]);incoming=out
        self.features=nn.Sequential(*layers)
        self.classifier=nn.Sequential(nn.Linear(8192,1024),nn.ReLU(),nn.Dropout(.5),
            nn.Linear(1024,1024),nn.ReLU(),nn.Dropout(.5),nn.Linear(1024,512),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,2))
    def forward(self,image):
        # Keras caffe preprocessing on grayscale replicated into BGR channels.
        value=image.repeat(1,3,1,1)-image.new_tensor([103.939,116.779,123.68])[None,:,None,None]
        value=self.features(value)
        # Keras Flatten() is NHWC, important for released H5 weight parity.
        return self.classifier(value.permute(0,2,3,1).reshape(len(value),-1))

def dog_image(image):
    value=gaussian(image,2,preserve_range=True)-gaussian(image,3,preserve_range=True)
    return np.interp(value,(value.min(),value.max()),(0,254)).astype(np.float64)

def patch_classes(truth):
    """Released >=5% class-2 OR intermediate class-1 occupancy rule."""
    second=(truth==2).mean(axis=(-2,-1));first=(truth==1).mean(axis=(-2,-1))
    defect=(second>.05)|((first>.05)&(first<.95))
    return np.where(defect,0,1).astype(np.int64) # original class order: defective, non_defective

def best_reference(classes):
    arr=np.asarray(classes,dtype=int);h,w=arr.shape
    def choose(connectivity):
        graph=nx.Graph();graph.add_nodes_from(range(h*w))
        offsets=[(-1,0),(0,-1)] if connectivity==4 else [(-1,0),(0,-1),(-1,-1),(-1,1)]
        for y in range(h):
            for x in range(w):
                for dy,dx in offsets:
                    yy,xx=y+dy,x+dx
                    if 0<=yy<h and 0<=xx<w and arr[y,x]==arr[yy,xx]:graph.add_edge(y*w+x,yy*w+xx)
        normal=np.flatnonzero(arr.ravel()==0)
        sub=graph.subgraph(normal if len(normal) else list(graph.nodes))
        values={}
        for node in sorted(sub.nodes):
            neighbors=nx.single_source_shortest_path_length(sub,node,cutoff=10)
            values[node]=sum(sum(v==i for v in neighbors.values())*(2*connectivity)**(1-i)/math.factorial(i) for i in range(1,11))
        node=max(values,key=values.get)
        return node,values[node]
    node,value=choose(4)
    if value==0:node,value=choose(8)
    return int(node)

def energy_tensor(patches,filters,blur=16):
    size=filters.shape[-1];pad=size//2
    responses=F.conv2d(F.pad(patches,(pad,pad,pad,pad),mode='reflect'),filters.flip(-2,-1))
    # ndimage.convolve with an even uniform kernel centers one pixel to the
    # right of uniform_filter's default: left=blur//2-1, right=blur//2.
    left=(blur-1)//2;right=blur//2
    return F.avg_pool2d(F.pad(responses.square(),(left,right,left,right)),blur,stride=1)*blur**2

def edge_factors(shape,edge=16,patch=128):
    yy,xx=np.indices(shape);factor=np.ones(shape)
    for n in range(24):
        factor*=np.where((yy>=edge+(.75+n/2)*patch)&(yy<edge+(1+n/2)*patch),2,1)
        factor*=np.where((xx>=edge+(.75+n/2)*patch)&(xx<edge+(1+n/2)*patch),2,1)
    top=yy<edge+.5*patch;left=xx<edge+.5*patch
    bottom=yy>shape[0]-edge-.25*patch;right=xx>shape[1]-edge-.25*patch
    factor*=np.where(top|left|bottom|right,2,1)
    for corner in (top&left,top&right,bottom&left,bottom&right):factor*=np.where(corner,2,1)
    return factor

def eigenfilter_score(image,reference,*,device='cuda:0',batch_size=12):
    samples=np.lib.stride_tricks.sliding_window_view(reference,(7,7)).reshape(-1,49)
    covariance=np.cov(samples.T)
    eigenvalues,eigenvectors=np.linalg.eig(covariance)
    filters=eigenvectors[eigenvalues.argsort()[::-1]].reshape(49,7,7) # preserve upstream row ordering
    if np.iscomplexobj(filters):raise ValueError('Complex covariance eigenvectors')
    kernels=torch.as_tensor(filters[:,None].copy(),dtype=torch.float64,device=device)
    with torch.inference_mode():
        good=energy_tensor(torch.as_tensor(reference[None,None],dtype=torch.float64,device=device),kernels)[0]
        ref=good.permute(1,2,0).reshape(-1,49)
        cov=np.cov(ref.cpu().numpy(),rowvar=False)
        inverse=np.linalg.inv(cov) # no unreported ridge/pseudoinverse replacement
        inv=torch.as_tensor(inverse,dtype=torch.float64,device=device)
        positions=[(y,x) for y in range(0,image.shape[0]-127,64) for x in range(0,image.shape[1]-127,64)]
        output=np.zeros(image.shape,np.float64)
        for start in range(0,len(positions),batch_size):
            coords=positions[start:start+batch_size]
            patches=np.stack([image[y:y+128,x:x+128] for y,x in coords])
            values=energy_tensor(torch.as_tensor(patches[:,None],device=device),kernels)
            diff=values.permute(0,2,3,1).reshape(len(coords),-1,49)-ref
            squared=((diff@inv)*diff).sum(-1)
            if squared.min() < -1e-5:raise ValueError('Nonpositive numerical Mahalanobis distance')
            distances=torch.sqrt(squared.clamp(min=0)).reshape(-1,128,128).cpu().numpy()
            distances[:,:16,:]=0;distances[:,-16:,:]=0;distances[:,:,:16]=0;distances[:,:,-16:]=0
            for (y,x),distance in zip(coords,distances):output[y:y+128,x:x+128]+=distance
    return output*edge_factors(image.shape)

def localvariance(image):
    mean,square=(cv2.boxFilter(v,-1,(20,20),borderType=cv2.BORDER_DEFAULT) for v in (image,image*image))
    return square-mean*mean

def postprocess(score,image,reference):
    var=localvariance(image);minimum=localvariance(reference).min()
    class2=cv2.GaussianBlur(var*255,(101,101),sigmaX=10)/255 < minimum/1.2
    positive=score[score>0]
    symmetry=np.zeros(score.shape,bool)
    if len(positive):
        cut=np.maximum(score,positive.min())
        if np.max(cut)>90:
            normalized=(cut-cut.min())/(cut.max()-cut.min())
            blurred=cv2.GaussianBlur(normalized*255,(101,101),sigmaX=6)/255
            _,binary=cv2.threshold(np.uint8(255*blurred),127,255,cv2.THRESH_TRIANGLE)
            symmetry=cv2.medianBlur(binary,19)>0
    result=symmetry.astype(np.uint8);result[class2]=2
    return result

def segment(model,image,*,device='cuda:0'):
    if image.ndim!=2 or min(image.shape)<128 or any(v%128 for v in image.shape):
        raise ValueError('Published pipeline requires full 128px tiling; do not silently rescale')
    pre=dog_image(image)
    patches=np.stack([pre[y:y+128,x:x+128] for y in range(0,image.shape[0],128) for x in range(0,image.shape[1],128)])
    probabilities=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(patches),16):
            logits=model(torch.as_tensor(patches[start:start+16,None],dtype=torch.float32,device=device))
            probabilities.extend(torch.softmax(logits,1)[:,0].cpu().numpy().tolist())
    classes=np.rint(probabilities).astype(int).reshape(image.shape[0]//128,image.shape[1]//128)
    index=best_reference(classes);reference=patches[index]
    score=eigenfilter_score(pre,reference,device=device)
    return postprocess(score,pre,reference),{'reference_patch':index,'defective_patch_count':int(classes.sum()),
                                          'patch_probabilities':probabilities},score
