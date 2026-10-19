"""Equal-data, no-pretraining comparison adapters. No production route changes.

Sparse expert positives stay positive/unknown. The shared occupancy penalty is
an explicit weak regularizer, not a declaration that unknown pixels are normal.
EMSeek uses the pinned upstream complete reference network, not a mock model.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .expert_student import make_expert_student, geometry_logits, expert_adaptation_loss

METHODS = ('phyevo_teacher_expert', 'same_student_expert_only', 'unet_expert_only', 'emseek_refunet_expert_only')

def make_folds(entries):
    eligible = [e for e in entries if e['action'] == 'correct_geometry' and e['geometry'] in ('point', 'interface')]
    groups = sorted({e['group'] for e in eligible})
    pairs = [(f'source-{i:02d}', [g], 'acquisition-held-out') for i, g in enumerate(groups)]
    al = [g for g in groups if g.startswith('sigma3-')]
    if len(al) == 2:
        pairs.append(('material-Al-held-out', al, 'Al-material-held-out-ZrO2-interface-training'))
    result = []
    for name, excluded, kind in pairs:
        train = [e for e in eligible if e['group'] not in excluded]
        test = [e for e in eligible if e['group'] in excluded]
        if not train or not test or {e['source_file_sha256'] for e in train} & {e['source_file_sha256'] for e in test}:
            raise ValueError('Source file leakage or empty fold')
        if not {e['geometry'] for e in test} <= {e['geometry'] for e in train}:
            raise ValueError('Held-out task has no matching training task')
        result.append({'fold_id': name, 'kind': kind, 'excluded_groups': excluded,
                       'train_case_numbers': [e['case_number'] for e in train],
                       'test_case_numbers': [e['case_number'] for e in test]})
    return result

class DoubleConv(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU(),
                         nn.Conv2d(b,b,3,padding=1), nn.BatchNorm2d(b), nn.ReLU())

class UNet(nn.Module):
    """Conventional four-scale U-Net, base width 32, learned up-convolutions."""
    def __init__(self):
        super().__init__()
        channels = [32,64,128,256,512]
        self.down = nn.ModuleList(DoubleConv(a,b) for a,b in zip([1]+channels, channels))
        self.up = nn.ModuleList(nn.ConvTranspose2d(a,b,2,stride=2) for a,b in zip(channels[:0:-1], channels[-2::-1]))
        self.merge = nn.ModuleList(DoubleConv(2*b,b) for b in channels[-2::-1])
        self.head = nn.Conv2d(32,5,1)
    def forward(self,x,reference=None):
        skips=[]
        for i, block in enumerate(self.down):
            if i:
                x=F.max_pool2d(x,2)
            x=block(x); skips.append(x)
        for up,merge,skip in zip(self.up,self.merge,skips[-2::-1]):
            x=up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x=F.interpolate(x,size=skip.shape[-2:],mode='bilinear',align_corners=False)
            x=merge(torch.cat([x,skip],1))
        return self.head(x)

def make_model(method, root):
    if method not in METHODS:
        raise ValueError('Unknown method')
    if method in METHODS[:2]:
        return make_expert_student(16)
    if method == 'unet_expert_only':
        return UNet()
    path = Path(root) / 'vendor/fair_baselines_20261014/emseek/emseek/models/unetpp_ref.py'
    spec = importlib.util.spec_from_file_location('pinned_emseek_ref', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Same resnet18/prompt=True as the official segmentor agent. Only the
    # supported class count changes for our geometry heads. NO weight download.
    return module.UnetPlusPlus(encoder_name='resnet18', pretrained=False,
                               prompt=True, classes=5)

def logits_for(model, method, image, geometry, reference=None):
    if method in METHODS[:2]:
        output=model(image)
    elif method == 'emseek_refunet_expert_only':
        if reference is None:
            raise ValueError('Reference-guided model requires a training reference')
        output=SimpleNamespace(geometry_logits=model(image.repeat(1,3,1,1), reference.repeat(1,3,1,1)))
    else:
        output=SimpleNamespace(geometry_logits=model(image))
    return geometry_logits(output,geometry), output

def shared_loss(logits, batch, *, teacher_output=None):
    losses=[]
    for i in range(len(logits)):
        weight=batch['human_weight'][i]*batch['valid_pixels'][i]
        if bool((weight>0).any()):
            bce=F.binary_cross_entropy_with_logits(logits[i],batch['human'][i],reduction='none')
            losses.append((bce*weight).sum()/weight.sum())
    human=torch.stack(losses).mean() if losses else logits.sum()*0
    valid=batch['valid_pixels']
    occupancy=(torch.sigmoid(logits)*valid).sum((1,2))/valid.sum((1,2)).clamp(min=1)
    ceiling=torch.where(batch['geometry_index']==1,0.03,0.15)
    prior=torch.relu(occupancy-ceiling).square().mean()
    teacher=logits.sum()*0
    if teacher_output is not None:
        _, parts=expert_adaptation_loss(teacher_output,batch,use_expert=True)
        teacher=parts['teacher_positive']+0.25*parts['teacher_background_prior']
    return 2*human+2*prior+teacher, {'human':human,'occupancy_prior':prior,'teacher':teacher}

def reference_bank(entries, arrays, *, excluded_groups=(), count=3, size=64):
    """At most three fixed positive exemplars/task, selected only from training.

    Pixel order, not test similarity or best test score, chooses exemplars.
    The bank is available to every method; only Ref-UNet consumes reference input.
    """
    if any(e['group'] in excluded_groups for e in entries):
        raise ValueError('Held-out source in reference bank')
    result={}; audit=[]
    for geo in sorted({e['geometry_index'] for e in entries}):
        candidates=sorted([e for e in entries if e['geometry_index']==geo],key=lambda e:e['case_number'])
        chosen=np.unique(np.linspace(0,len(candidates)-1,min(count,len(candidates)),dtype=int))
        patches=[]
        for i in chosen:
            e=candidates[int(i)]; plane=arrays[e['case_number']]
            ys,xs=np.where(plane['human']>0)
            if not len(ys):
                raise ValueError('No positive training reference')
            mid=len(ys)//2; h,w=plane['image'].shape
            y=int(np.clip(ys[mid]-size//2,0,max(0,h-size)))
            x=int(np.clip(xs[mid]-size//2,0,max(0,w-size)))
            patch=plane['image'][y:y+size,x:x+size]
            patch=np.pad(patch,((0,size-patch.shape[0]),(0,size-patch.shape[1])),mode='reflect').astype(np.float32)
            patches.append(patch)
            audit.append({'case_number':e['case_number'],'group':e['group'],'geometry_index':geo,
                          'yx':[y,x],'size':size,'sha256':hashlib.sha256(patch.tobytes()).hexdigest()})
        result[geo]=np.stack(patches)
    return result,audit

def batch_references(bank, geometries, *, step, device):
    values=[bank[int(g)][(step+i)%len(bank[int(g)])] for i,g in enumerate(geometries.detach().cpu().tolist())]
    return torch.as_tensor(np.stack(values)[:,None],device=device)

def predict(model, method, image, geometry, bank, *, device='cuda:0',tile_size=256,overlap=64):
    h,w=image.shape
    score=np.zeros((h,w),np.float32); weights=np.zeros_like(score)
    starts=lambda n: sorted(set(range(0,max(1,n-tile_size+1),tile_size-overlap))|{max(0,n-tile_size)})
    model.eval()
    with torch.inference_mode():
        for y in starts(h):
            for x in starts(w):
                tile=image[y:y+tile_size,x:x+tile_size]; th,tw=tile.shape
                ph,pw=max(32,32*((th+31)//32)),max(32,32*((tw+31)//32))
                tile=np.pad(tile,((0,ph-th),(0,pw-tw)),mode='reflect')
                tensor=torch.as_tensor(np.ascontiguousarray(tile)[None,None],device=device)
                geo=torch.tensor([geometry],device=device)
                if method=='emseek_refunet_expert_only':
                    values=[]
                    for patch in bank[geometry]:
                        ref=torch.as_tensor(patch[None,None],device=device)
                        z,_=logits_for(model,method,tensor,geo,ref)
                        values.append(torch.sigmoid(z))
                    value=torch.stack(values).mean(0)[0,:th,:tw].cpu().numpy()
                else:
                    z,_=logits_for(model,method,tensor,geo)
                    value=torch.sigmoid(z)[0,:th,:tw].cpu().numpy()
                win=np.maximum(np.outer(np.hanning(th),np.hanning(tw)),.05).astype(np.float32)
                score[y:y+th,x:x+tw]+=value*win; weights[y:y+th,x:x+tw]+=win
    return score/weights

def schedule_digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).digest()
