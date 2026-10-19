"""Procedural, image-independent dense training inputs for simulation transfer.

No experimental image, annotation, FFT fit, checkpoint or test statistic enters
generation. This is a texture benchmark, not a multislice electron simulation.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter

def make_training_texture(index:int,size:int=1024):
    if index not in range(37) or size<128:raise ValueError('Unregistered synthetic image')
    seed=83100+index;rng=np.random.default_rng(seed)
    yy,xx=np.mgrid[:size,:size]
    spacing=float(rng.uniform(10,22));theta=float(rng.uniform(0,np.pi));angle=float(rng.choice([np.pi/2,np.pi/3]))
    phase=rng.uniform(-np.pi,np.pi,3)
    def lattice(rotation):
        x=(np.cos(rotation)*xx+np.sin(rotation)*yy)/spacing
        y=(np.cos(rotation+angle)*xx+np.sin(rotation+angle)*yy)/(spacing*rng_ratio)
        signal=np.cos(2*np.pi*x+phase[0])+.85*np.cos(2*np.pi*y+phase[1])+.3*np.cos(2*np.pi*(x+y)+phase[2])
        return signal/signal.std()
    rng_ratio=float(rng.uniform(.85,1.15));crystal=lattice(theta)
    cx,cy=rng.uniform(.27,.73,2)*size;rx,ry=rng.uniform(.12,.24,2)*size
    mask=((xx-cx)/rx)**2+((yy-cy)/ry)**2<=1
    truth=np.zeros((size,size),np.uint8)
    kind='normal' if index<18 else 'amorphous' if index<28 else 'secondary_domain'
    if kind=='normal':image=crystal
    else:
        blend=gaussian_filter(mask.astype(float),1.8)
        if kind=='amorphous':
            other=gaussian_filter(rng.normal(size=(size,size)),rng.uniform(1,2.5))
            other=other/max(other.std(),1e-12)*rng.uniform(.25,1.)
            truth[mask]=2
        else:
            other=lattice(theta+rng.uniform(.17,.60));truth[mask]=1
        image=(1-blend)*crystal+blend*other
    image=image+.15*xx/size-.1*yy/size+rng.normal(0,rng.uniform(.03,.22),(size,size))
    lo,hi=np.percentile(image,[.5,99.5]);image=np.rint(np.clip((image-lo)/(hi-lo),0,1)*255).astype(np.uint8)
    return image,truth,{'seed':seed,'kind':kind,'spacing':spacing,'rotation_radians':theta,
                        'basis_angle_radians':angle,'size':size,'experimental_inputs':[]}

def require_synthetic_training(protocol):
    if protocol.get('training_source_kind')!='procedural_synthetic_only':
        raise PermissionError('Real Dennler archive is locked evaluation-only; training is prohibited')
    if len(protocol.get('training_rows',[]))!=37 or any(r.get('split')=='train' for r in protocol.get('rows',[])):
        raise PermissionError('Synthetic-only training inventory violated')
    if any(r.get('source_kind')!='procedural_synthetic' for r in protocol['training_rows']):
        raise PermissionError('Experimental data cannot enter synthetic training')
