#!/usr/bin/env python3
"""Synthetic-only training -> real images, including the full Dennler pipeline.

The earlier real-Dennler training attempts are DISQUALIFIED and refused. All 37
new training images are procedural textures. No real test image or label enters
generation, training, reference selection or checkpoint selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from benchmarks.run_dennler_reference_baseline import _decode_grayscale_tiff,_decode_label,predict_fourier,_binary_counts,_counts_to_metrics
from tfree_detection.dennler_full import DennlerVGG16,dog_image,patch_classes,segment
from tfree_detection.fair_synthetic import make_training_texture,require_synthetic_training
from tfree_detection.diagnostic_v4 import analyze_posthoc_v4
from tfree_detection.expert_supervision import contained_path,sha256_file,write_new_json,normalize_image
from tfree_detection.fair_baselines import METHODS,make_model,logits_for,shared_loss,reference_bank,batch_references,predict,schedule_digest

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'datasets/public/dennler_2021_hrstem_defects/hrstem_defects_dataset.zip'
OLD=ROOT/'outputs/dennler_posthoc_v4_diagnostic_v2/predictions'
OLD_FE=ROOT/'benchmarks/emseek_fecral_fixed_transfer_v1/predictions'
DENSE_METHODS=list(METHODS)+['dennler_vgg16_eigenfilter']

def read(p):return json.loads(p.read_text())

def prepare(output):
    if output.exists():raise FileExistsError(output)
    old=read(OLD/'prediction_manifest.json');rows=[]
    if sha256_file(ARCHIVE)!=old['archive_sha256']:raise ValueError('Changed source archive')
    with ZipFile(ARCHIVE) as archive:
        for r in old['images']:
            path=contained_path(ROOT,OLD/r['prediction_path'])
            if sha256_file(path)!=r['prediction_sha256']:raise ValueError('Changed physical teacher')
            if hashlib.sha256(archive.read(r['source_relative_path'])).hexdigest()!=r['source_sha256']:raise ValueError('Image provenance mismatch')
            rows.append({'id':r['image_id'],'filename':r['filename'],'category':r['category'],'image':r['source_relative_path'],
                         'sha256':r['source_sha256'],'physical':str(path.relative_to(ROOT)),'physical_sha256':r['prediction_sha256'],
                         'split':'test'})
    external=[]
    for number in (46,148):
        base=ROOT/'datasets/public/emseek_remaining_v1/metal_alloy_defects/FeCrAl'
        image=base/'image'/f'{number}.png';label=base/'label'/f'{number}.png'
        physical=OLD_FE/'arrays'/f'fecral-{number}.npz'
        external.append({'id':f'fecral-{number}','image':str(image.relative_to(ROOT)),'image_sha256':sha256_file(image),
                         'label':str(label.relative_to(ROOT)),'label_sha256':sha256_file(label),
                         'physical':str(physical.relative_to(ROOT)),'physical_sha256':sha256_file(physical)})
    output.mkdir(parents=True);assets=output/'synthetic_training';assets.mkdir()
    training=[]
    for number in range(37):
        image,truth,metadata=make_training_texture(number)
        teacher=analyze_posthoc_v4(image).any_defect_mask
        path=assets/f'synthetic-{number:02d}.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,image=image,truth=truth,teacher=teacher)
        training.append({'id':f'synthetic-{number:02d}','source_kind':'procedural_synthetic',
                         'category':'defective' if truth.any() else 'normal','path':str(path.relative_to(ROOT)),
                         'sha256':sha256_file(path),'generation':metadata})
        print(f'prepared synthetic {number+1}/37',flush=True)
    sources=['scripts/run_fair_dense.py','src/tfree_detection/dennler_full.py','src/tfree_detection/fair_baselines.py',
             'src/tfree_detection/fair_synthetic.py','src/tfree_detection/diagnostic_v4.py',
             'src/tfree_detection/expert_student.py','benchmarks/run_dennler_reference_baseline.py',
             'vendor/fair_baselines_20261014/emseek/emseek/models/unetpp_ref.py']
    p={'methods':DENSE_METHODS,'seeds':[23,41,67],'rows':rows,'external':external,'steps':800,'batch_size':16,'patch_size':128,
       'training_source_kind':'procedural_synthetic_only','training_rows':training,
       'archive_sha256':sha256_file(ARCHIVE),'source_sha256':{s:sha256_file(ROOT/s) for s in sources},
       'external_pretraining':False,'selection':'fixed800; no real test input is used to select or fit',
       'split':'37 synthetic textures train; all 61 Dennler and 2 FeCrAl real images evaluation-only',
       'classification':'released Dennler 5% class occupancy rule, defective index0; full VGG16 with 1024/1024/512 FCs',
       'learning_rate':{'dense_models':.001,'VGG16_from_scratch':.0001},'optimizer':'Adam','gradient_clip':5,
       'preprocessing':'same source crops/rotations/flips; VGG uses released DoG+Caffe, dense models common robust normalization',
       'dense_supervision':'synthetic known positive AND negative pixels, binary any-defect; common BCE plus occupancy regularizer; synthetic-image physics teacher only in first method',
       'reference':'three positive crops from training images only, same as expert comparison',
       'evaluation':'native threshold0.5 for dense networks; published full postprocessing for Dennler; binary any-defect IoU/F1',
       'claims':{'new_locked_test':False,'each_model_optimally_tuned':False,'FeCrAl_population_generalization':False,
                 'historical_physics_development_seen_Dennler_and_FeCrAl':True},
       'port':'full VGG16, graph normal-reference selection, 49 eigenfilters, Mahalanobis, variance, original postprocessing; float64 tensor batching'}
    require_synthetic_training(p);write_new_json(output/'protocol.json',p)

def checked(output):
    if (output/'DISQUALIFIED.json').exists():raise PermissionError('Quarantined prohibited-source run cannot be scored or reused')
    p=read(output/'protocol.json')
    require_synthetic_training(p)
    for path,digest in p['source_sha256'].items():
        if sha256_file(contained_path(ROOT,path))!=digest:raise ValueError('Producer changed')
    if sha256_file(ARCHIVE)!=p['archive_sha256']:raise ValueError('Archive changed')
    return p

def load_image(archive,row,truth=False):
    payload=archive.read(row['image'])
    if hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Input changed')
    image=_decode_grayscale_tiff(payload)
    if not truth:return image
    labels=_decode_label(archive.read(row['image'].replace('/images/','/labels/')),image.shape) if row['category']=='defective' else np.zeros(image.shape,np.uint8)
    return image,labels

def training_inputs(p):
    require_synthetic_training(p)
    rows=p['training_rows'];arrays={};entries=[];original={}
    for number,r in enumerate(rows):
        path=contained_path(ROOT,r['path'])
        if sha256_file(path)!=r['sha256']:raise ValueError('Synthetic training asset changed')
        with np.load(path,allow_pickle=False) as z:image,truth,teacher=z['image'],z['truth'],z['teacher'].astype(np.float32)
        arrays[number]={'image':normalize_image(image),'human':(truth>0).astype(np.float32),'teacher':teacher}
        original[number]={'dog':dog_image(image).astype(np.uint8),'truth':truth}
        entries.append({'case_number':number,'group':r['id'],'geometry_index':3,'category':r['category']})
    return rows,entries,arrays,original

def dense_batch(rng,entries,arrays,original,device):
    values={k:[] for k in ('image','human','teacher','dog','truth')};records=[]
    for i in range(16):
        category='defective' if i%2 else 'normal';eligible=[e for e in entries if e['category']==category]
        e=eligible[int(rng.integers(len(eligible)))];n=e['case_number']
        y,x=rng.integers(0,897,size=2);rotation=int(rng.integers(4));flip=bool(rng.integers(2))
        for key in values:
            src=arrays[n][key] if key in arrays[n] else original[n][key]
            patch=np.rot90(src[y:y+128,x:x+128],rotation)
            values[key].append(patch[:,::-1] if flip else patch)
        records.append([n,int(y),int(x),rotation,flip])
    batch={k:torch.as_tensor(np.ascontiguousarray(values[k]),dtype=torch.float32,device=device) for k in ('image','human','teacher')}
    batch['image']=batch['image'][:,None];batch['valid_pixels']=torch.ones_like(batch['human'])
    batch['human_weight']=torch.ones_like(batch['human']);batch['teacher_weight']=torch.ones_like(batch['human'])
    batch['geometry_index']=torch.full((16,),3,device=device,dtype=torch.long)
    dog=torch.as_tensor(np.ascontiguousarray(values['dog']),dtype=torch.float32,device=device)[:,None]
    classes=torch.as_tensor(patch_classes(np.asarray(values['truth'])),device=device)
    return batch,dog,classes,records

def work(args):
    output=contained_path(ROOT,args.output);p=checked(output)
    if args.method not in p['methods'] or args.seed not in p['seeds']:raise ValueError('Unregistered run')
    run=output/'runs'/args.method/f'seed-{args.seed}';run.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(args.seed);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    rows,entries,arrays,original=training_inputs(p)
    # Ref-UNet receives positive references; normal images do not invent positives.
    bank,bank_audit=reference_bank([e for e in entries if arrays[e['case_number']]['human'].any()],arrays)
    model=(DennlerVGG16() if args.method=='dennler_vgg16_eigenfilter' else make_model(args.method,ROOT)).cuda()
    lr=.0001 if args.method=='dennler_vgg16_eigenfilter' else .001
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=lr)
    rng=np.random.default_rng(args.seed);schedule=hashlib.sha256();start=time.monotonic()
    initial=hashlib.sha256()
    for name,value in model.state_dict().items():initial.update(name.encode());initial.update(value.detach().cpu().numpy().tobytes())
    model.train()
    for step in range(1,801):
        batch,dog,classes,records=dense_batch(rng,entries,arrays,original,'cuda:0');schedule.update(schedule_digest(records))
        optimizer.zero_grad(set_to_none=True)
        if args.method=='dennler_vgg16_eigenfilter':loss=F.cross_entropy(model(dog),classes)
        else:
            ref=batch_references(bank,batch['geometry_index'],step=step,device='cuda:0') if args.method=='emseek_refunet_expert_only' else None
            z,raw=logits_for(model,args.method,batch['image'],batch['geometry_index'],ref)
            loss,_=shared_loss(z,batch,teacher_output=raw if args.method=='phyevo_teacher_expert' else None)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        if step==1 or step%100==0:
            event={'step':step,'loss':float(loss.detach())}
            with (run/'training.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
            print(f'{args.method} seed{args.seed} step{step} loss{event["loss"]:.5f}',flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-start
    with (run/'state.pt').open('xb') as stream:torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':800},stream)
    write_new_json(run/'training_manifest.json',{'training_ids':[r['id'] for r in rows],'query_schedule_sha256':schedule.hexdigest(),
        'reference_bank':bank_audit,'state_sha256':sha256_file(run/'state.pt'),'initial_sha256':initial.hexdigest(),
        'parameters':sum(v.numel() for v in model.parameters()),'training_seconds':seconds,'learning_rate':lr,
        'protocol_sha256':sha256_file(output/'protocol.json')})
    predictions=[]
    def infer(image,identity,cohort):
        started=time.monotonic()
        if args.method=='dennler_vgg16_eigenfilter':mask,metadata,score=segment(model,image)
        else:
            score=predict(model,args.method,normalize_image(image),3,bank);mask=(score>=.5).astype(np.uint8);metadata={}
        if not np.isfinite(score).all():raise ValueError('Nonfinite prediction; do not hide failed sample')
        path=run/f'{identity}.npz'
        with path.open('xb') as stream:np.savez_compressed(stream,mask=mask,score=score.astype(np.float32))
        predictions.append({'id':identity,'cohort':cohort,'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path),
                            'prediction_seconds':time.monotonic()-started,'metadata':metadata})
        print(f'predicted {identity} seconds={predictions[-1]["prediction_seconds"]:.1f}',flush=True)
    with ZipFile(ARCHIVE) as archive:
        for r in p['rows']:
            if r['split']=='test':infer(load_image(archive,r),r['id'],'Dennler_sim_to_real')
    for r in p['external']:
        path=contained_path(ROOT,r['image'])
        if sha256_file(path)!=r['image_sha256']:raise ValueError('External image changed')
        infer(np.asarray(Image.open(path).convert('L')),r['id'],'FeCrAl_material_stress')
    write_new_json(run/'completed.json',{'predictions':predictions,'protocol_sha256':sha256_file(output/'protocol.json')})

def launch(output,gpu_ids='2,3,4,5,6,7'):
    prepare(output);p=checked(output)
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    states={int(a):(int(b),int(c)) for a,b,c in (line.split(',') for line in status.splitlines())}
    gpus=[int(g) for g in gpu_ids.split(',')]
    if len(gpus)!=len(set(gpus)) or any(g not in range(2,8) for g in gpus):raise ValueError('Use distinct authorized experiment GPUs 2-7')
    if any(states[g][0]>1024 or states[g][1]>5 for g in gpus):raise ValueError('GPU occupied')
    jobs=[(m,s) for s in p['seeds'] for m in p['methods']];active={};failed=[]
    workers=output/'workers';workers.mkdir()
    while jobs or active:
        for gpu in gpus:
            if gpu in active or not jobs:continue
            method,seed=jobs.pop(0);name=f'{method}-{seed}';stream=(workers/f'{name}.log').open('x')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),'PYTHONDONTWRITEBYTECODE':'1',
                 'CUBLAS_WORKSPACE_CONFIG':':4096:8','TMPDIR':str(ROOT/'.codex-tmp'),'MPLCONFIGDIR':str(ROOT/'.codex-tmp/matplotlib'),
                 'XDG_CACHE_HOME':str(ROOT/'.codex-tmp/cache'),'TORCH_HOME':str(ROOT/'.codex-tmp/torch'),
                 'OPENBLAS_NUM_THREADS':'2','OMP_NUM_THREADS':'2'}
            process=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_fair_dense.py'),'worker','--output',str(output),'--method',method,'--seed',str(seed)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[gpu]=(process,stream,name)
            with (workers/'processes.jsonl').open('a') as log:log.write(json.dumps({'pid':process.pid,'gpu':gpu,'job':name})+'\n')
            print(f'start dense gpu{gpu} pid{process.pid} {name}',flush=True)
        for gpu,(process,stream,name) in list(active.items()):
            if process.poll() is not None:
                stream.close()
                if process.returncode:failed.append(name)
                print(f'end dense {name} exit={process.returncode}',flush=True);del active[gpu]
        if active:time.sleep(2)
    write_new_json(output/'execution.json',{'expected_runs':15,'failed_runs':failed,'all_workers_finished':True})
    if failed:raise RuntimeError(f'Failed runs retained {failed}')

def evaluate(output):
    p=checked(output)
    if read(output/'execution.json')['failed_runs']:raise ValueError('Incomplete experiment')
    target=output/'analysis_v1';target.mkdir(exist_ok=False);truth={};physical={};cohorts={}
    with ZipFile(ARCHIVE) as archive:
        for r in p['rows']:
            if r['split']!='test':continue
            image,label=load_image(archive,r,True);truth[r['id']]=label>0;cohorts[r['id']]='Dennler_sim_to_real'
            path=contained_path(ROOT,r['physical'])
            if sha256_file(path)!=r['physical_sha256']:raise ValueError('Physical prediction changed')
            with np.load(path,allow_pickle=False) as z:physical[r['id']]={'physics_historical':z['any_defect_region'].astype(bool)}
            physical[r['id']]['fourier_bragg']=predict_fourier(image)>0
    for r in p['external']:
        image_path=contained_path(ROOT,r['image']);label_path=contained_path(ROOT,r['label']);path=contained_path(ROOT,r['physical'])
        if sha256_file(image_path)!=r['image_sha256'] or sha256_file(label_path)!=r['label_sha256'] or sha256_file(path)!=r['physical_sha256']:raise ValueError('External asset changed')
        image=np.asarray(Image.open(image_path).convert('L'));label=np.asarray(Image.open(label_path))
        truth[r['id']]=np.any(label[...,:3]!=0,axis=-1) if label.ndim==3 else label>0
        cohorts[r['id']]='FeCrAl_material_stress'
        with np.load(path,allow_pickle=False) as z:physical[r['id']]={'physics_historical':z['v04_score_only'].astype(bool)}
        physical[r['id']]['fourier_bragg']=predict_fourier(image)>0
    rows=[];audits=[]
    for seed in p['seeds']:
        schedules=set();initial={}
        for method in p['methods']:
            run=output/'runs'/method/f'seed-{seed}';m=read(run/'training_manifest.json');done=read(run/'completed.json')
            if m['protocol_sha256']!=sha256_file(output/'protocol.json') or done['protocol_sha256']!=m['protocol_sha256']:raise ValueError('Changed protocol')
            if sha256_file(run/'state.pt')!=m['state_sha256']:raise ValueError('Changed checkpoint')
            if m['training_ids']!=[r['id'] for r in p['training_rows']]:raise ValueError('Unequal training cohort')
            schedules.add(m['query_schedule_sha256']);initial[method]=m['initial_sha256']
            if sorted(r['id'] for r in done['predictions'])!=sorted(truth):raise ValueError('Missing test image')
            for r in done['predictions']:
                path=contained_path(ROOT,r['path'])
                if sha256_file(path)!=r['sha256']:raise ValueError('Changed prediction')
                with np.load(path,allow_pickle=False) as z:mask=z['mask']>0
                counts=_binary_counts(mask,truth[r['id']]);rows.append({'id':r['id'],'cohort':cohorts[r['id']],'method':method,'seed':seed,**counts})
        if len(schedules)!=1 or initial[p['methods'][0]]!=initial[p['methods'][1]]:raise ValueError('Unequal exposure or paired initialization')
        audits.append({'seed':seed,'query_schedule_equal':True,'paired_initialization_equal':True,'sha256':next(iter(schedules))})
        for identity,methods in physical.items():
            for method,mask in methods.items():rows.append({'id':identity,'cohort':cohorts[identity],'method':method,'seed':seed,**_binary_counts(mask,truth[identity])})
    groups=defaultdict(list)
    for r in rows:groups[(r['cohort'],r['method'],r['seed'])].append(r)
    per_seed=[]
    for (cohort,method,seed),values in groups.items():
        total={k:sum(r[k] for r in values) for k in ('true_positive_pixels','false_positive_pixels','false_negative_pixels','true_negative_pixels')}
        per_seed.append({'cohort':cohort,'method':method,'seed':seed,'images':len(values),**_counts_to_metrics(total)})
    summary=[]
    for cohort in sorted(set(cohorts.values())):
        for method in p['methods']+['physics_historical','fourier_bragg']:
            values=[r for r in per_seed if r['cohort']==cohort and r['method']==method]
            row={'cohort':cohort,'method':method,'images':values[0]['images']}
            for metric in ('iou','f1','precision','recall'):
                a=[v[metric] for v in values];row[metric+'_mean']=float(np.mean(a));row[metric+'_seed_std']=float(np.std(a,ddof=1))
            summary.append(row)
    write_new_json(target/'summary.json',{'rows':summary,'per_seed':per_seed,'per_image':rows,'fairness_audit':audits,'claims':p['claims']})
    with (target/'summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write_new_json(target/'seal.json',{'files':{f.name:sha256_file(f) for f in target.iterdir() if f.is_file()}})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['launch','worker','evaluate'])
    parser.add_argument('--output',default='outputs/fair_baselines_synthetic_transfer_20261014_v1');parser.add_argument('--method');parser.add_argument('--seed',type=int)
    parser.add_argument('--gpus',default='2,3,4,5,6,7')
    a=parser.parse_args()
    if a.command=='worker':work(a)
    elif a.command=='evaluate':evaluate(contained_path(ROOT,a.output))
    else:launch(contained_path(ROOT,a.output),a.gpus)
