#!/usr/bin/env python3
"""Same expert cohort -> WSe2 transfer. No WSe2 labels enter training/inference.

This is training-excluded material transfer, NOT a new untouched benchmark:
WSe2 was already a PhyEvo development dataset. Historical physics branches are
reported explicitly as historical references, never newly locked generalization.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from scripts.run_fair_baselines import protocol,worker,check_protocol
from tfree_detection.fair_baselines import make_model,reference_bank,predict
from tfree_detection.expert_supervision import contained_path,load_learning_dataset,normalize_image,sha256_file,write_new_json
from tfree_detection.rsi_evaluation import EvaluationProtocol,student_points,matched_expert_indices

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'datasets/public/emseek_wse2_atomic_defects'
HISTORIES={'physics_legacy_historical':'emseek_wse2_legacy_atomic_v0_2_predictions',
           'physics_periodic_historical':'emseek_wse2_periodic_v1_predictions'}

def read(path): return json.loads(path.read_text())

def prepare(output):
    if output.exists(): raise FileExistsError(output)
    dataset=ROOT/'outputs/expert_cycle_001/learning_dataset_v1'
    manifest,_=load_learning_dataset(ROOT,dataset)
    entries=[e for e in manifest['entries'] if e['action']=='correct_geometry' and e['geometry'] in ('point','interface')]
    p=protocol(dataset,manifest)
    p['folds']=[{'fold_id':'external-WSe2','kind':'training-excluded-material-transfer',
        'excluded_groups':['emseek_wse2_single_conservative_group'],
        'train_case_numbers':[e['case_number'] for e in entries],'test_case_numbers':[]}]
    p['source_hashes']['scripts/run_fair_external.py']=sha256_file(ROOT/'scripts/run_fair_external.py')
    rows=[]
    for number in range(1,108):
        img=contained_path(ROOT,BASE/'image'/f'{number}.png');lab=contained_path(ROOT,BASE/'label'/f'{number}.png')
        digest=sha256_file(img)
        if digest in {e['source_file_sha256'] for e in entries}: raise ValueError('Training/test duplicate')
        history={}
        for method,directory in HISTORIES.items():
            path=ROOT/'benchmarks/development_runs'/directory/'predictions'/f'image-{number}.json'
            record=read(path)
            if record['source_sha256']!=digest or record['annotations_read'] or record['status']!='success':
                raise ValueError('Historical source identity failed')
            history[method]={'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path)}
        rows.append({'number':number,'image':str(img.relative_to(ROOT)),'image_sha256':digest,
                     'label':str(lab.relative_to(ROOT)),'label_sha256':sha256_file(lab),'historical_physics':history})
    p['external_images']=rows
    p['external_evaluation']={'primary_step':800,'point_budgets':[20,50,100],'main_budget':50,
        'tolerance_px':6,'min_distance_px':4,'truth':'union of nonzero RGB author masks, 4-connected centroids',
        'metrics':'one-to-one precision recall F1 against author references, not chemical Gold',
        'historical_physics_tuning_on_WSe2_possible':True,'no_external_labels_used_to_fit_or_select_neural_models':True}
    output.mkdir(parents=True);write_new_json(output/'protocol.json',p)

def fit_and_predict(args):
    output=contained_path(ROOT,args.output)
    worker(SimpleNamespace(output=str(output),fold='external-WSe2',method=args.method,seed=args.seed))
    p,dataset=check_protocol(output);manifest,arrays=load_learning_dataset(ROOT,dataset)
    selected=[e for e in manifest['entries'] if e['case_number'] in p['folds'][0]['train_case_numbers']]
    bank,_=reference_bank(selected,arrays)
    run=output/'runs/external-WSe2'/args.method/f'seed-{args.seed}'
    model=make_model(args.method,ROOT).cuda()
    state=torch.load(run/'step-000800/state.pt',map_location='cuda:0',weights_only=False)
    model.load_state_dict(state['model']); del state
    target=run/'external_predictions';target.mkdir()
    records=[];began=time.monotonic()
    for row in p['external_images']:
        path=contained_path(ROOT,row['image'])
        if sha256_file(path)!=row['image_sha256']: raise ValueError('Changed transfer image')
        im=np.asarray(Image.open(path).convert('L'))
        score=predict(model,args.method,normalize_image(im),1,bank)
        out=target/f'image-{row["number"]:03d}.npz'
        with out.open('xb') as stream: np.savez_compressed(stream,score=score)
        records.append({'number':row['number'],'path':str(out.relative_to(ROOT)),'sha256':sha256_file(out)})
    write_new_json(run/'external_completed.json',{'predictions':records,'elapsed_seconds':time.monotonic()-began,
                                                 'protocol_sha256':sha256_file(output/'protocol.json')})

def launch(output):
    prepare(output);p=read(output/'protocol.json')
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    states={int(a):(int(b),int(c)) for a,b,c in (line.split(',') for line in status.splitlines())}
    gpus=list(range(2,8))
    if any(states[g][0]>1024 or states[g][1]>5 for g in gpus): raise ValueError('GPU occupied; no interference')
    jobs=[(m,s) for s in p['seeds'] for m in p['methods']];active={};failed=[]
    workers=output/'workers';workers.mkdir()
    while jobs or active:
        for gpu in gpus:
            if gpu in active or not jobs: continue
            method,seed=jobs.pop(0);name=f'{method}-{seed}';stream=(workers/f'{name}.log').open('x')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),
                 'PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                 'TORCH_HOME':str(ROOT/'.codex-tmp/torch'),'XDG_CACHE_HOME':str(ROOT/'.codex-tmp/cache'),
                 'MPLCONFIGDIR':str(ROOT/'.codex-tmp/matplotlib'),'TMPDIR':str(ROOT/'.codex-tmp')}
            cmd=[sys.executable,str(ROOT/'scripts/run_fair_external.py'),'worker','--output',str(output),'--method',method,'--seed',str(seed)]
            process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[gpu]=(process,stream,name)
            with (workers/'processes.jsonl').open('a') as log:log.write(json.dumps({'pid':process.pid,'gpu':gpu,'job':name})+'\n')
            print(f'start external gpu{gpu} pid{process.pid} {name}',flush=True)
        for gpu,(process,stream,name) in list(active.items()):
            if process.poll() is not None:
                stream.close()
                if process.returncode: failed.append(name)
                print(f'end external {name} exit={process.returncode}',flush=True);del active[gpu]
        if active: time.sleep(2)
    write_new_json(output/'execution.json',{'expected_runs':12,'failed_runs':failed,'all_workers_finished':True})
    if failed: raise RuntimeError(f'Failed runs retained {failed}')

def evaluate(output):
    p,_=check_protocol(output)
    if read(output/'execution.json')['failed_runs']: raise ValueError('Incomplete transfer experiment')
    target=output/'analysis_v1';target.mkdir(exist_ok=False)
    inventory={};audit=[]
    for seed in p['seeds']:
        schedules=set();initial={}
        for method in p['methods']:
            run=output/'runs/external-WSe2'/method/f'seed-{seed}'
            m=read(run/'step-000800/manifest.json');identity=read(run/'identity.json')
            if sha256_file(run/'step-000800/state.pt')!=m['state_sha256']:raise ValueError('Changed state')
            schedules.add(m['query_schedule_sha256']);initial[method]=identity['initial_state_sha256']
            pred=read(run/'external_completed.json')
            if pred['protocol_sha256']!=sha256_file(output/'protocol.json'):raise ValueError('Changed protocol')
            if sorted(r['number'] for r in pred['predictions'])!=list(range(1,108)):raise ValueError('Missing external score')
            for r in pred['predictions']:
                path=contained_path(ROOT,r['path'])
                if sha256_file(path)!=r['sha256']:raise ValueError('Changed external score')
                inventory[(seed,method,r['number'])]=path
        if len(schedules)!=1 or initial[p['methods'][0]]!=initial[p['methods'][1]]:raise ValueError('Unequal training')
        audit.append({'seed':seed,'query_schedule_equal':True,'paired_initialization_equal':True,'schedule_sha256':next(iter(schedules))})
    rows=[]
    for r in p['external_images']:
        label_path=contained_path(ROOT,r['label'])
        if sha256_file(label_path)!=r['label_sha256']:raise ValueError('Changed label')
        label=np.asarray(Image.open(label_path));mask=label[...,:3].any(-1)
        components,n=ndimage.label(mask)
        truth=np.asarray(ndimage.center_of_mass(mask,components,range(1,n+1))).reshape(-1,2)[:,::-1]
        histories={}
        for method,record in r['historical_physics'].items():
            path=contained_path(ROOT,record['path'])
            if sha256_file(path)!=record['sha256']:raise ValueError('Changed historical prediction')
            h=read(path);points=np.asarray(h['predicted_xy']).reshape(-1,2)
            scores=np.asarray(h['candidate_score']);indices=np.lexsort((points[:,0],points[:,1],-scores))
            histories[method]=points[indices]
        for seed in p['seeds']:
            scores={}
            for method in p['methods']:
                with np.load(inventory[(seed,method,r['number'])],allow_pickle=False) as z:scores[method]=z['score']
            for budget in p['external_evaluation']['point_budgets']:
                count=EvaluationProtocol(point_budget_per_512=budget).point_budget(mask.shape)
                predictions={m:student_points(s,count,4) for m,s in scores.items()}
                predictions.update({m:xy[:count] for m,xy in histories.items()})
                for method,xy in predictions.items():
                    tp=len(matched_expert_indices(xy,truth,6));fp=len(xy)-tp;fn=len(truth)-tp
                    rows.append({'image':r['number'],'seed':seed,'method':method,'budget':budget,'tp':tp,'fp':fp,'fn':fn})
    groups=defaultdict(list)
    for row in rows:groups[(row['method'],row['seed'],row['budget'])].append(row)
    per_seed=[]
    for (method,seed,budget),values in groups.items():
        tp,fp,fn=[sum(v[k] for v in values) for k in ('tp','fp','fn')]
        per_seed.append({'method':method,'seed':seed,'budget':budget,'tp':tp,'fp':fp,'fn':fn,
                         'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)})
    summary=[]
    for method in list(p['methods'])+list(HISTORIES):
        for budget in p['external_evaluation']['point_budgets']:
            values=[r for r in per_seed if r['method']==method and r['budget']==budget]
            row={'method':method,'budget':budget,'images':107,'conservative_source_groups':1}
            for metric in ('precision','recall','f1'):
                a=[v[metric] for v in values];row[metric+'_mean']=float(np.mean(a));row[metric+'_seed_std']=float(np.std(a,ddof=1))
            summary.append(row)
    write_new_json(target/'summary.json',{'summary':summary,'per_seed':per_seed,'per_image':rows,'training_audit':audit,
                                        'protocol':p['external_evaluation'],'score_files_verified':len(inventory)})
    with (target/'summary.csv').open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    write_new_json(target/'seal.json',{'files':{f.name:sha256_file(f) for f in target.iterdir() if f.is_file()}})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['launch','worker','evaluate'])
    parser.add_argument('--output',default='outputs/fair_baselines_external_20261014_v1')
    parser.add_argument('--method');parser.add_argument('--seed',type=int);a=parser.parse_args()
    if a.command=='worker':fit_and_predict(a)
    elif a.command=='evaluate':evaluate(contained_path(ROOT,a.output))
    else:launch(contained_path(ROOT,a.output))
