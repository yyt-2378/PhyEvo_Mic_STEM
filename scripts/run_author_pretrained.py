#!/usr/bin/env python3
"""Registered, inference-only author-weight comparison; labels scored separately.

Dennler native outputs retain the complete classifier/reference/eigenfilter/
variance pipeline. Its eigenfilter map is additionally evaluated with the same
expert point/area budget, explicitly as an adaptation (not a native point head).
No fine-tuning, label-dependent thresholds, or per-test model selection.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile
import numpy as np
from PIL import Image
from scipy import ndimage
import torch
from scripts.dennler_author_weights import load_author_model
from scripts.run_rsi_nested import idle_gpus
from scripts.run_fair_dense import load_image,ARCHIVE
from scripts.verify_rsi_cycle import require
from tfree_detection.dennler_full import segment
from tfree_detection.expert_supervision import contained_path,sha256_file,write_new_json,load_learning_dataset,verify_frozen_round
from tfree_detection.review_dataset import ReviewSample,load_review_plane
from tfree_detection.rsi_evaluation import EvaluationProtocol,Prediction,evaluate_case,student_points,student_mask,budgeted_physics_mask,matched_expert_indices
from benchmarks.run_dennler_reference_baseline import _binary_counts,_counts_to_metrics

ROOT=Path(__file__).resolve().parents[1]
DEFAULT='outputs/author_pretrained_20261014_v1'
BEST=ROOT/'outputs/best_release_20261014_v1'

def read(p):return json.loads(p.read_text())

def label_points(path):
    label=np.asarray(Image.open(path));mask=label[...,:3].any(-1)
    components,n=ndimage.label(mask)
    return np.asarray(ndimage.center_of_mass(mask,components,range(1,n+1))).reshape(-1,2)[:,::-1]

def prepare(output,model,parity):
    if output.exists():raise FileExistsError(output)
    receipt=read(parity);require(receipt['all_checks_passed'] and receipt['source_sha256']==sha256_file(model),'Author weight parity not established')
    require(receipt['adapter_sha256']==sha256_file(ROOT/'scripts/dennler_author_weights.py'),'Adapter changed since parity')
    dense=read(ROOT/'outputs/fair_baselines_synthetic_transfer_20261014_v1/protocol.json')
    dataset=ROOT/'outputs/expert_cycle_001/learning_dataset_v1';manifest,arrays=load_learning_dataset(ROOT,dataset)
    records={r['case_number']:r for r in verify_frozen_round(ROOT,contained_path(ROOT,manifest['frozen_bundle']))}
    split_paths={name:model.parent/f'{name}.txt' for name in ['train','test']}
    split_names={name:{s.split(': ',1)[1] for s in path.read_text().splitlines() if s.strip()} for name,path in split_paths.items()}
    require(not split_names['train']&split_names['test'],'Author split lists overlap')
    require(all(names<={r['filename'] for r in dense['rows']} for names in split_names.values()),'Author split filename missing from dataset')
    tasks=[]
    for r in dense['rows']:
        subset=('author_train_list' if r['filename'] in split_names['train'] else 'author_test_list' if r['filename'] in split_names['test']
                else 'unlisted_defective' if r['category']=='defective' else 'normal_training_status_unspecified')
        tasks.append({'kind':'dennler','id':r['id'],'input':{**r,'author_split':subset}})
    for r in dense['external']:tasks.append({'kind':'fecral','id':r['id'],'input':r})
    for e in manifest['entries']:
        if e['action']=='correct_geometry' and e['geometry'] in ('point','interface'):
            r=records[e['case_number']]
            tasks.append({'kind':'expert','id':f'case-{e["case_number"]:03d}','case_number':e['case_number'],
                'group':e['group'],'geometry':e['geometry'],'sample':r['sample'],
                'source_sha256':e['source_file_sha256'],'pixel_sha256':r['prediction']['pixel_sha256']})
    for r in read(ROOT/'outputs/fair_baselines_external_20261014_v1/protocol.json')['external_images']:
        tasks.append({'kind':'wse2','id':f'wse2-{r["number"]:03d}','input':r})
    for i,t in enumerate(tasks):t['index']=i
    source_names=['scripts/run_author_pretrained.py','scripts/dennler_author_weights.py','scripts/run_fair_dense.py',
        'scripts/run_fair_external.py','src/tfree_detection/dennler_full.py','src/tfree_detection/rsi_evaluation.py',
        'src/tfree_detection/review_dataset.py','src/tfree_detection/expert_supervision.py','benchmarks/run_dennler_reference_baseline.py']
    p={'model':str(model.relative_to(ROOT)),'model_sha256':sha256_file(model),'parity':str(parity.relative_to(ROOT)),
       'parity_sha256':sha256_file(parity),'tasks':tasks,'source_sha256':{f:sha256_file(ROOT/f) for f in source_names},
       'dataset':str(dataset.relative_to(ROOT)),'dataset_manifest_sha256':sha256_file(dataset/'manifest.json'),
       'best_release_manifest_sha256':sha256_file(BEST/'manifest.json'),'archive_sha256':dense['archive_sha256'],
       'evaluation':read(BEST/'manifest.json')['evaluation'],
       'inference_only':True,'training_performed':False,'label_dependent_selection':False,
       'native_pipeline':'full VGG16 -> graph normal reference -> 49 eigenfilters/Mahalanobis -> variance and released postprocessing',
       'padding':'reflect right/bottom to next multiple of128 if necessary; no resizing; crop outputs back to original field',
       'expert_adaptation':'rank raw eigenfilter score with exactly frozen point/area budgets; this is not native atomic-point detection; native interface mask and budget-thinned native mask also reported',
       'wse2_adaptation':'rank eigenfilter score at50 points per512square, min-distance4, tolerance6, same published-mask centroid truth',
       'author_split_lists':{name:{'path':str(path.relative_to(ROOT)),'sha256':sha256_file(path),'files':sorted(split_names[name])} for name,path in split_paths.items()},
       'pretraining_overlap':{'dennler':'22 explicit train,6 explicit test,3 unlisted defective,30 normals with unspecified training status; report strata separately, all61 not independent generalization',
          'expert':'different public collections; exact original-author training inventory not released, independence not proven',
          'fecral_wse2':'different published collections; no local tuning, but historical PhyEvo development and author overlap caveats apply'},
       'equal_training_data_claim':False,'new_blind_test':False}
    output.mkdir(parents=True);write_new_json(output/'protocol.json',p)
    print('REGISTERED',len(tasks),'inference images',flush=True)

def checked(output):
    p=read(output/'protocol.json')
    for path,digest in p['source_sha256'].items():require(sha256_file(contained_path(ROOT,path))==digest,'Inference producer changed')
    require(sha256_file(contained_path(ROOT,p['model']))==p['model_sha256'],'Author model changed')
    require(sha256_file(contained_path(ROOT,p['parity']))==p['parity_sha256'],'Parity changed')
    for r in p['author_split_lists'].values():require(sha256_file(contained_path(ROOT,r['path']))==r['sha256'],'Author split list changed')
    return p

def infer_image(task,archive):
    if task['kind']=='dennler':return load_image(archive,task['input'])
    if task['kind']=='expert':
        sample=ReviewSample.from_dict(task['sample']);path=contained_path(ROOT,sample.source_path)
        require(sha256_file(path)==task['source_sha256'],'Expert source changed')
        image=load_review_plane(sample,ROOT);a=np.ascontiguousarray(image);digest=hashlib.sha256()
        digest.update(str(a.dtype).encode());digest.update(np.asarray(a.shape,dtype=np.int64).tobytes());digest.update(a.view(np.uint8).tobytes())
        require(digest.hexdigest()==task['pixel_sha256'],'Reviewed pixels differ')
        return image
    r=task['input'];path=contained_path(ROOT,r['image']);require(sha256_file(path)==r['image_sha256'],'External source changed')
    return np.asarray(Image.open(path).convert('L'))

def work(output,shard,shards):
    p=checked(output);require(sha256_file(ARCHIVE)==p['archive_sha256'],'Dense image archive changed')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model,mapping=load_author_model(contained_path(ROOT,p['model']),'cuda:0');failed=[]
    with ZipFile(ARCHIVE) as archive:
        for t in p['tasks'][shard::shards]:
            dest=output/'predictions'/t['id'];dest.mkdir(parents=True,exist_ok=False);start=time.monotonic()
            try:
                image=infer_image(t,archive);h,w=image.shape;pad=((-h)%128,(-w)%128)
                padded=np.pad(image,((0,pad[0]),(0,pad[1])),mode='reflect') if any(pad) else image
                labels,meta,score=segment(model,padded,device='cuda:0');labels=labels[:h,:w];score=score[:h,:w]
                require(np.isfinite(score).all(),'Nonfinite author score')
                with (dest/'prediction.npz').open('xb') as f:np.savez_compressed(f,labels=labels,score=score)
                write_new_json(dest/'receipt.json',{'task':t,'status':'success','shape':[h,w],'padding':list(pad),
                    'prediction_sha256':sha256_file(dest/'prediction.npz'),'protocol_sha256':sha256_file(output/'protocol.json'),
                    'model_sha256':p['model_sha256'],'seconds':time.monotonic()-start,'metadata':meta,'labels_used_for_prediction':False})
                print(t['id'],'done',round(time.monotonic()-start,2),flush=True)
            except Exception as e:
                failed.append(t['id']);write_new_json(dest/'failure.json',{'task':t,'error':str(e),'traceback':traceback.format_exc()})
                print(t['id'],'FAILED',str(e),flush=True)
    write_new_json(output/f'shard-{shard:02d}.json',{'failed':failed,'shards':shards,'shard':shard,'model_mapping':mapping})
    require(not failed,'Some images failed; retained, not silently excluded')

def launch(output,gpus):
    checked(output);ids=[int(v) for v in gpus.split(',')];require(len(ids)==len(set(ids)),'Repeated GPU');idle_gpus(ids)
    logdir=output/'workers';logdir.mkdir();active=[]
    for shard,gpu in enumerate(ids):
        log=(logdir/f'shard-{shard}.log').open('x');env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),
            'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
            'TMPDIR':str(ROOT/'.codex-tmp'),'MPLCONFIGDIR':str(ROOT/'.codex-tmp/matplotlib')}
        proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--output',str(output),'--shard',str(shard),'--shards',str(len(ids))],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        active.append((proc,log,shard));print('start',shard,gpu,proc.pid,flush=True)
    write_new_json(logdir/'processes.json',{'workers':[{'pid':p.pid,'shard':s,'gpu':ids[s]} for p,_,s in active]})
    exitcodes=[]
    for proc,log,shard in active:code=proc.wait();log.close();exitcodes.append(code);print('end',shard,code,flush=True)
    write_new_json(output/'execution.json',{'exitcodes':exitcodes})
    require(not any(exitcodes),'Inference has failures; review all before scoring')

def score(output):
    p=checked(output);require(all(v==0 for v in read(output/'execution.json')['exitcodes']),'Incomplete inference')
    dataset=contained_path(ROOT,p['dataset']);require(sha256_file(dataset/'manifest.json')==p['dataset_manifest_sha256'],'Expert labels changed')
    manifest,arrays=load_learning_dataset(ROOT,dataset);entries={e['case_number']:e for e in manifest['entries']};ep=EvaluationProtocol(**p['evaluation'])
    dense=[];expert=[];wse=[];normal=defaultdict(list)
    with ZipFile(ARCHIVE) as archive:
        for t in p['tasks']:
            folder=output/'predictions'/t['id'];receipt=read(folder/'receipt.json');require(receipt['task']==t and not receipt['labels_used_for_prediction'],'Inference identity mismatch')
            require(receipt['protocol_sha256']==sha256_file(output/'protocol.json'),'Prediction protocol mismatch')
            require(sha256_file(folder/'prediction.npz')==receipt['prediction_sha256'],'Prediction changed')
            with np.load(folder/'prediction.npz',allow_pickle=False) as a:labels,values=a['labels'],a['score']
            if t['kind']=='expert':
                n=t['case_number'];e=entries[n];meta={'selection_uses_expert_labels':False,'author_pretrained':True,'adapted_from_region_detector':True}
                if e['geometry']=='point':
                    pred=Prediction('point',points_xy=student_points(values,ep.point_budget(values.shape),ep.point_min_distance_px),budget=ep.point_budget(values.shape),metadata=meta)
                else:pred=Prediction(e['geometry'],mask=student_mask(values,ep.area_budget(values.shape)),budget=ep.area_budget(values.shape),metadata=meta)
                preds={'author_eigenfilter_score_equal_budget':pred}
                if e['geometry']!='point':
                    preds['author_native_region']=Prediction(e['geometry'],mask=labels>0,metadata={'selection_uses_expert_labels':False,'budget_matched':False})
                    preds['author_native_region_equal_budget']=Prediction(e['geometry'],mask=budgeted_physics_mask(labels>0,ep.area_budget(values.shape)),budget=ep.area_budget(values.shape),metadata=meta)
                expert.append(evaluate_case(e,arrays[n],preds,protocol=ep));continue
            r=t['input']
            if t['kind']=='wse2':
                require(sha256_file(contained_path(ROOT,r['label']))==r['label_sha256'],'External label changed')
                truth=label_points(contained_path(ROOT,r['label']));points=student_points(values,ep.point_budget(values.shape),ep.point_min_distance_px)
                matched=len(matched_expert_indices(points,truth,ep.point_tolerance_px))
                wse.append({'id':t['id'],'matched':matched,'predicted':len(points),'truth':len(truth)});continue
            if t['kind']=='dennler':_,truth=load_image(archive,r,truth=True)
            else:
                require(sha256_file(contained_path(ROOT,r['label']))==r['label_sha256'],'FeCrAl label changed')
                truth=(np.asarray(Image.open(contained_path(ROOT,r['label'])).convert('L'))>0).astype(np.uint8)
            require(sha256_file(contained_path(ROOT,r['physical']))==r['physical_sha256'],'Historical physics changed')
            with np.load(contained_path(ROOT,r['physical']),allow_pickle=False) as old:
                if t['kind']=='dennler':methods={'author_pretrained_full':labels>0,'historical_physics_combined':old['any_defect_region'],
                    'historical_physics_score':old['coherence_loss_region'],'author_pretrained_class2':labels==2,'historical_physics_class2':old['coherence_loss_region']}
                else:methods={'author_pretrained_full':labels>0,'historical_physics_score':old['v04_score_only'],'historical_fourier_class2':old['fourier_bragg_class2']}
            for border in ([0,12] if t['kind']=='dennler' else [0]):
                sl=(slice(border,-border),slice(border,-border)) if border else (slice(None),slice(None))
                for name,pred in methods.items():
                    target=truth==2 if name.endswith('class2') and t['kind']=='dennler' else truth>0
                    counts=_binary_counts(pred[sl],target[sl]);dense.append({'id':t['id'],'kind':t['kind'],'author_split':r.get('author_split'),
                        'border':border,'method':name,'metrics':_counts_to_metrics(counts)})
                    if r.get('category')=='non_defective':normal[(name,border)].append(float(np.asarray(pred[sl],bool).mean()))
    sums=defaultdict(lambda:defaultdict(int));cover=defaultdict(list)
    for r in dense:
        for subset in ['all']+([r['author_split']] if r['author_split'] else []):
            for k,v in r['metrics'].items():
                if k.endswith('_pixels'):sums[(r['kind'],subset,r['border'],r['method'])][k]+=v
    dense_summary=[{'kind':k[0],'subset':k[1],'border':k[2],'method':k[3],'metrics':_counts_to_metrics(dict(v))} for k,v in sums.items()]
    for r in expert:
        for method,m in r['methods'].items():cover[(r['group'],method)].append(m['positive_coverage'])
    expert_summary=[{'group':k[0],'method':k[1],'cases':len(v),'mean':float(np.mean(v))} for k,v in cover.items()]
    totals={k:sum(r[k] for r in wse) for k in ['matched','predicted','truth']}
    totals.update({'precision':totals['matched']/max(1,totals['predicted']),'recall':totals['matched']/max(1,totals['truth']),
        'f1':2*totals['matched']/max(1,totals['predicted']+totals['truth']),'images':len(wse)})
    result={'protocol_sha256':sha256_file(output/'protocol.json'),'dense_summary':dense_summary,'expert_summary':expert_summary,
        'wse2_summary':totals,'dense_cases':dense,'expert_cases':expert,'wse2_cases':wse,
        'normal_control_false_positive_area':[{'method':k[0],'border':k[1],'images':len(v),'mean':float(np.mean(v))} for k,v in normal.items()]}
    write_new_json(output/'results.json',result)
    write_new_json(output/'seal.json',{'files':{str(f.relative_to(output)):sha256_file(f) for f in output.rglob('*') if f.is_file()}})
    print(json.dumps({k:v for k,v in result.items() if k.endswith('summary')},indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','launch','worker','score']);a.add_argument('--output',default=DEFAULT)
    a.add_argument('--model');a.add_argument('--parity');a.add_argument('--gpus',default='2,3,4,5,6,7');a.add_argument('--shard',type=int);a.add_argument('--shards',type=int)
    args=a.parse_args();out=contained_path(ROOT,args.output)
    if args.command=='prepare':prepare(out,contained_path(ROOT,args.model),contained_path(ROOT,args.parity))
    elif args.command=='launch':launch(out,args.gpus)
    elif args.command=='worker':work(out,args.shard,args.shards)
    else:score(out)
