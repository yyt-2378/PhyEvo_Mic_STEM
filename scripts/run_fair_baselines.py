#!/usr/bin/env python3
"""Run equal-data random-initialization baselines without touching old runs.

Query patches and augmentations are identical across methods. The primary
endpoint is fixed before fitting; outer scores never choose a checkpoint.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from tfree_detection.expert_supervision import contained_path,load_learning_dataset,sha256_file,write_new_json
from tfree_detection.fair_baselines import METHODS,make_folds,make_model,logits_for,shared_loss,reference_bank,batch_references,predict,schedule_digest
from tfree_detection.rsi_refinement import RefinedSampler,POLICIES

ROOT=Path(__file__).resolve().parents[1]
FILES=['scripts/run_fair_baselines.py','src/tfree_detection/fair_baselines.py',
       'src/tfree_detection/rsi_refinement.py','src/tfree_detection/expert_student.py',
       'src/tfree_detection/student.py','src/tfree_detection/expert_supervision.py',
       'vendor/fair_baselines_20261014/emseek/emseek/models/unetpp_ref.py']

def protocol(dataset,manifest):
    return {'format':'phyevo-equal-data-baselines-v1','created_date':'2026-10-14',
            'dataset':str(dataset.relative_to(ROOT)),'manifest_sha256':sha256_file(dataset/'manifest.json'),
            'source_hashes':{p:sha256_file(ROOT/p) for p in FILES},'methods':list(METHODS),
            'folds':make_folds(manifest['entries']),'seeds':[23,41,67],
            'steps':800,'checkpoints':[400,800],'primary_step':800,'batch_size':16,'patch_size':128,
            'optimizer':'Adam','learning_rate':0.001,'gradient_clip':5,'sampler_policy':'photometric',
            'threads':2,'external_pretraining':False,'author_checkpoints_allowed':False,
            'reference_policy':'3 deterministic training positive 64px crops/task; average at inference; no test click',
            'supervision':'existing corrected point/interface positive+unknown; accepted teacher labels excluded',
            'shared_loss':'2*known_positive_BCE + 2*relu(mean_probability - task_ceiling)^2',
            'occupancy_ceiling':{'point':0.03,'interface':0.15},
            'physics_addition':'teacher BCE only in phyevo_teacher_expert, same training images, no extra human labels',
            'emseek_adaptation':'unmodified official resnet18 dual-encoder reference network; supported classes=5, random initialization and shared sparse loss, training-only references; not reproduction of published scores or complete agent platform',
            'comparison_scope':'fixed-exposure controlled developmental comparison, not each architecture optimally tuned',
            'evaluation':{'point_budget_per_512':50,'point_tolerance_px':6,'point_min_distance_px':4,'area_budget_fraction':0.1},
            'budget_sensitivity':{'point':[20,50,100],'interface':[0.05,0.1,0.2]},
            'selection':'fixed primary800, checkpoint400 diagnostic; no test-best selection',
            'claims':{'independent_locked_external':False,'continual_RSI':False,'full_precision_F1':False,
                      'all_source_folds_are_material_holdout':False,'production_changed':False}}

def check_protocol(output):
    p=json.loads((output/'protocol.json').read_text())
    for path,digest in p['source_hashes'].items():
        if sha256_file(ROOT/path)!=digest:
            raise ValueError(f'Producer changed: {path}')
    dataset=contained_path(ROOT,p['dataset'])
    if sha256_file(dataset/'manifest.json')!=p['manifest_sha256']:
        raise ValueError('Learning dataset changed')
    return p,dataset

def worker(args):
    output=contained_path(ROOT,args.output); p,dataset=check_protocol(output)
    fold=next(f for f in p['folds'] if f['fold_id']==args.fold)
    if args.method not in p['methods'] or args.seed not in p['seeds']:
        raise ValueError('Unregistered method or seed')
    run=contained_path(ROOT,output/'runs'/args.fold/args.method/f'seed-{args.seed}')
    run.mkdir(parents=True,exist_ok=False)
    manifest,arrays=load_learning_dataset(ROOT,dataset)
    selected=[e for e in manifest['entries'] if e['case_number'] in fold['train_case_numbers']]
    validation=[e for e in manifest['entries'] if e['case_number'] in fold['test_case_numbers']]
    torch.set_num_threads(p['threads']); torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this registered run')
    model=make_model(args.method,ROOT).cuda()
    initial=hashlib.sha256()
    for name,value in model.state_dict().items():
        initial.update(name.encode()); initial.update(value.detach().cpu().numpy().tobytes())
    optimizer=torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=p['learning_rate'])
    rng=np.random.default_rng(args.seed)
    sampler=RefinedSampler(selected,arrays,POLICIES[p['sampler_policy']],excluded_groups=fold['excluded_groups'])
    bank,bank_audit=reference_bank(selected,arrays,excluded_groups=fold['excluded_groups'])
    write_new_json(run/'reference_bank.json',bank_audit)
    phash=sha256_file(output/'protocol.json')
    write_new_json(run/'identity.json',{'method':args.method,'seed':args.seed,'fold':fold,
                   'protocol_sha256':phash,'initial_state_sha256':initial.hexdigest(),
                   'parameter_count':sum(v.numel() for v in model.parameters()),
                   'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad),
                   'pretraining':None,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    began=time.monotonic(); train_seconds=0.; schedule=hashlib.sha256(); records=[]
    model.train()
    for step in range(1,p['steps']+1):
        t=time.monotonic()
        batch,crops=sampler.sample(rng,batch_size=p['batch_size'],patch_size=p['patch_size'],device='cuda:0',augmentation_seed=args.seed*1000000+step)
        schedule.update(schedule_digest(crops))
        ref=batch_references(bank,batch['geometry_index'],step=step,device='cuda:0') if args.method=='emseek_refunet_expert_only' else None
        optimizer.zero_grad(set_to_none=True)
        z,raw=logits_for(model,args.method,batch['image'],batch['geometry_index'],ref)
        loss,parts=shared_loss(z,batch,teacher_output=raw if args.method=='phyevo_teacher_expert' else None)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),p['gradient_clip']); optimizer.step()
        torch.cuda.synchronize(); train_seconds+=time.monotonic()-t
        if step==1 or step%100==0:
            event={'step':step,'loss':float(loss.detach()),'train_seconds':train_seconds,
                   **{k:float(v.detach()) for k,v in parts.items()}}
            with (run/'training.jsonl').open('a') as stream:
                stream.write(json.dumps(event)+'\n')
            print(f'{args.fold} {args.method} s{args.seed} {step}/800 loss={event["loss"]:.5f} time={train_seconds:.1f}',flush=True)
        if step in p['checkpoints']:
            target=run/f'step-{step:06d}';target.mkdir()
            with (target/'state.pt').open('xb') as stream:
                torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict() if step==p['steps'] else None,
                            'rng_numpy':rng.bit_generator.state,'rng_torch':torch.get_rng_state(),
                            'rng_cuda':torch.cuda.get_rng_state_all(),'step':step},stream)
            predictions=[]; t=time.monotonic()
            for e in validation:
                number=e['case_number']
                score=predict(model,args.method,arrays[number]['image'],e['geometry_index'],bank)
                if not np.isfinite(score).all():
                    raise ValueError('Nonfinite prediction')
                path=target/f'case-{number:03d}.npz'
                with path.open('xb') as stream:
                    np.savez_compressed(stream,score=score)
                predictions.append({'case_number':number,'group':e['group'],'path':str(path.relative_to(ROOT)),
                                    'sha256':sha256_file(path)})
            info={'step':step,'state_sha256':sha256_file(target/'state.pt'),'predictions':predictions,
                  'query_schedule_sha256':schedule.hexdigest(),'sampler_audit':sampler.audit.copy(),
                  'training_seconds':train_seconds,'prediction_seconds':time.monotonic()-t,
                  'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            write_new_json(target/'manifest.json',info);records.append(info)
            model.train()
    write_new_json(run/'completed.json',{'protocol_sha256':phash,'method':args.method,'seed':args.seed,'fold_id':args.fold,
                   'checkpoints':records,'elapsed_seconds':time.monotonic()-began,'pid':os.getpid()})

def launch(args):
    output=contained_path(ROOT,args.output)
    dataset=ROOT/'outputs/expert_cycle_001/learning_dataset_v1'
    manifest,_=load_learning_dataset(ROOT,dataset)
    p=protocol(dataset,manifest)
    if output.exists():
        if not args.resume or json.loads((output/'protocol.json').read_text())!=p:
            raise FileExistsError('Existing experiment must not be overwritten')
    else:
        output.mkdir(parents=True);write_new_json(output/'protocol.json',p)
    if shutil.disk_usage(ROOT).free<20*1024**3:
        raise RuntimeError('Insufficient disk for protected checkpoint retention')
    gpus=[int(g) for g in args.gpus.split(',')]
    status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    states={int(a):(int(b),int(c)) for a,b,c in (line.split(',') for line in status.splitlines())}
    if len(gpus)!=len(set(gpus)) or any(g not in states or states[g][0]>1024 or states[g][1]>5 for g in gpus):
        raise ValueError('Requested GPU occupied; do not interfere')
    jobs=[(f['fold_id'],m,s) for s in p['seeds'] for f in p['folds'] for m in p['methods']]
    active={}; failures=[]; workers=output/'workers';workers.mkdir(exist_ok=True)
    log=workers/'processes.jsonl'
    def event(row):
        with log.open('a') as stream:
            stream.write(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),**row})+'\n')
    while jobs or active:
        for gpu in gpus:
            if gpu in active or not jobs: continue
            fold,method,seed=jobs.pop(0)
            name=f'{fold}_{method}_{seed}'
            done=output/'runs'/fold/method/f'seed-{seed}'/'completed.json'
            if done.exists() and args.resume:
                if json.loads(done.read_text())['protocol_sha256']!=sha256_file(output/'protocol.json'):
                    raise ValueError('Completed run has another protocol')
                continue
            stream=(workers/f'{name}.log').open('x')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONPATH':str(ROOT/'src')+':'+str(ROOT),
                 'PYTHONDONTWRITEBYTECODE':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                 'TORCH_HOME':str(ROOT/'.codex-tmp/torch'),'XDG_CACHE_HOME':str(ROOT/'.codex-tmp/cache'),
                 'MPLCONFIGDIR':str(ROOT/'.codex-tmp/matplotlib'),'TMPDIR':str(ROOT/'.codex-tmp')}
            cmd=[sys.executable,str(ROOT/'scripts/run_fair_baselines.py'),'worker','--output',str(output),
                 '--fold',fold,'--method',method,'--seed',str(seed)]
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            active[gpu]=(proc,stream,name,done);event({'event':'start','pid':proc.pid,'gpu':gpu,'job':name})
            print(f'start gpu{gpu} pid{proc.pid} {name}',flush=True)
        for gpu,(proc,stream,name,done) in list(active.items()):
            if proc.poll() is not None:
                stream.close();event({'event':'finish','pid':proc.pid,'job':name,'exit_code':proc.returncode})
                if proc.returncode or not done.exists(): failures.append(name)
                print(f'end {name} exit={proc.returncode} remaining={len(jobs)}',flush=True)
                del active[gpu]
        if active: time.sleep(2)
    write_new_json(output/'execution.json',{'expected_runs':len(p['folds'])*len(p['methods'])*len(p['seeds']),
                   'failed_runs':failures,'all_workers_finished':True})
    if failures: raise RuntimeError(f'Failed runs retained: {failures}')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['launch','worker'])
    parser.add_argument('--output',default='outputs/fair_baselines_20261014_v1')
    parser.add_argument('--gpus',default='2,3,4,5,6,7');parser.add_argument('--resume',action='store_true')
    parser.add_argument('--fold');parser.add_argument('--method');parser.add_argument('--seed',type=int)
    args=parser.parse_args(); (launch if args.command=='launch' else worker)(args)

if __name__=='__main__': main()
