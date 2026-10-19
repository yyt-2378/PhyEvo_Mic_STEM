#!/usr/bin/env python3
"""Freeze the requested historical best collection, without pretending it is blind.

Original inputs/checkpoints are read-only. All new files are exclusive-create.
The selected five configurations were chosen after seeing outer scores; this is
an explicit scientific finding, not something a successful hash check removes.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from zipfile import ZipFile,ZIP_DEFLATED
import numpy as np
import torch
from scripts.analyze_rsi_refinement import load_analysis,metric_path,verified_metrics,macro_summary
from scripts.run_rsi_nested import idle_gpus
from scripts.verify_rsi_cycle import Audit,require
from tfree_detection.expert_supervision import contained_path,load_learning_dataset,verify_frozen_round,sha256_file,write_new_json
from tfree_detection.expert_student import restore_checkpoint
from tfree_detection.rsi_refinement import predict_with_embedding
from tfree_detection.rsi_feedback import feedback_predictions
from tfree_detection.rsi_evaluation import EvaluationProtocol,evaluate_case

ROOT=Path(__file__).resolve().parents[1]
DEFAULT='outputs/best_release_20261014_v1'
CHOICES={
    'guo-superscan-single-acquisition':('photometric',800,28.82,22.06),
    'guo_2021_mowte2_acquisition':('weak_roi',800,14.29,14.29),
    'guo_2021_zro2_twin_acquisition':('positive_crops',400,26.88,31.91),
    'sigma3-orientation-i':('source_sqrt',1200,3.77,4.00),
    'sigma3-orientation-ii':('source_sqrt',800,47.62,45.40),
}
COMBINATION_BRANCHES={group:('student_reranked_physics_plus_rescue' if group in
    ('guo-superscan-single-acquisition','guo_2021_mowte2_acquisition') else 'hybrid_fixed_budget') for group in CHOICES}
def read(p):return json.loads(p.read_text())

def audit_split(entries,fold,meta,bank):
    train=set(fold['train_case_numbers']);test=set(fold['validation_case_numbers'])
    require(train and test and not train&test,'Case train/test overlap')
    require(meta['train_case_numbers']==fold['train_case_numbers'],'Checkpoint training inventory changed')
    train_groups={entries[n]['group'] for n in train};test_groups={entries[n]['group'] for n in test}
    require(not train_groups&test_groups,'Acquisition train/test overlap')
    require(not {entries[n]['source_file_sha256'] for n in train}&{entries[n]['source_file_sha256'] for n in test},'Source-file train/test overlap')
    require(not train_groups&set(fold['excluded_groups']),'Excluded acquisition entered training')
    require(set(map(int,meta['sampler_audit']['by_case']))<=train,'Observed sampler used held-out case')
    require(set(meta['sampler_audit']['by_source'])<=train_groups,'Observed sampler used held-out source')
    require(sum(meta['sampler_audit']['by_case'].values())==meta['sampler_audit']['patches'],'Sampler count drift')
    require(bank['selection_uses_validation'] is False and {r['case_number'] for r in bank['cases']}<=train,'Reference bank used test')
    return {'case_disjoint':True,'source_file_disjoint':True,'acquisition_disjoint':True,
            'observed_sampler_train_only':True,'reference_bank_train_only':True,
            'training_cases':len(train),'test_cases':len(test)}

def pixel_duplicates(arrays,entries):
    known={};matches=[]
    for n in sorted(entries):
        a=arrays[n]['image'];key=(a.shape,hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
        for old in known.get(key,[]):
            if entries[old]['group']!=entries[n]['group']:matches.append([old,n])
        for k in range(4):
            b=np.rot90(a,k)
            for flip in (False,True):
                c=b[:,::-1] if flip else b
                key=(c.shape,hashlib.sha256(np.ascontiguousarray(c).tobytes()).hexdigest())
                known.setdefault(key,set()).add(n)
    return matches

def prepare(output):
    output=contained_path(ROOT,output)
    if output.exists():raise FileExistsError(output)
    analysis,ap,training,tp,dataset=load_analysis(ROOT/'outputs/rsi_cycle_002/analysis_v1')
    manifest,arrays=load_learning_dataset(ROOT,dataset)
    records={r['case_number']:r for r in verify_frozen_round(ROOT,contained_path(ROOT,manifest['frozen_bundle']))}
    entries={e['case_number']:e for e in manifest['entries']};audit=Audit(ROOT)
    require(manifest['admission']['development_retirement_authorized'],'Expert learning admission missing')
    audit.hashes(tp['source_hashes']);audit.hashes(ap['source_hashes'])
    for p in (training/'protocol.json',analysis/'analysis_protocol.json',dataset/'manifest.json',dataset/'seal.json'):
        audit.check(p)
    frozen=contained_path(ROOT,manifest['frozen_bundle']);audit.hashes(audit.read(frozen/'seal.json')['files'],frozen)
    for e in entries.values():
        audit.check(dataset/e['plane'],e['plane_sha256']);audit.check(e['source_path'],e['source_file_sha256'])
    duplicate=pixel_duplicates(arrays,entries)
    require(not duplicate,'Cross-source exact/D4 duplicate detected; audit before freezing')
    summary=audit.read(analysis/'summary.json');jobs=[];rows=[];recomputed=[]
    ep=EvaluationProtocol(**ap['evaluation'])
    for group,(policy,step,expected_student,expected_hybrid) in CHOICES.items():
        fold=next(f for f in tp['folds'] if f['stage']=='outer' and f['excluded_groups']==[group])
        for method,expected in [('student_expert',expected_student),(COMBINATION_BRANCHES[group],expected_hybrid)]:
            r=next(r for r in summary['outer_comparisons'] if r['group']==group and r['policy']==policy and r['step']==step and r['method']==method)
            require(round(r['mean']*100,2)==expected,'Requested historical number does not match source')
        for seed in tp['seeds']:
            checkpoint=training/policy/f'seed-{seed}'/fold['fold_id']/f'step-{step:06d}'
            meta=audit.read(checkpoint/'checkpoint.json');audit.check(checkpoint/'state.npz',meta['state_sha256'])
            require(meta['policy']==policy and meta['seed']==seed and meta['fold']==fold and meta['step']==step,'Checkpoint identity mismatch')
            require(meta['protocol_sha256']==sha256_file(training/'protocol.json'),'Checkpoint protocol mismatch')
            bank=audit.read(checkpoint/'training_embedding_bank.json')
            split=audit_split(entries,fold,meta,bank)
            metric=metric_path(analysis,policy,seed,fold);old=verified_metrics(metric,verify_inputs=True)
            audit.check(metric);audit.check(metric.with_suffix('.seal.json'))
            for number in fold['validation_case_numbers']:
                path=checkpoint/'held_out_scores'/f'case-{number:03d}.npz';receipt=audit.read(path.with_suffix('.json'))
                audit.check(path,receipt['score_sha256'])
                require(receipt['checkpoint_sha256']==sha256_file(checkpoint/'checkpoint.json'),'Score bound to different checkpoint')
                with np.load(path,allow_pickle=False) as a:score=a['score']
                plane=arrays[number];entry=entries[number]
                predictions=feedback_predictions(plane['image'],plane['teacher'],records[number]['prediction']['candidates'],score,entry['geometry'],protocol=ep)
                result=evaluate_case(entry,plane,predictions,protocol=ep)
                stored=next(r for r in old['rows'] if r['case_number']==number and r['step']==step)
                require(result['methods']==stored['methods'],'Fresh evaluation differs from stored metrics')
                result.update({'policy':policy,'seed':seed,'step':step})
                recomputed.append(result)
            job={'index':len(jobs),'group':group,'policy':policy,'step':step,'seed':seed,'fold':fold,
                 'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha256_file(checkpoint/'checkpoint.json'),
                 'state_sha256':meta['state_sha256'],'split_audit':split}
            jobs.append(job);print('verified',group,policy,step,seed,flush=True)
        row={'group':group,'policy':policy,'step':step,'selection':'outer-test-posthoc-development-best',
             'reported_combination_branch':COMBINATION_BRANCHES[group]}
        for method in ['physics_budgeted','student_expert','hybrid_fixed_budget','student_reranked_physics_plus_rescue']:
            expected=next(r for r in summary['outer_comparisons'] if r['group']==group and r['policy']==policy and r['step']==step and r['method']==method)
            actual=next(r for r in macro_summary(recomputed) if r['group']==group and r['method']==method)
            require(actual==expected,'Recomputed macro summary changed');row[method]=actual
        rows.append(row)
    output.mkdir(parents=True)
    sources=set(tp['source_hashes'])|set(ap['source_hashes'])|{'scripts/freeze_best_release.py','scripts/verify_rsi_cycle.py',
        'scripts/report_rsi_refinement.py','scripts/verify_rsi_refinement.py','tests/test_best_release_audit.py',
        'pyproject.toml','environments/expert-learning-cuda124.txt'}
    sources.update(str(p.relative_to(ROOT)) for p in (ROOT/'src/tfree_detection').rglob('*.py'))
    configs={records[n]['prediction']['config_path']:records[n]['prediction']['config_sha256'] for n in records}
    for p,digest in configs.items():audit.check(p,digest);sources.add(p)
    with ZipFile(output/'source_snapshot.zip','x',compression=ZIP_DEFLATED) as archive:
        for name in sorted(sources):archive.write(contained_path(ROOT,name),name)
    source_hashes={s:sha256_file(contained_path(ROOT,s)) for s in sources}
    for job in jobs:
        original=contained_path(ROOT,job['checkpoint']);dest=output/'checkpoints'/f'job-{job["index"]:02d}';dest.mkdir(parents=True)
        for name in ('checkpoint.json','state.npz','training_embedding_bank.json','training_embedding_bank.npz'):
            with (original/name).open('rb') as src,(dest/name).open('xb') as target:shutil.copyfileobj(src,target)
        require(sha256_file(dest/'state.npz')==job['state_sha256'],'Checkpoint copy differs')
        job['frozen_checkpoint']=str(dest.relative_to(ROOT))
    payload={'format':'phyevo-best-development-collection-v1','jobs':jobs,'table':rows,'dataset':str(dataset.relative_to(ROOT)),
             'input_sha256':audit.files,'source_sha256':source_hashes,'evaluation':ap['evaluation'],
             'metric':'case_macro_known_positive_coverage_then_three_seed_mean','seeds':tp['seeds'],
             'findings':{'direct_training_test_leakage_detected':False,'cross_source_exact_D4_duplicates':duplicate,
                         'test_score_informed_policy_checkpoint_selection':True,'expert_labels_model_assisted_not_blind_gold':True,
                         'single_deployable_model':False,'new_independent_test':False,'production_promotion':False,
                         'unobserved_cropped_source_overlap_excluded':False},
             'audit_scope':'Declared sources, observed sampler counts, references, checkpoint/score hashes, source code and metric replay; not proof of absence of all unknown leakage',
             'prior_strict_inner_selected_result':str((analysis/'selected_outer_cases.json').relative_to(ROOT))}
    write_new_json(output/'manifest.json',payload)
    write_new_json(output/'environment.json',{name:importlib.metadata.version(name) for name in
        ['torch','torchvision','numpy','scipy','scikit-image','h5py','Pillow','networkx','matplotlib']})
    write_new_json(output/'recomputed_cases.json',{'rows':recomputed})
    write_new_json(output/'seal.json',{'files':{str(p.relative_to(output)):sha256_file(p) for p in output.rglob('*') if p.is_file()}})
    print('FROZEN',output,flush=True)

def checked(output):
    p=read(output/'manifest.json')
    for name,digest in read(output/'seal.json')['files'].items():require(sha256_file(contained_path(output,name))==digest,'Frozen artifact changed')
    for name,digest in p['source_sha256'].items():require(sha256_file(contained_path(ROOT,name))==digest,'Frozen producer changed')
    return p

def worker(output,index):
    p=checked(output);job=p['jobs'][index];target=output/'gpu_replay'/f'job-{index:02d}';target.mkdir(parents=True,exist_ok=False)
    manifest,arrays=load_learning_dataset(ROOT,contained_path(ROOT,p['dataset']));entries={e['case_number']:e for e in manifest['entries']}
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model,_,_,_=restore_checkpoint(contained_path(ROOT,job['frozen_checkpoint']),device='cuda:0')
    checks=[]
    for n in job['fold']['validation_case_numbers']:
        score,embedding=predict_with_embedding(model,arrays[n]['image'],entries[n]['geometry_index'],device='cuda:0')
        source=contained_path(ROOT,job['checkpoint'])/'held_out_scores'/f'case-{n:03d}.npz'
        with np.load(source,allow_pickle=False) as z:
            exact=bool(np.array_equal(score,z['score']) and np.array_equal(embedding,z['embedding']))
            delta=float(np.max(np.abs(score-z['score'])))
        # Keep the actual rerun even if comparison fails; never hide it.
        path=target/f'case-{n:03d}.npz'
        with path.open('xb') as s:np.savez_compressed(s,score=score,embedding=embedding)
        checks.append({'case_number':n,'exact':exact,'max_abs_error':delta,'prediction_sha256':sha256_file(path)})
    write_new_json(target/'verification.json',{'job':job,'checks':checks,'all_exact':all(c['exact'] for c in checks),
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'manifest_sha256':sha256_file(output/'manifest.json')})
    require(all(c['exact'] for c in checks),'GPU replay changed; inspect retained predictions')

def replay(output,gpus):
    p=checked(output);ids=[int(g) for g in gpus.split(',')];require(len(ids)==len(set(ids)),'Repeated GPU')
    idle_gpus(ids);work=output/'workers';work.mkdir(exist_ok=False);jobs=list(range(len(p['jobs'])));active={};failed=[]
    while jobs or active:
        for gpu in ids:
            if gpu in active or not jobs:continue
            idle_gpus([gpu]);index=jobs.pop(0);log=(work/f'job-{index:02d}.log').open('x')
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONDONTWRITEBYTECODE':'1',
                 'TMPDIR':str(ROOT/'.codex-tmp'),'MPLCONFIGDIR':str(ROOT/'.codex-tmp/matplotlib'),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--output',str(output),'--index',str(index)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            active[gpu]=(proc,log,index)
            with (work/'processes.jsonl').open('a') as f:f.write(json.dumps({'pid':proc.pid,'gpu':gpu,'job':index})+'\n')
            print('replay start',index,gpu,proc.pid,flush=True)
        for gpu,(proc,log,index) in list(active.items()):
            if proc.poll() is not None:
                log.close();print('replay done',index,proc.returncode,flush=True)
                if proc.returncode:failed.append(index)
                del active[gpu]
        if active:time.sleep(2)
    write_new_json(output/'replay_execution.json',{'jobs':len(p['jobs']),'failed':failed})
    require(not failed,'GPU replay failure retained')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','replay','worker']);a.add_argument('--output',default=DEFAULT)
    a.add_argument('--gpus',default='2,3,4,5,6,7');a.add_argument('--index',type=int);args=a.parse_args();out=contained_path(ROOT,args.output)
    if args.command=='prepare':prepare(out)
    elif args.command=='replay':replay(out,args.gpus)
    else:worker(out,args.index)
