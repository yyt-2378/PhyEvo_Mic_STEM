#!/usr/bin/env python3
"""Verify all equal-data runs, then evaluate sealed predictions, never select them."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from scripts.run_fair_baselines import check_protocol
from scripts.evaluate_rsi_capabilities import spatial_controls
from tfree_detection.expert_supervision import contained_path,load_learning_dataset,verify_frozen_round,sha256_file,write_new_json
from tfree_detection.rsi_evaluation import EvaluationProtocol,Prediction,predict_methods,evaluate_case,student_points,student_mask

ROOT=Path(__file__).resolve().parents[1]
BUDGETS={'low':EvaluationProtocol(point_budget_per_512=20,area_budget_fraction=.05),
         'main':EvaluationProtocol(),
         'high':EvaluationProtocol(point_budget_per_512=100,area_budget_fraction=.20)}

def read(path): return json.loads(path.read_text())

def verify_checkpoint_copy(checkpoint,completion_copy,final_checkpoint):
    """Recognize ONLY the documented shallow-copy metadata bug, not score drift.

    The standalone manifest was written at its checkpoint; the completion list
    kept mutable nested counters. Scalar counts, schedules, weights and scores
    must still agree. The final 800-step copy must agree in full.
    """
    if checkpoint==completion_copy:return False
    if any(v!=completion_copy.get(k) for k,v in checkpoint.items() if k!='sampler_audit'):
        raise ValueError('Checkpoint identity mismatch')
    a,b=checkpoint['sampler_audit'],completion_copy['sampler_audit']
    if any(v!=b.get(k) for k,v in a.items() if k not in ('by_case','by_source')):
        raise ValueError('Checkpoint scalar audit mismatch')
    if any(b[k]!=final_checkpoint['sampler_audit'][k] for k in ('by_case','by_source')):
        raise ValueError('Unexplained checkpoint metadata drift')
    if sum(a['by_case'].values())!=a['patches'] or sum(a['by_source'].values())!=a['patches']:
        raise ValueError('Standalone checkpoint counts inconsistent')
    return True

def verify(training,p):
    execution=read(training/'execution.json')
    expected=len(p['methods'])*len(p['folds'])*len(p['seeds'])
    if execution['failed_runs'] or execution['expected_runs']!=expected:
        raise ValueError('Incomplete experiment; retain failures, do not silently exclude')
    phash=sha256_file(training/'protocol.json'); inventory={}; digests={}; audits=[]; costs=[]; warnings=[]
    for fold in p['folds']:
        for seed in p['seeds']:
            schedules={step:set() for step in p['checkpoints']}; initial={}; banks=[]
            for method in p['methods']:
                run=training/'runs'/fold['fold_id']/method/f'seed-{seed}'
                done=read(run/'completed.json'); identity=read(run/'identity.json')
                if done['protocol_sha256']!=phash or identity['fold']!=fold or identity['pretraining'] is not None:
                    raise ValueError('Run identity mismatch')
                if identity['seed']!=seed or identity['method']!=method:
                    raise ValueError('Method/seed mismatch')
                initial[method]=identity['initial_state_sha256']
                bank=read(run/'reference_bank.json');banks.append(bank)
                if any(r['case_number'] not in fold['train_case_numbers'] or r['group'] in fold['excluded_groups'] for r in bank):
                    raise ValueError('Test-informed reference')
                for step in p['checkpoints']:
                    checkpoint=run/f'step-{step:06d}'; m=read(checkpoint/'manifest.json')
                    copies=[r for r in done['checkpoints'] if r['step']==step]
                    if m['step']!=step or len(copies)!=1:
                        raise ValueError('Checkpoint identity mismatch')
                    if verify_checkpoint_copy(m,copies[0],done['checkpoints'][-1]):
                        warnings.append({'run':str(run.relative_to(ROOT)),'step':step,
                            'issue':'completion-copy nested sampler counters advanced to final step; standalone checkpoint manifest authoritative',
                            'weights_predictions_schedule_and_scalar_counts_unchanged':True})
                    schedules[step].add(m['query_schedule_sha256'])
                    if sha256_file(checkpoint/'state.pt')!=m['state_sha256']:
                        raise ValueError('Changed checkpoint')
                    if sorted(r['case_number'] for r in m['predictions'])!=sorted(fold['test_case_numbers']):
                        raise ValueError('Missing/duplicated test case')
                    for row in m['predictions']:
                        path=contained_path(ROOT,row['path']);digest=sha256_file(path)
                        if digest!=row['sha256']:
                            raise ValueError('Changed score')
                        key=(fold['fold_id'],seed,method,step,row['case_number'])
                        if key in inventory: raise ValueError('Duplicate prediction')
                        inventory[key]=path;digests[str(path.relative_to(ROOT))]=digest
                    if step==p['primary_step']:
                        costs.append({'fold':fold['fold_id'],'seed':seed,'method':method,
                                      'parameters':identity['parameter_count'],'trainable':identity['trainable_parameters'],
                                      'training_seconds':m['training_seconds'],'prediction_seconds':m['prediction_seconds'],
                                      'peak_gpu_GiB':m['peak_gpu_bytes']/1024**3})
            if any(len(values)!=1 for values in schedules.values()) or any(b!=banks[0] for b in banks):
                raise ValueError('Unequal query schedule or reference bank')
            if initial[p['methods'][0]]!=initial[p['methods'][1]]:
                raise ValueError('Paired physics ablation had unequal initialization')
            audits.append({'fold':fold['fold_id'],'seed':seed,'identical_query_schedule':True,
                           'identical_reference_bank':True,'paired_student_initialization_equal':True,
                           'schedule_sha256_by_step':{str(k):next(iter(v)) for k,v in schedules.items()}})
    return inventory,{'runs_verified':expected,'score_files_verified':len(inventory),'audits':audits,
                      'costs':costs,'score_sha256':digests,'protocol_sha256':phash,'metadata_warnings':warnings}

def build_predictions(plane,entry,record,scores,protocol,draw):
    result={}
    for method,score in scores.items():
        pair=predict_methods(plane['image'],plane['teacher'],record['prediction']['candidates'],score,score,entry['geometry'],protocol=protocol)
        result[method]=pair['student_expert']
        result['hybrid_'+method]=pair['hybrid_fixed_budget']
        result['physics_native']=pair['physics_native'];result['physics_budgeted']=pair['physics_budgeted']
    result.update(spatial_controls(plane['image'],entry['geometry'],entry['case_number'],protocol,draw))
    return result

def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        if not row['main_comparison_eligible']: raise ValueError('Noneligible label entered primary experiment')
        for method,metric in row['methods'].items():
            key=(row['fold'],row['group'],row['geometry'],row['step'],row['budget_name'],method,row['seed'])
            groups[key].append(metric)
    seeds=[]
    for key,metrics in sorted(groups.items()):
        fold,group,geometry,step,budget,method,seed=key
        count=sum(m['annotated_positive_count'] for m in metrics)
        matched=sum(m['matched_positive_count'] for m in metrics)
        seeds.append({'fold':fold,'group':group,'geometry':geometry,'step':step,'budget':budget,'method':method,'seed':seed,
                      'case_count':len(metrics),'positive_count':count,'matched':matched,
                      'micro_coverage':matched/count,'macro_coverage':float(np.mean([m['positive_coverage'] for m in metrics]))})
    combined=defaultdict(list)
    for r in seeds:
        combined[tuple(r[k] for k in ('fold','group','geometry','step','budget','method'))].append(r)
    result=[]
    for key,values in sorted(combined.items()):
        row=dict(zip(('fold','group','geometry','step','budget','method'),key))
        row.update({'case_count':values[0]['case_count'],'positive_count':values[0]['positive_count'],'seed_count':len(values)})
        for metric in ('micro_coverage','macro_coverage'):
            a=[v[metric] for v in values]
            row[metric+'_mean']=float(np.mean(a));row[metric+'_seed_std']=float(np.std(a,ddof=1)) if len(a)>1 else 0.
        row['by_seed']={str(v['seed']):v['micro_coverage'] for v in values};result.append(row)
    return seeds,result

def run(training,output):
    training,output=contained_path(ROOT,training),contained_path(ROOT,output)
    if output.exists(): raise FileExistsError('Analysis must be new')
    p,dataset=check_protocol(training)
    manifest,arrays=load_learning_dataset(ROOT,dataset)
    entries={e['case_number']:e for e in manifest['entries']}
    records={r['case_number']:r for r in verify_frozen_round(ROOT,contained_path(ROOT,manifest['frozen_bundle']))}
    output.mkdir(parents=True)
    write_new_json(output/'evaluation_protocol.json',{'training_protocol_sha256':sha256_file(training/'protocol.json'),
        'budgets':{k:asdict(v) for k,v in BUDGETS.items()},'primary_step':p['primary_step'],
        'hybrids':'same fixed 50% physics quota applied to EACH learned model; no tuning',
        'checkpoint_selection':'none','metric':'marked-positive coverage, not accuracy, precision, F1 or dense IoU',
        'aggregate':'per source, micro and macro; 3 seeds are not 3 independent acquisitions',
        'source_sha256':{s:sha256_file(ROOT/s) for s in ['scripts/analyze_fair_baselines.py','src/tfree_detection/rsi_evaluation.py','scripts/evaluate_rsi_capabilities.py']}})
    inventory,audit=verify(training,p);write_new_json(output/'integrity_audit.json',audit)
    rows=[]
    with (output/'per_case.jsonl').open('x') as stream:
        for fold in p['folds']:
            for draw,seed in enumerate(p['seeds']):
                for step in p['checkpoints']:
                    for number in fold['test_case_numbers']:
                        e,plane=entries[number],arrays[number]
                        scores={}
                        for method in p['methods']:
                            with np.load(inventory[(fold['fold_id'],seed,method,step,number)],allow_pickle=False) as z:
                                scores[method]=z['score']
                        for budget_name,protocol in BUDGETS.items():
                            predictions=build_predictions(plane,e,records[number],scores,protocol,draw)
                            row=evaluate_case(e,plane,predictions,protocol=protocol)
                            row.update({'fold':fold['fold_id'],'seed':seed,'step':step,'budget_name':budget_name})
                            rows.append(row);stream.write(json.dumps(row)+'\n')
                    print(f'evaluated {fold["fold_id"]} seed{seed} step{step}',flush=True)
    seeds,summary=summarize(rows)
    write_new_json(output/'summary.json',{'primary_step':p['primary_step'],'rows':summary,'per_seed':seeds,
        'claims':p['claims'],'case_evaluation_count':len(rows)})
    for name,values in [('summary.csv',summary),('per_seed.csv',seeds),('compute_cost.csv',audit['costs'])]:
        fields=[k for k in values[0] if k!='by_seed']
        with (output/name).open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(values)
    write_new_json(output/'seal.json',{'files':{str(f.relative_to(output)):sha256_file(f) for f in sorted(output.iterdir()) if f.is_file()}})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--training',default='outputs/fair_baselines_20261014_v1')
    parser.add_argument('--output',default='outputs/fair_baselines_20261014_v1/analysis_v1')
    a=parser.parse_args();run(a.training,a.output)
