#!/usr/bin/env python3
"""Independently score frozen-checkpoint GPU replays and seal their evidence."""
from pathlib import Path
import argparse
import json
from zipfile import ZipFile
import hashlib
import numpy as np
from scripts.freeze_best_release import checked,ROOT,DEFAULT,read
from scripts.verify_rsi_cycle import require,junit_counts
from tfree_detection.expert_supervision import contained_path,sha256_file,write_new_json,load_learning_dataset,verify_frozen_round
from tfree_detection.rsi_evaluation import EvaluationProtocol,evaluate_case
from tfree_detection.rsi_feedback import feedback_predictions

def verify(output,seal=False):
    p=checked(output)
    for path,digest in p['input_sha256'].items():require(sha256_file(contained_path(ROOT,path))==digest,f'Input changed: {path}')
    with ZipFile(output/'source_snapshot.zip') as z:
        require(set(z.namelist())==set(p['source_sha256']),'Source snapshot inventory differs')
        for name,digest in p['source_sha256'].items():require(hashlib.sha256(z.read(name)).hexdigest()==digest,'Archived source changed')
    manifest,arrays=load_learning_dataset(ROOT,contained_path(ROOT,p['dataset']))
    entries={e['case_number']:e for e in manifest['entries']}
    records={r['case_number']:r for r in verify_frozen_round(ROOT,contained_path(ROOT,manifest['frozen_bundle']))}
    old={(r['policy'],r['step'],r['seed'],r['case_number']):r for r in read(output/'recomputed_cases.json')['rows']}
    ep=EvaluationProtocol(**p['evaluation']);count=0
    for job in p['jobs']:
        folder=output/'gpu_replay'/f'job-{job["index"]:02d}';receipt=read(folder/'verification.json')
        require(receipt['all_exact'] and receipt['job']==job,'Replay job mismatch')
        require(receipt['manifest_sha256']==sha256_file(output/'manifest.json'),'Replay manifest mismatch')
        require([c['case_number'] for c in receipt['checks']]==job['fold']['validation_case_numbers'],'Replay case inventory')
        for c in receipt['checks']:
            n=c['case_number'];path=folder/f'case-{n:03d}.npz'
            require(sha256_file(path)==c['prediction_sha256'],'Replayed prediction changed')
            with np.load(path,allow_pickle=False) as z, np.load(contained_path(ROOT,job['checkpoint'])/'held_out_scores'/path.name,allow_pickle=False) as historical:
                for name in ['score','embedding']:require(np.array_equal(z[name],historical[name]),'Prediction is not bitwise identical')
                plane=arrays[n]
                preds=feedback_predictions(plane['image'],plane['teacher'],records[n]['prediction']['candidates'],z['score'],entries[n]['geometry'],protocol=ep)
                metric=evaluate_case(entries[n],plane,preds,protocol=ep)
            require(metric['methods']==old[job['policy'],job['step'],job['seed'],n]['methods'],'GPU-replayed metric differs')
            count+=1
    require(read(output/'replay_execution.json')=={'jobs':15,'failed':[]},'Replay execution incomplete')
    tests=junit_counts(output/'audit_tests.xml')
    result={'frozen_checkpoints':len(p['jobs']),'gpu_replayed_case_seed_predictions':count,'bitwise_identical':True,
        'metrics_recomputed_from_gpu_replay':True,'hashed_inputs':len(p['input_sha256']),'archived_source_files':len(p['source_sha256']),
        'audit_tests':tests,'findings':p['findings'],'verifier_sha256':sha256_file(Path(__file__))}
    if seal:
        for source,name in [('outputs/rsi_cycle_002/training_v1/protocol.json','training_protocol.json'),
                            ('outputs/rsi_cycle_002/analysis_v1/analysis_protocol.json','analysis_protocol.json')]:
            target=output/'protocols'/name;target.parent.mkdir(exist_ok=True)
            with target.open('xb') as f:f.write(contained_path(ROOT,source).read_bytes())
        write_new_json(output/'audit.json',result)
        write_new_json(output/'replay_seal.json',{'files':{str(f.relative_to(output)):sha256_file(f) for f in output.rglob('*') if f.is_file()}})
    else:
        require(read(output/'audit.json')==result,'Audit drift')
        for name,digest in read(output/'replay_seal.json')['files'].items():require(sha256_file(contained_path(output,name))==digest,'Replay evidence changed')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',default=DEFAULT);a.add_argument('--seal',action='store_true');args=a.parse_args()
    verify(contained_path(ROOT,args.output),args.seal)
