"""Additive GPU study: evidence-rich autonomous research, unchanged task engine.

No production edits, historical-score attribution, private source identifiers,
mandatory learning branch, or audit-guided selection. All model replies are raw.
"""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

import numpy as np

from .active_diagnostics import available_actions, diagnose, probe_proposals, validate_decision
from .archive import Archive
from .contracts import Protocol, Proposal, digest, engine_fingerprint, file_hash, inside, write_new
from .controller import Controller, baseline_proposal
from .evaluation import compare
from .executor import Executor
from .frontier_study import ancestry_order, frontier_gain, mean_coverage
from .learning_study import trace_check
from .meta_study import latest_success, receive_response
from .provider import PROPOSAL_SCHEMA, proposer_protocol
from .real_study import ROOT, load_study, read
from .sandbox import Sandbox

PREVIOUS=ROOT/'outputs/rsi_frontier_validation_20261017_v2'
FIELDS=ROOT/'outputs/rsi_frontier_validation_20261017_v1'
MODEL='gpt-5.6-sol'
EPISODES={'t01':3,'t02':4,'t03':3}

INSTRUCTIONS=(
    'Return exactly one complete Proposal JSON matching response_schema. This is an executable research '
    'decision, not a recommendation list. Choose action, parents, target and training_steps yourself; '
    'neither learning nor non-learning is required. Allowed actions: generate (0 parents), tune/repair (1), '
    'evolve (2 distinct). Use only supplied evidence. Parents must be visible in pool. '
    'predict(sample,state,context) returns a dict with finite numeric 2D score and bool support, both '
    'same shape as sample["image"]; support must be inside sample["valid"]. '
    'state is ALWAYS a dict mapping string keys to finite numeric numpy arrays; it is {} if no checkpoint. '
    'fit(samples,context) MUST return that same dict format, never a bare array or torch module. '
    'Fit alone receives human boolean positive masks and known annotation weights. Unlabelled pixels '
    'are unknown, not confirmed negatives. Physical/student fields can be fallible soft priors, not labels. '
    'Predict receives image/base/support/valid/physics/student/physics_valid/variation/domain_labels, '
    'but no human/known/source/case identifiers. image is a normalized raw 2D image. '
    'Context at execution contains seed, device, training_steps; fit may also receive parent_state. '
    'The fixed area fraction is 0.10 of the ENTIRE image, not of support. Labels never enter prediction. '
    'Available numpy/scipy/skimage/torch. No network, repository imports, filesystem access, new packages, '
    'evaluation changes, fixed annotation coordinates or source hard-coding. CUDA is available. '
    'Use context["seed"] and context["device"], bounded patches/batches and <4 GiB GPU working memory. '
    'If learning is chosen, do exactly training_steps optimizer updates, save actual trained tensors plus '
    'audit_initial_parameters, audit_final_parameters, audit_loss_trace (one loss/update), audit_steps. '
    'Those audit arrays must derive from real optimization, not invented traces. Predict must restore '
    'these learned tensors when no fit occurs. If not training, set training_steps=0 and do not optimize in predict. '
    'plan(context) must return action/parents/focus; task-score acceptance does not promote this policy. '
    'Explain direction choice and alternatives, a concrete mechanism and falsifier. No numerical success claims '
    'before execution. Pipeline and policy must be complete Python code, not markdown or pseudocode.')


def code_hashes():
    return {name:file_hash(Path(__file__).with_name(name)) for name in ('active_study.py','active_diagnostics.py')}


def initialize(directory):
    directory=inside(ROOT,directory);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'study.json').exists():raise FileExistsError('Use the frozen existing study, do not overwrite')
    library=read(PREVIOUS/'library.json'); proposals=dict(library['proposals']);sources={}
    for key in EPISODES:
        archive=PREVIOUS/key/'archive';path=archive/'archive.sqlite3'
        sources[str(path.relative_to(ROOT))]=file_hash(path)
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            # Include last pilot's new decisions, not its post-seal inference wrappers.
            for (body,) in db.execute("SELECT body FROM events WHERE kind='round_finished' ORDER BY seq"):
                identity=json.loads(body).get('candidate')
                if not identity:continue
                text,code,policy=db.execute('SELECT proposal,code_hash,policy_hash FROM candidates WHERE id=?',(identity,)).fetchone()
                value=json.loads(text)
                if (Proposal.from_dict(value).identity!=identity
                    or file_hash(archive/'programs'/identity/'pipeline.py')!=code
                    or file_hash(archive/'programs'/identity/'policy.py')!=policy):raise ValueError('Changed prior candidate')
                proposals[identity]=value
    probes={name:p.identity for name,p in probe_proposals()}
    for name,p in probe_proposals():proposals[p.identity]=asdict(p)
    write_new(directory/'library.json',{'proposals':proposals,'probes':probes,'prior_archives':sources,
        'scope':'Prior meta/learning/frontier program libraries plus prespecified diagnostic probes; not all paper models',
        'prior_library_sha256':file_hash(PREVIOUS/'library.json')})
    write_new(directory/'study.json',{
        'episodes':EPISODES,'model':MODEL,'minimum_rounds_per_episode':6,'maximum_rounds_per_episode':12,
        'stop_after_stagnant_rounds':4,'maximum_training_steps_per_candidate':4096,
        'sandbox_timeout_seconds':900,'new_proposals_not_forced_to_train':True,
        'actions_free_every_round':True,'prior_modulo_policy_is_not_enforced':True,
        'seeds':[23],'minimum_gain':.002,'maximum_group_regression':.01,'area_fraction':.1,
        'cohort':'development-exposed source-disjoint transfer; no new unseen-data claim',
        'stop_rule':'after at least 6 rounds, stop on 4 consecutive non-admissions, otherwise at 12; no audit peeking',
        'total_token_budget_imposed':False,'batch_horizon_is_a_preregistered_experimental_design':True,
        'maximum_directory_bytes':12*1024**3,'minimum_free_disk_bytes':8*1024**3,
        'implementation_sha256':code_hashes(),'engine_version':engine_fingerprint(),
        'library_sha256':file_hash(directory/'library.json'),
        'no_persistent_service':True,'current_production_release_is_not_modified':True})
    print(json.dumps({'initialized':str(directory),'library_records':len(proposals),'probes':len(probes)}),flush=True)


def load(directory,key):
    directory=inside(ROOT,directory);config=read(directory/'study.json')
    if (config['implementation_sha256']!=code_hashes() or config['engine_version']!=engine_fingerprint()
        or config['library_sha256']!=file_hash(directory/'library.json')):raise ValueError('Frozen implementation changed')
    episode=directory/key
    _,old,samples=load_study(FIELDS/key/'fields')
    p=replace(old,maximum_rounds=config['maximum_rounds_per_episode'],
        maximum_model_calls=config['maximum_rounds_per_episode'],maximum_output_tokens=12000,
        maximum_training_steps=config['maximum_training_steps_per_candidate'],timeout_seconds=config['sandbox_timeout_seconds'])
    archive=Archive(episode/'archive',p,create=not (episode/'archive').exists())
    e=Executor(archive,Sandbox(ROOT,ROOT/'.venvs/sam2',timeout=p.timeout_seconds,gpu=EPISODES[key]),samples)
    return directory,config,episode,archive,e


def prepare(directory,key):
    directory,config,episode,a,e=load(directory,key)
    try:
        if (episode/'prepared.json').exists():raise FileExistsError('Already prepared')
        library=read(directory/'library.json'); baseline=baseline_proposal().identity
        success=[];failed=[];probe_set=set(library['probes'].values())
        for identity,value in ancestry_order(library['proposals']):
            p=Proposal.from_dict(value);a.add(p)
            if p.target=='research_policy':continue
            trial,result=e.execute(identity)
            if 'rows' not in result:
                failed.append({'candidate':identity,'trial':trial,'error':result.get('error')});continue
            success.append({'candidate':identity,'trial':trial,'mean':mean_coverage(result),'diagnostic_probe':identity in probe_set})
            if identity==baseline and not a.events('baseline_registered'):
                a.select(identity,True,'raw_framework_reference',trial)
                a.event('baseline_registered',{'candidate':identity,'trial':trial})
            print(json.dumps({'trajectory':key,'reference':len(success),'candidate':identity[:12],'coverage':mean_coverage(result)}),flush=True)
        best=max(success,key=lambda r:(r['mean'],r['candidate']))
        historical=max((r for r in success if not r['diagnostic_probe']),key=lambda r:(r['mean'],r['candidate']))
        a.select(best['candidate'],True,'frozen_library_plus_diagnostic_probes_warm_start_not_new_gain',best['trial'])
        write_new(episode/'prepared.json',{'raw_baseline':baseline,
            'raw_start_mean':mean_coverage(latest_success(a,baseline)['receipt']),
            'historical_best':historical,'frozen_best':best,'references':success,'failed_references':failed,
            'source_audit':read(FIELDS/key/'fields/dataset.json')['source_audit'],
            'reference_fits_not_autonomous_training':True,'baseline_gpu':EPISODES[key]})
    finally:a.close()


def stop_status(events,config):
    bodies=[v['body'] for v in events]
    last=max((i+1 for i,v in enumerate(bodies) if v['status']=='accepted'),default=0)
    stagnant=len(bodies)-last
    stop=(len(bodies)>=config['maximum_rounds_per_episode'] or
          (len(bodies)>=config['minimum_rounds_per_episode'] and stagnant>=config['stop_after_stagnant_rounds']))
    return {'finished':stop,'rounds':len(bodies),'stagnant':stagnant}


def queue(directory,key):
    directory,config,episode,a,e=load(directory,key)
    try:
        if a.events('search_sealed'):raise ValueError('Search sealed')
        status=stop_status(a.events('round_finished'),config)
        if status['finished']:
            print(json.dumps({'trajectory':key,**status}),flush=True);return
        index=status['rounds']+1;folder=episode/'requests'/f'r{index:02d}'
        if (folder/'request.json').exists():print(json.dumps({'request':str(folder),'resumed':True}));return
        if shutil.disk_usage(ROOT).free<config['minimum_free_disk_bytes']:raise RuntimeError('Disk reserve reached')
        if sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())>config['maximum_directory_bytes']:
            raise RuntimeError('Experiment disk bound reached; preserve evidence')
        state=read(episode/'prepared.json');p=a.protocol
        context=Controller(a,e,SimpleNamespace(model=MODEL,calls=0),arm='full').context(index)
        context['research_decision']=available_actions(status['stagnant'])
        context['budget'].update(maximum_training_steps=p.maximum_training_steps,
            remaining_batch_rounds=p.maximum_rounds-index+1,requested_device='cuda',timeout_seconds=p.timeout_seconds)
        good=[t for t in a.trials() if t['split']=='validation' and t['outcome']=='ok']
        pre_best=max(mean_coverage(t['receipt']) for t in good)
        top=sorted(good,key=lambda t:mean_coverage(t['receipt']),reverse=True)
        ids=[a.incumbent()]
        for t in top:
            if t['candidate'] not in ids:ids.append(t['candidate'])
            if len(ids)>=3:break
        learned=next((t['candidate'] for t in top if a.candidate(t['candidate'])['proposal'].training_steps),None)
        if learned and learned not in ids:ids.append(learned)
        latest=a.events('round_finished')
        if latest and latest[-1]['body'].get('candidate') not in ids:
            last=latest[-1]['body'].get('candidate')
            if last:ids.append(last)
        context['reference_programs']=[asdict(a.candidate(i)['proposal'])|{'candidate_id':i} for i in ids]
        predictions={}
        for row in latest_success(a,a.incumbent())['receipt']['rows']:
            with np.load(inside(a.root,row['prediction']),allow_pickle=False) as z:
                predictions[row['case']]=(z['score'],z['support'])
        context['error_structure_and_trainability']=diagnose(e.samples,p,predictions)
        train=[s for s in e.samples if p.split(s.group)=='train']
        context['available_training']={'planes':len(train),'acquisition_groups':len({s.group for s in train}),
            'expert_positive_pixels':sum(int(s.human.sum()) for s in train),
            'new_labels_required':False,'maximum_steps':p.maximum_training_steps,'device':'cuda',
            'note':'Choosing learning is permitted with positive-unknown supervision, but never mandatory. '
                   'Decide from the observed failure modes; label diversity limitations alone do not establish that learning is useless.'}
        context['frontier_reference']={'raw_start':state['raw_start_mean'],
            'historical_library_best':state['historical_best']['mean'],
            'frozen_library_including_probes_best':state['frozen_best']['mean'],
            'pre_round_best':pre_best,'minimum_new_gain':p.minimum_gain,
            'diagnostic_probes_are_existing_capabilities_not_new_discoveries':True}
        context['task']='Choose and implement the next microscopy research experiment. Diagnose the failure mechanism, '
        context['task']+='select analysis/routing/learning and generate/tune/evolve/repair yourself, then supply complete executable code. '
        context['task']+='Prior conservative scalar tweaks stagnated; no restriction requires keeping their mechanism. '
        context['task']+='A valid support region is not ground truth. Cross-modal disagreement may contain complementarity, not only noise. '
        context['task']+='Consider both expected gain and experiment informativeness. Do not claim significance or infer transfer-audit outcomes.'
        payload={'model':MODEL,'context':context,'protocol':proposer_protocol(p),
                 'response_schema':PROPOSAL_SCHEMA,'instructions':INSTRUCTIONS}
        folder.mkdir(parents=True,exist_ok=False);write_new(folder/'request.json',payload)
        write_new(folder/'reservation.json',{'round':index,'incumbent':a.incumbent(),'pre_round_best':pre_best,
            'request_sha256':file_hash(folder/'request.json'),'visible_candidate_ids':[r['id'] for r in context['pool']]})
        a.event('active_request_reserved',{'round':index,'request_sha256':file_hash(folder/'request.json')})
        print(json.dumps({'trajectory':key,'round':index,'request':str(folder)}),flush=True)
    finally:a.close()


def verify_reuse(a,e,p,identity,receipt):
    trace={}
    for seed,checkpoint in receipt['checkpoints'].items():
        path=inside(a.root,checkpoint['path'])
        if file_hash(path)!=checkpoint['sha256']:raise ValueError('Training state changed')
        with np.load(path,allow_pickle=False) as z:trace[seed]=trace_check(dict(z),p.training_steps)
    if not trace:raise ValueError('Learning candidate produced no checkpoint')
    clone=replace(p,action='tune',parents=(identity,),training_steps=0,
                  observation='Verify frozen numeric-state inheritance without a second fit')
    replay=a.add(clone);trial,result=e.execute(replay)
    if 'rows' not in result or result['checkpoints']!=receipt['checkpoints']:raise ValueError('State reuse failed')
    for first,second in zip(receipt['rows'],result['rows']):
        if first['case']!=second['case']:raise ValueError('Different replay cohort')
        with np.load(inside(a.root,first['prediction']),allow_pickle=False) as x:
            with np.load(inside(a.root,second['prediction']),allow_pickle=False) as y:
                if any(not np.array_equal(x[k],y[k]) for k in ('score','support')):raise ValueError('Numeric reuse differs')
    return replay,trial,result,trace


def execute(directory,key):
    directory,config,episode,a,e=load(directory,key)
    try:
        status=stop_status(a.events('round_finished'),config)
        if status['finished'] or a.events('search_sealed'):raise ValueError('Batch closed')
        index=status['rounds']+1;folder=episode/'requests'/f'r{index:02d}'
        reserved=read(folder/'reservation.json');proposal=None
        try:
            proposal,model_receipt=receive_response(folder,a.protocol)
            validate_decision(proposal,reserved['visible_candidate_ids'],a.protocol)
        except (ValueError,TypeError,KeyError,SyntaxError) as error:
            output={'round':index,'status':'invalid_proposal','error':str(error),'incumbent':a.incumbent()}
        else:
            identity=a.add(proposal);trial,result=e.execute(identity)
            output={'round':index,'candidate':identity,'trial':trial,'model_receipt':model_receipt,
                'target':proposal.target,'action':proposal.action,'training_steps':proposal.training_steps,
                'hypothesis':proposal.hypothesis,'decision_reason':proposal.observation,
                'intervention':proposal.intervention}
            if 'rows' not in result:
                output.update(status='execution_failed',error=result.get('error'),incumbent=a.incumbent())
                a.remember(identity,trial,result.get('error'),'Technical failure; retain incumbent')
            else:
                reference=read(episode/'prepared.json');value=mean_coverage(result)
                gains=frontier_gain(value,reference['raw_start_mean'],reference['frozen_best']['mean'],reserved['pre_round_best'])
                decision=compare(result['rows'],latest_success(a,reserved['incumbent'])['receipt']['rows'],
                    latest_success(a,reference['raw_baseline'])['receipt']['rows'],a.protocol)
                decision['accepted']=bool(decision['accepted'] and gains['relative_to_all_pre_round_candidates']>=a.protocol.minimum_gain)
                promoted,promoted_trial=identity,trial
                if proposal.training_steps:
                    try:
                        promoted,promoted_trial,reused,trace=verify_reuse(a,e,proposal,identity,result)
                        output['actual_training']=trace;output['numeric_checkpoint_reuse_exact']=True
                    except (ValueError,KeyError,OSError) as error:
                        decision.update(accepted=False,reason='learning_trace_or_reuse_failed')
                        output['learning_verification_error']=str(error)
                a.select(promoted,decision['accepted'],decision,promoted_trial)
                a.remember(identity,trial,{'hypothesis':proposal.hypothesis,'gains':gains,'decision':decision},
                           {'split':'validation','direction':proposal.target,'training_steps':proposal.training_steps})
                output.update(status='accepted' if decision['accepted'] else 'retained_for_research',
                    coverage=value,gains=gains,decision=decision,incumbent=a.incumbent())
        write_new(folder/'result.json',output);a.event('round_finished',output)
        print(json.dumps({'trajectory':key,**output},ensure_ascii=False),flush=True)
    finally:a.close()


def audit(directory):
    # Freeze all independent searches before scoring any transfer source.
    for key in EPISODES:
        _,config,_,a,e=load(directory,key)
        try:
            if not stop_status(a.events('round_finished'),config)['finished']:raise ValueError('All trajectories must finish first')
            if not a.events('search_sealed'):a.event('search_sealed',{'incumbent':a.incumbent(),'rounds':len(a.events('round_finished'))})
        finally:a.close()
    rows=[]
    for key in EPISODES:
        directory,config,episode,a,e=load(directory,key)
        try:
            state=read(episode/'prepared.json');seal=a.events('search_sealed')[-1]['hash'];scores={}
            for role,identity in {'raw_start':state['raw_baseline'],'historical_best':state['historical_best']['candidate'],
                                 'frozen_best':state['frozen_best']['candidate'],'selected':a.incumbent()}.items():
                p=a.candidate(identity)['proposal']
                clone=replace(p,action='tune',parents=(identity,),training_steps=0,
                              observation='Frozen transfer evaluation without further fitting')
                cid=a.add(clone);trial,result=e.execute(cid,split='audit',audit_seal=seal)
                scores[role]={'candidate':identity,'inference_candidate':cid,'trial':trial,
                    'summary':result.get('summary'),'error':result.get('error'),'training_steps':0}
            rows.append({'trajectory':key,'prepared':state,'rounds':[v['body'] for v in a.events('round_finished')],
                         'transfer':scores,'archive_tail':a.verify()})
        finally:a.close()
    write_new(directory/'results.json',{'trajectories':rows,'audit_used_for_selection':False,
        'all_rounds_and_failures_included':True,'metric':'known_positive_area_coverage_at_fixed_area',
        'historical_data_development_exposure':True,'statistical_significance_claimed':False})
    print(json.dumps({'completed':str(directory/'results.json')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['initialize','prepare','queue','execute','audit'])
    parser.add_argument('directory');parser.add_argument('--trajectory',choices=EPISODES);args=parser.parse_args()
    directory=inside(ROOT,args.directory)
    with directory.with_name(directory.name+'.'+(args.trajectory or 'study')+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if args.command=='initialize':initialize(directory)
        elif args.command=='audit':audit(directory)
        else:
            if args.trajectory is None:parser.error('--trajectory is required')
            {'prepare':prepare,'queue':queue,'execute':execute}[args.command](directory,args.trajectory)
