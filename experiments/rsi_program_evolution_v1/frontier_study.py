"""Prospective bounded validation using the existing frozen RSI engine.

No production changes. Replay the entire prior program library per split, let
independent proposals choose whether to learn, and audit only after all searches
are sealed. Historical data exposure is explicitly not called a new blind test.
"""
import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil
import sqlite3
from types import SimpleNamespace

from .archive import Archive
from .contracts import Protocol, Proposal, digest, file_hash, inside, write_new
from .controller import Controller, baseline_proposal
from .evaluation import compare
from .executor import Executor
from .meta_study import emit_request, receive_response, latest_success, filtered_context
from .real_study import ROOT, load_study, prepare as prepare_fields, read
from .sandbox import Sandbox

OLD = ROOT / 'outputs/rsi_program_evolution_20261016_v1'
TRAJECTORIES = {'t01': 24, 't02': 6, 't03': 12}
MODEL = 'gpt-5.6-sol'


def mean_coverage(receipt):
    values = list(receipt['summary'].values())
    if len({v['metric'] for v in values}) != 1:
        raise ValueError('Do not average different task metrics')
    return sum(v['mean'] for v in values) / len(values)


def frontier_gain(value, start, frozen_best, pre_round_best):
    return {'relative_to_raw_start': value-start,
            'relative_to_frozen_library_best': value-frozen_best,
            'relative_to_all_pre_round_candidates': value-pre_round_best}


def ancestry_order(proposals):
    remaining=dict(proposals)
    done=set()
    result=[]
    while remaining:
        ready=sorted(k for k,v in remaining.items() if set(v['parents']) <= done)
        if not ready:
            raise ValueError('Missing parent or cyclic candidate ancestry')
        for key in ready:
            result.append((key,remaining.pop(key)))
            done.add(key)
    return result


def initialize(directory):
    directory = inside(ROOT, directory)
    directory.mkdir(parents=True, exist_ok=False)
    proposals, source_hashes = {}, {}
    for suffix in ('meta_live/archive', 'learning_live/archive'):
        folder = OLD / suffix
        path = folder / 'archive.sqlite3'
        source_hashes[str(path.relative_to(ROOT))] = file_hash(path)
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
            rows = db.execute('SELECT id,proposal,code_hash,policy_hash FROM candidates ORDER BY rowid').fetchall()
            for identity, text, code_hash, policy_hash in rows:
                value = json.loads(text)
                candidate = Proposal.from_dict(value)
                if (candidate.identity != identity
                    or file_hash(folder/'programs'/identity/'pipeline.py') != code_hash
                    or file_hash(folder/'programs'/identity/'policy.py') != policy_hash):
                    raise ValueError('Historical program content changed')
                proposals.setdefault(identity, value)
            if suffix.startswith('meta'):
                selected_policy = json.loads(db.execute(
                    "SELECT body FROM events WHERE kind='policy_selection' ORDER BY seq DESC LIMIT 1"
                ).fetchone()[0])
    write_new(directory/'library.json', {'proposals': proposals, 'source_sha256': source_hashes,
               'inherited_policy': selected_policy,
               'scope': 'All executable candidates in the prior meta and learning archives; '
                        'not all historical project checkpoints or paper tables.'})
    write_new(directory/'pilot_protocol.json', {
        'trajectories': TRAJECTORIES, 'rounds_per_trajectory': 3, 'maximum_proposal_sessions': 9,
        'requested_model': MODEL, 'inherited_policy': selected_policy['incumbent'],
        'maximum_training_steps_per_proposal': 64, 'device': 'cpu',
        'minimum_gain': .002, 'maximum_group_regression': .01,
        'area_fraction': .10, 'seeds': [23],
        'independent_proposer_contexts': True, 'independent_datasets_claimed': False,
        'all_searches_sealed_before_any_transfer_audit': True,
        'new_unseen_data_claimed': False, 'historically_development_exposed': True,
        'initial_incumbent': 'best freshly replayed frozen-library candidate on this validation split',
        'directions': ['analysis', 'routing', 'learning'], 'training_is_not_mandated': True,
        'reference_library_fitting_is_not_autonomous_training': True,
        'report_all_rounds_and_failures': True, 'statistical_significance_claimed': False,
        'matched_generic_or_ablation_control_in_this_pilot': False,
        'maximum_new_directory_bytes': 8*1024**3,
        'study_runner_sha256': file_hash(__file__), 'library_sha256': file_hash(directory/'library.json')})
    print(json.dumps({'initialized': str(directory), 'historical_candidates': len(proposals)}), flush=True)


def open_episode(directory, key):
    directory = inside(ROOT, directory)
    if key not in TRAJECTORIES:
        raise ValueError('Unknown trajectory')
    config = read(directory/'pilot_protocol.json')
    if file_hash(__file__) != config['study_runner_sha256'] or file_hash(directory/'library.json') != config['library_sha256']:
        raise ValueError('Frozen pilot runner/library changed')
    episode = directory/key
    field_source = read(episode/'fields_source.json')
    field_directory = inside(ROOT,field_source['path'])
    if file_hash(field_directory/'dataset.json') != field_source['dataset_sha256']:
        raise ValueError('Reused field provenance changed')
    _, _, samples = load_study(field_directory)
    p = read(episode/'protocol.json')
    protocol = Protocol(**{k: tuple(v) if k.endswith('_groups') or k=='seeds' else v for k,v in p.items()})
    archive = Archive(episode/'archive', protocol, create=not (episode/'archive').exists())
    executor = Executor(archive, Sandbox(ROOT, ROOT/'.venvs/sam2', timeout=180), samples)
    return episode, archive, executor


def prepare(directory, key, fields=None):
    directory = inside(ROOT, directory)
    episode = directory/key
    if (episode/'prepared.json').exists():
        raise FileExistsError('Already prepared; use the saved state')
    if fields is None:
        fields=episode/'fields'
        if not fields.exists():
            prepare_fields(fields, job_index=TRAJECTORIES[key], rounds=3)
    fields=inside(ROOT,fields)
    _, old, _ = load_study(fields)
    expected=read(ROOT/'outputs/autonomous_rsi_20261015_v1/protocol.json')['jobs'][TRAJECTORIES[key]]
    if old.validation_groups != (expected['validation_group'],) or old.audit_groups != (expected['outer_group'],):
        raise ValueError('Reused fields have the wrong source split')
    episode.mkdir(parents=True,exist_ok=True)
    if not (episode/'fields_source.json').exists():
        write_new(episode/'fields_source.json',{'path':str(fields.relative_to(ROOT)),
                    'dataset_sha256':file_hash(fields/'dataset.json')})
    protocol = replace(old, maximum_training_steps=64)
    if not (episode/'protocol.json').exists():
        write_new(episode/'protocol.json', asdict(protocol))
    episode, archive, executor = open_episode(directory, key)
    try:
        library = read(directory/'library.json')
        baseline = baseline_proposal().identity
        successful, failed = [], []
        for identity, value in ancestry_order(library['proposals']):
            proposal = Proposal.from_dict(value)
            assert archive.add(proposal) == identity
            if proposal.target == 'research_policy':
                continue
            trial, result = executor.execute(identity)
            if 'rows' not in result:
                failed.append({'candidate': identity, 'trial': trial})
                continue
            successful.append({'candidate': identity, 'trial': trial, 'mean': mean_coverage(result),
                               'training_steps': proposal.training_steps,
                               'role': 'frozen_library_reference_not_an_autonomous_decision'})
            if identity == baseline and not archive.events('baseline_registered'):
                archive.select(identity, True, 'raw_framework_reference', trial)
                archive.event('baseline_registered', {'candidate': identity, 'trial': trial})
            print(json.dumps({'trajectory': key, 'reference_replayed': len(successful),
                              'candidate': identity[:12], 'coverage': mean_coverage(result)}), flush=True)
        if failed:
            write_new(episode/'reference_failures.json', failed)
            raise RuntimeError('Reference library incomplete; do not claim its frontier or start search')
        best = max(successful, key=lambda r: (r['mean'], r['candidate']))
        archive.select(best['candidate'], True, 'frozen_library_warm_start_not_new_improvement', best['trial'])
        write_new(episode/'prepared.json', {'raw_baseline': baseline,
            'raw_start_mean': mean_coverage(latest_success(archive, baseline)['receipt']),
            'frozen_best': best, 'references': successful,
            'inherited_policy': library['inherited_policy']['incumbent'],
            'source_audit': read(fields/'dataset.json')['source_audit'],
            'actual_reference_execution': True})
    finally:
        archive.close()


def queue(directory, key):
    episode, archive, executor = open_episode(directory, key)
    try:
        if archive.events('search_sealed'):
            raise ValueError('Search sealed')
        index = len(archive.events('round_finished'))+1
        if index > 3:
            raise ValueError('Three-call budget exhausted')
        folder = episode/'requests'/f'r{index:02d}'
        if (folder/'request.json').exists():
            print(json.dumps({'request': str(folder), 'resumed': True}), flush=True)
            return
        if shutil.disk_usage(ROOT).free < 8*1024**3:
            raise RuntimeError('Insufficient free disk for bounded trial')
        total = sum(p.stat().st_size for p in inside(ROOT,directory).rglob('*') if p.is_file())
        if total > read(inside(ROOT,directory)/'pilot_protocol.json')['maximum_new_directory_bytes']:
            raise RuntimeError('Pilot storage budget reached')
        state = read(episode/'prepared.json')
        controller = Controller(archive, executor, SimpleNamespace(model=MODEL, calls=0), arm='full')
        context = controller.context(index)
        context['budget']['remaining_proposal_sessions'] = 4-index
        plan, policy_receipt = executor.policy(state['inherited_policy'], context)
        if plan is None:
            raise RuntimeError('Inherited research policy did not execute')
        context = filtered_context(context, plan)
        context['parent_programs'] = [asdict(archive.candidate(p)['proposal']) for p in plan['parents']]
        if not plan['parents']:
            context['reference_program'] = asdict(archive.candidate(archive.incumbent())['proposal'])
        visible = [t for t in archive.trials() if t['split']=='validation' and t['outcome']=='ok']
        pre_round_best = max(mean_coverage(t['receipt']) for t in visible)
        context['frontier_reference'] = {'raw_start': state['raw_start_mean'],
            'frozen_library_best': state['frozen_best']['mean'], 'pre_round_best': pre_round_best,
            'minimum_mean_gain_for_acceptance': archive.protocol.minimum_gain}
        train = [s for s in executor.samples if archive.protocol.split(s.group)=='train']
        context['available_training'] = {'samples': len(train),
            'acquisition_groups': len({s.group for s in train}),
            'known_positive_pixels': sum(int(s.human.sum()) for s in train),
            'unknown_pixels_are_not_negative': True, 'requested_device': 'cpu',
            'fit_timeout_seconds': 180, 'maximum_training_steps': 64}
        context['autonomous_direction_decision'] = (
            'Choose analysis, routing or learning from these diagnostics; no branch is required. '
            'Explain the chosen direction and why the alternatives are less suitable in observation/intervention. '
            'Report training_steps=0 if you choose not to train. If learning, implement real fit and reuse '
            'numeric state in predict; expose numeric optimization-step count and loss/parameter-change '
            'trace in the saved state for auditing. A measured gain over the raw start is not a new discovery '
            'unless it also exceeds the pre-round candidate frontier. Use only provided observations. '
            'Within the chosen parentage you may replace its mechanism, not merely adjust scalar weights. '
            'No labels/source identifiers are available at inference. Only propose a bounded executable experiment.')
        request_hash = emit_request(folder, context, archive.protocol)
        write_new(folder/'reservation.json', {'round': index, 'incumbent': archive.incumbent(),
            'pre_round_best': pre_round_best, 'policy': state['inherited_policy'],
            'policy_execution': policy_receipt, 'request_sha256': request_hash})
        archive.event('pilot_request_reserved', {'round': index, 'request_sha256': request_hash})
        print(json.dumps({'request': str(folder), 'round': index}), flush=True)
    finally:
        archive.close()


def execute(directory, key):
    episode, archive, executor = open_episode(directory, key)
    try:
        index = len(archive.events('round_finished'))+1
        if archive.events('search_sealed') or index > 3:
            raise ValueError('Search budget closed')
        folder = episode/'requests'/f'r{index:02d}'
        reservation, request = read(folder/'reservation.json'), read(folder/'request.json')
        context = request['context']
        try:
            proposal, model_receipt = receive_response(folder, archive.protocol)
            if (proposal.action != context['required_action'] or list(proposal.parents) != context['required_parents']
                    or proposal.target == 'research_policy'):
                raise ValueError('Proposal violates the frozen request')
            if proposal.training_steps > 0 and proposal.target != 'learning':
                raise ValueError('A proposal that trains must declare the learning direction')
        except (ValueError, TypeError, KeyError, SyntaxError) as error:
            output={'round':index,'status':'invalid_proposal','error':str(error),
                    'request_sha256':reservation['request_sha256'],'incumbent':archive.incumbent()}
            write_new(folder/'result.json',output)
            archive.event('round_finished',output)
            print(json.dumps(output),flush=True)
            return
        existing_hashes = {archive.candidate(r['id'])['code_hash'] for r in archive.pool()}
        identity = archive.add(proposal)
        trial, result = executor.execute(identity)
        output = {'round': index, 'candidate': identity, 'trial': trial,
            'model_receipt': model_receipt, 'target': proposal.target, 'training_steps': proposal.training_steps,
            'autonomous_direction_reason': proposal.intervention, 'hypothesis': proposal.hypothesis,
            'pipeline_already_in_pool': archive.candidate(identity)['code_hash'] in existing_hashes,
            'policy': reservation['policy'], 'request_sha256': reservation['request_sha256']}
        if 'rows' not in result:
            output.update(status='execution_failed', error=result.get('error'), incumbent=archive.incumbent())
            archive.remember(identity, trial, result.get('error'), 'Execution failure; retain prior model')
        else:
            state = read(episode/'prepared.json')
            decision = compare(result['rows'], latest_success(archive,reservation['incumbent'])['receipt']['rows'],
                               latest_success(archive,state['raw_baseline'])['receipt']['rows'],archive.protocol)
            value = mean_coverage(result)
            gains = frontier_gain(value, state['raw_start_mean'], state['frozen_best']['mean'], reservation['pre_round_best'])
            # A candidate beating a weak start is not promoted over a stronger prior pool member.
            decision['accepted'] = bool(decision['accepted'] and gains['relative_to_all_pre_round_candidates'] >= archive.protocol.minimum_gain)
            archive.select(identity, decision['accepted'], decision, trial)
            archive.remember(identity, trial, {'hypothesis':proposal.hypothesis,'gains':gains},
                             {'split':'validation','direction':proposal.target})
            output.update(status='accepted' if decision['accepted'] else 'retained_for_research',
                          coverage=value, gains=gains, decision=decision, checkpoints=result['checkpoints'],
                          incumbent=archive.incumbent())
        write_new(folder/'result.json', output)
        archive.event('round_finished', output)
        print(json.dumps(output, ensure_ascii=False), flush=True)
    finally:
        archive.close()


def seal_and_audit(directory):
    directory = inside(ROOT,directory)
    # Seal EVERY independent search before inspecting any new transfer result.
    for key in TRAJECTORIES:
        episode,a,e = open_episode(directory,key)
        try:
            if len(a.events('round_finished')) != 3:
                raise ValueError('All nine reserved rounds must finish before transfer audit')
            if not a.events('search_sealed'):
                a.event('search_sealed',{'incumbent':a.incumbent(),'rounds':3})
        finally:a.close()
    reports = []
    for key in TRAJECTORIES:
        episode,a,e = open_episode(directory,key)
        try:
            state = read(episode/'prepared.json')
            seal = a.events('search_sealed')[-1]['hash']
            targets = {'raw_start':state['raw_baseline'], 'frozen_library_champion':state['frozen_best']['candidate'],
                       'selected_after_search':a.incumbent()}
            audits = {}
            for role,identity in targets.items():
                p = a.candidate(identity)['proposal']
                # Never fit on audit, including when the selected candidate is a trained program.
                replay = replace(p,action='tune',parents=(identity,),training_steps=0,
                                 observation='Frozen transfer replay using parent numeric state; no new fitting')
                replay_id = a.add(replay)
                trial,result = e.execute(replay_id,split='audit',audit_seal=seal)
                audits[role] = {'original_candidate':identity,'inference_candidate':replay_id,'trial':trial,
                                'summary':result.get('summary'), 'error':result.get('error'),
                                'training_steps':0,'actual_execution':True}
            reports.append({'trajectory':key,'prepared':state,
                            'rounds':[v['body'] for v in a.events('round_finished')], 'audit':audits})
        finally:a.close()
    write_new(directory/'results.json',{'trajectories':reports,'audit_used_for_selection':False,
              'new_unseen_dataset_claimed':False,'all_nine_rounds_included':True,
              'statistical_significance_claimed':False,'metric':'known_positive_area_coverage_at_fixed_area'})
    print(json.dumps({'completed_trajectories':3,'report':str(directory/'results.json')}),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','prepare','queue','execute','audit'])
    p.add_argument('directory');p.add_argument('--trajectory',choices=list(TRAJECTORIES))
    p.add_argument('--fields',help='Reuse an already verified prepared-field bundle without copying data')
    args=p.parse_args()
    directory=inside(ROOT,args.directory)
    # One writer per trajectory. Locks do not kill processes or mutate services.
    lock=directory.with_name(directory.name+'.'+(args.trajectory or 'study')+'.lock')
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        if args.action=='initialize':initialize(directory)
        elif args.action=='audit':seal_and_audit(directory)
        else:
            if args.trajectory is None:p.error('--trajectory is required')
            if args.action=='prepare':prepare(directory,args.trajectory,args.fields)
            else:{'queue':queue,'execute':execute}[args.action](directory,args.trajectory)
