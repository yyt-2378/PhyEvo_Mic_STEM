"""Development-only whole-framework RSI with immutable source-held-out audit.

Each trajectory has a training-source retention replay, a held-out development
source, and a final transfer source. Historical algorithms were developed on
these data: this is NOT a new blind cohort. Current audit labels never enter a
request, fit, system selection, or research memory.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np

from .active_diagnostics import field_enrichment, selected_mask
from .archive import Archive
from .contracts import canonical, digest, engine_fingerprint, file_hash, inside, protected_snapshot, verify_snapshot, write_new
from .evaluation import summarize
from .real_study import ROOT, load_study
from .runtime_compat import ConfiguredSandbox
from .system_state import RAW_PIPELINE, SYSTEM_SCHEMA, SystemExecutor, SystemProposal, joint_admission, read


PRIOR = ROOT / 'outputs/rsi_active_direction_20261017_v1'
FIELDS = ROOT / 'outputs/rsi_frontier_validation_20261017_v1'
EPISODES = {'t01': 3, 't02': 4, 't03': 3}
MODEL = 'gpt-5.6-sol'
INSTRUCTIONS = '''Return exactly one SystemProposal JSON. Optimize the WHOLE system, not the best single branch.
Choose route_only, add_branch, or replace_branch yourself. Retain the protected b0 in active_branches;
old optional capabilities remain archived even when inactive. parent must equal current_system.
router_code defines route(sample,branches,state,context) returning {'branch': anonymous_name, 'reason': explanation}.
branches maps each active name to {'score':finite 2D array,'support':bool array}; return one name, not modified arrays.
sample contains image/base/support/valid/physics/student/physics_valid/variation/domain_labels, no labels or identifiers.
Only fit(samples,context) receives human and known; router fit samples also include a branches dict.
Do not mutate branch arrays. No material/source names, filenames, exact image-size dispatch, fixed coordinates or call-order dispatch,
network, filesystem access, repository imports, new packages, or evaluation changes. Use observable physical,
geometric, consistency or distribution evidence. A high score or self-consistency is not a guarantee of correctness.
All unlabelled pixels remain unknown, not confirmed negatives. Libraries numpy/scipy/skimage/torch are available.
You may train a new branch and/or the router, but training is not mandatory. Do exactly the declared optimizer steps.
fit returns a dict of finite numeric numpy arrays, with actual learned parameters plus audit_initial_parameters,
audit_final_parameters, audit_loss_trace and audit_steps. Respect context['device'] and context['seed'].
predict(sample,state,context) of a new branch returns score and support inside valid. State is always a dict.
For a new branch provide name (unused b-number), pipeline_code, training_steps, parent_branch (state donor or null),
replaces (old optional name only for replace_branch, otherwise null). For route_only set new_branch=null.
For replacement remove the replaced name from active_branches; archived program and weights are never destroyed.
All fits use D0 only. D0 scoring is training-source retention replay, NOT independent generalization;
D1 is source-held-out development, whose aggregate feedback may guide research. Never infer final audit scores.
The fixed candidate budget is 10% of the ENTIRE image. Admission requires equal-source mean gain >=0.002,
and no development source may lose >0.002 versus either the old system, current system, or best accepted history.
Explain why this adoption condition should distinguish applicable inputs and what result would falsify it.
Supply complete executable code, not recommendations. Do not claim measured success before execution.'''


def code_hashes():
    folder = Path(__file__).parent
    return {name: file_hash(folder / name) for name in ('system_state.py', 'system_study.py', 'runtime_compat.py')}


def initialize(directory):
    directory = inside(ROOT, directory)
    if directory.exists():
        raise FileExistsError('New whole-system study requires a new directory')
    previous = read(PRIOR / 'results.json')
    definitions = {'b0': {'pipeline_code': RAW_PIPELINE, 'training_steps': 0, 'parent_state': None}}
    origins = {}
    for key, name, steps in (('t01', 'b1', 0), ('t02', 'b2', 192), ('t03', 'b3', 0)):
        source = next(r for r in previous['trajectories'] if r['trajectory'] == key)
        identity = source['transfer']['selected']['candidate']
        path = PRIOR / key / 'archive/programs' / identity / 'pipeline.py'
        definitions[name] = {'pipeline_code': path.read_text(), 'training_steps': steps, 'parent_state': None}
        origins[name] = {'pipeline': str(path.relative_to(ROOT)), 'sha256': file_hash(path),
                         'weights': 'refit on the current training source; previous checkpoint is never imported'}
    protection = protected_snapshot(ROOT)
    for path in [PRIOR / n for n in ('study.json', 'library.json', 'results.json', 'summary.json')] + [
            PRIOR / key / 'archive/archive.sqlite3' for key in EPISODES]:
        protection[str(path.relative_to(ROOT))] = file_hash(path)
    directory.mkdir(exist_ok=False)
    write_new(directory / 'protected_before.json', protection)
    write_new(directory / 'library.json', {'definitions': definitions, 'origins': origins,
        'scope': 'Frozen previous capability code; no per-source hand-written router'})
    write_new(directory / 'protocol.json', {'episodes': EPISODES, 'model': MODEL, 'seeds': [23],
        'minimum_rounds': 3, 'maximum_rounds': 6, 'stop_after_stagnant_rounds': 3,
        'minimum_gain': .002, 'maximum_group_regression': .002, 'area_fraction': .1,
        'maximum_training_steps_per_component': 4096, 'timeout_per_stage': 900,
        'maximum_directory_bytes': 8 * 1024**3, 'minimum_free_disk_bytes': 8 * 1024**3,
        'implementation_sha256': code_hashes(), 'engine_fingerprint': engine_fingerprint(),
        'library_sha256': file_hash(directory / 'library.json'),
        'selection_unit': 'complete capability library + trained states + adoption program',
        'development': 'training-source retention replay plus held-out development source; equal-source mean',
        'audit': 'all trajectories sealed first; no audit-guided reselection or training',
        'historical_algorithms_have_development_exposure_to_all_three_sources': True,
        'new_blind_test': False, 'no_total_token_budget': True, 'no_production_change': True})
    print(json.dumps({'initialized': str(directory), 'capabilities': len(definitions)}), flush=True)


def load(directory, key):
    directory = inside(ROOT, directory); config = read(directory / 'protocol.json')
    if (config['implementation_sha256'] != code_hashes() or config['engine_fingerprint'] != engine_fingerprint()
            or config['library_sha256'] != file_hash(directory / 'library.json')):
        raise ValueError('Frozen whole-system experiment changed')
    _, old, samples = load_study(FIELDS / key / 'fields')
    protocol = replace(old, minimum_gain=config['minimum_gain'], maximum_group_regression=config['maximum_group_regression'],
        maximum_training_steps=4096, timeout_seconds=900, maximum_rounds=config['maximum_rounds'])
    episode = directory / key
    archive = Archive(episode / 'archive', protocol, create=not (episode / 'archive').exists())
    runner = SystemExecutor(ROOT, archive, ConfiguredSandbox(ROOT, ROOT / '.venvs/sam2', timeout=900, gpu=EPISODES[key]), samples)
    return directory, config, episode, archive, runner


def latest(archive, identity):
    return next(t for t in reversed(archive.trials(identity)) if t['split'] == 'validation' and t['outcome'] == 'ok')


def checkpoint(archive, runner):
    identity = archive.incumbent()
    folder, _ = runner.system(identity)
    runner.frozen_state(identity)
    memories = [m['id'] for m in archive.memories()]
    value = {'executable_system': identity, 'state_sha256': file_hash(folder / 'system_state.json'),
             'experience_ids': memories, 'experience_digest': digest(memories)}
    return dict(value, research_checkpoint=digest(value))


def prepare(directory, key):
    directory, _, episode, archive, runner = load(directory, key)
    try:
        if (episode / 'prepared.json').exists():
            raise FileExistsError('Already prepared')
        library = {}
        for name, definition in read(directory / 'library.json')['definitions'].items():
            library[name] = runner.capability(definition)
            result = runner.predict_capability(library[name], 'development')
            print(json.dumps({'trajectory': key, 'capability': name, 'development': result['summary']}), flush=True)
        identity = runner.register_initial(library)
        trial, result = runner.execute(identity)
        if 'rows' not in result:
            raise RuntimeError(result.get('error'))
        archive.select(identity, True, 'unchanged_old_system', trial)
        archive.event('baseline_registered', {'candidate': identity, 'trial': trial})
        state = checkpoint(archive, runner)
        write_new(episode / 'prepared.json', {'baseline': identity, 'baseline_trial': trial, 'checkpoint': state,
            'source_audit': read(FIELDS / key / 'fields/dataset.json')['source_audit'],
            'baseline_summary': result['summary'], 'capability_library': library})
    finally:
        archive.close()


def stop_status(archive, config):
    rows = [e['body'] for e in archive.events('system_round_finished')]
    last = max((i + 1 for i, r in enumerate(rows) if r['status'] == 'accepted'), default=0)
    stagnant = len(rows) - last
    return {'rounds': len(rows), 'stagnant': stagnant,
        'finished': len(rows) >= config['maximum_rounds'] or
        (len(rows) >= config['minimum_rounds'] and stagnant >= config['stop_after_stagnant_rounds'])}


def observable_features(sample, predictions, fraction):
    valid = sample.fields['valid']
    chosen = {name: selected_mask(p['score'], p['support'], fraction) for name, p in predictions.items()}
    pairwise = {a + '__' + b: float((chosen[a] & chosen[b]).sum() / max(1, max(chosen[a].sum(), chosen[b].sum())))
                for a in chosen for b in chosen if a < b}
    base = selected_mask(sample.fields['base'], sample.fields['support'], fraction)
    physics = selected_mask(sample.fields['physics'], sample.fields['physics_valid'], fraction)
    variation = sample.fields['variation'][valid]
    return {'physical_valid_fraction': float(sample.fields['physics_valid'].mean()),
        'base_physics_top_budget_overlap': float((base & physics).sum() / max(1, max(base.sum(), physics.sum()))),
        'variation_q10_q50_q90': np.quantile(variation, [.1, .5, .9]).tolist(),
        'branch_overlap': pairwise}


def diagnostics(runner, manifest):
    cohort = runner.cohorts('development')
    groups = sorted({s.group for s in cohort}, key=lambda g: runner.protocol.split(g) != 'train')
    alias = {g: 'D' + str(i) for i, g in enumerate(groups)}
    caches = {name: runner.predict_capability(binding, 'development') for name, binding in manifest['library'].items()}
    cases = []
    for sample in cohort:
        predictions = {}
        for name, cache in caches.items():
            with np.load(runner.checked_ref(cache['predictions'][sample.key]), allow_pickle=False) as arrays:
                predictions[name] = dict(arrays)
        cases.append({'group': alias[sample.group], 'role': 'training_retention_replay' if runner.protocol.split(sample.group) == 'train' else 'held_out_development',
            'observable': observable_features(sample, predictions, runner.protocol.area_fraction),
            'measured_branch_coverage': {name: next(r['coverage'] for r in cache['rows'] if r['case'] == sample.key) for name, cache in caches.items()},
            'positive_field_percentiles': {field: field_enrichment(sample.fields[field], sample.human, sample.fields['valid'])
                for field in ('variation', 'physics', 'student')}})
    # Cases have no identifiers, dimensions, coordinates, raw images or masks.
    return {'anonymous_development_observations': cases,
            'branch_mean_by_development_source': {name: {alias[g]: v['mean'] for g, v in cache['summary'].items()} for name, cache in caches.items()}}, alias


def anonymize(value, aliases):
    if isinstance(value, dict):
        return {aliases.get(k, k): anonymize(v, aliases) for k, v in value.items() if k not in {'case', 'fit_groups'}}
    if isinstance(value, list):
        return [anonymize(v, aliases) for v in value]
    return aliases.get(value, value) if isinstance(value, str) else value


def queue(directory, key):
    directory, config, episode, archive, runner = load(directory, key)
    try:
        status = stop_status(archive, config)
        if archive.events('search_sealed'):
            raise ValueError('Search already sealed')
        if status['finished']:
            print(json.dumps({'trajectory': key, **status}), flush=True); return
        index = status['rounds'] + 1; folder = episode / 'requests' / f'r{index:02d}'
        if folder.exists():
            print(json.dumps({'request': str(folder), 'resumed': True}), flush=True); return
        if shutil.disk_usage(ROOT).free < config['minimum_free_disk_bytes']:
            raise RuntimeError('Disk reserve reached')
        if sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()) > config['maximum_directory_bytes']:
            raise RuntimeError('New experiment storage bound reached')
        identity = archive.incumbent(); manifest = runner.frozen_state(identity)
        evidence, aliases = diagnostics(runner, manifest)
        programs = {}
        for name, binding in manifest['library'].items():
            cap = read(runner.checked_ref(binding))
            programs[name] = {'pipeline_code': runner.checked_ref(cap['program']).read_text(),
                'checkpoint_present': bool(cap['checkpoint']), 'checkpoint_sha256': cap['checkpoint']['sha256'] if cap['checkpoint'] else None,
                'training_trace': cap['training']['trace'] if cap['training'] else None}
        context = {'current_system': identity, 'checkpoint': checkpoint(archive, runner),
            'active_branches': manifest['active_branches'], 'capability_library': programs,
            'current_router_code': (Path(archive.candidate(identity)['directory']) / 'router.py').read_text(),
            'current_system_summary': anonymize(latest(archive, identity)['receipt']['summary'], aliases),
            'experience': anonymize(archive.memories(), aliases), 'diagnostics': evidence,
            'round': index, 'remaining_batch_rounds': config['maximum_rounds'] - index + 1,
            'new_annotation_required': False, 'gpu_available': True}
        payload = {'model': MODEL, 'instructions': INSTRUCTIONS, 'response_schema': SYSTEM_SCHEMA, 'context': context}
        serialized = canonical(payload)
        if any(g in serialized for g in runner.protocol.train_groups + runner.protocol.validation_groups + runner.protocol.audit_groups):
            raise ValueError('Named source leaked into model request')
        folder.mkdir(parents=True, exist_ok=False); write_new(folder / 'request.json', payload)
        slot = (index - 1) * 3 + list(EPISODES).index(key) + 1
        packet = directory / 'model_packets' / f'u{slot:03d}'; packet.mkdir(parents=True, exist_ok=False)
        with (packet / 'request.json').open('xb') as stream:
            stream.write((folder / 'request.json').read_bytes())
        write_new(packet / 'transport.json', {'source': str(folder.relative_to(ROOT)), 'request_sha256': file_hash(folder / 'request.json')})
        write_new(folder / 'reservation.json', {'round': index, 'checkpoint': context['checkpoint'], 'request_sha256': file_hash(folder / 'request.json')})
        archive.event('whole_system_request_reserved', {'round': index, 'checkpoint': context['checkpoint'], 'request_sha256': file_hash(folder / 'request.json')})
        print(json.dumps({'trajectory': key, 'round': index, 'packet': str(packet)}), flush=True)
    finally:
        archive.close()


def publish(packet, origin):
    packet = inside(ROOT, packet); transport = read(packet / 'transport.json')
    folder = inside(ROOT, transport['source'])
    if any(file_hash(path) != transport['request_sha256'] for path in (packet / 'request.json', folder / 'request.json')):
        raise ValueError('Reserved input changed')
    raw = (packet / 'raw_response.json').read_text(); value = json.loads(raw)
    SystemProposal(**value)
    if not origin:
        raise ValueError('Independent proposer origin required')
    write_new(folder / 'response.json', {'raw_text': raw, 'proposal': value, 'request_sha256': transport['request_sha256'],
        'requested_model': MODEL, 'origin': origin, 'returned_model': None, 'usage': None})
    write_new(folder / 'response.ready.json', {'response_sha256': file_hash(folder / 'response.json')})
    write_new(packet / 'published.json', {'response_sha256': file_hash(folder / 'response.json'), 'origin': origin})
    print(json.dumps({'published': str(folder), 'origin': origin}), flush=True)


def execute(directory, key):
    _, config, episode, archive, runner = load(directory, key)
    try:
        status = stop_status(archive, config)
        if status['finished'] or archive.events('search_sealed'):
            raise ValueError('Closed search')
        index = status['rounds'] + 1; folder = episode / 'requests' / f'r{index:02d}'
        reserved = read(folder / 'reservation.json'); response = read(folder / 'response.json')
        if (file_hash(folder / 'response.json') != read(folder / 'response.ready.json')['response_sha256']
            or response['request_sha256'] != file_hash(folder / 'request.json')
            or json.loads(response['raw_text']) != response['proposal']
            or reserved['checkpoint'] != checkpoint(archive, runner)):
            raise ValueError('Proposal, input or whole-system inheritance binding changed')
        proposal = SystemProposal(**response['proposal']); identity = None
        result = {'round': index, 'operation': proposal.operation, 'origin': response['origin'],
            'request_sha256': response['request_sha256'], 'response_sha256': file_hash(folder / 'response.json'),
            'hypothesis': proposal.hypothesis, 'intervention': proposal.intervention,
            'parent_checkpoint': reserved['checkpoint']}
        try:
            current = archive.incumbent(); parent_manifest = runner.frozen_state(current)
            identity = runner.register_system(proposal, parent_manifest)
            trial, receipt = runner.execute(identity)
            result.update(candidate=identity, trial=trial)
            if 'rows' not in receipt:
                raise RuntimeError(receipt['error'])
            accepted_ids = {e['body']['candidate'] for e in archive.events('selection') if e['body']['accepted']}
            summaries = [latest(archive, i)['receipt']['summary'] for i in accepted_ids]
            retained = {g: max(s[g]['mean'] for s in summaries) for g in receipt['summary']}
            baseline = read(episode / 'prepared.json')['baseline']
            decision = joint_admission(receipt['rows'], latest(archive, current)['receipt']['rows'],
                                       latest(archive, baseline)['receipt']['rows'], archive.protocol, retained)
            replay_trial, replay = runner.execute(identity, replay=True)
            if 'rows' not in replay or replay['router_checkpoint'] != receipt['router_checkpoint']:
                raise RuntimeError('Whole-system checkpoint replay failed')
            for case, ref in receipt['predictions'].items():
                with np.load(runner.checked_ref(ref), allow_pickle=False) as a:
                    with np.load(runner.checked_ref(replay['predictions'][case]), allow_pickle=False) as b:
                        if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                            raise RuntimeError('Frozen system changed its inference')
            if {r['case']: r['branch'] for r in receipt['rows']} != {r['case']: r['branch'] for r in replay['rows']}:
                raise RuntimeError('Adoption decisions changed on frozen replay')
            archive.select(identity, decision['accepted'], decision, trial)
            archive.remember(identity, trial, {'hypothesis': proposal.hypothesis, 'decision': decision, 'summary': receipt['summary']},
                             {'scope': 'development_and_training_retention_only', 'operation': proposal.operation})
            result.update(status='accepted' if decision['accepted'] else 'retained_for_research', decision=decision,
                          summary=receipt['summary'], choices=receipt['choices'], replay_trial=replay_trial, exact_system_replay=True)
        except (ValueError, KeyError, RuntimeError, TypeError, SyntaxError, OSError) as error:
            result.update(status='execution_failed', error=str(error))
            if identity:
                trials = archive.trials(identity)
                if trials:
                    archive.remember(identity, trials[-1]['id'], str(error), 'failed whole-system proposal; retain incumbent')
        result['checkpoint'] = checkpoint(archive, runner)
        write_new(folder / 'result.json', result)
        archive.event('system_round_finished', result)
        print(json.dumps({'trajectory': key, **result}, ensure_ascii=False), flush=True)
    finally:
        archive.close()


def audit(directory):
    # Checking every stop condition precedes any audit inference.
    for key in EPISODES:
        _, config, _, archive, _ = load(directory, key)
        try:
            if not stop_status(archive, config)['finished']:
                raise ValueError('Every search must finish before audit')
        finally:
            archive.close()
    for key in EPISODES:
        _, _, _, archive, runner = load(directory, key)
        try:
            if not archive.events('search_sealed'):
                archive.event('search_sealed', checkpoint(archive, runner))
        finally:
            archive.close()
    results = []
    for key in EPISODES:
        directory, _, episode, archive, runner = load(directory, key)
        try:
            prepared = read(episode / 'prepared.json'); scores = {}
            for role, identity in {'old_system': prepared['baseline'], 'selected_system': archive.incumbent()}.items():
                if identity in [v['candidate'] for v in scores.values()]:
                    scores[role] = next(v for v in scores.values() if v['candidate'] == identity)
                    continue
                trial, receipt = runner.execute(identity, split='audit')
                scores[role] = {'candidate': identity, 'trial': trial, 'receipt': receipt}
            results.append({'trajectory': key, 'prepared': prepared,
                'rounds': [e['body'] for e in archive.events('system_round_finished')],
                'selected_development': latest(archive, archive.incumbent())['receipt'],
                'audit': scores, 'checkpoint': checkpoint(archive, runner), 'archive_tail': archive.verify()})
        finally:
            archive.close()
    write_new(directory / 'results.json', {'trajectories': results, 'audit_used_for_selection': False,
        'whole_system_selection_executed': True, 'all_attempts_included': True,
        'protected': verify_snapshot(ROOT, read(directory / 'protected_before.json'))})
    print(json.dumps({'completed': str(directory / 'results.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('initialize', 'prepare', 'queue', 'publish', 'execute', 'audit'))
    parser.add_argument('directory'); parser.add_argument('--trajectory', choices=EPISODES); parser.add_argument('--origin')
    args = parser.parse_args(); directory = inside(ROOT, args.directory)
    if args.command == 'publish':
        publish(directory, args.origin)
    else:
        with directory.with_name(directory.name + '.' + (args.trajectory or 'study') + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.command == 'initialize':
                initialize(directory)
            elif args.command == 'audit':
                audit(directory)
            else:
                if args.trajectory is None:
                    parser.error('--trajectory is required')
                {'prepare': prepare, 'queue': queue, 'execute': execute}[args.command](directory, args.trajectory)
