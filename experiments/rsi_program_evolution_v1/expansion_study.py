"""Paired prospective research on a historically exposed microscopy cohort.

Frozen complete system versus route-only research versus full capability
research. All original sources are now development/replay evidence, including
the previously failed transfer source. Rotation stress is held back until all
searches close; it is NOT a new independent material or acquisition.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil

import numpy as np

from .archive import Archive
from .contracts import canonical, digest, engine_fingerprint, file_hash, inside, protected_snapshot, verify_snapshot, write_new
from .executor import sample_fingerprint
from .expansion_state import CompactSystemExecutor, expansion_admission, library_gaps, selected_novel_gain, transformed_audit
from .real_study import ROOT, load_study
from .runtime_compat import ConfiguredSandbox
from .system_state import SYSTEM_SCHEMA, SystemProposal, read
from .system_study import INSTRUCTIONS as SYSTEM_INSTRUCTIONS, anonymize, checkpoint, diagnostics, latest, load as old_load


PRIOR = ROOT / 'outputs/rsi_whole_system_20261017_v2'
FIELDS = ROOT / 'outputs/rsi_frontier_validation_20261017_v1'
ARMS = ('route', 'full')
SEEDS = (23, 41, 67)
MODEL = 'gpt-5.6-sol'
SCHEMA = {'type': 'object', 'additionalProperties': False, 'required': ['research_decision', 'system'],
    'properties': {'system': SYSTEM_SCHEMA,
        'research_decision': {'type': 'object', 'additionalProperties': False,
            'required': ['diagnosis', 'operation_target', 'evidence', 'alternatives', 'new_capacity_test'],
            'properties': {'diagnosis': {'enum': ['routing_gap', 'capability_gap', 'mixed', 'inconclusive']},
                'operation_target': {'enum': ['adoption', 'analysis', 'student']},
                **{k: {'type': 'string'} for k in ('evidence', 'alternatives', 'new_capacity_test')}}}}}
INSTRUCTIONS = SYSTEM_INSTRUCTIONS + '''
This experiment returns an envelope {research_decision, system}; system follows SystemProposal.
Research is on historically exposed data. D0 is the training-retention source; D1 and D2 are
weight-fit-excluded development sources, NOT fresh blind datasets. All original sources are evaluated.
Distinguish routing_gap (an existing capability works but is not selected) from capability_gap
(the existing library itself misses positives). Diagnose before choosing adoption/analysis/student.
The old-library per-case best is an offline ceiling for choosing one fixed-budget existing branch,
not an executable strategy or an answer available at inference. A new branch that beats that ceiling
provides stronger evidence of additional capacity than improvement over a weak initial branch.
The frozen baseline is the strongest complete previously audited system, including its actual
adoption program and learned states, not b0 alone. Training-source-only improvement is insufficient:
mean gain over D1/D2 must also reach 0.002 and no source may regress beyond 0.002.
Do not force training; choose it only when evidence supports the intervention. Existing annotations
are available to fit, and no new expert annotation is needed. Router plus branch training steps must
sum to at most 4096. Explain alternatives and how execution could distinguish your mechanism.
The research arm may restrict the operation set; obey context.allowed_operations.
No manual source lookup, exact shape/coordinate dispatch, hidden labels, or inferred test scores.
'''


def implementations():
    names = ('expansion_state.py', 'expansion_study.py', 'system_state.py', 'system_study.py', 'runtime_compat.py')
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def choose_reference(results):
    candidates = []
    for record in results['trajectories']:
        for role in ('old_system', 'selected_system'):
            audit = record['audit'][role]
            if 'rows' not in audit['receipt']:
                continue
            if role == 'selected_system':
                development = record['selected_development']['summary']
            else:
                development = record['prepared']['baseline_summary']
            summary = {**development, **audit['receipt']['summary']}
            candidates.append({'trajectory': record['trajectory'], 'role': role,
                'candidate': audit['candidate'], 'summary': summary,
                'source_equal_mean': float(np.mean([v['mean'] for v in summary.values()]))})
    best = max(candidates, key=lambda r: (r['source_equal_mean'], r['candidate']))
    return best, candidates


def program_identity(code):
    return digest(ast.dump(ast.parse(code), include_attributes=False))


def known_capabilities(reference):
    """Both arms receive all successful prior whole-system capability programs.

    Duplicate learned programs retain only the chosen fold's weights. A state
    fitted on a current development source must never be silently imported.
    """
    episode = PRIOR / reference['trajectory']
    spec = read(episode / 'archive/programs' / reference['candidate'] / 'system_spec.json')
    known = set()
    for binding in spec['library'].values():
        cap = read(episode / binding['path'])
        known.add(program_identity((episode / cap['program']['path']).read_text()))
    extras = []
    for key in ('t01', 't02', 't03'):
        for path in sorted((PRIOR / key / 'capabilities').glob('*/capability.json')):
            cap = read(path); program = inside(PRIOR / key, cap['program']['path'])
            if file_hash(program) != cap['program']['sha256']:
                raise ValueError('Historical capability code changed')
            fingerprint = program_identity(program.read_text())
            if fingerprint in known:
                continue
            if cap['checkpoint']:
                raise ValueError('Previously unknown learned branch requires an explicit training-exclusion review')
            known.add(fingerprint)
            extras.append({'trajectory': key, 'binding': {'path': str(path.relative_to(PRIOR / key)),
                'sha256': file_hash(path)}, 'program_identity': fingerprint})
    return extras, sorted(known)


def initialize(directory):
    directory = inside(ROOT, directory)
    if directory.exists():
        raise FileExistsError('Use a new study destination')
    prior = read(PRIOR / 'results.json')
    reference, compared = choose_reference(prior)
    extras, known_programs = known_capabilities(reference)
    fields = FIELDS / reference['trajectory'] / 'fields'
    _, old_protocol, samples = load_study(fields)
    training = tuple(sorted({s.group for s in samples if old_protocol.split(s.group) == 'train'}))
    validation = tuple(sorted({s.group for s in samples} - set(training)))
    if len(training) != 1 or len(validation) != 2:
        raise ValueError('Expected one training and two weight-fit-excluded replay sources')
    combined = samples + transformed_audit(samples)
    protocol = replace(old_protocol, train_groups=training, validation_groups=validation,
        audit_groups=tuple('stress-' + g for g in sorted({s.group for s in samples})),
        data_version=sample_fingerprint(combined), maximum_rounds=6, maximum_model_calls=6,
        maximum_output_tokens=16000, maximum_training_steps=4096, timeout_seconds=900,
        minimum_gain=.002, maximum_group_regression=.002)
    protected = protected_snapshot(ROOT)
    for path in PRIOR.rglob('*'):
        if path.is_file() and (path.suffix in {'.json', '.sqlite3'} or path.name in {'pipeline.py', 'router.py'}):
            protected[str(path.relative_to(ROOT))] = file_hash(path)
    directory.mkdir(exist_ok=False)
    write_new(directory / 'protected_before.json', protected)
    write_new(directory / 'protocol.json', {'evaluation': asdict(protocol), 'seeds': list(SEEDS), 'arms': list(ARMS),
        'rounds_per_arm': 6, 'minimum_free_disk_bytes': 8 * 1024**3,
        'reference': reference, 'reference_candidates': compared,
        'additional_known_capabilities': extras, 'known_program_identities': known_programs,
        'prior_results_sha256': file_hash(PRIOR / 'results.json'),
        'fields_manifest_sha256': file_hash(fields / 'dataset.json'),
        'implementation_sha256': implementations(), 'engine_fingerprint': engine_fingerprint(),
        'model': MODEL, 'study_scope': 'prospective research decisions on historically exposed original sources',
        'original_sources_all_development': True, 'new_blind_dataset': False,
        'final_check': 'held-back 90-degree equivariance stress; not independent source generalization',
        'stress_not_used_for_search_or_selection': True,
        'novelty': 'selected new-branch gain over the frozen old-library per-case selection ceiling',
        'primary_control': 'equally budgeted routing-only research from the same complete baseline',
        'no_production_change': True, 'no_total_token_budget': True})
    print(json.dumps({'initialized': str(directory), 'reference': reference,
                      'samples': len(samples), 'comparison': 'route-only versus full capability research'}), flush=True)


def load(directory, arm, seed):
    directory = inside(ROOT, directory); config = read(directory / 'protocol.json')
    if (config['implementation_sha256'] != implementations() or config['engine_fingerprint'] != engine_fingerprint()
            or config['prior_results_sha256'] != file_hash(PRIOR / 'results.json')):
        raise ValueError('Frozen experiment implementation or input changed')
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Unknown paired arm or seed')
    fields = FIELDS / config['reference']['trajectory'] / 'fields'
    if file_hash(fields / 'dataset.json') != config['fields_manifest_sha256']:
        raise ValueError('Source manifest changed')
    _, old, samples = load_study(fields)
    combined = samples + transformed_audit(samples)
    values = config['evaluation']
    protocol = replace(old, **{k: tuple(v) if k.endswith('_groups') or k == 'seeds' else v for k, v in values.items()})
    protocol = replace(protocol, seeds=(seed,))
    if sample_fingerprint(combined) != protocol.data_version:
        raise ValueError('Frozen sample values changed')
    episode = directory / f'{arm}-s{seed}'
    archive = Archive(episode / 'archive', protocol, create=not (episode / 'archive').exists())
    runner = CompactSystemExecutor(ROOT, archive,
        ConfiguredSandbox(ROOT, ROOT / '.venvs/sam2', timeout=900, gpu=3 if arm == 'full' else 4),
        combined, directory / 'shared_inputs')
    return directory, config, episode, archive, runner


def prepare(directory, arm, seed):
    directory, config, episode, archive, runner = load(directory, arm, seed)
    try:
        if (episode / 'prepared.json').exists():
            raise FileExistsError('Already prepared')
        sources = {}
        try:
            for key in {config['reference']['trajectory']} | {r['trajectory'] for r in config['additional_known_capabilities']}:
                _, _, _, previous_archive, previous_runner = old_load(PRIOR, key)
                sources[key] = (previous_archive, previous_runner)
            previous, old_runner = sources[config['reference']['trajectory']]
            extras = [(sources[r['trajectory']][1], r['binding']) for r in config['additional_known_capabilities']]
            identity = runner.import_system(old_runner, config['reference']['candidate'], extras)
            historical = previous.memories()
        finally:
            for previous_archive, _ in sources.values():
                previous_archive.close()
        trial, receipt = runner.execute(identity)
        if 'rows' not in receipt:
            raise RuntimeError(receipt.get('error'))
        expected = config['reference']['summary']
        if any(receipt['summary'][g]['mean'] != v['mean'] for g, v in expected.items()):
            raise ValueError('Imported complete baseline does not reproduce frozen source scores')
        archive.select(identity, True, 'exact_frozen_best_complete_system', trial)
        archive.event('baseline_registered', {'candidate': identity, 'trial': trial})
        archive.remember(identity, trial, {'historical_research': historical},
                         {'scope': 'shared pre-experiment historical experience, not new improvement'})
        state = runner.frozen_state(identity)
        write_new(episode / 'prepared.json', {'baseline': identity, 'trial': trial,
            'checkpoint': checkpoint(archive, runner), 'summary': receipt['summary'],
            'baseline_library': state['library'], 'reproduced_all_frozen_source_scores': True})
        print(json.dumps({'arm': arm, 'seed': seed, 'baseline': receipt['summary']}, ensure_ascii=False), flush=True)
    finally:
        archive.close()


def parse_proposal(value, arm, incumbent, library):
    if set(value) != {'research_decision', 'system'}:
        raise ValueError('Explicit diagnostic decision and system proposal required')
    research = value['research_decision']
    if set(research) != set(SCHEMA['properties']['research_decision']['properties']):
        raise ValueError('Incomplete research diagnosis')
    if research['diagnosis'] not in {'routing_gap', 'capability_gap', 'mixed', 'inconclusive'}:
        raise ValueError('Unknown diagnosis')
    if research['operation_target'] not in {'adoption', 'analysis', 'student'}:
        raise ValueError('Unknown intervention target')
    if not all(isinstance(research[k], str) and research[k].strip() for k in ('evidence', 'alternatives', 'new_capacity_test')):
        raise ValueError('Evidence, alternatives and an expansion test are required')
    proposal = SystemProposal(**value['system']).validate(incumbent, library)
    if arm == 'route' and (proposal.operation != 'route_only' or research['operation_target'] != 'adoption'):
        raise ValueError('Routing-only control cannot create a capability')
    total_steps = proposal.router_training_steps + (proposal.new_branch['training_steps'] if proposal.new_branch else 0)
    if total_steps > 4096:
        raise ValueError('Matched total optimization budget exceeded')
    if research['operation_target'] == 'student' and not (proposal.new_branch and proposal.new_branch['training_steps']):
        raise ValueError('Student intervention must actually train a new analysis branch')
    if research['operation_target'] == 'analysis' and not proposal.new_branch:
        raise ValueError('An analysis intervention must implement a new branch')
    return proposal, research


def queue(directory, arm, seed):
    directory, config, episode, archive, runner = load(directory, arm, seed)
    try:
        rounds = archive.events('expansion_round_finished')
        if len(rounds) >= config['rounds_per_arm'] or archive.events('search_sealed'):
            print(json.dumps({'finished': True, 'rounds': len(rounds)}), flush=True); return
        folder = episode / 'requests' / f'r{len(rounds)+1:02d}'
        if folder.exists():
            print(json.dumps({'request': str(folder / 'request.json'), 'resumed': True}), flush=True); return
        if shutil.disk_usage(ROOT).free < config['minimum_free_disk_bytes']:
            raise RuntimeError('Disk reserve reached; no existing evidence is removed')
        identity = archive.incumbent(); manifest = runner.frozen_state(identity)
        evidence, old_aliases = diagnostics(runner, manifest)
        groups = list(runner.protocol.train_groups) + list(runner.protocol.validation_groups)
        aliases = {group: 'D' + str(i) for i, group in enumerate(groups)}
        evidence = anonymize(evidence, {old_aliases[g]: aliases[g] for g in groups})
        prepared = read(episode / 'prepared.json')
        current = latest(archive, identity)['receipt']
        branch_rows = {name: runner.predict_capability(binding, 'development')['rows']
                       for name, binding in manifest['library'].items()}
        frozen_rows = {name: runner.predict_capability(binding, 'development')['rows']
                      for name, binding in prepared['baseline_library'].items()}
        programs = {}
        for name, binding in manifest['library'].items():
            cap = read(runner.checked_ref(binding))
            programs[name] = {'pipeline_code': runner.checked_ref(cap['program']).read_text(),
                'checkpoint_present': bool(cap['checkpoint']),
                'checkpoint_sha256': cap['checkpoint']['sha256'] if cap['checkpoint'] else None}
        context = {'current_system': identity, 'checkpoint': checkpoint(archive, runner),
            'arm': arm, 'allowed_operations': ['route_only'] if arm == 'route' else ['route_only', 'add_branch', 'replace_branch'],
            'current_router_code': (Path(archive.candidate(identity)['directory']) / 'router.py').read_text(),
            'active_branches': manifest['active_branches'], 'capability_library': programs,
            'current_summary': anonymize(current['summary'], aliases),
            'frozen_complete_baseline': anonymize(prepared['summary'], aliases),
            'gap_diagnosis': anonymize(library_gaps(current['rows'], branch_rows), aliases),
            'frozen_library_gap': anonymize(library_gaps(current['rows'], frozen_rows), aliases),
            'observable_and_error_evidence': evidence,
            'experience': anonymize(archive.memories(), aliases),
            'all_attempt_feedback': anonymize([r['body'] for r in rounds], aliases),
            'round': len(rounds)+1, 'remaining_rounds': config['rounds_per_arm']-len(rounds),
            'new_annotation_required': False, 'training_source': 'D0', 'gpu_available': True,
            'historical_cohort_not_new_blind_test': True}
        # Remove raw paths, identities and source names from ancillary feedback.
        payload = {'model': MODEL, 'instructions': INSTRUCTIONS, 'response_schema': SCHEMA, 'context': context}
        serialized = canonical(payload)
        if any(g in serialized for g in runner.protocol.train_groups + runner.protocol.validation_groups + runner.protocol.audit_groups):
            raise ValueError('Named source in model input')
        folder.mkdir(parents=True, exist_ok=False)
        write_new(folder / 'request.json', payload)
        write_new(folder / 'reservation.json', {'checkpoint': context['checkpoint'],
            'request_sha256': file_hash(folder / 'request.json')})
        archive.event('expansion_request_reserved', {'round': len(rounds)+1,
            'request_sha256': file_hash(folder / 'request.json'), 'checkpoint': context['checkpoint']})
        print(json.dumps({'request': str(folder / 'request.json'), 'arm': arm, 'seed': seed}), flush=True)
    finally:
        archive.close()


def publish(folder, origin):
    folder = inside(ROOT, folder)
    raw = (folder / 'raw_response.json').read_text(); proposal = json.loads(raw)
    request = read(folder / 'request.json'); context = request['context']
    parse_proposal(proposal, context['arm'], context['current_system'], context['capability_library'])
    if not origin:
        raise ValueError('Independent proposer origin required')
    write_new(folder / 'response.json', {'raw_text': raw, 'proposal': proposal, 'origin': origin,
        'request_sha256': file_hash(folder / 'request.json'), 'requested_model': MODEL,
        'returned_model': None, 'usage': None})
    write_new(folder / 'response.ready.json', {'response_sha256': file_hash(folder / 'response.json')})
    print(json.dumps({'published': str(folder), 'origin': origin}), flush=True)


def execute(directory, arm, seed):
    _, config, episode, archive, runner = load(directory, arm, seed)
    try:
        rounds = archive.events('expansion_round_finished'); index = len(rounds)+1
        if index > config['rounds_per_arm'] or archive.events('search_sealed'):
            raise ValueError('Closed search')
        folder = episode / 'requests' / f'r{index:02d}'
        reservation = read(folder / 'reservation.json'); response = read(folder / 'response.json')
        if (reservation['checkpoint'] != checkpoint(archive, runner)
                or response['request_sha256'] != file_hash(folder / 'request.json')
                or file_hash(folder / 'response.json') != read(folder / 'response.ready.json')['response_sha256']
                or json.loads(response['raw_text']) != response['proposal']):
            raise ValueError('Changed proposal or inherited whole-system state')
        parent = archive.incumbent(); manifest = runner.frozen_state(parent)
        proposal, research = parse_proposal(response['proposal'], arm, parent, manifest['library'])
        result = {'round': index, 'origin': response['origin'], 'research_decision': research,
            'operation': proposal.operation, 'parent_checkpoint': reservation['checkpoint'],
            'request_sha256': response['request_sha256'], 'response_sha256': file_hash(folder / 'response.json')}
        identity = None
        try:
            identity = runner.register_system(proposal, manifest)
            trial, receipt = runner.execute(identity)
            result.update(candidate=identity, trial=trial)
            if 'rows' not in receipt:
                raise RuntimeError(receipt.get('error'))
            prepared = read(episode / 'prepared.json')
            accepted = {e['body']['candidate'] for e in archive.events('selection') if e['body']['accepted']}
            histories = [latest(archive, i)['receipt']['summary'] for i in accepted]
            retained = {g: max(s[g]['mean'] for s in histories) for g in receipt['summary']}
            decision = expansion_admission(receipt['rows'], latest(archive, parent)['receipt']['rows'],
                latest(archive, prepared['baseline'])['receipt']['rows'], archive.protocol, retained)
            replay_trial, replay = runner.execute(identity, replay=True)
            if 'rows' not in replay or replay['router_checkpoint'] != receipt['router_checkpoint']:
                raise RuntimeError('Frozen state replay failed')
            for case, ref in receipt['predictions'].items():
                with np.load(runner.checked_ref(ref), allow_pickle=False) as a:
                    with np.load(runner.checked_ref(replay['predictions'][case]), allow_pickle=False) as b:
                        if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                            raise RuntimeError('Changed predictions on frozen replay')
            if {r['case']: r['branch'] for r in replay['rows']} != {r['case']: r['branch'] for r in receipt['rows']}:
                raise RuntimeError('Changed adoption on frozen replay')
            frozen_rows = {name: runner.predict_capability(binding, 'development')['rows']
                           for name, binding in prepared['baseline_library'].items()}
            novelty = selected_novel_gain(receipt['rows'], frozen_rows, set(prepared['baseline_library']))
            archive.select(identity, decision['accepted'], decision, trial)
            archive.remember(identity, trial, {'research_decision': research, 'decision': decision,
                'summary': receipt['summary'], 'novelty': novelty}, {'scope': 'historical-development-only'})
            result.update(status='accepted' if decision['accepted'] else 'retained_for_research',
                summary=receipt['summary'], choices=receipt['choices'], decision=decision, novelty=novelty,
                proposed_analysis_code_already_known=(program_identity(proposal.new_branch['pipeline_code'])
                    in config['known_program_identities']) if proposal.new_branch else None,
                replay_trial=replay_trial, exact_replay=True)
        except (ValueError, KeyError, RuntimeError, TypeError, SyntaxError, OSError) as error:
            result.update(status='execution_failed', error=str(error), candidate=identity)
            # This event is always included in next request, even if fit failed before registration.
        result['checkpoint'] = checkpoint(archive, runner)
        write_new(folder / 'result.json', result)
        archive.event('expansion_round_finished', result)
        print(json.dumps({'arm': arm, 'seed': seed, **result}, ensure_ascii=False), flush=True)
    finally:
        archive.close()


def audit(directory):
    directory = inside(ROOT, directory)
    for arm in ARMS:
        for seed in SEEDS:
            _, config, _, archive, runner = load(directory, arm, seed)
            try:
                if len(archive.events('expansion_round_finished')) != config['rounds_per_arm']:
                    raise ValueError('All paired searches must finish before stress audit')
            finally:
                archive.close()
    for arm in ARMS:
        for seed in SEEDS:
            _, _, _, archive, runner = load(directory, arm, seed)
            try:
                if not archive.events('search_sealed'):
                    archive.event('search_sealed', checkpoint(archive, runner))
            finally:
                archive.close()
    trajectories = []
    for arm in ARMS:
        for seed in SEEDS:
            _, _, episode, archive, runner = load(directory, arm, seed)
            try:
                prepared = read(episode / 'prepared.json'); evaluated = {}; outcomes = {}
                for role, identity in {'frozen': prepared['baseline'], 'selected': archive.incumbent()}.items():
                    if identity not in evaluated:
                        trial, receipt = runner.execute(identity, split='audit')
                        evaluated[identity] = {'trial': trial, 'receipt': receipt}
                    outcomes[role] = evaluated[identity]
                trajectories.append({'arm': arm, 'seed': seed, 'prepared': prepared,
                    'rounds': [e['body'] for e in archive.events('expansion_round_finished')],
                    'development': latest(archive, archive.incumbent())['receipt'],
                    'stress_audit': outcomes, 'checkpoint': checkpoint(archive, runner), 'archive_tail': archive.verify()})
            finally:
                archive.close()
    write_new(directory / 'results.json', {'trajectories': trajectories,
        'protected': verify_snapshot(ROOT, read(directory / 'protected_before.json')),
        'new_blind_test': False, 'production_replaced': False})
    print(json.dumps({'completed': str(directory / 'results.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('initialize', 'prepare', 'queue', 'publish', 'execute', 'audit'))
    parser.add_argument('directory'); parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS); parser.add_argument('--origin')
    args = parser.parse_args(); directory = inside(ROOT, args.directory)
    if args.command == 'publish':
        publish(directory, args.origin)
    else:
        lock_path = directory.with_name(directory.name + '.' + (f'{args.arm}-{args.seed}' if args.arm else 'study') + '.lock')
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.command == 'initialize':
                initialize(directory)
            elif args.command == 'audit':
                audit(directory)
            else:
                if args.arm is None or args.seed is None:
                    parser.error('--arm and --seed are required')
                {'prepare': prepare, 'queue': queue, 'execute': execute}[args.command](directory, args.arm, args.seed)
