"""New paired capability research with executed pre-admission stability checks.

No code or decisions in the sealed v3 study are changed. Both arms inherit its
best complete development system and every measured prior capability. rot90 is
development evidence; flip_lr and rot180 are withheld until every search seals.
All images remain historically exposed data, not independent new acquisitions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from . import bank_study as previous
from .archive import Archive
from .contracts import (canonical, digest, engine_fingerprint, file_hash, inside,
                        protected_snapshot, verify_snapshot, write_new)
from .executor import sample_fingerprint
from .expansion_state import expansion_admission, library_gaps, selected_novel_gain
from .real_study import ROOT, load_study
from .research_state_v2 import (MASK_FEATURE_CONTRACT, ResearchBank, allowed_actions,
                               next_branch_name, parse_action, response_schema)
from .runtime_compat import ConfiguredSandbox
from .stability_diagnostics import field_information, ordinal_tie_probe, prediction_stability, transform_array
from .stability_state import StabilitySystemExecutor, stability_admission
from .system_state import read
from .system_study import anonymize, latest


SOURCE = ROOT / 'outputs/rsi_research_bank_20261017_v3'
ARMS, SEEDS, MODEL = previous.ARMS, previous.SEEDS, previous.MODEL
ROUND_EVENT = 'stable_research_round_finished'
DEVELOPMENT_TRANSFORMS = ('rot90',)
HELD_BACK_TRANSFORMS = ('flip_lr', 'rot180')
INSTRUCTIONS = previous.INSTRUCTIONS + '''
This NEW study also measures a declared rot90 development stability probe before admission.
All fields, validity masks and evaluation targets undergo the SAME exact pixel permutation;
frozen programs and numerical states are actually rerun. No fit occurs during a probe.
A system must retain each source's original coverage within 0.002 AND its best previously
accepted score on this transform, in addition to the unchanged original gain requirements.
Missing or failed stability executions cannot pass. A stable system without gains is not enough.
observable_field_information reports constant/tied fields and the behavior of a legacy ordinal
ranking operation under permutation; this is diagnostic evidence, not a repair prescription or
proof that it explains all program failures. current_development_stability compares actual scores,
selected regions and branch choices. Distinguish algorithm instability from adoption instability.
At least one admitted system with measured new capacity must actually be inherited by a later
successful research execution; a new gain appearing only at the last proposal is not sufficient.
You may use existing mechanisms, design new analysis, or train a student when evidence supports it.
Retained research assets may be combined in a later proposal even if an earlier whole system failed.
Do not optimize for a particular image orientation or add exact-coordinate exceptions. Final
unseen-in-this-search transforms are checked only after all proposals close; they are still
stress checks of known images, not a new independent dataset. Do not request their answers.
'''


def implementation_hashes():
    names = ('stable_study.py', 'stable_report.py', 'stability_state.py', 'stability_diagnostics.py',
             'bank_study.py', 'bank_report.py', 'research_state_v2.py', 'active_diagnostics.py',
             'expansion_study.py', 'expansion_state.py', 'system_state.py', 'system_study.py',
             'real_study.py', 'learning_study.py', 'runtime_compat.py')
    return {n: file_hash(Path(__file__).with_name(n)) for n in names}


def choose_baseline(results):
    candidates = []
    for record in results['trajectories']:
        summary = record['development']['summary']
        candidates.append({'arm': record['arm'], 'seed': record['seed'],
            'candidate': record['checkpoint']['system']['executable_system'], 'summary': summary,
            'equal_source_mean': float(np.mean([v['mean'] for v in summary.values()]))})
    if not candidates:
        raise ValueError('Completed source trajectories required')
    return max(candidates, key=lambda r: (r['equal_source_mean'], r['candidate'])), candidates


def collect_known(source, baseline):
    known, extras = set(), []
    _, _, _, archive, runner = previous.load(source, baseline['arm'], baseline['seed'])
    try:
        for binding in runner.frozen_state(baseline['candidate'])['library'].values():
            known.add(previous.capability_key(runner, binding))
    finally:
        archive.close()
    for arm in ARMS:
        for seed in SEEDS:
            _, _, _, archive, runner = previous.load(source, arm, seed)
            try:
                # ResearchBank.entries checks executed feedback, code and states.
                for entry in ResearchBank(runner).entries().values():
                    binding = entry['binding']; key = previous.capability_key(runner, binding)
                    if key not in known:
                        known.add(key)
                        extras.append({'arm': arm, 'seed': seed, 'binding': binding, 'state_identity': key})
            finally:
                archive.close()
    return extras, sorted(known)


def held_back_samples(samples, transforms):
    result = []
    for transform in transforms:
        for sample in samples:
            shape = sample.fields['image'].shape
            if sample.geometry != 'interface' or any(v.shape != shape for v in sample.fields.values()):
                raise ValueError('Held-back stress requires aligned scalar interface fields')
            move = lambda a: np.ascontiguousarray(transform_array(a, transform))
            prefix = f'held-{transform}-'
            result.append(replace(sample, key=prefix + sample.key, group=prefix + sample.group,
                fields={k: move(v) for k, v in sample.fields.items()}, human=move(sample.human),
                known=move(sample.known), points_xy=np.empty((0, 2))))
    return result


def initialize(directory, tests):
    directory, tests = inside(ROOT, directory), inside(ROOT, tests)
    if directory.exists():
        raise FileExistsError('A separate new study destination is required')
    summary, results = read(SOURCE / 'summary.json'), read(SOURCE / 'results.json')
    if summary['results_sha256'] != file_hash(SOURCE / 'results.json'):
        raise ValueError('Source study audit and verification must finish first')
    suites = ET.parse(tests).getroot().findall('.//testsuite')
    if not suites or not sum(int(s.get('tests', 0)) for s in suites) or any(
            int(s.get('errors', 0)) or int(s.get('failures', 0)) for s in suites):
        raise ValueError('Passing integration preflight evidence required')
    baseline, compared = choose_baseline(results)
    extras, known = collect_known(SOURCE, baseline)
    old_config = read(SOURCE / 'protocol.json')
    fields = previous.previous.FIELDS / old_config['reference']['trajectory'] / 'fields'
    _, old_protocol, samples = load_study(fields)
    combined = samples + held_back_samples(samples, HELD_BACK_TRANSFORMS)
    values = old_config['evaluation']
    protocol = replace(old_protocol, train_groups=tuple(values['train_groups']),
        validation_groups=tuple(values['validation_groups']),
        audit_groups=tuple(sorted({s.group for s in combined[len(samples):]})),
        seeds=SEEDS, data_version=sample_fingerprint(combined), engine_version=engine_fingerprint(),
        maximum_rounds=12, maximum_model_calls=12, maximum_output_tokens=16000,
        maximum_training_steps=4096, timeout_seconds=900, minimum_gain=.002, maximum_group_regression=.002)
    config = {'evaluation': asdict(protocol), 'source': str(SOURCE.relative_to(ROOT)),
        'source_hashes': {n: file_hash(SOURCE / n) for n in ('protocol.json', 'results.json', 'summary.json')},
        'fields': str(fields.relative_to(ROOT)), 'fields_manifest_sha256': file_hash(fields / 'dataset.json'),
        'baseline': baseline, 'baseline_candidates': compared, 'imports': extras, 'known_state_identities': known,
        'implementations': implementation_hashes(), 'engine_fingerprint': engine_fingerprint(),
        'development_transforms': list(DEVELOPMENT_TRANSFORMS), 'held_back_transforms': list(HELD_BACK_TRANSFORMS),
        'readonly_input_pools': [str((SOURCE / 'shared_inputs').relative_to(ROOT))],
        'preflight_tests': {'path': str(tests.relative_to(ROOT)), 'sha256': file_hash(tests)},
        'rounds_per_arm': 12, 'seeds': list(SEEDS), 'arms': list(ARMS), 'model': MODEL,
        'minimum_free_disk_bytes': 8 * 1024**3, 'new_blind_dataset': False,
        'no_total_token_budget': True, 'production_replaced': False,
        'study_scope': 'prospective paired capability creation and stable integration on historical development data'}
    required = read(SOURCE / 'completion_requirements.json')
    required.update(scope=config['study_scope'], baseline='strongest complete v3 development system plus all measured research assets')
    required['matched_research']['proposals_per_arm'] = 12
    required['required_evidence']['all_declared_final_transforms_retained'] = True
    required['required_evidence']['admitted_new_capacity_used_by_later_successful_execution'] = True
    required['development_transforms'] = list(DEVELOPMENT_TRANSFORMS)
    required['final_transforms'] = list(HELD_BACK_TRANSFORMS)
    protection = protected_snapshot(ROOT)
    for path in SOURCE.rglob('*'):
        if path.is_file() and (path.suffix in {'.json', '.sqlite3'} or path.name in {'pipeline.py', 'router.py'}):
            protection[str(path.relative_to(ROOT))] = file_hash(path)
    directory.mkdir(exist_ok=False)
    write_new(directory / 'protocol.json', config)
    write_new(directory / 'completion_requirements.json', required)
    write_new(directory / 'protected_before.json', protection)
    print(json.dumps({'initialized': str(directory), 'baseline': baseline, 'known_states': len(known)}), flush=True)


def load(directory, arm, seed):
    directory = inside(ROOT, directory); config = read(directory / 'protocol.json')
    if (config['implementations'] != implementation_hashes()
            or config['engine_fingerprint'] != engine_fingerprint()):
        raise ValueError('Frozen stable-study implementation changed')
    source = inside(ROOT, config['source'])
    if any(file_hash(source / n) != h for n, h in config['source_hashes'].items()):
        raise ValueError('Prior study evidence changed')
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError('Unknown research arm or seed')
    fields = inside(ROOT, config['fields'])
    if file_hash(fields / 'dataset.json') != config['fields_manifest_sha256']:
        raise ValueError('Source data manifest changed')
    _, original, samples = load_study(fields)
    samples += held_back_samples(samples, config['held_back_transforms'])
    values = config['evaluation']
    protocol = replace(original, **{k: tuple(v) if k.endswith('_groups') or k == 'seeds' else v for k, v in values.items()})
    protocol = replace(protocol, seeds=(seed,))
    if sample_fingerprint(samples) != protocol.data_version:
        raise ValueError('Frozen data changed')
    episode = directory / f'{arm}-s{seed}'
    archive = Archive(episode / 'archive', protocol, create=not (episode / 'archive').exists())
    runner = StabilitySystemExecutor(ROOT, archive,
        ConfiguredSandbox(ROOT, ROOT / '.venvs/sam2', timeout=900, gpu=3 if arm == 'full' else 4),
        samples, directory / 'shared_inputs', config['development_transforms'],
        [inside(ROOT, p) for p in config['readonly_input_pools']])
    return directory, config, episode, archive, runner


def field_diagnostics(runner):
    observations = []
    for sample in runner.cohorts('development'):
        fields, valid = sample.fields, sample.fields['valid']
        values = {}
        for name in ('base', 'physics', 'student', 'variation'):
            mask = valid & fields['physics_valid'] if name == 'physics' else valid
            values[name] = {'mask': 'valid & physics_valid' if name == 'physics' else 'valid',
                'information': field_information(fields[name], mask),
                'ordinal_rank_probe': ordinal_tie_probe(fields[name], mask)}
        observations.append({'case': sample.key, 'group': sample.group, 'fields': values})
    return observations


def probes(runner, candidate):
    return {t: {'trial': trial, 'receipt': receipt}
            for t in runner.development_transforms
            for trial, receipt in [runner.probe_system(candidate, t)]}


def stability_feedback(runner, candidate):
    original = latest(runner.archive, candidate)['receipt']
    records = []
    for transform, measured in probes(runner, candidate).items():
        receipt = measured['receipt']
        if 'rows' not in receipt:
            records.append({'transform': transform, 'error': receipt.get('error')})
            continue
        moved = {r['case']: r for r in receipt['rows']}
        for row in original['rows']:
            key = row['case']; other = moved[transform + '-' + key]
            with np.load(runner.checked_ref(original['predictions'][key]), allow_pickle=False) as a:
                with np.load(runner.checked_ref(receipt['predictions'][other['case']]), allow_pickle=False) as b:
                    diagnostic = prediction_stability(a['score'], a['support'], b['score'], b['support'],
                        transform, runner.protocol.area_fraction, original_branch=row['branch'], transformed_branch=other['branch'])
            records.append({'case': key, 'group': row['group'], 'transform': transform,
                'original_coverage': row['coverage'], 'transformed_coverage': other['coverage'],
                'original_branch': row['branch'], 'transformed_branch': other['branch'],
                'observable_diagnostic': diagnostic})
    return records


def prepare(directory, arm, seed):
    _, config, episode, archive, runner = load(directory, arm, seed)
    readers = {}
    try:
        if (episode / 'prepared.json').exists():
            raise FileExistsError('Already prepared')
        source = inside(ROOT, config['source']); reference = config['baseline']
        for key in {(reference['arm'], reference['seed'])} | {(e['arm'], e['seed']) for e in config['imports']}:
            _, _, _, old_archive, old_runner = previous.load(source, *key)
            readers[key] = old_archive, old_runner
        origin = readers[reference['arm'], reference['seed']][1]
        extras = [(readers[e['arm'], e['seed']][1], e['binding']) for e in config['imports']]
        baseline = runner.import_system(origin, reference['candidate'], extras)
        trial, receipt = runner.execute(baseline)
        if 'rows' not in receipt or any(receipt['summary'][g]['mean'] != v['mean'] for g, v in reference['summary'].items()):
            raise ValueError('Best complete baseline did not reproduce exactly')
        archive.select(baseline, True, 'exact inherited complete baseline; stability not assumed', trial)
        archive.event('baseline_registered', {'candidate': baseline, 'trial': trial})
        bank = ResearchBank(runner); bank.initialize()
        measured = probes(runner, baseline)
        if any('rows' not in p['receipt'] for p in measured.values()):
            raise ValueError('Baseline development probe execution failed')
        write_new(episode / 'prepared.json', {'baseline': baseline, 'trial': trial, 'summary': receipt['summary'],
            'baseline_library': bank.inherited_manifest()['library'], 'checkpoint': bank.checkpoint(),
            'development_stability': measured, 'field_diagnostics': field_diagnostics(runner)})
        print(json.dumps({'arm': arm, 'seed': seed, 'prepared': True, 'known_capabilities': len(bank.entries())}), flush=True)
    finally:
        for old_archive, _ in readers.values():
            old_archive.close()
        archive.close()


def queue(directory, arm, seed):
    _, config, episode, archive, runner = load(directory, arm, seed)
    try:
        rounds = archive.events(ROUND_EVENT)
        if len(rounds) >= config['rounds_per_arm'] or archive.events('search_sealed'):
            print(json.dumps({'finished': True, 'rounds': len(rounds)}), flush=True); return
        folder = episode / 'requests' / f'r{len(rounds)+1:02d}'
        if folder.exists():
            print(json.dumps({'request': str(folder / 'request.json'), 'resumed': True}), flush=True); return
        if shutil.disk_usage(ROOT).free < config['minimum_free_disk_bytes']:
            raise RuntimeError('Disk reserve reached; all evidence preserved')
        prepared = read(episode / 'prepared.json'); bank = ResearchBank(runner)
        current = archive.incumbent(); manifest = bank.inherited_manifest()
        aliases = {g: f'D{i}' for i, g in enumerate(runner.protocol.train_groups + runner.protocol.validation_groups)}
        rows = latest(archive, current)['receipt']['rows']
        branch_rows = {n: read(runner.checked_ref(e['feedback']))['rows'] for n, e in bank.entries().items()}
        old_rows = {n: branch_rows[n] for n in prepared['baseline_library']}
        prior = []
        for event in rounds:
            old = episode / 'requests' / f'r{event["body"]["round"]:02d}'
            raw = read(old / 'response.json')['raw_text']
            try:
                value = json.loads(raw)
            except ValueError:
                value = {'invalid_raw_text': raw}
            prior.append({'round': event['body']['round'], 'status': event['body']['status'],
                          'proposal': value, 'response_sha256': file_hash(old / 'response.json')})
        context = {'current_system': current, 'checkpoint': bank.checkpoint(),
            'allowed_actions': allowed_actions(arm), 'next_branch_name': next_branch_name(manifest['library']),
            'active_branches': manifest['active_branches'], 'capability_library': anonymize(bank.catalog(), aliases),
            'current_router_code': (Path(archive.candidate(current)['directory']) / 'router.py').read_text(),
            'current_summary': anonymize(latest(archive, current)['receipt']['summary'], aliases),
            'frozen_best_complete_summary': anonymize(prepared['summary'], aliases),
            'bank_gap_diagnosis': anonymize(library_gaps(rows, branch_rows), aliases),
            'frozen_library_gap': anonymize(library_gaps(rows, old_rows), aliases),
            'observable_feature_definitions': MASK_FEATURE_CONTRACT,
            'anonymous_observations': anonymize(previous.observations(runner), aliases),
            'observable_field_information': anonymize(prepared['field_diagnostics'], aliases),
            'current_development_stability': anonymize(stability_feedback(runner, current), aliases),
            'development_transforms': config['development_transforms'],
            'experience': anonymize(archive.memories(), aliases), 'prior_proposals': prior,
            'all_attempt_feedback': anonymize([e['body'] for e in rounds], aliases),
            'round': len(rounds)+1, 'remaining_rounds': config['rounds_per_arm']-len(rounds),
            'training_source': 'D0', 'gpu_available': True, 'new_annotation_required': False,
            'historically_exposed_sources': True}
        payload = {'model': MODEL, 'instructions': INSTRUCTIONS, 'response_schema': response_schema(arm), 'context': context}
        if any(g in canonical(payload) for g in runner.protocol.train_groups + runner.protocol.validation_groups + runner.protocol.audit_groups):
            raise ValueError('Named source leaked into model input')
        folder.mkdir(parents=True, exist_ok=False)
        write_new(folder / 'request.json', payload)
        write_new(folder / 'reservation.json', {'checkpoint': context['checkpoint'], 'request_sha256': file_hash(folder / 'request.json')})
        archive.event('stable_request_reserved', {'round': len(rounds)+1, 'checkpoint': context['checkpoint'],
                                                'request_sha256': file_hash(folder / 'request.json')})
        print(json.dumps({'request': str(folder / 'request.json'), 'arm': arm, 'seed': seed}), flush=True)
    finally:
        archive.close()


def retained_stability(archive, runner, accepted):
    measured = {identity: probes(runner, identity) for identity in accepted}
    return {transform: {g: max(m[transform]['receipt']['summary'][g]['mean'] for m in measured.values())
                       for g in runner.protocol.train_groups + runner.protocol.validation_groups}
            for transform in runner.development_transforms}


def execute(directory, arm, seed):
    _, config, episode, archive, runner = load(directory, arm, seed)
    try:
        index = len(archive.events(ROUND_EVENT))+1
        if index > config['rounds_per_arm'] or archive.events('search_sealed'):
            raise ValueError('Closed search')
        folder = episode / 'requests' / f'r{index:02d}'
        request, reservation = read(folder / 'request.json'), read(folder / 'reservation.json')
        response = read(folder / 'response.json'); bank = ResearchBank(runner)
        if (reservation['checkpoint'] != bank.checkpoint() or request['context']['checkpoint'] != reservation['checkpoint']
                or file_hash(folder / 'request.json') != reservation['request_sha256']
                or response['request_sha256'] != reservation['request_sha256']
                or response['raw_text'] != (folder / 'raw_response.json').read_text()
                or file_hash(folder / 'response.json') != read(folder / 'response.ready.json')['response_sha256']):
            raise ValueError('Changed request, raw response or inherited state')
        result = {'round': index, 'origin': response['origin'], 'parent_checkpoint': reservation['checkpoint'],
            'request_sha256': reservation['request_sha256'], 'response_sha256': file_hash(folder / 'response.json'),
            'candidate': None, 'actual_execution': False}
        parent = archive.incumbent(); manifest = bank.inherited_manifest()
        try:
            value = json.loads(response['raw_text'])
            proposal, diagnosis = parse_action(value, arm, parent, manifest['library'])
        except (ValueError, TypeError, KeyError, AttributeError, SyntaxError) as error:
            result.update(status='invalid_proposal', error=f'{type(error).__name__}: {error}', counted_against_proposal_budget=True)
        else:
            identity = None
            result.update(action=value['action'], diagnosis=diagnosis, operation=proposal.operation)
            try:
                result['actual_execution'] = True
                identity = runner.register_system(proposal, manifest); result['candidate'] = identity
                trial, receipt = runner.execute(identity); result['trial'] = trial
                if 'rows' not in receipt:
                    result['research_bank_additions'] = bank.retain_measured(identity)
                    raise RuntimeError(receipt.get('error'))
                if proposal.new_branch:
                    _, spec = runner.system(identity)
                    result['analysis_replay'] = previous.analysis_replay(runner, spec['library'][proposal.new_branch['name']])
                result['research_bank_additions'] = bank.retain_measured(identity)
                replay_trial, replay = runner.execute(identity, replay=True)
                if 'rows' not in replay or replay['router_checkpoint'] != receipt['router_checkpoint']:
                    raise RuntimeError('Whole-system state replay failed')
                for case, ref in receipt['predictions'].items():
                    with np.load(runner.checked_ref(ref), allow_pickle=False) as a:
                        with np.load(runner.checked_ref(replay['predictions'][case]), allow_pickle=False) as b:
                            if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                                raise RuntimeError('Whole-system replay changed predictions')
                if {r['case']: r['branch'] for r in receipt['rows']} != {r['case']: r['branch'] for r in replay['rows']}:
                    raise RuntimeError('Whole-system replay changed adoption')
                measured = probes(runner, identity)
                prepared = read(episode / 'prepared.json')
                accepted = {e['body']['candidate'] for e in archive.events('selection') if e['body']['accepted']}
                history = [latest(archive, c)['receipt']['summary'] for c in accepted]
                retained = {g: max(s[g]['mean'] for s in history) for g in receipt['summary']}
                quality = expansion_admission(receipt['rows'], latest(archive, parent)['receipt']['rows'],
                    latest(archive, prepared['baseline'])['receipt']['rows'], archive.protocol, retained)
                stability_reference = retained_stability(archive, runner, accepted)
                decision = stability_admission(quality, receipt['rows'], {t: p['receipt'] for t, p in measured.items()},
                    config['development_transforms'], archive.protocol.maximum_group_regression, stability_reference)
                frozen = {n: read(runner.checked_ref(b).parent / 'development/predictions.json')['rows']
                          for n, b in prepared['baseline_library'].items()}
                novelty = selected_novel_gain(receipt['rows'], frozen, set(prepared['baseline_library']))
                archive.select(identity, decision['accepted'], decision, trial)
                archive.remember(identity, trial, {'action': value['action'], 'diagnosis': diagnosis,
                    'decision': decision, 'summary': receipt['summary'], 'novelty': novelty,
                    'stability_trials': {t: p['trial'] for t, p in measured.items()}}, {'scope': 'historical development and declared stability only'})
                result.update(status='accepted' if decision['accepted'] else 'retained_for_research',
                    summary=receipt['summary'], choices=receipt['choices'], decision=decision, novelty=novelty,
                    development_stability=measured, stability_reference=stability_reference,
                    replay_trial=replay_trial, exact_replay=True)
            except (ValueError, KeyError, RuntimeError, TypeError, SyntaxError, OSError) as error:
                result.update(status='execution_failed', error=str(error), candidate=identity)
        result['checkpoint'] = bank.checkpoint()
        write_new(folder / 'result.json', result); archive.event(ROUND_EVENT, result)
        print(json.dumps({'arm': arm, 'seed': seed, **result}, ensure_ascii=False), flush=True)
    finally:
        archive.close()


def audit(directory):
    directory = inside(ROOT, directory)
    for arm in ARMS:
        for seed in SEEDS:
            _, config, _, archive, _ = load(directory, arm, seed)
            try:
                if len(archive.events(ROUND_EVENT)) != config['rounds_per_arm']:
                    raise ValueError('Every search must finish before held-back execution')
            finally:
                archive.close()
    for arm in ARMS:
        for seed in SEEDS:
            _, _, _, archive, runner = load(directory, arm, seed)
            try:
                if not archive.events('search_sealed'):
                    archive.event('search_sealed', ResearchBank(runner).checkpoint())
            finally:
                archive.close()
    records = []
    for arm in ARMS:
        for seed in SEEDS:
            _, _, episode, archive, runner = load(directory, arm, seed)
            try:
                prepared = read(episode / 'prepared.json'); measured = {}; audits = {}
                for role, candidate in {'frozen': prepared['baseline'], 'selected': archive.incumbent()}.items():
                    if candidate not in measured:
                        trial, receipt = runner.execute(candidate, split='audit')
                        measured[candidate] = {'trial': trial, 'receipt': receipt}
                    audits[role] = measured[candidate]
                records.append({'arm': arm, 'seed': seed, 'prepared': prepared,
                    'rounds': [e['body'] for e in archive.events(ROUND_EVENT)],
                    'development': latest(archive, archive.incumbent())['receipt'], 'stress_audit': audits,
                    'development_stability': probes(runner, archive.incumbent()),
                    'checkpoint': ResearchBank(runner).checkpoint(), 'archive_tail': archive.verify()})
            finally:
                archive.close()
    write_new(directory / 'results.json', {'trajectories': records, 'production_replaced': False,
        'protected': verify_snapshot(ROOT, read(directory / 'protected_before.json')), 'new_blind_test': False})
    print(json.dumps({'completed': str(directory / 'results.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('initialize', 'prepare', 'queue', 'publish', 'execute', 'audit'))
    parser.add_argument('directory'); parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS); parser.add_argument('--origin'); parser.add_argument('--tests')
    args = parser.parse_args(); directory = inside(ROOT, args.directory)
    if args.command == 'publish':
        previous.publish(directory, args.origin)
    else:
        label = f'{args.arm}-{args.seed}' if args.arm else 'study'
        with directory.with_name(directory.name + '.' + label + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.command == 'initialize':
                if not args.tests:
                    parser.error('--tests is required')
                initialize(directory, args.tests)
            elif args.command == 'audit':
                audit(directory)
            else:
                if args.arm is None or args.seed is None:
                    parser.error('--arm and --seed are required')
                {'prepare': prepare, 'queue': queue, 'execute': execute}[args.command](directory, args.arm, args.seed)
