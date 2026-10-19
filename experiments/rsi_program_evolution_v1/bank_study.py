"""New paired study with persistent research assets and explicit observables.

This does not resume or modify the sealed expansion batch. Both arms start at
its strongest admitted complete system and receive every successfully measured
prior capability, including rejected ones. Reusing them is not a new discovery.
"""
import argparse
import fcntl
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from . import expansion_study as previous
from .active_diagnostics import field_enrichment, label_shape, selected_mask
from .contracts import canonical, digest, file_hash, inside, protected_snapshot, verify_snapshot, write_new
from .expansion_state import expansion_admission, library_gaps, selected_novel_gain
from .research_state_v2 import MASK_FEATURE_CONTRACT, ResearchBank, allowed_actions, mask_observables, next_branch_name, parse_action, response_schema
from .real_study import ROOT
from .system_state import read
from .system_study import anonymize, latest


SOURCE = ROOT / 'outputs/rsi_capability_expansion_20261017_v2'
ARMS, SEEDS, MODEL = previous.ARMS, previous.SEEDS, previous.MODEL
ROUND_EVENT = 'bank_round_finished'
INSTRUCTIONS = '''Return exactly {diagnosis, action, system} following response_schema.
The diagnosis describes the observed problem, not a mandatory operation. A routing control can
diagnose a capability gap but can only choose action=route_existing. Do NOT add operation or
operation_target fields. The action alone determines the executor operation.
Choose from context.allowed_actions. No operation or training is mandatory. route_existing may
activate ANY program in capability_library, including inactive/rejected research assets. Their
archival does not mean they are deployed. All original branches remain available for retention.
For new analysis/learning use exactly context.next_branch_name. Set parent=current_system.
Always retain b0 in active_branches. New branch fields: name, pipeline_code, training_steps,
parent_branch (a bank alias donating frozen numerical weights or null), replaces (only for replace
actions, otherwise null). Remove a replaced optional branch from active_branches but not the bank.
create_analysis/replace_analysis declare zero branch training steps. train_student/replace_student
must actually train. Total router plus branch optimizer steps <=4096. No forced small architecture.
router_code defines route(sample,branches,state,context), returning one active branch name plus
reason; do not modify its score/support arrays. Each branches[name] has score and support.
A new pipeline defines predict(sample,state,context), returning finite 2D score and boolean support
inside valid. sample has image/base/support/valid/physics/student/physics_valid/variation/domain_labels.
Only fit(samples,context) sees human and known. All unmarked pixels are unknown, NOT negatives.
Only D0 can enter fitting; D1/D2 are weight-fit-excluded development sources. All sources are
historically exposed; this is not a fresh blind dataset. Do not request new annotations.
fit returns finite numeric numpy arrays: actual parameters and audit_initial_parameters,
audit_final_parameters, audit_loss_trace, audit_steps. Execute exactly the declared steps, respect
context.device and context.seed; parent weights are context.parent_state, when requested.
No network, filesystem access, repository imports, new packages, labels at inference, source/material
names, exact image-size or coordinate dispatch, sample index/call-order dispatch, or evaluator changes.
numpy/scipy/skimage/torch are available. Inference must replay exactly in reversed case order.
Use observable physical/geometry/disagreement evidence. Feature definitions specify denominators:
global image fraction and fraction within support are DIFFERENT quantities; unavailable is null.
positive_error_partition divides all expert-positive mass into disjoint missing-validity,
missing-support, nonpositive-score, eligible-but-unselected, and selected categories. It describes
the CURRENT executed system, not just its initial teacher support. These are development diagnostics,
not an inference-time input. Outside support cannot be recovered by reordering scores alone.
The candidate budget is 10% of the entire image, NOT 10% of support. Equal-source system gain must
reach 0.002, mean D1/D2 gain must also reach 0.002, and no source may regress >0.002 against baseline,
current, or best accepted history. A new mechanism can enter the research bank without deployment.
It can then be refined, adopted, or used as a weight parent by later proposals. No manual answer table.
prior_proposals contains the exact earlier proposals from this trajectory, including rejected
adoption programs and failed code. They are research evidence, NOT automatically accepted versions.
Old-library casewise best is an OFFLINE diagnostic ceiling, never a runtime selector. Genuine added
capacity must exceed that ceiling when actually selected, not just improve on a weak branch.
Provide concrete executable mechanisms and falsifiers, not measured-success claims before execution.
'''


def extension_hashes():
    return {name: file_hash(Path(__file__).with_name(name))
            for name in ('bank_study.py', 'research_state_v2.py', 'bank_report.py')}


def choose_baseline(results):
    options = []
    for row in results['trajectories']:
        summary = row['development']['summary']
        options.append({'arm': row['arm'], 'seed': row['seed'],
            'candidate': row['checkpoint']['executable_system'], 'summary': summary,
            'equal_source_mean': float(np.mean([v['mean'] for v in summary.values()]))})
    if not options:
        raise ValueError('Completed whole-system results are required')
    return max(options, key=lambda r: (r['equal_source_mean'], r['candidate'])), options


def capability_key(runner, binding):
    cap = read(runner.checked_ref(binding))
    program = runner.checked_ref(cap['program'])
    if cap['checkpoint']:
        runner.checked_ref(cap['checkpoint'])
    if set(cap['weight_training_groups']) & set(runner.protocol.validation_groups + runner.protocol.audit_groups):
        raise ValueError('Imported weights used an excluded training source')
    return digest({'program': previous.program_identity(program.read_text()),
                   'state': cap['checkpoint']['sha256'] if cap['checkpoint'] else None})


def collect_known(source, baseline):
    """All executed prior capabilities, not just hand-picked winners."""
    known, entries = set(), []
    _, _, _, archive, runner = previous.load(source, baseline['arm'], baseline['seed'])
    try:
        for binding in runner.frozen_state(baseline['candidate'])['library'].values():
            known.add(capability_key(runner, binding))
    finally:
        archive.close()
    for arm in ARMS:
        for seed in SEEDS:
            _, _, episode, archive, runner = previous.load(source, arm, seed)
            try:
                events = archive.events('capability_predictions_recorded')
                for path in sorted((episode / 'capabilities').glob('*/capability.json')):
                    feedback = path.parent / 'development/predictions.json'
                    if not feedback.is_file():
                        continue
                    binding = runner.ref(path)
                    key = capability_key(runner, binding)
                    if key in known:
                        continue
                    cap = read(path)
                    recorded = [e['body'] for e in events if e['body']['capability'] == cap['identity']
                                and e['body']['split'] == 'development']
                    if len(recorded) != 1 or recorded[0]['sha256'] != file_hash(feedback):
                        raise ValueError('Prior measurement is not bound to its execution')
                    known.add(key)
                    entries.append({'arm': arm, 'seed': seed, 'binding': binding, 'state_identity': key})
            finally:
                archive.close()
    return entries, sorted(known)


def initialize(directory):
    directory = inside(ROOT, directory)
    if directory.exists():
        raise FileExistsError('A new study destination is required')
    source = SOURCE
    summary, results = read(source / 'summary.json'), read(source / 'results.json')
    if summary['results_sha256'] != file_hash(source / 'results.json'):
        raise ValueError('Previous audit/report did not finish with bound results')
    tests = source / 'bank_study_preflight_v4.xml'
    suites = ET.parse(tests).getroot().findall('.//testsuite')
    if not suites or not sum(int(s.get('tests', 0)) for s in suites) or any(
            int(s.get('failures', 0)) or int(s.get('errors', 0)) for s in suites):
        raise ValueError('New study integration tests have not passed')
    baseline, compared = choose_baseline(results)
    extras, known = collect_known(source, baseline)
    config = read(source / 'protocol.json')
    config.update(bank_source=str(source.relative_to(ROOT)),
        bank_source_hashes={n: file_hash(source / n) for n in ('protocol.json', 'results.json', 'summary.json')},
        bank_reference=baseline, bank_reference_candidates=compared, bank_imports=extras,
        known_capability_state_identities=known, extension_sha256=extension_hashes(),
        preflight_tests={'path': str(tests.relative_to(ROOT)), 'sha256': file_hash(tests)},
        rounds_per_arm=8, study_scope='paired persistent research-bank study on historical development data',
        final_check='previously used 90-degree transform, now a regression/equivariance stress, not a fresh holdout',
        new_blind_dataset=False, comparison='full research versus route-existing-only, same complete best and bank',
        bank_retention_is_not_system_admission=True,
        novelty='newly selected program/state must exceed all known measured prior capabilities at fixed budget')
    config['evaluation'] = dict(config['evaluation'], maximum_rounds=8, maximum_model_calls=8)
    protection = protected_snapshot(ROOT)
    for path in source.rglob('*'):
        if path.is_file() and (path.suffix in {'.json', '.sqlite3'} or path.name in {'pipeline.py', 'router.py'}):
            protection[str(path.relative_to(ROOT))] = file_hash(path)
    requirements = read(source / 'completion_requirements.json')
    requirements.update(scope=config['study_scope'], baseline='strongest admitted complete v2 system plus ALL measured known prior capabilities')
    requirements['matched_research']['proposals_per_arm'] = 8
    requirements['required_evidence']['selected_stress_vs_own_original_within_tolerance'] = True
    requirements['stress_semantics'] = config['final_check']
    directory.mkdir(exist_ok=False)
    write_new(directory / 'protocol.json', config)
    write_new(directory / 'protected_before.json', protection)
    write_new(directory / 'completion_requirements.json', requirements)
    print(json.dumps({'initialized': str(directory), 'baseline': baseline,
                      'known_capability_states': len(known), 'additional_imports': len(extras)}), flush=True)


def load(directory, arm, seed):
    directory = inside(ROOT, directory)
    config = read(directory / 'protocol.json')
    if config['extension_sha256'] != extension_hashes():
        raise ValueError('Frozen research-bank implementation changed')
    source = inside(ROOT, config['bank_source'])
    if any(file_hash(source / n) != h for n, h in config['bank_source_hashes'].items()):
        raise ValueError('Previous study evidence changed')
    # Reuse unchanged input, GPU, evaluation, staging and isolation contracts.
    return previous.load(directory, arm, seed)


def prepare(directory, arm, seed):
    _, config, episode, archive, runner = load(directory, arm, seed)
    readers = {}
    try:
        if (episode / 'prepared.json').exists():
            raise FileExistsError('Already prepared')
        source = inside(ROOT, config['bank_source']); reference = config['bank_reference']
        keys = {(reference['arm'], reference['seed'])} | {(e['arm'], e['seed']) for e in config['bank_imports']}
        for a, s in sorted(keys):
            _, _, _, previous_archive, previous_runner = previous.load(source, a, s)
            readers[a, s] = (previous_archive, previous_runner)
        origin = readers[reference['arm'], reference['seed']][1]
        extras = [(readers[e['arm'], e['seed']][1], e['binding']) for e in config['bank_imports']]
        baseline = runner.import_system(origin, reference['candidate'], extras)
        trial, receipt = runner.execute(baseline)
        if 'rows' not in receipt or any(receipt['summary'][g]['mean'] != v['mean'] for g, v in reference['summary'].items()):
            raise ValueError('Strongest complete baseline failed exact source-score reproduction')
        archive.select(baseline, True, 'exact strongest admitted complete baseline', trial)
        archive.event('baseline_registered', {'candidate': baseline, 'trial': trial})
        bank = ResearchBank(runner)
        bank.initialize()
        write_new(episode / 'prepared.json', {'baseline': baseline, 'trial': trial,
            'summary': receipt['summary'], 'baseline_library': bank.inherited_manifest()['library'],
            'checkpoint': bank.checkpoint(), 'all_prior_capabilities_are_known': True})
        print(json.dumps({'arm': arm, 'seed': seed, 'prepared': True, 'known_capabilities': len(bank.entries())}), flush=True)
    finally:
        for old_archive, _ in readers.values():
            old_archive.close()
        archive.close()


def positive_error_partition(fields, human, score, support, fraction):
    """Label-side accounting with exactly the frozen area evaluator's budget."""
    human = np.asarray(human, bool); valid = np.asarray(fields['valid'], bool)
    score = np.asarray(score); support = np.asarray(support)
    if (human.ndim != 2 or valid.shape != human.shape or score.shape != human.shape
            or support.shape != human.shape or support.dtype != bool
            or not np.isfinite(score).all() or np.any(support & ~valid)):
        raise ValueError('Invalid aligned prediction for positive-error accounting')
    chosen = selected_mask(score, support, fraction)
    total = int(human.sum()); budget = max(1, int(np.ceil(fraction * human.size)))
    eligible = support & (score > 0)
    masks = {'outside_valid': ~valid, 'valid_outside_support': valid & ~support,
        'supported_nonpositive_score': support & ~eligible,
        'eligible_not_selected': eligible & ~chosen, 'selected': chosen}
    return {'denominator': 'all expert-positive pixels, including those outside support',
        'known_positive_pixels': total, 'whole_image_candidate_budget': budget,
        'fractions': {k: float((human & mask).sum()/total) if total else None for k, mask in masks.items()},
        'support_only_coverage_upper_bound': min(budget, int((human & support).sum()))/total if total else None,
        'initial_support_only_coverage_upper_bound': min(budget, int((human & valid & fields['support']).sum()))/total if total else None,
        'precision_measured': False}


def observations(runner):
    output = []
    predictions = latest(runner.archive, runner.archive.incumbent())['receipt']['predictions']
    for sample in runner.cohorts('development'):
        fields = sample.fields
        valid = fields['valid'] & np.isfinite(fields['variation'])
        values = fields['variation'][valid]
        with np.load(runner.checked_ref(predictions[sample.key]), allow_pickle=False) as arrays:
            errors = positive_error_partition(fields, sample.human, arrays['score'], arrays['support'], runner.protocol.area_fraction)
        output.append({'group': sample.group, 'mask_features': mask_observables(fields),
            'positive_error_partition': errors,
            'variation_q10_q50_q90_within_valid': np.quantile(values, [.1, .5, .9]).tolist() if values.size else None,
            'expert_geometry_summary': label_shape(sample.human),
            'positive_field_percentiles': {f: field_enrichment(fields[f], sample.human, fields['valid'])
                for f in ('base', 'variation', 'physics', 'student')}})
    return output


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
            raise RuntimeError('Disk reserve reached; prior evidence is preserved')
        bank = ResearchBank(runner); manifest = bank.inherited_manifest(); current = archive.incumbent()
        aliases = {g: f'D{i}' for i, g in enumerate(runner.protocol.train_groups + runner.protocol.validation_groups)}
        prepared = read(episode / 'prepared.json')
        rows = latest(archive, current)['receipt']['rows']
        branch_rows = {name: read(runner.checked_ref(entry['feedback']))['rows'] for name, entry in bank.entries().items()}
        frozen = {name: branch_rows[name] for name in prepared['baseline_library']}
        proposals = []
        for event in rounds:
            old = episode / 'requests' / f'r{event["body"]["round"]:02d}'
            response = read(old / 'response.json')
            try:
                value = json.loads(response['raw_text'])
            except ValueError:
                value = {'invalid_raw_text': response['raw_text']}
            proposals.append({'round': event['body']['round'], 'status': event['body']['status'],
                'proposal': value, 'response_sha256': file_hash(old / 'response.json')})
        context = {'current_system': current, 'checkpoint': bank.checkpoint(),
            'allowed_actions': allowed_actions(arm), 'next_branch_name': next_branch_name(manifest['library']),
            'active_branches': manifest['active_branches'], 'capability_library': anonymize(bank.catalog(), aliases),
            'current_router_code': (Path(archive.candidate(current)['directory']) / 'router.py').read_text(),
            'current_summary': anonymize(latest(archive, current)['receipt']['summary'], aliases),
            'frozen_best_complete_summary': anonymize(prepared['summary'], aliases),
            'bank_gap_diagnosis': anonymize(library_gaps(rows, branch_rows), aliases),
            'frozen_library_gap': anonymize(library_gaps(rows, frozen), aliases),
            'observable_feature_definitions': MASK_FEATURE_CONTRACT,
            'anonymous_observations': anonymize(observations(runner), aliases),
            'experience': anonymize(archive.memories(), aliases),
            'prior_proposals': proposals,
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
        archive.event('bank_request_reserved', {'round': len(rounds)+1, 'checkpoint': context['checkpoint'],
            'request_sha256': file_hash(folder / 'request.json')})
        print(json.dumps({'request': str(folder / 'request.json'), 'arm': arm, 'seed': seed}), flush=True)
    finally:
        archive.close()


def publish(folder, origin):
    folder = inside(ROOT, folder)
    if not isinstance(origin, str) or not origin:
        raise ValueError('Independent model origin required')
    raw = (folder / 'raw_response.json').read_text()
    # Invalid JSON is preserved, then consumes one attempt in execute.
    write_new(folder / 'response.json', {'raw_text': raw, 'origin': origin,
        'request_sha256': file_hash(folder / 'request.json'), 'requested_model': MODEL,
        'returned_model': None, 'usage': None})
    write_new(folder / 'response.ready.json', {'response_sha256': file_hash(folder / 'response.json')})
    print(json.dumps({'published': str(folder), 'origin': origin}), flush=True)


def analysis_replay(runner, binding):
    """Actually rerun a new analysis branch, not just cached branch selection."""
    cap_path = runner.checked_ref(binding); cap = read(cap_path)
    folder = cap_path.parent / 'reverse_replay'
    original = runner.predict_capability(binding, 'development')
    if (folder / 'receipt.json').exists():
        raise FileExistsError('Analysis replay is immutable and may not be overwritten')
    folder.mkdir(exist_ok=False)
    cohort = list(reversed(runner.cohorts('development')))
    runner.stage(folder / 'input', cohort, state=cap['checkpoint'])
    execution = runner.sandbox.run(runner.checked_ref(cap['program']).parent, folder / 'input', folder / 'output')
    if execution['status'] != 'ok':
        raise RuntimeError('Analysis reverse replay did not complete')
    predictions = {}
    for index, sample in enumerate(cohort):
        path = folder / 'output' / f'prediction-{index:04d}.npz'
        with np.load(path, allow_pickle=False) as a:
            with np.load(runner.checked_ref(original['predictions'][sample.key]), allow_pickle=False) as b:
                if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                    raise RuntimeError('Analysis changes with case order or frozen-state reuse')
        predictions[sample.key] = runner.ref(path)
    write_new(folder / 'receipt.json', {'binding': binding, 'checkpoint': cap['checkpoint'],
        'execution': execution, 'predictions': predictions, 'exact_reverse_replay': True})
    return runner.ref(folder / 'receipt.json')


def execute(directory, arm, seed):
    _, config, episode, archive, runner = load(directory, arm, seed)
    try:
        rounds = archive.events(ROUND_EVENT); index = len(rounds)+1
        if index > config['rounds_per_arm'] or archive.events('search_sealed'):
            raise ValueError('Closed search')
        folder = episode / 'requests' / f'r{index:02d}'
        request = read(folder / 'request.json'); reservation = read(folder / 'reservation.json')
        response = read(folder / 'response.json'); bank = ResearchBank(runner)
        if (reservation['checkpoint'] != bank.checkpoint()
                or request['context']['checkpoint'] != reservation['checkpoint']
                or file_hash(folder / 'request.json') != reservation['request_sha256']
                or response['request_sha256'] != reservation['request_sha256']
                or response['raw_text'] != (folder / 'raw_response.json').read_text()
                or file_hash(folder / 'response.json') != read(folder / 'response.ready.json')['response_sha256']):
            raise ValueError('Request, raw reply, or inherited research state changed')
        result = {'round': index, 'origin': response['origin'], 'parent_checkpoint': reservation['checkpoint'],
            'request_sha256': reservation['request_sha256'], 'response_sha256': file_hash(folder / 'response.json'),
            'candidate': None, 'actual_execution': False}
        parent = archive.incumbent(); manifest = bank.inherited_manifest()
        try:
            value = json.loads(response['raw_text'])
            proposal, diagnosis = parse_action(value, arm, parent, manifest['library'])
        except (ValueError, TypeError, KeyError, AttributeError, SyntaxError) as error:
            result.update(status='invalid_proposal', error=f'{type(error).__name__}: {error}',
                counted_against_proposal_budget=True)
        else:
            result.update(action=value['action'], diagnosis=diagnosis, operation=proposal.operation)
            identity = None
            try:
                result['actual_execution'] = True
                identity = runner.register_system(proposal, manifest)
                result['candidate'] = identity
                trial, receipt = runner.execute(identity)
                result['trial'] = trial
                if 'rows' not in receipt:
                    # Preserve measured sub-capabilities even if the router failed.
                    result['research_bank_additions'] = bank.retain_measured(identity)
                    raise RuntimeError(receipt.get('error'))
                if proposal.new_branch:
                    _, spec = runner.system(identity)
                    binding = spec['library'][proposal.new_branch['name']]
                    result['analysis_replay'] = analysis_replay(runner, binding)
                result['research_bank_additions'] = bank.retain_measured(identity)
                prepared = read(episode / 'prepared.json')
                accepted = {e['body']['candidate'] for e in archive.events('selection') if e['body']['accepted']}
                histories = [latest(archive, i)['receipt']['summary'] for i in accepted]
                retained = {g: max(s[g]['mean'] for s in histories) for g in receipt['summary']}
                decision = expansion_admission(receipt['rows'], latest(archive, parent)['receipt']['rows'],
                    latest(archive, prepared['baseline'])['receipt']['rows'], archive.protocol, retained)
                replay_trial, replay = runner.execute(identity, replay=True)
                if 'rows' not in replay or replay['router_checkpoint'] != receipt['router_checkpoint']:
                    raise RuntimeError('Whole-system state replay failed')
                for case, reference in receipt['predictions'].items():
                    with np.load(runner.checked_ref(reference), allow_pickle=False) as a:
                        with np.load(runner.checked_ref(replay['predictions'][case]), allow_pickle=False) as b:
                            if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                                raise RuntimeError('Whole-system predictions changed on replay')
                if {r['case']: r['branch'] for r in replay['rows']} != {r['case']: r['branch'] for r in receipt['rows']}:
                    raise RuntimeError('Adoption changed on replay')
                frozen = {n: read(runner.checked_ref(b).parent / 'development/predictions.json')['rows']
                          for n, b in prepared['baseline_library'].items()}
                novelty = selected_novel_gain(receipt['rows'], frozen, set(prepared['baseline_library']))
                archive.select(identity, decision['accepted'], decision, trial)
                archive.remember(identity, trial, {'action': value['action'], 'diagnosis': diagnosis,
                    'decision': decision, 'summary': receipt['summary'], 'novelty': novelty}, {'scope': 'historical development only'})
                result.update(status='accepted' if decision['accepted'] else 'retained_for_research',
                    summary=receipt['summary'], choices=receipt['choices'], decision=decision, novelty=novelty,
                    replay_trial=replay_trial, exact_replay=True)
            except (ValueError, KeyError, RuntimeError, TypeError, SyntaxError, OSError) as error:
                result.update(status='execution_failed', error=str(error), candidate=identity)
        result['checkpoint'] = bank.checkpoint()
        write_new(folder / 'result.json', result)
        archive.event(ROUND_EVENT, result)
        print(json.dumps({'arm': arm, 'seed': seed, **result}, ensure_ascii=False), flush=True)
    finally:
        archive.close()


def audit(directory):
    directory = inside(ROOT, directory)
    for arm in ARMS:
        for seed in SEEDS:
            _, config, _, archive, runner = load(directory, arm, seed)
            try:
                if len(archive.events(ROUND_EVENT)) != config['rounds_per_arm']:
                    raise ValueError('Every paired search must finish first')
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
                prepared = read(episode / 'prepared.json'); measured = {}; outcomes = {}
                for role, identity in {'frozen': prepared['baseline'], 'selected': archive.incumbent()}.items():
                    if identity not in measured:
                        trial, receipt = runner.execute(identity, split='audit')
                        measured[identity] = {'trial': trial, 'receipt': receipt}
                    outcomes[role] = measured[identity]
                records.append({'arm': arm, 'seed': seed, 'prepared': prepared,
                    'rounds': [e['body'] for e in archive.events(ROUND_EVENT)],
                    'development': latest(archive, archive.incumbent())['receipt'], 'stress_audit': outcomes,
                    'checkpoint': ResearchBank(runner).checkpoint(), 'archive_tail': archive.verify()})
            finally:
                archive.close()
    write_new(directory / 'results.json', {'trajectories': records, 'production_replaced': False,
        'protected': verify_snapshot(ROOT, read(directory / 'protected_before.json')),
        'new_blind_test': False, 'stress_previously_observed': True})
    print(json.dumps({'completed': str(directory / 'results.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('initialize', 'prepare', 'queue', 'publish', 'execute', 'audit'))
    parser.add_argument('directory'); parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--seed', type=int, choices=SEEDS); parser.add_argument('--origin')
    args = parser.parse_args(); directory = inside(ROOT, args.directory)
    if args.command == 'publish':
        publish(directory, args.origin)
    else:
        label = f'{args.arm}-{args.seed}' if args.arm else 'study'
        with directory.with_name(directory.name + '.' + label + '.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if args.command in ('initialize', 'audit'):
                {'initialize': initialize, 'audit': audit}[args.command](directory)
            else:
                if args.arm is None or args.seed is None:
                    parser.error('--arm and --seed are required')
                {'prepare': prepare, 'queue': queue, 'execute': execute}[args.command](directory, args.arm, args.seed)
