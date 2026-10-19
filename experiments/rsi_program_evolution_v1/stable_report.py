"""Verify the paired stable-study execution chain before claiming expansion."""
import argparse
from copy import deepcopy
import html
import json
import xml.etree.ElementTree as ET

import numpy as np

from .bank_report import paired_decision
from .contracts import file_hash, inside, verify_snapshot, write_new
from .evaluation import evaluate
from .expansion_state import expansion_admission, selected_novel_gain
from .frontier_report import table
from .research_state_v2 import ResearchBank, parse_action
from .stable_study import ARMS, SEEDS, ROUND_EVENT, load
from .stability_state import stability_admission
from .system_state import read
from .real_study import ROOT


def paired_stability_decision(full, route, config, required, novelty):
    checks = {}
    for transform in config['held_back_transforms']:
        value = deepcopy(full)
        for role in ('frozen', 'selected'):
            summary = full['stress_audit'][role]['receipt'].get('summary', {})
            prefix = f'held-{transform}-'
            value['stress_audit'][role]['receipt']['summary'] = {
                'stress-' + g.removeprefix(prefix): r for g, r in summary.items() if g.startswith(prefix)}
        checks[transform] = paired_decision(value, route, config, required, novelty)
    if not checks:
        raise ValueError('Declared held-back transform checks required')
    first = next(iter(checks.values()))
    inherited_new = []
    groups = config['evaluation']['validation_groups']
    for index, row in enumerate(full['rounds']):
        gains = row.get('novelty', {}).get('mean_selected_new_gain_over_old_casewise_ceiling', {})
        if (row['status'] != 'accepted' or not set(groups) <= set(gains)
                or np.mean([gains[g] for g in groups]) < required['minimum_positive_selected_new_capacity_gain_over_old_library_ceiling']):
            continue
        later = [r for r in full['rounds'][index+1:]
                 if r['status'] in {'accepted', 'retained_for_research'} and r.get('actual_execution')
                 and r.get('parent_checkpoint', {}).get('system', {}).get('executable_system') == row['candidate']]
        if later:
            inherited_new.append(row['candidate'])
    return dict(first, held_back_checks=checks,
                admitted_new_capacity_inherited=bool(inherited_new), inherited_new_capacity_states=inherited_new,
                passed=all(v['passed'] for v in checks.values()) and bool(inherited_new),
                stress_vs_frozen_retained=all(v['stress_vs_frozen_retained'] for v in checks.values()),
                stress_vs_own_original_retained=all(v['stress_vs_own_original_retained'] for v in checks.values()))


def verify(directory, results):
    config = read(directory / 'protocol.json')
    proof = {'protected': verify_snapshot(ROOT, read(directory / 'protected_before.json')),
        'independent_proposals': 0, 'paired_initial_inputs': 0, 'recomputed_system_rows': 0,
        'recomputed_capability_rows': 0, 'verified_reverse_analysis_rows': 0,
        'recomputed_admission_decisions': 0, 'invalid_attempts_counted': 0}
    keys = ('current_system', 'active_branches', 'capability_library', 'current_router_code', 'current_summary',
        'frozen_best_complete_summary', 'bank_gap_diagnosis', 'frozen_library_gap', 'observable_feature_definitions',
        'anonymous_observations', 'observable_field_information', 'current_development_stability',
        'development_transforms', 'training_source', 'round', 'remaining_rounds')
    for seed in SEEDS:
        pair = [read(directory / f'{a}-s{seed}/requests/r01/request.json')['context'] for a in ARMS]
        if any(pair[0][k] != pair[1][k] for k in keys):
            raise ValueError('Paired scientific starting evidence differs')
        proof['paired_initial_inputs'] += 1
    origins = set()
    for record in results['trajectories']:
        _, _, episode, archive, runner = load(directory, record['arm'], record['seed'])
        try:
            bank = ResearchBank(runner); trials = {t['id']: t for t in archive.trials(visible_only=False)}
            if bank.checkpoint() != record['checkpoint'] or archive.verify() != record['archive_tail']:
                raise ValueError('Reported final executable state changed')
            seals = archive.events('search_sealed')
            if len(seals) != 1 or seals[0]['body'] != record['checkpoint']:
                raise ValueError('Reported state was not uniquely sealed')
            events = [e['body'] for e in archive.events(ROUND_EVENT)]
            if events != record['rounds'] or len(events) != config['rounds_per_arm']:
                raise ValueError('Missing or changed research attempts')
            parent = record['prepared']['checkpoint']
            base = trials[record['prepared']['trial']]['receipt']
            accepted = {record['prepared']['baseline']: base}
            stability_history = {record['prepared']['baseline']: record['prepared']['development_stability']}
            for row in events:
                folder = episode / 'requests' / f'r{row["round"]:02d}'
                request, response = read(folder / 'request.json'), read(folder / 'response.json')
                if (row != read(folder / 'result.json') or row['parent_checkpoint'] != parent
                        or request['context']['checkpoint'] != parent
                        or response['request_sha256'] != file_hash(folder / 'request.json')
                        or row['request_sha256'] != response['request_sha256']
                        or row['response_sha256'] != file_hash(folder / 'response.json')
                        or file_hash(folder / 'response.json') != read(folder / 'response.ready.json')['response_sha256']
                        or response['raw_text'] != (folder / 'raw_response.json').read_text()):
                    raise ValueError('Request, original response, or inherited state changed')
                if response['origin'] in origins or response['origin'] != row['origin']:
                    raise ValueError('Independent proposer origin missing or repeated')
                origins.add(response['origin'])
                error = None
                try:
                    parse_action(json.loads(response['raw_text']), record['arm'], request['context']['current_system'],
                                 request['context']['capability_library'])
                except (ValueError, TypeError, KeyError, AttributeError, SyntaxError) as failure:
                    error = f'{type(failure).__name__}: {failure}'
                if row['status'] == 'invalid_proposal':
                    if (error != row['error'] or not error or row['actual_execution'] or row['candidate'] is not None
                            or row['checkpoint'] != parent or not row['counted_against_proposal_budget']):
                        raise ValueError('Invalid proposal was repaired, executed, or escaped its budget')
                    proof['invalid_attempts_counted'] += 1
                elif error:
                    raise ValueError('Invalid proposal reached execution')
                if row.get('candidate'):
                    _, child = runner.system(row['candidate'])
                    _, old = runner.system(parent['system']['executable_system'])
                    if any(child['library'].get(n) != b for n, b in old['library'].items()):
                        raise ValueError('Prior capability was rebound')
                if row.get('analysis_replay'):
                    replay = read(runner.checked_ref(row['analysis_replay']))
                    path = runner.checked_ref(replay['binding']); cap = read(path)
                    original = read(path.parent / 'development/predictions.json')
                    if (replay['checkpoint'] != cap['checkpoint'] or not replay['exact_reverse_replay']
                            or set(replay['predictions']) != set(original['predictions'])):
                        raise ValueError('New analysis replay changed cohort or numerical state')
                    for case, ref in replay['predictions'].items():
                        with np.load(runner.checked_ref(ref), allow_pickle=False) as a:
                            with np.load(runner.checked_ref(original['predictions'][case]), allow_pickle=False) as b:
                                if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                                    raise ValueError('New analysis is not case-order reproducible')
                        proof['verified_reverse_analysis_rows'] += 1
                if row['status'] in {'accepted', 'retained_for_research'}:
                    receipt = trials[row['trial']]['receipt']
                    before = accepted[parent['system']['executable_system']]
                    retained = {g: max(r['summary'][g]['mean'] for r in accepted.values()) for g in base['summary']}
                    quality = expansion_admission(receipt['rows'], before['rows'], base['rows'], runner.protocol, retained)
                    reference = {t: {g: max(h[t]['receipt']['summary'][g]['mean'] for h in stability_history.values())
                                     for g in base['summary']} for t in config['development_transforms']}
                    probes = row['development_stability']
                    for t, probe in probes.items():
                        if trials[probe['trial']]['receipt'] != probe['receipt'] or trials[probe['trial']]['split'] != 'smoke':
                            raise ValueError('Stability decision is detached from actual execution')
                        if probe['receipt']['router_checkpoint'] != receipt['router_checkpoint']:
                            raise ValueError('Stability probe changed router weights')
                    expected = stability_admission(quality, receipt['rows'], {t: p['receipt'] for t, p in probes.items()},
                        config['development_transforms'], runner.protocol.maximum_group_regression, reference)
                    if expected != row['decision'] or reference != row['stability_reference']:
                        raise ValueError('Recorded admission is not reproducible')
                    if expected['accepted'] != (row['status'] == 'accepted'):
                        raise ValueError('Adoption ignored the executed admission result')
                    proof['recomputed_admission_decisions'] += 1
                    if expected['accepted']:
                        accepted[row['candidate']] = receipt; stability_history[row['candidate']] = probes
                parent = row['checkpoint']
            if parent != record['checkpoint']:
                raise ValueError('Final state did not follow from the last recorded round')
            samples = {s.key: s for s in runner.samples}
            for t in config['development_transforms']:
                samples.update({s.key: s for s in runner.cohorts('stability-' + t)})
            def check_rows(receipt, counter):
                for row in receipt.get('rows', []):
                    sample = samples[row['case']]
                    if row['group'] != sample.group or row['seed'] != record['seed']:
                        raise ValueError('Prediction source/seed mismatch')
                    with np.load(runner.checked_ref(receipt['predictions'][row['case']]), allow_pickle=False) as z:
                        actual = evaluate(sample.validation_target(), dict(z), runner.protocol)
                    if any(row.get(k) != v for k, v in actual.items()):
                        raise ValueError('Saved arrays do not reproduce the measured result')
                    proof[counter] += 1
            for trial in trials.values():
                if trial['split'] == 'audit':
                    event = next(e for e in archive.events('trial_recorded') if e['body']['trial'] == trial['id'])
                    if event['seq'] <= seals[0]['seq'] or trial['receipt'].get('router_training'):
                        raise ValueError('Final stress was accessed before sealing or refitted')
                check_rows(trial['receipt'], 'recomputed_system_rows')
            for event in archive.events('capability_predictions_recorded'):
                body = event['body']
                path = episode / 'capabilities' / body['capability'] / body['split'] / 'predictions.json'
                if file_hash(path) != body['sha256']:
                    raise ValueError('Capability execution evidence changed')
                check_rows(read(path), 'recomputed_capability_rows')
            for candidate in archive.pool():
                folder, _ = runner.system(candidate['id'])
                if (folder / 'system_state.json').exists():
                    runner.frozen_state(candidate['id'])
        finally:
            archive.close()
    proof['independent_proposals'] = len(origins)
    tests = inside(ROOT, config['preflight_tests']['path'])
    if file_hash(tests) != config['preflight_tests']['sha256']:
        raise ValueError('Preflight test evidence changed')
    suites = ET.parse(tests).getroot().findall('.//testsuite')
    proof['tests'] = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if not proof['tests']['tests'] or proof['tests']['failures'] or proof['tests']['errors']:
        raise ValueError('Preflight did not pass')
    return proof


def report(directory):
    directory = inside(ROOT, directory)
    results, config = read(directory / 'results.json'), read(directory / 'protocol.json')
    required = read(directory / 'completion_requirements.json')['required_evidence']
    proof = verify(directory, results); indexed = {(r['arm'], r['seed']): r for r in results['trajectories']}
    criteria = []
    for seed in SEEDS:
        full, route = indexed['full', seed], indexed['route', seed]
        if full['prepared']['summary'] != route['prepared']['summary']:
            raise ValueError('Paired baselines differ')
        _, _, _, archive, runner = load(directory, 'full', seed)
        try:
            library = full['prepared']['baseline_library']
            rows = {n: read(runner.checked_ref(b).parent / 'development/predictions.json')['rows'] for n, b in library.items()}
            novelty = selected_novel_gain(full['development']['rows'], rows, set(library))
        finally:
            archive.close()
        criteria.append(paired_stability_decision(full, route, config, required, novelty))
    summary = {'verification': proof, 'paired_criteria': criteria,
        'batch_requirements_satisfied': sum(c['passed'] for c in criteria) >= required['minimum_successful_paired_trajectories'],
        'results_sha256': file_hash(directory / 'results.json'), 'report_source_sha256': file_hash(__file__),
        'production_replaced': False, 'new_blind_test': False,
        'metric': 'known-positive coverage at fixed 10% entire-image budget, not accuracy or IoU',
        'scope': 'new capability creation AND stable integration; historical development sources'}
    write_new(directory / 'summary.json', summary)
    rows = [[c['seed'], f'{100*c["frozen_mean"]:.2f}%', f'{100*c["route_only_mean"]:.2f}%',
             f'{100*c["full_mean"]:.2f}%', f'{100*c["selected_new_capacity_gain"]:+.3f} pp',
             c['accepted_updates'], c['inherited_updates'], c['stress_vs_own_original_retained'], c['passed']] for c in criteria]
    page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSI 新能力与稳定继承</title>'
    page += '<style>body{max-width:1250px;margin:40px auto;font:17px Arial,sans-serif;line-height:1.7}table{border-collapse:collapse}td,th{border:1px solid #ccd9d6;padding:9px}pre{white-space:pre-wrap}</style>'
    page += '<h1>PhyEvo-Mic：创造新能力并稳定继承</h1><p>从上一批最强完整系统出发；两组共享全部已测能力。旋转开发检查参与准入，另两种变换仅在全部封存后检查。已知图像的变换不等于独立新采集数据。</p>'
    page += table(['种子', '冻结系统', '只改采用', '完整 RSI', '新能力超旧库贡献', '准入', '实际继承', '最终稳定', '通过'], rows)
    page += '<pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></html>'
    with (directory / 'report.html').open('x') as stream:
        stream.write(page)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory'); args = parser.parse_args()
    report(args.directory)
