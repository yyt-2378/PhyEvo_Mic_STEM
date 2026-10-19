"""Verify real research-bank trajectories and report, without changing selection."""
import argparse
import html
import json
import xml.etree.ElementTree as ET

import numpy as np

from .bank_study import ARMS, SEEDS, ROUND_EVENT, load
from .contracts import file_hash, inside, verify_snapshot, write_new
from .evaluation import evaluate
from .expansion_state import selected_novel_gain
from .frontier_report import table
from .real_study import ROOT
from .research_state_v2 import ResearchBank, parse_action
from .system_state import read


def paired_decision(full, route, config, requirement, novelty):
    groups = config['evaluation']['validation_groups']
    tol = config['evaluation']['maximum_group_regression']
    mean = lambda summary: float(np.mean([summary[g]['mean'] for g in groups]))
    baseline = mean(full['prepared']['summary'])
    new, control = mean(full['development']['summary']), mean(route['development']['summary'])
    novel_gain = float(np.mean([novelty['mean_selected_new_gain_over_old_casewise_ceiling'][g] for g in groups]))
    accepted = sum(r['status'] == 'accepted' for r in full['rounds'])
    inherited = sum(full['rounds'][i-1]['status'] == 'accepted' for i in range(1, len(full['rounds'])))
    retained = all(v['mean'] >= full['prepared']['summary'][g]['mean'] - tol
                   for g, v in full['development']['summary'].items())
    old_stress = full['stress_audit']['frozen']['receipt'].get('summary', {})
    new_stress = full['stress_audit']['selected']['receipt'].get('summary', {})
    relative = bool(old_stress and new_stress.keys() == old_stress.keys()
        and all(new_stress[g]['mean'] >= old_stress[g]['mean'] - tol for g in old_stress))
    absolute = bool(new_stress and set(new_stress) == {'stress-' + g for g in full['development']['summary']}
        and all(new_stress['stress-' + g]['mean'] >= v['mean'] - tol for g, v in full['development']['summary'].items()))
    passed = (new-baseline >= requirement['minimum_full_over_frozen_nontraining_source_mean_gain']
        and new-control >= requirement['minimum_full_over_route_only_nontraining_source_mean_gain']
        and novel_gain >= requirement['minimum_positive_selected_new_capacity_gain_over_old_library_ceiling']
        and accepted >= requirement['minimum_accepted_updates_per_successful_full_trajectory']
        and inherited >= 1 and retained and relative and absolute)
    return {'seed': full['seed'], 'frozen_mean': baseline, 'route_only_mean': control, 'full_mean': new,
        'full_over_frozen_gain': new-baseline, 'full_over_route_only_gain': new-control,
        'selected_new_capacity_gain': novel_gain, 'accepted_updates': accepted, 'inherited_updates': inherited,
        'original_sources_retained': retained, 'stress_vs_frozen_retained': relative,
        'stress_vs_own_original_retained': absolute, 'passed': passed}


def verify(directory, results):
    config = read(directory / 'protocol.json')
    proof = {'protected': verify_snapshot(ROOT, read(directory / 'protected_before.json')),
        'independent_proposals': 0, 'recomputed_system_rows': 0, 'recomputed_capability_rows': 0,
        'verified_analysis_replay_rows': 0, 'invalid_attempts_counted': 0, 'paired_initial_inputs': 0}
    paired_keys = ('current_system', 'active_branches', 'capability_library', 'current_router_code',
        'current_summary', 'frozen_best_complete_summary', 'bank_gap_diagnosis', 'frozen_library_gap',
        'observable_feature_definitions', 'anonymous_observations', 'round', 'remaining_rounds', 'training_source')
    for seed in SEEDS:
        contexts = [read(directory / f'{arm}-s{seed}/requests/r01/request.json')['context'] for arm in ARMS]
        if any(contexts[0][k] != contexts[1][k] for k in paired_keys):
            raise ValueError('Unequal initial scientific inputs')
        proof['paired_initial_inputs'] += 1
    origins = set()
    for record in results['trajectories']:
        _, _, episode, archive, runner = load(directory, record['arm'], record['seed'])
        try:
            bank = ResearchBank(runner)
            if bank.checkpoint() != record['checkpoint'] or archive.verify() != record['archive_tail']:
                raise ValueError('Final system/bank/memory state changed')
            seals = archive.events('search_sealed')
            if len(seals) != 1 or seals[0]['body'] != record['checkpoint']:
                raise ValueError('Search was not sealed at the reported checkpoint')
            events = [e['body'] for e in archive.events(ROUND_EVENT)]
            if events != record['rounds'] or len(events) != config['rounds_per_arm']:
                raise ValueError('Missing or altered research attempts')
            parent = record['prepared']['checkpoint']
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
                    raise ValueError('Changed response, request or inherited state')
                if response['origin'] in origins or response['origin'] != row['origin']:
                    raise ValueError('Expected independent proposer origins')
                origins.add(response['origin'])
                error = None
                try:
                    parse_action(json.loads(response['raw_text']), record['arm'], request['context']['current_system'],
                                 request['context']['capability_library'])
                except (ValueError, TypeError, KeyError, AttributeError, SyntaxError) as failure:
                    error = f'{type(failure).__name__}: {failure}'
                if row['status'] == 'invalid_proposal':
                    if error != row['error'] or not error or row['actual_execution'] or row['candidate'] is not None:
                        raise ValueError('Invalid proposal handling cannot be reproduced')
                    if row['checkpoint'] != parent or not row['counted_against_proposal_budget']:
                        raise ValueError('Invalid proposal changed state or escaped budget')
                    proof['invalid_attempts_counted'] += 1
                elif error is not None:
                    raise ValueError('Invalid model proposal was executed')
                if row.get('candidate'):
                    _, child = runner.system(row['candidate'])
                    _, old = runner.system(row['parent_checkpoint']['system']['executable_system'])
                    if any(child['library'].get(n) != b for n, b in old['library'].items()):
                        raise ValueError('Old capability was overwritten')
                if row.get('analysis_replay'):
                    replay = read(runner.checked_ref(row['analysis_replay']))
                    cap_path = runner.checked_ref(replay['binding'])
                    cap = read(cap_path)
                    original = read(cap_path.parent / 'development/predictions.json')
                    if replay['checkpoint'] != cap['checkpoint'] or not replay['exact_reverse_replay']:
                        raise ValueError('Analysis replay used a different state')
                    if set(replay['predictions']) != set(original['predictions']):
                        raise ValueError('Analysis replay cohort changed')
                    for key, ref in replay['predictions'].items():
                        with np.load(runner.checked_ref(ref), allow_pickle=False) as a:
                            with np.load(runner.checked_ref(original['predictions'][key]), allow_pickle=False) as b:
                                if any(not np.array_equal(a[k], b[k]) for k in ('score', 'support')):
                                    raise ValueError('Analysis replay is not exact')
                        proof['verified_analysis_replay_rows'] += 1
                parent = row['checkpoint']
            if parent != record['checkpoint']:
                raise ValueError('Last attempt did not lead to the final research state')
            samples = {s.key: s for s in runner.samples}
            def check_rows(receipt, counter):
                for row in receipt.get('rows', []):
                    with np.load(runner.checked_ref(receipt['predictions'][row['case']]), allow_pickle=False) as arrays:
                        actual = evaluate(samples[row['case']].validation_target(), dict(arrays), archive.protocol)
                    if any(row.get(k) != v for k, v in actual.items()):
                        raise ValueError('Predictions do not reproduce reported metric')
                    proof[counter] += 1
            for trial in archive.trials(visible_only=False):
                if trial['split'] == 'audit':
                    event = next(e for e in archive.events('trial_recorded') if e['body']['trial'] == trial['id'])
                    if event['seq'] <= seals[0]['seq'] or trial['receipt'].get('router_training'):
                        raise ValueError('Stress evaluation preceded seal or refitted the router')
                check_rows(trial['receipt'], 'recomputed_system_rows')
            for entry in bank.entries().values():
                check_rows(read(runner.checked_ref(entry['feedback'])), 'recomputed_capability_rows')
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
        raise ValueError('Preflight tests did not pass')
    return proof


def report(directory):
    directory = inside(ROOT, directory)
    results, config = read(directory / 'results.json'), read(directory / 'protocol.json')
    required = read(directory / 'completion_requirements.json')['required_evidence']
    proof = verify(directory, results)
    lookup = {(r['arm'], r['seed']): r for r in results['trajectories']}
    criteria = []
    for seed in SEEDS:
        full, route = lookup['full', seed], lookup['route', seed]
        if full['prepared']['summary'] != route['prepared']['summary']:
            raise ValueError('Paired baselines differ')
        _, _, _, archive, runner = load(directory, 'full', seed)
        try:
            library = full['prepared']['baseline_library']
            rows = {n: read(runner.checked_ref(b).parent / 'development/predictions.json')['rows'] for n, b in library.items()}
            novelty = selected_novel_gain(full['development']['rows'], rows, set(library))
        finally:
            archive.close()
        criteria.append(paired_decision(full, route, config, required, novelty))
    summary = {'verification': proof, 'paired_criteria': criteria,
        'batch_requirements_satisfied': sum(c['passed'] for c in criteria) >= required['minimum_successful_paired_trajectories'],
        'results_sha256': file_hash(directory / 'results.json'), 'report_source_sha256': file_hash(__file__),
        'production_replaced': False, 'new_blind_test': False,
        'metric': 'known-positive coverage at fixed 10% entire-image area, NOT accuracy or IoU',
        'stress': 'Previously observed transformed evidence; regression check, not fresh generalization'}
    write_new(directory / 'summary.json', summary)
    rows = [[c['seed'], f'{100*c["frozen_mean"]:.2f}%', f'{100*c["route_only_mean"]:.2f}%',
        f'{100*c["full_mean"]:.2f}%', f'{100*c["selected_new_capacity_gain"]:+.3f} pp',
        c['accepted_updates'], c['inherited_updates'], c['stress_vs_own_original_retained'], c['passed']] for c in criteria]
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSI 研究能力积累与整体保留</title>
<style>body{max-width:1300px;margin:30px auto;padding:20px;font:17px Arial,"Noto Sans CJK SC",sans-serif;line-height:1.7;background:#f7faf9;color:#243e42}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #ccd9d6;padding:9px}th{background:#dcece6}pre{white-space:pre-wrap}</style>
<h1>PhyEvo-Mic：新能力创造与可复用研究资产</h1>
<p>自主诊断 → 选择分析／学习／采用操作 → 隔离执行 → 固定预算整体评价 → 准入或保留研究资产 → 下一轮实际继承。</p>
<p>两组从上一批最佳完整系统出发，共享所有已测能力。旧能力复用不计为本轮新发现；未准入程序和权重可以继续研究，但不会自动上线。全部数据为历史开发来源，旋转检查不是新独立数据集。</p>'''
    page += table(['种子', '冻结系统', '只改采用', '完整 RSI', '新能力超过旧库的贡献', '准入次数', '下一轮继承', '旋转保持自身原分数', '通过'], rows)
    page += '<h2>全部研究尝试</h2><ul>'
    for trajectory in results['trajectories']:
        for row in trajectory['rounds']:
            path = f'{trajectory["arm"]}-s{trajectory["seed"]}/requests/r{row["round"]:02d}'
            page += f'<li>{html.escape(path)}：{html.escape(row["status"])} · <a href="{path}/request.json">输入</a> · <a href="{path}/raw_response.json">原始提案</a> · <a href="{path}/result.json">实测</a></li>'
    page += '</ul><h2>逐项核验</h2><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></html>'
    with (directory / 'report.html').open('x') as stream:
        stream.write(page)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory'); args = parser.parse_args()
    report(args.directory)
