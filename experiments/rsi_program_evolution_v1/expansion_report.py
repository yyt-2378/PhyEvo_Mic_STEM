"""Evidence-backed paired capability-expansion results; no experiment mutation."""
import argparse
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .contracts import file_hash, inside, verify_snapshot, write_new
from .evaluation import evaluate
from .expansion_state import selected_novel_gain
from .expansion_transport import validation_error
from .expansion_study import ARMS, SEEDS, PRIOR, checkpoint, implementations, load, program_identity
from .frontier_report import NAMES, table
from .real_study import ROOT
from .system_state import read


def mean(summary, groups):
    return float(np.mean([summary[g]['mean'] for g in groups]))


def verify(directory, results):
    config = read(directory / 'protocol.json')
    if config['implementation_sha256'] != implementations():
        raise ValueError('Frozen implementation changed')
    proof = {'protected': verify_snapshot(ROOT, read(directory / 'protected_before.json')),
             'independent_proposals': 0, 'verified_arrays': 0, 'recomputed_predictions': 0,
             'accepted_states_used_in_next_request': 0, 'initial_scientific_inputs_paired': 0,
             'invalid_proposals_counted': 0}
    paired_keys = ('current_system', 'current_router_code', 'active_branches', 'capability_library',
        'current_summary', 'frozen_complete_baseline', 'gap_diagnosis', 'frozen_library_gap',
        'observable_and_error_evidence', 'round', 'remaining_rounds', 'training_source')
    for seed in SEEDS:
        inputs = [read(directory / f'{arm}-s{seed}/requests/r01/request.json')['context'] for arm in ARMS]
        if any(inputs[0][key] != inputs[1][key] for key in paired_keys):
            raise ValueError('Paired research arms received unequal initial scientific evidence')
        proof['initial_scientific_inputs_paired'] += 1
    origins = set()
    for record in results['trajectories']:
        _, _, episode, archive, runner = load(directory, record['arm'], record['seed'])
        try:
            if archive.verify() != record['archive_tail'] or checkpoint(archive, runner) != record['checkpoint']:
                raise ValueError('Final archive/state binding changed')
            seals = archive.events('search_sealed')
            if len(seals) != 1 or seals[0]['body'] != record['checkpoint']:
                raise ValueError('Unsealed search')
            if len(record['rounds']) != config['rounds_per_arm']:
                raise ValueError('Missing research attempts')
            parent = record['prepared']['checkpoint']
            for index, row in enumerate(record['rounds'], 1):
                folder = episode / 'requests' / f'r{index:02d}'
                request = read(folder / 'request.json'); response = read(folder / 'response.json')
                if (folder / 'raw_response.json').read_text() != response['raw_text']:
                    raise ValueError('Original model response changed')
                parsed, proposal_error = validation_error(response['raw_text'], request['context'])
                if row['status'] == 'invalid_proposal':
                    if (proposal_error != row['error'] or proposal_error is None
                            or row['candidate'] is not None or row['actual_execution'] is not False
                            or row['checkpoint'] != row['parent_checkpoint']
                            or not row['counted_against_proposal_budget']
                            or row['transport_source_sha256'] != file_hash(Path(__file__).with_name('expansion_transport.py'))):
                        raise ValueError('Invalid proposal rejection is not reproducible')
                    proof['invalid_proposals_counted'] += 1
                elif proposal_error is not None:
                    raise ValueError('Invalid proposal was scientifically executed')
                if (row != read(folder / 'result.json') or row['parent_checkpoint'] != parent
                        or request['context']['checkpoint'] != parent
                        or response['request_sha256'] != file_hash(folder / 'request.json')
                        or row['response_sha256'] != file_hash(folder / 'response.json')
                        or parsed != response['proposal']
                        or file_hash(folder / 'response.json') != read(folder / 'response.ready.json')['response_sha256']):
                    raise ValueError('Changed proposal or inherited experience')
                if response['origin'] in origins:
                    raise ValueError('Expected independent proposer sessions')
                origins.add(response['origin']); parent = row['checkpoint']
                if index > 1 and record['rounds'][index-2]['status'] == 'accepted':
                    proof['accepted_states_used_in_next_request'] += 1
                if row.get('candidate'):
                    _, child = runner.system(row['candidate'])
                    _, old = runner.system(row['parent_checkpoint']['executable_system'])
                    if not all(child['library'].get(k) == v for k, v in old['library'].items()):
                        raise ValueError('Old capability binding was replaced')
            if parent != record['checkpoint']:
                raise ValueError('Final experience state mismatch')
            samples = {s.key: s for s in runner.samples}
            for trial in archive.trials(visible_only=False):
                receipt = trial['receipt']
                if trial['split'] == 'audit':
                    event = next(e for e in archive.events('trial_recorded') if e['body']['trial'] == trial['id'])
                    if event['seq'] <= seals[0]['seq'] or receipt.get('router_training'):
                        raise ValueError('Stress was exposed before seal or used for training')
                for row in receipt.get('rows', []):
                    path = runner.checked_ref(receipt['predictions'][row['case']])
                    with np.load(path, allow_pickle=False) as arrays:
                        measured = evaluate(samples[row['case']].validation_target(), dict(arrays), archive.protocol)
                    if any(row.get(k) != v for k, v in measured.items()):
                        raise ValueError('Saved prediction does not reproduce score')
                    proof['verified_arrays'] += 1; proof['recomputed_predictions'] += 1
            for candidate in archive.pool():
                folder, spec = runner.system(candidate['id'])
                if (folder / 'system_state.json').exists():
                    runner.frozen_state(candidate['id'])
                for binding in spec['library'].values():
                    cap = read(runner.checked_ref(binding))
                    if set(cap['weight_training_groups']) & set(archive.protocol.validation_groups + archive.protocol.audit_groups):
                        raise ValueError('New branch checkpoint fitted to validation/stress source')
        finally:
            archive.close()
    proof['independent_proposals'] = len(origins)
    suites = ET.parse(directory / 'tests.xml').getroot().findall('.//testsuite')
    proof['tests'] = {k: sum(int(s.get(k, 0)) for s in suites) for k in ('tests', 'failures', 'errors', 'skipped')}
    if not proof['tests']['tests'] or proof['tests']['failures'] or proof['tests']['errors']:
        raise ValueError('Related tests did not pass')
    if proof['invalid_proposals_counted']:
        suites = ET.parse(directory / 'transport_tests.xml').getroot().findall('.//testsuite')
        proof['transport_tests'] = {k: sum(int(s.get(k, 0)) for s in suites)
            for k in ('tests', 'failures', 'errors', 'skipped')}
        if not proof['transport_tests']['tests'] or proof['transport_tests']['failures'] or proof['transport_tests']['errors']:
            raise ValueError('Proposal rejection tests did not pass')
        proof['transport_policy_sha256'] = file_hash(directory / 'transport_rejection_policy.json')
    return proof


def report(directory):
    directory = inside(ROOT, directory)
    result = read(directory / 'results.json'); config = read(directory / 'protocol.json')
    requirement = read(directory / 'completion_requirements.json')['required_evidence']
    proof = verify(directory, result)
    lookup = {(t['arm'], t['seed']): t for t in result['trajectories']}
    paired = []; retention = []; rounds = []; criteria = []; links = []
    groups = config['evaluation']['validation_groups']
    tolerance = config['evaluation']['maximum_group_regression']
    for seed in SEEDS:
        full, route = lookup['full', seed], lookup['route', seed]
        if full['prepared']['summary'] != route['prepared']['summary']:
            # Seeds are identical inside a pair; summaries include their seed key.
            raise ValueError('Paired arms did not start at the same complete baseline')
        baseline = mean(full['prepared']['summary'], groups)
        new = mean(full['development']['summary'], groups)
        control = mean(route['development']['summary'], groups)
        _, _, _, archive, runner = load(directory, 'full', seed)
        try:
            frozen_library = full['prepared']['baseline_library']
            frozen_rows = {n: read(runner.checked_ref(b).parent / 'development/predictions.json')['rows'] for n, b in frozen_library.items()}
            state = runner.frozen_state(full['checkpoint']['executable_system'])
            known = set(frozen_library)
            for name, binding in state['library'].items():
                cap = read(runner.checked_ref(binding))
                if program_identity(runner.checked_ref(cap['program']).read_text()) in config['known_program_identities']:
                    known.add(name)
            novelty = selected_novel_gain(full['development']['rows'], frozen_rows, known)
        finally:
            archive.close()
        novel_gain = float(np.mean([novelty['mean_selected_new_gain_over_old_casewise_ceiling'][g] for g in groups]))
        accepted = [r for r in full['rounds'] if r['status'] == 'accepted']
        inherited = sum(full['rounds'][i-1]['status'] == 'accepted' for i in range(1, len(full['rounds'])))
        retained = all(v['mean'] >= full['prepared']['summary'][g]['mean'] - tolerance
                       for g, v in full['development']['summary'].items())
        stress_old = full['stress_audit']['frozen']['receipt'].get('summary', {})
        stress_new = full['stress_audit']['selected']['receipt'].get('summary', {})
        stress_pass = bool(stress_old and stress_old.keys() == stress_new.keys()
            and all(stress_new[g]['mean'] >= stress_old[g]['mean'] - tolerance for g in stress_old))
        success = (new-baseline >= requirement['minimum_full_over_frozen_nontraining_source_mean_gain']
            and new-control >= requirement['minimum_full_over_route_only_nontraining_source_mean_gain']
            and novel_gain >= requirement['minimum_positive_selected_new_capacity_gain_over_old_library_ceiling']
            and len(accepted) >= requirement['minimum_accepted_updates_per_successful_full_trajectory']
            and inherited >= 1 and retained and stress_pass)
        criteria.append({'seed': seed, 'full_over_frozen_gain': new-baseline,
            'full_over_route_only_gain': new-control, 'selected_new_capacity_gain': novel_gain,
            'accepted_updates': len(accepted), 'inherited_updates': inherited,
            'original_sources_retained': retained, 'rotation_stress_retained': stress_pass, 'passed': success})
        paired.append([seed, f'{100*baseline:.2f}%', f'{100*control:.2f}%', f'{100*new:.2f}%',
            f'{100*(new-control):+.2f} pp', f'{100*novel_gain:+.2f} pp', success])
        for g, v in full['development']['summary'].items():
            retention.append([seed, NAMES[g], '训练来源回放' if g in config['evaluation']['train_groups'] else '未参与本批拟合的开发来源',
                f'{100*full["prepared"]["summary"][g]["mean"]:.2f}%', f'{100*v["mean"]:.2f}%'])
    for record in result['trajectories']:
        key = f'{record["arm"]}-s{record["seed"]}'
        for row in record['rounds']:
            rounds.append([key, row['round'], row['research_decision']['diagnosis'],
                row['research_decision']['operation_target'], row['operation'], row['status'], row.get('exact_replay', False)])
            folder = f'{key}/requests/r{row["round"]:02d}'
            links.append(f'<p>{key} · {row["round"]}：<a href="{folder}/request.json">可见证据</a> · '
                f'<a href="{folder}/response.json">独立模型原始提案</a> · <a href="{folder}/result.json">实际结果</a></p>')
    successes = sum(r['passed'] for r in criteria)
    summary = {'verification': proof, 'paired_criteria': criteria,
        'batch_requirements_satisfied': successes >= requirement['minimum_successful_paired_trajectories'],
        'results_sha256': file_hash(directory / 'results.json'),
        'completion_requirements_sha256': file_hash(directory / 'completion_requirements.json'),
        'report_source_sha256': file_hash(Path(__file__)), 'production_replaced': False,
        'metric': 'known-positive coverage at fixed 10% whole-image candidate budget; not accuracy/IoU',
        'historical_cohort': True, 'stress_is_not_independent_acquisition': True}
    write_new(directory / 'summary.json', summary)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSI 能力扩展与持续保留</title>
<style>body{max-width:1300px;margin:30px auto;padding:0 24px;font:17px Arial,"Noto Sans CJK SC",sans-serif;line-height:1.7;color:#223f47;background:#f6f8f8}h1,h2{color:#17626a}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}th,td{padding:9px;border:1px solid #cad7d8;text-align:left}th{background:#e0efed}.note{background:#fff3d6;padding:15px}pre{white-space:pre-wrap}</style>
<h1>PhyEvo-Mic：自主能力扩展，而非仅优化调用</h1>
<p>能力缺口诊断 → 自主选择研究操作 → 新能力与采用条件 → 完整系统准入 → 状态和经验继承。</p>
<p class="note">已有公开来源全部用于本批开发／回放，专家不需重新标注。新增权重只在训练来源拟合。最终旋转压力测试不参与研究，但不是新的材料或独立采集泛化测试。所有百分数为固定候选面积下的已标正例覆盖率。</p>'''
    page += f'<p>从已完成审计的最佳完整系统出发，两个研究组共享全部 11 种已有程序。{successes}/3 对达到预先声明的本批能力扩展、效果提升与连续保留要求。</p>'
    page += '<h2>1. 同起点、同研究轮数的配对结果</h2>' + table(['种子', '冻结系统', '仅改调用', '完整 RSI', '完整−仅调用', '新能力超过旧库逐例上限的贡献', '本批要求'], paired)
    page += '<p>上表平均仅包括没有参与本批权重拟合的开发来源，按来源等权。旧库逐例上限仅用于离线归因，绝不是运行时看答案的选择程序。新能力贡献可以为负；不会只保留正值。</p>'
    page += '<h2>2. 完整 RSI 的逐来源保留</h2>' + table(['种子', '来源', '用途', '冻结系统', '最终系统'], retention)
    page += '<h2>3. 全部研究尝试</h2>' + table(['轨迹', '轮次', '诊断', '研究目标', '操作', '准入', '冻结重放一致'], rounds)
    page += '<h2>4. 决策与执行证据</h2>' + ''.join(links)
    page += '<h2>5. 协议与核验</h2><p><a href="protocol.json">冻结协议</a> · <a href="completion_requirements.json">预先声明的完成要求</a> · <a href="results.json">全部结果</a> · <a href="summary.json">逐项核验</a> · <a href="tests.xml">测试</a></p>'
    page += '<p>未替换生产版本，未自动启动长期服务。允许学生训练并不等于实际发生了学生训练，具体操作以原始提案和优化日志为准。</p><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre></html>'
    with (directory / 'report.html').open('x') as stream:
        stream.write(page)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('directory'); args = parser.parse_args()
    report(args.directory)
