"""Read-only verification and an evidence-linked whole-system RSI handoff."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .contracts import digest, file_hash, inside, verify_snapshot, write_new
from .evaluation import evaluate
from .frontier_report import NAMES, table
from .real_study import ROOT
from .system_state import read
from .system_study import checkpoint, code_hashes, load


def verify(directory, results):
    protocol = read(directory / 'protocol.json')
    if protocol['implementation_sha256'] != code_hashes():
        raise ValueError('Frozen implementation changed')
    origins = set(); checked = set(); inherited = 0; old_bindings = 0; executed = 0; recomputed_audit_cases = 0
    protected = verify_snapshot(ROOT, read(directory / 'protected_before.json'))
    for row in results['trajectories']:
        key = row['trajectory']; _, _, episode, archive, runner = load(directory, key)
        try:
            if archive.verify() != row['archive_tail'] or checkpoint(archive, runner) != row['checkpoint']:
                raise ValueError('Whole-system history or checkpoint changed')
            seals = archive.events('search_sealed')
            if len(seals) != 1 or seals[0]['body'] != row['checkpoint']:
                raise ValueError('Missing immutable system seal')
            previous = row['prepared']['checkpoint']
            for index, round_result in enumerate(row['rounds'], 1):
                folder = episode / 'requests' / f'r{index:02d}'
                request = read(folder / 'request.json'); response = read(folder / 'response.json')
                if (request['context']['checkpoint'] != previous or round_result['parent_checkpoint'] != previous
                    or file_hash(folder / 'request.json') != response['request_sha256']
                    or file_hash(folder / 'response.json') != read(folder / 'response.ready.json')['response_sha256']
                    or json.loads(response['raw_text']) != response['proposal']
                    or read(folder / 'result.json') != round_result):
                    raise ValueError('Whole-system proposal/experience inheritance changed')
                if response['origin'] in origins:
                    raise ValueError('Proposers were not distinct fresh sessions')
                origins.add(response['origin'])
                if any(g in (folder / 'request.json').read_text() for g in archive.protocol.train_groups + archive.protocol.validation_groups + archive.protocol.audit_groups):
                    raise ValueError('Source identifiers in public model input')
                if index > 1 and row['rounds'][index-2]['status'] == 'accepted':
                    inherited += 1
                previous = round_result['checkpoint']
                candidate = round_result.get('candidate')
                if candidate:
                    _, child = runner.system(candidate)
                    _, parent = runner.system(response['proposal']['parent'])
                    if not all(child['library'].get(k) == v for k,v in parent['library'].items()):
                        raise ValueError('An older capability or its learned state was overwritten')
                    old_bindings += len(parent['library'])
            if previous != row['checkpoint']:
                raise ValueError('Final inherited state mismatch')
            for entry in archive.pool():
                folder, spec = runner.system(entry['id'])
                if (folder / 'system_state.json').exists():
                    runner.frozen_state(entry['id'])
                for binding in spec['library'].values():
                    cap_path = runner.checked_ref(binding); cap = read(cap_path)
                    if file_hash(cap_path.parent / 'definition.json') != cap['definition_sha256']:
                        raise ValueError('Capability training definition changed')
                    for ref in [cap['program']] + ([cap['checkpoint']] if cap['checkpoint'] else []):
                        checked.add(str(runner.checked_ref(ref)))
                    if set(cap['weight_training_groups']) & set(archive.protocol.audit_groups):
                        raise ValueError('A final-audit source entered weight training')
            for event in archive.events('capability_predictions_recorded'):
                body = event['body']; path = episode / 'capabilities' / body['capability'] / body['split'] / 'predictions.json'
                if file_hash(path) != body['sha256']:
                    raise ValueError('Capability feedback changed')
                if body['split'] == 'audit' and event['seq'] <= seals[0]['seq']:
                    raise ValueError('Audit feedback existed before sealing')
                for ref in read(path)['predictions'].values():
                    checked.add(str(runner.checked_ref(ref)))
            for trial in archive.trials(visible_only=False):
                if digest({k:trial[k] for k in ('candidate','split','outcome','receipt')}) != trial['id']:
                    raise ValueError('Trial binding changed')
                receipt = trial['receipt']; executed += 1
                for ref in receipt.get('predictions', {}).values():
                    checked.add(str(runner.checked_ref(ref)))
                if trial['split'] == 'audit':
                    if receipt.get('router_training'):
                        raise ValueError('Router was retrained during audit')
                    event = next(e for e in archive.events('trial_recorded') if e['body']['trial'] == trial['id'])
                    if event['seq'] <= seals[0]['seq']:
                        raise ValueError('Audit trial before seal')
                    samples = {sample.key: sample for sample in runner.samples}
                    for measured_row in receipt.get('rows', []):
                        sample = samples[measured_row['case']]
                        with np.load(runner.checked_ref(receipt['predictions'][sample.key]), allow_pickle=False) as arrays:
                            recomputed = evaluate(sample.validation_target(), dict(arrays), archive.protocol)
                        if any(measured_row.get(k) != value for k, value in recomputed.items()):
                            raise ValueError('Audit metric does not match the saved prediction')
                        recomputed_audit_cases += 1
        finally:
            archive.close()
    suites = ET.parse(directory / 'tests.xml').getroot().findall('.//testsuite')
    tests = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if not tests['tests'] or tests['failures'] or tests['errors']:
        raise ValueError('Focused tests failed')
    return {'protected':protected, 'tests':tests, 'independent_proposers':len(origins),
        'actual_system_trials':executed, 'verified_program_checkpoint_prediction_files':len(checked),
        'audit_case_metrics_recomputed_from_saved_predictions':recomputed_audit_cases,
        'older_capability_bindings_preserved_across_proposals':old_bindings,
        'accepted_systems_used_in_next_request':inherited,
        'executable_and_experience_inheritance_verified':True,
        'no_audit_retraining_or_reselection':True}


def report(directory):
    directory = inside(ROOT,directory); results = read(directory / 'results.json')
    proof = verify(directory,results)
    development=[]; library_comparison=[]; audit=[]; audit_results=[]; rounds=[]; details=[]; counts=Counter()
    for trajectory in results['trajectories']:
        key=trajectory['trajectory']; prepared=trajectory['prepared']; source=prepared['source_audit']
        known_means={}
        for name,binding in prepared['capability_library'].items():
            capability=inside(directory/key,binding['path'])
            summary=read(capability.parent/'development/predictions.json')['summary']
            known_means[name]=sum(v['mean'] for v in summary.values())/len(summary)
        best_known=max(known_means,key=known_means.get)
        selected=trajectory['selected_development']['summary']
        selected_mean=sum(v['mean'] for v in selected.values())/len(selected)
        library_comparison.append([key,best_known,f'{100*known_means[best_known]:.2f}%',
            f'{100*selected_mean:.2f}%',f'{100*(selected_mean-known_means[best_known]):+.2f} pp'])
        for group,old in prepared['baseline_summary'].items():
            new=trajectory['selected_development']['summary'][group]['mean']
            role='独立来源开发评价' if group==source['validation_group'] else '训练来源旧能力回放'
            development.append([key,NAMES.get(group,group),role,f"{100*old['mean']:.2f}%",f'{100*new:.2f}%',f"{100*(new-old['mean']):+.2f} pp"])
        old=trajectory['audit']['old_system']['receipt']; new=trajectory['audit']['selected_system']['receipt']
        if 'summary' in old and 'summary' in new:
            group=source['audit_group']; x=old['summary'][group]['mean']; y=new['summary'][group]['mean']
            audit.append([key,NAMES[group],f'{100*x:.2f}%',f'{100*y:.2f}%',f'{100*(y-x):+.2f} pp',str(new['choices'])])
            audit_results.append({'trajectory':key,'source':group,'old':x,'new':y,'delta':y-x,
                'retained_within_protocol_tolerance':y>=x-read(directory/'protocol.json')['maximum_group_regression']})
        else:
            audit.append([key,NAMES[source['audit_group']],'—','—','—','执行失败；保留原始记录'])
            audit_results.append({'trajectory':key,'source':source['audit_group'],'execution_failed':True,
                'retained_within_protocol_tolerance':False})
        for r in trajectory['rounds']:
            counts['proposals']+=1;counts[r['operation']]+=1;counts[r['status']]+=1
            proposal=read(directory/key/'requests'/f'r{r["round"]:02d}'/'response.json')['proposal']
            counts['router_training_proposals']+=int(proposal['router_training_steps']>0)
            counts['new_branch_training_proposals']+=int(bool(proposal['new_branch'] and proposal['new_branch']['training_steps']>0))
            values='；'.join(NAMES.get(g,g)+f" {100*v['mean']:.2f}%" for g,v in r.get('summary',{}).items())
            rounds.append([key,r['round'],r['operation'],values or '执行未完成',r['status'],r.get('exact_system_replay',False)])
            folder=f"{key}/requests/r{r['round']:02d}"
            details.append(f'<h3>{key} · {r["round"]}</h3><p>'+html.escape(r['hypothesis'])+'</p><p>'+html.escape(r['intervention'])+
                f'</p><p><a href="{folder}/request.json">可见开发证据</a> · <a href="{folder}/response.json">原始系统提案</a> · <a href="{folder}/result.json">实际系统评价</a></p>')
            if r.get('error'):details.append('<pre>'+html.escape(r['error'])+'</pre>')
        identity=trajectory['checkpoint']['executable_system']
        details.append(f'<p>{key} 最终系统：<a href="{key}/archive/programs/{identity}/system_spec.json">完整能力库</a> · '
            f'<a href="{key}/archive/programs/{identity}/system_state.json">权重状态绑定</a> · '
            f'<a href="{key}/archive/programs/{identity}/router.py">实际采用规则</a></p>')
    summary={'counts':dict(counts),'verification':proof,'results_sha256':file_hash(directory/'results.json'),
             'sealed_transfer_results':audit_results,
             'all_sealed_transfer_sources_retained':all(r['retained_within_protocol_tolerance'] for r in audit_results),
             'report_source_sha256':file_hash(Path(__file__)),'production_replaced':False,
             'metric':'known-positive area coverage at 10% whole-image budget; not accuracy/IoU',
             'historical_development_exposure':True,'new_blind_test':False}
    write_new(directory/'summary.json',summary)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>PhyEvo-Mic 整体框架 RSI</title>
<style>body{max-width:1380px;margin:32px auto;padding:0 24px;background:#f6f8f8;color:#233943;font:17px Arial,"Noto Sans CJK SC",sans-serif;line-height:1.7}h1,h2{color:#17626a}table{border-collapse:collapse;width:100%;background:white;margin:20px 0}td,th{border:1px solid #d6e0e1;padding:9px;text-align:left}th{background:#deeeee}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eee;padding:12px}.note{border-left:5px solid #b99742;background:#fff4d7;padding:16px}</style>
<h1>PhyEvo-Mic：完整系统的能力积累与自主采用</h1>
<p>2026-10-17 · 新增实验层 · 正式框架、既有模型、专家标注与原结果不改动</p>
<p>能力库＋分支权重＋采用规则＋研究经验 → 自主系统提案 → 隔离执行 → 多来源开发准入 → 整体继承或保留旧系统。</p>
<div class="note">指标为固定整图 10% 面积预算下的专家已标正例覆盖率，不是准确率或 IoU。三个来源已有历史开发接触；本轮检验新的系统集成与实际来源留出推理，不称为全新的盲测。</div>'''
    page+=f'<p>完成 {counts["proposals"]} 次独立系统提案，接纳 {counts["accepted"]} 次整体更新。采用规则由模型给出，不按材料名、文件名或测试答案拼接。</p>'
    page+='<h2>本轮实际改变</h2><p>此前优秀程序作为已知能力入库，训练型分支在每条轨迹自己的训练来源上重新拟合，不跨折搬运已见测试来源的权重。基线仍调用原框架 b0；其他分支入库不等于自动接管。研究模型可以只改采用规则，也可以新增或替换可选分析分支、训练新分支或选择程序。旧程序及权重均保留。</p>'
    page+='<p>选择程序只能返回一个有效分支；执行器将实际输出与该分支保存的数值逐值核对。每次成功候选还会倒序重放图像，验证冻结权重和采用决定不依赖调用顺序。未接纳候选的经验仍进入下一轮，但不替换执行版本。</p>'
    page+='<h2>1. 完整系统开发准入与旧能力回放</h2>'+table(['轨迹','来源','用途','旧系统','最终系统','变化'],development)
    page+='<p>开发目标按来源等权，不让帧多的来源掩盖单图来源。平均收益至少 0.2 个百分点；任何来源相对旧系统、当前系统及已接纳历史最好值下降不得超过 0.2 个百分点。训练来源回放检验能力保留，不能当作独立泛化证据。</p>'
    page+='<h3>与初始能力库的最强单分支比较</h3>'+table(['轨迹','开发集最强单分支','该分支双来源平均','新系统双来源平均','变化'],library_comparison)
    page+='<p>这里每条轨迹选择一个固定分支用于两个开发来源，不是逐图挑最高分。该比较区分“胜过原起点”与“胜过已有单分支能力”；路由组合的提升不自动等于发现了新检测算法，也不是最终留出成绩。</p>'
    page+='<h2>2. 全部封存后的来源迁移</h2>'+table(['轨迹','最终留出来源','旧系统','新系统自动选择','变化','实际分支调用次数'],audit)
    passed=sum(r['retained_within_protocol_tolerance'] for r in audit_results)
    page+=f'<p>按冻结的下降容限，{passed}/{len(audit_results)} 条轨迹保住最终留出来源。开发准入通过不自动等于未知来源保留通过；上表不因结果不理想而更换版本。</p>'
    page+='<p>上表使用已经封存的采用程序实际运行，不是各分支测试最高分的拼接。留出结果不参与本批提案、准入、重训或改选。以前单个 Al ORII 分支的迁移失败记录仍保留，本批另测完整系统。</p>'
    page+='<p>三条轨迹分别继承自己的完整系统和训练状态；这是同一进化机制的三次来源划分验证，不是把三个最终系统的最好数字拼成一个已部署版本。当前只有一个随机种子，来源内多帧也不能当作多个独立材料。</p>'
    page+='<h2>3. 全部迭代</h2>'+table(['轨迹','轮次','操作','完整系统开发成绩','决定','冻结状态重放一致'],rounds)
    page+='<h2>4. 决策、实现与状态</h2>'+''.join(details)
    page+='<h2>5. 核验材料</h2><p><a href="protocol.json">冻结协议</a> · <a href="library.json">初始能力来源</a> · <a href="results.json">完整结果</a> · <a href="summary.json">核验汇总</a> · <a href="tests.xml">相关测试</a></p>'
    page+='<p>这是受控实验，不自动发布为生产默认。模型提案仍由会话调度，没有启动长期服务。小规模来源留出结果不等于任意新材料上的保证；对既有能力的复用不计作重新发明算法。</p>'
    page+='<pre>'+html.escape(json.dumps(proof,ensure_ascii=False,indent=2))+'</pre></html>'
    with (directory/'report.html').open('x',encoding='utf-8') as stream:stream.write(page)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();report(args.directory)
