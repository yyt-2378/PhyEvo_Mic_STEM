"""Evidence-linked report for the additive autonomous-direction GPU study."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from .active_diagnostics import field_enrichment
from .active_study import EPISODES, PREVIOUS, load, code_hashes
from .contracts import digest, file_hash, inside, verify_snapshot, write_new
from .frontier_report import NAMES, table
from .real_study import ROOT, read


def verify(directory,results):
    config=read(directory/'study.json'); library=read(directory/'library.json')
    if config['implementation_sha256']!=code_hashes():raise ValueError('Frozen code changed')
    if file_hash(PREVIOUS/'library.json')!=library['prior_library_sha256']:raise ValueError('Prior library changed')
    for name,expected in library['prior_archives'].items():
        if file_hash(inside(ROOT,name))!=expected:raise ValueError('Prior archive changed')
    protection=verify_snapshot(ROOT,read(directory/'preflight_gpu3/protected_before.json'))
    checked=set();origins=set();records=[];inherited_updates=0
    for item in results['trajectories']:
        key=item['trajectory'];_,_,episode,a,e=load(directory,key)
        try:
            if a.verify()!=item['archive_tail']:raise ValueError('Finished event history changed')
            events=a.events(); rounds=a.events('round_finished');sealed=a.events('search_sealed')
            if len(sealed)!=1 or [r['body'] for r in rounds]!=item['rounds']:raise ValueError('Incomplete result history')
            previous=item['prepared']['frozen_best']['candidate']
            for index,event in enumerate(rounds,1):
                folder=episode/'requests'/f'r{index:02d}'
                request=read(folder/'request.json');reservation=read(folder/'reservation.json')
                response=read(folder/'response.json');marker=read(folder/'response.ready.json')
                if (file_hash(folder/'request.json')!=response['request_sha256']
                    or file_hash(folder/'response.json')!=marker['response_sha256']
                    or json.loads(response['raw_text'])!=response['proposal']
                    or reservation['incumbent']!=previous
                    or request['context']['incumbent']!=previous
                    or read(folder/'result.json')!=event['body']):raise ValueError('Request/response/state binding changed')
                if response['origin'] in origins:raise ValueError('Proposer contexts not independent')
                origins.add(response['origin'])
                text=(folder/'request.json').read_text()
                if any(g in text for g in a.protocol.train_groups+a.protocol.validation_groups+a.protocol.audit_groups):
                    raise ValueError('Named source in public request')
                if index>1 and rounds[index-2]['body']['status']=='accepted':inherited_updates+=1
                previous=event['body']['incumbent']
            if previous!=sealed[0]['body']['incumbent']:raise ValueError('Seal changed selected model')
            trials=a.trials(visible_only=False)
            for entry in a.pool():a.candidate(entry['id'])
            for t in trials:
                if digest({k:t[k] for k in ('candidate','split','outcome','receipt')})!=t['id']:raise ValueError('Trial changed')
                receipt=t['receipt']
                if t['split']=='audit':
                    event=next(v for v in events if v['kind']=='trial_recorded' and v['body']['trial']==t['id'])
                    if event['seq']<=sealed[0]['seq'] or receipt.get('requested_training_steps_per_seed',0):
                        raise ValueError('Audit before sealing or audit retraining')
                assets={**receipt.get('artifacts',{}),**{v['path']:v['sha256'] for v in receipt.get('checkpoints',{}).values()}}
                for name,expected in assets.items():
                    path=inside(a.root,name)
                    if str(path) not in checked:
                        if file_hash(path)!=expected:raise ValueError('Prediction/checkpoint changed')
                        checked.add(str(path))
            selected=a.candidate(previous)
            selected_trial=next(t for t in reversed(trials) if t['candidate']==previous
                and t['split']=='validation' and t['outcome']=='ok')
            checkpoints={seed:{'path':f"{key}/archive/{v['path']}",'sha256':v['sha256']}
                for seed,v in selected_trial['receipt'].get('checkpoints',{}).items()}
            diagnostics={}
            for split in ('validation','audit'):
                cohort=[s for s in e.samples if a.protocol.split(s.group)==split]
                diagnostics[split]={'planes':len(cohort),'positive_field_percentiles':{
                    field:np.mean([field_enrichment(s.fields[field],s.human,s.fields['valid'])[
                        'positive_percentile_q10_q50_q90'] for s in cohort],axis=0).tolist()
                    for field in ('variation','physics','student','base')}}
            records.append({'trajectory':key,'trials':len(trials),'proposals':len(rounds),
                'source_exclusion':item['prepared']['source_audit']['withheld_sources_absent_from_both_weight_training_stages'],
                'retained_candidate':previous,'retained_pipeline':f'{key}/archive/programs/{previous}/pipeline.py',
                'retained_pipeline_sha256':selected['code_hash'],'retained_checkpoints':checkpoints,
                'post_seal_diagnostics_not_used_for_selection':diagnostics})
        finally:a.close()
    suites=ET.parse(directory/'tests_final.xml').getroot().findall('.//testsuite')
    test_counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    if not test_counts['tests'] or test_counts['failures'] or test_counts['errors']:
        raise ValueError('Focused validation suite did not pass')
    source_hashes={name:file_hash(Path(__file__).with_name(name)) for name in
                  ('active_study.py','active_diagnostics.py','active_report.py','runtime_compat.py','test_active_study.py')}
    compatibility=None
    if (directory/'compatibility_replay/result.json').is_file():
        replay=read(directory/'compatibility_replay/result.json')
        provenance=read(directory/'compatibility_replay/provenance.json')
        if (replay['source_pipeline_sha256']!=replay['replayed_pipeline_sha256']
            or replay['included_in_original_rsi_results'] or replay['selected_for_production']
            or provenance['study_results_sha256']!=file_hash(directory/'results.json')):
            raise ValueError('Infrastructure replay changed scientific attribution')
        compatibility={'status':replay['status'],'unchanged_pipeline':True,
            'excluded_from_rsi_results':True,'result_sha256':file_hash(directory/'compatibility_replay/result.json')}
    return {'protected':protection,'focused_tests':test_counts,'implementation_sha256':source_hashes,
            'post_seal_compatibility_replay':compatibility,
            'independent_proposer_sessions':len(origins),
            'verified_prediction_checkpoint_files':len(checked),'trajectories':records,
            'accepted_updates_used_in_a_later_request':inherited_updates}


def report(directory):
    directory=inside(ROOT,directory); results=read(directory/'results.json');proof=verify(directory,results)
    comparisons=[];rounds=[];transfers=[];details=[];folds=[];learners=[];counts=Counter();positive_transfers=0
    frontier_trajectories=0; transfer_gain_ge_development_margin=0
    for t in results['trajectories']:
        key=t['trajectory'];state=t['prepared'];name=NAMES[state['source_audit']['validation_group']]
        sources=state['source_audit']
        folds.append([key,'、'.join(NAMES.get(g,g) for g in sources['new_fit_groups']),
            len(sources['new_fit_cases']),name,NAMES[sources['audit_group']]])
        accepted=[r for r in t['rounds'] if r['status']=='accepted']
        selected=accepted[-1]['coverage'] if accepted else state['frozen_best']['mean']
        frontier_trajectories+=selected>state['frozen_best']['mean']
        comparisons.append([name,f"{100*state['raw_start_mean']:.2f}%",f"{100*state['historical_best']['mean']:.2f}%",
            f"{100*state['frozen_best']['mean']:.2f}%",f'{100*selected:.2f}%',
            f"{100*(selected-state['frozen_best']['mean']):+.2f} pp",len(accepted)])
        for row in t['rounds']:
            counts['proposals']+=1;counts[row.get('target','invalid')]+=1;counts[row['status']]+=1
            counts['trained_and_trace_verified']+=bool(row.get('actual_training'))
            counts['accepted_training']+=bool(row.get('actual_training')) and row['status']=='accepted'
            rounds.append([key,row['round'],row.get('action','—'),row.get('target','—'),row.get('training_steps',0),
                f"{100*row['coverage']:.2f}%" if 'coverage' in row else '执行未完成',
                f"{100*row['gains']['relative_to_all_pre_round_candidates']:+.2f} pp" if 'gains' in row else '—',row['status']])
            folder=f"{key}/requests/r{row['round']:02d}"
            details.append(f"<h3>{key} · {row['round']}</h3><p>"+html.escape(row.get('decision_reason',''))+'</p><p>'+
                html.escape(row.get('intervention',''))+f'</p><p><a href="{folder}/request.json">可见证据</a> · '
                f'<a href="{folder}/response.json">原始提案</a> · <a href="{folder}/result.json">实际结果</a></p>')
            if row.get('error'):details.append('<pre>'+html.escape(row['error'])+'</pre>')
            if row.get('actual_training'):
                details.append('<pre>'+html.escape(json.dumps(row['actual_training'],ensure_ascii=False,indent=2))+'</pre>')
                for seed,trace in row['actual_training'].items():
                    learners.append([key,row['round'],seed,trace['steps'],trace['parameters'],
                        f"{trace['first_loss']:.4f} → {trace['last_loss']:.4f}",
                        '逐值一致' if row.get('numeric_checkpoint_reuse_exact') else '未通过',
                        '接纳' if row['status']=='accepted' else '拒绝更新'])
        def score(role):
            v=t['transfer'][role]['summary']
            return next(iter(v.values()))['mean'] if v else None
        old,new,raw=score('frozen_best'),score('selected'),score('raw_start')
        if old is not None and new is not None:
            positive_transfers+=bool(accepted) and new>old
            transfer_gain_ge_development_margin+=bool(accepted) and new-old>=.002
            transfers.append([name+' → '+NAMES[state['source_audit']['audit_group']],f'{100*raw:.2f}%' if raw is not None else '—',f'{100*old:.2f}%',
                f'{100*new:.2f}%',f'{100*(new-old):+.2f} pp',
                '没有新更新' if not accepted else ('正向保留' if new>old else ('容限内未明显退化' if new>=old-.01 else '迁移退化'))])
        else:transfers.append([name,'—','—','—','—','迁移执行失败'])
    summary={'counts':dict(counts),'positive_transfer_trajectories':positive_transfers,'verification':proof,
        'new_frontier_trajectories':frontier_trajectories,
        'transfer_gain_at_least_0_2_percentage_points':transfer_gain_ge_development_margin,
        'results_sha256':file_hash(directory/'results.json'),'current_release_replaced':False,
        'metric':'known-positive coverage at fixed 10% total image area, not accuracy/IoU',
        'no_new_unseen_dataset_or_statistical_significance_claim':True}
    write_new(directory/'summary.json',summary)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSI 自主方向与 GPU 实测</title>
<style>body{max-width:1350px;margin:32px auto;padding:0 24px;background:#f6f8f8;color:#233943;font:17px Arial,"Noto Sans CJK SC",sans-serif;line-height:1.7}h1,h2{color:#17626a}table{border-collapse:collapse;width:100%;background:white;margin-bottom:24px}td,th{border:1px solid #d6e0e1;padding:9px;text-align:left}th{background:#deeeee}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px;background:#eee;padding:12px}.note{border-left:5px solid #b99742;background:#fff4d7;padding:16px}p{overflow-wrap:anywhere}</style>
<h1>PhyEvo-Mic：自主研究方向与新增益验证</h1><p>2026-10-17 · 独立 GPU 沙盒 · 原框架和专家标注保持不动</p>'''
    page+=f'<p>完成 {counts["proposals"]} 次独立提案，接受 {counts["accepted"]} 次任务更新；自主选择学习 {counts["learning"]} 次，实际训练与状态继承核验通过 {counts["trained_and_trace_verified"]} 次。</p>'
    page+='<div class="note">所有数值都是固定整图 10% 面积预算下的专家正例覆盖率，不是准确率或 IoU。未知区域不作负例。数据已有开发接触，当前为前瞻执行的来源隔离实验，不是新的外部盲测；不把多个平面当成多个独立采集来源。</div>'
    page+='<p>计算方式：每张图按预测得分排序，在有效支持域内最多选取整图像素数的 10%；覆盖率 = 选中区域与专家已标正例的交集面积 ÷ 专家已标正例面积，再按冻结协议汇总。因此，这个百分数不能与此前文章中的 IoU 直接比较。</p>'
    page+='<h2>这轮改变了什么</h2><p>去掉微调/交叉的轮次限制，让模型选择操作、父版本、分析/协同/训练方向；扩展匿名诊断到支持域遗漏、区域内排序错误、信号互补和训练来源差异；明确字典 checkpoint 接口；为模型选择的训练提供 CUDA 和较充分的单次运行空间。没有要求它必须训练。</p>'
    page+='<p>每来源至少六轮；连续四轮未准入后结束，最多十二轮作为单批实验设计。所有来源搜索完成后再封存迁移评价，不依迁移结果追加提案。单次 4,096 训练步上限、900 秒超时用于控制实验执行，不设本任务总 token 预算。</p>'
    page+='<h2>实验分工与数据隔离</h2>'+table(['轨迹','本轮可新增训练的来源','训练平面数','开发评价来源','封存后迁移来源'],folds)
    page+='<p>这里列的是本轮新增拟合数据，不是继承模型的全部训练集。继承权重可能还用过其他来源；记录已核验开发评价来源和迁移来源未进入两个权重训练阶段。本批使用种子 23，三个来源，不作统计显著性结论。</p>'
    page+='<h2>1. 新收益是否超过已知能力？</h2>'+table(['开发来源','原起点','重测历史库最佳','含诊断探测的起点','最终保留','真正新增','更新次数'],comparisons)
    page+='<p>十个预设单信号/方向探测也被算入既有能力。若改进只是切换到这些已测方案，不计作新发现；历史参照重训也不计作自主训练决策。“超过已知最好”仅指本次冻结且可执行的候选库，不等于超过论文中全部模型。</p>'
    page+='<h2>2. 完整提案与执行</h2>'+table(['轨迹','轮次','操作','方向','训练步数','覆盖率','超过轮前最好','决定'],rounds)
    page+='<h3>自主选择学习的实际执行</h3>'+table(['轨迹','轮次','种子','优化步数','训练参数数','首末训练损失','冻结权重复用','决定'],learners)
    page+='<p>这里是模型自主设计的轻量学习分支，不是大规模分割网络重训。已证明它能根据诊断选择训练、执行并保留有效权重；尚未通过“总是训练／从不训练”等策略对照证明选择时机最优。</p>'
    page+='<h2>3. 封存后的跨来源表现</h2>'+table(['迁移','目标来源原框架','更新前方案','最终版本','相对更新前变化','解读'],transfers)
    page+='<p>正向保留只表示同一研究轨迹的新方案相对其更新前方案，在迁移来源上仍有增益；不自动表示超过目标来源原框架或当地最佳方案。原框架列提供额外参照，没有用于事后改选。训练记录通过核验也不等于学到的残差获得了有效泛化。</p>'
    page+='<p>迁移只推理，不重训、不改选。若新版本退化，如实记录，不将开发集准入直接写成生产发布或普遍泛化。研究方向由独立模型会话产生，调度仍经过当前会话；没有建立长期后台服务。</p>'
    diagnostic_rows=[]
    for record in proof['trajectories']:
        for split,stats in record['post_seal_diagnostics_not_used_for_selection'].items():
            for field,percentiles in stats['positive_field_percentiles'].items():
                diagnostic_rows.append([record['trajectory'],'开发' if split=='validation' else '迁移',
                    field,*[f'{100*v:.2f}%' for v in percentiles]])
    page+='<h3>封存后检查：信号在新来源上是否仍然有效？</h3>'+table(
        ['轨迹','来源角色','输入信号','正例信号排名 Q10','排名中位数','排名 Q90'],diagnostic_rows)
    page+='<p>这是专家正例在整图该信号分布中的百分位，多个平面取各自统计的均值；不是检测得分。只用于封存后的原因分析，没有反馈给本批提案。高分信号在另一来源上可能不再指向缺陷，因而“开发场景改进”不自动等于“可以替代其他来源的原分支”。</p>'
    page+='<h2>4. 可复核的决策链</h2>'+''.join(details)
    page+='<h2>5. 文件与核验</h2><p><a href="results.json">全部结果</a> · <a href="study.json">冻结协议</a> · <a href="library.json">候选库</a> · <a href="summary.json">完整性核验</a> · <a href="tests_final.xml">相关测试</a></p>'
    for record in proof['trajectories']:
        page+=f'<p>{record["trajectory"]}：<a href="{record["retained_pipeline"]}">最终保留程序</a>'
        for seed,checkpoint in record['retained_checkpoints'].items():
            page+=f' · <a href="{checkpoint["path"]}">种子 {seed} 的冻结权重</a>'
        page+='。仅为本实验保留版本，未替换正式发布。</p>'
    page+='<p>本轮暴露的 skimage 间接导入 matplotlib 配置目录问题，另用 opt-in ConfiguredSandbox 适配器修复，并实测导入与隔离不变。适配器只设置 /work 内的配置位置，没有在冻结研究中替换 Sandbox，也没有覆盖第七轮失败记录或将人工环境修复算作模型发现。</p>'
    if proof['post_seal_compatibility_replay'] is not None:
        page+='<p>封存后原程序兼容性重放：<a href="compatibility_replay/result.json">执行结果</a> · <a href="compatibility_replay/provenance.json">程序与环境来源</a>。这项基础设施检查不参与上面的研究轮次、增益或版本选择。</p>'
    page+='<pre>'+html.escape(json.dumps(proof,ensure_ascii=False,indent=2))+'</pre></html>'
    with (directory/'report.html').open('x',encoding='utf-8') as f:f.write(page)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();report(args.directory)
