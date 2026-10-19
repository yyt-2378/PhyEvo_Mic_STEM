"""Small local HTML handoff for the prospective frontier pilot; no new metrics."""
import argparse
import html
import json
from pathlib import Path
from .archive import Archive
from .contracts import Protocol, digest, file_hash, inside, write_new, verify_snapshot
from .real_study import ROOT, read

NAMES={'sigma3-orientation-i':'Al ORI','sigma3-orientation-ii':'Al ORII',
       'guo_2021_zro2_twin_acquisition':'ZrO₂'}


def table(headers,rows):
    esc=lambda x:html.escape(str(x))
    return '<table><thead><tr>'+''.join('<th>'+esc(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+esc(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def verify_evidence(directory):
    """Read and bind the finished run; never reopen search or change decisions."""
    config=read(directory/'pilot_protocol.json')
    if (file_hash(Path(__file__).with_name('frontier_study.py'))!=config['study_runner_sha256']
            or file_hash(directory/'library.json')!=config['library_sha256']):
        raise ValueError('Frozen pilot runner or library changed')
    checked={}; origins=[];bound_artifacts=set(); inherited_updates=0;action_counts={}
    for key in config['trajectories']:
        p=read(directory/key/'protocol.json')
        protocol=Protocol(**{k:tuple(v) if k.endswith('_groups') or k=='seeds' else v for k,v in p.items()})
        archive=Archive(directory/key/'archive',protocol)
        try:
            events=archive.events(); rounds=archive.events('round_finished'); seals=archive.events('search_sealed')
            if len(rounds)!=3 or len(seals)!=1:
                raise ValueError('Incomplete or unsealed trajectory')
            previous=read(directory/key/'prepared.json')['frozen_best']['candidate']
            for index,event in enumerate(rounds,1):
                folder=directory/key/'requests'/f'r{index:02d}'
                request=read(folder/'request.json');reservation=read(folder/'reservation.json')
                response=read(folder/'response.json');marker=read(folder/'response.ready.json')
                if (reservation['incumbent']!=previous or request['context']['incumbent']!=previous
                        or file_hash(folder/'request.json')!=response['request_sha256']
                        or file_hash(folder/'response.json')!=marker['response_sha256']
                        or json.loads(response['raw_text'])!=response['proposal']
                        or read(folder/'result.json')!=event['body']):
                    raise ValueError('Round inheritance or raw proposal binding changed')
                request_text=(folder/'request.json').read_text()
                if any(g in request_text for g in protocol.train_groups+protocol.validation_groups+protocol.audit_groups):
                    raise ValueError('Named source exposed in proposer request')
                if index>1 and rounds[index-2]['body']['status']=='accepted':
                    inherited_updates+=1
                origins.append(response['origin']);previous=event['body']['incumbent']
                action=response['proposal']['action']
                action_counts[action]=action_counts.get(action,0)+1
            if seals[0]['body']['incumbent']!=previous:
                raise ValueError('Selected version differs from sealed version')
            for entry in archive.pool():
                archive.candidate(entry['id'])
            trials=archive.trials(visible_only=False)
            for trial in trials:
                if digest({k:trial[k] for k in ('candidate','split','outcome','receipt')})!=trial['id']:
                    raise ValueError('Trial identity changed')
                receipt=trial['receipt']
                if trial['split']=='audit':
                    recorded=next(e for e in events if e['kind']=='trial_recorded' and e['body']['trial']==trial['id'])
                    if recorded['seq']<=seals[0]['seq'] or receipt.get('requested_training_steps_per_seed',0)!=0:
                        raise ValueError('Audit before sealing or audit fitting detected')
                artifacts={**receipt.get('artifacts',{}),
                    **{v['path']:v['sha256'] for v in receipt.get('checkpoints',{}).values()}}
                for name,expected in artifacts.items():
                    path=inside(archive.root,name)
                    if str(path) not in bound_artifacts:
                        if file_hash(path)!=expected:raise ValueError('Prediction/checkpoint artifact changed')
                        bound_artifacts.add(str(path))
            checked[key]={'archive_tail':archive.verify(),'trials':len(trials),'rounds':3,
                          'audit_trials':sum(t['split']=='audit' for t in trials)}
        finally:archive.close()
    if len(origins)!=9 or len(set(origins))!=9:
        raise ValueError('Expected nine distinct independent proposer sessions')
    return {'archives':checked,'unique_proposer_sessions':len(set(origins)),
            'prediction_and_checkpoint_files_verified':len(bound_artifacts),
            'actual_research_actions':action_counts,
            'accepted_new_incumbents_used_in_a_later_request':inherited_updates,
            'raw_response_binding_and_anonymous_requests_verified':True}


def report(directory):
    directory=inside(ROOT,directory)
    source=read(directory/'results.json')
    comparisons=[];rounds=[];audits=[];directions=[];splits=[]
    total_new_frontier=total_train=accepted_train=retained_transfer=0
    for trajectory in source['trajectories']:
        key=trajectory['trajectory']; prepared=trajectory['prepared']
        name=NAMES[prepared['source_audit']['validation_group']]
        request=read(directory/key/'requests/r01/request.json')['context']
        splits.append([key,', '.join(NAMES.get(g,g) for g in prepared['source_audit']['new_fit_groups']),
            request['available_training']['samples'],name,
            NAMES[prepared['source_audit']['audit_group']]])
        successful=[r for r in trajectory['rounds'] if 'coverage' in r]
        accepted=[r for r in successful if r['status']=='accepted']
        chosen=accepted[-1]['coverage'] if accepted else prepared['frozen_best']['mean']
        comparisons.append([name,f"{100*prepared['raw_start_mean']:.2f}%",
            f"{100*prepared['frozen_best']['mean']:.2f}%",f"{100*chosen:.2f}%",
            f"{100*(chosen-prepared['frozen_best']['mean']):+.2f} pp",len(accepted)])
        for row in trajectory['rounds']:
            trained=row.get('training_steps',0)>0
            total_train+=trained;accepted_train+=trained and row['status']=='accepted'
            novel=row.get('gains',{}).get('relative_to_all_pre_round_candidates',float('-inf'))>=.002
            total_new_frontier+=novel
            rounds.append([key,row['round'],row.get('target','—'),row.get('training_steps','—'),
                f"{100*row['coverage']:.2f}%" if 'coverage' in row else '未获得有效分数',
                f"{100*row['gains']['relative_to_all_pre_round_candidates']:+.2f} pp" if 'gains' in row else '—',
                {'accepted':'接受更新','retained_for_research':'保留旧版本','execution_failed':'执行失败',
                 'invalid_proposal':'提案格式/协议失败'}.get(row['status'],row['status'])])
            directions.append(f"<h3>{key} · 第 {row['round']} 轮</h3><p>"+
                              html.escape(row.get('autonomous_direction_reason',row.get('error','—')))+"</p>")
            folder=f"{key}/requests/r{row['round']:02d}"
            directions.append(f'<p><a href="{folder}/request.json">原始可见输入</a> · '
                              f'<a href="{folder}/response.json">原始模型提案</a> · '
                              f'<a href="{folder}/result.json">实测与准入决定</a></p>')
            if row.get('error'):
                directions.append('<pre>'+html.escape(row['error'])+'</pre>')
        results=trajectory['audit']
        get=lambda role:next(iter(results[role]['summary'].values()))['mean'] if results[role]['summary'] else None
        old,new=get('frozen_library_champion'),get('selected_after_search')
        if old is not None and new is not None:
            retained_transfer+=bool(accepted) and new>=old-.01
            audits.append([name+' → '+NAMES[prepared['source_audit']['audit_group']],
                           f'{100*old:.2f}%',f'{100*new:.2f}%',f'{100*(new-old):+.2f} pp',
                           '有新更新且未超过预定退化容限' if accepted and new>=old-.01 else
                           ('新更新在迁移来源退化' if accepted else '无新准入更新；不是增益迁移证据')])
        else:audits.append([name,'—','—','—','审核执行失败，不能认定通过'])
    evidence=verify_evidence(directory)
    protection=verify_snapshot(ROOT,read(ROOT/'outputs/rsi_frontier_validation_20261017_v1/preflight/protected_before.json'))
    library=read(directory/'library.json')
    for path,digest in library['source_sha256'].items():
        if file_hash(inside(ROOT,path))!=digest:raise ValueError('Historical archive changed')
    summary={'trajectories':3,'proposal_rounds':sum(len(t['rounds']) for t in source['trajectories']),
             'rounds_exceeding_pre_round_frontier_by_at_least_0_2pp':total_new_frontier,
             'autonomous_training_choices':total_train,'accepted_training_choices':accepted_train,
             'updated_trajectories_retained_within_1pp_transfer_tolerance':retained_transfer,
             'accepted_new_updates':sum(r['status']=='accepted' for t in source['trajectories'] for r in t['rounds']),
             'failed_proposal_rounds':sum(r['status'] in {'invalid_proposal','execution_failed'}
                 for t in source['trajectories'] for r in t['rounds']),
             'protected':protection,'execution_evidence':evidence,
             'source_results_sha256':file_hash(directory/'results.json'),
             'interpretation':'Development-exposed source transfer pilot, no new-blind-test, '
                              'independent-dataset, general-agent-superiority or continuous-gain claim.'}
    write_new(directory/'summary.json',summary)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>RSI 自主方向与新增益验证</title>
<style>body{font:17px Arial,"Noto Sans CJK SC",sans-serif;color:#203341;background:#f5f7f7;max-width:1220px;margin:36px auto;padding:0 24px;line-height:1.65}h1,h2{color:#135e64}table{border-collapse:collapse;width:100%;background:white;margin:18px 0 30px}th,td{border:1px solid #d3dfe1;padding:10px;text-align:left}th{background:#e5f0f0}.note{background:#fff5da;padding:16px;border-left:5px solid #c59638}p,pre{overflow-wrap:anywhere}pre{white-space:pre-wrap;font-size:14px;background:#eee;padding:12px}</style>
<h1>PhyEvo-Mic：自主研究方向与新最优验证</h1><p>2026-10-17 · 三条来源轮换轨迹 · 每条三次独立提案 · 保留全部结果</p>'''
    page+='<p>本轮实际测试：超过已经可用的候选库最优值、由提案模型自行选择是否训练、冻结后检验跨来源保留。专家没有重新标注，当前最佳发布没有被替换。</p>'
    page+=('<p><strong>本轮结论：</strong>九次提案中，'+str(summary['accepted_new_updates'])+
           ' 次达到新版本准入条件；自主选择训练 '+str(total_train)+' 次；有新更新并在迁移来源保留到预定容限内的轨迹 '+
           str(retained_transfer)+' 条。未产生新更新的轨迹只证明保留旧版本，不能计作持续提升成功。</p>')
    page+='<div class="note">指标为固定 10% 面积预算下的专家已知正例覆盖率，不是准确率或 IoU。原图来源和历史算法已在开发中接触；本轮是前瞻执行的开发迁移实验，不是全新盲测。三个来源也不等于三个独立数据集。</div>'
    page+='<h2>实验怎样安排</h2>'+table(['轨迹','本轮拟合来源','拟合图像/平面数','开发选择来源','封存后迁移来源'],splits)
    page+='<p>每条轨迹从该来源重新实测的旧库冠军出发，最多三次新提案；阈值、数据划分、评价器和候选库均提前冻结。继承的研究策略选择操作和父版本，新提案自行选择分析、协同或学习方向。新训练上限 64 步、CPU、单次拟合超时 180 秒；仅一个评测种子，不是多随机种子统计。原有学生预测来自经过来源排除审计的权重，并非由本表唯一一组数据从零训练。</p>'
    page+='<p>每张图：覆盖率 = 入选像素中命中的专家正例像素数 ÷ 专家正例总像素数。入选像素不超过整张图面积的 10%，且只从有效候选区域按分数选取；来源分数为其图像/平面的算术平均。未标注像素不作为负例，本指标不测量误报率。</p>'
    page+='<h2>1. 相对旧起点与已有最优，分别提高多少？</h2>'+table(
        ['开发来源','原框架起点','重测候选库最优','迭代后保留','超过候选库','接受轮数'],comparisons)
    page+='<p>冻结库包括此前 meta/learning 归档的全部 18 个候选记录（其中一个是研究策略）；按本轮相同来源划分重新执行。不是跨不同指标直接比较论文里的最佳数字。旧学习程序按当前训练来源重训作为参照，不计作系统本轮自主决定训练。</p>'
    page+='<h2>2. 九次提案完整记录</h2>'+table(['轨迹','轮次','自主选择','训练步数','开发覆盖','超过轮前最好','准入'],rounds)
    page+=f'<p>自主选择训练：{total_train} 次；其中获准更新：{accepted_train} 次。未选择训练不自动等于错误，但不能据此宣称已经验证训练带来的收益。</p>'
    page+='<h2>3. 全部搜索封存后的迁移检查</h2>'+table(['来源迁移','冻结库冠军','迭代后版本','变化','解读'],audits)
    page+='<p>迁移阶段只推理，不再训练或改选；比较的是开发集选中的同一版本，不在迁移结果中挑赢家。模型调用由当前会话中转，未建立脱离会话的长期服务。</p>'
    page+='<p>模型调度指定 GPT-5.6 Sol，各提案来自不继承对话的独立会话。记录九次提案会话及原始回复；运行环境未提供后端模型认证或 token 用量，未据此推算费用。</p>'
    page+='<p>“未退化”按预先约定的 1 个百分点容限判断；它不等于在新来源获得正增益。没有通过开发准入的新版本，也不能用原版本不变来声称改进已成功迁移。</p>'
    page+='<h2>本轮暴露出的具体问题</h2>'
    page+='<p>三步预算与继承策略的探索时机不匹配：该策略只在轮次为 3 的倍数、并且剩余提案数大于 1 时触发交叉演化；本轮到第三步只剩一次，因此不会触发交叉。实际执行了 '+html.escape(str(evidence['actual_research_actions']))+'。这不是用结果倒推的新规则，而是封存策略代码中的条件。后续应另立协议验证预算感知的探索策略，不能在本轮结束后修改条件并仍称同一实验。</p>'
    page+='<p>当前诊断主要提供整体覆盖、支持域遗漏比例和物理/学生相关性，没有直接告诉模型哪些具体错误模式可通过训练修复。结合每条轨迹只有一个新增拟合来源、CPU 64 步上限，九次均未启动训练；这些是解释当前选择的线索，不是已经验证的唯一原因。下一轮应补充匿名的错例类别统计和可学习性探测，检验系统在什么条件下选择训练，而非人为强制它选择学习。</p>'
    page+='<p>一次失败来自参数状态接口：实际推理传入字典，但模型将其当作数组读取；现有请求对状态格式的说明不够明确。后续修复提案解决了执行错误，不过检测结果仍未改善。接口修复成功与科学指标提升分别记账。</p>'
    page+='<h2>4. 原始方向选择理由</h2>'+''.join(directions)
    page+='<h2>5. 可复核记录</h2><p><a href="results.json">全部结果</a> · <a href="pilot_protocol.json">冻结协议</a> · <a href="library.json">冻结候选库</a> · <a href="summary.json">核验摘要</a></p></html>'
    with (directory/'report.html').open('x',encoding='utf-8') as f:f.write(page)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();report(args.directory)
