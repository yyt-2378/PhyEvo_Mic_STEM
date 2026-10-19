#!/usr/bin/env python3
"""Create data-derived Chinese figures, tables and PDF from sealed outcomes."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

from scripts.run_rsi_continual import ROOT, all_jobs, load_protocol, read, run_path, verified
from tfree_detection.expert_supervision import contained_path, load_learning_dataset, sha256_file, write_new_json
from tfree_detection.rsi_continual import predictions

GROUPS = {'guo-superscan-single-acquisition': 'SuperScan 点位',
          'guo_2021_mowte2_acquisition': 'MoWTe₂ 点位',
          'guo_2021_zro2_twin_acquisition': 'ZrO₂ 界面',
          'sigma3-orientation-i': 'Al ORI 界面', 'sigma3-orientation-ii': 'Al ORII 界面'}
ARMS = {'fixed_replay': '固定采样', 'error_feedback': '错误反馈'}


def setup_font():
    path = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_manager.fontManager.addfont(str(path))
    plt.rcParams.update({'font.family': [font_manager.FontProperties(fname=str(path)).get_name(), 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10, 'pdf.fonttype': 42})


def report(training, output):
    protocol, dataset = load_protocol(training)
    summary = verified(training/'summary.json')
    verification = verified(training/'verification.json')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    setup_font()
    rows, comparisons = summary['rows'], summary['primary_comparison']
    lookup = {(r['group'], r['arm'], r['round'], r['method']): r for r in rows}
    def val(group, arm, rnd, method):
        return 100*lookup[(group, arm, rnd, method)]['mean_coverage']
    fields = ['group', 'geometry', 'arm', 'round', 'method', 'mean_coverage']
    with (output/'all_rounds.csv').open('x', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows({k: r[k] for k in fields} for r in rows)
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, group in zip(axes.flat, GROUPS):
        for arm, color in [('fixed_replay', '#57697a'), ('error_feedback', '#087e8b')]:
            for method, style, suffix in [('student_expert', '--', '当前学生'), ('gated_release', '-', '保留机制后的版本')]:
                values = [val(group, arm, r, method) for r in range(1, 5)]
                ax.plot(range(1, 5), values, style, color=color, marker='o', lw=1.8, label=ARMS[arm]+'·'+suffix)
        ax.axhline(val(group, 'fixed_replay', 1, 'physics_budgeted'), color='#ba773b', lw=1, label='同预算物理')
        ax.set_title(GROUPS[group]); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel('轮次，每轮 400 步')
        ax.set_ylabel('已标正例覆盖率（%）'); ax.set_ylim(bottom=0); ax.grid(alpha=.2)
    axes.flat[-1].axis('off')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc='center', frameon=False, fontsize=11)
    figure.suptitle('完整四轮轨迹｜三个种子均值｜实线的稳定可能来自保留旧版本，不等于每轮新模型都改善', fontsize=14)
    figure.savefig(output/'01_all_round_trajectories.png', dpi=170); plt.close(figure)
    figure, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for i, item in enumerate(comparisons):
        delta = item['difference_pp']
        ax.barh(i, delta, color='#087e8b' if delta >= 0 else '#ba554a', alpha=.8)
        ax.scatter(item['seed_differences_pp'], [i]*3, c='#183444', s=25, zorder=3)
        ax.text(delta+(1 if delta >= 0 else -1), i+.18, f'{delta:+.2f} pp', ha='left' if delta >= 0 else 'right')
    ax.set_yticks(range(5), [GROUPS[x['group']] for x in comparisons]); ax.invert_yaxis()
    ax.axvline(0, c='#596773', lw=1); ax.grid(axis='x', alpha=.2)
    ax.set_xlabel('第四轮 错误反馈加版本保留 − 固定采样加版本保留（百分点）')
    ax.set_title('预先确定的主比较｜圆点为三个种子的配对差值，不是独立材料重复')
    figure.savefig(output/'02_compute_matched_primary.png', dpi=180); plt.close(figure)
    gate_rows = []
    for arm, seed, fold in all_jobs(protocol):
        run = run_path(training, arm, seed, fold)
        for r in range(1, 5):
            f = verified(run/f'round-{r:02d}'/'feedback.json')
            for d in f['decisions']:
                gate_rows.append({'arm': arm, 'seed': seed, 'fold': fold['fold_id'], 'round': r,
                                  'geometry': d['geometry'], 'accepted': d['accepted'],
                                  'selected_round': d['selected']['round'], 'branch': d['selected']['branch']})
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, arm in zip(axes, ARMS):
        accepted = [sum(x['accepted'] for x in gate_rows if x['arm'] == arm and x['round'] == r) for r in range(1, 5)]
        total = [sum(1 for x in gate_rows if x['arm'] == arm and x['round'] == r) for r in range(1, 5)]
        ax.bar(range(1, 5), accepted, label='接纳新版本', color='#087e8b')
        ax.bar(range(1, 5), np.array(total)-accepted, bottom=accepted, label='保留旧版本', color='#cbd4d9')
        ax.set_xticks(range(1, 5)); ax.set_title(ARMS[arm]); ax.set_ylabel('折 × 种子 × 几何任务的决定数')
    axes[1].legend(frameon=False)
    figure.suptitle('控制数据的接纳与保留记录｜不是外层测试的改善次数')
    figure.savefig(output/'03_acceptance_history.png', dpi=180); plt.close(figure)
    write_new_json(output/'gate_history.json', gate_rows)

    manifest, planes = load_learning_dataset(ROOT, dataset)
    entries = {e['case_number']: e for e in manifest['entries']}
    records = {r['case_number']: r for r in map(json.loads, (contained_path(ROOT, manifest['frozen_bundle'])/'cases.jsonl').read_text().splitlines())}
    # Fixed representative IDs and seed, NOT best-by-result examples.
    example_files = []
    for index, numbers in enumerate(([63, 2], [103, 105, 135]), 4):
        fig, axes = plt.subplots(len(numbers), 4, figsize=(14, 3.8*len(numbers)), squeeze=False, constrained_layout=True)
        for ri, n in enumerate(numbers):
            e, plane = entries[n], planes[n]
            fold = next(f for f in protocol['folds'] if f['partition']['outer_group'] == e['group'])
            run = run_path(training, 'error_feedback', 23, fold)
            evaluation = verified(run/'outer_evaluation.json')
            row = next(r for r in evaluation['rows'] if r['round'] == 4 and r['case_number'] == n)
            def proposal(r):
                with np.load(run/f'round-{r:02d}'/'outer_scores'/f'case-{n:03d}.npz', allow_pickle=False) as archive:
                    return predictions(plane['image'], plane['teacher'], records[n]['prediction']['candidates'], archive['score'], e['geometry'])
            latest = proposal(4)
            route = row['route']
            released = latest if route['round'] in (0, 4) else proposal(route['round'])
            panels = [None, latest['physics_budgeted'], latest['student_expert'], released[route['branch']]]
            titles = ['已有专家正标注', '同预算物理', '第 4 轮当前学生', f'选定版本：第 {route["round"]} 轮']
            methods = [None, 'physics_budgeted', 'student_expert', 'gated_release']
            for ci, (ax, p, title, method) in enumerate(zip(axes[ri], panels, titles, methods)):
                ax.imshow(plane['image'], cmap='gray', vmin=0, vmax=1)
                if e['geometry'] == 'point':
                    xy = np.asarray(e['points_xy'])
                    ax.scatter(xy[:, 0], xy[:, 1], facecolors='none', edgecolors='#35d6c5', s=30, lw=.8)
                    if p is not None and len(p.points_xy):
                        ax.scatter(p.points_xy[:, 0], p.points_xy[:, 1], facecolors='none', edgecolors='#ffc557', s=18, lw=.65)
                else:
                    ax.contour(plane['human'], levels=[.5], colors=['#35d6c5'], linewidths=.8)
                    if p is not None and p.mask.any():
                        rgba = np.zeros((*p.mask.shape, 4)); rgba[p.mask] = (1., .72, .2, .6)
                        ax.imshow(rgba)
                if method:
                    title += f'\n覆盖 {100*row["methods"][method]["positive_coverage"]:.2f}%'
                ax.set_title(title, fontsize=11); ax.set_xticks([]); ax.set_yticks([])
            axes[ri, 0].set_ylabel(f'{GROUPS[e["group"]]}\n病例 {n}，seed 23', fontsize=11)
        fig.suptitle('固定病例举例｜青绿为专家正标注，黄色为候选｜未标注处不自动判作误报', fontsize=13)
        name = f'{index:02d}_fixed_examples.png'
        fig.savefig(output/name, dpi=160); plt.close(fig); example_files.append(name)

    positives = sum(c['difference_pp'] > 0 for c in comparisons)
    negatives = sum(c['difference_pp'] < 0 for c in comparisons)
    zero = len(comparisons)-positives-negatives
    opening = (f'四轮、两条等步数训练线、三个种子和五个整组留出来源已经完成。预先确定的第四轮主比较中，'
               f'错误反馈加版本保留相对固定采样加版本保留，在 {positives} 个来源提高、{negatives} 个来源下降、{zero} 个来源持平。'
               '这是对错误反馈增益的直接检验，而不是从全部轮次中挑最高分。')
    table = [['场景', '同预算物理', '固定采样加保留', '错误反馈加保留', '差值 pp']]
    for c in comparisons:
        table.append([GROUPS[c['group']], f"{val(c['group'], 'fixed_replay', 4, 'physics_budgeted'):.2f}%",
                      f"{100*c['fixed_gated']:.2f}%", f"{100*c['feedback_gated']:.2f}%", f"{c['difference_pp']:+.2f}"])
    secondary = [['场景', '固定采样学生', '反馈学生', '反馈与物理融合', '同预算物理']]
    for group in GROUPS:
        secondary.append([GROUPS[group], f"{val(group, 'fixed_replay', 4, 'student_expert'):.2f}%",
            f"{val(group, 'error_feedback', 4, 'student_expert'):.2f}%",
            f"{val(group, 'error_feedback', 4, 'student_reranked_physics_plus_rescue'):.2f}%",
            f"{val(group, 'fixed_replay', 4, 'physics_budgeted'):.2f}%"])
    mechanisms = ('上一轮训练正标注的漏检与相对历史最好成绩的遗忘，改变下一轮病例和裁块采样。每轮仍复习全部来源；'
                  '控制数据决定是否接纳新检查点，不通过则保留旧版本。默认推理仍用原物理，显式研究开关才启用冻结选择；'
                  '学生只改变候选关注顺序和补漏，不修改原物理测量。')
    boundaries = ('主指标是固定预算下的已标正例覆盖：点位 6 像素容差、每 512×512 像素最多 50 点；界面占图像面积 10%。'
                  '它不是总体正确率、精确率或 F1。多帧与三个随机种子不是独立材料重复。现有队列曾用于开发；'
                  '内部控制是同采集的保留帧或图块，跨来源结论以外层成绩为准，不能宣称新锁定外测或普遍不退步。')
    claim = ('可支撑的论点是：物理框架可与专家监督学生构成可审计、可拒绝退步的多轮反馈机制，并在固定预算、等步数对照下描述其适用边界。'
             '是否对某一来源有效，应引用表中配对结果。不能将闭环实现完成写成已经证明所有场景持续正向 RSI。')
    diagnosis = ('瓶颈不只在训练。第四轮学生单独比较，错误反馈在三个界面来源的均值高于固定采样，而两个点位来源更低；'
                 '加入控制选择后，ORI 和 ORII 的主比较却转为负值。控制规则选错跨来源检查点，可能抵消已有学生收益。'
                 'SuperScan 主比较提高也不代表其末轮学生更好：末轮学生反而低于固定采样，说明训练与选择的贡献不能混为一谈。')
    physical_value = ('ZrO₂ 的第四轮固定比例融合覆盖为 17.05%，高于该反馈学生的 6.18% 和物理的 12.36%；'
                      '三个种子的融合均不低于物理，其中一个持平。这个单一采集来源支持保留物理候选的补充价值，'
                      '不证明所有材料都应使用固定比例融合：SuperScan 融合覆盖反而低于其学生。')
    decision = {'default_changed': False, 'production_promotions': 0,
                'verdict': 'research_only_no_universal_continual_improvement_claim',
                'basis': 'observed fixed-round paired regressions; not a post-hoc routing policy',
                'primary_comparison': comparisons,
                'next_hypothesis': 'validate continual proposals on acquisition-excluded inner sources; do not replace this experiment results',
                'existing_expert_labels_preserved': True}
    write_new_json(output/'adoption_assessment.json', decision)
    lines = ['# PhyEvo Mic 多轮错误反馈结果', '', opening, '', '## 预先确定的第四轮比较', '',
             '| '+' | '.join(table[0])+' |', '| '+' | '.join(['---']*5)+' |']
    lines += ['| '+' | '.join(r)+' |' for r in table[1:]]
    lines += ['', boundaries, '', '## 学生与版本选择分开看', '',
              '以下是预先声明的第四轮组件比较，不是根据外层分数重新指定部署分支。', '',
              '| '+' | '.join(secondary[0])+' |', '| '+' | '.join(['---']*5)+' |']
    lines += ['| '+' | '.join(r)+' |' for r in secondary[1:]]
    lines += ['', diagnosis, '', physical_value, '',
              '本轮任务构成和控制留出不同于上一轮六策略训练，因此不能把本轮与上一轮最大值之差直接归因于错误反馈。等量对照仅指本轮两条训练线。', '',
              '## 实际闭环', '', mechanisms, '',
              '## 完整轨迹', '', '![四轮轨迹](01_all_round_trajectories.png)', '',
              '![等步数主比较](02_compute_matched_primary.png)', '',
              f'控制阶段共接纳 {verification["accepted_control_updates"]} 次、保留旧版本 {verification["rejected_control_updates"]} 次；这不是测试改善次数。', '',
              '![版本接纳](03_acceptance_history.png)', '', '## 固定病例举例', '']
    for name in example_files:
        lines += [f'![固定病例]({name})', '']
    lines += ['## 文章表述边界', '', claim, '',
              '## 可复核材料', '',
              f'30 条训练轨迹、120 个检查点、{verification["outer_rows"]} 条逐病例逐轮记录。两臂首轮模型、优化器和随机状态逐数组一致；全部外层指标由保存分数独立重算。', '',
              '完整均值和轮次表为 `all_rounds.csv`；逐病例、逐种子分数和候选选择均保留在上一级实验目录。运行协议、检查点、父轮反馈和评分文件通过哈希关联。', '',
              '未删除或覆盖原始数据、专家标注、旧模型和旧实验。未修改专家服务，未自动提交或推送 GitHub，未晋升生产。', '']
    with (output/'results_zh.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    body = ParagraphStyle('body', fontName='STSong-Light', fontSize=11, leading=17, spaceAfter=10)
    title = ParagraphStyle('title', parent=body, fontSize=21, leading=29, textColor=colors.HexColor('#173c53'))
    heading = ParagraphStyle('heading', parent=body, fontSize=15, leading=22, spaceBefore=6)
    small = ParagraphStyle('small', parent=body, fontSize=9, leading=13)
    def para(s, style=body):
        # Standard CID body fonts lack Unicode subscript digits; preserve the
        # chemistry labels as readable plain digits rather than missing glyphs.
        s = s.translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789'))
        return Paragraph(s.replace('&', '&amp;').replace('<', '&lt;'), style)
    def img(name, maxh):
        from PIL import Image as PILImage
        with PILImage.open(output/name) as im:
            w, h = im.size
        scale = min(720/w, maxh/h)
        return Image(str(output/name), width=w*scale, height=h*scale)
    content = [para('PhyEvo Mic 多轮错误反馈结果', title), para(opening), para('预先确定的第四轮主比较', heading)]
    native_table = Table([[para(x, small) for x in row] for row in table], colWidths=[150, 110, 150, 150, 100])
    native_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5f0f2')),
                                    ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#ccd7dd')),
                                    ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    content += [native_table, Spacer(1, 12), para(boundaries, small), para(claim), PageBreak(),
                para('机制与接纳记录', title), para(mechanisms),
                para('两条训练线各 1600 步，首轮完全相同。错误反馈只影响之后的训练采样；控制决定先封存，外层后评分。'),
                para('五个外层模型是交叉验证产物，不是按测试材料身份挑选最优模型的部署系统。'),
                img('03_acceptance_history.png', 270), PageBreak(),
                para('四轮轨迹与保留机制', title), img('01_all_round_trajectories.png', 440),
                para('虚线为当轮新学生，实线为控制决定采用的版本。实线没有下降可能只是没有接纳退步模型。', small), PageBreak(),
                para('错误反馈是否优于相同训练步数', title), img('02_compute_matched_primary.png', 330),
                para('配对差值同时保留三个种子的离散性。不能把 15 个来源与种子组合当作 15 个独立采集实验，也不汇总不同几何指标伪造总体准确率。'),
                para(claim), PageBreak(), para('学生改善与版本选择是两个问题', title)]
    secondary_table = Table([[para(x, small) for x in row] for row in secondary], colWidths=[150, 130, 120, 150, 110])
    secondary_table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5f0f2')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, -1), .3, colors.HexColor('#ccd7dd')),
        ('TOPPADDING', (0, 0), (-1, -1), 7), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    content += [secondary_table, Spacer(1, 12), para(diagnosis), para(physical_value),
                para('组件比较按固定第四轮完整报告，不将某一来源的较好组件事后拼成新的测试分数。保留原默认框架；这些研究检查点没有生产晋升。', small), PageBreak()]
    for i, name in enumerate(example_files):
        content += [para('固定病例的候选与专家标注', title), img(name, 435),
                    para('病例按预先固定编号选取，使用 seed 23；没有根据最好成绩挑案例。界面宽带覆盖不等于中心线定位精度，黄色候选不代表已确认缺陷。', small)]
        if i < len(example_files)-1:
            content.append(PageBreak())
    doc = SimpleDocTemplate(str(output/'PhyEvo_Mic_continual_feedback_zh.pdf'), pagesize=(842, 595),
                            rightMargin=45, leftMargin=45, topMargin=28, bottomMargin=28,
                            title='PhyEvo Mic 多轮错误反馈结果', author='PhyEvo-Mic')
    def footer(canvas, doc):
        canvas.setFont('STSong-Light', 9); canvas.setFillColor(colors.HexColor('#617482'))
        canvas.drawString(45, 17, '现有开发队列研究证据 · 正例覆盖，不是总体正确率')
        canvas.drawRightString(797, 17, str(doc.page))
    doc.build(content, onFirstPage=footer, onLaterPages=footer)
    sources = set(protocol['source_hashes']) | {'scripts/report_rsi_continual.py', 'scripts/verify_rsi_continual.py',
        'scripts/run_rsi_continual_detection.py', 'src/tfree_detection/rsi_continual_runtime.py'}
    with ZipFile(output/'producer_source.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for name in sorted(sources):
            archive.write(contained_path(ROOT, name), name)
    files = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_new_json(output/'seal.json', {'files': files, 'summary_sha256': sha256_file(training/'summary.json'),
                                      'verification_sha256': sha256_file(training/'verification.json'),
                                      'source_hashes': {p: sha256_file(ROOT/p) for p in sorted(sources)}})
    print(json.dumps({'output': str(output), 'positive_sources': positives, 'negative_sources': negatives, 'ties': zero}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training', type=Path, default=Path('outputs/rsi_cycle_003'))
    p.add_argument('--output', type=Path, default=Path('outputs/rsi_cycle_003/report_v1'))
    a = p.parse_args()
    report(contained_path(ROOT, a.training), contained_path(ROOT, a.output))
