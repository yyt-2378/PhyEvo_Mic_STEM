#!/usr/bin/env python3
"""Concise Chinese PDF and three useful summary figures from verified results."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager
import numpy as np
from PIL import Image
from tfree_detection.expert_supervision import contained_path,sha256_file,write_new_json,load_learning_dataset,normalize_image
from tfree_detection.config import load_config

ROOT=Path(__file__).resolve().parents[1]
EXPERT=ROOT/'outputs/fair_baselines_20261014_v1'
EXTERNAL=ROOT/'outputs/fair_baselines_external_20261014_v1'
DENSE=ROOT/'outputs/fair_baselines_synthetic_transfer_20261014_v1'
LABELS={'physics_budgeted':'原物理 同预算','phyevo_teacher_expert':'物理教师＋学生','same_student_expert_only':'同结构 仅专家',
        'unet_expert_only':'U-Net 仅专家','emseek_refunet_expert_only':'EMSeek Ref-UNet',
        'hybrid_phyevo_teacher_expert':'物理与学生固定组合','dennler_vgg16_eigenfilter':'Dennler 完整流程',
        'physics_historical':'历史物理分支','fourier_bragg':'Fourier/Bragg',
        'physics_legacy_historical':'历史位点残差分支','physics_periodic_historical':'历史周期反事实分支'}
GROUPS={'guo-superscan-single-acquisition':'SuperScan 点位','guo_2021_mowte2_acquisition':'MoWTe2 点位',
        'guo_2021_zro2_twin_acquisition':'ZrO2 界面','sigma3-orientation-i':'Al ORI 界面','sigma3-orientation-ii':'Al ORII 界面'}
MODELS=['phyevo_teacher_expert','same_student_expert_only','unet_expert_only','emseek_refunet_expert_only']

def read(path):return json.loads(path.read_text())

def sealed(directory):
    s=read(directory/'seal.json')
    for path,digest in s['files'].items():
        if sha256_file(contained_path(directory,path))!=digest:raise ValueError('Changed analysis')
    return read(directory/'summary.json')

def percent(value,std=None):
    return f'{100*value:.2f}%' if std is None else f'{100*value:.2f} ± {100*std:.2f}'

def text_page(title,paragraphs):
    fig=plt.figure(figsize=(12.8,8));fig.text(.06,.91,title,fontsize=23,color='#173b52',weight='bold')
    y=.81
    for paragraph in paragraphs:
        lines=[]
        for line in paragraph.split('\n'):
            # Short source prose is prewrapped; CJK fallback keeps lines visible.
            lines.extend(line[i:i+57] for i in range(0,len(line),57))
        fig.text(.065,y,'\n'.join(lines),fontsize=13,va='top',linespacing=1.65,color='#243e4e')
        y-=.05*len(lines)+.035
    return fig

def table_page(title,columns,rows,footnote,*,widths=None):
    fig,ax=plt.subplots(figsize=(12.8,8));ax.axis('off')
    fig.text(.06,.92,title,fontsize=22,color='#173b52',weight='bold')
    table=ax.table(cellText=rows,colLabels=columns,cellLoc='center',loc='upper center',colWidths=widths,
                   bbox=[0,.23,1,.65])
    table.auto_set_font_size(False);table.set_fontsize(10.5)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('#d8e4e8')
        if r==0:cell.set_facecolor('#18515c');cell.set_text_props(color='white',weight='bold')
        else:cell.set_facecolor('#eff6f5' if r%2 else '#ffffff')
    fig.text(.06,.17,footnote,fontsize=11,va='top',linespacing=1.6,color='#425968')
    fig.subplots_adjust(left=.04,right=.96,bottom=.1,top=.88)
    return fig

def pixel_overlap_audit():
    manifest,arrays=load_learning_dataset(ROOT,ROOT/'outputs/expert_cycle_001/learning_dataset_v1')
    if not manifest['admission']['development_retirement_authorized'] or manifest['admission']['purpose']!='user_authorized_expert_learning_cycle':
        raise ValueError('Expert learning admission missing')
    p=read(EXTERNAL/'protocol.json');numbers=p['folds'][0]['train_case_numbers'];fingerprints={}
    for number in numbers:
        image=arrays[number]['image']
        for rotation in range(4):
            for flip in (False,True):
                a=np.rot90(image,rotation)
                if flip:a=a[:,::-1]
                digest=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                fingerprints[(a.shape,digest)]=number
    matches=[]
    for row in p['external_images']:
        image=normalize_image(np.asarray(Image.open(contained_path(ROOT,row['image'])).convert('L')))
        key=(image.shape,hashlib.sha256(image.tobytes()).hexdigest())
        if key in fingerprints:matches.append({'test_image':row['number'],'training_case':fingerprints[key]})
    if matches:raise ValueError(f'Normalized pixel duplicate: {matches}')
    records=[json.loads(s) for s in (ROOT/manifest['frozen_bundle']/'cases.jsonl').read_text().splitlines()]
    configs={}
    for row in records:
        if row['case_number'] not in numbers:continue
        pred=row['prediction'];path=contained_path(ROOT,pred['config_path'])
        if sha256_file(path)!=pred['config_sha256']:raise ValueError('Physics configuration changed')
        c=load_config(path)
        if c.stage2.enabled or c.stage2.dino.enabled or c.stage2.sam3.enabled:raise ValueError('Pretrained teacher component enabled')
        configs[str(path.relative_to(ROOT))]={'sha256':sha256_file(path),'DINO_enabled':False,'SAM_enabled':False}
    return {'training_cases':len(numbers),'test_images':107,'exact_robust_normalized_pixel_duplicates_including_D4':matches,
            'expert_admission':manifest['admission'],'physics_teacher_configs':configs,
            'does_not_prove_absence_of_cropped_or_semantically_related_sources':True}

def build(output):
    output=contained_path(ROOT,output)
    if output.exists():raise FileExistsError(output)
    expert=sealed(EXPERT/'analysis_v2');external=sealed(EXTERNAL/'analysis_v1');dense=sealed(DENSE/'analysis_v1')
    output.mkdir(parents=True)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=font).get_name(),'DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':11,'pdf.fonttype':3})
    write_new_json(output/'pixel_overlap_audit.json',pixel_overlap_audit())
    index={(r['fold'],r['group'],r['method'],r['step'],r['budget']):r for r in expert['rows']}
    def get(group,method,step=800,budget='main',material=False):
        prefix='material-' if material else 'source-'
        return next(r for key,r in index.items() if key[0].startswith(prefix) and key[1:]==(group,method,step,budget))
    methods=['physics_budgeted']+MODELS+['hybrid_phyevo_teacher_expert']
    dense_labels={**LABELS,'same_student_expert_only':'同结构 仅合成真值',
                  'unet_expert_only':'U-Net 合成监督','emseek_refunet_expert_only':'EMSeek 合成监督'}
    def positive_area(cohort):
        rows=[r for r in dense['per_image'] if r['method']=='phyevo_teacher_expert' and r['cohort']==cohort]
        return sum(r['true_positive_pixels']+r['false_positive_pixels'] for r in rows)/sum(
            sum(r[k] for k in ('true_positive_pixels','false_positive_pixels','false_negative_pixels','true_negative_pixels')) for r in rows)
    with PdfPages(output/'PhyEvo_Mic_fair_baseline_results_zh.pdf') as pdf:
        def save(fig,png=None):
            fig.canvas.draw()
            renderer=fig.canvas.get_renderer()
            for label in fig.texts:
                box=label.get_window_extent(renderer)
                if box.x0<0 or box.y0<0 or box.x1>fig.bbox.width or box.y1>fig.bbox.height:
                    raise ValueError('Report text would be clipped; retain attempt and use a new output directory')
            if png:fig.savefig(output/png,dpi=180,bbox_inches='tight')
            pdf.savefig(fig);plt.close(fig)
        save(text_page('PhyEvo-Mic 公平重训与迁移比较',[
            '已完成 99 次训练，均不使用外部预训练权重。每项实验固定训练数据、裁块、随机种子与测试规则。',
            '专家队列：4 种学习方法 × 6 个留出划分 × 3 个种子，共 72 次。\nWSe2 迁移：4 种方法 × 3 个种子，共 12 次，全部使用相同的 151 例训练。',
            '区域外推：5 种方法 × 3 个种子，共 15 次，只使用相同的 37 张合成训练图。\n新增完整 Dennler VGG16＋图参考选择＋特征滤波流程，不再只比较 Fourier。',
            '这是固定训练机会下的完整网络比较，不是每个架构充分调参后的性能上限。\n主检查点固定 800 步；400 步仅作诊断，没有按外层成绩挑最好一版。',
            '既有数据曾参与项目开发。本文区分训练排除、采集来源留出、材料留出；\n不把这些开发性结果写成全新锁定外测，也不据此宣称持续 RSI 已获证实。']))
        rows=[]
        for group in GROUPS:
            row=[GROUPS[group]]
            for m in methods:
                r=get(group,m);row.append(percent(r['micro_coverage_mean'],r['micro_coverage_seed_std']))
            rows.append(row)
        save(table_page('专家队列  同预算已标阳性覆盖率', ['场景']+[LABELS[m].replace(' ','\n') for m in methods],rows,
            '按已标阳性总数加权的微平均，单位 %；± 为 3 个种子标准差，不是总体置信区间。\n点位：每 512×512 最多 50 点，6 px 一对一匹配；界面：最多标记 10% 面积。\n指标不是正确率或 IoU。未标区域不是已确认阴性，不能从这组标注计算完整精确率。'),
            '01_expert_equal_budget.png')
        rows=[]
        for group in list(GROUPS)[-2:]:
            rows.append([GROUPS[group]]+[percent(get(group,m,material=True)['micro_coverage_mean'],get(group,m,material=True)['micro_coverage_seed_std']) for m in methods])
        save(table_page('真正排除铝材料  同时留出 ORI 和 ORII',['测试场景']+[LABELS[m].replace(' ','\n') for m in methods],rows,
            '训练中的界面监督只来自 ZrO2，训练仍保留同样的点位来源。\nORI→ORII 的单来源留出属于同材料迁移；本页才是训练中同时排除两种 Al 取向。\n每个取向只有一组采集，多个帧和三个种子不等于多个独立材料实验。'))
        fig,axes=plt.subplots(2,3,figsize=(12.8,8));colors=['#008b83','#687586','#de9d29','#a44474']
        for ax,group in zip(axes.flat,GROUPS):
            for method,color in zip(MODELS,colors):
                rows=[get(group,method,step=s) for s in (400,800)]
                ax.errorbar([400,800],[100*r['micro_coverage_mean'] for r in rows],
                    yerr=[100*r['micro_coverage_seed_std'] for r in rows],label=LABELS[method],color=color,marker='o',capsize=3)
            ax.set_title(GROUPS[group]);ax.set_xticks([400,800]);ax.set_ylabel('已标阳性覆盖率 %');ax.grid(alpha=.15)
        axes.flat[-1].axis('off');axes.flat[-1].legend(*axes.flat[0].get_legend_handles_labels(),loc='center',frameon=False)
        fig.suptitle('训练更久不保证迁移更好  800 步仍是预先固定主结果',fontsize=18)
        fig.tight_layout(rect=[0,0,1,.94]);save(fig)
        rows=[]
        for m in MODELS+list(HIST for HIST in ('physics_legacy_historical','physics_periodic_historical')):
            r=next(r for r in external['summary'] if r['method']==m and r['budget']==50)
            rows.append([LABELS[m]]+[percent(r[k+'_mean'],r[k+'_seed_std']) for k in ('precision','recall','f1')])
        save(table_page('WSe2  训练中未使用的材料数据',['方法','作者参考精确率 %','作者参考召回率 %','作者参考 F1 %'],rows,
            '107 张图，16,030 个作者参考点，按一个保守来源组处理。相同数量上限与 6 px 匹配容差。\n前四行全部只用相同的专家数据从零训练；不使用 WSe2 标签选参或提供参考块。\n最后两行是历史物理参考，曾在 WSe2 上开发，不能当作全新外推优势的证据。\nEMSeek 是官方完整 Ref-UNet 在本监督协议下重训，不是作者论文分数或完整 Agent 平台。'),
            '02_WSe2_transfer.png')
        rows=[]
        for m in MODELS+['dennler_vgg16_eigenfilter','physics_historical','fourier_bragg']:
            r=next(r for r in dense['rows'] if r['method']==m and r['cohort']=='Dennler_sim_to_real')
            rows.append([dense_labels[m]]+[percent(r[k+'_mean'],r[k+'_seed_std']) for k in ('iou','f1','precision','recall')])
        save(table_page('合成训练到 Dennler  完整模型区域外推',['方法','IoU %','F1 %','精确率 %','召回率 %'],rows,
            '所有学习模型只训练同样的 37 张程序生成纹理；61 张 Dennler 真实图全部仅用于评价。\n目标是作者标注的任意缺陷区域。学习分割网络固定阈值 0.5；Dennler 使用原完整后处理。\n各模型均从零训练 800 步；VGG Adam 学习率 1e-4，其余 1e-3，均在评分前固定。\n这不是作者的真实图训练方案。历史物理分支仅作开发参考。\n'
            f'警示：教师学生预测阳性面积 {positive_area("Dennler_sim_to_real")*100:.2f}%，接近整图；高召回不代表有效检测。'),
            '03_dense_complete_baselines.png')
        rows=[]
        for m in MODELS+['dennler_vgg16_eigenfilter','physics_historical','fourier_bragg']:
            cells=[dense_labels[m]]
            for identity in ('fecral-46','fecral-148'):
                vals=[]
                for r in dense['per_image']:
                    if r['method']==m and r['id']==identity:
                        tp=r['true_positive_pixels'];fp=r['false_positive_pixels'];fn=r['false_negative_pixels']
                        vals.append(tp/max(tp+fp+fn,1))
                cells.append(percent(float(np.mean(vals)),float(np.std(vals,ddof=1))))
            rows.append(cells)
        save(table_page('FeCrAl  跨材料压力测试逐图结果',['方法','图 46 任意非零作者掩膜 IoU %','图 148 任意非零作者掩膜 IoU %'],rows,
            '学习模型仅用相同 37 张合成纹理训练，测试 FeCrAl 时不微调，不以测试标签选阈值。\n作者掩膜含多个颜色编码；这里只评价所有非零区域的并集，不臆造元素或缺陷类别语义。\n两张图、一个未解析来源组：不能估计材料总体能力。\n'
            f'教师学生预测阳性面积 {positive_area("FeCrAl_material_stress")*100:.2f}%，已经退化为全图阳性，不能宣称优势。'))
        audit=read(EXPERT/'analysis_v2/integrity_audit.json')
        rows=[]
        for m in MODELS:
            vals=[r for r in audit['costs'] if r['method']==m]
            rows.append([LABELS[m],f'{vals[0]["parameters"]:,}',f'{np.median([r["training_seconds"] for r in vals]):.1f}',
                         f'{max(r["peak_gpu_GiB"] for r in vals):.2f}'])
        save(table_page('计算规模与公平性的适用范围',['学习方法','参数量','800 步训练秒数 中位数','峰值显存 GiB'],rows,
            '同数据、同裁块、同种子不等于同 FLOPs，也不等于各架构都已最优。\nRef-UNet 另外接收同训练集内固定的正例参考块，推理平均 3 个参考；没有额外来源数据。\n物理教师为训练图提供规则生成的监督，是待检验的方法差异，不是新增人工标注。\n相同结构对照具有逐项相同的初始权重；所有方法的查询裁块哈希逐组核对一致。'))
        save(text_page('如何解读：正结果与失败都保留',[
            'WSe2：同数据从零训练，教师学生 F1 64.31%，EMSeek 适配基线 35.89%。\n支持物理教师在这一材料迁移上的帮助，不代表对 EMSeek 原论文或产品的超越。',
            '专家来源留出：Al ORII 的仅专家同结构学生更好，SuperScan 的 U-Net 更好。\nZrO2 的固定组合均值超过双方；不存在一条路线在所有场景都取胜。',
            '合成到真实：完整 Dennler IoU 29.02%，Fourier 23.93%，教师学生 15.02%。\n教师学生在 Dennler 上几乎全图阳性，在 FeCrAl 上全图阳性，是失败案例。',
            '现阶段应主张可解释的能力边界，而不是持续 RSI 已成立。\n各方法的最佳训练参数仍需相同的训练侧验证预算，不能用外测结果反向选最优。']))
        save(text_page('复现状态与结论边界',[
            '99 次纳入比较的训练与预测均完成。专家队列核验 4,824 个分数文件；\nWSe2 迁移核验 1,284 个；合成到真实模型另有 945 个测试预测。',
            '完整 Dennler 实现保留 VGG16、图参考选择、49 个特征滤波器、\nMahalanobis 距离、局部方差和原后处理。保留原代码特征向量的行排序约定。',
            '一批误用锁定 Dennler 图训练的 15 个模型已标记作废，不进入任何结果表。\n新模型重新随机初始化，只训练程序合成图；入口已阻止被作废批次重用。\n原数据与历史结果未修改。另保留解码失败、计数浅拷贝及权重下载失败记录。',
            '证据能回答：固定数据预算下，哪些场景适合物理先验，哪些适合学习网络；\n不能据此宣布对所有材料优于 EMSeek，或 RSI 已形成持续正收益。',
            '下一步应给各模型相同的训练侧调参预算，再检验新独立采集来源。\n已有专家标注继续使用，不需要让专家重做这轮标注。']))
    sources=['scripts/run_fair_baselines.py','scripts/analyze_fair_baselines.py','scripts/run_fair_external.py',
             'scripts/run_fair_dense.py','scripts/report_fair_baselines.py','scripts/prepare_fair_baselines.py',
             'scripts/verify_fair_baseline_sources.py','src/tfree_detection/fair_baselines.py',
             'scripts/quarantine_dense_training_violation.py',
             'src/tfree_detection/dennler_full.py','src/tfree_detection/fair_synthetic.py',
             'tests/test_fair_baselines.py','tests/test_dennler_full.py','tests/test_fair_synthetic.py',
             'environments/fair-baselines.txt']
    with ZipFile(output/'producer_source.zip','x',compression=ZIP_DEFLATED) as archive:
        for name in sources:archive.write(contained_path(ROOT,name),name)
    inputs=[EXPERT/'protocol.json',EXPERT/'analysis_v2/seal.json',EXTERNAL/'protocol.json',EXTERNAL/'analysis_v1/seal.json',
            DENSE/'protocol.json',DENSE/'analysis_v1/seal.json',ROOT/'vendor/fair_baselines_20261014/verified_source_provenance.json',
            ROOT/'outputs/fair_baselines_dense_20261014_v2/DISQUALIFIED.json']
    write_new_json(output/'result_seal.json',{'inputs':{str(p.relative_to(ROOT)):sha256_file(p) for p in inputs},
        'sources':{s:sha256_file(ROOT/s) for s in sources},
        'files':{p.name:sha256_file(p) for p in output.iterdir() if p.is_file()}})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='outputs/fair_baseline_report_20261014_v3')
    build(parser.parse_args().output)
