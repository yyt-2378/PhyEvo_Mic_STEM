#!/usr/bin/env python3
"""Verify inference artifacts and produce a provenance-qualified comparison.

The native runner's normal-control field is empty because its historical rows
use category='normal', not 'non_defective'. Independently recompute that field
here, preserving the registered inference code/results and recording the fix.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from scripts.run_author_pretrained import checked,ROOT,DEFAULT,read
from scripts.verify_rsi_cycle import require,junit_counts
from tfree_detection.expert_supervision import sha256_file,contained_path,write_new_json

NAMES={'guo-superscan-single-acquisition':'SuperScan 点位','guo_2021_mowte2_acquisition':'MoWTe₂ 点位',
       'guo_2021_zro2_twin_acquisition':'ZrO₂ 界面','sigma3-orientation-i':'Al ORI 界面','sigma3-orientation-ii':'Al ORII 界面'}

def main():
    output=ROOT/DEFAULT;p=checked(output);r=read(output/'results.json')
    for name,digest in read(output/'seal.json')['files'].items():require(sha256_file(contained_path(output,name))==digest,'Inference artifact changed')
    require(r['protocol_sha256']==sha256_file(output/'protocol.json'),'Scoring protocol mismatch')
    tasks=p['tasks'];require(len(tasks)==321 and len({t['id'] for t in tasks})==321,'Incomplete registered cohort')
    receipts=[read(output/'predictions'/t['id']/'receipt.json') for t in tasks]
    require(all(x['status']=='success' and x['model_sha256']==p['model_sha256'] for x in receipts),'Missing or wrong author prediction')
    for shard in range(6):require(not read(output/f'shard-{shard:02d}.json')['failed'],'Failed shard')
    best=read(ROOT/'outputs/best_release_20261014_v1/manifest.json')
    require(sha256_file(ROOT/'outputs/best_release_20261014_v1/manifest.json')==p['best_release_manifest_sha256'],'Best release changed')
    normal=defaultdict(list)
    for t in tasks:
        if t['kind']!='dennler' or t['input']['category']!='normal':continue
        row=t['input']
        require(sha256_file(contained_path(ROOT,row['physical']))==row['physical_sha256'],'Historical normal prediction changed')
        with np.load(output/'predictions'/t['id']/'prediction.npz',allow_pickle=False) as a,np.load(contained_path(ROOT,row['physical']),allow_pickle=False) as h:
            masks={'author_pretrained_full':a['labels']>0,'historical_physics_combined':h['any_defect_region'],
                   'historical_physics_score':h['coherence_loss_region']}
        for name,mask in masks.items():normal[name].append(float(np.asarray(mask,bool).mean()))
    require(all(len(v)==30 for v in normal.values()),'Normal controls incomplete')
    normal_summary={name:{'images':len(v),'mean_false_positive_area':float(np.mean(v)),'max_false_positive_area':float(np.max(v))} for name,v in normal.items()}
    expert_table=[]
    for b in best['table']:
        native=next(x for x in r['expert_summary'] if x['group']==b['group'] and x['method']=='author_eigenfilter_score_equal_budget')
        expert_table.append({'group':b['group'],'cases':native['cases'],'physics':b['physics_budgeted']['mean'],
            'frozen_student':b['student_expert']['mean'],'frozen_combination':b[b['reported_combination_branch']]['mean'],
            'combination_branch':b['reported_combination_branch'],'author_score_adapted':native['mean']})
    interface=[]
    for group in ['guo_2021_zro2_twin_acquisition','sigma3-orientation-i','sigma3-orientation-ii']:
        cases=[x for x in r['expert_cases'] if x['group']==group]
        interface.append({'group':group,'native_coverage':float(np.mean([x['methods']['author_native_region']['positive_coverage'] for x in cases])),
            'native_area_fraction':float(np.mean([x['methods']['author_native_region']['selected_area_fraction'] for x in cases])),
            'thinned_coverage':next(x['mean'] for x in r['expert_summary'] if x['group']==group and x['method']=='author_native_region_equal_budget')})
    counts=Counter(t['kind'] for t in tasks)
    splits=Counter(t['input']['author_split'] for t in tasks if t['kind']=='dennler')
    parity=read(contained_path(ROOT,p['parity']));tests=junit_counts(ROOT/'vendor/author_pretrained_20261014_v1/adapter_tests_v2.xml')
    em=ROOT/'vendor/author_pretrained_20261014_v1/emseek';emdata=read(em/'data_archive_inventory.json');emcode=read(em/'code_inventory.json')
    require(not emdata['weight_like_members'],'EMSeek weights found; evaluate instead of claiming unavailable')
    findings={'cohorts':dict(counts),'dennler_author_split_counts':dict(splits),'successful_predictions':len(receipts),
        'author_parameter_parity':{'tensors':parity['parameter_tensors'],'probability_max_abs_difference':parity['probability_max_abs_difference'],
             'float64_layer_max_difference':max(x['max_abs_difference'] for x in parity['float64_layer_checks'])},
        'tests':tests,'expert_table':expert_table,'interface_native_budget_audit':interface,'normal_controls_corrected':normal_summary,
        'normal_control_correction':'Original scorer checked non_defective, while source category is normal; recalculated from all30 immutable predictions, no inference change',
        'emseek_author_weights':{'status':'not_found_in_checked_public_assets','code_zip_members':len(emcode),'data_zip_members':len(emdata['members']),
            'data_zip_central_directory_only':True,'random_weights_substituted':False,'same_data_trained_models_not_author_weights':True},
        'model_sha256':p['model_sha256'],'results_sha256':sha256_file(output/'results.json'),'reporter_sha256':sha256_file(Path(__file__))}
    write_new_json(output/'comparison_audit.json',findings)
    def value(kind,subset,method,border=0,metric='iou'):
        return next(x['metrics'][metric] for x in r['dense_summary'] if (x['kind'],x['subset'],x['method'],x['border'])==(kind,subset,method,border))*100
    lines=['# 作者预训练模型直接迁移对比','',
        '已使用 Dennler 作者发布的完整 VGG16 权重及分类、参考块选择、特征滤波、局部方差分割流水线，完成 321 张图的直接推理；没有微调、训练或根据测试标签调阈值。EMSeek 的作者任务权重未在已核查的官方发布资产中找到，因此没有冒充完成其预训练评测。','',
        '结论不是 PhyEvo-Mic 全面领先。Dennler 在其明确留出的原任务测试图及 Al ORI 界面上有优势；我们冻结的开发最佳学生或组合在 SuperScan、ZrO₂、Al ORII 的已标注正例覆盖上更高。这些跨训练来源的直接迁移比较，不等于相同训练资源下的模型排名。','',
        '## 与确认保留的最佳结果逐场景对照','',
        '以下均为同一专家病例、同一预测预算和同一匹配指标下的覆盖率百分数。点位最多每 512×512 像素 50 个，容差 6 像素；界面最多 10% 面积。我们列出的是三种子均值，作者模型为固定单个已发布权重。','',
        '| 场景 | 原物理 | 冻结学生 | 原表组合 | 作者预训练分数图转接 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for x in expert_table:lines.append(f"| {NAMES[x['group']]} | {x['physics']*100:.2f} | {x['frozen_student']*100:.2f} | {x['frozen_combination']*100:.2f} | {x['author_score_adapted']*100:.2f} |")
    lines+=['',
        '作者方法原生输出缺陷区域，而非原子缺陷点。最后一列使用完整方法的连续特征滤波分数，按同一预算取局部峰或高分区域，是明确标记的任务转接诊断；不能称为作者原生原子检测成绩。','',
        '我们五行使用不同的已选配置，且前两行组合为学生重排物理候选并补点，后三行为固定预算组合。选配置及组合分支参考过外层测试分数，属于事后开发最佳集合。作者权重训练了其他任务和数据，并有原作者预训练背景。预算一致并不意味着训练信息、选优机会和任务完全一致。','',
        '## 作者原生界面区域输出','',
        '原生区域覆盖不能直接与限定 10% 面积的结果比较。下面同时报告预测面积与预算缩减后的覆盖，防止大面积涂满造成虚假的高覆盖。','',
        '| 场景 | 原生覆盖率 | 原生平均预测面积 | 缩减至最多 10% 面积后的覆盖率 |',
        '| --- | ---: | ---: | ---: |']
    for x in interface:lines.append(f"| {NAMES[x['group']]} | {x['native_coverage']*100:.2f}% | {x['native_area_fraction']*100:.2f}% | {x['thinned_coverage']*100:.2f}% |")
    lines+=['','## 原任务完整分割对比','',
        '此处使用完整区域标签，指标是像素 micro IoU，不是上面的专家覆盖率。原作者附带 train.txt 与 test.txt：22 张缺陷图明确用于训练，6 张明确列为测试；另 3 张缺陷图及 30 张正常图的训练用途未列明。','',
        '| 范围与指标 | 作者预训练完整方法 | 历史物理组合 | 历史物理 score 分支 |',
        '| --- | ---: | ---: | ---: |']
    for title,subset,border in [('全部 61 张，任意缺陷 IoU','all',0),('作者测试清单 6 张，任意缺陷 IoU','author_test_list',0),('作者测试清单 6 张，裁边 12 px 的 IoU','author_test_list',12)]:
        lines.append(f"| {title} | {value('dennler',subset,'author_pretrained_full',border):.2f}% | {value('dennler',subset,'historical_physics_combined',border):.2f}% | {value('dennler',subset,'historical_physics_score',border):.2f}% |")
    lines+=['',f"全部 61 张的 Class-2 区域 IoU：作者完整模型 {value('dennler','all','author_pretrained_class2'):.2f}%，历史物理分支 {value('dennler','all','historical_physics_class2'):.2f}%。这纠正了此前仅用 Fourier/Bragg 代替完整作者模型所造成的过强结论。",'',
        '全部 61 张的成绩混有作者训练图，只作回溯描述；6 张明确测试图更适合讨论作者任务内留出效果，但不是新材料外推，也不是本项目从未使用过的全新盲测。我们的历史物理分支曾用于该数据集的开发，不能据此宣称严格公平未见测试的提升。','',
        '| 30 张正常图的平均误报面积 | 百分比 |','| --- | ---: |']
    for method,title in [('author_pretrained_full','作者完整模型'),('historical_physics_combined','历史物理组合'),('historical_physics_score','历史物理 score 分支')]:
        lines.append(f"| {title} | {normal_summary[method]['mean_false_positive_area']*100:.2f}% |")
    lines+=['','正常对照从 30 张原始预测独立重算。原 results.json 的正常对照字段因 normal／non_defective 名称不一致为空；补充核算见 comparison_audit.json，不改写原推理产物或已登记的代码。此问题不影响像素 IoU、专家覆盖或其他病例指标。','',
        '## 不同数据来源上的直接迁移','',
        f"FeCrAl 的 2 张图：作者完整方法任意缺陷 IoU 为 {value('fecral','all','author_pretrained_full'):.2f}%，历史物理 score 分支为 {value('fecral','all','historical_physics_score'):.2f}%。两者绝对成绩都低，且仅 2 张图，不能推断整个材料类别的泛化表现。",'',
        f"WSe₂ 的 107 张图：将作者区域分数转成同预算点位后，F1 为 {r['wse2_summary']['f1']*100:.2f}%，precision 为 {r['wse2_summary']['precision']*100:.2f}%，recall 为 {r['wse2_summary']['recall']*100:.2f}%。这反映该区域算法直接转接原子缺陷定位的局限，不代表其原任务无效。该来源此前已用于本项目开发，不重新称作全新盲测。",'',
        '## 权重来源与实现核验','',
        '- Dennler 权重来自 [作者 Zenodo 发布记录](https://zenodo.org/records/5792917)，134,380,387 字节，MD5 为 c130bb1ffecba7903213bb7d246065f5；旧的不完整下载保留，没有使用。',
        '- 完整算法对照 [作者代码](https://github.com/nkdnnlr/TEMDefectClassification)，固定提交 72d37c3ca5a84cfa39337c7243a80e3ffcc0f648。保留所有 13 个卷积层和 4 个全连接层，以及完整后处理，未用较小替代模型。',
        f"- 34 个参数张量严格加载；Keras 与 Torch 单精度概率最大差异为 {parity['probability_max_abs_difference']:.3g}，双精度逐层最大差异为 {findings['author_parameter_parity']['float64_layer_max_difference']:.3g}。Keras 使用 Torch 后端，此核验不是原版 TensorFlow 二进制运行环境的逐位复刻。",
        '- 12 个针对性测试通过，包括原作者参考块选择、滤波能量、边缘处理、Mahalanobis 距离及分割后处理的数值对照。',
        '- 非 128 整数倍的输入只向右／下反射补齐并裁回原视野，不缩放。原实现主要面向方形图，这种矩形处理是明示的适配。','',
        '## EMSeek 尚缺的资产','',
        '已核查 [官方仓库](https://github.com/PEESEgroup/EMSeek)、其 release 附件、作者 Hugging Face 模型列表，以及论文指向的 [Zenodo 发布记录](https://zenodo.org/records/18340509)。代码 ZIP 的 59 项和数据 ZIP 的 43,582 项目录未见任务权重；大数据 ZIP 仅远程读取中央目录，没有下载完整 3.8 GB 数据集。','',
        '官方配置需要 SegMentor/unet_ref.pth 或 task1.pth 至 task5.pth。项目此前已有的 EMSeek state.pt 是我们自己训练的网络，不是作者权重；DINO 通用权重也不能替代 EMSeek 完整任务模型。预训练 EMSeek 这一行保持“未评测”，需要作者可下载的真实检查点后才能完成。未向作者发送消息、未购买服务，也未用随机参数补位。','',
        '## 复现入口','',
        '- 最佳版本冻结入口：benchmarks/BEST_RELEASE_v1.yaml；审计说明：docs/best_release_audit_zh.md。',
        '- 作者推理协议及逐图输出：outputs/author_pretrained_20261014_v1/protocol.json、predictions/。',
        '- 原始指标及校验：results.json、seal.json；补充比较审计：comparison_audit.json。',
        '- 执行脚本：scripts/run_author_pretrained.py；权重加载与一致性脚本：scripts/dennler_author_weights.py。',
        '- 所有已有数据、标注和成功结果保留；本次没有删除、Git 提交或推送。','']
    dest=ROOT/'docs/author_pretrained_comparison_zh.md'
    with dest.open('x') as f:f.write('\n'.join(lines))
    write_new_json(output/'comparison_seal.json',{'files':{str(f.relative_to(ROOT)):sha256_file(f) for f in
        [output/'comparison_audit.json',dest,Path(__file__),ROOT/'vendor/author_pretrained_20261014_v1/adapter_tests_v2.xml',contained_path(ROOT,p['parity'])]}})
    print('REPORT',dest,flush=True);print(json.dumps(normal_summary,indent=2),flush=True)

if __name__=='__main__':main()
