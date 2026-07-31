# PhyEvo-Mic v0.4 实施状态与下一步实验入口

> 审计日期：2026-07-30
> 设计标准：
> [`self_improving_platform_roadmap_zh.md`](self_improving_platform_roadmap_zh.md)

## 2026-07-30：SAM2.1-large 真实 GPU 实例验证完成

平台已复用本机既有 `sam` 环境和 SAM2.1-large checkpoint，在实验分子筛局部非晶图与
公开 ZrO₂ 孪晶/两相界面图上完成真实 CUDA 推理，没有重新安装 PyTorch，也没有训练或
微调模型。新增 `run_sam2_boundary_worker.py`，复用候选级文件协议、Policy A 物理门和
逐候选回退。

分子筛 3 个候选产生 24 个 proposal，1 个通过、2 个因拓扑改变局部回退；ZrO₂ 使用
16 个成对晶域 prompts 形成 8 个界面 proposal，因物理核心召回仅 `0.593` 且拓扑改变
而回退。两次 worker 推理分别为 `9.92 s` 和 `8.53 s`，GPU 峰值均约 `1.31 GiB`。
这证明了真实模型路径和安全回退，不是准确率声明，因为两张图目前都没有独立 Gold 边界。
实例六列图、环境、DINO frozen 流程和复现命令见
[`sam2_live_and_dino_frozen_validation_zh.md`](sam2_live_and_dino_frozen_validation_zh.md)。
当前完整自动回归为 120 项通过。

## 2026-07-29：SAM 3 候选级物理约束边界层完成

平台已把 SAM 3 接成“可拒绝的几何提案器”，而不是新的物理 detector。当前实现包括：

- `stage2/sam3_refine.py`：候选级 point/box/coarse-mask ensemble、共识图和接受门；
- Policy A 硬约束：某候选核心召回 `<0.95` 时 proposal 保留供诊断，只回退该候选；
- v2 SHA-256 文件协议：worker 保存 candidate/role/ensemble 独立 proposal，不再整图 OR；
- 两相/孪晶界使用 domain A/domain B → shared boundary，不把界面当填充区域；
- stable core、possible extent、boundary uncertainty、最终几何和六面板报告；
- `final_geometry_mask` 已接入候选表、主 overlay、expert review 和 Boundary
  F1/HD95/ASSD 比较入口；
- live worker 记录 runtime/peak RSS/peak device memory，assisted/blinded review 可统计
  几何修正率和专家时间；
- 仅对 `amorphous_region`、`phase_boundary` 开放，默认关闭；
- 11 项 SAM 定向测试、5 项边界评估测试和两类 prepare-only 端到端运行。

参考非晶图产生 3 个 coarse 候选和 24 组物理 prompts；公开 ZrO₂ 图产生 1 个 coarse
界面候选和 16 组 paired-domain prompts。数字孪生门控验证为 1 个候选通过、1 个候选
局部回退，Boundary F1 从 0.584 提升到 1.000。因本机没有隔离 SAM 3 环境与授权
checkpoint，真实图状态如实保持 `proposal_pending`，该数字孪生结果没有被声称为真实
SAM 精度。实现、输出解释和未完成的 Gold 边界校准见
[`sam3_boundary_refinement_zh.md`](sam3_boundary_refinement_zh.md)。
当前完整自动回归为 116 项通过。

## 2026-07-29：公开实验 4D-STEM 第二波完成准入

当前生产 registry 已升级为 v6：313 个资产全部通过 SHA-256，分区为
`train=280`、`development_test=32`、`locked_external_test=1`、`calibration=0`。
逐资产接口已升级为
[`sample_manifest_v7.jsonl`](../datasets/standardized/sample_manifest_v7.jsonl)：
155 个读取 adapter ready、141 个可直接进入 detector，3 个需显式选择、2 个需派生
4D 物理图、7 个仍需来源专用 adapter。
NMC/LGPS、WSe₂ raw 1657 + 作者坐标、SrTiO₃ 原子分辨正常对照、辐照 Al + calibration
和聚合物室温 4D-STEM 已完成下载、固定校验和、acquisition-safe sealed intake 和容器
清单。TiO₂/NMC 两个 acquisition 已共同完成无训练界面 smoke test。
加入 SAM 3 边界层之前的完整自动回归为 103 项通过。

新增数据扩展了原子缺陷、界面、正常/OOD、辐照和取向物理覆盖，但没有改变
`calibration=0` 的投稿阻塞；原始 MIB/DM4/WSe₂ reconstruction 仍需各自流式 adapter，
不能把“已下载”写成“二维 detector 已验证”。详见
[`public_stem_dataset_expansion_2026_07_29_zh.md`](public_stem_dataset_expansion_2026_07_29_zh.md)。

## 2026-07-28：Class1 群关系任务完成

平台新增第五个 training-free 任务 `symmetry_relation`：自动多参考 patch、平移/旋转/
镜像/子晶格交换群对齐、五物理教师和 region/interface/line/point 四几何输出。Dennler
Class1 的 61 图事后重测 61/61 成功；最终保守引导扩展 IoU 从 `0.1251` 提高到
`0.1891`，F1 从 `0.2224` 提高到 `0.3181`，Recall 从 `0.2162` 提高到 `0.5338`，
正常图平均误报面积从 `3.5067%` 变为 `3.5150%`。

位错芯 v1 曾因端点判据错误产生 `0.2346` 的假高 IoU，已在 v2 修复；修复后芯点区域
IoU `0.0022`，明确保留为需要原子图 Burgers circuit 验证的 proxy。完整方法、图、seal、
限制和复现命令见
[`class1_symmetry_relation_final_zh.md`](class1_symmetry_relation_final_zh.md)，失败迭代见
[`class1_failed_iterations_archive_zh.md`](class1_failed_iterations_archive_zh.md)。

Dennler 已揭盲，以上仍不能用作新的 zero-shot 或独立外测主张。

## 2026-07-28：Dennler v0.4 事后诊断重测

在完整保留 v0.3 locked 负结果的前提下，项目新增了独立的尺度感知 coherence-loss
诊断分支，并按“触边候选保留、边缘可靠性降低”的规则重测 61 张 Dennler 图。61/61
预测成功；Class 2 IoU 从 `0.0118` 升到 `0.6108`、Recall 从 `0.0118` 升到
`0.9683`。保守的任意缺陷组合 IoU 为 `0.6186`。

局部波矢晶域实验分支没有通过正常对照门控，因此 v2 没有把它并入主掩膜：Class 1 主结果
仍为历史 IoU `0.1251`，实验分支作为 ablation 完整保留。这说明 Class 2 的主要欠覆盖
机制已经被修复，但 Class 1/结构边界尚未解决。

Dennler 标签此前已经揭盲，以上只能称为 post-hoc diagnostic，不能重新包装成 zero-shot
或新的 locked external validation。方法、完整数值、逐例图、seal 和复现命令见
[`dennler_posthoc_v4_diagnostic_zh.md`](dennler_posthoc_v4_diagnostic_zh.md)。

## 2026-07-27：可由代码完成的环节已继续前推

本节覆盖本文后续较早的状态描述。当前已新增：

- `partition_registry_v5.yaml`：298 个校验资产；SuperScan、Al Σ3、EMSeek
  16 场景均衡子集和首个实验 4D-STEM acquisition 均已纳管；
- `sample_manifest_v4.jsonl`：逐资产格式、shape/dtype、EMD/HDF5 channel/axes、stack selection
  与直接可运行任务；
- 正式 `detect-atomic-consensus` 入口：lattice residual 与 periodic counterfactual
  共识候选为 Silver，单教师分歧保留为 unresolved；
- 可选 `--line-topology`：线/界面骨架、端点、分叉、长度、线性度与方向，不改变 mask；
- `assess-temporal-support`：stack 漂移校正、时间 MAD/瞬态 z/一致性和逐候选时序状态，
  不做硬过滤；
- SuperScan 与 ZrO₂ 的真实 smoke outputs，以及 19 个公开来源的可机读运行状态。

完整边界和复现命令见
[`non_human_loop_completion_status_zh.md`](non_human_loop_completion_status_zh.md)。
数据资产 v1–v4、平台 v0.2–v0.4 和测试收益的同口径图表见
[`version_iteration_comparison_zh.md`](version_iteration_comparison_zh.md)。
仍不能由代码替代的核心是：独立实验 Gold、两位专家共识、全新未消费 locked test、
真实 multislice/采集参数验证、跨实验室效益和显微镜在线闭环。

## 结论

路线图要求的软件骨架已经从“四个共享 CLI 的独立 detector”迭代为一个可审计的
physics-teacher → evidence-student → active-review → calibration/release 闭环。

但“平台代码完成”和“Nature 级方法学证据完成”不是一回事。当前自动审计结果为：

```text
软件协议与安全门禁：已完成
自动测试：116 项通过（本文后续的 79 项是新增模块前的历史快照）
四任务冻结性能保留：100%
证据学生合成消融：通过（IoU +0.061）
WSe₂ 双物理教师开发域：site F1 0.7633（不是独立外测）
独立 Gold calibration：0 个资产
locked external test：1 个归档已完成预注册盲测，任意缺陷 IoU 0.1396
双专家前瞻性 pilot：100 候选/69 图已封存，尚未收到专家 response
跨材料/跨实验室学生增益：尚未验证
```

因此当前可以发布为研究框架 v0.4 和实验协议，但不能声称已实现真实世界持续增精或已达到
投稿终点。

## 已实现的统一架构

```text
Input + SHA-256 provenance
        │
        ▼
Frozen v0.2 physics teacher
  ├─ amorphous_region
  ├─ atomic_point_defect
  ├─ stacking_fault
  └─ phase_boundary
        │
        ▼
EvidenceBundle v2
  ├─ node: occupancy/intensity/displacement
  ├─ edge: phase/strain/rotation/coherence
  ├─ region: Fourier/Patterson/domain/specimen
  ├─ field: acquisition/time/dose context
  ├─ q_k(x) reliability + valid mask
  └─ violation × support × family × epistemic state
        │
        ▼
Multi-hypothesis resolver
  probable / artifact / unresolved + ranked alternatives
        │
        ├──────────────► candidate_table.jsonl + append-only annotation ledger
        │
        ▼
Evidence-preserving student
  ├─ geometry head
  ├─ violation head
  ├─ evidence head
  └─ reliability/OOD heads
        │
        ▼
Train-only active selection
  uncertainty + disagreement + OOD + diversity + physics − cost
        │
        ▼
Two-reviewer consensus
  Gold / Unresolved / artifact + annotation time
        │
        ▼
Gold-only calibration + OOD evaluation
        │
        ▼
Locked-test non-regression gate
  accept release OR audited rejection
```

## Phase 0–1：基线与统一证据

已完成：

- 冻结
  [`baseline_v0_2_manifest.json`](../benchmarks/baseline_v0_2_manifest.json)；
- `EvidenceBundle`、`ReliabilityBundle`、`DefectHypothesis`；
- 节点—边—区域—场字段和四层开放世界缺陷本体；
- reliability、invalid、OOD、artifact、unresolved 与 legacy confidence 分离；
- Gold/Silver/Unresolved 与来源规则；
- 四任务多假设能量、resolver ranking 和候选证据向量；
- 输出独立证据可靠性与候选表。

基线门禁要求每项 IoU、precision、recall 至少保留 95%；当前全部为 100%。

## Phase 1.5：原子 motif 反事实与双教师共识

新增的周期反事实教师不依赖亮原子峰能否正确归属到子晶格，而是从 FFT 估计直接晶格
\(a,b\)，比较 \(\pm a,\pm b,\pm(b-a)\) 平移前后的局部 motif。六个方向使用最小
disagreement 聚合，要求候选与所有周期对应位置都不一致。它与冻结 v0.2 位点配准教师按
物理尺度容差形成双教师共识；匹配规则为“容差内最大基数优先，再最小定位距离”，所有
单教师候选均保留为 Unresolved。

在 107 张 EMSeek WSe₂ train-domain 图像上：

| 分支 | Precision | Recall | F1 |
|---|---:|---:|---:|
| v0.2 位点配准 | 0.4895 | 0.9226 | 0.6396 |
| 周期反事实 | 0.7367 | 0.6874 | 0.7112 |
| 双教师共识 v2 | 0.8914 | 0.6674 | 0.7633 |

共识层产生 12,001 个 Silver 候选，另外 2,957 个 periodic-only 和 18,212 个
legacy-only 候选进入主动审阅池。该结果只证明同一 train group 内的教师互补性：阈值
`q=0.997` 是开发选择，数据没有独立正常对照、calibration 或新材料 test；图像 56、66、
71 仍为零共识。因此这一阶段的软件能力已实现，跨域科学门禁仍未通过。

方法、选择性风险和哈希见
[`periodic_counterfactual_and_consensus_zh.md`](periodic_counterfactual_and_consensus_zh.md)。

## Phase 2：数据、校准与 OOD

软件完成：

- train/calibration/development/locked 四分区；
- revision-pinned 外部下载、逐文件 SHA-256 和禁止覆盖；
- prospective intake plan、license/标注资质/group provenance 准入；
- acquisition/sample/group 泄漏检查；
- locked test 禁止主动学习；
- Gold 禁止被弱标签覆盖；
- candidate、配置、像素和教师版本 provenance；
- temperature/isotonic、Brier/ECE、risk–coverage、split conformal；
- train-reference + calibration-threshold Mahalanobis OOD；
- data/model/domain uncertainty 分离；
- 模型 lineage 与联合非退化发布门禁。

当前注册状态：

| 分区 | 当前文件资产数 | split groups |
|---|---:|---:|
| train | 215 | 1 |
| calibration | 0 | 0 |
| development_test | 4 | 4 |
| locked_external_test | 1 | 1 |

train 中的 215 个文件资产来自一个保守 EMSeek WSe₂ group：107 图像、107 作者标注和 1 个
source manifest。只有 107 张图像具有 active-learning eligibility。公开数据说明使用
consensus annotations，但 reviewer 数未公开，且 acquisition/sample 关系无法可靠还原；
因此平台将其登记为 `author_curated` training-only，而不是 Gold，也没有从同一 group
切出 calibration/test。

当前 registry 是
[`partition_registry_v2.yaml`](../datasets/partition_registry_v2.yaml)。被 Dennler 协议
封存的 v1 [`partition_registry.yaml`](../datasets/partition_registry.yaml) 继续保持
不变，确保已消费外测可复核。

已有本地图和公开 Guo 数据都已经影响过开发，所以没有伪装成 locked test。另行下载的
Dennler III–V HRSTEM 数据在任何 detector 运行前按 SHA-256 冻结为一个完整 split group；
归档含 31 张专家像素标签 defect 图和 30 张 normal 图。协议先冻结任务映射、三个主终点、
配置、代码、失败处理和指标，再生成并封存 183/183 个预测，最后才读取标签。

锁定结果没有支持预期的零样本普适性：非晶/模糊/束损伤复合区 IoU 为 0.0118，
次级晶体对称性联合 IoU 为 0.1251，任意专家缺陷 IoU 为 0.1396。该负结果完整保留，
Dennler 资产也永久退出训练、校准和阈值选择。单一 JEOL ARM200F/协作来源本来也不等于
跨实验室验证；以后改进必须使用新的开发数据，并另行寻找全新 locked test。

详见
[`locked_external_test_result_zh.md`](locked_external_test_result_zh.md)。

## Phase 3：证据学生

学生模型和目标构造已完成。早期证据多头发生负迁移；一次看似为正的结果又因可靠性语义
错误被作废。平台保留全部失败/作废记录。修正可靠性，并让几何头显式融合
violation/evidence/reliability 预测后，在 3 个未见旋转/噪声合成条件上：

| 模型 | IoU | Dice | Precision | Recall |
|---|---:|---:|---:|---:|
| Evidence-preserving | 0.6648 | 0.7987 | 0.7912 | 0.8063 |
| Hard pseudo label | 0.6037 | 0.7528 | 0.6268 | 0.9423 |

当前结果是修正语义后的合成软件门禁，不是跨材料门禁。真实 WSe₂ train group 已经准入，
但 evidence student 尚未在它上面重新训练；在没有独立 Gold calibration 和新 locked
test 之前，不能用同组作者标签完成“训练—调参—证明提升”的闭环，也不能继续根据当前
3 个 held-out synthetic cases 调权重。

## Phase 4：主动学习和专家协议

已实现：

- 六因素 rank aggregation；
- 专家时间预算与最大候选数约束；
- 非 train 分区自动排除并计数；
- point/line/region/interface 的低成本复核动作；
- 可离线打开的 HTML crop/overlay 审阅界面；
- 点、线、轮廓点击修正和原图坐标导出；
- blinded bundle 不包含 overlay、teacher score、预测类型或 evidence vector；
- 导出 JSON 的 assistance/action/geometry 完整性验证；
- assisted/blinded 条件记录；
- 至少两位独立 reviewer 一致才生成 Gold；
- 同一 reviewer 对同一 candidate 的重复投票自动拒绝；
- disagreement 自动保留为 Unresolved；
- cycle 状态机禁止跳过标注、校准、评估直接发布。

当前公开二维 STEM 队列另已实现正式在线入口：200 个已有像素标签样本被硬排除；354 个
无标签平面合并为 154 个病例；全部病例主视图和354个平面都有预计算预测。Streamlit 支持
六类结论和 point/line/interface/region 几何，SQLite 以“一个当前结果＋完整不可变历史”
持久化，支持断点续审、当前/历史 JSONL 下载和 expert-verified Silver 训练导出。真实账本
仍为0条，平台完成不等于专家工作完成。详见
[`expert_review_platform_2d_zh.md`](expert_review_platform_2d_zh.md)。

基于 WSe₂ 双教师结果，平台已经 label-blind 地抽取并封存 100 个候选，覆盖 69 张图：
25 个共识点、25 个 periodic-only、25 个 legacy-only 和 25 个随机 sham prompts。
assisted/blinded bundle 使用相同 opaque candidate ID；盲包不含分支、score、预测类型或
evidence。协议 seal 为
`9ee5e5f0beda1d38574a164444aeeb10fe2a8df243afef8b2e94ee77e4894125`。

但尚未收到任何真实专家 response，私有分层 key 也必须继续隐藏。仍需至少两位 STEM 专家按
冻结分配执行，记录标注时间、接受/拒绝/修正次数、专家一致性和模型提示偏差；在此之前
Phase 4 科学门禁仍为未完成。

演示和使用协议见
[`expert_review_workflow_zh.md`](expert_review_workflow_zh.md)。其中 blinded 只隐藏模型
输出，候选 crop 仍由算法选择；正式研究若评价候选发现偏差，还必须加入独立随机窗口。

## Phase 5：反事实数字孪生

四任务都能生成 normal/defect 配对，固定：

- 晶格角度；
- 照明；
- Gaussian/剂量噪声参数；
- 模糊；
- 同一耦合随机实现；
- 联合强度归一化。

只改变声明的结构干预。当前生成器用于验证学习逻辑，不具备 multislice 的厚度、像差、
探测器角度和真实元素散射，因此不能作为实验精度证据。

## Phase 6：为什么仍未实现完整 RL

路线图明确要求在真实前瞻性循环前不做完整 RL。当前 rank aggregation 更容易消融，也不会
把 physics teacher 偏差直接变成奖励。contextual bandit 只有在积累多轮“选择—专家时间—
独立验证增益”记录后才有可识别奖励；自主采集还需要显微镜接口、剂量约束和硬件合作。

## 使用和验收命令

```bash
# 全部自动测试
mkdir -p .codex-tmp
PYTHONPATH=src pytest --basetemp .codex-tmp/pytest-manual-run

# 冻结 detector 非退化
PYTHONPATH=src python scripts/check_baseline_regression.py

# 数据分区与泄漏审计
PYTHONPATH=src python -m tfree_detection validate-registry \
  --registry datasets/partition_registry_v2.yaml \
  --project-root .

# sealed 外部数据准入
PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/emseek_wse2_train_v1.sealed.yaml \
  --verify-files

# 查看统一 ontology
PYTHONPATH=src python -m tfree_detection ontology

# 学生 vs hard pseudo label 合成消融
PYTHONPATH=src python benchmarks/run_student_ablation.py

# 路线图阶段审计
PYTHONPATH=src python scripts/audit_roadmap_status.py
```

## 真正解除投稿阻塞的下一批工作

这部分不能靠继续写代码伪造完成，需要实验数据和专家协作：

1. 新采集或获得带 acquisition/sample provenance 的数据，建立独立 Gold calibration；
2. 前瞻性锁定一个从未参与开发的新材料/新仪器/新实验室集合；
3. 由两位材料/STEM 专家独立复核首批 50–100 个候选；
4. 在同一专家时间预算下比较 random、entropy、diversity、disagreement；
5. 在未见材料上复验 evidence student 是否仍优于 hard pseudo label；
6. 把解析 twin 升级为带真实成像参数的 multislice counterfactual pairs；
7. 全部通过 accuracy、calibration、risk–coverage、OOD 和 locked hash 门禁后才升级模型；
8. 再决定是否进入 contextual bandit 或显微镜自主采集。

Dennler v1 已经完成揭盲，不能再充当改进后模型的“新”外测；第 2 项必须寻找另一批从未
查看标签的数据。当前 WSe₂ 结果已经证明高召回位点教师与高精度周期教师可以互补，但它
仍是一个训练组内部的开发结果。下一优先级不是继续堆新 backbone，而是获取可按采集单元
切分的 Gold calibration/normal controls，执行双专家 pilot，并用全新 locked test 检验
这种选择性共识是否跨尺度和材料成立。

这也是目前最诚实、最短的“从可运行框架到投稿方法”的路径。

## 2026-07-27：类别与公开数据扩展设计（随后已实现）

针对当前四任务类别不足的问题，项目先新增了一版设计参考：

- [`public_data_and_defect_ontology_expansion_zh.md`](public_data_and_defect_ontology_expansion_zh.md)
  说明七个缺陷家族、公开来源分级、数据—标签 adapter 和六周执行计划；
- [`defect_ontology_v2_draft.yaml`](../schemas/defect_ontology_v2_draft.yaml) 把观测违反、
  空间/时间支撑、候选物理诊断和认识状态拆开，共登记 8 个家族（含开放集出口）和
  42 个二级标签；
- [`public_dataset_candidate_registry_v1.yaml`](../datasets/intake/public_dataset_candidate_registry_v1.yaml)
  登记 19 个公开来源候选，包含许可、数据规模、模态、可覆盖缺陷和准入门禁。

这三项目前是“设计和候选目录”，不是已经下载或训练完成的数据扩展。生产 ontology、
`partition_registry_v2.yaml` 和现有 benchmark 均未改动；许可不明的来源仍处于
quarantine。下一实现步骤是为 EMSeek 剩余子集、Guo SuperScan 和 Al Σ3 数据建立 sealed
intake plan，再实现统一的 `dataset_label_adapter`。

## 2026-07-27：Open-world v0.4 已实现

上述“下一实现步骤”现已完成，详细验收见
[`open_world_v0_4_implementation_zh.md`](open_world_v0_4_implementation_zh.md)：

- EMSeek 剩余 16 个科学子集已冻结 28,334 个 LFS SHA-256，未批量下载 4.20 GB；
- Guo SuperScan 与 Al Σ3 已下载、通过 MD5 + SHA-256，并分别建立可
  `--verify-files` 的 development sealed plan；
- 已实现五套 provenance-preserving `dataset_label_adapter`；
- EvidenceBundle 升级为 v2，四 detector 同时评分多个物理/伪影解释；
- resolver 已写入 `probable / artifact / unresolved` 和完整 hypothesis ranking；
- 旧 mask、score 和 legacy `defect_type` 保留用于回归。

`partition_registry_v2.yaml` 仍未改写，三个新 plan 尚未正式 admit。Gold calibration 和全新
locked external test 仍是投稿门禁。

## 2026-07-27：跨数据集固定样本 smoke test

已用同一套 open-world v0.4 输出协议实际运行 6 个固定样本，覆盖分子筛、EMSeek WSe₂、
Guo Al 位错、Guo ZrO₂ twin、Guo SuperScan stack 和 Al Σ3 EMD。总览、逐图解释、
WSe₂ 作者点参考匹配及可复现命令见
[`open_world_dataset_gallery_zh.md`](open_world_dataset_gallery_zh.md)。

这次没有按输出挑图，也没有进行模型训练。结果暴露出两个需要优先解决的问题：

- WSe₂ 固定图 reference recall 为 0.958、precision 为 0.513，当前单一位点教师明显过检；
- Al 位错几何候选被 resolver 判成 `artifact`，说明线缺陷 subtype 还缺拓扑、位移场和
  多帧一致性证据。

ZrO₂ 与 Σ3 均保持 `unresolved`，后者因 EMD UUID channel 语义未知只算 I/O smoke test。
这些结果不会改写既有 benchmark，也不能代替 Gold calibration 或新的 locked external
test。

可视化随后按材料解释习惯修正：原子位点继续用小圆圈，非晶区、线缺陷和相界面改为真实
candidate mask 轮廓，不再使用可能覆盖大片正常晶区的 bounding box。当前公开数据的文件
校验、sealed plan、生产 registry、标签 adapter 和剩余缺口见
[`public_dataset_standardization_status_zh.md`](public_dataset_standardization_status_zh.md)。
