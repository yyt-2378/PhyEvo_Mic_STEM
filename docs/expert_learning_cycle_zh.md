# 已有专家标注怎样进入 PhyEvo-Mic 闭环

本轮不再要求专家标注。154 个正式病例已封存；其中 153 例具有可用的正标注或已接受预测，
1 例拒绝预测保留作审计，不被擅自变成整图无缺陷标签。原始数据库、修订历史、原图和原教师预测不覆盖。

## 本轮真正完成了什么

1. 校验标注、原图像素、预测及源文件哈希，生成独立、不可覆盖的学习版本。
2. 专家点位独立于教师有效区获得监督；没有圈出的地方仍是未知。
3. 51 例闭合界面圈解释为界面范围，不能当成精确中心线或沿外框的细线。
4. 原 `EvidencePreservingStudent` 的几何分支进入真实训练，物理证据仍由原教师提供。
   本轮没有相应监督的 evidence/reliability/OOD 辅助头置零并冻结，不输出虚构的物理或可靠性结论。
5. 使用整采集组留出验证：5 折、2 个学生训练分支、3 个随机种子，共 30 个留出模型；
   另训练 3 个使用全部可用标注的模型供实验性推理，总计 33 次 GPU 训练。
6. 每个模型训练 1,200 步，batch=32、原像素 patch=128、base channels=16；
   本轮使用空闲 H100 GPU 2–7，并未使用运行其他任务的 GPU 0–1。
7. 保存权重、Adam 状态、NumPy/Torch 随机状态，支持恢复；权重使用不含 pickle 的 NPZ。
8. 接入 CLI/API 的可选学生分支，单独输出模型来源和候选，不覆盖原教师 mask 或改变默认行为。
9. 闭环状态机记录训练、探索性评估和发布拒绝：不把缺少独立校准的单专家辅助标注包装成生产级验证。

这不是“训练脚本跑完就默认升级”。本轮的实际决定是：**保留原框架默认行为，学生作为可显式启用的实验性候选分支。**

## 标签怎样使用

| 已有结果 | 使用方式 |
|---|---|
| 专家修正点位 | 半径 3 px 的局部正监督；邻近未知区不被教师背景拉回零 |
| 闭合界面圈 | 多边形内部为弱监督范围，不虚构中心线 |
| 区域圈 | 圈内正监督；圈外不自动成为人工负样本 |
| 接受预测 | 保留经过认可的教师正标签，监督权重与人工修正区分 |
| 明确无缺陷 | 才允许整图成为对应任务的人工负监督；本批没有这类记录 |
| 拒绝、伪影、无法判断 | 不擅自生成负标签；本批拒绝病例不进入训练 |

教师非空预测的背景只作为有明确来源的弱伪标签先验，不称为专家确认无缺陷。
教师完全没有预测时，不使用其全零掩膜制造背景真值。不同图像的监督单独归一化，
防止宽界面标注的像素数量淹没少量原子点。

本轮仅使用专家实际查看并保存坐标的平面，不自动把标注复制到其他通道。
病例上的多个缺陷名称不能自动变成每个点的具体化学或缺陷亚型真值。

## 数据划分和结论范围

原审核批次在旧注册表中是 `development_test`，不会静默修改为 `train`。
新学习版本中的 `admission` 明确记录：用户授权本轮学习、原用途退役、原注册表不变、
这些训练过或消费过的来源不能继续称作学生独立测试集。锁定测试及校准分区禁止这样转用。

两个点位来源组和三个界面来源组分别整组留出。SuperScan 的 100 帧仍只是一个采集组，
不能随机切帧后声称有 100 个独立样品。区域只有一个来源组，无法验证跨组泛化；线缺陷只有拒绝记录，
维持原物理分支。两位专家分工标注也不等于同一病例的双专家共识。

本轮结果是 **模型辅助单专家标签上的开发性分组交叉验证**，不是 Gold 校准、盲法评估或锁定外部验证。
没有穷尽负标注，所以不报告虚假的完整 precision/F1。点位报告固定候选预算下已标点找回率；
界面报告固定 10% 候选面积下专家范围覆盖率。

## 实际结果：三随机种子平均

| 留出采集组 | 仅学教师 | 教师＋专家 | 指标 |
|---|---:|---:|---|
| SuperScan 整段序列 | 13.2% | 16.1% | 已标点找回率，每 512×512 最多 50 个候选 |
| MoWTe2 单视野 | 14.3% | 9.5% | 同上；只有 7 个明确标注点 |
| ZrO2 单视野 | 14.3% | 0.9% | 固定 10% 候选面积的界面范围覆盖率 |
| Al Σ3 / ORI | 0.3% | 3.5% | 同上，仍低于随机面积覆盖期望 10% |
| Al Σ3 / ORII | 9.4% | 24.6% | 同上 |

这里比较的是两个学生训练分支，不要把“仅学教师学生”的指标误写成原物理教师的直接指标。
SuperScan 加入专家后不同种子的结果约 5.6%–22.2%，波动明显，不能宣称稳健提升。
ORII 有正向信号，但不能推广成全材料成功，ZrO2 的退化必须保留。

### 呼应动机的现象

- **教师无输出不等于无缺陷**：21 例教师为空但专家画出了缺陷，其中点位 4 例、界面 17 例。
- **监督语义比标签数量更重要**：808 个明确点位中 563 个距边缘不到 24 px。
  专家点位与原教师掩膜零重叠不能直接解读成原模型召回率为零，也可能反映补充漏检式标注。
- **学习不能自动等于升级**：不同材料、几何和随机种子表现差异明显。
  数据产生新学生，验证结果决定保留原教师还是允许升级，发布门禁是闭环的一部分。
- **不能只看“有提升”**：加入空间随机和单调边缘位置对照；发现单调对照偏向最外缘后，
  又补充了均匀 24/32 px 边缘条带的事后敏感性分析，未据此重新训练或调参。
  SuperScan 的 24 px 条带对照约 7.2%，学生约 16.1%；这支持继续研究，但不消除跨组和种子不稳定性。
- **排序改善不等于置信度可靠**：界面组中分数超过 0.5 的平均像素比例随来源与种子约为
  74.5%–99.2%。因此实验性推理不用 0.5 生成几乎全图阳性的掩膜，而与验证一致，展示
  得分最高的 10% 候选范围。这个固定预算也可能在正常图像上产生候选，不能据此判断缺陷是否存在。

## 入口和文件

本地结果不进入 Git：

- 当前报告：`outputs/expert_cycle_001/report_release/PhyEvo_Mic_专家标注学习闭环报告.pdf`
- 当前必要图：同目录 `01_annotation_evidence.png`、`02_group_cv_comparison.png`、
  `03_held_out_points.png`、`04_held_out_interfaces.png`
- 正式输入：`outputs/expert_cycle_001/frozen_round/`
- 学习数据：`outputs/expert_cycle_001/learning_dataset_v1/`
- 三种子实验：`outputs/expert_cycle_001/student_gpu_s23_v1/`、`student_gpu_s41_v1/`、`student_gpu_s67_v1/`
- 可用模型之一：`outputs/expert_cycle_001/student_gpu_s23_v1/all_expert_labels/step-001200/`
- 每次实验的 `protocol.json`、`cycle_summary.json`、`cycle_lineage.json`、`release_decision.json` 均保留。
- 报告内 `training_producer_sources.zip` 保存与训练协议哈希一致的实际源码；不是只记一个无法恢复的哈希。

`report_v1` / `report_v2` 是字体渲染未通过检查的中间报告，`report_final` 有示例标题重叠；
均保留其数值与源码快照，不作为当前展示入口。以 `report_release` 为准。
原稿件、benchmark 和标注均没有清理或覆盖。

## 重现与恢复

依赖固定在 `environments/expert-learning-cuda124.txt`，本轮使用项目原有的 `.venvs/sam2` 隔离环境。
不更改系统 CUDA/驱动，不做全局安装。安装日志提示该环境原有的可选 `python-docx`/`diptest`
依赖不完整，本轮训练不使用它们，未扩大任务去修改这些无关组件。

生成**新的**数据版本时才执行（已有目录拒绝覆盖）：

```bash
PYTHONPATH=src .venvs/sam2/bin/python scripts/run_expert_learning_cycle.py prepare \
  --frozen-round outputs/expert_cycle_001/frozen_round \
  --output outputs/your_new_cycle/learning_dataset \
  --retire-development
```

复现 seed 23 的训练配置。使用新的输出目录；先检查 GPU 空闲情况，调度器会拒绝占用中的设备：

```bash
PYTHONPATH=src .venvs/sam2/bin/python scripts/run_expert_learning_cycle.py train \
  --dataset outputs/expert_cycle_001/learning_dataset_v1 \
  --output outputs/your_new_cycle/student_s23 \
  --seed 23 --steps 1200 --batch-size 32 --patch-size 128 --base-channels 16 \
  --checkpoint-every 400 --threads 2 --device cuda:0 --gpus 2,3
```

中断恢复时，对**同一命令、同一输出、同一代码和数据**追加 `--resume`。
不删除旧 checkpoint，不从头重复训练。若协议、代码或数据不同，恢复会拒绝，应新建实验版本。

显式启用实验性学生，不改变原来的普通 `detect` 命令默认行为：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src .venvs/sam2/bin/python -m tfree_detection.cli detect \
  --input datasets/public/guo_2021_atomic_defects/Recording_of_SuperScan_img_stack.ndata1 \
  --frame-index 60 --config configs/atomic_point_defect.yaml \
  --student-checkpoint outputs/expert_cycle_001/student_gpu_s23_v1/all_expert_labels/step-001200 \
  --allow-experimental-student --student-device cuda:0 \
  --output outputs/your_new_shadow_inference
```

原教师结果仍是原文件；学生结果在 `expert_student_shadow/`，组合入口为
`expert_assisted_summary.json`。点候选采用固定数量预算，界面/区域采用固定 10% 面积预算；
它们是未校准的候选排序，即使没有缺陷也可能列出候选，
不能把“列出了点”解释为“已经确认存在缺陷”。全标签模型在已训练病例上的演示不是留出验证。

学习记录按用户要求保持 `deferred`，不要求补做学习问答或专家审核。

## 最终验收与改动清单

当前唯一入口清单为 `outputs/expert_cycle_001/CURRENT_EXPERT_CYCLE.json`。
它绑定数据、模型、报告及验收凭据，不依据文件名中的 `final` 猜测最新版本。

- 数据保全：154 例正式最新审核与冻结版本的 revision ID / payload hash 相同，
  157 条历史保存事件及 1 条未选用的旧 reviewer 当前记录均保留；7 个原始源文件校验一致。
- 训练核验：33 个模型均在 GPU 上完成 1,200 步；重新核算全部 912 个留出病例预测记录，
  与保存的点位命中/界面范围覆盖数值一致。这是重复的模型—病例预测记录数，不是 912 个独立病例。
- 恢复核验：实际重新执行 seed 23 的 `--resume`，11 个模型恢复成功且未重复优化；
  409 个已有权重、评估、协议与训练历史文件哈希不变。
- 推理核验：病例 63 的点位分支和病例 135 的界面分支使用原始输入运行；
  原物理 mask 与审核时保存的 mask 一致，学生结果单独输出。该演示使用全标签模型，不能当泛化指标。
- 代码验证：13 个相关测试文件，77 passed，无跳过、无失败；有 14 个已有依赖弃用警告。
  未运行与本次改动无关的旧稿件/PPT 整库测试。
- PDF 检查：6 页，已检查中文、数字、图例及示例行标题；Poppler 有字体类型警告，
  实际渲染和中文文本提取正常。未修改系统字体或全局环境。

本轮代码和文档改动（原先已有的其他工作区改动保留，不混入本轮归属）：

| 用途 | 文件 |
|---|---|
| 标注冻结、数据转换、训练、汇报 | `scripts/freeze_expert_round.py`、`scripts/run_expert_learning_cycle.py`、`scripts/report_expert_learning_cycle.py` |
| 监督语义、学生训练、实验性推理 | `src/tfree_detection/expert_supervision.py`、`src/tfree_detection/expert_student.py`、`src/tfree_detection/expert_runtime.py` |
| CLI 接口和闭环状态 | `src/tfree_detection/cli.py`、`src/tfree_detection/cycle.py` |
| 回归测试 | `tests/test_freeze_expert_round.py`、`tests/test_expert_supervision.py`、`tests/test_expert_student.py`、`tests/test_expert_cycle_runner.py`、`tests/test_expert_cycle_report.py`、`tests/test_expert_runtime.py`、`tests/test_cycle.py` |
| 环境版本 | `environments/expert-learning-cuda124.txt` |
| 文档与 deferred 记录 | `README.md`、`docs/index_zh.md`、本文、`docs/learning/index.md`、`docs/learning/entries/2026-10-13-expert-supervision-independent-of-teacher.md` |

没有删除文件、修改原标注或重启审核服务，也没有提交或推送 Git。
本轮没有新专家共识、独立 Gold 校准或独立锁定学生测试；生产替换被明确拒绝，
这是已完成实验的发布结论，不是要求专家再来标注。
