# 非 Human-loop 环节收口报告

日期：2026-07-27  
平台版本：v0.4  
原则：能由现有数据、确定性规则和自动测试完成的先完成；只有确实需要新专家判断、
新实验或外部实验室的事项才保留为 human/external gate。

## 结论

本轮把“数据放在本地、算法分别存在、benchmark 离线可跑”进一步收口为可复现的平台入口：

```text
versioned registry + per-asset manifest
                ↓
明确 channel / frame / task adapter
                ↓
单物理教师或双物理教师检测
                ↓
EvidenceBundle + open-world resolver
                ↓
point / contour / line topology / temporal support
                ↓
Silver / unresolved 候选和 human-review queue
```

没有训练新的神经网络，也没有用作者标签调当前推理阈值。新增双教师、拓扑和时序路径仍是
training-free；作者标注仍只作为 train-domain 开发参照，不是实验 Gold。

## 已经由代码完成

### 1. 生产数据入口升级为 registry v4

权威文件：
[`partition_registry_v4.yaml`](../datasets/partition_registry_v4.yaml)

| 分区 | 资产数 |
|---|---:|
| train | 280 |
| development_test | 10 |
| locked_external_test | 1 |
| calibration | 0 |
| 合计 | 291 |

registry SHA-256：

```text
51a4d07dff2e00c3923279e5b358ea8f6487cc27bbaf0254d86d162b41861395
```

新增正式资产包括：

- Guo SuperScan：1 个不可拆分的 100 帧 acquisition；
- Al Σ3：2 个多通道 EMD、EBSD、simulation 和 README，共 5 个 provenance group；
- EMSeek 均衡子集：16 个材料/场景组，每组按文件 ID 的 SHA-256 排序选 2 对
  image–annotation，共 32 对、64 个科学文件和 1 个选择 manifest。

均衡选择过程没有读取 label pixels。64 个下载文件共 65,770,693 bytes，全部按冻结的
Hugging Face LFS SHA-256 校验通过。完整 4.21 GB 远程范围仍由原 sealed plan 固定，
以后可以增量 admit；合并器已支持“完全相同的既有资产复用”，但仍拒绝内容跨分区。

### 2. 逐资产可执行 manifest

权威文件：

- [`sample_manifest_v3.jsonl`](../datasets/standardized/sample_manifest_v3.jsonl)
- [`sample_manifest_v3.summary.json`](../datasets/standardized/sample_manifest_v3.summary.json)
- [`sample_manifest.schema.json`](../schemas/sample_manifest.schema.json)

每个资产记录 registry/provenance、role、bytes、format、shape/dtype、container 内容概况、
channel/frame 结构、直接 detector task、task adapter、是否必须显式选择。

摘要：

| 项目 | 数量 |
|---|---:|
| image / annotation / metadata / container | 146 / 139 / 4 / 2 |
| 需要 EMD channel 选择 | 2 |
| 需要 frame/projection 选择 | 3 |
| 无需额外选择即可进入已有 detector | 129 |

### 3. EMD 与 NData 采集元数据

`inspect-emd` 现在自动解析每个 UUID channel 的：

- detector name/index；
- instrument model；
- acceleration voltage；
- illumination mode 与 camera length；
- pixel size/unit；
- metadata frame count；
- 首末帧 detector 是否一致。

Sigma3_ORI 自动识别 5 个通道：BF、HAADF、DF2、HAADF、DF4；Sigma3_ORII 也识别 5 个
通道。两者均为 Titan、300 kV。框架仍拒绝在多通道 EMD 中静默猜 channel。

### 4. 原子双物理教师正式入口

代码：
[`atomic_consensus.py`](../src/tfree_detection/atomic_consensus.py)

```bash
python -m tfree_detection.cli detect-atomic-consensus \
  --input image.png \
  --config configs/atomic_point_defect.yaml \
  --output outputs/atomic_consensus_run
```

两个独立教师分别是：

1. 晶格配准后的 presence/intensity/displacement/extra-peak residual；
2. 直接晶格平移的 periodic counterfactual motif disagreement。

一对一空间匹配后的状态：

- 两教师同意：`confirmed_candidate + Silver + probable`；
- 只有一个教师：保留为 `uncertain_candidate + unresolved`；
- 不读取标签，不产生 Gold，不把分歧候选丢掉。

真实 WSe₂ smoke 样本中：legacy 312 个、periodic 127 个、共识 117 个，另有
195/10 个单教师候选进入 unresolved。该结果用于证明统一入口可运行，不是新的独立泛化
分数；完整 train-domain benchmark 的既有共识 F1 仍为 0.7633。

### 5. 线/界面 topology，而不是 box 语义

代码：
[`line_topology.py`](../src/tfree_detection/line_topology.py)

```bash
python -m tfree_detection.cli detect \
  --input image.tif \
  --config configs/phase_boundary.yaml \
  --line-topology \
  --output outputs/interface_topology_run
```

它对已有 line/interface mask 计算 skeleton、端点、分叉、连通分量、近似长度、PCA
线性度和主轴方向，并给出 simple/curved/branching/fragmented topology state。它不改变
原 mask，也不声称已经得到 Burgers vector、stacking sequence 或 phase identity。

ZrO₂ 真实 smoke 输出：
[`zro2_phase_boundary_topology_v1`](../outputs/zro2_phase_boundary_topology_v1)
中检测到 1 个界面，topology 为 `simple_coherent_line`。

### 6. SuperScan 时序支持

代码：
[`temporal_support.py`](../src/tfree_detection/temporal_support.py)

```bash
python -m tfree_detection.cli assess-temporal-support \
  --input movie.ndata1 \
  --config configs/stacking_fault.yaml \
  --reference-frame-index 50 \
  --max-frames 9 \
  --output outputs/temporal_support_run
```

流程是逐帧 robust normalization、phase-correlation 亚像素漂移配准、异常漂移拒绝、
temporal median/MAD、瞬态 robust-z 和逐候选一致性。它只做辅助排序，不自动删除候选。

真实 SuperScan smoke：
[`superscan_temporal_support_v1`](../outputs/superscan_temporal_support_v1)。
5 个抽样帧中 4 个通过配准；frame 0 因约 241 px 的异常位移被拒绝。参考帧的 12 个线候选
均记录为跨已接受帧的 persistent appearance。这只能说明外观稳定，不能单独证明它们是真实
层错。

### 7. 公开来源运行状态与安全物化

- 19 来源运行状态：
  [`public_dataset_operational_status_v2.json`](../datasets/intake/public_dataset_operational_status_v2.json)
- 均衡子集 sealed plan：
  [`emseek_balanced_subset_train_v1.sealed.yaml`](../datasets/intake/emseek_balanced_subset_train_v1.sealed.yaml)
- 下载/校验报告：
  [`emseek_balanced_subset_v1_report.json`](../outputs/data_materialization/emseek_balanced_subset_v1_report.json)

物化器支持已有文件复验、`.part` 断点续传、Range/Content-Range 一致性检查、指数退避和
最终字节数/SHA-256 门禁。本轮首个远端 TLS 失败暴露并修复了这些缺口。

## 仍然必须由人或新外部证据完成

以下事项不是继续写脚本就能科学成立：

1. **实验 Gold calibration**：至少两位独立专家共识，且同时包含阳性与阴性 control；
2. **新 locked external test**：必须是从未参与 ontology、阈值或方法设计的新 acquisition；
3. **缺陷机制确认**：vacancy/substitution、Burgers vector、stacking sequence、phase
   identity 需要独立成像、谱学、原子图或材料专家证据；
4. **真实 multislice 对照**：需要厚度、取向、剂量、像差、探测器角度等可信采集参数；
5. **跨材料/跨实验室学生增益**：当前 student 代码与合成门禁存在，但不能用同一 train
   group 自证持续提升；
6. **显微镜在线 RL/bandit**：需要仪器接口、安全边界和 prospective acquisition。

## 下一步顺序

不应先扩大 RL。正确顺序是：

1. 把已封存的 100-candidate 双盲 review packet 交给两位专家；
2. 同时冻结一套全新的 calibration acquisition 和一套新 locked acquisition；
3. 用当前 registry v4 跑跨材料 coverage/OOD 与 failure taxonomy，不改 locked 阈值；
4. Gold 到位后才做 calibration、学生模型真实消融和 non-regression release；
5. 最后再进入在线采集。

## 自动验证

- `python -m compileall -q src scripts`：通过；
- `pytest -q`：88/88 通过；
- registry v4：291/291 文件字节数和 SHA-256 通过；
- balanced subset sealed plan：65/65 资产通过；
- 下载目录：无残留 `.part`；
- 原子 consensus、ZrO₂ topology、SuperScan temporal support 三条真实 smoke 路径通过。

本轮只删除过一个由失败下载明确产生、且大小已超过 sealed 期望值的中间文件：

```text
datasets/public/emseek_remaining_v1/nanoparticles/Au@Ge/image/22.png.part
```

它不是数据集资产；随后对应正式文件已重新下载并通过 SHA-256。没有删除或覆盖任何原始、
清洗后或已纳管数据。
