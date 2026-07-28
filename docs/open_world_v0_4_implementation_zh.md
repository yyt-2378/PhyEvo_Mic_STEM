# Open-world v0.4 实现与验收说明

版本：0.4.0（2026-07-27）

## 这次真正完成了什么

本轮把上一版“类别扩展草案”推进成了三条可运行链路：

```text
公开来源
  └─ checksum/source revision → sealed intake plan

外部论文标签
  └─ dataset_label_adapter → 原标签 + 统一本体 + 备选解释 + 资格等级

四个 training-free detector
  └─ candidate geometry + EvidenceBundle
       └─ 多 hypothesis energy
            └─ resolver → probable / artifact / unresolved
```

旧 detector 的二值 mask、score 和 legacy `defect_type` 保留。新增 resolver 不参与候选生成，
只解释现有候选，因此可以直接比较 v0.2/v0.3 定位结果，不会因类别体系变化而偷偷改变 IoU。

## 三个 sealed intake plan

### 1. EMSeek 剩余子集

封存文件：

- [`emseek_remaining_v1_source_manifest.json`](../datasets/intake/emseek_remaining_v1_source_manifest.json)
- [`emseek_remaining_train_v1.sealed.yaml`](../datasets/intake/emseek_remaining_train_v1.sealed.yaml)

固定 Hugging Face revision：

```text
8f4a987c78af743098459e5d467f2c1b5f5c11b3
```

范围：

- 8 个 atom-column 材料：DyScO₃、LiCoO₂、MoS₂、NiO、Pt、Si、SrTiO₃、graphene；
- FeCrAl irradiation damage；
- 7 个 nanoparticle 材料/载体组合；
- WSe₂ 明确排除，因为已经由旧 plan 单独准入；
- `single_atom_catalysts` 明确排除，因为该目录在固定 revision 中不存在。

封存统计：

| 项目 | 数值 |
|---|---:|
| 科学数据 groups | 16 |
| provenance manifest group | 1 |
| 远程 LFS 文件 | 28,334 |
| 计划内文件（含 manifest） | 28,335 |
| 远程总容量 | 4,197,421,991 bytes |
| bulk download | 否 |
| plan seal | `84e824ce473c48f07daca2f9649eaeb61d371d3a33a1cda8bc61327b48e96701` |

Hugging Face LFS oid 作为远程内容 SHA-256 冻结。当前 plan 可以验证 seal，但由于 4.20 GB
数据尚未下载，`admit` 的强制 `--verify-files` 会失败。这正是预期门禁：有计划不等于已经
准入。

### 2. Guo SuperScan

文件：

- `Recording_of_SuperScan_img_stack.ndata1`
- 104,862,360 bytes
- Zenodo MD5 已验证，本地 SHA-256 已写入 plan
- plan：
  [`guo_superscan_development_v1.sealed.yaml`](../datasets/intake/guo_superscan_development_v1.sealed.yaml)
- seal：
  `85e71a4d1f4c850b0b526a86d21268875da5862597b4f24e901e1acf8dae1096`

它被登记为一个不可拆分的 `development_test` acquisition，任务覆盖点异常、线缺陷和时间
演化。没有作者像素真值，因此 `annotation_qualification=none`，不能当 Gold。

### 3. Al Σ3

文件和 provenance 被拆成五组：

| group | 文件 | 角色 |
|---|---|---|
| `sigma3-orientation-i` | `Sigma3_ORI.emd` | 实验 STEM |
| `sigma3-orientation-ii` | `Sigma3_ORII.emd` | 实验 STEM |
| `sigma3-ebsd` | `EBSD_data.osc` | 取向辅助证据 |
| `sigma3-simulation` | `simulation.zip` | 独立 simulation truth 来源 |
| `sigma3-readme` | `README.md` | provenance |

总容量 278,438,842 bytes。每个文件同时通过 Zenodo MD5 和本地 SHA-256。plan：
[`al_sigma3_development_v1.sealed.yaml`](../datasets/intake/al_sigma3_development_v1.sealed.yaml)，
seal：
`b35496c32c6c0ba830bf0eb26d7b6e3de6c601ac6b9c1b23d3083232d2912031`。

实验数据和 simulation group 没有混在一起；simulation Gold 今后必须与 experimental Gold
分表报告。

三个 plan 都没有合并进 `partition_registry_v2.yaml`。因此当前正式训练/校准/测试统计没有
被静默改变。

## NData/EMD 多帧读取

准入后检查发现 Guo SuperScan 是 `100 × 512 × 512`，Sigma3 ORI/ORII 各包含 5 个 EMD
detector channels，每个 channel 又有 20–30 帧。旧加载器不能安全处理这类输入，v0.4 已
增加显式选择：

```bash
# 先列出 EMD 中所有 Data/Image/.../Data，不猜 detector channel
PYTHONPATH=src python -m tfree_detection inspect-emd \
  --input datasets/public/al_sigma3_twin_boundaries/Sigma3_ORI.emd

# 选择一个 channel 和一帧
PYTHONPATH=src python -m tfree_detection detect \
  --input datasets/public/al_sigma3_twin_boundaries/Sigma3_ORI.emd \
  --dataset-path 'Data/Image/<channel-id>/Data' \
  --frame-index 0 \
  --config configs/phase_boundary.yaml \
  --output outputs/sigma3_ori_channel_frame_v1

# 或显式做帧轴投影
PYTHONPATH=src python -m tfree_detection detect \
  --input datasets/public/guo_2021_atomic_defects/Recording_of_SuperScan_img_stack.ndata1 \
  --projection mean \
  --config configs/atomic_point_defect.yaml \
  --output outputs/guo_superscan_mean_v1
```

`frame-index` 与 `projection` 互斥。多 channel EMD 未给 `dataset-path`、多帧 stack 未给
frame/projection 时都会拒绝运行，避免把时间轴误当 RGB，或擅自挑一个探测器通道。实际
Sigma3 detector channel 的物理身份仍需解析 EMD metadata/论文说明，当前不根据 UUID 猜测。

真实文件 smoke test 已通过：

| 输入 | 路径选择 | 完整管线结果 |
|---|---|---|
| Guo SuperScan | frame 0，512×512 | atomic detector → 5 hypothesis maps → resolver |
| Sigma3 ORI | 第一个 channel 的 mean projection，256×256 | phase detector → 3 hypothesis maps → `interface_subtype_unresolved` |

这只证明 I/O 和端到端协议可运行。由于没有 Gold 且 Sigma3 channel 身份尚未确认，不构成
检测准确率或物理类别正确性的证据。

## dataset_label_adapter

实现位置：
[`dataset_label_adapter.py`](../src/tfree_detection/dataset_label_adapter.py)。

已注册五套 adapter：

| adapter | 关键语义保护 |
|---|---|
| EMSeek | WSe₂ mask → `atomic_column_intensity_anomaly` Silver，不直接叫 vacancy；particle mask 不是 defect mask |
| Guo | Al partial dislocation 保留为 dislocation/stacking-fault composite；SuperScan 是 unlabeled sequence |
| Al Σ3 | 实验边界为 Silver probable；simulation truth 单独为 simulation Gold |
| Dennler | Class 1 为 secondary-symmetry composite；Class 2 为 amorphous/blur/beam composite |
| STEMGym | vacancy/substitution/GB/phase truth 可以 hard train，但资格固定为 simulation Gold |

每条转换同时输出：

- 原始 source label 和 payload；
- adapter/version；
- semantic role；
- ontology family/label；
- alternative hypotheses；
- epistemic state；
- label tier 和 annotation qualification；
- hard-label eligibility；
- 缺少的辅助证据。

JSON Schema：
[`dataset_label_adapter.schema.json`](../schemas/dataset_label_adapter.schema.json)。

命令：

```bash
PYTHONPATH=src python -m tfree_detection label-adapters

PYTHONPATH=src python -m tfree_detection adapt-labels \
  --input external_labels.jsonl \
  --output adapted_labels.v1.jsonl
```

输出存在时拒绝覆盖，避免标签 provenance 被原地改写。

## EvidenceBundle v2 与 resolver

生产 ontology 已更新为
[`defect_ontology.yaml`](../schemas/defect_ontology.yaml)，EvidenceBundle protocol 升级为
2.0。

四个 detector 当前比较的 hypothesis：

| detector | 同时比较的解释 |
|---|---|
| amorphous | local amorphous、partial disorder、local strain/deformation、defocus/blur artifact |
| atomic | vacancy-like、substitution/intensity、interstitial/adatom、displaced column、broad fallback |
| stacking | stacking fault、dislocation、lattice bending/rotation、scan/coherence artifact、broad fallback |
| interface | heterophase、twin/orientation domain、broad interface fallback |

resolver 位置：
[`resolver.py`](../src/tfree_detection/resolver.py)。

它不会使用 detector 分支名作为类别，而是对每个候选计算：

- mean/p90 hypothesis energy；
- 区域 support fraction；
- 独立证据数量因子；
- top-1/top-2 margin；
- measurement reliability；
- physical 与 artifact hypothesis 的竞争。

新增候选字段：

```text
resolved_family
resolved_label
epistemic_state
hypothesis_ranking
missing_auxiliary_evidence
```

新增 decision code：

```text
artifact   = 5
unresolved = 6
```

典型行为：

- 原子存在和强度证据都能解释同一异常时，保留
  `atomic_site_subtype_unresolved`；
- 只有 domain signature transition、没有 symmetry/phase identity 时，保留
  `interface_subtype_unresolved`；
- sharpness/coherence artifact 解释明显超过物理解释时，输出 `artifact`；
- 物理解释达到可靠性和 margin 门槛时输出 `probable`，但仍是 Silver，不是 Gold。

## 数据准入状态新增

`AdmissionState` 增加 `development_candidate`。它只能包含 `development_test` group，并且
不能进入主动学习。这样 Guo/Σ3 这类已经影响过设计、但许可和 checksum 完整的数据，不必被
错误写成 train，也不必伪装成 quarantine。

## 验收命令

```bash
# 两个本地数据 plan：seal + 每个文件 SHA-256
PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/guo_superscan_development_v1.sealed.yaml \
  --verify-files

PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/al_sigma3_development_v1.sealed.yaml \
  --verify-files

# EMSeek：验证远程清单和 plan seal，不声称本地文件存在
PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/emseek_remaining_train_v1.sealed.yaml

# adapter + resolver 定向测试
PYTHONPATH=src pytest \
  tests/test_label_adapter.py tests/test_resolver.py tests/test_data_intake.py

# 四 detector 定位不退化
PYTHONPATH=src python scripts/check_baseline_regression.py
```

## 仍然没有完成的科学门禁

1. EMSeek 4.20 GB 还没有选择性下载和本地 checksum 复验，因此没有 admitted。
2. Guo/Σ3 虽已本地验证，但尚未生成专家 candidate-level 标注，仍不是 calibration Gold。
3. resolver threshold 是透明软件默认值，不是 Gold calibration 拟合结果。
4. `probable` 不能解释为真实概率；真实 ECE/Brier 仍需独立 calibration partition。
5. 化学异常需要 EELS/EDS 或可信 forward model；时间因果需要完整 dose/time metadata。
6. 需要用新实验室、新材料、未提前看标签的来源重新建立 locked external test。

因此 v0.4 的准确定位是：**开放世界类别和数据准入基础设施已可运行，科学校准和跨实验室
证明尚未完成。**
