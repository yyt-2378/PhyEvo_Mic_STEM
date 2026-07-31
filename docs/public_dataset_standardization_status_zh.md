# 公开数据集标准化整理状态

审计日期：2026-07-27

> 2026-07-29 更新：当前权威入口为
> [`partition_registry_v6.yaml`](../datasets/partition_registry_v6.yaml)（313 个资产）、
> [`sample_manifest_v7.jsonl`](../datasets/standardized/sample_manifest_v7.jsonl) 和
> [`public_dataset_operational_status_v6.json`](../datasets/intake/public_dataset_operational_status_v6.json)。
> v7 为每个资产增加了非破坏式 `tfree-data-interface-v1`；当前数据总表、接口状态和
> adapter 队列见
> [当前数据资产总结与统一接口](data_collection_and_interface_standardization_zh.md)。
> NMC/LGPS、WSe₂ raw 1657、SrTiO₃、辐照 Al 和聚合物 4D-STEM 的实际下载、容器清单、
> 标准化和测试见
> [第二波公开实验 STEM 报告](public_stem_dataset_expansion_2026_07_29_zh.md)。

> 2026-07-28 更新：最新权威入口已经升级为
> [`partition_registry_v5.yaml`](../datasets/partition_registry_v5.yaml)（298 个资产）和
> [`sample_manifest_v4.jsonl`](../datasets/standardized/sample_manifest_v4.jsonl)。
> 新增实验 4D-STEM、WSe₂ 三维坐标、21 来源检索和实际测试结果见
> [公开实验 STEM 数据扩展报告](public_stem_dataset_expansion_2026_07_28_zh.md)。

> 最新覆盖说明（同日 v0.4 收口）：本文后半部分保留了 registry v2 时点的历史审计，
> 其中“SuperScan/Σ3 尚未 admit”“EMD channel 未解析”“没有逐样本 manifest”已经过时。
> 当前权威入口是
> [`partition_registry_v4.yaml`](../datasets/partition_registry_v4.yaml) 和
> [`sample_manifest_v3.jsonl`](../datasets/standardized/sample_manifest_v3.jsonl)。

最新状态：

- registry v4 共 291 个资产，SHA-256 全部复验通过：`train=280`、
  `development_test=10`、`locked_external_test=1`、`calibration=0`；
- SuperScan 1 个 100 帧 acquisition 与 Al Σ3 的 5 个 provenance group 已正式纳管；
- 291 条逐资产 manifest 已记录 role、shape、dtype、container、任务、分区和显式选择要求；
- 两个 Σ3 EMD 已自动解析 BF/HAADF/DF2/DF4、Titan、300 kV、pixel size、
  camera length，并验证 detector 在 stack 首末帧一致；
- 19 个候选来源的非历史化运行状态见
  [`public_dataset_operational_status_v2.json`](../datasets/intake/public_dataset_operational_status_v2.json)；
- EMSeek 16 个材料/场景组已用文件 ID 哈希无标签选出 32 对均衡子集，64 个科学文件
  （65,770,693 bytes）全部下载校验并正式纳管；其余远程范围仍未整体下载，
  实验 Gold calibration 仍为 0。

## 一句话结论

目前已经完成了“来源目录 → 固定版本/校验和 → 分组与分区 → 标签 adapter → 统一
EvidenceBundle”的治理骨架；WSe₂ 和既有 Guo 静态图已经进入生产 registry，SuperScan 与
Al Σ3 已在本地完整校验并封存为 staging plan。

但是，这还不能叫“公开数据集库已经完整建成”：

- EMSeek 剩余 4.21 GB 目前只有远程文件清单，图像尚未批量下载；
- SuperScan 和 Al Σ3 尚未正式合并进生产 registry；
- calibration 分区仍为 0；
- 公开作者标注没有被冒充为实验 Gold；
- EMD channel、采集条件和部分 source label 语义仍待人工补齐。

比较准确的表述是：**治理层基本成形，文件准入层完成一半以上，实验 Gold 和跨来源
benchmark 层仍是主要短板。**

## 生产 registry：已经正式纳管的部分

权威文件：
[`partition_registry_v2.yaml`](../datasets/partition_registry_v2.yaml)。

本次重新执行文件级 SHA-256 验证，220 个资产全部通过。registry hash 为：

```text
cb4cf6241c8f2381c47c4fc4bedc952da7b68c6bca28d8ac86c8bff30d441f73
```

| 分区 | 资产数 | 当前含义 |
|---|---:|---|
| `train` | 215 | EMSeek WSe₂：107 图 + 107 作者点标签 + 1 source manifest |
| `development_test` | 4 | 本地分子筛 1 个 + Guo 静态图 3 个 |
| `locked_external_test` | 1 | Dennler 完整 archive，已经揭盲消费，只能保留历史结果 |
| `calibration` | 0 | 当前最重要的缺口 |

当前 registry 中的数据集：

| dataset_id | 资产数 | 状态 |
|---|---:|---|
| `emseek_wse2_atomic_defects` | 215 | 已纳管；全部文件校验通过 |
| `guo_2021_atomic_defects` | 3 | MoWTe₂、Al partial dislocation、ZrO₂ twin 已纳管 |
| `dennler_2021_hrstem_defects` | 1 | consumed locked test，不得回流训练或调参 |
| `local_zeolite_reference` | 1 | development reference |

生产 registry 使用 acquisition/sample/lab 的显式 group 作为 split 单元，而不是把同一张大图
切成 patch 后随机分配。这解决了最危险的相邻 patch 泄漏问题。

## 已封存、已校验，但尚未正式 admit

### Guo SuperScan

计划：
[`guo_superscan_development_v1.sealed.yaml`](../datasets/intake/guo_superscan_development_v1.sealed.yaml)

| 项目 | 状态 |
|---|---|
| groups / files | 1 / 1 |
| 大小 | 104,862,360 bytes |
| 本地文件 | 完整 |
| checksum + seal | 已验证 |
| 目标分区 | `development_test` |
| annotation qualification | `none` |
| 生产 registry | 尚未 admit |

整个 movie 被保留为一个不可拆分 acquisition，避免把不同帧随机拆进不同分区。它可以用于
帧间一致性、扫描伪影和缺陷演化研究，但没有候选级 Gold。

### Al Σ3 twin boundaries

计划：
[`al_sigma3_development_v1.sealed.yaml`](../datasets/intake/al_sigma3_development_v1.sealed.yaml)

| 项目 | 状态 |
|---|---|
| groups / files | 5 / 5 |
| 大小 | 278,438,842 bytes |
| 本地文件 | 完整 |
| checksum + seal | 已验证 |
| 目标分区 | 5 个 `development_test` groups |
| annotation qualification | 4 个 `none`，1 个 `simulation_truth` |
| 生产 registry | 尚未 admit |

实验 EMD、EBSD 和 simulation 被拆成关联但独立的 provenance groups；simulation truth 不会
与 experimental Gold 混报。目前最大的具体问题是 EMD 内部 UUID channel 尚未映射到
HAADF/ABF 等真实探测器语义。

## 已有 sealed manifest，但图像还不在本地

### EMSeek remaining

计划：
[`emseek_remaining_train_v1.sealed.yaml`](../datasets/intake/emseek_remaining_train_v1.sealed.yaml)

| 项目 | 状态 |
|---|---|
| science groups | 16 |
| 总 groups | 17（含 provenance manifest） |
| 文件 | 28,335 |
| 期望大小 | 4,214,705,597 bytes |
| 本地存在 | 1 个 manifest |
| 本地缺失 | 28,334 个远程 LFS 文件 |
| plan seal | 有效 |
| 全文件校验 | 未执行、也无法通过 |
| 生产 registry | 未 admit |

已覆盖的计划范围包括 FeCrAl irradiation、8 个 atom-column 材料和 7 个 nanoparticle
组合。这里“28,335 个文件已标准化”的准确含义是远程 revision、LFS SHA-256、文件角色和
分组已经冻结，并不表示图像已经下载。

下一步有两个合规选择：

1. 按 sealed plan 下载完整 4.21 GB，完成文件校验后整体 admit；
2. 先定义一个新的、材料均衡的小规模 sealed subset plan，再选择性下载。

不能只随手下载几张效果好的图然后沿用原 plan，这会破坏预注册和可复现性。

## 标签标准化已经做到哪里

代码：
[`dataset_label_adapter.py`](../src/tfree_detection/dataset_label_adapter.py)；
协议：
[`dataset_label_adapter.schema.json`](../schemas/dataset_label_adapter.schema.json)。

目前有 5 套 source-specific adapter：

- `emseek-v1`
- `guo-2021-v1`
- `al-sigma3-v1`
- `dennler-2021-v1`
- `stemgym-v1`

adapter 不会覆盖原作者标签，而是同时保存：

```text
original source label
        +
semantic role
ontology family / label
alternative hypotheses
spatial support
Gold / Silver / Unresolved tier
required auxiliary evidence
hard-label-training eligibility
```

几个关键安全规则已经落实：

- EMSeek 点标注只能升为 Silver 原子异常定位，不能直接宣称 vacancy Gold；
- Guo Al 文件名只能给出 dislocation/stacking-fault composite，仍需要 Burgers 或专家 trace；
- Dennler class 2 保留为 blur/amorphous/beam-damage composite；
- simulation Gold 与 experimental Gold 分开；
- 无法解释的 source label 进入 `unknown_structural_anomaly`，不强行塞入四个旧类别。

## 19 个公开来源候选的现状

候选目录：
[`public_dataset_candidate_registry_v1.yaml`](../datasets/intake/public_dataset_candidate_registry_v1.yaml)。

目录共 19 个来源：

| catalog 状态 | 数量 |
|---|---:|
| `candidate` | 7 |
| `candidate_large` | 4 |
| `quarantine_license` | 5 |
| `partially_ingested` | 2 |
| `consumed_locked_external_test` | 1 |

其中 5 个许可不清楚的来源仍在 quarantine，没有因为“数据可能有用”就下载进正式
dataset tree。

需要注意：这个 v1 候选目录早于后续 sealed intake 实现，其中关于“SuperScan 尚未
ingest”和“Σ3 仍只是 candidate”的 `local_state/next_action` 已经过时。为了保持历史审计，
没有在本次状态问答中改写旧目录；当前真实状态应以 sealed plan、生产 registry 和本文为准。
下一版应建立 `public_dataset_candidate_registry_v1_1.yaml`，显式记录
`catalogued → sealed → files_verified → admitted` 状态机。

## 文件格式统一与“标准化”的边界

当前采用的是非破坏式标准化：保留论文作者原文件，通过统一 loader 和 provenance 描述读取，
不把 TIFF、NData、EMD 强行转存后覆盖原始数据。

已支持：

- PNG / TIFF 二维图；
- NData 1 stack 的显式 `frame_index` 或 projection；
- EMD 的 dataset 枚举、显式 UUID channel 和 projection；
- 输入文件 SHA-256、字节数、原始 dtype/shape 和选帧参数记录；
- 所有 detector 输出统一为 EvidenceBundle v2 和 candidate table。

尚未完成：

- 通用的逐图 `sample_manifest.parquet/jsonl`；
- 全部来源统一的像素尺寸、剂量、dwell time、探测器和晶向字段；
- EMD UUID channel 的物理语义映射；
- OME-Zarr 等大规模流式缓存层；
- 多模态 EBSD–STEM 坐标配准。

因此目前是“统一访问与统一证据协议”，还不是把所有公开数据物理重编码成一个完全同构的
数据库。

## 下一步最应该做的四件事

1. ~~建一个新的 `partition_registry_v3`，正式 admit 已验证的 SuperScan 和 Al Σ3；~~
   已完成，并继续升级为当前 `partition_registry_v4`；
2. 为 EMSeek remaining 先冻结材料均衡的小规模 subset，而不是直接用 4.21 GB 做无目标
   bulk download；
3. 生成逐样本标准 manifest，补齐 pixel size、detector、dose/time、material、orientation
   和 label semantics；
4. 单独寻找 acquisition-aware 的实验 Gold calibration 和全新 locked test。当前
   `calibration=0`，这是比继续扩充候选目录更严重的投稿阻塞。

## 本次审计命令

```bash
PYTHONPATH=src python -m tfree_detection validate-registry \
  --registry datasets/partition_registry_v2.yaml --project-root .

PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/emseek_wse2_train_v1.sealed.yaml --verify-files

PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/guo_superscan_development_v1.sealed.yaml --verify-files

PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/al_sigma3_development_v1.sealed.yaml --verify-files

PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/emseek_remaining_train_v1.sealed.yaml
```

前三个本地 plan 和生产 registry 均通过文件验证；EMSeek remaining 最后一条只验证 seal，
没有宣称远程文件已在本地。
