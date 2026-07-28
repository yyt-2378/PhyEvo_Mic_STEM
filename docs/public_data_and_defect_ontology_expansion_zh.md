# 公开数据与缺陷类型扩展方案

版本：v1.0（2026-07-27）

> 实现状态更新：本文最后提出的三个最小任务已经在 open-world v0.4 中完成。实际 sealed
> plan、adapter、resolver、验收命令和剩余门禁见
> [`open_world_v0_4_implementation_zh.md`](open_world_v0_4_implementation_zh.md)。

## 先说结论

当前平台不是“少加几个类别”就能完善。真正的问题是：现有 4 个任务把**看到的现象**和
**推断的物理原因**绑在了一起。例如，原子柱变暗可能是空位、元素替换、厚度变化、失焦
或扫描伪影；局部周期性变差可能是非晶、束流损伤、漂移、弯曲或第二相。只增加
`vacancy / dislocation / interface / amorphous` 四五个平面类别，会制造更多看似精确、
实际证据不足的误判。

本次建议把平台改成“开放世界证据本体”：

```text
输入图像与辅助信息
        │
        ├── 观测：哪里违反了什么结构规律？
        │       occupancy / displacement / strain / phase-slip /
        │       periodicity-loss / diffraction-change / artifact ...
        │
        ├── 支撑：点、线、区域、界面还是整场？
        │
        ├── 候选诊断：空位样、位错样、晶界、相界、非晶、束流损伤……
        │
        └── 认识状态：confirmed / probable / composite /
                        unresolved / artifact / OOD
```

也就是说，平台可以先可靠地说“这里是一个带位移和应变证据的线状异常”，再根据原子图、
倒空间、时间序列、剂量或 EELS/EDS 把它升级为“位错”。没有足够辅助信息时保留
`unresolved`，这不是失败，而是科学上诚实的输出。

生产机器可读版本见
[`defect_ontology.yaml`](../schemas/defect_ontology.yaml)，完整设计参考保留在
[`defect_ontology_v2_draft.yaml`](../schemas/defect_ontology_v2_draft.yaml)；公开来源候选及许可状态见
[`public_dataset_candidate_registry_v1.yaml`](../datasets/intake/public_dataset_candidate_registry_v1.yaml)。

## 目前的数据覆盖缺口

当前正式 registry 中：

- 本地拉伸分子筛图覆盖局部非晶；
- Guo 数据覆盖 MoWTe₂ 点异常、Al 局部位错和 ZrO₂ 孪晶边界；
- EMSeek 只接入了 107 张 WSe₂ 图及作者点标注，而且来源关系不明，只能保守地视为一个
  train group；
- Dennler 是已经消费过的一次性外部测试，不能再用于训练、校准、主动学习或调阈值；
- `calibration` 仍为空，没有双专家 Gold positive + Gold negative，因此目前还不能声称
  得到了真实概率校准。

主要缺口是：

1. 原子位点只覆盖“点异常”，没有可靠拆分空位、替位、间隙/吸附原子、反位和复合缺陷。
2. 线缺陷把层错和位错合在一起，缺少位错核、晶界、孪晶、取向畴墙及界面位错。
3. 区域缺陷偏向完全非晶，缺少部分失序、短程有序变化、束流损伤和相变前沿。
4. 缺少化学、偏振、倒空间、剂量/时间等辅助证据。
5. 缺少失焦、漂移、污染、厚度/弯曲和探测器异常等“伪缺陷”负类。
6. 材料和实验室覆盖窄，无法检验跨晶体结构、跨仪器和跨采集条件的泛化。

## 建议的七个一级家族

| 一级家族 | 建议二级类型 | 最低需要的证据 |
|---|---|---|
| 原子位点异常 | 空位样、替位样、间隙/吸附原子样、反位样、点缺陷复合体 | 原子柱占位/强度；元素身份最好有化学信息或可信成像前向模型 |
| 晶格形变与拓扑 | 位错核/位错线、层错、局部应变、晶格转动、纳米孔洞 | 实空间相滑移或位移场；最好同时有应变/倒空间证据 |
| 畴、对称性与界面 | 晶界、孪晶界、异相界面、取向畴、铁电畴、第二对称性区域 | 界面两侧的结构/取向/衍射差异 |
| 有序度与相干性退化 | 局部非晶、部分失序、晶体-非晶界面、束流损伤、相变前沿 | 实空间周期性 + 倒空间或时间/剂量一致性 |
| 化学与谱学异常 | 掺杂/成分异常、界面偏析、氧化态/价态变化 | HAADF 化学对比不能单独定性，优先 EDS/EELS 或模拟支撑 |
| 颗粒、表面与形貌 | 台阶、颗粒-载体界面、析出/夹杂、晶面/形状异常、界面粗糙度 | 轮廓、表面结构和上下文；必要时多焦面/层析/ptychography |
| 采集伪影与质量 | 失焦、漂移/扫描畸变、振动重影、污染、厚度/弯曲、坏点/条纹、真空边缘 | 采集元数据、帧间一致性、正反扫或多模态对照 |

另设 `open_set_resolution`，用于保存未知结构异常、复合异常、OOD 和无效测量。它不是第八种
物理缺陷，而是平台拒绝过度解释的出口。

## 数据不能都用同一种方式接入

公开数据按用途分四档：

| 档位 | 含义 | 能做什么 | 不能做什么 |
|---|---|---|---|
| A | 已有 mask/框/点，接近可用 | 训练/开发 label adapter、验证标注接口 | 作者标签不能自动称为双专家 Gold |
| B | 原始数据丰富，带时间、剂量、坐标、衍射或模拟配对 | 专家重标、证据适配、多模态一致性研究 | 没有重标前不能当像素真值 |
| C | 大规模无标签真实图或正常对照 | 自监督特征、OOD、困难负样本、正常分布建模 | 不能直接计算缺陷分割精度 |
| D | 仿真或数字孪生 | 可控扰动、灵敏度曲线、原子真值、主动采集环境 | 不能代替跨实验室真实数据验证 |

这四档必须分别报告。否则仿真上的高 IoU、作者标签上的复现和双专家实验 Gold 会被错误地
混成一个数字。

## 已核实的公开来源

### 第一波：最值得立即接入

| 来源 | 新增覆盖 | 标签/证据 | 许可 | 建议 |
|---|---|---|---|---|
| [EMSeek](https://huggingface.co/datasets/gary23ai/EMSeek) | WSe₂ 点缺陷之外，还有 FeCrAl 辐照损伤、原子柱、纳米颗粒 | mask、框、点/提示 | Apache-2.0 | 冻结 revision 后按子集和 provenance 分开准入；跳过尚无数据的 single-atom-catalyst 占位目录 |
| [Guo 2021](https://zenodo.org/records/5520169) | MoWTe₂、Al 位错、ZrO₂ 孪晶和 SuperScan 时间信息 | 原图及 104.9 MB 电影 | CC BY 4.0 | 现有三张图继续留在 development；补接电影作为单独 acquisition group |
| [Al Σ3 孪晶界](https://zenodo.org/records/6414955) | 孪晶界、界面缺陷、disconnection | STEM + EBSD + 原子模拟 | CC BY 4.0 | 建实验/模拟关联组，避免同源信息跨分区 |
| [4D-STEM 点缺陷浓度](https://zenodo.org/records/7041997) | 辐照 Al、Au 热循环、应变和空位浓度 | 4D-STEM + calibration + 剂量信息 | CC BY 4.0 | 全库约 23.1 GB，先接约 839.5 MB 的辐照 Al 和校准文件 |
| [快速 4D-STEM 混相映射](https://zenodo.org/records/20366781) | TiO₂/SiO₂ 和 NMC/LGPS 晶体-非晶/混相界面 | 两个 H5 及处理代码 | CC BY 4.0 | 301.7 MB，适合率先建立 real-space + reciprocal-space 证据适配器 |
| [STEMGym](https://huggingface.co/datasets/johnpolat/STEMGym) | 空位、替位、晶界、相界、Pt/非晶碳颗粒 | 原子位置/种类、缺陷 mask、phase map | CC BY 4.0 | 只作为模拟 Gold 和受控压力测试，不能替代真实实验 Gold |

第一波没有把文件立即全部下载。原因不是数据越多越好，而是要先冻结版本、确认每个采集组、
规划磁盘用量并写 sealed intake plan。特别是 4D-STEM 数据可能从几百 MB 到几十 GB，
盲目整库下载只会产生无法审计的数据堆。

### 第二波：补足时间、畴、伪影和开放世界

| 来源 | 价值 | 当前门禁 |
|---|---|---|
| [Warwick Electron Microscopy Dataset](https://wrap.warwick.ac.uk/id/eprint/142120/) | 19,769 STEM、17,266 TEM 和 98,340 仿真 exit wave；适合自监督、OOD 和困难负样本 | CC BY 4.0；无缺陷 mask，且采集分组/元数据需审计 |
| [Graphene STEM Standard](https://doi.org/10.13139/ORNLNCCS/1773705) | 原子分辨序列，适合束流损伤、缺陷演化和时间一致性 | 主目录页未明确显示数据许可，先 quarantine |
| [Sm-BFO STEM](https://doi.org/10.13139/ORNLNCCS/1773704) | 原子坐标、偏振和成分序列，补铁电畴/相变 | 主目录页未明确显示数据许可，先 quarantine |
| [WS₂ 束流诱导缺陷追踪](https://www.nature.com/articles/s41524-019-0152-9) | 点缺陷、扩展缺陷、第二相形核和 100 帧演化 | 论文和代码公开不等于数据复用许可明确；算法 mask 不是 Gold |
| [受控剂量 SEI 束流伪影](https://zenodo.org/records/14555388) | 把束流伪影变成正式反例类，而不是归到“模糊/非晶” | CC BY 4.0；约 9.3 GB，先选小型剂量对照序列 |
| [Ayyubi CdTe/SrTiO₃](https://github.com/RAW-Ayyubi/defect-classification-in-stem-images) | 多图像类型、开放世界聚类、人工 vacancy/patch 对照 | 仓库未检测到明确 license；人工缺陷必须和实验缺陷分开 |

### 第三波：扩充尺度、表面和仪器域

| 来源 | 新增方向 | 许可与规模 |
|---|---|---|
| [层状材料大视场取向图](https://zenodo.org/records/3784335) | graphene/MoS₂ 晶界、取向、应变、多层区 | CC BY 4.0，约 114 GB；先看 82×60 小样例 |
| [高分辨 SrTiO₃ 4D-STEM](https://zenodo.org/records/5113449) | 原子柱和倒空间正常对照 | CC BY 4.0；不是缺陷标签集 |
| [MerlinEM 4D-STEM](https://zenodo.org/records/10813049) | 探测器/仪器域、4D 管线正常对照 | CC BY 4.0，约 8.5 GB |
| [金属-载体界面 ptychography](https://zenodo.org/records/19201546) | 颗粒-载体界面重构、表面原子排列 | CC BY 4.0，约 7.1 GB |
| [GAA 晶体管 3D 应变与粗糙度](https://zenodo.org/records/15882443) | 异质界面、应变、粗糙度、多模态一致性 | CC BY 4.0，约 18.2 GB |
| [TEM ImageNet v1.3](https://sites.google.com/view/temimagenet/home) | 多材料模拟原子坐标、分割、clean/noisy 对 | GitHub 仓库未检测到明确 license，保持 quarantine |

## 数据来源怎样映射到类别

不是每个数据集各建一套标签，而是通过 adapter 映射到同一证据本体。例如：

| 外部标签/数据 | 不应直接解释为 | 推荐映射 |
|---|---|---|
| EMSeek WSe₂ mask | 已确认 vacancy | `atomic_column_intensity_anomaly`，Silver；有额外证据再细分 |
| Dennler Class 1 | 一定是层错 | `secondary_symmetry_region`，且允许晶界/畴/相界多个候选 |
| Dennler Class 2 | 一定是非晶 | `amorphous OR defocus/blur OR beam-damage` 的 composite |
| STEMGym vacancy | 真实实验 Gold | `gold_simulation.vacancy_like`，单独报告 |
| 4D-STEM phase map | 原子像素分割 Gold | `diffraction-signature-change` 证据，加专家标注后形成界面 Gold |
| 时间序列中新出现的模糊区 | 非晶 | 先记 `temporal-change + coherence-loss`，结合剂量判断束流损伤 |

代码上现已新增 `dataset_label_adapter` 层：先保留原始标签和原作者语义，再产生平台统一
标签，两者都写 provenance，绝不原地覆盖。

## 具体执行顺序

### 阶段 0：先冻结规则，不碰测试集

1. v2 的七个物理家族和 `open_set_resolution` 已进入生产 ontology；以后以版本化方式更新。
2. 把本文看过标签说明的来源限定为 train/development 候选。
3. 另找一个尚未查看图像和标签的新实验室/新来源，在下载前只凭元数据冻结 checksum 和协议，
   作为未来真正的 locked external test。

### 阶段 1：六个第一波来源准入

1. 每个来源记录 DOI/Hub revision、license、bytes、SHA-256、material、lab、microscope、
   sample 和 acquisition。
2. 先做 metadata-only dry run，再选择性下载；禁止一开始下载 23–114 GB 的整库。
3. 同一电影、同一样品、同一模拟母结构的裁片不能跨分区。
4. 每个外部标签写 adapter 和 qualification：`author_curated / algorithmic / simulation_truth /
   expert_consensus`。

### 阶段 2：建立“证据路由器”，不再由检测分支决定类型

所有检测器只产生 EvidenceBundle。新 resolver 同时比较多个家族：

```text
候选区域
  ├─ 原子图：occupancy / intensity / displacement
  ├─ 晶格图：phase-slip / strain / rotation
  ├─ 有序度：periodicity / coherence / short-range order
  ├─ 倒空间：diffraction signature
  ├─ 时间轴：出现、持续、扩张、随剂量变化
  └─ 质量轴：defocus / drift / contamination / support
                    ↓
       多假设排序 + unresolved/artifact/OOD
```

这一步是方法学统一的核心：SAM 可以负责候选区域的几何细化，DINO 类特征可以负责跨材料
相似性和开放集聚类，但二者都不直接决定“这一定是 vacancy/非晶”。物理证据和辅助信息负责
诊断。

### 阶段 3：建立真正的 Gold calibration

从 train pool 中主动挑选“高不确定、高分歧、高覆盖价值”的候选，请至少两位材料/电镜专家
独立标注；同时标正常晶格、伪影和边界负类。建议第一版目标：

- 7 个一级家族全部有样本；
- 每个重要二级类型至少来自 2 个独立 source groups，达不到就标为 exploratory；
- 至少覆盖 5 类材料体系、3 个数据来源实验室、4 种成像/证据模态；
- calibration 同时含 Gold positive、Gold hard negative 和 Gold normal；
- 专家分歧原样保留为 `unresolved`，不使用多数票强行制造确定标签。

### 阶段 4：评价不只看一个 IoU

建议至少报告：

- 定位：mask IoU、boundary F1、point average precision；
- 类型：一级家族 macro-F1、二级类型 hierarchical F1；
- 开放集：unknown AUROC、OOD false-positive rate；
- 科学可靠性：risk–coverage、选择性准确率、校准 ECE/Brier；
- 跨域：leave-one-material-out、leave-one-lab/source-out；
- 物理一致性：时序一致性、实验/模拟一致性、实空间/倒空间一致性；
- 人效：每增加 100 个专家 Gold，平台性能提升多少。

不要把所有来源随机切 patch 后汇总一个数字。论文主表应以 **source-group 级跨域结果** 为主，
否则最容易发生相邻帧、同一样品和同一母结构泄漏。

## 六周可交付计划

| 周 | 交付物 |
|---|---|
| 第 1 周 | 六个第一波来源的 license/revision/容量核验与 sealed intake 草案 |
| 第 2 周 | EMSeek 子集、Guo movie、Σ3 数据 adapter；小规模可重复下载清单 |
| 第 3 周 | 4D-STEM real/reciprocal evidence adapter；STEMGym 模拟真值 adapter |
| 第 4 周 | 多家族 resolver、artifact/OOD 出口和 v2 EvidenceBundle |
| 第 5 周 | 专家审阅包：候选、上下文、证据、多个假设和辅助信息缺口 |
| 第 6 周 | 第一轮 Gold calibration、leave-one-source-out 基线和失败案例图册 |

## 本轮明确不做的事

- 不把已看过标签的数据重新命名为“全新 locked test”。
- 不因仓库可访问就默认有训练许可；`NOASSERTION` 一律 quarantine。
- 不把论文中算法生成的 mask 当专家 Gold。
- 不把仿真真值和实验 Gold 混报。
- 不把 4D-STEM、EELS/EDS 或时间序列强行压成普通 PNG 后丢掉元数据。
- 不批量下载几十到上百 GB 数据，直到 provenance、分组和磁盘预算都冻结。

## 三个最小任务的完成状态与下一步

原计划的三项任务均已在 v0.4 完成：

1. EMSeek 剩余真实子集、Guo SuperScan 和 Al Σ3 都有 checksum-sealed intake plan；
2. 通用 `dataset_label_adapter` 已覆盖 EMSeek、Guo、Al Σ3、Dennler 和 STEMGym；
3. 四个 detector 已输出 EvidenceBundle v2，多假设 resolver 支持 `artifact` 和
   `unresolved`。

下一轮最小任务转为：

1. 不整库下载 EMSeek，而是按材料/任务和磁盘预算选择首批 group，完成本地
   `--verify-files` 后再生成新 registry；
2. 解析 Sigma3 EMD detector metadata、Guo 时间轴和 dose/acquisition 信息，避免只依赖
   frame projection；
3. 用 adapter + resolver 生成首批跨来源 review pool，请两位专家复核 point/line/interface
   和 artifact hard negatives；
4. 建立真正的 Gold calibration，并将 resolver score/margin 从透明默认值升级为独立校准
   风险；
5. 再接入首个 4D-STEM real/reciprocal adapter，验证非晶/异相界面的双空间一致性。
