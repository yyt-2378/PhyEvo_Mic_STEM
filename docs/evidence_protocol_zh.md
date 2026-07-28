# 统一结构证据与开放世界解析协议 v2.0

当前四个 training-free 检测器的掩膜和数值算法保持不变，但输出已通过
`EvidenceBundle` 统一为节点—边—区域—场四级结构证据。v2 在 EvidenceBundle 后新增
multi-hypothesis resolver：任务分支只负责产生候选几何和证据，不再直接决定最终物理亚型。
这一层把“测到了什么”“该测量是否有效”“有哪些物理解释”“为什么暂时不能确认”分开记录。

## 1. 四种尺度

| 尺度 | 结构对象 | 当前证据 |
|---|---|---|
| node | 原子柱或期望晶格位点 | presence、intensity、displacement、extra peak |
| edge | 晶格连接及局部相位场 | phase jump、strain、rotation、coherence |
| region | 周期结构和晶畴 | Fourier/Patterson、sharpness、domain、specimen support |
| field | 整幅图像或时空采集场 | 全场漂移、探测器/扫描状态、时间和剂量上下文 |

每一个证据场均包含：

- `values`：实际测量或异常响应；
- `reliability`：该证据在每个像素的可信程度，范围为 0–1；
- `valid_mask`：该物理量是否可以在该位置解释；
- `source_key`：它来自哪个原始任务证据图；
- `scale × violation_type × role × direction`：统一物理语义。

## 2. 缺陷本体

缺陷不再只是互不相关的类名，而由四层表达：

```text
观测违反类型 × 空间支撑 × 候选物理家族 × 认识状态
```

例如：

- 原子空位：`occupancy × point`；
- 层错/位错：`phase-slip × line`；
- 局部非晶：`periodicity-loss/coherence-loss × region`；
- 两相或孪晶界面：`domain-change × interface`。

同一证据可以支持多个解释。例如 `domain-change × interface` 可以同时支持异相界面和
孪晶/取向畴边界；没有 symmetry indexing 时，resolver 输出
`interface_subtype_unresolved`，不会由 `phase_boundary` 分支名强行决定类别。

机器可读定义见
[`defect_ontology.yaml`](../schemas/defect_ontology.yaml)，协议清单的 JSON Schema
见 [`evidence_bundle.schema.json`](../schemas/evidence_bundle.schema.json)。

## 3. 可靠性、无效区和拒答

`confidence` 暂时保留用于向后兼容，但它只是当前图像内的相对排序值，不能解释为真实正确
概率。平台新增独立的可靠性和决策状态：

| 状态 | 含义 |
|---|---|
| `confirmed_candidate` | 物理教师内部证据一致且测量可靠；仍不是 Gold 或校准概率 |
| `uncertain_candidate` | 是候选，但证据可靠性或独立支持不足 |
| `out_of_domain` | OOD 模块判定超出已验证域 |
| `invalid_measurement` | 边界、真空、配准失败或低相干使物理量不可解释 |
| `artifact` | 采集退化解释超过当前可支持的物理缺陷解释 |
| `unresolved` | 多个物理假设太接近，或没有假设达到透明门槛 |
| `background` | 有效测量范围内未被当前教师选为候选 |

当前可靠性来源为：

- 局部非晶：Fourier 有效边界、完整窗口上下文及向内部渐增的可靠性；`support_count`
  仅表示超过阈值的异常证据组数，不再误作测量有效性；
- 原子点缺陷：非刚性晶格配准可靠性；
- 层错/位错：相位有效边界与局部晶格相干度；
- 两相界面：界面有效区与晶体样品支持；
- GPA 第二阶段：额外使用 `gpa_validity × gpa_lattice_coherence`。

这些可靠性仍是物理测量质量，不是标签正确率。概率校准必须使用独立 Gold calibration
partition。

## 4. 标签等级和来源

物理教师只能自动产生：

- `silver`：可靠、内部一致的 physics pseudo candidate；
- `unresolved`：需要主动送审的冲突、低可靠或无效候选。

物理教师不能创建 `gold`。Gold 只能来自专家共识或可验证模拟真值，且不能被伪标签覆盖。
每个候选均记录 `candidate_id`、教师版本、任务、来源种类、质量标记和候选证据向量。

## 5. 多假设能量与 resolver

每个任务现在注册多个稀疏、非负、可审计的解释。例如原子位点分支同时比较
`vacancy_like / substitution_or_intensity_anomaly / interstitial_or_adatom_like /
displaced_atom_column`；非晶分支同时比较非晶、部分失序、局部应变和失焦/模糊。
统一引擎对证据进行鲁棒变换，并按局部可靠性门控：

\[
E_h(x)=
\frac{\sum_k w_{hk}\log(1+\rho(e_k(x)))q_k(x)}
{\sum_k w_{hk}q_k(x)}
\]

只有满足最少独立证据数的位置才产生能量。resolver 在每个候选区域汇总 mean/p90 energy，
显式记录证据数量因子，并比较最低 score、第一/第二解释 margin、区域 support fraction、
measurement reliability 和 artifact hypothesis。

resolver 输出 `resolved_family / resolved_label / epistemic_state /
hypothesis_ranking / missing_auxiliary_evidence`。`probable` 仍只是 Silver；当空位与替位、
层错与位错、孪晶与异相界面不能区分时，合法输出是 `unresolved`。最终二值候选掩膜仍由
冻结的 task-specific detector 给出，所以分类语义扩展不会静默改变定位基线。

## 6. 输出文件

`save_result(...)` 新增：

- `evidence_protocol.json`：证据字段、可靠性摘要、假设和 provenance；
- `reliability_map.npy`：任务级测量可靠性；
- `decision_map.npy`：状态编码图；
- `hypothesis_maps.npz`：统一假设能量；
- `evidence_reliability_maps.npz`：每项证据各自的可靠性。
- `candidate_table.jsonl`：同时保存 legacy `defect_type` 和 resolver 的开放世界诊断。

完整数组与 JSON 清单分开保存，避免把大数组嵌入不可读的报告。

## 7. 基线门禁

冻结基线见
[`baseline_v0_2_manifest.json`](../benchmarks/baseline_v0_2_manifest.json)。运行：

```bash
PYTHONPATH=src python scripts/check_baseline_regression.py
```

会重新生成四项确定性合成任务，并逐项检查 IoU、precision 和 recall 是否至少保留原值的
95%。统一证据 v2 和 resolver 不改变旧掩膜，因此定位基线应当是 100% 保留。resolver 只
允许改变候选的解释、认识状态和送审优先级。

## 8. 外部数据标签适配

不同论文的标签不能直接拼接。统一入口
`tfree_detection.dataset_label_adapter` 会同时保存：

- 原始 `source_label`、原始 payload 和 annotation qualification；
- 统一 family/label、备选假设和缺少的辅助证据；
- `author_curated / simulation_truth / expert_consensus` 等来源等级；
- 是否允许作为 hard-label training target。

例如 EMSeek WSe₂ mask 只映射为 Silver
`atomic_column_intensity_anomaly`，不会自动变成 vacancy；Dennler Class 2 保留为
`amorphous OR blur OR beam-damage` composite；STEMGym vacancy 可以是 simulation Gold，
但必须与 experimental Gold 分开报告。

查看适配规则：

```bash
PYTHONPATH=src python -m tfree_detection label-adapters
```

批量转换 JSON/JSONL：

```bash
PYTHONPATH=src python -m tfree_detection adapt-labels \
  --input external_labels.jsonl \
  --output adapted_labels.v1.jsonl
```

输出文件存在时命令拒绝覆盖。

## 9. 多帧 NData 与 EMD

`load_grayscale(...)` 和 CLI 支持显式 `frame_index` 或 `mean/median/max projection`。
Velox EMD 先用 `inspect-emd` 列出 `Data/Image/.../Data`，多 detector channel 时必须给出
`dataset_path`。平台不会把 microscopy stack 的第三轴默认当 RGB，也不会在多个 EMD
channel 中静默选择一个。
