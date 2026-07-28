# 数据治理、Gold 标注与校准协议

这一层负责阻止三类最危险的“伪提升”：相邻 patch 泄漏、伪标签覆盖 Gold、以及用已经看过的
测试数据选择主动学习样本。

## 数据分区

平台固定使用四个分区：

| 分区 | 允许训练 | 允许拟合校准器 | 允许主动学习 | 用途 |
|---|---:|---:|---:|---|
| `train` | 是 | 否 | 是 | 学生训练和候选送审池 |
| `calibration` | 否 | 是 | 否 | Gold-only 概率和 conformal 校准 |
| `development_test` | 否 | 否 | 否 | 算法和阈值开发比较 |
| `locked_external_test` | 否 | 否 | 永不允许 | 最终跨域评价 |

切分单位由 `split_group_id` 固定，同一采集序列、样品或显式组不能跨分区。由同一原图裁出的
patch 或派生数据必须继承父资产的分区和 split group。

当前开发使用
[`partition_registry_v2.yaml`](../datasets/partition_registry_v2.yaml)。原有本地图和 Guo
公开图已经影响过方法开发，因此诚实地登记为 `development_test`。Dennler 等人的 III–V
HRSTEM 专家标注归档在任何 detector 运行前完成 checksum 冻结，作为一个不可拆分的
`locked_external_test` group。随后按照预注册协议完成 183 次预测封存和一次性揭盲；
任意专家缺陷的零样本 IoU 为 0.1396，不能支持跨域泛化主张。该资产已经消费，今后永不
用于训练、校准、主动学习或阈值开发。

Dennler seal 同时冻结了 v1
[`partition_registry.yaml`](../datasets/partition_registry.yaml)，所以新增数据不能原地
改写 v1，只能生成新版本 registry。v2 当前有：

| 分区 | 文件资产 | split groups |
|---|---:|---:|
| `train` | 215 | 1 |
| `calibration` | 0 | 0 |
| `development_test` | 4 | 4 |
| `locked_external_test` | 1 | 1 |

train 来自 EMSeek WSe₂：107 张图、107 张作者点标注和 1 个 source manifest。由于公开
acquisition/sample 关系无法还原，全部文件被放进同一个保守 group，不能从中再切出
calibration/test。公开 reviewer 数未知，因此标注登记为 `author_curated`，不是 Gold。
当前 `calibration` 仍为空，所以平台不会声称已经完成真实概率校准或跨实验室验证。

具体协议与结果分别见
[`locked_external_test_protocol_zh.md`](locked_external_test_protocol_zh.md) 和
[`locked_external_test_result_zh.md`](locked_external_test_result_zh.md)。

## 外部数据准入

[`data_intake.py`](../src/tfree_detection/data_intake.py) 在 registry 之前增加一层 sealed
intake plan。一次准入必须记录并验证：

- source URI、不可变 revision 和 license SPDX；
- group、acquisition、sample、lab、microscope、material provenance；
- 每个文件的 role、bytes、SHA-256 和 image/annotation `pair_id`；
- annotation qualification、reviewer 数和 positive/negative/mixed control role；
- active-learning eligibility。

门禁规则包括：

- license 未知只能留在 quarantine；
- 同内容文件不能跨 partition 重复；
- 只有 train 图像可以进入主动学习；
- locked test 永不进入主动学习；
- calibration 必须是至少两位专家共识，并同时含 Gold positive 和 Gold negative/normal；
- author-curated 或单专家标签不能被描述为 Gold。

当前 EMSeek 的 source manifest、sealed plan 和准入哈希见
[`periodic_counterfactual_and_consensus_zh.md`](periodic_counterfactual_and_consensus_zh.md)。

验证命令：

```bash
PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/emseek_wse2_train_v1.sealed.yaml \
  --verify-files

PYTHONPATH=src python -m tfree_detection validate-registry \
  --registry datasets/partition_registry_v2.yaml \
  --project-root .
```

## 候选 provenance

每次检测记录：

- 原始文件 SHA-256（文件输入时）；
- 原始像素、形状和 dtype 的 SHA-256；
- 完整配置 SHA-256；
- physics teacher 和证据协议版本；
- 每个候选的稳定 ID、bbox、几何、违反类型、证据向量、可靠性和质量标记。

`save_result(...)` 现在会输出 `candidate_table.jsonl`。为避免静默改写标签来源，如果目标表
已存在，保存会拒绝覆盖，要求使用新的版本化运行目录。

## Gold / Silver / Unresolved

标注历史采用 append-only ledger：

- 物理教师不能创建 Gold；
- 专家共识 Gold 至少需要两位 reviewer；
- 可验证数字孪生真值可以是 Gold，但必须保留模拟来源；
- 一旦候选已有 Gold，后续 Silver/Unresolved 不能覆盖它；
- 修订通过 `parent_annotation_id` 形成谱系，而不是原地改写。

## 校准

[`calibration.py`](../src/tfree_detection/calibration.py) 提供：

- temperature scaling：适合学生模型 logit；
- PAVA isotonic calibration：适合单调但尺度未知的候选分数；
- Brier score 和 ECE；
- risk–coverage 曲线；
- binary split conformal prediction sets。

拟合接口会强制每个样本满足：

```text
partition == calibration
label_tier == gold
同时包含 Gold positive 和 Gold normal/negative
```

Conformal 输出仅在 calibration 与部署样本可交换时有覆盖意义。新材料、新仪器或新实验室
必须先经过 OOD 检查，发生分布漂移后应重新校准，不能沿用固定覆盖保证。
