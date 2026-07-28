# Dennler III–V HRSTEM 锁定外测预注册

> 协议：`dennler-2021-hrstem-zero-shot-v1`  
> 状态：在解包标签、运行 detector 和查看指标之前冻结  
> 数据 DOI：<https://doi.org/10.5281/zenodo.5792917>  
> 方法论文：<https://doi.org/10.1016/j.micron.2021.103069>

## 目的

这个外测不是新的开发集，而是检验当前冻结的 training-free 物理教师是否能在完全没有标签
适配、阈值调整和样本筛选的情况下迁移到外部 III–V HRSTEM 图像。归档 SHA-256 为：

```text
abef46a0183ba08b925c64807d5e914cd97967444d9fd3e0799924896f0614ec
```

机器可执行协议见
[`locked_dennler_v1_protocol.yaml`](../benchmarks/locked_dennler_v1_protocol.yaml)。
协议 seal 同时冻结注册表、归档、配置和推理/评分代码的逐文件 SHA-256。

## 为什么必须分成两个阶段

```text
冻结协议与代码
      ↓
仅访问 images，生成全部 61 张图 × 3 个 detector 的预测
      ↓
prediction_manifest + 每个 NPZ 的 SHA-256
      ↓
prediction_seal
      ↓
在 append-only ledger 登记本次 prediction-set hash
      ↓
首次读取 labels 并评分
```

`predict` 阶段只枚举协议中的 `defective/images` 和 `non_defective/images`，不枚举
`labels`。`score` 阶段在读第一个标签像素前，先把 prediction-set SHA-256 追加到揭盲账本。
此后同一资产如果换成任何不同预测集合，评分器会拒绝，防止看见结果后调参再报最好一次。

## 预先冻结的类别映射

数据作者的标签含义不是我们四任务 ontology 的一一对应，因此预先冻结为：

| 外测目标 | 专家真值 | 冻结预测 |
|---|---|---|
| 模糊/非晶/束损伤复合区 | 白色 class 2 | `amorphous_region` |
| 次级晶体对称性区 | 灰色 class 1 | `stacking_fault ∪ phase_boundary` |
| 任意专家缺陷 | class 1 或 2 | 三个 detector 的并集 |

`stacking_fault` 和 `phase_boundary` 对 class 1 的单独结果只作为诊断，不能在揭盲后选较好者
充当主结果。

## 冻结指标

- 微平均 pixel IoU、Dice、precision、recall、F1 和 specificity；
- defect 图上的 image-level IoU、boundary F1 和 ASSD；
- 30 张 normal 对照上的平均、P95 和最大假阳性面积比例；
- 所有 detector 失败必须保留；预注册规则按全零预测计分；
- 边界容差固定为 2 pixels。

不会在该外测上拟合 temperature、isotonic、conformal 或 OOD 阈值。当前没有独立 Gold
calibration 分区，因此 reliability 只能作为描述性量，不能解释成概率。

## 解释边界

1. class 2 同时包含模糊、非晶和束流缺陷，不能把它称为“纯非晶金标准”；
2. class 1 没有区分层错、晶畴或相界，不能据此评价细分类正确性；
3. 数据包含多种 III–V 材料和 ADF/BF 模式，但来自一个命名仪器/协作档案，不能证明
   leave-one-lab-out 泛化；
4. 该实验只评价冻结解析教师，不评价尚未在真实 Gold 数据上训练的 evidence student；
5. 无论结果好坏，外测资产永不进入主动学习、阈值开发或模型训练。

## 执行命令

以下命令必须按顺序执行，且所有输出目录拒绝覆盖：

```bash
# 1. 在标签解包前创建一次性 protocol seal
PYTHONPATH=src python scripts/run_locked_external_test.py seal

# 2. seal 验证通过后才允许解包
PYTHONPATH=src python scripts/prepare_locked_hrstem.py

# 3. 只生成预测，不读标签
PYTHONPATH=src python scripts/run_locked_external_test.py predict \
  --output benchmarks/locked_runs/dennler_v1_predictions

# 4. 用唯一 sealed prediction set 揭盲
PYTHONPATH=src python scripts/run_locked_external_test.py score \
  --predictions benchmarks/locked_runs/dennler_v1_predictions \
  --output benchmarks/locked_runs/dennler_v1_score
```

如果预测阶段中断，不删除失败目录；换一个明确的新目录重跑，并在揭盲前只选择完整且
`failed_predictions` 已记录的预测集。一旦发生首次揭盲，就不能再更换 prediction set。
