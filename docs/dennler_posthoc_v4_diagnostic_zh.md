# Dennler v0.4 事后诊断重测报告

> 完成日期：2026-07-28  
> 科学状态：`posthoc_diagnostic_not_locked`  
> 结论先行：新版确实修复了 Dennler Class 2 大面积 coherence-loss 区域“几乎不报”的
> 主要机制；Class 1 晶域/结构边界分支仍未取得净增益。

## 1. 为什么这次不能叫新的 zero-shot 外测

Dennler v0.3 的预测先封存、后揭盲，是有效的一次性 locked test。其标签已经在
v0.4 开发前被查看，因此本次只能回答：

> 已知 v0.3 为什么失败后，新的尺度、相干性和候选几何机制能否解释并修复这一失败？

本次预测仍严格分成两个阶段：

1. `predict` 只读取 61 张原图、归档哈希和历史只读结构掩膜，不读取标签；
2. 61/61 张预测成功并写入 seal 后，`score` 才读取标签。

这能防止评分代码反向污染同一次预测，但不能把已经揭盲的 Dennler 重新变成独立外测。
本报告中的提升只能称为**事后机制诊断**，不能称为新 zero-shot、独立泛化或投稿主证据。

协议见
[`dennler_posthoc_v4_diagnostic_protocol.yaml`](../benchmarks/dennler_posthoc_v4_diagnostic_protocol.yaml)。

## 2. 新版到底改了什么

### 2.1 Class 2：从固定像素阈值改为晶格尺度相干性

新版不训练神经网络，也没有使用 SAM 或 DINO。处理链是：

```text
原图稳健归一化
    ↓
FFT 全局倒易峰 → 两个倒易基矢 → 直接晶格尺度
    ↓
复数解调两个晶格 carrier
    ├─ carrier amplitude loss：周期原子相是否消失
    └─ local high-pass loss：原子尺度纹理能量是否下降
    ↓
0.68 × carrier loss + 0.32 × high-pass loss
    ↓
按估计晶格间距设置平滑、形态学半径和最小区域面积
    ↓
coherence-loss region
```

它检测的是“局部清晰原子相相对于同图参考区域明显丢失”，因此既能覆盖真正非晶，也会对
失焦、束损伤、强模糊等相同观测表型响应。这与 Dennler Class 2 本身的复合语义一致，但
尚不能在没有辅助信息时把这些物理原因彼此分开。

### 2.2 触边策略采用 A：保留候选，降低可靠性

旧思路会把触碰图像边缘的连通区域整个删除；但 Dennler 中很多真实模糊区本来就贯穿视野
边界。新版把“有没有物理候选”和“这一位置的测量有多可靠”拆开：

- 二值候选不因触边被删除；
- 最外侧可靠性为 `0.25`；
- 随离边缘距离增加，在约四个晶格周期内渐变到 `1.00`；
- 可靠性作为单独 evidence map 输出，不反过来截断物理区域。

合成单元测试中，贯穿上边缘的真实模糊带覆盖率为 `88.3%`，且顶部候选仍连续存在。

![边缘可靠性规则](../outputs/dennler_posthoc_v4_diagnostic_v2/figures/03_border_reliability_invariant.png)

### 2.3 Class 1：局部波矢晶域提案仍是实验分支

新版还尝试用两个解调 carrier 的局部波矢/形变向量做稳健双原型聚类，并要求聚类边界同时
出现足够强的局部形变梯度。该物理门控把正常图平均误报面积从 v1 实验分支的 `0.1266`
降到 v2 的 `0.0488`，但仍没有通过净增益门槛：

- 局部波矢实验分支 Class 1 IoU：`0.0386`；
- 历史结构主分支 Class 1 IoU：`0.1251`；
- 两者并集 IoU：`0.1098`；
- 两者并集正常图平均误报面积：`0.0820`，高于历史分支的 `0.0351`。

所以 v2 把它隔离成 ablation，只保存候选和证据图，不让它进入保守主结果。主结果为：

```text
secondary_symmetry_region = 历史只读 stacking/phase 分支
any_defect_region = 新 coherence-loss ∪ 历史只读结构分支
```

这不是宣称历史 Class 1 已经足够好，而是防止一个尚未验证的分支用高召回掩盖大量误报。

## 3. 完整结果

### 3.1 v0.3 与 v0.4 v2 主结果

| 目标 | 版本 | IoU | Precision | Recall | F1 | 正常图平均误报面积 |
|---|---|---:|---:|---:|---:|---:|
| Class 2 coherence-loss | v0.3 | 0.0118 | 0.9319 | 0.0118 | 0.0233 | — |
| Class 2 coherence-loss | v0.4 v2 | **0.6108** | 0.6232 | **0.9683** | **0.7584** | **0.0001** |
| Class 1 secondary symmetry | v0.3 | 0.1251 | 0.2291 | 0.2162 | 0.2224 | 0.0351 |
| Class 1 secondary symmetry | v0.4 v2 主分支 | 0.1251 | 0.2291 | 0.2162 | 0.2224 | 0.0351 |
| 任意专家缺陷 | v0.3 | 0.1396 | 0.4522 | 0.1680 | 0.2450 | — |
| 任意专家缺陷 | v0.4 v2 保守组合 | **0.6186** | **0.7010** | **0.8403** | **0.7644** | 0.0352 |

与 v0.3 相比：

- Class 2 IoU `+0.5990`，Recall `+0.9565`，F1 `+0.7350`；
- Class 2 Precision `-0.3086`：从“几乎不报但报了通常对”变成“基本都能覆盖但仍会多报”；
- 任意缺陷 IoU `+0.4790`，Precision `+0.2488`，Recall `+0.6723`，F1 `+0.5194`；
- Class 1 主分支没有变化，也没有伪装成提升。

![总指标对比](../outputs/dennler_posthoc_v4_diagnostic_v2/figures/01_metric_comparison.png)

### 3.2 为什么保守组合比最初 v1 实验组合更好

最初的 v0.4 v1 直接把局部波矢晶域提案并入总掩膜，结果：

| 版本/组合 | 任意缺陷 IoU | Precision | Recall | 正常图平均误报面积 |
|---|---:|---:|---:|---:|
| v0.4 v1 全并集 | 0.4586 | 0.4962 | 0.8582 | 0.1568 |
| v0.4 v2 保守组合 | **0.6186** | **0.7010** | 0.8403 | **0.0352** |
| v0.4 v2 实验增强并集 | 0.5345 | 0.5865 | 0.8576 | 0.0821 |

v2 不是靠继续扩大掩膜获得高分，而是通过失败隔离，把无净增益的晶域分支从主结果拿掉。
这也是平台需要长期保留的发布治理原则：新分支可以生成候选和消融结果，但只有通过正常
对照、几何匹配指标和全新 locked test 后才进入主输出。

## 4. 逐例解释

![逐例结果](../outputs/dennler_posthoc_v4_diagnostic_v2/figures/02_example_gallery.png)

- **A：ADF x10M_05，典型修复。** Class 2 真值占图像 `54.2%`；v0.3 IoU 为
  `0.000`，v0.4 为 `0.955`。原子相消失带被完整覆盖。
- **B：ADF x10M_02，仍会过分割。** Class 2 真值只占 `5.4%`，新版把右侧更广的低
  相干区域也检出，IoU 只有 `0.110`。这可能是标签只圈了最严重区域，也可能是当前阈值对
  局部对比度变化过敏；在新的 Gold calibration 前不能靠这一张图继续调阈值。
- **C：BF x10M_19，Class 1 历史分支局部有效。** v0.3 与 v0.4 主结果 IoU 都为
  `0.661`；这明确表示新实验晶域分支没有替换主分支。
- **D：正常 BF x8M_22。** 新 coherence-loss 没有误报，但保守总掩膜仍继承了历史结构
  分支的局部误报。Class 1 正常控制仍是下一轮重点。

## 5. 结构边界数值为什么不能当作结论

v2 的实验性 `structural_boundary` 对 Dennler Class 1 区域标签的像素 IoU 为 `0.0251`，
正样本平均 boundary F1 为 `0.0114`。这同时包含两层问题：

1. 当前局部波矢边界分支本身不够稳；
2. Dennler Class 1 提供的是粗粒度区域，不是独立的单像素/窄线界面 Gold，把窄边界与整块
   区域做像素 IoU 本来就几何不匹配。

因此下一版需要分别准备 region、line/interface 和 point 的几何匹配真值，界面任务用
容差 boundary F1、平均轮廓距离和拓扑连续性评估，不能再用一个 box 或一个区域 IoU 代表
全部结构缺陷。

## 6. 下一步应该做什么

### 可立即在代码中继续做

1. 冻结本次 Class 2 机制和边缘不变量，不再根据 Dennler 标签调整阈值；
2. 为 Class 1 增加局部参考晶格配准、位移不连续、应变张量、Burgers/disregistry 和线拓扑
   证据，保持每个物理分支可独立消融；
3. 把候选区域与可靠性图完整接入统一 `EvidenceBundle`，使主动审阅能优先采样“高物理
   违反、低可靠性”的区域；
4. 为 region、interface、line、point 建立不同的评价器和发布门禁；
5. 继续保存实验分支失败结果，不允许消融分支自动进入 release mask。

### 必须依赖新证据，不能靠当前代码自证

1. 找到一套从未查看标签的新 coherence-loss 数据，预注册阈值、正常对照和主指标；
2. 至少两位 STEM 专家分别标注“可见低相干区域”和“推断物理原因”，避免把观测与因果
   混为一个标签；
3. 获取真正的界面/层错窄线 Gold，以几何匹配指标验证 Class 1；
4. 只在独立 calibration 上决定 reliability-to-risk 映射，再对全新 locked test 一次揭盲。

## 7. 复现入口

```bash
# 预测：不读取标签；输出目录必须不存在
PYTHONPATH=src python scripts/run_dennler_posthoc_v4.py predict \
  --output outputs/dennler_posthoc_v4_diagnostic_v2/predictions

# 预测封存后评分
PYTHONPATH=src python scripts/run_dennler_posthoc_v4.py score \
  --predictions outputs/dennler_posthoc_v4_diagnostic_v2/predictions \
  --output outputs/dennler_posthoc_v4_diagnostic_v2/score

# 可视化
PYTHONPATH=src python scripts/visualize_dennler_posthoc_v4.py

# 新机制与失败隔离测试
PYTHONPATH=src pytest -q tests/test_diagnostic_v4.py
```

关键产物：

- [预测 seal](../outputs/dennler_posthoc_v4_diagnostic_v2/predictions/prediction_seal.json)
- [完整评分](../outputs/dennler_posthoc_v4_diagnostic_v2/score/posthoc_metrics.json)
- [评分 seal](../outputs/dennler_posthoc_v4_diagnostic_v2/score/score_seal.json)
- [指标图](../outputs/dennler_posthoc_v4_diagnostic_v2/figures/01_metric_comparison.png)
- [逐例图](../outputs/dennler_posthoc_v4_diagnostic_v2/figures/02_example_gallery.png)
- [边缘规则图](../outputs/dennler_posthoc_v4_diagnostic_v2/figures/03_border_reliability_invariant.png)

