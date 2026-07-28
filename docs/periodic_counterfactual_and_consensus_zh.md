# 周期反事实与双物理教师共识：原子点缺陷开发域验证

> 状态：2026-07-26，development/train-domain evidence  
> 结论边界：本页结果不是独立校准、locked external test 或跨材料泛化证据。

## 1. 这一轮解决了什么

冻结的 v0.2 原子位点教师先定位亮原子柱，再拟合晶格并读取位点残差。在 WSe₂ 图像上，
它表现为高召回、低精度：很多真实缺陷能被覆盖，但正常位点也容易成为候选。新增的周期
反事实教师从另一条物理路径出发：

> 晶体在直接晶格矢量平移后应看到相同局部 motif；真正的局部缺陷应同时违反多个方向的
> 周期等价关系。

两种教师不共享监督标签、训练参数或检测机制。平台保留它们的一致和不一致，而不是把一个
教师静默当作另一个教师的真值：

```text
bright-site registration teacher ── high recall ──┐
                                                   ├─ one-to-one consensus → Silver
periodic counterfactual teacher ─ high precision ─┘
                  teacher-only branches → Unresolved / active-review pool
```

## 2. 周期反事实分数

由全图 Fourier 峰估计直接晶格基矢 \(a,b\)，构造
\(\mathcal V=\{\pm a,\pm b,\pm(b-a)\}\)。先对图像做固定尺度平滑，再计算同一张观测内的
平移反事实差：

\[
D_v(x)=\left|G_{\sigma}*I(x)-G_{\sigma}*I(x+v)\right|,
\qquad v\in\mathcal V .
\]

最终异常证据为

\[
S(x)=G_{\tau}*\min_{v\in\mathcal V}D_v(x).
\]

使用最小值的含义是：只有当某位置与所有可用周期对应位置都不一致时，\(S(x)\) 才会高。
如果只是某一个邻居受噪声、边缘或局部插值影响，至少还有一个正常周期对应位置把最小值
压低，因此不会仅凭单方向异常报缺陷。边界宽度按最大晶格平移量排除，候选在每张图内部
使用 `q=0.997` 上尾阈值和与晶格间距成比例的最小峰间距。

这仍是开发过的数值策略：`q=0.997` 不能被解释成跨域校准后的概率阈值。实现位于
[`periodic_counterfactual.py`](../src/tfree_detection/periodic_counterfactual.py)。

## 3. 双教师共识与不确定分支

周期反事实教师作为 primary，v0.2 位点配准教师作为 secondary。配对容差固定为 primary
推断晶格间距的 0.35 倍。`site_matching_v2` 在容差图内先最大化一对一匹配数，再在最大
匹配数下最小化定位距离：

1. 匹配上的两个点取坐标和分数的算术平均，生成 teacher-consensus Silver 候选；
2. 周期教师独有点保留为 `periodic_only_uncertain`；
3. 位点配准教师独有点保留为 `legacy_only_uncertain`；
4. 任一教师都不能创建 Gold；Gold 仍要求至少两位独立专家共识或可验证模拟真值。

当前实现见
[`site_consensus.py`](../src/tfree_detection/site_consensus.py) 和
[`site_matching.py`](../src/tfree_detection/site_matching.py)。

### 为什么必须是 `site_matching_v2`

“先对所有距离做普通 Hungarian，再丢弃超过容差的边”不保证得到容差内最大匹配数；
存在加入候选反而减少真阳性的反例。旧开发分数和旧共识结果因此完整保留，但标记为
superseded。当前结果统一采用词典序目标：

```text
primary objective   = maximum cardinality inside tolerance
secondary objective = minimum total localization distance
```

评估器和教师共识现在使用同一版本规则，并有显式反例测试。

## 4. 数据准入与泄漏控制

数据来自 [EMSeek Hugging Face 数据页](https://huggingface.co/datasets/gary23ai/EMSeek)，
固定 revision：

```text
8f4a987c78af743098459e5d467f2c1b5f5c11b3
```

当前 WSe₂ 子集包含 107 张 1024×1024 图像和 107 张 RGB 点标注，共 92,121,442 bytes。
下载器逐文件验证 LFS SHA-256、禁止覆盖并仅续传 `.part`；准入计划另含一个不可变 source
manifest，因此注册了 215 个 train 文件资产。

由于公开元数据没有给出足以可靠拆分 acquisition/sample 的信息，107 对图像和标注被
保守地放在同一个 `emseek_wse2_single_conservative_group`：

- 不能从这一组再切出独立 calibration 或 locked test；
- 图像可以进入主动学习，标注和 source manifest 不可以；
- 论文描述为 consensus annotations，但公开 reviewer 数未知，因此准入等级是
  `author_curated`，不是 Gold；
- 检测预测先封存，评分阶段才读取标注像素；
- 标签评估目标只取所有非黑连通域中心，不推断蓝/绿 palette 的缺陷亚类语义。

准入文件：

- [`emseek_wse2_source_manifest.json`](../datasets/intake/emseek_wse2_source_manifest.json)
- [`emseek_wse2_train_v1.sealed.yaml`](../datasets/intake/emseek_wse2_train_v1.sealed.yaml)
- [`partition_registry_v2.yaml`](../datasets/partition_registry_v2.yaml)

数据准入逻辑见
[`data_intake.py`](../src/tfree_detection/data_intake.py)，包括 license gate、group provenance、
image/annotation pair 检查、跨分区重复内容拒绝、calibration 双专家与正/负对照门禁，以及
locked asset 永久禁止主动学习。

## 5. 开发域结果

全部结果来自同一个 WSe₂ train group，三条分支使用相同的 107 张图、相同点真值和
`site_matching_v2`。点匹配容差为每图推断晶格间距的 0.35 倍。

| 方法 | 预测点 | 匹配点 | Precision | Recall | F1 | 平均定位误差/px | P95/px |
|---|---:|---:|---:|---:|---:|---:|---:|
| v0.2 位点配准教师 | 30,213 | 14,789 | 0.4895 | 0.9226 | 0.6396 | 0.691 | 1.359 |
| 周期反事实教师 | 14,958 | 11,019 | 0.7367 | 0.6874 | 0.7112 | 1.150 | 2.236 |
| 双教师共识 v2 | 12,001 | 10,698 | 0.8914 | 0.6674 | 0.7633 | 0.835 | 1.636 |

双教师共识相对 v0.2 教师的 F1 增加 0.1237，相对周期反事实教师增加 0.0521。它将位点配准
教师的高召回与周期反事实教师的高精度转化成一个更低风险的自动 Silver 层，但没有解决
全部漏检：图像 56、66、71 因 v0.2 教师没有候选而产生零共识。

当前汇总图：

- [教师互补性与逐图分布](../benchmarks/development_runs/emseek_wse2_two_teacher_consensus_v2_score/teacher_consensus_summary_final.png)
- [图像 23 的原图、作者标注、共识状态和开发域 TP/FP/FN](../benchmarks/development_runs/emseek_wse2_two_teacher_consensus_v2_score/teacher_consensus_overlay_image23_final.png)

图中的圆表示点中心和匹配状态，不是原子尺寸的 segmentation mask。

## 6. Selective triage：把风险变成审阅策略

两个教师的联合候选池共有 33,170 个点。确认层和两个不一致层形成可审计的
risk–coverage 路径：

| 自动/审阅层 | 候选池 coverage | Precision | Recall | F1 | selective risk |
|---|---:|---:|---:|---:|---:|
| 仅双教师共识 | 0.3618 | 0.8914 | 0.6674 | 0.7633 | 0.1086 |
| 共识 + periodic-only | 0.4509 | 0.7367 | 0.6875 | 0.7112 | 0.2633 |
| 全部候选，含 legacy-only | 1.0000 | 0.4556 | 0.9427 | 0.6143 | 0.5444 |

分支计数为：

```text
confirmed consensus       12,001
periodic-only uncertain    2,957
legacy-only uncertain     18,212
active-review pool        21,169
```

因此平台可以按任务成本选择：

- 高置信自动预标注使用共识层；
- 希望补少量召回时，先审阅 periodic-only；
- 需要接近完整候选发现时，再审阅规模较大的 legacy-only；
- 所有不一致仍保留为 Unresolved，不能因排队优先级而被自动改成 Gold。

基于这三层，平台已从 69 张图封存一个 100-candidate pilot：三个教师状态各 25 个，另加
25 个 random sham prompts；选择不读取作者标签。协议和 expert 入口见
[`expert_review_workflow_zh.md`](expert_review_workflow_zh.md)。当前 response 数为 0，因此
本页仍不报告专家一致性或 Gold 数量。

## 7. 可复现产物与哈希

| 对象 | SHA-256 |
|---|---|
| source manifest 文件 | `41891936c965b4a636c6ad9671a706d8ba79c82706e5fc7cea2bf89cd88c8a0e` |
| sealed intake 内容 | `6bb2a95e35bce5ae500a19a735996bb14b1a77d819018efc593d837351cdb1dd` |
| registry v2 内容 | `cb4cf6241c8f2381c47c4fc4bedc952da7b68c6bca28d8ac86c8bff30d441f73` |
| 周期教师预测集 | `a0dc515cdd7d6dbfcec2862989033cbb5e3fb8ccbd10a5ccfc7f80a6a2fbb829` |
| 位点配准教师预测集 | `adc9479e3e1e89d68734fcdb21dc16fde633b92e029c5405d12658b906c2e1de` |
| 双教师共识 v2 预测集 | `137569c3d2ff8c576765b2f45ba83d2c1ea64d496481da311141096441c60dde` |
| 周期教师 v2 score 内容 | `a8f6c3cd5c3fafa17ee34e45e952379bbb88e045b03de7d32ca59e46a5dd5b8b` |
| 位点配准教师 v2 score 内容 | `e56f9914abfd583c679f6b0b430e2cdb5f3430e268161a7b74f5a51e6db2006f` |
| 双教师共识 v2 score 内容 | `93c5bda10624ebe0c898bbe3499a171b341a80307d17a99c80fea8937c303dcc` |
| selective triage 内容 | `66ebbc65455933600d20131da88be5c33037998286e306d5f341be000fe8ef9c` |
| 最终 summary 图 | `edaf7696be828d4c85fd2b8032e623c55817a2ca4dcf028b19bfa2625a3368b1` |
| 最终 overlay 图 | `39d6fb8f147ff3bdbb60778aa8e86ee124550685fc027bdbfc347868b0af7c89` |

当前 JSON 指标分别位于：

- [`periodic development_metrics.json`](../benchmarks/development_runs/emseek_wse2_periodic_v1_score_v2/development_metrics.json)
- [`legacy development_metrics.json`](../benchmarks/development_runs/emseek_wse2_legacy_atomic_v0_2_score_v2/development_metrics.json)
- [`consensus development_metrics.json`](../benchmarks/development_runs/emseek_wse2_two_teacher_consensus_v2_score/development_metrics.json)
- [`consensus_triage_metrics.json`](../benchmarks/development_runs/emseek_wse2_two_teacher_consensus_v2_score/consensus_triage_metrics.json)

验证准入和当前注册表：

```bash
PYTHONPATH=src python scripts/manage_data_intake.py verify \
  --plan datasets/intake/emseek_wse2_train_v1.sealed.yaml \
  --verify-files

PYTHONPATH=src python -m tfree_detection validate-registry \
  --registry datasets/partition_registry_v2.yaml \
  --project-root .
```

预测、共识和评分命令会拒绝覆盖已有目录；复现时必须使用新的版本化输出路径。主入口为
[`run_periodic_counterfactual_benchmark.py`](../scripts/run_periodic_counterfactual_benchmark.py)。

## 8. 当前不能声称什么

这一轮支持的是“两个互补物理教师在已准入 WSe₂ 开发域上形成更好的选择性候选层”，不支持：

- 独立材料、独立样品、独立 acquisition 或跨实验室泛化；
- 概率校准、conformal coverage 或正常图像假阳性率；
- 蓝/绿 palette 对应的精细缺陷亚类识别；
- EMSeek 标注已经满足本平台的双专家 Gold 定义；
- `q=0.997` 是可直接迁移到其他像素尺寸或对比度的固定阈值；
- 三个零共识图像已解决；
- student 已在真实数据上训练并优于 physics teacher；
- 旧 Dennler locked test 可以被重新用于调参或再次证明改进。

下一道科学门禁仍是：按真实 acquisition/sample 建立独立 Gold calibration，并在一个从未
用于开发的新材料或新实验室集合上前瞻性封存模型和指标后一次性测试。
