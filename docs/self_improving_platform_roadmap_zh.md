# 从 Training-Free 检测器到物理对齐、自提升的显微缺陷平台

> 版本：路线图草案 v1  
> 日期：2026-07-26  
> 目标：把当前四任务原型发展为具有统一方法学、可信度控制、主动学习闭环和跨域验证的
> 显微结构异常分析平台。

## 2026-07-28 Class1 路线更新

路线图中的“统一物理表示”已向前推进一步：新增 `symmetry_relation`，以参考晶格下的群
关系残差统一表达平移域、旋转/镜像域、界面、相位滑移和位错芯代理，并由扫描伪影教师
竞争解释。Dennler 已消费数据上的事后结果表明 Class1 guided IoU/F1 分别从
`0.1251/0.2224` 提高到 `0.1891/0.3181`，但 Precision 仍约 `0.23`。

因此下一步冻结该实现，不再消费 Dennler 调参；需要在新 locked test 上按 region、
interface/line 和 point 三种评估协议分别验证，并增加原子图 Burgers circuit 与真正时序
扫描证据。详见
[`class1_symmetry_relation_final_zh.md`](class1_symmetry_relation_final_zh.md)。

## 2026-07-27 实施补记

路线图中不依赖新专家意见或新实验的部分已进一步实现：公开数据 registry v4 与逐资产
manifest、EMD detector/尺度自动解析、原子双物理教师统一推理、线/界面 topology 描述、
SuperScan 漂移校正时序支持，以及候选来源状态机。所有这些证据都保持 training-free，
并且不会把作者标签冒充实验 Gold。

下一阶段真正不可被继续写代码“代做”的门槛已经收敛为：

1. 两位独立材料/显微专家对预注册候选完成盲审并形成实验 Gold；
2. 取得一套从未参与设计/调参的新材料、新 acquisition 或新实验室 locked test；
3. 提供可验证的厚度、取向、剂量、像差等参数，完成真实 multislice 对照；
4. 在学生模型和 training-free teacher 间做跨材料 prospective 非退化验证；
5. 最后才进入显微镜在线 acquisition/bandit/RL。

因此主动学习、学生模型和 RL 代码可以继续存在并接受自动测试，但在上述证据到位前，
不能把“软件闭环”写成“真实持续提高已经被证明”。

## 1. 先给结论

当前框架已经证明了一个重要起点：不依赖人工标签，也不微调神经网络，仅依靠周期结构、
晶格配准、相位和局部倒易空间证据，就能对局部非晶、原子点缺陷、层错/位错和两相界面
给出可解释候选。

但若要形成统一且有竞争力的方法论文，下一步不能只是：

```text
training-free 生成掩膜
        ↓
用这些掩膜训练 U-Net / SAM
        ↓
再加一个强化学习循环
```

原因有三点：

1. 当前四个任务共享入口和输出协议，但内部仍是四套相对独立的算法，尚未形成统一表示；
2. Cellpose 2.0 和 μSAM 已经实现“模型预测—人工修正—重新训练”，单独加入该闭环并不
   构成足够的新颖性；
3. 若强化学习直接以当前 training-free 结果为奖励，系统会学习复制教师的偏差，形成
   confirmation bias 和 reward hacking，而不一定接近物理真值。

更强的主线应当是：

> **Physics teacher → evidence-preserving student → disagreement-driven active learning**

即：

- training-free 模块不再只输出一个二值掩膜，而输出多尺度物理证据、可靠性和不确定性；
- 学习模型同时学习“缺陷是什么”和“为什么是缺陷”；
- 系统主动选择物理教师、学生模型和专家最不一致的数据；
- 最终输出经过独立校准，可以拒答，而不是强制对每张图给出高置信结论。

本文将这个下一代框架暂称为：

> **PhyEvo-Mic：Physics-Evidence-Guided Self-Evolving Microscopy**

名称可以以后更换，核心是“物理证据教师 + 自提升学习器”，而不是某个特定网络。

---

## 2. 当前平台达到了什么，还缺什么

### 2.1 已经具备的能力

| 层 | 当前实现 |
|---|---|
| 输入 | 8/16/32-bit 灰度图，保留原始尺度元数据 |
| 公共入口 | 统一 `TrainingFreeDetectionFramework` 和任务注册表 |
| 局部非晶 | Fourier、Patterson、清晰度多证据融合 |
| 原子点缺陷 | 原子峰、自动多子晶格、非刚性配准、位点强度/位移/存在性残差 |
| 层错/位错 | 两载波复解调、相位突变、曲率、剪切/旋转、相干度 |
| 两相界面 | 局部极坐标 Fourier 签名、双原型分域、界面追踪 |
| 第二阶段 | GPA 应变/相干度和冻结 DINO 纹理偏离验证 |
| 输出 | 掩膜、连续分数、证据图、区域 JSON、配置和诊断图 |
| 验证 | 确定性合成真值、公开数据复现、17 个自动测试 |

当前四任务合成基准平均 IoU 为 0.650。局部非晶已较稳健，界面和线缺陷仍受窗口尺度、
边缘响应及形态学定位精度影响。公开数据没有穷尽式逐像素真值，因此目前只能说明结构一致性
和可复现性，不能说明真实世界精度。

### 2.2 方法学上的主要缺口

#### 缺口 A：统一的是软件接口，不是科学表示

现在的 `DetectionTask` 统一了调用方式，但四个任务分别构造自己的分数、阈值和置信度。
“非晶分数 3.0”和“相界分数 0.9”没有公共含义，也没有共享的物理状态表示。

#### 缺口 B：当前置信值不是经过标签校准的概率

当前 confidence 主要由当前图像中的 median/MAD、阈值距离和证据组数量得到。它适合排序，
但不能解释为“90% 概率为真实缺陷”，也不能给出跨材料覆盖保证。

#### 缺口 C：没有真实金标准和跨域评估

当前公开实验来自单个数据档案。缺少：

- 多材料、不同晶格对称性和不同缺陷密度；
- 不同显微镜、探测器、剂量、扫描方向和实验室；
- 多位专家独立标注和一致性分析；
- 留一材料、留一仪器、留一实验室的真正 OOD 测试。

#### 缺口 D：DINO/SAM 仍是外围工具

当前 DINO 只对非晶候选做冻结特征验证；SAM 尚未参与正式输出。这种保守设计是正确的，
但也意味着平台尚未拥有能够跨任务快速推理和从反馈中成长的学习层。

#### 缺口 E：没有数据闭环和版本治理

还没有统一记录：

- 一个标签来自物理教师、数字孪生还是专家；
- 专家修改了哪些像素、类型或伪影标记；
- 哪个模型版本看过哪些数据；
- 哪些数据属于训练、校准、开发和永不触碰的最终测试集。

---

## 3. 建议的统一科学表示：多尺度结构证据图

### 3.1 从四个检测器变成一个结构解释问题

设原始观测为 \(I\)，结构先验为 \(S\)，成像和几何干扰参数为 \(\theta\)。统一问题写成：

\[
\hat I = G(S,\theta), \qquad R = I-\hat I
\]

其中 \(G\) 可以是解析晶格/PSF 模型、局部相位模型、Fourier 原型，或以后加入的 multislice
模拟器。平台要回答的不是“图像像不像某类缺陷”，而是：

> 正常结构在允许的成像噪声和形变下，能否解释当前观测？

不同缺陷只是“正常解释失败”的不同空间尺度与拓扑形式。

### 3.2 统一成节点—边—区域三级证据

建议把当前证据改造成一个 **Structural Evidence Graph**：

#### 节点级证据：0D 原子/原子柱

\[
e_v = [
r_{\mathrm{presence}},
r_{\mathrm{intensity}},
r_{\mathrm{displacement}},
r_{\mathrm{shape}}
]
\]

用于空位、置换、低强度柱、偏移和间隙柱。

#### 边级证据：1D 配位与晶格连接

\[
e_e = [
\Delta\phi,
\epsilon_{xx},
\epsilon_{yy},
\epsilon_{xy},
\omega,
c_{\mathrm{lattice}}
]
\]

用于键长/角度变化、相位滑移、层错、位错和局部扫描断裂。

#### 区域级证据：2D 周期结构与域

\[
e_r = [
H_{\mathrm{FFT}},
C_{\mathrm{peak}},
P_{\mathrm{Patterson}},
C_{\mathrm{sharpness}},
d_{\mathrm{domain}},
q_{\mathrm{specimen}}
]
\]

用于局部非晶、晶畴、两相界面、真空和失焦区域。

每个证据同时携带可靠性 \(q_k(x)\)，例如：

- Fourier 有效边界；
- 晶格配准匹配比例；
- GPA 相干度；
- 局部样品/真空支持；
- 原始信噪比和饱和信息。

### 3.3 用“违反类型 × 几何支撑”定义缺陷本体

建议不要把类别仅写成扁平字符串，而使用两个轴：

| 轴 | 例子 |
|---|---|
| 违反类型 | occupancy、intensity、displacement、phase-slip、coherence-loss、domain-change |
| 空间支撑 | point、line、region、interface |

于是：

- 空位 = `occupancy × point`；
- 层错 = `phase-slip × line`；
- 局部非晶 = `coherence-loss × region`；
- 两相界面 = `domain-change × interface`。

这样新增缺陷时不必再发明完全独立的检测协议，也能支持多标签和复合缺陷。

### 3.4 统一假设能量

对缺陷假设 \(h\) 和候选区域 \(\Omega\)，定义：

\[
E_h(\Omega)
=
\sum_{k}
w_{hk}\,
\rho\!\left(e_k(\Omega)\right)\,
q_k(\Omega)
\]

其中：

- \(e_k\) 是节点、边、区域证据；
- \(q_k\) 是可靠性门控；
- \(\rho\) 是鲁棒变换；
- \(w_{hk}\) 表示某类缺陷需要哪些证据。

第一版可以使用透明、人工定义的稀疏 \(w_{hk}\)。以后再让模型学习权重，但必须保持非负、
稀疏和可审计，避免网络通过无物理意义的背景纹理完成分类。

---

## 4. Training-free 引擎的新角色：物理证据教师

### 4.1 不再只保存硬掩膜

每个候选应该保存：

```yaml
candidate_id:
geometry: point | line | region | interface
type_distribution:
evidence_vector:
evidence_maps:
reliability_maps:
teacher_soft_mask:
teacher_uncertainty:
quality_flags:
source:
  kind: physics_pseudo | digital_twin | human_gold
  detector_version:
  parameters:
```

训练使用软分布和可靠性权重，不把一个阈值后的二值掩膜伪装成绝对真值。

### 4.2 伪标签分成三档

| 档位 | 条件 | 用途 |
|---|---|---|
| Gold | 专家共识或可验证模拟真值 | 训练、校准、最终评估 |
| Silver | 多个独立物理证据一致，且位于高可靠区 | 带权训练 |
| Unresolved | 教师证据冲突、低相干、OOD 或疑似成像伪影 | 主动送审，不训练为硬标签 |

高置信正常区域也要进入数据集。只收集缺陷候选会让学生不知道真实背景分布。

### 4.3 保留教师与学生的独立性

物理教师不能被学生模型反向更新，否则两者会逐渐失去独立性。建议：

- 物理教师参数由明确的算法版本和校准集管理；
- 学生可以学习教师证据；
- 学生发现的大量新候选只能进入待审核队列；
- 专家确认后才能修改 gold 数据或教师规则。

---

## 5. 学习层：Evidence-Preserving Student

### 5.1 学生不应只预测一个分割掩膜

建议使用一个共享编码器和四类输出头：

1. **几何头**：point / line / region / interface；
2. **缺陷头**：违反类型和复合类别；
3. **物理证据头**：回归教师的 occupancy、phase、strain、coherence、domain 等证据；
4. **可靠性/OOD 头**：输出拒答风险和域外程度。

主干可以从小型 ConvNeXt、U-Net、ViT 或显微领域自监督编码器开始。不要把创新点押在
“使用最新 backbone”上；核心贡献应是证据监督和自提升机制。

### 5.2 建议的损失

\[
\mathcal L =
\mathcal L_{\mathrm{gold}}
+ \lambda_p q\,\mathcal L_{\mathrm{pseudo}}
+ \lambda_e \mathcal L_{\mathrm{evidence}}
+ \lambda_s \mathcal L_{\mathrm{symmetry}}
+ \lambda_c \mathcal L_{\mathrm{consistency}}
+ \lambda_u \mathcal L_{\mathrm{uncertainty}}
\]

其中：

- \(\mathcal L_{\mathrm{gold}}\)：专家/模拟真值监督；
- \(\mathcal L_{\mathrm{pseudo}}\)：可靠性加权的软伪标签；
- \(\mathcal L_{\mathrm{evidence}}\)：学生复现物理证据；
- \(\mathcal L_{\mathrm{symmetry}}\)：旋转、镜像、晶格平移的等变/不变约束；
- \(\mathcal L_{\mathrm{consistency}}\)：弱/强增强下的一致性；
- \(\mathcal L_{\mathrm{uncertainty}}\)：学习何时拒答。

用于定量 STEM 的增强必须尊重物理：

- 可以：旋转、镜像、平移、计数噪声、剂量、轻度模糊、扫描漂移；
- 谨慎：任意强度变换、非物理局部形变；
- 禁止：会生成或删除原子峰的生成式增强直接作为定量真值。

### 5.3 先做域内自监督，再做伪标签蒸馏

当前冻结 DINO 来自自然图像。下一步可先用大量无标签显微 patch 做 masked image modeling
或 DINO-style 自监督预训练，但增强必须保持晶格对称性和定量结构。随后再进行证据蒸馏。

这一步可能提高跨材料表征，但不是首要里程碑。若没有足够多样的无标签数据，先用小模型和
明确物理证据更可信。

---

## 6. 主动学习：系统应该主动问什么

### 6.1 不只按照网络熵选样本

建议的样本价值分数：

\[
A_i =
\alpha U_i
+ \beta D_i^{\mathrm{student-teacher}}
+ \gamma O_i^{\mathrm{OOD}}
+ \delta V_i^{\mathrm{diversity}}
+ \eta P_i^{\mathrm{violation}}
- \lambda C_i^{\mathrm{annotation}}
\]

分别表示：

- 学生不确定性；
- 学生与物理教师的分歧；
- 域外程度；
- 与已标注数据的多样性；
- 物理一致性违反程度；
- 专家标注成本。

特别有价值的不是“模型最不确定的像素”，而是：

1. 学生很自信但违反物理教师；
2. 教师很自信但学生在多种增强下不稳定；
3. 教师和学生都不确定，但样本位于新的材料/成像域；
4. 不同专家经常分歧的候选；
5. 可能是新缺陷类型而非现有类别边界的候选。

### 6.2 专家交互应支持低成本标签

不应要求每次都画完整像素掩膜。按成本分级：

| 交互 | 适用对象 |
|---|---|
| 接受/拒绝候选 | 点缺陷和明显非晶 |
| 选择缺陷类型或“未知” | 分类 |
| 点选预期/观测原子 | 晶格配准纠错 |
| 拖动线或端点 | 层错/界面 |
| 局部刷子修正 | 必须精细评价的区域 |
| 标为成像伪影 | 漂移、失焦、饱和、真空 |

SAM/μSAM 可以用于把点、框、粗线转成可编辑轮廓，但最终标签来源仍应记录为专家修正，而
不是 SAM 自身。

### 6.3 第一阶段不用强化学习，先用可解释排序

主动样本选择先采用：

- 不确定性；
- teacher–student disagreement；
- core-set / embedding 多样性；
- 物理违反；
- 成本权重。

用 Pareto 排序或加权 rank aggregation 即可。它容易做消融，也不会因奖励稀疏而不稳定。

---

## 7. 强化学习应该放在哪里

### 7.1 不建议的用法

不建议定义：

```text
reward = 学生预测与 training-free 标签的一致程度
```

这会使学生复制教师的系统误差，尤其是在：

- 大面积相变破坏“正常多数”假设；
- 扫描畸变模拟位错；
- 厚度/倾斜模拟强度缺陷；
- Fourier 边缘响应模拟线缺陷。

同样不建议让 RL 直接修改 gold 标签或为了提高同一测试集指标不断调阈值。

### 7.2 合理的第一种用法：上下文 bandit 选择标注任务

状态：

```text
样本 embedding + 物理证据 + 学生不确定性 + OOD + 预计标注成本
```

动作：

```text
选择下一张图 / 候选 / 标签类型 / 标注粒度
```

奖励：

```text
独立验证集性能增益
+ 校准改善
+ 新域覆盖
- 专家时间成本
```

这本质上是 contextual bandit，比完整 RL 更符合单步“下一批标什么”的问题。

### 7.3 更有科学价值的第二种用法：自主显微采集

当平台可以连接显微镜后，RL/贝叶斯优化可以选择：

- 下一视野位置；
- 放大倍率；
- dwell time 和电子剂量；
- 是否补拍另一扫描方向；
- 是否切换成分/谱学通道；
- 是否对不确定候选做高分辨复扫。

奖励可以是：

\[
r =
\mathrm{information\ gain}
- \lambda_{\mathrm{dose}}\mathrm{dose}
- \lambda_{\mathrm{time}}\mathrm{time}
- \lambda_{\mathrm{damage}}\mathrm{damage\ risk}
\]

这一层可能形成真正强的新颖性：平台不仅分析已有图像，还能主动获取最能区分缺陷与伪影的
证据。但它需要硬件接口、剂量安全和真实在线实验，不应作为第一阶段的必要条件。

---

## 8. 反事实数字孪生：比普通数据增强更重要

建议为每种结构生成成对数据：

\[
(I_{\mathrm{normal}}, I_{\mathrm{defect}})
\quad\text{under identical nuisance parameters}
\]

两张图使用相同的：

- 厚度；
- 离焦和像差；
- 剂量/Poisson 噪声；
- 扫描漂移；
- 倾斜；
- PSF；
- 探测器响应。

只改变结构缺陷。这样可构造反事实差异，帮助模型区分：

> 成像条件导致的外观变化 vs 原子结构变化。

数字孪生数据至少覆盖：

- 多种晶格对称性和多子晶格；
- 空位、置换、位移、间隙、复合缺陷；
- 层错、位错、孪晶、相界；
- 局部非晶化程度和渐变过渡；
- 未知杂质、束流损伤和低剂量；
- 严格的 normal-only 负样本。

已有工作已经使用 DFT 结构和 multislice 模拟构建数百种缺陷数字孪生，并验证低剂量、
未知杂质和晶界等条件。因此数字孪生本身不是新颖点；我们的贡献应是“反事实配对 +
物理证据图 + 主动选择真实域差距”的组合。

---

## 9. 可信度、校准和拒答

### 9.1 明确三类不确定性

| 类型 | 来源 | 对策 |
|---|---|---|
| 数据不确定性 | 噪声、失焦、剂量、真空、饱和 | 可靠区和 QC flag |
| 模型不确定性 | 学生参数、少样本类别 | ensemble / augmentation consistency |
| 域不确定性 | 新材料、新仪器、新成像条件 | OOD 距离和主动送审 |

### 9.2 单独保留校准集

使用不参与训练和阈值开发的 gold calibration set，对：

- 候选存在性；
- 缺陷类别集合；
- 掩膜边界区间；
- 原子位置误差；

进行温度校准、isotonic calibration 或 conformal prediction。

最终输出应允许：

```text
confirmed_candidate
uncertain_candidate
out_of_domain
invalid_measurement
```

而不是所有像素都必须归入某个缺陷类别。

### 9.3 不能把 conformal 保证说得过头

Conformal prediction 依赖校准和测试数据的交换性。跨仪器或跨材料发生分布漂移时，必须先
触发 OOD，再重新校准；不能声称一个固定阈值对所有实验室永远有效。

---

## 10. 数据与版本治理

建议建立四个永不混用的数据分区：

```text
train/
calibration/
development_test/
locked_external_test/
```

分组必须按“采集序列/样品/实验室”切分，而不是随机裁 patch，否则相邻 patch 会造成严重
泄漏。

每次模型更新记录：

```yaml
model_version:
parent_version:
training_data_hashes:
pseudo_teacher_version:
human_annotation_version:
calibration_set_version:
metrics_before:
metrics_after:
accepted_for_release:
```

基本规则：

1. gold 标签不可被伪标签覆盖；
2. locked external test 永不进入主动学习队列；
3. 每次循环都与固定基线比较；
4. 只有准确率、校准和 OOD 安全指标均不退化时才升级生产模型；
5. 保存失败模型和原因，但不让失败模型继续产生伪标签。

---

## 11. 投稿级验证设计

### 11.1 四条评价轴

#### 检测正确性

- 点缺陷：site-level precision/recall/F1、亚像素定位误差；
- 区域缺陷：IoU、Dice、instance AP；
- 线/界面：boundary F1、平均对称表面距离、拓扑连通性；
- 分类：macro-F1、复合缺陷多标签 AP。

#### 可信度

- ECE、Brier score；
- risk–coverage 曲线；
- conformal empirical coverage；
- OOD AUROC/AUPR；
- 拒答后保留样本的错误率。

#### 数据效率

- 性能—专家标注时间曲线；
- 达到固定性能所需候选数/像素数；
- random、entropy、diversity、disagreement、bandit 的对比；
- 从 0、1%、5%、10%、20% gold 标签开始的学习曲线。

#### 普适性与效率

- leave-one-material-out；
- leave-one-microscope-out；
- leave-one-lab-out；
- 剂量、漂移、旋转、缩放、厚度和失焦鲁棒性；
- CPU/GPU 时间、显存、每百万像素耗时。

### 11.2 必须比较的基线

1. 当前 task-specific training-free 模块；
2. Patterson + OCSVM 等无监督基线；
3. 原子定位/高斯拟合基线；
4. DINO 特征 + 最近邻/one-class；
5. 默认 SAM、μSAM 或适合 EM 的交互分割；
6. 标准 U-Net/Mask R-CNN/现代分割模型；
7. 只用伪标签训练的学生；
8. 只用专家标签训练的学生；
9. 数字孪生训练模型；
10. 完整“证据监督 + 主动学习 + 校准”模型。

### 11.3 关键消融

- 去掉结构证据头；
- 去掉可靠性门控；
- 去掉反事实配对；
- 去掉 symmetry/equivariance loss；
- 去掉 teacher–student disagreement；
- 主动学习换成随机；
- bandit 换成固定策略；
- 去掉 OOD 和拒答；
- 自然图像 DINO vs 显微自监督编码器；
- 统一模型 vs 四个独立模型。

### 11.4 专家用户研究

至少记录：

- 每位专家独立标注时间；
- 接受、拒绝、修正轮廓和修改类型的次数；
- 专家间一致性；
- 先看模型建议是否改变专家偏差；
- training-free、学生和主动学习三个版本的效率；
- 专家最终对科学可解释性的评分。

---

## 12. Nature Methods 目标的现实判断

Nature Methods 的官方范围强调生命科学方法、立即可用性、强验证和重要生物学应用。它也
接受显微、结构生物学和生物数据机器学习方法，但当前“分子筛/材料原子缺陷检测”本身不属于
其最核心范围。

因此有两条路线：

### 路线 A：保持材料科学主线

更自然的高水平目标包括：

- Nature Machine Intelligence；
- Nature Computational Science；
- Nature Communications；
- npj Computational Materials；
- 专业显微学期刊。

这条路线可把“结构证据教师 + 反事实数字孪生 + 自主采集”做到很深。

### 路线 B：真正按 Nature Methods 设计

需要增加一个重要生命科学应用，并证明平台能处理生物电子显微中的重复结构。例如：

- cryo-EM / cryo-ET 中微管晶格缺口、protofilament 变化或修复位点；
- 病毒衣壳装配缺陷；
- 细菌 S-layer 周期结构异常；
- 纤毛/轴丝周期结构和局部断裂；
- 膜蛋白二维晶格或其他规则超分子结构。

此时统一对象不再局限为“无机晶格”，而是“可由参考结构解释的重复纳米结构”。必须与生物
实验团队合作，并展示一个该方法促成的新生物学发现，而不只是把材料代码跑在一张生物图上。

如果没有生物合作数据，建议不要为了期刊名称勉强包装成 Nature Methods；先把方法做到
Nature Machine Intelligence / Nature Computational Science 级别更合理。

---

## 13. 建议的论文主张

不要把主张写成：

> 我们组合了 Fourier、DINO、SAM、主动学习和强化学习。

这会被认为是模块堆叠。

更好的主张是：

> We introduce a physics-evidence teacher that converts unlabeled microscopy
> images into reliability-aware structural supervision, enabling a student
> model to learn across point, line, regional and interfacial defects. A
> disagreement-driven expert loop continuously expands domain coverage while
> calibrated abstention prevents self-confirming errors.

对应三项核心贡献：

1. **统一结构证据图**：把点、线、区域和界面缺陷表示为节点—边—区域的结构违反；
2. **证据保持式学习**：学生不仅学习掩膜，还学习教师物理证据和可靠性；
3. **可审计自提升闭环**：基于专家—教师—学生分歧主动学习，并通过独立校准和锁定测试集
   防止伪标签自我强化。

数字孪生和 RL 是增强贡献，不应遮盖这三个核心。

---

## 14. 推荐实施顺序

### Phase 0：确定论文边界（2 周）

- 决定材料方法路线还是 Nature Methods 生物路线；
- 定义目标用户和核心科学问题；
- 冻结当前 v0.2 代码、基准和结果作为不可变 baseline；
- 写出一句论文主张和三条不可替代贡献。

**停止条件：** 如果核心贡献仍只能描述为“组合已有模块”，不要开始大规模训练。

### Phase 1：统一证据层（4–6 周）

- 新增 `EvidenceBundle`、`ReliabilityBundle`、`DefectHypothesis`；
- 把四任务现有 evidence maps 映射到节点—边—区域协议；
- 建立统一 ontology；
- 让统一假设引擎在当前四任务上达到不低于原 task-specific 版本 95% 的性能；
- 增加跨任务诊断和公共置信字段。

**这是最优先的代码工作。**

### Phase 2：gold 数据和校准（6–10 周，可与 Phase 1 并行收集）

- 建立版本化数据仓和标签 provenance；
- 制定多专家标注协议；
- 建立 train / calibration / locked test；
- 收集跨材料、剂量和仪器数据；
- 实现 OOD、risk–coverage 和初步 conformal 校准。

**停止条件：** 没有独立 gold test 时，不进入“持续提高精度”的结论。

### Phase 3：物理证据学生（6–10 周）

- 先训练小型共享编码器；
- 实现几何、缺陷、证据、可靠性四头；
- 比较 hard pseudo label、soft pseudo label 和 evidence distillation；
- 验证统一学生是否优于四个独立学生；
- 保留 training-free 教师作为回退和审计路径。

### Phase 4：主动学习与专家工具（6–8 周）

- 候选队列和低成本修正 UI；
- random / entropy / diversity / teacher–student disagreement 对比；
- 以专家时间而非只以标注数量评估；
- 在多个新域做前瞻性循环，而不是离线重复抽样。

### Phase 5：反事实数字孪生（8–12 周）

- 构建同 nuisance 的 normal/defect 对；
- 系统覆盖剂量、漂移、离焦、厚度和未知杂质；
- 用真实 OOD 失败反向指导模拟参数覆盖；
- 禁止使用 locked test 调数字孪生。

### Phase 6：bandit / 自主采集（可选，12 周以上）

- 先用 contextual bandit 选择标注；
- 有显微镜接口后再做视野、剂量和复扫策略；
- 引入安全约束和剂量预算；
- 与随机扫描、网格扫描、Bayesian optimization 比较。

---

## 15. 未来六周最值得做的具体任务

优先级从高到低：

1. 冻结 `v0.2-training-free-baseline` 的代码、配置、数据清单和指标；
2. 写出统一缺陷 ontology 和 `EvidenceBundle` schema；
3. 把四任务 evidence maps 转换到统一节点—边—区域命名；
4. 实现统一 reliability、invalid 和 abstain 语义；
5. 建立候选级数据表和 provenance；
6. 设计专家标注协议及 50–100 个 pilot 候选的双人复核；
7. 做第一个 evidence-preserving student，与 hard pseudo-label student 对比；
8. 在确定 gold/calibration/locked split 后，才实现主动选择；
9. 暂不做完整 RL；
10. 同时寻找跨实验室材料数据或生物 EM 合作数据。

第一阶段成功的最低判据建议为：

- 四任务都能通过统一 evidence schema 表达；
- 统一引擎不明显退化；
- confidence 与 invalid/reliability 明确分离；
- gold、silver、unresolved 三类标签有版本和来源；
- 学生在未见材料上优于只学硬伪标签的基线；
- 主动选择在相同专家时间下优于随机选择；
- 模型在 OOD 时能够增加拒答，而不是更加自信。

---

## 16. 文献定位与我们必须超越的工作

1. [Nature Methods Aims & Scope](https://www.nature.com/nmeth/aims)：强调生命科学、
   强方法验证、实际应用和重要生物学问题。
2. [Cellpose 2.0](https://doi.org/10.1038/s41592-022-01663-4)：已经实现 human-in-the-loop
   标注与快速再训练，说明“加人工闭环”本身不是新颖点。
3. [Segment Anything for Microscopy / μSAM](https://doi.org/10.1038/s41592-024-02580-4)：
   已经覆盖 LM/EM、交互/自动分割、用户微调和用户研究，显示投稿级平台需要广泛比较和
   可用工具。
4. [Guo et al., unsupervised atomic defect detection](https://doi.org/10.1038/s41524-021-00642-1)：
   使用 Patterson 与 OCSVM 检测点、线和孪晶缺陷，是当前平台需要明确比较的无监督基线。
5. [Digital twins and deep-learning segmentation of defects](https://doi.org/10.1063/5.0181080)：
   已有约 600 个 DFT 缺陷结构及 multislice 模拟，数字孪生本身不能单独作为新颖性。
6. [SIGMA-ML](https://doi.org/10.1038/s41524-025-01756-6)：已经使用真实 OOD 失败指导
   合成训练数据更新，并报告自调节流程带来的性能提升。
7. [FAST autonomous scanning](https://doi.org/10.1038/s41467-023-40339-1)：展示神经网络、
   路径规划和仪器控制的自主高分辨扫描，说明 RL/主动采集需要真实仪器与剂量收益验证。
8. [Conformal semantic image segmentation](https://openaccess.thecvf.com/content/CVPR2024W/SAIAD/html/Mossina_Conformal_Semantic_Image_Segmentation_Post-hoc_Quantification_of_Predictive_Uncertainty_CVPRW_2024_paper.html)：
   提供分割预测的事后不确定性量化思路，可作为候选集和拒答校准基线。

---

## 17. 最后的优先判断

如果只能选一件事，下一步应当做：

> **统一结构证据图 + 带 provenance 的 gold/silver/unresolved 数据协议。**

因为：

- 没有统一证据图，学习层只是把四个独立算法的输出混在一起；
- 没有数据来源和锁定测试，主动学习会变成不可审计的自我强化；
- 没有可靠性和拒答，强化学习只会放大现有置信分数的缺陷；
- 一旦这两层稳固，学生模型、SAM 辅助、主动学习、数字孪生和自主扫描都能自然接入。

强化学习是后续可能很强的一层，但不是当前最重要的一层。当前最有论文价值的转折，是让
training-free 检测器从一组规则升级为一个统一、可解释、可校准、能教授学习模型的
**physics-evidence teacher**。

---

## 18. 2026-07-26 实施检查点

本节只记录路线图执行状态，不改变上述预注册式科学标准。

### 已经落地

- Phase 0–1：v0.2 基线冻结、统一 EvidenceBundle、可靠性/拒答语义和四任务非退化门禁；
- Phase 2 软件：版本化 registry、sealed intake plan、license/group/checksum 门禁、
  Gold-only calibration、OOD、conformal 和 locked release gate；
- Phase 3 软件：evidence-preserving student 与 hard-pseudolabel 合成对照；
- Phase 4 软件：train-only 主动排序、assisted/blinded UI、双 reviewer 共识和不可变 ledger；
- Phase 5 软件：四任务同 nuisance 解析 counterfactual twins；
- 一次真正的预注册外测：Dennler III–V 归档已一次性揭盲，任意专家缺陷 IoU 0.1396，
  诚实否定了当前解析教师的广泛零样本迁移；
- 一个真实 train-domain 扩展：107 张 EMSeek WSe₂ 图像按一个保守 group 准入；新增周期
  反事实教师，并与高召回位点教师形成双物理教师共识。当前 site precision 0.8914、
  recall 0.6674、F1 0.7633，所有不一致点保留为主动审阅池。
- Phase 4 执行准备：从 69 张图 label-blind 地抽取 100 个 opaque 候选，平衡共识、
  periodic-only、legacy-only 和 random sham prompt，封存 assisted/blinded bundle 和
  私有分层 key；真实专家 response 仍为 0。

详细实现证据见
[`implementation_status_zh.md`](implementation_status_zh.md) 和
[`periodic_counterfactual_and_consensus_zh.md`](periodic_counterfactual_and_consensus_zh.md)。

### 仍未满足，因此路线图目标尚未完成

- EMSeek acquisition/sample 元数据不足，107 对数据仍是单一 train group；
- 作者标注 reviewer 数未知，不能升级为本平台定义的 Gold；
- `calibration=0`，没有真实正常对照支持概率、risk 或 conformal 覆盖主张；
- evidence student 尚未在真实数据上完成预注册训练和独立校准；
- 100-candidate pilot 已封存但尚无双专家 response，不能把“包已生成”算成 pilot 完成；
- Dennler 已消费，任何改进后的模型都必须使用全新的 locked external test；
- 解析 twins 还不是实验参数验证过的 multislice；
- contextual bandit/RL 继续有意延后，直到多轮真实“选择—专家成本—独立增益”可识别。

因此当前最短路径已经从“继续添加 detector”转为：

```text
新 acquisition/sample-resolved 数据
        ↓
双专家 Gold calibration + normal controls
        ↓
冻结真实 student/active-learning 对照
        ↓
全新材料/仪器/实验室 locked external test
        ↓
通过 accuracy + calibration + risk–coverage + OOD 门禁
```

在这条链路完成前，WSe₂ 数字只能称为 train-domain development evidence，不能作为跨材料
普适性或持续自提升的最终证明。
