可以，而且我认为这条路线**技术上可行**。但最可靠的架构并不是直接做：

> Grounding DINO 输入“atomic vacancy” → SAM 分割缺陷

而是把问题重新定义为：

> **给定理想晶格骨架，对实际 STEM 图像进行配准，检测相对于理想骨架的稀疏结构残差。**

理想晶格、子晶格类型、配位关系和附着原子位置，是比自然语言提示强得多的先验。Grounding DINO、SAM、DINOv3 应当作为辅助感知模块，而不是最终物理判据。

我建议把整个系统设计成一个：

> 实现状态（2026-07-25）：该方案已经落地为统一四任务平台。除原有
> `amorphous_region` 外，现已实现 `atomic_point_defect`、
> `stacking_fault` 和 `phase_boundary`，加入自动多子晶格推断、平滑非刚性
> 配准、倒易载波相位滑移、两域 Fourier 签名分割、公开数据清单和统一基准。
> 公开实验验证使用 `10.5281/zenodo.5520169`，完整操作与限制见 `README.md`。

## Skeleton-Conditioned Training-Free Atomic Defect Detector

即“**骨架条件化、冻结权重、无需标注训练的原子缺陷检测器**”。

---

# 一、先明确什么叫 training-free

这里建议采用严格且可实现的定义：

* 不需要人工标注数据集；
* 不对 Grounding DINO、SAM、DINOv3 等模型做反向传播或微调；
* 所有预训练网络权重冻结；
* 允许对当前图像做晶格配准、PSF 拟合、鲁棒统计、PCA、最近邻检索等数值计算；
* 可以使用用户提供的一张正常骨架和一个缺陷示例。

严格来说，PCA、OCSVM 会“拟合”当前图像的统计分布，但没有训练神经网络。若要求连这种统计拟合也不要，可以退化为模板残差、归一化互相关和 MAD 阈值。

---

# 二、为什么不能把 Grounding DINO 作为原子缺陷主检测器

Grounding DINO 本质上是图像—文本开放词汇检测器，输入图像和文本，输出与文本匹配的目标框。它适合检测“图中存在的对象”，而空位、缺失原子、断裂配位等往往恰恰是“某个对象不存在”。([arXiv][1])

对于原子分辨 STEM，还有三个特殊问题：

1. **原子列极小且高度重复。**
   在整幅图缩放到基础模型输入尺寸后，单个原子列可能只占很少的特征单元。

2. **缺陷语义不等于自然图像语义。**
   模型可能理解“hole”“broken object”，但不一定理解某种特定晶相中某个子晶格的硫空位或配位氧缺失。

3. **SAM 会被提示诱导。**
   即使一个预期原子位置实际上是空的，向 SAM 给出正点提示，它仍可能在噪声或邻近原子上产生一个掩膜。因此，SAM 的掩膜不能直接作为“该原子存在”的证据。

截至 2026 年，SAM 3 已支持文本、点、框、掩膜和图像示例提示，并能尝试找出所有与一个示例概念相匹配的实例，因此它比传统的 Grounding DINO + SAM 更适合“给一个缺陷例子，搜索类似缺陷”的场景。但在原子分辨图像中，它仍应当只负责**候选生成和轮廓支持**，不能代替晶格与强度判据。([GitHub][2])

---

# 三、推荐的总体结构

```text
16-bit STEM 原图
      +
理想晶格骨架/晶胞/子晶格/配位边
      +
可选：一个已知缺陷示例
      │
      ▼
定量预处理与扫描畸变校正
      │
      ▼
FFT 初始晶格估计 + 原子柱亚像素定位
      │
      ▼
理想骨架 → 实际图像的鲁棒非刚性配准
      │
      ├───────────────┐
      ▼               ▼
物理生成模型残差       DINOv3 特征异常分支
      │               │
      ├───────────────┤
      ▼               ▼
晶格图匹配            SAM3 局部提示/示例检索
      │               │
      └───────融合─────┘
              │
              ▼
空位 / 置换 / 偏移 / 间隙原子 /
附着原子缺失 / 位错 / 相界 / 成像伪影
```

其中真正的主干是：

> **晶格配准 + 分析式图像生成 + 稀疏残差 + 图匹配**

SAM 和 DINO 是增强器。

---

# 四、第一步：把“规整骨架”表示为晶格图

最好不要只给一张骨架图片，而是转成结构化表示：

[
S=(V,E,B,\tau,\mathcal G)
]

其中：

* (V)：理想原子或原子柱节点；
* (E)：预期配位、键或附着关系；
* (B=[b_1,b_2])：二维晶格基矢；
* (\tau_i)：节点所属子晶格、元素类型或角色；
* (\mathcal G)：允许的旋转、镜像等晶体对称操作。

每个节点可以包含：

```text
site_id
fractional_coordinate: [u, v]
sublattice: A / B / O / attachment
expected_species
expected_relative_intensity
expected_neighbors
optional_or_required
```

对于你说的“骨架附着原子”，尤其应该显式标记：

```text
backbone node ── attachment node
```

这样检测就不再只是判断某处亮不亮，而可以判断：

> 主骨架原子仍存在，但它所对应的附着节点或配位边消失了。

---

# 五、第二步：原子柱定位与骨架配准

## 1. 保留两条图像链路

建议同时保留：

* **定量链路**：原始 16-bit/32-bit 强度，不做破坏强度关系的对比度拉伸；
* **视觉模型链路**：百分位归一化后复制成三通道，供 SAM、DINO 使用。

不要用生成式超分辨或扩散去噪作为定量检测输入，因为它可能生成不存在的原子峰。

## 2. 初始晶格估计

利用 FFT 或自相关找到倒易晶格峰，估计：

* 晶格方向；
* 晶格间距；
* 全局旋转；
* 缩放比例；
* 初始平移。

## 3. 原子柱候选

先用传统且可解释的方法：

* LoG/DoG；
* 局部极值；
* 匹配滤波；
* 二维高斯拟合；
* 多高斯联合拟合。

Atomap 就是针对原子分辨 STEM 的开源工具，可以通过二维高斯拟合定位原子柱并提取位置、形状和强度；其定量模块还支持高斯强度以及 Voronoi、watershed 积分。对于亚像素位置和强度，它通常比直接依赖 SAM 掩膜更合适。([atomap.org][3])

## 4. 鲁棒非刚性配准

理想坐标 (q_i) 到图像坐标的映射可写成：

[
\phi(q)=Aq+b+d(q)
]

其中：

* (Aq+b)：全局仿射变换；
* (d(q))：由扫描漂移、样品弯曲和局部应变造成的平滑形变场。

拟合时必须使用 Huber、Tukey 或 RANSAC 一类鲁棒损失，并迭代排除异常点。否则算法会为了“解释”一个空位而把整个晶格拉弯。

匹配可采用带空节点的二分图匹配：

* 预期节点没有对应观测峰：空位候选；
* 观测峰没有对应预期节点：间隙原子或吸附原子候选；
* 匹配成功但距离异常：位移原子候选。

---

# 六、最关键的一步：分析式生成与残差，而不是直接分割

根据已配准的骨架，在图像中生成“该区域如果完全正常应该长什么样”。

可以使用如下模型：

[
\hat I(x)=b(x)+\sum_i a_{\tau_i}(x),
h_{\theta_i}\bigl(x-\phi(q_i)\bigr)
]

其中：

* (b(x))：缓慢变化的背景；
* (a_{\tau_i}(x))：对应子晶格的局部原子柱幅度；
* (h_{\theta_i})：从正常原子柱估计出的 PSF 或二维高斯/椭圆高斯；
* (\phi(q_i))：骨架配准后的预期位置。

这里的幅度不要设成全局常数，而应允许随位置缓慢变化，以消除厚度、照明、探测器响应和样品倾斜造成的强度梯度。

得到残差：

[
R(x)=I(x)-\hat I(x)
]

不同缺陷在残差中会产生不同的结构：

| 缺陷类型           | 典型残差信号                |
| -------------- | --------------------- |
| 空位             | 预期位置出现负的 PSF 形残差      |
| 置换、掺杂          | 位置正常，但中心幅度正偏或负偏       |
| 原子位移           | 一侧正、一侧负的偶极残差          |
| 间隙原子、吸附原子      | 非预期位置出现正的局部峰          |
| 附着原子缺失         | 主骨架正常，但附着节点位置出现负残差和缺边 |
| 分裂原子柱、dumbbell | 单峰模型拟合差，双峰模型显著更优      |
| 位错、晶界          | 连续区域的几何残差和配位数同时异常     |

这相当于一个“analysis by synthesis”过程：不是询问模型“这是不是缺陷”，而是比较：

> 正常晶格能否解释这块观测数据？

---

# 七、可以直接用局部模型选择完成缺陷分类

对于每个预期原子位置，可拟合几个竞争模型：

[
\begin{aligned}
M_0 &: \text{正常原子，位置和幅度受约束}\
M_v &: \text{空位，幅度为 0}\
M_s &: \text{置换，位置正常但幅度自由}\
M_d &: \text{位移，幅度正常但位置自由}\
M_2 &: \text{双峰或分裂原子柱}\
M_e &: \text{增加一个额外峰}
\end{aligned}
]

比较各模型的鲁棒残差和 BIC/AIC：

[
\mathrm{BIC}_m
==============

n\log\left(\frac{\mathrm{RSS}_m}{n}\right)
+
k_m\log n
]

这会给出：

* 最可能的缺陷类别；
* 相对于正常模型的证据强度；
* 亚像素位置、幅度、宽度和不确定度。

全程不需要训练分类网络。

---

# 八、SAM 系列应当怎么接入

## 1. 不要对整张原子图运行自动分割

更合理的做法是：

1. 根据骨架自动生成每个预期原子的正点提示；
2. 在邻近原子或晶格间隙放负点；
3. 给出一个只包含局部一至数个晶胞的小框；
4. 对每个局部窗口做批量 SAM 解码；
5. 比较多种提示组合下掩膜是否稳定。

MatSAM 已证明，“先利用材料结构先验自动生成提示点，再调用 SAM”比直接对显微图像应用原始 SAM 更合理；其工作主要针对 SEM/光镜尺度的相和晶界，而原子尺度需要把提示生成改成晶格骨架驱动。([arXiv][4])

## 2. 对 vacancy 不要相信正点掩膜

空位没有可分割实体。因此：

* SAM 掩膜只能告诉你局部纹理支持是否稳定；
* 原子是否存在，必须由原始强度、PSF 拟合和匹配结果决定；
* 若正点、框、负点稍微变化，掩膜就大幅变化，应将该结果标为低可信。

## 3. 使用 SAM 3 的示例提示搜索已知缺陷

用户只标一个缺陷时，不要只截取“缺失的空白中心”，而应该截取：

* 缺陷中心；
* 一至两圈近邻；
* 主骨架和附着原子的相对关系；
* 缺陷的锚点位置。

SAM 3 可以用图像示例提示寻找相似概念实例，但输出必须再经过晶格位置、残差和图结构过滤。([GitHub][2])

## 4. μSAM 可作为备选

μSAM 提供针对光学和电子显微图像微调过的模型，并支持点、框、自动实例分割和 tiled embedding。它在一般 EM 数据上的域适应程度高于原始 SAM，但其论文结果并不能直接证明它能可靠处理原子分辨晶格，因此仍需要实测。([Nature][5])

---

# 九、DINOv3 最适合做“异常特征分支”

DINOv3 的优势是密集视觉特征，而不是直接输出缺陷类别。官方工作强调其冻结特征在多种视觉任务上的迁移能力。([AI.Meta][6])

推荐流程如下：

1. 将每个晶胞或原子局部窗口旋转到统一晶格方向；
2. 按子晶格分别处理；
3. 从几何残差最低的一批晶胞中构造“正常原型”；
4. 提取冻结的 DINOv3 patch token；
5. 计算候选晶胞到正常特征子空间或正常原型的距离。

一种简单形式是：

[
z_i^{\mathrm{feat}}
===================

\left|
f_i-\mu-UU^\top(f_i-\mu)
\right|_2
]

其中 (U) 是正常晶胞特征的低维 PCA 子空间。

近期的 SubspaceAD 正是采用冻结的 DINOv2 patch 特征和 PCA 正常子空间，在不训练神经网络的情况下通过重建残差做少样本异常检测。这里可以借用它的结构，把输入单元从普通工业图像 patch 换成**已经按晶格配准的晶胞 patch**，并测试 DINOv3 是否优于 DINOv2。([arXiv][7])

需要强调：DINO 特征是辅助证据。对于非常微弱的低 Z 原子缺失，原始积分强度和物理残差往往比自然图像预训练特征更敏感。

---

# 十、用户给一个缺陷示例时，应该比较“缺陷残差”，而不是原图

原始晶胞绝大部分内容都是重复晶格，直接比较两个原图 patch，模型主要比较到的是正常晶格。

更好的表示是：

[
P_i^\Delta
==========

P_i-P_{\tau_i}^{\mathrm{normal}}
]

以及特征残差：

[
f_i^\Delta
==========

f(P_i)-f(P_{\tau_i}^{\mathrm{normal}})
]

对于用户标记的缺陷示例 (d)，计算：

[
s_i=
\max_{g\in\mathcal G}
\left[
\alpha,\mathrm{NCC}
\left(P_i^\Delta,gP_d^\Delta\right)
+
\beta,\cos
\left(f_i^\Delta,g f_d^\Delta\right)
------------------------------------

\gamma D_{\mathrm{graph}}
\right]
]

其中：

* (\mathcal G)：晶体允许的旋转和镜像；
* NCC：残差图归一化互相关；
* 余弦相似度：DINO 特征残差相似度；
* (D_{\mathrm{graph}})：局部晶格图编辑距离。

这个方法可以理解为：

> 先减去“大家都有的正常晶格”，再寻找与给定缺陷相同的破坏模式。

对于附着原子破坏，这通常比直接对整个晶胞做视觉相似检索更有效。

---

# 十一、加入 Patterson 描述子处理裁剪偏移

即使晶格已经配准，局部晶胞裁剪仍可能存在几个像素的偏移。可以对每个晶胞计算 Patterson 图，也就是局部图像的自相关或傅里叶幅度平方。

已有工作使用 Patterson 函数消除晶胞裁剪中的平移误差，再用 OCSVM 检测点缺陷、线缺陷和孪晶界；其代码和数据也公开。([Nature][8])

在本方案中可以采用两种方式：

* 严格 training-free：计算 Patterson 图到正常中位模板的距离；
* 较宽松的无标注模式：PCA + OCSVM 或 PCA 子空间异常分数。

Patterson 描述子会弱化平移影响，但因为其主要保留原子间矢量关系，可能混淆部分镜像或方向信息，所以应和规范化原图 patch、位移方向及晶格图一起使用。

---

# 十二、无训练的证据融合

每个原子位点最终可以得到：

[
F_i=
{
f_{\mathrm{occupancy}},
f_{\mathrm{position}},
f_{\mathrm{amplitude}},
f_{\mathrm{shape}},
f_{\mathrm{residual}},
f_{\mathrm{graph}},
f_{\mathrm{DINO}},
f_{\mathrm{SAM}}
}
]

所有特征应当按照子晶格和局部区域做鲁棒归一化：

[
z_{ik}
======

\frac{
|f_{ik}-\operatorname{median}(f_{jk})|
}{
1.4826,\operatorname{MAD}(f_{jk})+\epsilon
}
]

不要用整张图的一个全局强度阈值，因为厚度、倾斜和 illumination gradient 会让不同区域的正常原子强度不同。

无需学习融合权重，可以采用：

* 可靠度加权的 top-3 平均；
* 物理残差、图结构和 DINO 三分支投票；
* “两种独立证据一致”才标高置信。

推荐置信等级：

* **高置信**：物理模型和晶格图同时异常；
* **中置信**：物理异常，同时 DINO/SAM 支持；
* **待复核**：只有基础模型认为异常；
* **成像伪影嫌疑**：异常不依附于任何晶格节点，或呈现扫描条纹、整行偏移。

---

# 十三、缺陷类型可以用规则直接解释

| 分类        | 判据                         |
| --------- | -------------------------- |
| 空位        | 预期节点未匹配，积分强度低，负残差明显        |
| 附着原子缺失    | 主骨架匹配正常，附着节点缺失，局部图度数减少     |
| 置换/掺杂     | 位置正常，幅度或散射强度显著异常           |
| 位移原子      | 原子存在但位置残差大，出现偶极残差          |
| 间隙原子/吸附原子 | 非预期位置出现额外峰，无法匹配到骨架         |
| 分裂原子柱     | 双峰模型优于单峰模型                 |
| 位错        | 配位数、边长、Burgers 回路或局部晶格相位异常 |
| 相界/孪晶界    | 连续区域由另一个晶格骨架解释得更好          |
| 扫描伪影      | 同一行或方向上大量位点同时异常，且与晶格拓扑无关   |

特别重要的是：**相变区域不能被误报为成百上千个点缺陷**。应先允许多个候选晶相骨架竞争拟合；若一整片区域由第二套骨架解释得更好，应分类为相域或相界。

---

# 十四、如何做到快速

不要对数千个原子逐个完整运行大模型。采用两阶段结构：

## 第一阶段：全图廉价筛选

对所有位点计算：

* 配准残差；
* 积分强度；
* 高斯拟合残差；
* 图匹配；
* 生成模型残差。

这一步主要是 FFT、KD-tree、局部拟合和稀疏图运算。

## 第二阶段：只验证可疑位点

仅对异常分数较高或不确定的候选：

* 提取 DINOv3 特征；
* 运行 SAM3 局部点/框提示；
* 与用户提供的缺陷示例比较。

同时：

* 每个大 tile 只计算一次 SAM/DINO 图像 embedding；
* 所有点提示批量解码；
* tile 之间保留至少一个晶胞的重叠；
* 原子位点匹配使用 KD-tree 局部门控，不构造全局 (N\times N) 代价矩阵。

若确实需要 Grounding DINO，可以只用于粗尺度 ROI，并使用切片推理。官方 Grounded-SAM-2 也为高分辨率密集小目标提供了 SAHI 切片方案。但 Grounding DINO 1.5/1.6 在该官方链路中需要 API token；要做完全离线、可复现实验，应优先采用原始本地 Grounding DINO、SAM3 本地模型，或者干脆不使用 Grounding DINO。([GitHub][9])

---

# 十五、我最推荐的第一版系统

第一版不要从 Grounding DINO 开始，按照下面的优先级实现：

1. **Atomap/二维高斯原子柱定位**；
2. **FFT + RANSAC + 平滑形变场完成骨架配准**；
3. **根据正常原子估计 PSF，渲染理想 STEM 图并计算残差**；
4. **预期节点—观测峰带空节点匹配**；
5. **局部晶格图编辑和附着边缺失检测**；
6. **MAD/经验分位数输出异常分数**；
7. **DINOv3 做未知异常排序**；
8. **SAM3 用于局部轮廓和一个缺陷示例的相似实例搜索**；
9. **Grounding DINO 仅用于晶体区域、颗粒边界或污染区域的粗定位**。

最终每个检测结果建议输出：

```json
{
  "site_id": 1824,
  "position_pixel": [1213.4, 834.8],
  "position_physical": [12.73, 8.76],
  "sublattice": "attachment_atom",
  "defect_type": "missing_attachment",
  "confidence": 0.94,
  "occupancy_ratio": 0.18,
  "displacement_normalized": 0.07,
  "graph_degree_delta": -1,
  "physics_residual_score": 6.2,
  "dino_similarity_to_exemplar": 0.87,
  "sam_mask_stability": 0.41
}
```

---

# 十六、这条路线的真实边界

这套方法可行的前提是：

* 原子柱在图像中实际可分辨；
* 提供的晶相、投影方向和骨架基本正确；
* 正常区域占多数，或者能指定一块完整参考区域；
* 缺陷会造成可测的强度、位置、形状或配位变化。

它不能自动解决以下信息论上的问题：

* 多个三维结构在二维投影中完全等价；
* 原子缺失的信号低于噪声和仪器 PSF；
* 厚度、倾斜、焦差造成的强度变化与置换原子完全不可区分；
* 高缺陷密度导致图中不存在足够的“正常多数”；
* 不同晶相混在一起但只提供了一套骨架。

因此最稳妥的结论是：

> **用骨架和物理残差完成“是否异常”的判断，用图结构完成“哪里断了”的解释，用 SAM3 和 DINOv3完成候选补充、示例检索和复杂纹理验证。**

这会比单独使用 Grounding DINO + SAM 更快、更可解释，也更接近真正的 training-free 原子缺陷检测。下一步最有价值的输入是一张未经压缩的 16-bit STEM 原图、像素标尺、成像模式以及一份带子晶格和附着边的骨架坐标。

[1]: https://arxiv.org/abs/2303.05499?utm_source=chatgpt.com "Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection"
[2]: https://github.com/facebookresearch/sam3 "GitHub - facebookresearch/sam3: The repository provides code for running inference and finetuning with the Meta Segment Anything Model 3 (SAM 3), links for downloading the trained model checkpoints, and example notebooks that show how to use the model. · GitHub"
[3]: https://atomap.org/?utm_source=chatgpt.com "Welcome to Atomap's documentation! — Atomap 0.4.3.dev1 ..."
[4]: https://arxiv.org/html/2401.05638v2?utm_source=chatgpt.com "MatSAM: Efficient Extraction of Microstructures of Materials ..."
[5]: https://www.nature.com/articles/s41592-024-02580-4 "Segment Anything for Microscopy | Nature Methods"
[6]: https://ai.meta.com/research/publications/dinov3/ "DINOv3 | Research - AI at Meta"
[7]: https://arxiv.org/html/2602.23013v2?utm_source=chatgpt.com "Training-Free Few-Shot Anomaly Detection via Subspace ..."
[8]: https://www.nature.com/articles/s41524-021-00642-1 "Defect detection in atomic-resolution images via unsupervised learning with translational invariance | npj Computational Materials"
[9]: https://github.com/idea-research/grounded-sam-2 "GitHub - IDEA-Research/Grounded-SAM-2: Grounded SAM 2: Ground and Track Anything in Videos with Grounding DINO, Florence-2 and SAM 2 · GitHub"
