# Open-world v0.4 跨数据集测试与可视化

日期：2026-07-27  
性质：固定样本的 development / smoke test，不是新的 locked external benchmark

## 先看总图

[跨数据集总览图](../outputs/open_world_dataset_gallery_v04_contours/cross_dataset_overview.png)

![跨数据集总览图](../outputs/open_world_dataset_gallery_v04_contours/cross_dataset_overview.png)

这次统一测试了 PNG、TIFF、NData stack 和多通道 EMD 四种输入形式。每一行都使用同一
显示规则：

1. 左：固定且未修改的输入；
2. 中：在原子图上画候选及 resolver 状态；
3. 右：在同一原子图上半透明叠加主要物理证据。

颜色含义：

- 绿色：`probable`，当前证据较支持某个物理解释；
- 橙色：`unresolved`，候选存在，但 subtype 证据不够；
- 品红色：`artifact`，伪影解释比当前物理解释更强；
- 青色轮廓：detector 产生的候选几何范围。

原子点任务用小圆圈标出位点；非晶区、线缺陷和界面任务则沿真实 candidate mask 画轮廓，
不再用覆盖正常晶区的 bounding box。右图的青色轮廓显示同一候选几何。

## 测试协议

本轮没有按结果挑图：

- 分子筛：用户给出的唯一 reference；
- EMSeek WSe₂：数值编号最小的 `image/1.png`；
- Al 位错和 ZrO₂ twin：公开数据中带明确语义文件名的完整二维图；
- Guo SuperScan：预先固定 frame 0；
- Al Σ3：预先固定字典序第一个 EMD image UUID，并对 30 帧做 mean projection。

本轮没有训练或拟合神经网络。每张图只进行 detector 自带的单图统计估计，例如晶格基矢、
原子峰和稳健阈值。候选数量是管线输出；没有候选级 Gold 时，它不是精度。

完整机器可读摘要：
[`gallery_summary.json`](../outputs/open_world_dataset_gallery_v04_contours/gallery_summary.json)。

## 六个样本分别说明

| 数据 / 任务分支 | 输出 | resolver 结果 | 应当怎样读 |
|---|---:|---|---|
| 拉伸分子筛 / 非晶区 | 3 个候选，面积 1.880% | 2 `probable`，1 `artifact` | 两个候选被解释为 `local_amorphous_region`；中间较清晰度相关的区域被分到 `defocus_or_blur`。没有像素 Gold，不能由此计算非晶 IoU。 |
| EMSeek WSe₂ image 1 / 原子点 | 312 个候选，面积 1.493% | 312 `probable` | subtype 假设为 254 vacancy-like、41 displaced-column、17 intensity anomaly；这些是 resolver 假设，不是作者提供的 subtype 真值。 |
| Guo Al partial dislocation / 线缺陷 | 2 个几何候选，面积 0.776% | 2 `artifact` | detector 抓到了中心相位跳变，但 `scan_or_coherence_artifact` 得分超过 `stacking_fault_or_dislocation_line`。这是当前 resolver 的真实失败/保守拒绝，不应包装成正确分类。 |
| Guo monoclinic ZrO₂ twin / 相界面 | 1 个候选，面积 1.319% | 1 `unresolved` | 界面几何清楚，但目前只有局部倒空间签名，尚不足以在 twin、heterophase 和 orientation-domain 中定 subtype。 |
| Guo SuperScan frame 0 / 原子点 | 43 个候选，面积 1.855% | 43 `probable` | 证明 stack 读取、显式选帧和原子位点链路可运行；没有候选级 Gold，因此不是 43 个“正确缺陷”。 |
| Al Σ3 ORI EMD / 相界面 | 2 个候选，面积 4.684% | 2 `unresolved` | 只证明多通道 EMD 的显式选 channel + projection + detector 流程能跑通。UUID 尚未映射到真实 detector 信号，所以必须标为 smoke-only。 |

对应的原子相保留诊断图：

- [拉伸分子筛](../outputs/open_world_dataset_gallery_v04_contours/figures/zeolite_local_amorphization.png)
- [EMSeek WSe₂ image 1](../outputs/open_world_dataset_gallery_v04_contours/figures/emseek_wse2_image_1.png)
- [Guo Al partial dislocation](../outputs/open_world_dataset_gallery_v04_contours/figures/guo_al_partial_dislocation.png)
- [Guo monoclinic ZrO₂ twin](../outputs/open_world_dataset_gallery_v04_contours/figures/guo_zro2_twin_boundary.png)
- [Guo SuperScan frame 0](../outputs/open_world_dataset_gallery_v04_contours/figures/guo_superscan_frame_0.png)
- [Al Σ3 ORI EMD](../outputs/open_world_dataset_gallery_v04_contours/figures/al_sigma3_ori_channel_0_mean.png)

每张诊断图的六个格子依次是：原图、候选状态、测量可靠性、主要 detector 证据、最高分解释、
第二解释或作者参考点。所有标量图都画在同一幅原子图上，而不是把特征矩阵误当成新的
显微图。

## WSe₂ 作者点参考检查

[WSe₂ image 1 一对一参考匹配图](../outputs/open_world_dataset_gallery_v04_contours/wse2_image1_reference_alignment.png)

![WSe₂ image 1 一对一参考匹配图](../outputs/open_world_dataset_gallery_v04_contours/wse2_image1_reference_alignment.png)

用当前图像推断出的 `5.43 px` 物理容差和 `site_matching_v2` 最大基数优先的一对一规则：

| 项目 | 数值 |
|---|---:|
| detector 候选 | 312 |
| 作者参考点 | 167 |
| 一对一匹配 | 160 |
| reference precision | 0.513 |
| reference recall | 0.958 |
| reference F1 | 0.668 |
| 平均定位误差 | 0.663 px |
| P95 定位误差 | 1.217 px |

机器可读结果：
[`development_reference_check.json`](../outputs/open_world_dataset_gallery_v04_contours/development_reference_check.json)。

这里只能说“与作者点参考的一致程度”，不能说 subtype accuracy，也不能把单张 development
图当外部泛化性能。它清楚显示当前单一位点教师的性格是高召回、低精度：167 个作者点中只
漏 7 个，但另外给出 152 个作者标注之外的候选。下一轮原子点任务应该优先把已经实现的
periodic counterfactual 教师和两教师共识纳入默认 open-world 路径，而不是继续放宽阈值。

## 这轮测试暴露出的真正改进点

### 1. 原子点：重点是减少过检

WSe₂ 单图的 0.958 recall 已经很高，主要损失来自 0.513 precision。下一步要让
site-residual 与 periodic counterfactual 两个独立物理教师互相确认，并把单教师候选保留为
`unresolved`，而不是全部升为 `probable`。

### 2. 线缺陷：需要拓扑和帧间一致性

Al 位错的 detector 几何是有响应的，但 resolver 更信伪影假设。需要增加 Burgers
circuit / 位错拓扑、相位绕转、长程位移场以及多帧稳定性证据，才能区分真实位错与扫描/
相干伪影。这个例子适合成为下一版 resolver 的最小失败回归样本。

### 3. 相界面：定位与命名必须继续分开

ZrO₂ 和 Σ3 都证明了界面定位链路可运行，但没有晶向标定、衍射关系或成像元数据时，
`unresolved` 比武断命名成 twin 更科学。下一步要把 orientation relationship、局部 FFT
峰对应和外部 metadata 作为 auxiliary evidence。

### 4. EMD：先解决 channel 语义

当前代码已经拒绝在多通道 EMD 中静默猜 channel；本轮虽然显式选了第一个 UUID，但这只
够做 I/O smoke test。必须从 Velox metadata 或数据论文说明把 UUID 映射到 HAADF/ABF 等
真实信号后，才允许进入物理性能报告。

## 输出结构

```text
outputs/open_world_dataset_gallery_v04_contours/
├── cross_dataset_overview.png
├── gallery_summary.json
├── development_reference_check.json
├── wse2_image1_reference_alignment.png
├── figures/
│   └── 六张原子相保留诊断图
└── runs/
    └── 每个样本的 summary、resolved config、mask、overlay、
        EvidenceBundle protocol 和 candidate_table.jsonl
```

为了不制造大体积重复文件，本轮运行关闭了 `.npy/.npz` 保存；可审计的 JSON、候选表、
配置和图像均保留。输入文件及任何数据集文件都没有修改或删除。

## 复现

```bash
cd /home/aiprogram/project/yaotian/next_generation_characterization/training_free_detection

PYTHONPATH=src python scripts/build_open_world_dataset_gallery.py \
  --output outputs/open_world_dataset_gallery_v04_contours_reproduction

PYTHONPATH=src python scripts/score_open_world_gallery_reference.py \
  --gallery outputs/open_world_dataset_gallery_v04_contours_reproduction
```

脚本拒绝覆盖已有输出目录或已有参考检查文件；复现时必须使用新的版本化目录。
