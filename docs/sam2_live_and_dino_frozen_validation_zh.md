# SAM2 真实实例与 DINO frozen 流程说明

> 验证日期：2026-07-30
> 结论：SAM2.1-large 已在两张实验 STEM 图上完成真实 GPU 推理；DINO 阶段此前也使用了
> 真实冻结权重，但那次留档运行记录的是 CPU 推理。两者都没有训练或微调模型。

## 1. 三个模块各自负责什么

| 模块 | 回答的问题 | 是否决定缺陷存在 | 是否训练 |
|---|---|---:|---:|
| physics teacher | 哪里违反周期性、相干性或晶域连续性 | 是，产生 coarse candidate | 否 |
| frozen DINO | 候选局部的表征与正常晶格参考差多少 | 否，只补充表型差异证据 | 否 |
| frozen SAM2 | 已知候选附近更合理的像素边界在哪里 | 否，只提出可拒绝的几何 proposal | 否 |

因此，SAM2 不是新的缺陷分类器。平台先用物理教师发现候选，再把正点、负点、框和粗
mask 交给 SAM2。SAM2 的 proposal 只有通过核心召回、外侧泄漏、面积、物理对比、
prompt 稳定性、边界证据和拓扑门，才进入 `final_geometry_mask`；失败时只回退当前候选。

## 2. 本机实际使用的环境

```text
Python        /home/aiprogram/anaconda3/envs/sam/bin/python
Python 版本   3.10.15
PyTorch       2.4.1+cu121
torchvision   0.19.1+cu121
CUDA runtime  12.1
GPU           NVIDIA H100 80GB
SAM2          /home/aiprogram/project/yaotian/sam2
checkpoint    sam2.1_hiera_large.pt
```

没有为这次验证重新安装 CPU 或 GPU 版 PyTorch。平台主进程仍可使用项目环境，只把 SAM2
推理通过文件协议交给上述已有环境中的独立 worker。

## 3. 两个真实实例的结果

总览图：

[`outputs/sam2_real_instance_gallery_v1/sam2_real_instance_comparison.png`](../outputs/sam2_real_instance_gallery_v1/sam2_real_instance_comparison.png)

| 实验图 | 物理候选 | SAM2 prompts | 通过 / 回退 | worker 时间 | GPU 峰值 |
|---|---:|---:|---:|---:|---:|
| 拉伸后局部非晶分子筛 | 3 | 24 | 1 / 2 | 9.92 s | 1.31 GiB |
| ZrO₂ 孪晶/两相界面 | 1 | 16（A/B 两侧），8 个界面 proposal | 0 / 1 | 8.53 s | 1.31 GiB |

### 分子筛

- 候选 2 通过全部门：核心召回 `1.000`、面积比 `1.071`、prompt 平均 IoU
  `0.991`、边界证据对齐 `0.845`。最终面积从 `26,166` px 细化为 `28,025` px。
- 候选 1 和 3 的核心都被保留，但 SAM2 共识把单一 coarse 区域切成了 3 个连通块；
  `preserve_candidate_topology=true` 因而拒绝它们，并逐候选恢复原 coarse mask。
- 整图状态是 `partially_accepted`，不是“全部成功”或“全部失败”。

### ZrO₂ 界面

- 物理分支先给出两侧晶域，worker 分别提示 domain A 和 domain B，再转换为共享界面，
  而不是用 box 填满一大片区域。
- SAM2 的视觉分数很高且 8 次 prompt 较稳定，但共识界面只覆盖物理核心的 `0.593`，
  低于 Policy A 的 `0.950`；同时从 1 个连通界面变成 3 段。
- 因此 proposal 留在诊断中，最终几何严格回退到物理 coarse 界面。这说明 SAM2 自己的
  高分不能越过材料物理门。

这两张图没有人工 Gold 边界，所以这里只能证明“真实模型推理、门控和回退正常工作”，
不能据此宣称边界准确率提高。

## 4. 总览图六列怎样读

1. **Physics coarse**：红线是训练自由物理检测器给出的原始候选。
2. **Prompt ensemble 0**：绿色 `+` 是缺陷/晶域正点，红色 `×` 是正常外侧负点，青框
   限制本候选的局部作用范围。
3. **Live SAM2 proposal**：8 次轻微扰动 prompt 的共识包含区；这是 SAM2.1-large
   的真实输出，不是手绘示意。
4. **Final geometry**：青色表示门控通过后采用 SAM2 细化，黄色表示该候选被拒绝并恢复
   coarse 几何。
5. **Boundary uncertainty**：prompt 集成在边缘处的不一致；亮区表示换一组提示后边界更
   容易变化。它不是经过 Gold 校准的概率。
6. **Physics-gate audit**：逐候选列出核心召回、面积比、稳定性、通过/回退状态和原因。

## 5. DINO frozen 当时究竟怎样跑

留档目录
[`outputs/reference_amorphous_stage2_final`](../outputs/reference_amorphous_stage2_final)
不是模拟 embedding。它加载本地 DINO ViT-S/8 权重
`vendor/models/dino-vits8/dino_deitsmall8_pretrain.pth`，SHA-256 为
`55c8b267f479b614ec20858947ce361ba804846c193e20be74263a096bb6d75e`，然后：

1. 物理检测器先产生 3 个候选，DINO 不负责发现候选。
2. 对每个候选建立 core、transition 和 reference 三个空间带。
3. 每带最多固定采样 16 个中心，截取 `128×128` 灰度 patch，经 1/99 百分位归一化；
   复制成 RGB、缩放到 `224×224`，再做 ImageNet normalization。
4. DINO ViT-S/8 设为 `eval()`，全部参数 `requires_grad_(False)`，并在
   `torch.inference_mode()` 中提取归一化的 384 维 CLS embedding。
5. reference embedding 的逐维中位数作为该候选的局部正常原型；异常距离是
   `1 - cosine_similarity`。
6. 用 reference 距离的 median/MAD 产生 robust z-score；`z≥3` 为 strong，
   `1.5≤z<3` 为 moderate，再与物理证据并列报告。

三处候选的 core z-score 分别为 `3.603`、`7.130`、`2.815`；对应 DINO 判定为
strong、strong、moderate。历史输出的 provenance 记录 `device: cpu`，含义是那一次
推理进程只看到了 CPU，不是代码要求 DINO 必须使用 CPU，也不代表本机没有 GPU。

DINO 当前最大的技术限制是“空间带约束 patch 中心，而非保证整个 patch 都在该带内”，
所以靠近边界的 patch 可能混入相邻区域。它证明候选表征与局部正常晶格不同，但没有 Gold
时不能证明类别或边界准确。

## 6. 复现命令

使用现有 SAM 环境和 CUDA：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tfree_detection detect \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous_sam2.yaml \
  --output outputs/reference_amorphous_sam2_live_v2
```

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python -m tfree_detection detect \
  --input datasets/public/guo_2021_atomic_defects/ZrO2_monoclinic_twin_boundary.tif \
  --config configs/phase_boundary_sam2.yaml \
  --output outputs/zro2_phase_boundary_sam2_live_v2
```

当前配置中的 `stage2.sam3`、交换协议名和少数产物文件名仍保留早期 SAM3 原型命名，以
兼容既有结果；真正使用的后端由 `backend: sam2` 和输出 provenance 决定。SAM2 官方
也支持视频，但本平台本次只接入了 `SAM2ImagePredictor` 的单图路径，尚未把
`SAM2VideoPredictor` 作为时间序列缺陷跟踪器接入。

## 7. 下一步怎样判断“真的更好”

最重要的下一步不是继续看轮廓是否顺眼，而是制作一批独立、双人复核的像素级 Gold：

- region：非晶核心、过渡带和不确定边缘分层标注；
- interface：中心线与容许宽度带分开标注；
- 同时保存 coarse、SAM2 proposal、门控后的 final；
- 比较 Boundary F1、HD95、ASSD、核心召回和专家修订时间；
- 用 calibration 集调门，只在 locked external test 上报告最终收益。
