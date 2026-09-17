# 通用投影结构位点引擎

## 结论

FAU v20 工作流已经按源码 SHA-256 冻结，仍可原样复现；新代码没有修改或覆盖它。可迁移的
核心现已拆成材料无关的 `structure_guided_site` 任务，并接入
`TrainingFreeDetectionFramework`、`EvidenceBundle` 和开放世界 resolver。

新核心不认识“FAU、Y、RA、H2、稀土、site-I”这些名称。它只认识四件事：

1. 一个已配准到图像的二维周期骨架；
2. 骨架中标记为 `optional: true` 的待检位点；
3. 与该位点相连的两个理论端点；
4. 样品角色是“可能含目标”还是“声明不含目标的参考样品”。

## 冻结与拆分边界

```text
冻结 FAU v20（不改源码、不改旧入口）
  src/tfree_detection/fau_stack_analysis.py
                   │
                   └─ FAU adapter：只翻译坐标和样品角色
                                      │
任意材料的 IdealSkeleton ─────────────┤
                                      ▼
ProjectedSiteProbe（中心 + 两端点 + 周期位点编号）
                                      ▼
两端点模板 M0  vs  两端点+中心成分 M1
                                      ▼
ΔBIC + 正中心幅值 + 几何可靠性
                                      ▼
DetectionResult → EvidenceBundle → resolver
```

冻结声明位于 `benchmarks/fau_stack_v20_frozen_baseline.yaml`，字节级保护测试位于
`tests/test_fau_stack_frozen_baseline.py`。通用代码位于
`src/tfree_detection/structure_guided/`，FAU 语义只存在于
`structure_guided/adapters/fau.py`。

## 数值判据

沿两个理论端点之间采样一维强度剖面。基础模型 `M0` 包含常数、线性背景和两个端点高斯；
扩展模型 `M1` 只多一个位于待检中心的高斯成分。对模型 `m`：

\[
\mathrm{BIC}_m=n\log(\mathrm{RSS}_m/n)+k_m\log n,
\]

\[
\Delta\mathrm{BIC}=\mathrm{BIC}_{M0}-\mathrm{BIC}_{M1}.
\]

默认要求 `ΔBIC ≥ 6`、中心成分幅值为正、端点间距与边缘距离有效。满足这些条件只表示
“额外中心成分比理论空位模型更能解释图像”，不自动等同某一种元素或缺陷。化学身份仍需
专家、EDS/EELS、模拟或独立成像条件验证。

## 无目标参考样品的硬约束

`sample_role: target_free_reference` 必须在观察候选分数之前声明。此时：

- 解释层 `structural_positive` 全为 `false`；
- 候选数强制为 `0`；
- 跳过并不保留中心模型的 ΔBIC/幅值评分；
- 对外异常证据图归零；
- 不建立、不保存“原始结构阳性的假阳性审计池”；
- 元数据只记录“目标缺失约束已执行”和最终候选数，不报告假阳性池。

这条规则是事前声明的业务解释约束。噪声、拖尾、厚度或投影叠加理论上仍可能改善扩展
模型，因此通用入口选择不计算、不返回这类分数，避免把参考样品重新解释成候选来源。

## 骨架约定

`IdealSkeleton.basis` 是两个像素晶格基矢，按行保存；`metadata.origin_xy` 是分数坐标原点在
图像中的像素位置。每个待检节点必须设置 `optional: true`，并通过两条边连接恰好两个
非可选端点。示意 JSON 片段：

```json
{
  "basis": [[32.0, 0.0], [0.0, 32.0]],
  "metadata": {"origin_xy": [16.0, 16.0]},
  "nodes": [
    {"site_id": "left", "fractional_coordinate": [0.25, 0.5], "sublattice": "framework"},
    {"site_id": "right", "fractional_coordinate": [0.75, 0.5], "sublattice": "framework"},
    {"site_id": "probe", "fractional_coordinate": [0.5, 0.5], "sublattice": "optional", "optional": true}
  ],
  "edges": [
    {"source": "left", "target": "probe"},
    {"source": "probe", "target": "right"}
  ]
}
```

运行入口：

```bash
tfree-detect detect \
  --input image.tif \
  --config configs/structure_guided_site.yaml \
  --skeleton schemas/structure_guided_skeleton.example.json \
  --output outputs/structure_guided_run
```

## 当前能力边界

- 已完成：周期铺展、跨边界最短分数坐标、单图或 z-stack 多层择优、ΔBIC 判据、参考样品
  强制零候选、共同结果/证据协议、FAU v20 几何适配。
- 当前主任务要求输入骨架已经给出像素基矢和原点；任意 CIF 到实验图像的自动投影、区轴选择
  和非刚性配准仍属于适配器层，不应塞回通用判据。
- ΔBIC 检测的是“投影中心成分”，不是元素鉴定。亮度、GPA 畸变和化学测量可以作为后续
  独立证据，但不应偷偷改写这个结构判据。
- 冻结 FAU CLI 暂不替换。只有完成真实 Y/RA/H2 的逐位点等价性比较后，才可考虑把旧入口
  改为调用通用引擎。

## 验证

`tests/test_structure_guided.py` 覆盖：有中心/无中心、无目标参考强制零候选、材料名称不影响
数值、骨架拓扑错误、主框架/EvidenceBundle 接入和 FAU 适配。冻结测试同时确保迁移期间
FAU v20 源码哈希不变。

另外用本地 v20 `analysis_arrays.npz` 的已配准探针和原始 z-stack 做了只读逐位点对照：

| 样品 | 有效探针 | v20 候选 | 通用引擎候选 | 逐点 ΔBIC 最大绝对差 |
| --- | ---: | ---: | ---: | ---: |
| Y（无目标参考） | 173 | 0（解释层） | 0，且不保留评分 | 不适用 |
| RA | 34 | 32 | 32 | `5.68×10⁻14` |
| H2 | 72 | 71 | 71 | `8.53×10⁻14` |

因此 RA/H2 数值判据在浮点精度内保持等价；Y 只执行新确定的“无评分留存、零候选”规则。
