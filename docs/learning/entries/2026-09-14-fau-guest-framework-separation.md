# FAU 投影模板中的稀土客体—骨架分离

- Kind: algorithm
- First seen: 2026-09-14
- Status: applied
- Confidence: 2
- Related change: `src/tfree_detection/fau_stack_analysis.py`、`analyze-fau-zstacks`
- Last review: 2026-09-14

> 同日更正：本条“最亮总体作为客体”的硬分离会高估稀土。骨架与候选分开所有权的原则
> 继续保留，但当前候选必须通过[畸变区与局部极亮交集门](2026-09-14-distortion-gated-local-extreme-rare-earth.md)；
> 跨层重复仅作旁证。
>
> 2026-09-16 再更新：畸变区/局部极亮交集也不再拥有最终候选；当前采用
> [骨架优先 site-I 结构模型、邻近等价位点亮度和 GPA 后验旁证](2026-09-16-skeleton-first-site-i-local-gpa.md)。

## Why this appeared

新的输入是三组完整 TIFF z-stack 和一个纯 SiO2 FAU.cif。任务既要从图像找稀土，又要用
CIF 计算 Y/RA/H2 的骨架变化。CIF 不含稀土，如果把所有亮峰都送入最近模板点匹配，客体
会被错误吞成 Si/O，并产生很大的假位移。

## Mental model

观测必须分成两个所有权不同的集合：最亮且跨焦面复现的峰是 `unmatched_guest_candidate`；
CIF 只拥有暗/中亮的 O/Si 骨架匹配。局部应变来自稀土周围已匹配骨架，不能来自把稀土
本身硬贴到理想骨架点的残差。跨视野稀土只在规范 CIF 单胞坐标中比较总体分布。

## Minimal reproduction

```bash
TMPDIR="$PWD/.codex-tmp" PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python -m pytest -p no:cacheprovider -q \
  tests/test_fau_stack_analysis.py::test_rare_earth_guest_is_never_consumed_by_framework_match \
  tests/test_fau_stack_analysis.py::test_guest_coordinates_remove_image_cell_basis_and_phase_nuisance
```

## Prediction

用户确认：FAU.cif 是 Si/O 标准骨架；图像中最亮的稀土峰必须一直留作外部客体，不能因
离某个模板点近就被改名为 Si/O。骨架形变应从它周围匹配到的 Si/O 推断。

## Observation

最小合成反例中，四个骨架点的真实位移都小于 0.5 px；额外稀土点若按最近模板错误匹配，
会制造 4.801 px 假位移。应用硬分离后，四个骨架点正确匹配，稀土保持
`unmatched_guest_candidate`。真实数据中 CIF 显式 192 个对称操作展开为 O 384、Si 192，
用户给定的 `[101]` 在三图中相对非等价低指数候选均得到最高周期相关；它与先前无先验
图像搜索得到的 `[110]` 是立方 FAU 中的对称等价方向。

## Project integration map

```text
TIFF QYX stack
  ├─ raw-detail focus window + cross-plane persistence
  ├─ occupied ordered brightness mixture
  │    └─ brightest → unmatched RE guest candidate
  └─ dim/mid peaks → user-supplied CIF [101] O/Si framework matcher
       ├─ local framework strain / hotspot proxy
       └─ projected-cell basis → RA/H2 vs Y global strain

guest xy → undo image basis + template GL transform + phase → canonical CIF uv
         → ensemble distribution only
```

## Theory and boundaries

模板映射为 `x=B(Mu+φ+n)`，因此客体规范坐标必须使用
`u=M^{-1}(B^{-1}x-φ) mod 1`。只减图像平移或只取 `B^{-1}x` 会把独立的基矢/相位选择
误报为迁移。局部小应变取位移梯度的对称部分；没有弹性常数时最大剪应变只能叫热点代理。

## Alternatives

未把 CIF 增补虚构稀土坐标，因为没有占位先验；未做跨图逐原子最近邻轨迹，因为视野不同；
未把 z-stack 第一轴当空间 z，因为 TIFF 无物理步长。可以在未来加入已标定多重散射模拟，
但不能由当前图像自动杜撰。

## Failure modes

- 最亮总体仍可能混入投影重叠柱、厚度或扫描伪影；
- 单图 O/Si 覆盖率不足时，逐点畸变只能探索性使用；
- FFT 基矢同时包含真实晶格变化和扫描仿射畸变；
- 不同像素标定会直接污染跨图全局应变；
- CIF 相位或基矢未规范化会制造假稀土迁移。

## User contribution

用户给出两项决定性约束：4D-STEM 亮度顺序为“稀土最亮、Si 次之、O 最暗”；FAU.cif
是 Si/O 标准骨架，稀土必须作为骨架外客体。第二项被直接固化在
`test_rare_earth_guest_is_never_consumed_by_framework_match`，第一项由非空有序混合测试覆盖。

## Validation

- 相关测试：15 passed；包括安全 CIF、非空亮度总体、焦面持续性、低指数合成投影、
  客体不被吞、规范单胞坐标、已知仿射应变恢复。
- 真实运行：Y/RA/H2 均按用户给定 `[101]` 拟合并由非等价候选旁证；输出 CSV、NPZ、
  JSON、中文报告和六张审计图。近/远稀土候选的畸变没有一致单调差异，未建立因果关系。
- 当前解释门槛：Y/H2 逐点结果 `exploratory_only`；RA 周期/局部拟合通过，但化学身份未验证。

## Retrieval questions

1. 为什么一个最亮峰刚好靠近 CIF 的 Si 位点时，仍不能直接把它当 Si？
2. 三图视野不同时，为什么规范单胞内的总体分布可以比较，而逐原子轨迹不可以？

## Learning debt

需要真实扫描像素标定、少量专家或 EDS/EELS 点标、厚度/成像参数和 multislice 模拟来
验证化学身份与位移系统误差；真实应力还需要材料的取向相关弹性张量。

## Review history

- 2026-09-14：用户确认客体—骨架硬分离；合成反例、真实 z-stack 和 CIF `[110]` 闭环完成。
