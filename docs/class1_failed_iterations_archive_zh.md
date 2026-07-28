# Class1 失败迭代归档

本文件把不再作为推荐入口的 Class1 尝试集中登记。相关源码和结果没有删除，因为它们仍是
可复现负结果或 Class2 最佳结果的一部分；最终推荐入口只有
`symmetry_relation`、`run_class1_symmetry_benchmark.py` 和 v2 sealed 产物。

| 迭代 | 结果 | 失败原因 | 状态 |
|---|---|---|---|
| v0.3 historical | IoU 0.1251，F1 0.2224 | 固定分位数产生正常响应；line/region 几何混合 | 只读历史控制 |
| v0.4 domain v1/v2 | domain IoU 约 0.03 | 局部波矢/应变不能表达同频率群相位域；正常纹理分离更强 | deprecated ablation |
| augmented v0.4 | IoU 约 0.11，正常误报约 8.2% | 直接并集提高覆盖但破坏正常对照 | deprecated ablation |
| symmetry final v1 | 芯点表面 IoU 0.2346 | 8 邻域普通线像素被误当端点，点掩膜异常膨胀 | **科学作废** |
| symmetry final v2 | guided IoU 0.1891，F1 0.3181 | 有真实增益，但仍是已揭盲事后诊断 | 当前推荐 |

## v1 芯点假高分的审计

错误条件使用 `neighbor_count <= 2`。在 8 邻域骨架中，普通线像素恰好通常有两个邻居，
因此整条线被当作端点邻域，`dislocation_core_point` 在第一张图膨胀到 176,879 像素。
v1 芯点 IoU `0.2346` 不是点检测成功，而是与 Dennler 粗区域 Gold 偶然重叠。

v2 改成：

1. 只接受 `neighbor_count <= 1` 的内部端点；
2. 相位线长度至少为四个晶胞；
3. 每个端点邻域最多保留一个 Burgers 不闭合峰；
4. 输出仍称 proxy，并记录骨架分量、支持端点和种子数。

修复后芯点 IoU 降到 `0.0022`，但新域、界面和相位线指标不变；最终 guided 指标仍比历史
明显提高。这是“修复物理语义优先于保留漂亮数字”的必要负结果。

## 为什么不删除旧脚本和 v1 产物

- `run_dennler_posthoc_v4.py` 与 `visualize_dennler_posthoc_v4.py` 仍复现 Class2 的最佳
  coherence-loss 结果，不能当作纯 Class1 垃圾删除；
- v1 预测是成功完成但科学语义错误的审计证据，不属于可随意删除的失败运行临时文件；
- 历史文档、seal 和图仍引用这些路径，删除会破坏项目可追溯性。

因此这里采用“唯一推荐入口 + deprecated 归档”而不是物理删除。若以后需要清理，必须先
给出精确文件列表、确认引用迁移，并继续保护所有数据集与最终结果。

