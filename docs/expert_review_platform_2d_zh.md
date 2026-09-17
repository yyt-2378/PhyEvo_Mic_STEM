# 二维 STEM 专家评审平台

## 当前范围

本评审平台只处理可直接呈现为二维实空间图像的样本，不包含 4D-STEM 衍射帧或其
虚拟探测器派生图。统一清单为
`datasets/standardized/review_2d_manifest_v1.jsonl`，共 554 条逻辑图像平面：

| 数据来源 | 逻辑图像数 | 像素标签状态 | 主要审核用途 |
| --- | ---: | --- | --- |
| EMSeek WSe2 | 107 | 作者像素掩膜 | 仅自动评测，不进入人工界面 |
| EMSeek 平衡子集 | 32 | 作者像素掩膜 | 仅自动评测，不进入人工界面 |
| Dennler 2021 | 61 | 31 个掩膜、30 个空掩膜 | 锁定外部测试 |
| Guo 2021 | 103 | 无像素 Gold | 高优先级专家标注 |
| Al Sigma3 | 250 | 只有边界存在先验，无像素轨迹 | 高优先级边界描绘 |
| 本地分子筛 | 1 | 只有非晶区域先验 | 方法开发复核 |

“来源说明中存在边界”不能当作像素标签。平台将它记录为 `source_prior_only`，
必须经过专家描绘才能成为 expert-verified Silver；只有独立专家共识后才可能升级为实验
Gold。

## 两个完全分离的池

根据当前专家工作策略，554 张图像不再全部进入人工界面：

- **专家待标注池：354 张。**只包括 Guo 103 张、Al Sigma3 250 张和本地分子筛
  1 张；这些样本没有像素标签。
- **自动评测参考池：200 张。**包括 EMSeek 139 张和 Dennler 61 张；已有像素掩膜
  或明确的正常空掩膜，不再让专家重复标注或审核。

待标注池的可执行清单是
`datasets/standardized/expert_review_queue_v1.jsonl`。生成器设置硬门禁：
`pixel_mask` 和 `implicit_empty_mask` 都不能进入该队列。

354 个图像平面进一步合并为 **154 个实际审核案例**：

- Guo：103 个单平面案例；
- Al Sigma3：250 个平面按同一 frame 的 5 个探测器通道合并成 50 个案例；
- 本地分子筛：1 个案例。

因此专家不会在 BF、HAADF、DF2、DF4 上重复描绘同一条 Al 边界。界面默认显示 HAADF，
允许切换其他通道确认边界，但一次提交的边界对该物理 frame 的 5 个通道共同有效。这样
避免 200 次重复边界描绘。案例清单位于
`datasets/standardized/expert_review_cases_v1.jsonl`。

五个通道必须保持相同 frame index 和二维 shape；正式训练导出仍保留参考通道及全部五个
sample ID。若后续通道配准门发现非零漂移，应保存通道变换，而不能直接复制像素 mask。

## 数据划分

当前清单采用 `source-group-isolated-v1`：

- train：384；
- validation：107；
- test：63。

其中专家待标注平面为 train 251、validation 103、test 0；合并后是 train 131、
validation 23 个审核案例。已有标签的锁定测试集仍负责
最终评测，不进入专家界面。被当前模型预测辅助过的人工标签也不应反过来作为同一模型的
无偏 test Gold。

同一 acquisition、时间序列或保守来源组只能出现在一个 split。特别是：

- Guo SuperScan 的 100 帧整体进入 train，不随机拆帧；
- Al Sigma3 ORI 的 150 个通道平面进入 train，ORII 的 100 个平面进入
  validation；
- EMSeek WSe2 缺少可靠 acquisition 分组元数据，因此 107 张整体保留在 train；
- Dennler 61 张整体保留为锁定外部 test，不能进入模型辅助主动学习队列。

这里的 split 主要为后续主动学习和可训练学生模型预留。当前 training-free 检测器本身
不使用 train split 拟合参数。

## 图像与预测显示

读取原始数据时不复制、不改写源文件：

- PNG/TIFF 直接读取；
- NData 必须明确 `frame_index`；
- EMD 必须同时明确 `dataset_path` 与 `frame_index`。

专家界面使用固定颜色约定：

- 橙色填充：training-free 预测区域；
- 青色边界：training-free 最终几何；
- 专家队列不显示绿色作者标签，因为该队列保证没有像素标签；
- source prior 只用于提示任务类型，不会渲染成伪造标签。

已有标签样本在独立的自动评测流程中计算指标，不进入专家审核页面。

## 启动正式在线审核界面

项目环境已包含 Streamlit 时：

```bash
PYTHONPATH=src streamlit run apps/review_dataset_viewer.py
```

在同一局域网提供访问：

```bash
PYTHONPATH=src streamlit run apps/review_dataset_viewer.py \
  --server.address 0.0.0.0 --server.port 8501
```

不使用自有域名、通过 HTTPS 隧道向受邀专家开放时，必须设置非空共享访问码，并让
Streamlit 只监听回环地址：

```bash
TFREE_REVIEW_ACCESS_CODE='<strong-shared-code>' \
  PYTHONPATH=src streamlit run apps/review_dataset_viewer.py \
  --server.address 127.0.0.1 --server.port 8502 --server.headless true

tailscale funnel --bg 8502
```

当前阶段性审核使用 Tailscale Funnel，固定公网入口为
`https://phyevo-review.tailf74731.ts.net/`。当前 tailnet 使用 Personal 免费方案，没有购买
域名或付费资源。项目本地守护器会监控并重启 Tailscale 与 Streamlit 子进程；服务器必须
保持开机、联网且不能重启。服务器重启、上游网络故障或 Tailscale 服务异常仍会造成中断，
因此这不是带可用性 SLA 的托管服务。

共享访问码在任何病例、预测或 SQLite 加载之前校验；缺失、空值或错误都会立即停止应用。
访问码只通过运行进程的环境变量注入，不得写入仓库、文档或命令脚本。负责人应定期检查
`.codex-tmp/public-review-runtime-status.json`、公网健康端点和守护事件日志；若页面无法访问，
先确认服务器仍在线，再恢复项目本地守护器。

专家先输入唯一的 `Reviewer ID`。主页面使用单张可缩放大图，默认隐藏预测；按需打开的
模型轮廓为琥珀细线，专家几何为青绿空心点/细线。支持六种结论：接受预测、修正几何、拒绝预测、
无缺陷、artifact 和 unknown。

可直接发给专家的一页说明见
[二维 STEM 专家审核：一页操作说明](expert_annotation_quick_guide_zh.md)。

点缺陷直接点击位点；线缺陷、位错、相界面和晶界使用顺序控制点；非晶区域在专用 Canvas
上按住鼠标沿外缘连续拖动。Canvas 会把浏览器坐标换算成源像素坐标；绘图、撤销和清空在
`st.form` 内暂存，不触发全页重运行。专家点击两个保存按钮之一时，几何、缺陷类型和备注才
作为同一次表单事务回传。Canvas 工具栏支持缩放、全屏、拖动查看、删除单点、撤销/恢复
和清空。屏幕亮度、轮廓浓度与深浅配色只影响显示，不改变图像或源坐标。页面记录的是
预测层实际被显示过的状态；未查看预测时无法提交接受/拒绝预测。“保存并进入下一例”
的成功提示在跳转后仍显示。

权威账本默认是
`datasets/expert_annotations/review_store_v1.sqlite3`。Streamlit 页面重运行、刷新或浏览器
断开不会删除审核：每个 `(project, reviewer, case)` 只有一个当前有效结果，真实修改会生成
新的不可变修订，重复点击或相同内容重试不会增加版本。界面可分别下载当前结果和完整历史
JSONL。数据库及 `-wal`/`-shm` 文件属于不可删除数据资产，部署时必须备份。

`Reviewer ID` 是记录身份，不是互联网认证。共享访问码加 HTTPS Funnel 只适合
封闭、受邀专家组的阶段性审核：它阻止未持码者加载数据，但不提供个人撤权、MFA 或机构级
身份认证。若访问码泄露，应精确停止当前 Funnel 和 Streamlit 进程，换码后重启；长期或更高
敏感度部署仍应使用具备独立账号策略的访问代理。

## 资产门禁与审核导出

启动前或数据变化后执行：

```bash
PYTHONPATH=src python scripts/audit_expert_review_platform.py
```

门禁要求：354 个无标签平面恰好覆盖 154 个案例；现有像素标签数为 0；test 案例数为 0；
154 个主视图和全部 354 个图像平面都有通过校验和验证的预测。

审核冻结后导出一位专家的当前有效结果：

```bash
PYTHONPATH=src python scripts/export_expert_reviews.py \
  --reviewer-id expert-01 \
  --output datasets/expert_annotations/exports/expert-01-v1
```

导出目录拒绝覆盖。接受预测、修正几何和无缺陷会生成训练可用记录；拒绝预测、artifact 和
unknown 会保留为审核结论，但不进入学生模型训练。点/线/区域的源坐标是权威标注，PNG
mask 只是给像素模型使用的派生栅格视图。单专家导出仍是 expert-verified Silver，不自动
称为 Gold。

## 预计算主任务预测

```bash
PYTHONPATH=src python scripts/precompute_2d_review_predictions.py \
  --output outputs/expert_review_2d_predictions_v1 \
  --workers 8
```

每个样本只保存紧凑二值预测掩膜、候选框元数据、配置哈希、输入像素哈希和校验和，不为
每张图重复生成大型诊断图。这样专家切换样本时不需要等待检测器现场运行。

## 科学审核原则

1. 已有像素标签或明确空掩膜的样本不进入人工队列。
2. 作者掩膜属于 author-curated Silver，只用于自动评测，不会因跳过复核而升级为 Gold。
3. source prior 只能帮助确定审核任务，不能作为空间监督。
4. 锁定 test 不参与主动选择、提示调参或模型辅助标注。
5. assisted/blinded 条件必须写入每条审核记录。
6. 接受、拒绝、类型修正、artifact、unknown 和几何修正都是有效结论。
7. 双专家一致才能自动生成实验 Gold；单专家结果记为 expert-verified Silver。
