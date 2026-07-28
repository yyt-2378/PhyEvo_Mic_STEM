# 双专家离线审阅工作流

平台现在可以把 physics teacher 候选打包成无需服务器的 HTML 审阅界面。界面支持：

- `accept` / `reject`；
- 改变缺陷类型或标为 unknown/artifact；
- 对 point、line/interface、region 做点、线和轮廓修正；
- 自动记录候选级耗时；
- 导出符合 `review_response.schema.json` 的 JSON；
- 严格区分 assisted 与 blinded 条件；
- 两位独立 reviewer 一致时才生成 Gold。

## 两种实验条件

辅助条件展示候选覆盖、预测类型、证据和可靠性：

[`assisted review demo`](../outputs/review_pilot_demo_assisted/index.html)

盲法条件只包含原始 crop、候选 ID、bbox 和所需几何交互。bundle 内不会写入 assisted
overlay，也不会在 packet/HTML 中泄露 teacher score、缺陷类型或 evidence vector：

[`blinded review demo`](../outputs/review_pilot_demo_blinded/index.html)

这里的 blinded 仅指“隐藏模型判断”，不是隐藏候选选择过程：crop 仍来自候选生成器。若要
评价模型是否改变专家的候选发现率，还必须另设从原始视野独立抽样的窗口；不能把本界面的
blinded 条件说成完全没有算法影响。同一专家也不应依次查看同一候选的两个版本，否则会有
记忆泄漏。

这两个 demo 使用已经属于 `development_test` 的分子筛参考图，只验证交互流程，不产生
投稿所需的 Gold calibration，也不把开发图转成“新”外测。

## 已封存的 WSe₂ 100-candidate pilot

现在另有一个真正来自 train 数据、但尚未由专家执行的多图 pilot 包：

- 100 个候选，覆盖 69 张 WSe₂ 图像；
- 25 个双教师共识、25 个 periodic-only、25 个 legacy-only；
- 25 个不读取模型候选或作者标签、均匀抽取的 sham prompt controls；
- 选择 seed、父预测集、原图哈希、两种 bundle 和私有分层 key 均已封存；
- 选择过程中没有枚举或读取作者 annotation 文件。

专家入口：

- [blinded 100-candidate bundle](../benchmarks/prospective_pilots/emseek_wse2_disagreement_pilot_v1/blinded/index.html)
- [assisted 100-candidate bundle](../benchmarks/prospective_pilots/emseek_wse2_disagreement_pilot_v1/assisted/index.html)

正式分配前必须读取
[`protocol.yaml`](../benchmarks/prospective_pilots/emseek_wse2_disagreement_pilot_v1/protocol.yaml)；
其 seal 是
[`protocol_seal.json`](../benchmarks/prospective_pilots/emseek_wse2_disagreement_pilot_v1/protocol_seal.json)。
每个 reviewer–candidate 组合只能看到一种 condition，assisted/blinded 顺序要跨 reviewer
平衡。同一专家不能先看 blinded 再看同一候选 assisted。

`private_selection_key.json` 含隐藏分层，不得在所有专家 response 文件完成并封存哈希前
分发给 reviewer。blinded packet 已自动检查不包含共识分支、teacher score、defect type
或 evidence vector。当前 seal 状态仍是 `awaiting_external_reviewers`：准备好 100 个
候选不等于完成双专家 pilot，也不会自动产生 Gold。

可从同一父预测集重新生成一个新版本包：

```bash
PYTHONPATH=src python scripts/build_emseek_prospective_pilot.py \
  --output benchmarks/prospective_pilots/emseek_wse2_disagreement_pilot_v2 \
  --pilot-id emseek-wse2-disagreement-pilot-v2 \
  --seed 20260727
```

生成器拒绝覆盖已有包。新版本必须重新写 protocol 并封存，不能改动 v1。

## 生成 bundle

```bash
PYTHONPATH=src python scripts/build_review_bundle.py \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous.yaml \
  --output outputs/review_pilot_assisted_v1 \
  --condition assisted \
  --dataset-id local_zeolite_reference \
  --partition development_test

PYTHONPATH=src python scripts/build_review_bundle.py \
  --input reference_scale_radial_wiener.png \
  --config configs/amorphous.yaml \
  --output outputs/review_pilot_blinded_v1 \
  --condition blinded \
  --dataset-id local_zeolite_reference \
  --partition development_test
```

输出目录拒绝覆盖，并保存 source/config、所有 crop、packet 和 HTML 的 SHA-256。代码禁止把
`locked_external_test` 候选打包进审阅队列。

专家直接打开 `index.html`，填写匿名 reviewer ID，逐项操作后点击
“Export review_responses.json”。几何点会转换回原图像素坐标。

## 验证单个专家导出

```bash
PYTHONPATH=src python scripts/validate_review_responses.py \
  --responses reviewer_a_responses.json \
  --packets outputs/review_pilot_assisted_v1/review_packets.json
```

验证器检查：

- candidate 必须属于对应 bundle；
- action 必须适合 point/line/region/interface；
- assisted/blinded 标志不能被导出文件篡改；
- 接受或几何修正必须给出缺陷标签；
- 点/线/轮廓必须分别提供至少 1/2/3 个坐标；
- 一个导出中每个 candidate 只能有一条响应。

## 合并双专家共识

```bash
PYTHONPATH=src python scripts/compile_review_consensus.py \
  --submission reviewer_a_responses.json \
    outputs/review_pilot_assisted_v1/review_packets.json \
  --submission reviewer_b_responses.json \
    outputs/review_pilot_blinded_v1/review_packets.json \
  --annotation-version prospective-pilot-v1 \
  --output annotations/prospective-pilot-v1-consensus.json
```

同一 reviewer 不能对同一 candidate 重复投票。至少两个不同 reviewer 在接受状态与标签集合
上完全一致才生成 `expert_consensus` Gold；单人、unknown 或任何分歧都保持 Unresolved。

## 前瞻性 pilot 的实验设计

正式 50–100 候选 pilot 应在候选出现前冻结：

1. 资产分区和 acquisition/sample/lab group；
2. 候选选择策略与专家时间预算；
3. reviewer 匿名 ID 和 assisted/blinded 交叉顺序；
4. 缺陷 ontology、artifact 定义和停止规则；
5. 一致性、耗时、接受/拒绝/修正率以及模型提示偏差的指标。

当前只是把可执行工具准备好。没有两位真实 STEM 专家完成前瞻性审阅前，Phase 4 的科学
门禁仍然是未完成。
