# Streamlit 重运行与专家审核持久状态

- Kind: architecture
- First seen: 2026-07-31
- Status: applied
- Confidence: 2
- Related change: `apps/review_dataset_viewer.py`、`review_store.py` 与 `review_export.py`
- Last review: 2026-07-31

## Why this appeared

专家需要在一周内审核数百张二维 STEM 图像。浏览器刷新、网络短暂断开或 Streamlit
组件交互都不能丢失已提交判断，也不能让同一专家对同一样本被重复统计。

## Mental model

Streamlit 的 Python 脚本会在组件交互时重新执行，`st.session_state` 只属于当前
WebSocket 会话。页面状态适合保存“当前看到第几张”，不适合充当科研审核记录。

高频浏览器草稿还需要独立的事务边界：Canvas、缺陷类型和备注位于同一个 `st.form` 时，
`setStateValue` 只更新表单的待提交状态，不会因每个控制点重跑脚本；两个 form submit button
之一被点击后，完整表单才一次进入 Python。表单批处理解决交互闪烁，SQLite 仍负责持久事实，
二者不能互相替代。

可靠设计需要把展示状态和持久事实分开：页面可随时重建；服务器存储以
`(review_project_id, reviewer_id, case_id)` 定位审核对象，每次保存必须幂等，并记录
assisted/blinded、几何坐标、耗时和版本。当前策略是只保留一个当前有效投影，同时保留
完整不可变修订历史。

## Minimal reproduction

`tests/test_expert_review_store.py` 使用合成样本执行两次保存：

1. 同一专家对同一病例提交 `accept_prediction`；
2. 随后提交 `correct_geometry`；
3. 关闭并重新打开 store；
4. 检查当前结果数量和修订历史数量。

同一内容的重复点击和新 click ID 重试也会执行，验证不会新增修订。

## Prediction

Codex 于 2026-07-31 询问：重新打开页面后应显示“两条并列结果”，还是“一个当前有效
结果＋完整修改历史”。用户随后要求按目标中的推荐语义完成平台，即采用后者。

同日用户预测表单化后的行为：连续绘制5个点但不保存时，页面不重运行且数据库修订数不变；
点击保存后只进行一次提交并新增一版记录。

## Observation

第一次保存 `accept_prediction`、第二次保存 `correct_geometry` 后，历史长度为2，当前结果
长度为1且指向第二版。重新构造 store 后状态不变。相同内容重试保持历史长度为1。真实
Streamlit AppTest 提交后重新构造整个应用，完成进度由0恢复为1。

表单复现中，Canvas组件和“保存当前”“保存并进入下一例”两个按钮拥有相同且非空的
`form_id`，按钮均为form submitter；选择修正几何并挂载Canvas但未保存时，审核历史仍为空。

## Project integration map

`review_2d_manifest_v1.jsonl` 定义样本和 split；预测目录提供不可变模型输出；Streamlit
入口加载两者并显示图像。`review_canvas.py` 在表单内拥有高频绘图草稿；表单提交后
`review_dataset_viewer.py` 才校验整组输入。`review_store.py` 拥有专家提交事实；正式 app
只能通过 store 保存和恢复；`review_export.py` 读取当前投影并保留修订 provenance。页面
session 只保存未提交草稿和导航状态。

## Theory and boundaries

幂等键避免刷新或重复点击制造重复有效结论；修订历史用于审计。`st.form` 只阻止表单内控件
的逐次重运行，表单外的结论选择、病例切换或通道切换仍会重运行页面，因此专家应先选择任务
和通道再开始绘图。它不解决专家身份认证、双专家任务分配或跨服务器数据库复制。

## Alternatives

只使用 `st.session_state` 实现最简单，但刷新页面会丢失记录。Canvas每个点都立即回调Python
能实时显示服务端坐标，却会产生全页遮罩、点击排队和闪烁。增加独立“同步绘制”按钮可以减少
重运行，但会增加专家遗漏步骤；同一表单的原生提交边界更直接。让专家每天手工下载 JSON
可以保留现有离线流程，但不满足在线断点续审与集中进度统计。

## Failure modes

- 页面刷新后审核消失；
- 双击提交产生重复记录；
- 修订覆盖旧值且无法审计；
- 首次判断前泄露作者标签；
- locked test 进入模型辅助审核；
- 多位专家共享同一 reviewer ID。
- Canvas被移出form或保存按钮不再是form submitter，导致逐点全页重运行；
- 专家绘图后切换表单外的病例或通道，未提交草稿被新页面替换。

## User contribution

用户选择了“一个当前有效结果＋完整修改历史”的状态语义，并明确了六类专家动作、四种
几何及 train/validation 与 locked test 的边界；实现据此固定数据库不变量。
用户进一步明确并预测了表单事务语义：绘制5点期间不重运行、不写库，保存时才一次提交。

## Validation

- `tests/test_expert_review_store.py`：重开、幂等、非法 test 和进度边界通过；
- `tests/test_review_streamlit_app.py`：真实154病例提交、应用重载以及Canvas/保存按钮共享
  form transaction boundary通过；
- Canvas、表单、存储和平台相邻回归：16项通过；
- 全项目回归：150项通过；
- 实际 Streamlit 服务健康端点返回 `ok`，随后精确停止该测试进程。

## Retrieval questions

1. 为什么 `st.session_state` 不能作为专家审核的唯一事实来源？
2. 幂等审核键应包含哪些标识，为什么不能只使用 sample ID？

## Learning debt

公网身份认证和多服务器数据库复制尚未实现；当前部署边界是可信局域网单服务器。真实浏览器
仍需按用户贡献的5点场景确认不再出现全页变暗。

## Review history

2026-07-31：用户选择单一当前投影和完整历史；合成复现及真实应用重载均符合预测，状态
更新为 `applied`，confidence 2。

2026-07-31：用户预测5点绘制应在浏览器表单内批处理，保存后才产生一次提交。组件和两个
保存按钮的form ID契约、未保存零历史以及完整回归符合预测；confidence保持2，等待真实鼠标
验证后再复核。
