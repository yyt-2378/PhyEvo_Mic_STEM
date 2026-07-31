# 浏览器双向 Canvas 与源像素几何

- Kind: architecture
- First seen: 2026-07-31
- Status: deferred
- Confidence: 1
- Related change: `review_canvas.py` 与 `review_dataset_viewer.py`
- Last review: —

## Why this appeared

正式审核页曾用 Plotly selection 模拟点、线和区域绘制。专家实际点击后没有任何坐标回传，
说明数据选择事件不能充当可靠的标注输入层。

## Mental model

浏览器 Canvas 拥有未提交的绘图状态；每次点击或一次区域拖动结束后，组件通过 Streamlit
V2 `setStateValue` 把完整几何回传给 Python。回传值始终使用原图像素 `(x, y)`，而不是
浏览器缩放后的 CSS 坐标。Python 只接收、校验并暂存草稿；点击正式保存后，SQLite 才拥有
不可变审核事实。页面缩放不能改变源像素几何，组件清空必须显式回传 `null`。

## Minimal reproduction

`tests/test_review_canvas.py` 使用32×48合成图验证底图嵌入、geometry type、坐标裁剪、最少
点数和清空语义。`tests/test_review_streamlit_app.py` 选择“修正几何”并验证真实154病例应用能
挂载双向组件。

## Prediction

用户通过截图和实际操作预测旧实现的结果：“在这里点击没有任何作用”。此次紧急修复未暂停
要求用户预测新组件行为，因此学习检查点记为 deferred。

## Observation

旧界面显示 Plotly 缩放/框选工具栏，而不是标注工具，确认交互模型与说明不一致。新组件在
Python侧完成载荷、坐标和挂载测试。真实鼠标测试进一步发现：第一个点附近可重合，后续点随
距离增大而漂移。根因是 Canvas 同时使用 `width: 100%`、`max-height` 和 `object-fit: contain`，
图像实际内容在元素外框内发生了留白缩放，而坐标仍按整个外框换算。

## Project integration map

`review_dataset_viewer.py` 提供图像、预测mask、geometry hint和当前草稿；
`review_canvas.py` 在浏览器绘制并回传源像素坐标；`ReviewSubmission` 再执行几何门禁；
`ReviewStore` 只保存明确提交的结果。

## Theory and boundaries

CSS显示尺寸和原始像素尺寸必须分离，但浏览器中只能存在一套等比例缩放。Canvas内容框由原图
宽高和可用视口共同计算；坐标换算使用扣除四边边框后的content box，不能使用带边框、留白或
`object-fit`的外框。点和折线在 pointer down 时提交；区域在 pointer up 时一次提交，避免鼠标
移动期间持续重运行。

## Alternatives

第三方 drawable-canvas 包需要增加和维护外部依赖；继续使用 Plotly selection 已被真实操作
证伪。内联 Streamlit V2组件不需要Node构建或新增运行服务，因此被采用。

## Failure modes

- 点击事件未回传；
- 页面缩放、边框或object-fit留白造成坐标渐进偏移；
- 区域拖动每帧触发重运行；
- 清空后旧草稿被重新注入；
- 少于最小点数仍被提交；
- 组件状态被误当成已经持久化的审核事实。

## User contribution

用户实际打开线上平台，提供无法点击的截图并指出操作无效。这一真实失败暴露了仅靠无头
Streamlit启动测试无法覆盖的浏览器交互缺口。

## Validation

- `tests/test_review_canvas.py`：4项通过，新增左上、中心、右下多点坐标回归；
- Canvas、存储、导出和Streamlit相邻回归：16项通过；
- 全项目回归：150项通过；
- 8502服务保持健康并热加载新源码。

## Retrieval questions

1. 为什么浏览器显示坐标不能直接保存为源图像像素坐标？
2. 为什么区域拖动只在 pointer up 时回传，而不是每次 pointer move 都回传？

## Learning debt

等待用户用真实鼠标确认点、折线和区域三种交互；之后补充浏览器端自动化测试。

## Review history

尚无。
