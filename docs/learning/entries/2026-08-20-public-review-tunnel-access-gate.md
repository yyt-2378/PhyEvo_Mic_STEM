# 无域名公网隧道与数据前置访问门禁

- Kind: architecture
- First seen: 2026-08-20
- Status: applied
- Confidence: 2
- Related change: `apps/review_dataset_viewer.py`、`tests/test_review_streamlit_app.py` 与 Tailscale Funnel 部署
- Last review: 2026-08-20

## Why this appeared

外部专家需要从任意互联网接入网络访问远程服务器上的审核平台，但项目不准备购买或配置
专用域名。直接公开 Streamlit 端口会暴露病例、预测和 SQLite 写入口，因此公网连通和应用
访问门禁必须作为同一个安全边界设计。

## Mental model

Tailscale Funnel 只负责把固定 `ts.net` HTTPS 地址转发到服务器回环端口，不拥有审核身份和标注事实。
Streamlit 共享访问码必须先于审核资产加载执行：未认证会话只渲染密码表单并立即停止；认证
成功后才读取队列、预测并打开 SQLite。Reviewer ID 继续承担科研记录身份，不能代替访问码。
SQLite 仍是唯一持久事实，隧道和浏览器会话都可重建。

## Minimal reproduction

聚焦复现位于 `tests/test_review_streamlit_app.py::test_access_code_gate_blocks_assets_before_authentication`。
它使用项目内临时 SQLite 路径运行真实 Streamlit AppTest，依次提交缺失、错误和正确访问码，
检查错误阶段没有病例指标且数据库文件不存在，正确阶段才出现154病例并创建 store。

## Prediction

Codex 询问：错误访问码时应允许看到病例但禁止保存，还是完全不加载病例、预测和数据库。
用户选择 B：错误访问码时不加载病例、预测或数据库。

## Observation

门禁测试在缺失和错误访问码阶段均未创建 SQLite，也没有渲染任何病例指标；正确访问码后
才加载154个病例。实际 Streamlit 随后仅绑定 `127.0.0.1:8502`，公网请求由 HTTPS Funnel
转发，健康端点返回 `ok`。

## Project integration map

浏览器先连接 Tailscale 固定 HTTPS 地址，Funnel 将流量转到回环 Streamlit。`main()` 在
`st.set_page_config` 后立即调用 `_require_access_code()`；只有该函数返回后，缓存加载器、
资产审计、Reviewer ID 和 `ReviewStore` 才能执行。专家提交仍由 `ReviewStore.save()` 原子写入
本地数据库，隧道不保存或复制标注。

## Theory and boundaries

共享访问码是受邀小组的粗粒度门禁，不是个人身份系统。它适合无域名、阶段性审核，但不能
独立撤销某位专家，也不提供 MFA。Tailscale Funnel 的 DNS 地址固定到已认证节点，项目本地
守护器会自动重启退出的 Funnel 与 Streamlit 子进程；服务器关机、重启或断网仍会中断访问。
任何隧道中断都不会影响已提交的 SQLite 数据。

## Alternatives

直接开放公网端口缺少 TLS 和门禁，因此未采用。Cloudflare Quick Tunnel 的随机地址会随
进程重建而变化，不符合连续审核需要。Streamlit Community Cloud 不保证本地 SQLite 持久性，
并需要迁移本地资产。Tailscale Funnel 可以在 Personal 免费方案下复用固定 `ts.net` 地址，
因此用于本次无自有域名的阶段性部署；长期高敏感部署仍需独立账号和机构级访问代理。

## Failure modes

- 访问码校验放在资产加载之后，导致未认证请求仍读取受保护数据；
- Streamlit 监听所有网卡，使同网段访问绕过 tunnel；
- 访问码写入仓库、文档或进程参数；
- tunnel 退出后链接失效，但负责人误以为服务仍在线；
- 多位专家共用 Reviewer ID，破坏双专家独立性。

## User contribution

用户明确选择聚焦测试的不变量 B：错误访问码时病例、预测和数据库都不得加载。生产调用顺序
和 `test_access_code_gate_blocks_assets_before_authentication` 据此固定。

## Validation

- `tests/test_review_streamlit_app.py`：3项通过；
- 官方发布摘要与项目内 Tailscale 二进制 SHA-256 一致；
- Streamlit 监听 `127.0.0.1:8502`，本地健康端点返回 `ok`；
- Tailscale Funnel 公网 HTTPS 首页返回 HTTP 200，WebSocket 返回 101；
- 项目本地守护器在 Streamlit 子进程退出后约4秒启动替代进程并恢复健康端点。

## Retrieval questions

1. 为什么访问码校验必须发生在 `_samples()`、`_prediction_index()` 和 `ReviewStore()` 之前？
2. tunnel 进程退出后，哪些状态会丢失，哪些审核事实仍然保留？

## Learning debt

共享访问码仍不是个人身份认证；守护器也不能跨服务器重启自动恢复。长期部署需要可单独撤权
的账号策略、主机级开机恢复和外部可用性监控。

## Review history

2026-08-20：用户选择未认证时完全不加载数据；聚焦 AppTest 和实际回环加公网 tunnel 部署均
符合预测，状态记为 `applied`，confidence 2。随后因 Quick Tunnel 临时地址失效，切换到
Personal 免费方案上的固定 Tailscale Funnel，并验证公网 HTTPS、WebSocket 和子进程自动恢复。
