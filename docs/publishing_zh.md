# GitHub 发布说明

目标仓库：<https://github.com/yyt-2378/PhyEvo_Mic_STEM>

## 公开范围

首次提交应包含可复现代码与轻量研究材料：

- `src/`、`configs/`、`scripts/`、`tests/`；
- `schemas/`、协议文件和不含像素数据的轻量 registry/manifest；
- `README.md`、`plan.md`、`docs/`；
- 四张精选说明图和 Class1 的轻量指标/封存摘要；
- Python 包元数据与 CI 配置。

以下资产只保留在本地：

- 原始或标准化公开数据；
- 完整 `outputs/`、预测数组和中间缓存；
- 下载的预训练权重与 `vendor/`；
- benchmark 预测、私有 selection key、专家审阅临时包；
- `.env`、令牌、凭据和本地虚拟环境。

`.gitignore` 是发布边界的一部分。不要使用 `git add -f` 绕过它。

## 发布前检查

在项目根目录运行：

```bash
gh --version
gh auth status
pytest -q
python -m tfree_detection.cli --help
```

然后检查待提交范围和大文件：

```bash
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

不得提交第三方数据、下载权重、审阅私钥或 GitHub token。仓库目前没有添加开源许可证；
正式公开复用前应由项目负责人选择许可证。

## 首次提交

本项目是新仓库，目标远端为空。完成 GitHub CLI 安装与认证后，使用本地 Git 建立历史：

```bash
git init
git add \
  .github .gitignore README.md plan.md pyproject.toml \
  src configs scripts tests schemas benchmarks datasets docs \
  reference_scale_radial_wiener.png
git status --short
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:yyt-2378/PhyEvo_Mic_STEM.git
git push -u origin main
```

这次直接推送 `main` 是项目负责人的明确要求；后续开发建议使用功能分支和 pull request。
若当前安全环境不允许读取工作区外的 SSH 密钥，可把远端改为 HTTPS，并使用 GitHub CLI
的认证凭据；不要把 token 写入命令、配置文件或仓库。

## 发布后检查

- GitHub 首页能正确渲染架构 SVG 和四张结果图；
- Actions 中完整测试通过；
- 仓库没有原始数据、完整输出数组、权重或私有评测文件；
- 安装命令、最小检测命令和文档链接均可用；
- release/tag 只在锁定评测和版本说明准备完成后创建。
