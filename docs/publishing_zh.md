# GitHub 发布说明

目标仓库：<https://github.com/yyt-2378/PhyEvo_Mic_STEM>

## 公开范围

当前为已有仓库的功能分支发布，不重新初始化仓库、不覆盖 `main`。本次同步包含：

- 当前 `src/`、`configs/`、`scripts/`、`tests/` 及程序级 RSI 的 `experiments/`；
- `schemas/`、协议文件和不含像素数据的轻量 registry/manifest；
- `README.md`、`plan.md`、`docs/`；
- 已有公开夹具和说明图（不新增上传本地实验结果）；
- Python 包元数据与 CI 配置。

以下资产只保留在本地：

- 原始或标准化公开数据；
- 完整 `outputs/`、预测数组和中间缓存；
- 下载的预训练权重与 `vendor/`；
- benchmark 预测、私有 selection key、专家审阅临时包；
- `.env`、令牌、凭据和本地虚拟环境。
- 稿件、PPT、论文构建脚本、个人学习记录及本轮未明确纳入的本地说明材料。

`.gitignore` 是发布边界的一部分。不要使用 `git add -f` 绕过它。

## 发布前检查

源码快照沿用包版本 `0.4.0`，用功能分支提交哈希固定，不为整理目录而修改科学引擎。
先运行不写文件的语法检查：

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_source_release.py --profile syntax
```

Linux 下有 bubblewrap 时，使用现有隔离环境执行源码专用测试：

```bash
PYTHONDONTWRITEBYTECODE=1 .venvs/sam2/bin/python scripts/check_source_release.py --profile core
PYTHONDONTWRITEBYTECODE=1 .venvs/sam2/bin/python scripts/check_source_release.py --profile rsi
# 审核界面使用已安装 Streamlit 的解释器；不会启动服务或连接真实数据库。
PYTHONDONTWRITEBYTECODE=1 python scripts/check_source_release.py --profile review
```

`core` 使用 Git 可见的 `tests/test_*.py`，不收集已被忽略的稿件／本地结果检查。
`rsi` 显式运行最新引擎及其关键依赖的 20 个测试文件；不声称覆盖全部历史编排。
两个依赖官方下载源码的对照测试（其中一个另需 OpenCV）被明确排除并记入回执；
其他可选依赖缺失会在 pytest 的 skip 汇总中显示。发布时必须报告这些排除与跳过。

运行器先制作源码副本，逐文件校验哈希，再在断网、只读源码的进程内执行。
`outputs/`、稿件、第三方下载目录、原图与真实标注不会被复制或挂载。
每次的 `manifest.json`、`pytest.log`、`result.json` 和 `work/junit.xml` 仅保存在
新建的 `.codex-tmp/source-release-*`，不覆盖旧目录、不清理任何已有文件。
暂存白名单后，在相同命令末尾添加 `--staged`，验证精确的 Git index 字节；
未暂存的数据清单改动与未跟踪的本地稿件脚本不会混入该源码副本。
不提供解除隔离的降级选项；隔离或依赖不可用时应修正环境，不能记为测试通过。

核心测试所需库由现有项目环境提供：项目基础依赖、pytest、PyTorch，以及对应测试的
Pillow/tifffile/h5py、文档报告依赖；审核测试另需 `review` extra。
运行器本身不安装软件或下载权重。默认 `pytest -q` 在有本地资产的工作区可能额外收集
稿件测试，因此不能代替上述明确范围的源码验收。

然后检查待提交范围和大文件：

```bash
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

不得提交第三方数据、下载权重、审阅私钥或 GitHub token。仓库目前没有添加开源许可证；
正式公开复用前应由项目负责人选择许可证。

## 提交与功能分支推送

先确认当前工作树中的既有改动，逐项审查白名单；禁止 `git add .`、`git add -f` 或整库强制清理。
`datasets/manifest.yaml` 等不属于本轮整理的改动留在工作区，不随本轮提交。
已在本地完成但尚未提交的框架源代码可以在验证通过后按白名单同步；本次不修改其算法。

```bash
git switch -c codex/framework-source-freeze-20261019
# git add 后面必须是本次审查通过的精确文件清单，不是整个工作区。
git status --short
git diff --cached --check
git diff --cached --stat
git diff --cached
git commit -m "chore: freeze framework sources and organize release entrypoints"
git push -u origin codex/framework-source-freeze-20261019
```

使用已有远端与正常认证通道，不读取凭据文件、不修改全局 Git/SSH/Codex 配置。
若认证失败，保留本地提交并报告失败，不重设远端或强制推送。
源码删除需另外列出精确文件并获得确认；保留代码依赖比按版本号批量删文件更重要。

## 2026-10-19 源码整理验收

| 范围 | 通过 | 跳过 | 显式排除 | 失败／错误 |
| --- | ---: | ---: | ---: | ---: |
| 核心代码与脚本 | 605 | 12 | 2 | 0 |
| 最新程序级 RSI（20 个测试文件） | 187 | 4 | 0 | 0 |
| 专家审核与界面 | 45 | 0 | 0 | 0 |

三个范围部分重叠，不能相加当成不同测试总数。核心环境未安装 Streamlit，
因此界面在有 Streamlit 的现有解释器中单独运行并通过。
其余跳过项是未分发的第三方源代码、FAU CIF、DINO checkpoint，以及未显式启用的 GPU 测试。
两项显式排除只涉及官方下载源码的对照，不是隐藏算法回归。
以上是本地源码测试，不是远端 Actions 状态，也没有复核或更改论文实验分数。

本轮新增源码专用检查入口，修正审计脚本对 JSON 同长度快速改写的缓存漏检，
并将一个依赖真实旧输出目录的覆盖保护测试改用临时夹具；未改变检测算法。
经逐项授权，仅删除了 `experiments/rsi_program_evolution_v1/route_probe.py` 和
`experiments/rsi_program_evolution_v1/storage_preflight.py` 两个一次性旧脚本。
历史依赖与其他源文件保留；`outputs/`、专家标注、稿件和权重没有被修改或清理。

## 发布后检查

- GitHub 首页能正确渲染架构 SVG 和四张结果图；
- 本地各测试范围的实际通过／失败／跳过已记录；远端 Actions 状态另行确认，不混为一谈；
- 仓库没有原始数据、完整输出数组、权重或私有评测文件；
- 安装命令、最小检测命令和文档链接均可用；
- 核对远端功能分支 SHA 等于本地提交；不自动合并 `main` 或创建 release/tag。
