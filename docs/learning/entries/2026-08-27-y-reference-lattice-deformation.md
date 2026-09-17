# 以 Y 为参考的跨图晶格配准与可识别变形

- Kind: algorithm
- First seen: 2026-08-27
- Status: reproduced
- Confidence: 2
- Related change: `src/tfree_detection/reference_deformation.py`、`tfree-detect compare-reference`
- Last review: 2026-08-27

## Why this appeared

Y 被明确指定为标准结构模板，而 RA、H2 的像素大小、视野和取向不同。逐图 GPA 只能回答“相对各自平均晶格的局部波动”，不能直接回答“相对 Y 的变形”，因此需要新增跨图参考比较。

## Mental model

先用两组晶格基矢求出目标图到 Y 的对应关系。等向缩放和刚体旋转被当作成像、裁剪与坐标配准量移除；相似变换解释不了的基矢形状差异才进入全局无量纲畸变。目标图自身 GPA 的常量偏置先用可靠区中位数归零，再旋转到 Y 坐标并与全局形状差异合成。核心不变量是：仅改变放大倍数和刚体取向，不应产生相对 Y 的形状应变。没有共同地标不能恢复绝对平移；没有物理像素标定不能恢复等向晶格膨胀；没有弹性常数不能把应变换成 Pa/GPa 应力。

## Minimal reproduction

当前已有合成基矢测试覆盖已知方向性畸变。学习检查点保留的边界测试是：构造一个只经过等向缩放和旋转、没有形状变化的目标晶格，验证两个全局主应变都接近 0。

## Prediction

用户预测 A：如果目标晶格与 Y 完全相同，只是拍摄时放大 1.3 倍并旋转 5°，配准后全局主应变应接近 0。

## Observation

新增 `test_pure_zoom_and_rotation_do_not_create_shape_strain`。合成目标准确恢复 1.3 倍目标到参考缩放和 5°目标到参考旋转，同时两个全局主应变在 `1e-10%` 容差内为 0；用户预测与观察一致。

## Project integration map

CLI 加载同一 GPA 配置并关闭不参与张量计算的 DINO/SAM 推理；训练自由框架为 Y、RA、H2 生成逐图 GPA 证据；参考比较模块匹配基矢、分解配准相似变换、计算全局对数应变并合并局部 GPA；保存器输出 JSON、CSV、NPZ 和可视化，且拒绝覆盖非空结果目录。

## Theory and boundaries

全局量使用基矢映射的极分解和对数应变；局部量使用小应变 GPA。结果适合比较方向性拉伸、压缩、剪切和热点排序，不适合在缺少共同视野时做逐像素模板相减。

## Alternatives

特征点/光流配准可在同一视野且存在唯一地标时给出位移场；当前图案高度重复且未确认同一视野，因此没有采用其输出作为绝对迁移。

## Failure modes

- 把像素缩放当晶格膨胀：由纯缩放边界测试约束。
- 选错基矢顺序或正负号：枚举有向符号置换并禁止镜像映射。
- 把低相干区域当热点：所有统计受 GPA validity 和 coherence 门控。
- 把应力代理当真实应力：输出中明确记录缺少弹性常数，禁止使用 Pa/GPa。

## User contribution

用户指定 Y 是标准结构模板，并明确 RA、H2 的尺寸与视野不同，需要框架自行配准。这决定了参考所有权和不可直接逐像素相减的边界。用户还独立预测纯缩放与旋转在配准后不应留下形状应变，该预测现已固化为自动化边界测试。

## Validation

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q tests/test_reference_deformation.py tests/test_stage2.py tests/test_fourdstem.py`
- Observed: 9 passed；另有一条既有 NumPy/scikit-image 弃用警告。
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider -q tests/test_reference_deformation.py::test_pure_zoom_and_rotation_do_not_create_shape_strain`
- Observed: 1 passed；纯缩放与旋转边界成立。
- `git diff --check`
- Observed: passed。

## Retrieval questions

1. 为什么不同图片的像素周期不能直接解释为晶格常数变化？
2. 要把应变热点换成 GPa，应补充哪类材料和成像信息？

## Learning debt

尚未获得三张图的物理像素标定、共同视野地标和材料弹性常数，因此绝对等向膨胀、平移及 Pa/GPa 应力仍不可识别。

## Review history

- 2026-08-27：首次引入；已说明可识别量边界，等待学习检查点回答。
- 2026-08-27：用户预测 A（纯缩放/旋转不产生形状应变）；合成边界测试通过，状态更新为 reproduced、置信度 2。
