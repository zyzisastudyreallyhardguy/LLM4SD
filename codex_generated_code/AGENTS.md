# AGENTS Notes For `codex_generated_code`

## Purpose

这个目录用于存放“把 `synthesize_model_response/deepresearch` 里的规则响应，整理成可执行 feature 提取代码”的产物。

这里的代码不是普通的规则文本归档，而是后续训练与评估会直接调用的 feature extractor。

因此，每次新增一个任务时，目标都不是“把规则原样翻译成 Python”，而是：

1. 提取出可计算、可复现的分子特征
2. 用稳定的接口暴露这些特征
3. 让它能够无缝接入 `feature_backends` 和后续 RandomForest 训练

## Source Of Truth

每次开始新任务时，默认输入来源是：

- `synthesize_model_response/deepresearch/<Task>/...`

最常见的情况是类似：

- `synthesize_model_response/deepresearch/BBB_Martins/deepresearch_pk_response.txt`

如果 deepresearch 目录结构与现有任务不同，先确认真实源文件位置，再开始生成代码。

## Required Output Structure

每个任务建议采用如下结构：

```text
codex_generated_code/
  <Task>/
    __init__.py
    <task_lower>_rule_features.py
```

例如：

```text
codex_generated_code/
  BBB_Martins/
    __init__.py
    bbb_martins_rule_features.py
```

## Required Public Interface

为了方便接入 `feature_backends`，每个任务生成的主模块都应尽量提供下面这些函数：

```python
featurize_smiles(smiles: str) -> dict[str, float]
featurize_smiles_list(smiles_list, *, include_smiles: bool = False, on_error: str = "raise") -> pd.DataFrame
get_feature_names() -> list[str]
get_feature_descriptions() -> dict[str, str]
```

如果任务中存在明确跳过的规则组，也建议提供：

```python
get_skipped_rule_groups() -> list[str]
```

`__init__.py` 应该把这些核心接口导出，方便上层直接 import。

## Interface Expectations

所有面向训练的 feature 代码都应满足下面这些约束：

- 输入是 SMILES
- 输出必须是数值特征，不要返回字符串、列表或对象
- 缺失值统一使用 `math.nan`
- 批量接口返回 `DataFrame`
- 列名顺序要稳定
- 同一个分子多次计算结果要一致
- 不依赖在线 API
- 不依赖需要额外人工交互的运行时环境

## Conversion Strategy

将 deepresearch 响应转成代码时，遵守下面的思路。

### 1. 先提炼“可计算的内容”

不是所有文字规则都能直接实现。优先保留：

- RDKit 可直接计算的描述符
- 明确阈值规则
- 可转成 0/1 flag 的结构条件
- 可转成连续数值的化学性质

对于明显无法本地稳定实现的内容，例如：

- 需要实验测量的数据
- 需要外部数据库的数据
- 需要复杂专用模型但当前仓库没有可靠依赖的数据

不要硬编造，应该明确跳过，并记录到 `get_skipped_rule_groups()`。

### 2. 不要把“规则文字”当接口

最终训练使用的是 feature 列，不是原始文本。

因此不要依赖：

- `Calculate ...`
- `Check ...`
- 原始项目符号顺序

应该直接设计稳定的 feature name，例如：

- `mol_weight`
- `tpsa_le_90`
- `most_basic_pka_le_8`

### 3. 允许一个文本规则拆成多个 feature

一个 deepresearch 规则经常可以拆成：

- 一个连续描述符
- 一个或多个阈值 flag
- 一个组合规则分数

这是允许的，而且通常比机械“一条规则对应一个函数”更适合训练。

### 4. 所有特征都要能直接进入表格

`featurize_smiles()` 返回的字典要能直接转成一行表。

也就是说：

- 每个键是固定 feature 名
- 每个值是 `float` 或可转成 `float` 的数

### 5. pKa 相关特征优先复用现有实现链路

如果某个任务需要：

- `most_basic_pka`
- `most_acidic_pka`
- `num_basic_sites`
- `num_acidic_sites`
- `estimated_logd_ph74`
- `base_protonated_fraction_ph74`
- 其他依赖 pKa 预测的衍生特征

不要自己重新发明一套 pKa 接口或随意换实现。

优先参考：

- `codex_generated_code/BBB_Martins/bbb_martins_rule_features.py`
- `/data1/tianang/Projects/Intern-S1/tools/pka_related_tools.py`

推荐做法：

- 像 `BBB_Martins` 一样封装 `_get_pka_predictor()`
- 通过 `sys.path` 接入 `Intern-S1`
- 从 `tools.pka_related_tools` 导入 `_get_pka_predictor`
- 使用 `predictor.predict(mol)` 获取 `base_sites_1` / `acid_sites_1`
- 再在当前任务模块里把它们整理成稳定的数值特征

实现原则：

- 先复用现有 `MolGpKa` 使用方式，再做任务级特征加工
- 只要任务需要 pKa 或 logD 相关特征，就默认走 `MolGpKa` 链路，不要增加类似 `LLM4SD_DISABLE_MOLGPKA` 的环境变量开关，也不要为了提速跳过它
- 如果 pKa helper 不可用，返回 `math.nan`，不要伪造结果
- 如果只需要 basic pKa，也尽量沿用同一条 helper 链路，避免不同任务之间口径不一致

## Recommended Module Design

每个任务主模块建议包含这些部分：

1. feature 规格定义
2. 公共接口函数
3. RDKit / 其他本地依赖的加载封装
4. 单分子特征计算逻辑
5. 批量特征计算逻辑
6. 可选 CLI

建议保留一个空模板或有序模板，用来保证所有分子输出列一致。

## Recommended CLI

为了方便人工检查和批量导出，建议像 `BBB_Martins` 一样提供命令行入口，支持：

- `--smiles`
- `--input_csv`
- `--smiles_col`
- `--output_csv`
- `--include_smiles`
- `--on_error`
- `--show_metadata`

这不是强制，但强烈推荐，因为它能明显降低后续排查成本。

## Validation Checklist

每次新建或修改一个任务模块后，至少检查下面这些点：

1. 单分子是否能跑通
- 例如：`python <task_module>.py --smiles "CCO"`

2. `featurize_smiles()` 是否返回稳定字典

3. `featurize_smiles_list()` 是否返回 `DataFrame`

4. `get_feature_names()` 与实际输出列是否一致

5. 批量模式遇到坏分子时，`on_error="nan"` 是否能正常工作

6. 没有把不可实现的规则偷偷硬编码成伪结果

## Integration Checklist

只把代码放进 `codex_generated_code` 还不够。

如果这个任务后续要接训练流程，还要检查：

1. 是否需要在 `feature_backends/` 下增加对应 adapter
2. 是否需要在 `feature_backends/registry.py` 里注册 backend
3. 是否需要更新 `feature_backends/README.md`
4. 是否需要更新 `eval.py` / 其他训练入口的数据接线

核心原则：

- `codex_generated_code` 负责“实现特征”
- `feature_backends` 负责“统一接入接口”

不要把这两层职责混在一起。

## Naming Guidance

建议：

- 任务目录名与任务名保持一致
- 主文件名用小写下划线风格
- feature 名尽量短、明确、稳定

避免：

- 用随机后缀命名 feature
- 用原始 response 中很长的整句做列名
- 用会频繁变化的临时命名

## Missingness Policy

如果某个特征在某个分子上无法合理定义：

- 返回 `math.nan`

不要：

- 默默返回 0
- 用字符串 `"NA"`
- 用任意常数代替

除非这个 0 本身就有明确化学语义。

## When In Doubt

如果 deepresearch 规则存在歧义，优先遵守下面顺序：

1. 可计算性
2. 接口稳定性
3. 与训练流程兼容
4. 化学语义尽量保真

不要为了“表面上覆盖更多规则”而引入不稳定、不可验证或无法接训练的实现。

## Maintenance Rule

如果某个任务模块的公共接口发生变化，请同步检查：

- 对应任务目录下的 `__init__.py`
- `feature_backends` 中是否有受影响的 adapter
- `feature_backends/README.md` 是否还准确

## Current Practical Standard

目前 `BBB_Martins` 是这个目录下的参考实现。

后续新增任务时，应尽量向它看齐，尤其是：

- 公开接口
- 批量处理方式
- metadata 暴露方式
- 与 `feature_backends` 的兼容性
