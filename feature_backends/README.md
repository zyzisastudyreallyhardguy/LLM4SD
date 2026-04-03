# feature_backends 说明

这个目录负责把“不同来源的特征提取逻辑”包装成统一接口，供后面的训练与评估脚本直接调用。

现在这层的目标有两个：

1. 让 `code_gen_and_eval.py` 和 `eval.py` 不需要知道特征到底来自哪里。
2. 让以后新增新的特征模块时，只需要补一个新的 backend，而不需要反复改 RandomForest 主流程。

## 当前整体设计

训练脚本现在遵循下面这条链路：

1. 通过 `load_feature_backend(...)` 加载一个 backend。
2. backend 负责把一批 `SMILES` 转成 `DataFrame`。
3. `prepare_feature_matrices(...)` 负责统一做列对齐、缺失值处理和可选标准化。
4. 处理后的矩阵再送入下游模型，例如 RandomForestClassifier / RandomForestRegressor，或者 `eval.py --estimator linear` 对应的 LogisticRegression / LinearRegression。

这意味着：

- 特征来源可以替换
- 训练逻辑可以复用
- 现在导出 feature importance 时也能共用同一套列名顺序

## 文件说明

### `__init__.py`

作用：

- 对外暴露这个目录最常用的两个入口：
  - `load_feature_backend`
  - `prepare_feature_matrices`

你通常不需要直接改这个文件，除非：

- 新增了新的公共函数，想从包顶层直接导出

示例：

```python
from feature_backends import load_feature_backend, prepare_feature_matrices
```

### `registry.py`

作用：

- 根据名字加载具体的 backend
- 屏蔽不同 backend 的实现差异

当前支持：

- `generated_rules`
- `bbb_martins`
- `dili`
- `clintox`
- `herg`
- `pampa_ncats`
- `skin_reaction`
- `pgp_broccatelli`
- `carcinogens_lagunin`
- `ames`
- `bioavailability_ma`
- `hia_hou`
- `cyp2c9_substrate_carbonmangels`
- `cyp2d6_substrate_carbonmangels`
- `cyp3a4_substrate_carbonmangels`
- `sarscov2_3clpro_diamond`
- `sarscov2_vitro_touret`

核心函数：

```python
load_feature_backend(name, *, function_names=None, namespace=None)
```

使用方式：

```python
backend = load_feature_backend("bbb_martins")
```

也可以加载 DILI DeepResearch 特征：

```python
backend = load_feature_backend("dili")
```

也可以加载 ClinTox DeepResearch 特征：

```python
backend = load_feature_backend("clintox")
```

也可以加载 hERG DeepResearch 特征：

```python
backend = load_feature_backend("herg")
```

也可以加载 PAMPA_NCATS DeepResearch 特征：

```python
backend = load_feature_backend("pampa_ncats")
```

Skin Reaction DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("skin_reaction")
```

P-gp Broccatelli DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("pgp_broccatelli")
```

Carcinogens_Lagunin DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("carcinogens_lagunin")
```

AMES DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("ames")
```

Bioavailability_Ma DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("bioavailability_ma")
```

HIA_Hou DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("hia_hou")
```

CYP2C9_Substrate_CarbonMangels DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("cyp2c9_substrate_carbonmangels")
```

CYP2D6_Substrate_CarbonMangels DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("cyp2d6_substrate_carbonmangels")
```

CYP3A4_Substrate_CarbonMangels DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("cyp3a4_substrate_carbonmangels")
```

SARSCoV2_3CLPro_Diamond DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("sarscov2_3clpro_diamond")
```

SARSCoV2_Vitro_Touret DeepResearch 特征也是同样的用法：

```python
backend = load_feature_backend("sarscov2_vitro_touret")
```

如果使用 `generated_rules`，需要额外提供：

- `function_names`
- `namespace`

例如：

```python
backend = load_feature_backend(
    "generated_rules",
    function_names=valid_function_names,
    namespace=globals(),
)
```

什么时候需要改它：

- 新增一个 backend 时，要在这里注册名字和加载逻辑

### `bbb_martins.py`

作用：

- 把 `codex_generated_code/BBB_Martins/bbb_martins_rule_features.py` 包装成统一 backend 接口

它本身是一个 adapter，核心职责不是重新计算特征，而是把现有 BBB 模块接到统一框架里。

当前提供的方法包括：

- `name`
- `get_feature_names()`
- `get_feature_descriptions()`
- `featurize_smiles(smiles)`
- `featurize_smiles_list(smiles_list, on_error="raise")`

使用方式：

```python
backend = load_feature_backend("bbb_martins")
df = backend.featurize_smiles_list(["CCO", "CCN"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 BBB Martins 规则特征
- 不想再走 LLM 生成函数执行那条路径

什么时候需要改它：

- BBB Martins 特征模块路径变化
- BBB 模块接口变化
- 想给这个 backend 增加更多元信息接口

### `dili.py`

作用：

- 把 `codex_generated_code/DILI/dili_rule_features.py` 包装成统一 backend 接口

和 `bbb_martins.py` 一样，它是一个 adapter，本身不重新定义特征，只负责把 DILI 特征模块接到统一框架里。

当前提供的方法包括：

- `name`
- `get_feature_names()`
- `get_feature_descriptions()`
- `featurize_smiles(smiles)`
- `featurize_smiles_list(smiles_list, on_error="raise")`

使用方式：

```python
backend = load_feature_backend("dili")
df = backend.featurize_smiles_list(["CCO", "O=[N+]([O-])c1ccccc1"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 DILI DeepResearch 结构警报和理化性质特征
- 希望在 TDC `DILI` 子任务上复用统一训练与预处理流程

### `clintox.py`

作用：

- 把 `codex_generated_code/ClinTox/clintox_rule_features.py` 包装成统一 backend 接口

它和 `bbb_martins.py` / `dili.py` 一样，是一个 adapter，本身不重新定义特征，只负责把 ClinTox 的理化窗口、pKa 代理和结构警报特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("clintox")
df = backend.featurize_smiles_list(["CCO", "Nc1ccccc1"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 ClinTox DeepResearch 特征
- 希望在 MoleculeNet `clintox` 或 TDC `ClinTox` 任务上复用统一训练与预处理流程

### `herg.py`

作用：

- 把 `codex_generated_code/hERG/herg_rule_features.py` 包装成统一 backend 接口

它和 `bbb_martins.py` / `dili.py` / `clintox.py` 一样，是一个轻量 adapter，本身不重新定义特征，只负责把 hERG 的理化窗口、pKa/logD 代理、碱性中心和结构警报特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("herg")
df = backend.featurize_smiles_list(["CCO", "CN(C)CCOc1ccc(cc1)C(c1ccccc1)=C(C#N)C#N"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 hERG DeepResearch 特征
- 希望在 TDC `hERG` 子任务上复用统一训练与预处理流程

### `skin_reaction.py`

作用：

- 把 `codex_generated_code/Skin_Reaction/skin_reaction_rule_features.py` 包装成统一 backend 接口

它和 `bbb_martins.py` / `dili.py` 一样，是一个轻量 adapter，本身不重新定义特征，只负责把 Skin Reaction 的结构警报和理化性质特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("skin_reaction")
df = backend.featurize_smiles_list(["CCO", "O=C(CCl)Cl"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 Skin_Reaction DeepResearch 特征
- 希望在 TDC `Skin_Reaction` 子任务上复用统一训练与预处理流程

### `pampa_ncats.py`

作用：

- 把 `codex_generated_code/PAMPA_NCATS/pampa_ncats_rule_features.py` 包装成统一 backend 接口

和 `bbb_martins.py` / `dili.py` 一样，它是一个轻量 adapter，本身不重新定义特征，只负责把 PAMPA_NCATS 的被动渗透相关特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("pampa_ncats")
df = backend.featurize_smiles_list(["CCO", "CC(=O)O"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 PAMPA_NCATS DeepResearch 理化性质、logD/pKa 代理和酸性基团特征
- 希望在 TDC `PAMPA_NCATS` 子任务上复用统一训练与预处理流程

### `pgp_broccatelli.py`

作用：

- 把 `codex_generated_code/Pgp_Broccatelli/pgp_broccatelli_rule_features.py` 包装成统一 backend 接口

它和 `bbb_martins.py` / `dili.py` / `skin_reaction.py` 一样，是一个轻量 adapter，本身不重新定义特征，只负责把 P-gp 抑制相关的理化性质、芳香性、碱性中心和疏水 bulky motif 代理特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("pgp_broccatelli")
df = backend.featurize_smiles_list(["CCO", "CN(C)CCOc1ccc(cc1)C(c1ccccc1)=C(C#N)C#N"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 Pgp_Broccatelli DeepResearch 特征
- 希望在 TDC `Pgp_Broccatelli` 子任务上复用统一训练与预处理流程

### `carcinogens_lagunin.py`

作用：

- 把 `codex_generated_code/Carcinogens_Lagunin/carcinogens_lagunin_rule_features.py` 包装成统一 backend 接口

它和 `bbb_martins.py` / `dili.py` / `skin_reaction.py` / `pgp_broccatelli.py` 一样，是一个轻量 adapter，本身不重新定义特征，只负责把 Carcinogens_Lagunin 的致癌结构警报、平面多芳香代理和理化性质特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("carcinogens_lagunin")
df = backend.featurize_smiles_list(["c1ccc2cc3ccccc3cc2c1", "N(CCCl)CCCl"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 Carcinogens_Lagunin DeepResearch 特征
- 希望在 TDC `Carcinogens_Lagunin` 子任务上复用统一训练与预处理流程

### `ames.py`

作用：

- 把 `codex_generated_code/AMES/ames_rule_features.py` 包装成统一 backend 接口

它和 `bbb_martins.py` / `dili.py` / `skin_reaction.py` / `pgp_broccatelli.py` / `carcinogens_lagunin.py` 一样，是一个轻量 adapter，本身不重新定义特征，只负责把 AMES 的突变原性结构警报、融合多芳香体系代理和基础理化性质特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("ames")
df = backend.featurize_smiles_list(["Nc1ccccc1", "O=[N+]([O-])c1ccccc1"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 AMES DeepResearch 特征
- 希望在 TDC `AMES` 子任务上复用统一训练与预处理流程

### `bioavailability_ma.py`

作用：

- 把 `codex_generated_code/Bioavailability_Ma/bioavailability_ma_rule_features.py` 包装成统一 backend 接口

它和 `bbb_martins.py` / `dili.py` / `pampa_ncats.py` 一样，是一个轻量 adapter，本身不重新定义特征，只负责把口服生物利用度相关的规则窗口、芳香性/饱和度/手性代理，以及 pKa/logD 代理特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("bioavailability_ma")
df = backend.featurize_smiles_list(["CCO", "CCN(CC)CC"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 Bioavailability_Ma DeepResearch 特征
- 希望在 TDC `Bioavailability_Ma` 子任务上复用统一训练与预处理流程

### `hia_hou.py`

作用：

- 把 `codex_generated_code/HIA_Hou/hia_hou_rule_features.py` 包装成统一 backend 接口

它和 `bioavailability_ma.py` / `bbb_martins.py` 一样，是一个轻量 adapter，本身不重新定义特征，只负责把 HIA 的口服吸收理化窗口、Palm/Veber/Ghose 规则，以及 pKa 驱动的肠道 pH 中性分数代理特征接到统一框架里。

使用方式：

```python
backend = load_feature_backend("hia_hou")
df = backend.featurize_smiles_list(["CC(=O)Nc1ccccc1", "CCO"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 HIA_Hou DeepResearch 特征
- 希望在 TDC `HIA_Hou` 子任务上复用统一训练与预处理流程

### `cyp2c9_substrate_carbonmangels.py`

作用：

- 把 `codex_generated_code/CYP2C9_Substrate_CarbonMangels/cyp2c9_substrate_carbonmangels_rule_features.py` 包装成统一 backend 接口

和其他 DeepResearch backend 一样，它本身不重新定义特征，只负责把 CYP2C9 substrate 任务的理化窗口、pKa/logD 代理、酸性 motif，以及 benzylic/allylic/alpha-heteroatom 氧化位点 proxy 接到统一框架里。

当前提供的方法包括：

- `name`
- `get_feature_names()`
- `get_feature_descriptions()`
- `featurize_smiles(smiles)`
- `featurize_smiles_list(smiles_list, on_error="raise")`

使用方式：

```python
backend = load_feature_backend("cyp2c9_substrate_carbonmangels")
df = backend.featurize_smiles_list(["CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O", "Cn1cnc2n(C)c(=O)n(C)c(=O)c12"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 CYP2C9_Substrate_CarbonMangels DeepResearch 特征
- 希望在 TDC `CYP2C9_Substrate_CarbonMangels` 子任务上复用统一训练与预处理流程

### `cyp2d6_substrate_carbonmangels.py`

作用：

- 把 `codex_generated_code/CYP2D6_Substrate_CarbonMangels/cyp2d6_substrate_carbonmangels_rule_features.py` 包装成统一 backend 接口

和其他 DeepResearch backend 一样，它本身不重新定义特征，只负责把 CYP2D6 substrate 任务的理化窗口、pKa/logD 代理、碱性中心、芳香平面性，以及 benzylic/allylic/alkoxy 氧化位点 proxy 接到统一框架里。

当前提供的方法包括：

- `name`
- `get_feature_names()`
- `get_feature_descriptions()`
- `featurize_smiles(smiles)`
- `featurize_smiles_list(smiles_list, on_error="raise")`

使用方式：

```python
backend = load_feature_backend("cyp2d6_substrate_carbonmangels")
df = backend.featurize_smiles_list(["CN(C)CCOc1ccc(cc1)C(C)=O", "CC(C)(Oc1ccc(Cl)cc1)C(=O)O"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 CYP2D6_Substrate_CarbonMangels DeepResearch 特征
- 希望在 TDC `CYP2D6_Substrate_CarbonMangels` 子任务上复用统一训练与预处理流程

### `cyp3a4_substrate_carbonmangels.py`

作用：

- 把 `codex_generated_code/CYP3A4_Substrate_CarbonMangels/cyp3a4_substrate_carbonmangels_rule_features.py` 包装成统一 backend 接口

和其他 DeepResearch backend 一样，它本身不重新定义特征，只负责把 CYP3A4 substrate 任务的体积/脂溶性窗口、芳香性和柔性、heterocycle 与 tertiary-amine 或 basic-center proxy、pKa/logD 代理、卤素，以及保守的 steroid-like fused-ring penalty 接到统一框架里。

当前提供的方法包括：

- `name`
- `get_feature_names()`
- `get_feature_descriptions()`
- `featurize_smiles(smiles)`
- `featurize_smiles_list(smiles_list, on_error="raise")`

使用方式：

```python
backend = load_feature_backend("cyp3a4_substrate_carbonmangels")
df = backend.featurize_smiles_list(["CN1CCN(CC1)C(=O)OC[C@H]2CO[C@@H](n3ccnc3N)[C@H]2O", "CC(C)NCC(O)COc1cccc2ccccc12"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 CYP3A4_Substrate_CarbonMangels DeepResearch 特征
- 希望在 TDC `CYP3A4_Substrate_CarbonMangels` 子任务上复用统一训练与预处理流程

### `sarscov2_3clpro_diamond.py`

作用：

- 把 `codex_generated_code/SARSCoV2_3CLPro_Diamond/sarscov2_3clpro_diamond_rule_features.py` 包装成统一 backend 接口

和其他 DeepResearch backend 一样，它本身不重新定义特征，只负责把 3CLpro 特征模块接到统一框架里。

当前提供的方法包括：

- `name`
- `get_feature_names()`
- `get_feature_descriptions()`
- `featurize_smiles(smiles)`
- `featurize_smiles_list(smiles_list, on_error="raise")`

使用方式：

```python
backend = load_feature_backend("sarscov2_3clpro_diamond")
df = backend.featurize_smiles_list(["CCO", "N#CCC(=O)N1CCC(C(N)=O)CC1"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 SARSCoV2_3CLPro_Diamond DeepResearch 特征
- 希望在 TDC `SARSCoV2_3CLPro_Diamond` 子任务上复用统一训练与预处理流程

### `sarscov2_vitro_touret.py`

作用：

- 把 `codex_generated_code/SARSCoV2_Vitro_Touret/sarscov2_vitro_touret_rule_features.py` 包装成统一 backend 接口

和其他 DeepResearch backend 一样，它本身不重新定义特征，只负责把 SARS-CoV-2 体外活性任务的理化窗口、pKa/logD 代理，以及 PAINS/反应性/金属螯合等 liability 特征接到统一框架里。

当前提供的方法包括：

- `name`
- `get_feature_names()`
- `get_feature_descriptions()`
- `featurize_smiles(smiles)`
- `featurize_smiles_list(smiles_list, on_error="raise")`

使用方式：

```python
backend = load_feature_backend("sarscov2_vitro_touret")
df = backend.featurize_smiles_list(["CCO", "CCOc1ccc(cc1)C(=O)NCCN"], on_error="nan")
```

适用场景：

- 想直接使用整理好的 SARSCoV2_Vitro_Touret DeepResearch 特征
- 希望在 TDC `SARSCoV2_Vitro_Touret` 子任务上复用统一训练与预处理流程

### `generated_rules.py`

作用：

- 把当前仓库里“由 LLM 生成并 `exec` 到内存中的规则函数”包装成统一 backend

这个 backend 是为了兼容项目原来的主流程。

内部思路：

1. 接收一组函数名
2. 从传入的 `namespace` 中找到对应函数
3. 对每个 SMILES 调用这些函数
4. 把结果拼成 `DataFrame`

当前类：

```python
GeneratedRulesBackend
```

典型用法：

```python
backend = load_feature_backend(
    "generated_rules",
    function_names=valid_function_names,
    namespace=globals(),
)
df = backend.featurize_smiles_list(smiles_list, on_error="nan")
```

注意事项：

- 如果某个函数报错，会打印错误并把该位置记成 `NaN`
- 如果输入 SMILES 无效，行为由 `on_error` 控制

适用场景：

- 保持现有 synthesize / inference / all 代码生成流程兼容
- 将老的规则执行链路统一接入新 backend 架构

### `preprocessing.py`

作用：

- 对 backend 生成的 `DataFrame` 做统一预处理
- 这是目前 feature 对齐最关键的一层

核心函数：

```python
prepare_feature_matrices(train_df, valid_df, test_df, *, scale_features=False)
```

这个函数会做几件事：

1. 把各列尽量转成数值
2. 只根据训练集决定保留哪些列
3. 删除训练集中含 `NaN` 的列
4. 让 valid/test 用同样的列顺序 `reindex`
5. 用 `SimpleImputer(strategy="median")` 补缺失值
6. 如果需要，则再做 `StandardScaler`

返回值：

```python
X_train, X_valid, X_test, surviving_columns
```

其中 `surviving_columns` 很重要，因为它记录了进入模型的最终特征顺序，后面做 feature importance 时会直接用到。

使用方式：

```python
X_train, X_valid, X_test, surviving_columns = prepare_feature_matrices(
    train_df,
    valid_df,
    test_df,
    scale_features=True,
)
```

为什么这层重要：

- 修复了原先 train / valid / test 分别 `dropna()` 的脆弱做法
- 保证了列顺序稳定
- 为 rule importance 导出提供准确的列名映射基础

## 训练脚本里怎么用

### 在 `code_gen_and_eval.py` 里

现在典型调用方式是：

```python
feature_backend = build_feature_backend(valid_function_names)
evaluation(feature_backend)
```

如果不是 `generated_rules`，则会直接：

```python
feature_backend = build_feature_backend()
evaluation(feature_backend)
```

### 在 `eval.py` 里

使用方式与上面一致，也是先构造 backend，再调用统一的 `evaluation(feature_backend)`。

现在这两个训练入口在 `clf.fit(...)` 之后都会直接复用 `surviving_columns` / `surviving_feature_names` 去导出：

- `*_rule_importance_per_seed.csv`
- `*_rule_importance_summary.csv`

也就是说，这层不只是“为以后做 importance 打基础”，而是已经参与当前 importance 导出的列名对齐。

## 命令行使用方式

两个入口脚本都支持：

```bash
--feature_backend
```

`eval.py` 还支持：

```bash
--estimator
```

`--estimator` 当前可选值：

- `rf`，默认，保持现有 RandomForest 行为
- `linear`，分类任务使用 `LogisticRegression`，回归任务使用 `LinearRegression`

当前可选值：

- `generated_rules`
- `bbb_martins`
- `dili`
- `clintox`
- `herg`
- `pampa_ncats`
- `skin_reaction`
- `pgp_broccatelli`
- `carcinogens_lagunin`
- `ames`
- `bioavailability_ma`
- `hia_hou`
- `cyp2c9_substrate_carbonmangels`
- `cyp2d6_substrate_carbonmangels`
- `cyp3a4_substrate_carbonmangels`
- `sarscov2_3clpro_diamond`
- `sarscov2_vitro_touret`

例如直接跑 BBB 试点：

```bash
python eval.py --dataset bbbp --feature_backend bbb_martins
```

如果想用线性基线而不是 RandomForest：

```bash
python eval.py --dataset bbbp --feature_backend bbb_martins --estimator linear
```

如果要跑 TDC 的 DILI 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask DILI --feature_backend dili
```

如果要跑 TDC 的 Skin_Reaction 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask Skin_Reaction --feature_backend skin_reaction
```

如果要跑 TDC 的 Pgp_Broccatelli 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask Pgp_Broccatelli --feature_backend pgp_broccatelli
```

如果要跑 TDC 的 Carcinogens_Lagunin 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask Carcinogens_Lagunin --feature_backend carcinogens_lagunin
```

如果要跑 TDC 的 AMES 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask AMES --feature_backend ames
```

如果要跑 TDC 的 Bioavailability_Ma 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask Bioavailability_Ma --feature_backend bioavailability_ma
```

如果要跑 TDC 的 HIA_Hou 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask HIA_Hou --feature_backend hia_hou
```

如果要跑 TDC 的 CYP2C9_Substrate_CarbonMangels 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask CYP2C9_Substrate_CarbonMangels --feature_backend cyp2c9_substrate_carbonmangels
```

如果要跑 TDC 的 CYP2D6_Substrate_CarbonMangels 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask CYP2D6_Substrate_CarbonMangels --feature_backend cyp2d6_substrate_carbonmangels
```

如果要跑 TDC 的 CYP3A4_Substrate_CarbonMangels 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask CYP3A4_Substrate_CarbonMangels --feature_backend cyp3a4_substrate_carbonmangels
```

如果要跑 TDC 的 SARSCoV2_Vitro_Touret 任务，可以直接：

```bash
python eval.py --dataset TDC --subtask SARSCoV2_Vitro_Touret --feature_backend sarscov2_vitro_touret
```

如果继续跑旧的代码生成规则流程，可以保持默认：

```bash
python eval.py --dataset bbbp --feature_backend generated_rules
```

## 以后如果新增一个 backend，建议怎么做

建议步骤：

1. 在 `feature_backends/` 里新增一个文件，例如 `my_backend.py`
2. 在里面实现与现有 backend 一致的接口
3. 在 `registry.py` 里注册这个名字
4. 尽量返回 `DataFrame`
5. 保证列名稳定、顺序稳定、语义清晰

推荐至少实现这些方法：

```python
get_feature_names()
featurize_smiles(smiles)
featurize_smiles_list(smiles_list, on_error="raise")
```

如果有条件，也建议补：

```python
get_feature_descriptions()
```

## 当前扩展原则

这里的代码最好始终遵守下面几条：

- backend 只负责“提特征”，不负责训练模型
- 训练脚本不直接写死某个具体任务的特征逻辑
- 列对齐规则统一放在 `preprocessing.py`
- 做 feature importance 时要复用 `surviving_columns`

## 当前已知限制

- `bbb_martins` backend 里有 pKa 相关计算，运行可能比较慢
- `dili` backend 主要依赖 RDKit 子结构匹配和描述符，通常比 `bbb_martins` 更轻
- `herg` backend 同时包含 RDKit 结构警报和 MolGpKa 派生特征，首次运行时也可能因为 MolGpKa 初始化而变慢
- `skin_reaction` backend 也主要依赖 RDKit 子结构匹配和描述符，可直接复用 TDC 预计算缓存
- `pgp_broccatelli` backend 在需要 pKa 特征时会调用 MolGpKa，首次运行可能比较慢
- `pampa_ncats` backend 在需要 pKa/logD 特征时也会调用 MolGpKa，首次运行可能比较慢
- `carcinogens_lagunin` backend 同时包含结构警报和 pKa/logD 代理，首次运行时也可能因为 MolGpKa 初始化而变慢
- `ames` backend 主要依赖 RDKit 子结构匹配和描述符，通常可以直接复用 TDC 预计算缓存
- `bioavailability_ma` backend 同时包含理化窗口与 MolGpKa 派生特征，首次运行时也可能因为 MolGpKa 初始化而变慢
- `hia_hou` backend 会计算肠道 pH 下的中性分数和 logD 代理，首次运行时也可能因为 MolGpKa 初始化而变慢
- `sarscov2_vitro_touret` backend 同时包含 RDKit 结构警报与 MolGpKa 派生特征，首次运行时也可能因为 MolGpKa 初始化而变慢
- `generated_rules` backend 依赖外部 `exec` 过的函数命名空间
- 除了静态注册 backend 之外，registry 现在还支持按命名约定动态加载 feedback 变体 backend，例如 `dili_feedback_v001`
- feedback 变体 backend 的代码位于 `codex_generated_code_variants/`，并通过统一的 module-backed adapter 接入现有训练流程

## 给后续维护者的建议

如果你修改了以下内容，通常应该同步更新本 README：

- 新增或删除 backend
- backend 的接口变化
- 预处理逻辑变化
- 训练脚本接入方式变化
- 命令行参数变化

这样可以保证这层抽象一直“代码和说明一致”。

## Feedback 变体 Backend

如果你在做 train-error-driven feature feedback loop，不要直接修改 `codex_generated_code/` 里的原始特征代码。

应该使用：

```bash
python feature_feedback_loop.py init-variant --base_backend dili --variant_backend dili_feedback_v001 --subtask DILI
```

这样会在 `codex_generated_code_variants/` 下创建一个版本化副本 backend。

之后你可以像普通 backend 一样直接调参和评估：

```bash
python tune_tdc_rf.py --subtask DILI --feature_backend dili_feedback_v001
python eval.py --dataset TDC --subtask DILI --feature_backend dili_feedback_v001
```

这些 feedback 变体 backend 会自动拥有独立的：

- TDC 预计算特征缓存
- tuned RF 参数 JSON
- checkpoint bundle
- eval 输出文件

也就是说，它们与原始 backend 兼容，但不会污染原始 backend 的结果产物。
