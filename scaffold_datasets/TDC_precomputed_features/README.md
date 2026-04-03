# TDC Precomputed Features

这个目录用于缓存 `eval.py` 为 TDC 数据集预计算的分子特征。

当前主要服务于：

- `--dataset TDC`
- `--subtask BBB_Martins`
- `--feature_backend bbb_martins`

## 作用

`bbb_martins` 特征提取包含较重的分子描述符与 pKa 相关计算。

为了避免每次训练前都重复从头计算，`eval.py` 现在会：

1. 在开始训练前检查缓存文件是否存在
2. 如果不存在，则并行计算并保存
3. 如果存在，则直接读取缓存

## 目录结构

缓存会按 task 单独存放，例如：

```text
scaffold_datasets/TDC_precomputed_features/
  BBB_Martins/
    BBB_Martins_train_bbb_martins_features.csv
    BBB_Martins_valid_bbb_martins_features.csv
```

## 说明

- 当前 TDC 源数据只有 `train` 和 `valid`，没有独立 `test`
- 因此 `eval.py` 在 TDC 模式下会对 `test` 回退使用 `valid`
- 如果需要强制重算缓存，可以在 `eval.py` 中传：

```bash
--force_recompute_features
```
