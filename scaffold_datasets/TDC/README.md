# TDC Dataset Conversion

这个目录存放从：

`/data1/tianang/Projects/Intern-S1/DataPrepare/TDC_prepended/KNN_3_per_label`

转换过来的 TDC 数据。

## 数据格式

源数据是按行存储的 `.jsonl` 文件，每行至少包含：

- `drug`: 分子的 SMILES
- `Y`: 标签

转换后每个任务会放在自己的子目录里，目录名就是任务名。

目录结构示例：

```text
scaffold_datasets/TDC/
  BBB_Martins/
    BBB_Martins_train.csv
    BBB_Martins_valid.csv
  ClinTox/
    ClinTox_train.csv
    ClinTox_valid.csv
```

每个任务目录中的 CSV 文件命名格式为：

- `<task>_train.csv`
- `<task>_valid.csv`

CSV 中统一包含两列：

- `smiles`
- `<task>`

例如：

- `BBB_Martins_train.csv`
- `BBB_Martins_valid.csv`

对应列为：

- `smiles`
- `BBB_Martins`

## 转换脚本

运行：

```bash
python scaffold_datasets/TDC/convert_tdc_knn3_to_scaffold_csv.py
```

脚本会：

1. 扫描 `train/` 和 `valid/` 下的全部 `.jsonl`
2. 将每个 task 转为单独 CSV
3. 每个 task 输出到自己的子目录
4. 额外生成 `conversion_summary.csv`

## 注意

- 当前源目录只包含 `train` 和 `valid`，没有 `test`
- 这里不会伪造 `test` 数据
- 如果以后源目录新增 `test`，可以在转换脚本里把 `SPLITS` 扩展进去
