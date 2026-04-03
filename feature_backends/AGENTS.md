# AGENTS Notes For `feature_backends`

## Purpose

这个目录存放特征后端抽象层与共享预处理逻辑。

## Maintenance Rule

如果这里的代码发生了重大更新，请一并更新同目录下的 `README.md`，保证文档说明与当前实现保持一致。

这里所说的“重大更新”包括但不限于：

- 新增或删除 backend 文件
- 修改 `registry.py` 中支持的 backend 名称或加载方式
- 修改某个 backend 的公开接口
- 修改 `preprocessing.py` 中的列对齐、缺失值处理、标准化逻辑
- 修改训练脚本对这层的接入方式，导致 README 中的使用方法不再准确

## Update Expectation

更新 README 时，至少检查这些内容是否仍然正确：

- 每个文件的职责说明
- 每个 backend 的用途和调用方式
- `load_feature_backend(...)` 的使用方式
- `prepare_feature_matrices(...)` 的行为描述
- 命令行示例
- 新增功能或限制说明

## Goal

目标不是写很长的文档，而是保证：

- 新同事第一次看到这个目录时能快速理解结构
- 以后新增 backend 时知道应该改哪里
- 文档不会因为代码演化而过时
