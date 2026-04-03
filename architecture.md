# LLM4SD Codebase Architecture

## 核心流程图

```mermaid
flowchart TD
    A[("科学文献/数据集\nscaffold_datasets/")] --> B

    subgraph PROMPT["1. Prompt 构建"]
        B["create_prompt.py\n- 生成 synthesis/inference 提示词\n- 写入 prompt_file/*.json"]
    end

    subgraph RULE_GEN["2. 规则生成 (Rule Generation)"]
        C["synthesize.py\n- 调用开源LLM(Falcon/Galactica等)\n- 从文献知识提取规则\n→ synthesize_model_response/"]
        D["inference.py\n- 调用开源LLM\n- 从数据中归纳规则\n→ inference_model_response/"]
        E["summarize_rules.py\n- 调用GPT-4\n- 去重/汇总 inference rules\n→ summarized_inference_rules/"]
    end

    subgraph CODE_GEN["3. 规则→代码 转换 ⭐ 核心"]
        F["code_gen_and_eval.py\nauto_gen_code()\n\nprompt = 规则文本\ngpt-4-turbo 生成 Python 函数\n正则提取 def xxx(mol): ..."]
        G["validate_functions()\n\n- exec() 执行生成代码\n- 用 mol=Chem.MolFromSmiles('CC') 测试\n- 函数必须返回 int/float\n- 若失败: 调用 GPT-4 修复(最多3次)\n- 仍失败则丢弃该函数"]
        H["exec_code()\n\n- 对每个 SMILES 分子\n- 调用所有合法函数\n- 生成特征向量"]
    end

    subgraph EVAL["4. 模型训练与评估"]
        I["evaluation()\n\n- RandomForest 训练\n- 分类: ROC-AUC\n- 回归: RMSE / MAE\n→ eval_result/"]
    end

    subgraph STORAGE["存储"]
        R1["synthesize_model_response/\n原始合成规则(文本)"]
        R2["inference_model_response/\n原始推断规则(文本)"]
        R3["summarized_inference_rules/\n去重后推断规则"]
        R4["eval_code_generation_repo/\n生成的Python函数代码"]
        R5["eval_result/\n评估结果 JSON"]
    end

    B --> C
    B --> D
    D --> E

    C --> F
    E --> F

    F --> G
    G -->|"valid_function_names"| H
    H -->|"特征矩阵"| I

    C --> R1
    D --> R2
    E --> R3
    G --> R4
    I --> R5

    style CODE_GEN fill:#ffe0b2,stroke:#e65100,stroke-width:2px
    style F fill:#fff3e0,stroke:#ef6c00
    style G fill:#fff3e0,stroke:#ef6c00
```

---

## 规则→代码 关键函数详解

```mermaid
sequenceDiagram
    participant Rules as 规则文本
    participant Prompt as prompt 模板<br/>(code_gen_and_eval.py:40)
    participant GPT4 as GPT-4-turbo
    participant Regex as 正则提取<br/>(line 112)
    participant Exec as exec() 验证<br/>(line 172)
    participant RF as RandomForest

    Rules->>Prompt: 拼接规则文本
    Prompt->>GPT4: ChatCompletion.create()
    GPT4-->>Regex: 生成代码文本
    Regex-->>Exec: 提取 def xxx(mol): 函数

    loop 对每个函数
        Exec->>Exec: globals()[fn](mol='CC')
        alt 返回 int/float
            Exec-->>RF: 加入 valid_functions
        else 抛异常/返回None
            Exec->>GPT4: 发送错误信息修复(最多3次)
            GPT4-->>Exec: 修复后代码
        end
    end

    RF->>RF: 训练分类/回归模型
    RF-->>RF: 输出 ROC-AUC/RMSE
```

---

## 文件结构

```
LLM4SD/
├── create_prompt.py          # 生成 prompt 模板
├── synthesize.py             # 文献知识 → 规则(开源LLM)
├── inference.py              # 数据归纳 → 规则(开源LLM)
├── summarize_rules.py        # 规则去重(GPT-4)
├── code_gen_and_eval.py      # ⭐ 规则→代码 + 评估(GPT-4)
├── eval.py                   # 独立评估脚本
│
├── run_others.sh             # 主流程: bbbp/bace/clintox/...
├── run_tox21.sh              # tox21 流程
├── run_sider.sh              # sider 流程
├── run_qm9.sh                # qm9 流程
│
├── llm4sd_models.json        # RandomForest 超参数配置
├── prompt_file/
│   ├── synthesize_prompt.json
│   └── inference_prompt.json
│
├── scaffold_datasets/        # 数据集 (bbbp/bace/tox21/...)
├── synthesize_model_response/  # 合成规则原始输出
├── inference_model_response/   # 推断规则原始输出
├── summarized_inference_rules/ # 去重后规则
├── eval_code_generation_repo/  # GPT-4 生成的Python函数
└── eval_result/                # 最终评估结果
```

---

## 关键技术点

| 组件 | 技术 |
|------|------|
| 规则生成 LLM | Falcon-7b/40b, Galactica-6.7b/30b, ChemLLM-7b, ChemDFM-13B |
| 代码生成 LLM | GPT-4-turbo (`auto_gen_code`, line 94) |
| 规则去重 LLM | GPT-4 (`summarize_rules.py`) |
| 分子描述符 | RDKit, Mordred |
| 下游分类/回归 | scikit-learn RandomForest |
| 代码验证 | Python `exec()` + dummy mol 测试 |
