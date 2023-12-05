# Large Language Model for Scientific Discovery (LLM4SD)
LLM4SD is an open-source initiative that aims to leverage large language models for scientific discovery. We have now **released the complete code** :satisfied:.

## Code Description

### QuickStart:
:star2: **First**, requirements are shown in the **requirements.txt**. Please use the requirements.txt to create the environment for running LLM4SD.

:star2: **Second**, please **put your Openai API key** in the **bash file** before you run the bash file. The Openai API will be used to call GPT-4 to conduct text summarisation for knowledge inference information and automatic code generation.

To run tasks for ["bbbp" "bace" "clintox" "esol" "freesolv" "hiv" "lipophilicity"]. Please run:
```
bash run_others.sh
```

To run tasks for "Tox21" and "Sider". Please run:
```
bash run_tox21.sh
```

```
bash run_sider.sh
```

To run tasks for "Qm9". Please run:
```
bash run_qm9.sh
```

### The Process of LLM4SD Code Pipeline:
In the bash file, the LLM4SD is conducted in the following process:


👉: "**Knowledge synthesize** from the literature", this step will call python synthesize.py
The synthesized rules are stored under the prior_knowledge folder.

👉: "**Knowledge inference** from data", this step will call python inference.py
The inferred rules are stored under the data_knowledge folder.

👉: "**Inferred Knowledge Summarization**", this step will call python summarize_rules.py
The summarized rules are stored under the summarized_inference_rules folder. --> The purpose of this step is to drop duplicate rules.

👉: "**Automatic Code Generation & Evaluation**", this step will call python auto_gen_and_eval.py
This step will automatically generate the code using GPT-4 and run experiments to get the model performance. Please note that, in practice, human experts would review the code before usage. However, even with automatic code generation and direct evaluation, the code achieves pretty much the same performance.

**📓Notes:** We have also provided **an advanced automatic code generation tool** based on the newly released **OpenAI Assistant**. If you are interested in trying the assistant version of code generation, please check out the "code_gen.py" and "eval.py" files in the folder "LLM4SD-gpt4-demo".

PS: To obtain an explanation, you can use the information provided by the trained interpretable model and structure a prompt to let an LLM explain the result as shown in the paper.

### Direct Evaluation:
A direct evaluation of the generated code of a specific task. You can run:
```
python eval.py --dataset ${dataset} --subtask "{subtask_name}" --model ${model_name} --knowledge_type ${knowledge_type} [if evaluating inference code or combined code specify --num_samples ${number of responses during inference}]
```

A direct evaluation of all generated code in all tasks. You can run:
```
python eval_code.sh
```

## Architecture of LLM4SD


<div align="center">

<img width="843" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/cbfc09d3-ac63-4889-a15d-5471439e1e70">

</div>

## Web-based application developed based on LLM4SD (Will be released soon)

### **Comments are welcome to help us improve the web-based application:exclamation::exclamation::exclamation:**

#### 1.Knowledge Synthesis (Derive Knowledge from Scientific Literature) ##
<div align="center">
<img width="843" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/4c187835-72ea-477e-be57-0030f41a552f">
<img width="843" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/04e29487-6fdc-4a08-b064-e159f3e76c60">
</div>

#### 2.Knowledge Inference (Derive Knowledge from Analyzing Scientific Data) ##
<div align="center">
<img width="843" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/46d15716-c294-4c60-8cc1-47235f1a32d2">
<img width="843" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/f404d5e5-6f34-4c7f-8f53-8ad9274f7cd0">
</div>

#### 3.Prediction with Explanation (Explaining how the Prediction is derived) ##
<div align="center">
<img width="843" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/5bc95560-b53a-44e4-b478-3ad1b901f61b">
<img width="600" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/707c6585-592a-4e1c-9fd6-22e73ca313a9">
</div>
