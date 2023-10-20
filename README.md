# LLM4SD
LLM4SD is an open-source initiative that aims to leverage large language models for scientific discovery. We have **released the demo code** for a task **predicting the blood-brain barrier permeability** of molecules :satisfied:. **(Subsequent code will be released soon.)**

## Demo Description
In the demo for LLM4SD using **Galactica-6.7b** as LLM encoder to solve the blood-brain barrier permeability prediction problem.

:star2: Requirements are shown in the **requirement.txt**

:star2: **First**, to conduct "**Knowledge synthesize** from the literature", run the following command: python synthesize.py
The synthesized rules are stored under the prior_knowledge folder.

:star2: **Second**, to conduct "**Knowledge inference** from data", run the following command: python inference.py
The inferred rules are stored under the data_knowledge folder.

:star2: **Before conducting "Interpretable Model Training"**, we need to conduct code generation based on the extracted rules. There are two ways to achieve this.
1. Human experts write the code based on the generated rules.
2. Using code generation tools, e.g., GPT4, and Code Llamma and then human experts review and modify the generated code. 

In this demo, we adopt the second way. An illustrated code example is shown in the code_generation_repo folder. The code is generated based on the combination of the example synthesized rules in the prior_knowledge folder, and the example inferred rules in the data_knowledge folder. The generated code is stored under the code_generation_repo folder.

:star2: **Finally** to conduct **"Interpretable Model Training"** and obtain the model performance results. Run the following command: python eval.py --knowledge_type 'all'

Ps: To obtain an explanation, you can use the information provided by the trained interpretable model and structure a prompt to let an LLM explain the result.

## Architecture of LLM4SD


<div align="center">

<img width="843" alt="image" src="https://github.com/zyzisastudyreallyhardguy/LLM4SD/assets/75228223/cbfc09d3-ac63-4889-a15d-5471439e1e70">

</div>

## Web-based application developed based on LLM4SD (Will be release soon)

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
