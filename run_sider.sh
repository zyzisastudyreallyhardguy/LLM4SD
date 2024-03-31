############################################################################################################
# Code for the paper: Large Language Models for Scientific Synthesis, Inference and Explanation
# Code Authors: Yizhen Zheng, Huan Yee Koh, Jiaxin Ju
# https://github.com/zyzisastudyreallyhardguy/LLM4SD/tree/main
#
# * Please install packages in requirment.txt before running this file
# 
# * To automate the entire process, we recommend utilizing GPT-4-turbo for summarizing inference rules and
# generating the corresponding code. Please ensure you have entered your API_KEY before initiating the run.
############################################################################################################

API_KEY='' # OpenAI API key for gpt4 summarization and code generation

MODEL="galactica-6.7b" # ("falcon-40b" "galactica-30b") Falcon-7B fails in sider tasks

SIDER_SUBTASK=("cardiac disorders" "neoplasms benign, malignant and unspecified (incl cysts and polyps)"
               "hepatobiliary disorders" "metabolism and nutrition disorders" "product issues" "eye disorders"
               "investigations" "musculoskeletal and connective tissue disorders" "gastrointestinal disorders"
               "social circumstances" "immune system disorders" "reproductive system and breast disorders"
               "general disorders and administration site conditions" "endocrine disorders"
               "surgical and medical procedures" "vascular disorders" "blood and lymphatic system disorders"
               "skin and subcutaneous tissue disorders" "congenital, familial and genetic disorders"
               "infections and infestations" "respiratory, thoracic and mediastinal disorders" "psychiatric disorders"
               "renal and urinary disorders" "pregnancy, puerperium and perinatal conditions"
               "ear and labyrinth disorders" "nervous system disorders"
               "injury, poisoning and procedural complications")

# Step 1: Generate prior knowledge and data knowledge prompt json file
# Please mannually add your own prompt if your task is not in the above dataset list
echo "Processin step 1: Generating Prompt files ..."
python create_prompt.py --task synthesize
python create_prompt.py --task inference

# Processing Sider Dataset
echo "Processing Sider dataset ..."
for subtask in "${SIDER_SUBTASK[@]}"; do
    #Step 2: Extract rules from prior knowledge
    echo "Processin step 2 for Sider-$subtask: LLM for Scientific Synthesize" 
    python synthesize.py --dataset sider --subtask "${subtask}" --model ${MODEL} --output_folder "synthesize_model_response"
    
    # Step 3: Knowledge inference from Data
    echo "Processin step 3 for Sider-$subtask: LLM for Scientific Inference"
    python inference.py --dataset sider --subtask "${subtask}" --model ${MODEL} --list_num 30 --output_folder "inference_model_response"
    
    # Step 4: Summarize inference rules generated from the last step
    echo "Processin step 4 for Sider-$subtask: Summarize rules from gpt4"
    python summarize_rules.py --input_model_folder ${MODEL} --dataset sider --subtask "${subtask}" --list_num 30 --api_key ${API_KEY} \
                              --output_folder "summarized_inference_rules"

    # Step 5: Interpretable model training and Evaluation
    # Results in our paper are stored in eval_result folder and the generated code files are in eval_code_generation_repo folder
    # To avoid overwriting we here provide another folder name
    # **** Must run synthesize and inference setting before run all setting ****
    echo "Processin step 5 for Sider-$subtask: Interpretable model training and Evaluation"
    python code_gen_and_eval.py --dataset sider --subtask "${subtask}" --model ${MODEL} --knowledge_type "synthesize" \
                                --api_key ${API_KEY} --output_dir "llm4sd_results" --code_gen_folder "llm4sd_code_generation"
    
    python code_gen_and_eval.py --dataset sider --subtask "${subtask}" --model ${MODEL} --knowledge_type "inference" --list_num 30 \
                                --api_key ${API_KEY} --output_dir "llm4sd_results" --code_gen_folder "llm4sd_code_generation"
    
    python code_gen_and_eval.py --dataset sider --subtask "${subtask}" --model ${MODEL} --knowledge_type "all" --list_num 30 \
                                --api_key ${API_KEY} --output_dir "llm4sd_results" --code_gen_folder "llm4sd_code_generation"
done