############################################################################################################
# Evaluation code for the paper: Large Language Models for Scientific Synthesis, Inference and Explanation
# Code Authors: Yizhen Zheng, HuanYee Koh, Jiaxin Ju
# https://github.com/zyzisastudyreallyhardguy/LLM4SD/tree/main
# 
# Please install packages in requirment.txt before running this file
# Direct run this script to evluate the following datasets (Evaluating QM9 dataset takes more time)
# Our results are stroed in the directory "eval_result"
# To avoid overwrite and change the output directory, please add: --output_dir new_folder_name
############################################################################################################

DATASET=("bbbp" "bace" "clintox" "esol" "freesolv" "hiv" "lipophilicity")

TOX21_SUBTASK=("nr-ar-lbd" "nr-ahr" "nr-ar" "nr-aromatase" "nr-er" "nr-er-lbd" "nr-ppar-gamma"
               "sr-are" "sr-atad5" "sr-hse" "sr-mmp" "sr-p53")
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
QM9_SUBTASK=("alpha" "c_v" "Delta_epsilon" "epsilon_HOMO" "epsilon_LUMO" "G" "H" "mu" "R^2" "U_0" "U" "ZPVE")

# Falcon-7B fails in some tasks, so we have two sets of Models
MODEL_1=("falcon-7b" "falcon-40b" "galactica-6.7b" "galactica-30b")
MODEL_2=("falcon-40b" "galactica-6.7b" "galactica-30b")

# Evaluate Datasets: BBBP, BACE, Clintox, ESOL, Freesolv, HIV, Lipophilicity
for dataset in "${DATASET[@]}"; do
  for model in "${MODEL_1[@]}"; do
      echo "Processing $dataset with model $model"
      python eval.py --dataset ${dataset} --subtask "" --model ${model} --knowledge_type synthesize
      python eval.py --dataset ${dataset} --subtask "" --model ${model} --knowledge_type inference --num_samples 30
      python eval.py --dataset ${dataset} --subtask "" --model ${model} --knowledge_type inference --num_samples 50
      python eval.py --dataset ${dataset} --subtask "" --model ${model} --knowledge_type all --num_samples 30
      python eval.py --dataset ${dataset} --subtask "" --model ${model} --knowledge_type all --num_samples 50
  done
done

# Evaluate 12 tasks in Tox21
for model in "${MODEL_2[@]}"; do
  for subtask in "${TOX21_SUBTASK[@]}"; do
      echo "Processing $subtask in tox21 with model $model"
      python eval.py --dataset tox21 --subtask ${subtask} --model ${model} --knowledge_type synthesize
      python eval.py --dataset tox21 --subtask ${subtask} --model ${model} --knowledge_type inference --num_samples 30
      python eval.py --dataset tox21 --subtask ${subtask} --model ${model} --knowledge_type inference --num_samples 50
      python eval.py --dataset tox21 --subtask ${subtask} --model ${model} --knowledge_type all --num_samples 30
      python eval.py --dataset tox21 --subtask ${subtask} --model ${model} --knowledge_type all --num_samples 50
  done
done

# Evaluate 27 tasks in Sider
for model in "${MODEL_2[@]}"; do
  for subtask in "${SIDER_SUBTASK[@]}"; do
      echo "Processing $subtask in sider with model $model"
      python eval.py --dataset sider --subtask "${subtask}" --model ${model} --knowledge_type synthesize
      python eval.py --dataset sider --subtask "${subtask}" --model ${model} --knowledge_type inference --num_samples 30
      python eval.py --dataset sider --subtask "${subtask}" --model ${model} --knowledge_type inference --num_samples 50
      python eval.py --dataset sider --subtask "${subtask}" --model ${model} --knowledge_type all --num_samples 30
      python eval.py --dataset sider --subtask "${subtask}" --model ${model} --knowledge_type all --num_samples 50
  done
done

# Evaluate 12 tasks in QM9
for model in "${MODEL_2[@]}"; do
  for subtask in "${QM9_SUBTASK[@]}"; do
      echo "Processing $subtask in qm9 with model $model"
      python eval.py --dataset qm9 --subtask ${subtask} --model ${model} --knowledge_type synthesize
      python eval.py --dataset qm9 --subtask ${subtask} --model ${model} --knowledge_type inference --num_samples 50
      python eval.py --dataset qm9 --subtask ${subtask} --model ${model} --knowledge_type all --num_samples 50
  done
done