from turtle import pos
import pandas as pd
import argparse
import random
from transformers import BitsAndBytesConfig
import copy
from transformers import AutoTokenizer
import transformers
import torch
import os
import json
from tqdm import tqdm

from accelerate import infer_auto_device_map, init_empty_weights

def load_dataset(dataset_name):
    df = pd.read_csv(dataset_name)
    df_str = df.iloc[:].to_string(index=False, header=False)
    df_str_list = df_str.split('\n')
    df_str_list = [f.strip() for f in df_str_list]
    return df_str_list

def get_hf_tokenizer_pipeline(model, is_8bit=False):
    """Return HF tokenizer for the model"""
    model = model.lower()
    if model == "galactica-6.7b":
        hf_model = "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
    else:
        raise NotImplementedError(f"Cannot find Hugging Face tokenizer for model {model}.")
    model_kwargs = {}
    quantization_config = None
    if is_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0,)
    model_kwargs['quantization_config'] = quantization_config
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False, padding_side="left", trust_remote_code=True,)
    model = transformers.AutoModelForCausalLM.from_pretrained(hf_model, device_map = 'cuda:0', cache_dir = '/home/yzhe0006/az20_scratch/yzhe0006/.cache', torch_dtype=torch.float16, trust_remote_code = True, **model_kwargs)
    return tokenizer, model


def get_data_knowledge_prompt(task):
    prompt_file = os.path.join(args.prompt_folder, args.prompt_file)
    with open(prompt_file, 'r') as f:
        prompt_dict = json.load(f)
    print(f'Extracting {task} task(s) data knowledge prompt ....')
    task = task.lower()
    if task in list(prompt_dict.keys()):
        prompt = prompt_dict[task]
    else:
        raise NotImplementedError(f"""No data knowledge prompt for task {args.dataset}.""")
    return prompt


def get_token_limit(model, for_response=False):
    """Returns the token limitation of provided model"""
    model = model.lower()
    if for_response:  # For get response
        if model in ['falcon-7b', 'falcon-40b',"galactica-6.7b", "galactica-30b"]:
            num_tokens_limit = 2048 
    else:  # For split input list
        if model in ['falcon-7b', 'falcon-40b',"galactica-6.7b", "galactica-30b"]:
            num_tokens_limit = round(2048*3/4)  # 1/4 part for the response, 512 tokens
        else:
            raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit


def split_smile_list(smile_content_list, dk_prompt, tokenizer,list_num):  
    """
    Each list can be directly fed into the model
    """
    token_limitation = get_token_limit(args.model)  # Get input token limitation for current model

    if args.model in ['falcon-7b', 'falcon-40b']:
        system_prompt = ("{instruction}\n")
    elif args.model in ["galactica-6.7b", "galactica-30b"]:
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n### Response:\n")
        
    all_smile_content = dk_prompt + '\n'+'\n'.join(smile_content_list)
    formatted_all_smile = system_prompt.format_map({'instruction': all_smile_content})
    token_num_all_smile = len(tokenizer.tokenize(formatted_all_smile))
    if token_num_all_smile > token_limitation:  # Need to do split
        list_of_smile_label_lists = []
        for _ in tqdm(range(list_num)):  # Generate request number of sub lists
            current_list = []
            cp_smile_content_list = copy.deepcopy(smile_content_list)
            current_prompt = system_prompt.format_map({'instruction': dk_prompt})  # only data knowledge prompt, without smile&label
            current_prompt_len = len(tokenizer.tokenize(current_prompt))
            while current_prompt_len <= token_limitation:
                if cp_smile_content_list:
                    smile_label = random.choice(cp_smile_content_list)  # Randomly select an element
                    smile_prompt = dk_prompt + '\n' + '\n'.join(current_list) + '\n' + smile_label
                    smile_input_prompt = system_prompt.format_map({'instruction': smile_prompt})
                    current_prompt_len = len(tokenizer.tokenize(smile_input_prompt))
                    if current_prompt_len > token_limitation:
                        cp_smile_content_list.remove(smile_label)  # Maybe this smile string is too long, remove it and try to add another shorter one
                    else:
                        current_list.append(smile_label)
                        cp_smile_content_list.remove(smile_label)  # no duplicated smile string in one sub-list
                else:
                    break

            list_of_smile_label_lists.append(current_list)
    else:
        list_of_smile_label_lists = [[sc + '\n' for sc in smile_content_list]]
    return list_of_smile_label_lists


def get_model_response(model, list_of_smile_label_lists, model_run, dk_prompt,tokenizer):
    input_list = [dk_prompt +'\n' + '\n'.join(s) for s in list_of_smile_label_lists]
    model = model.lower()
    if model in ["galactica-6.7b", "galactica-30b"]:
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n### Response:\n")
    elif model in ['falcon-7b', 'falcon-40b']:
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n### Response:\n")
    response_list = []
    if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b"]:
        for smile_label in input_list:
            smile_prompt = system_prompt.format_map({'instruction': smile_label.strip()})
            len_smile_prompt = len(tokenizer.tokenize(smile_prompt))
            input_ids = tokenizer.encode(smile_prompt, return_tensors='pt')
            input_ids = input_ids.to(model_run.device)
            print(smile_prompt)
            max_new_token = get_token_limit(model, for_response=True) - len_smile_prompt
            max_new_token = max_new_token - 100
            output = model_run.generate(
                input_ids,
                min_new_tokens=0,
                max_new_tokens=max_new_token,
                min_length=0,    # or any other value
                do_sample=False,
                num_beams=1,
                temperature=0.2,
                repetition_penalty=1.07,
                num_return_sequences=1,
                top_p=0.75,
                top_k=40,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
            generated_text = decoded_output
            if model in ["galactica-6.7b", "galactica-30b"]:
                generated_text = generated_text.split('### Response:\n')[1]
            elif model in ['falcon-7b', 'falcon-40b']:
                generated_text = generated_text.split('### Response:\n')[1]
            print(generated_text)
            response_list.append(generated_text)
                
    else:
        raise NotImplementedError(f"""get_model_response() is not implemented for model {model}.""")
    return response_list


def main():
    file_folder = os.path.join(args.input_folder, args.dataset)
    train_file_name = args.dataset + '_train.csv'
    train_file_path = os.path.join(file_folder, train_file_name)
    smile_label_list = load_dataset(train_file_path)

    tokenizer, pipeline = get_hf_tokenizer_pipeline(args.model)
    dk_prompt = get_data_knowledge_prompt(args.dataset)

    list_of_smile_label_lists = split_smile_list(smile_label_list, dk_prompt, tokenizer,args.list_num)
    print(f'Split into {len(list_of_smile_label_lists)} lists')
    

    output_file_name = f"{args.model}_{args.dataset}_dk_response_sample_{args.list_num}.txt"
    output_file_folder = os.path.join(args.output_folder, args.model, args.dataset)
    if not os.path.exists(output_file_folder):
        os.makedirs(output_file_folder)
    output_file = os.path.join(output_file_folder, output_file_name)
    print(f'Start getting response from model {args.model}....')
    response_list = get_model_response(args.model, list_of_smile_label_lists, pipeline, dk_prompt, tokenizer)
    with open(output_file, 'w') as f:
        for response in response_list:
            f.write(response)
            f.write("\n\n================================\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_folder', type=str, default='prompt_file', help='data knowledge prompt file folder')
    parser.add_argument('--prompt_file', type=str,default='data_knowledge_prompt.json', help='prior knowledge prompt json file')
    parser.add_argument('--input_folder', type=str, default='scaffold_datasets', help="load training dataset")
    parser.add_argument('--output_folder', type=str, default='data_knowledge', help='data knowledge output folder')
    parser.add_argument('--dataset', type=str, default='bbbp', help='dataset name', choices=['bbbp', 'tox21', 'sider', 'clintox', 'hiv', 'bace', 'esol', 'freesolv', 'lipophilicity'])
    parser.add_argument('--list_num', type=int,default=6, help='number of lists for model inference')
    parser.add_argument('--model', type=str, default="galactica-6.7b", help='model for data knowledge', choices=['falcon-7b', 'falcon-40b',"galactica-6.7b", "galactica-30b"])
    args = parser.parse_args()

    main()
