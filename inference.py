import pandas as pd
import argparse
import random
import copy
from transformers import BitsAndBytesConfig
from transformers import AutoTokenizer
import transformers
import torch
import os
import json
from tqdm import tqdm


def load_dataset(dataset_name):
    df = pd.read_csv(dataset_name)
    df_str = df.iloc[:].to_string(index=False, header=False)
    df_str_list = df_str.split('\n')
    df_str_list = [f.strip() for f in df_str_list]
    return df_str_list


def get_hf_tokenizer_pipeline(model, is_8bit=False):
    """Return HF tokenizer for the model"""
    model = model.lower()
    if model == 'falcon-7b':
        hf_model = "tiiuae/falcon-7b-instruct"
    elif model == 'falcon-40b':
        hf_model = "tiiuae/falcon-40b-instruct"
        is_8bit = True
    elif model == "galactica-6.7b":
        hf_model = "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
    elif model == "galactica-30b":
        hf_model = "GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k"
        is_8bit = True
    else:
        raise NotImplementedError(f"Cannot find Hugging Face tokenizer for model {model}.")
    model_kwargs = {}
    quantization_config = None
    if is_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0,)
    model_kwargs['quantization_config'] = quantization_config
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False, padding_side="left", trust_remote_code=True,)
    pipeline = transformers.pipeline("text-generation", model=hf_model, tokenizer=tokenizer, torch_dtype=torch.float16,
                                     trust_remote_code=True, use_fast=False, device_map='auto', model_kwargs=model_kwargs,)
    return tokenizer, pipeline


def get_inference_prompt():
    prompt_file = os.path.join(args.prompt_folder, args.prompt_file)
    with open(prompt_file, 'r') as f:
        prompt_dict = json.load(f)
    print(f'Extracting {args.dataset} {args.subtask} task(s) Inference prompt ....')
    if args.dataset in list(prompt_dict.keys()) and not args.subtask:
        prompt = prompt_dict[args.dataset]
    elif args.subtask:
        prompt = prompt_dict[args.dataset][args.subtask]
    else:
        raise NotImplementedError(f"""No data knowledge prompt for task {args.dataset} {args.subtask}.""")
    return prompt


def get_token_limit(model, for_response=False):
    """Returns the token limitation of provided model"""
    model = model.lower()
    if for_response:  # For get response
        if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b"]:
            num_tokens_limit = 2048 
    else:  # For split input list
        if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b"]:
            num_tokens_limit = round(2048*3/4)  # 1/4 part for the response, 512 tokens
        else:
            raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit


def get_system_prompt():
    # prompt format depends on the LLM, you can add the system prompt here
    model = args.model.lower()
    if model in ['falcon-7b', 'falcon-40b']:
        system_prompt = ("{instruction}\n")
    elif model in ["galactica-6.7b", "galactica-30b"]:
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n### Response:\n")
    else:
        raise NotImplementedError(f"""No system prompt setting for the model: {model} .""")
    return system_prompt


def split_smile_list(smile_content_list, dk_prompt, tokenizer, list_num):
    """
    Each list can be directly fed into the model
    """
    token_limitation = get_token_limit(args.model)  # Get input token limitation for current model
    system_prompt = get_system_prompt()
    all_smile_content = dk_prompt + '\n'+'\n'.join(smile_content_list)
    formatted_all_smile = system_prompt.format_map({'instruction': all_smile_content})
    token_num_all_smile = len(tokenizer.tokenize(formatted_all_smile))
    if token_num_all_smile > token_limitation:  # Need to do split
        list_of_smile_label_lists = []
        for _ in tqdm(range(list_num)):  # Generate request number of sub lists
            current_list = []
            cp_smile_content_list = copy.deepcopy(smile_content_list)
            current_prompt = system_prompt.format_map({'instruction': dk_prompt})  # only inference prompt, without smile&label
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


def get_model_response(model, list_of_smile_label_lists, pipeline, dk_prompt, tokenizer):
    input_list = [dk_prompt +'\n' + '\n'.join(s) for s in list_of_smile_label_lists]
    system_prompt = get_system_prompt()
    response_list = []
    if model.lower() in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b"]:
        for smile_label in input_list:
            smile_prompt = system_prompt.format_map({'instruction': smile_label.strip()})
            len_smile_prompt = len(tokenizer.tokenize(smile_prompt))
            print(smile_prompt)
            max_new_token = get_token_limit(model, for_response=True) - len_smile_prompt
            text_generator = pipeline(
                smile_prompt,
                min_new_tokens=0,
                max_new_tokens=max_new_token,
                do_sample=False,
                num_beams=3,  # beam search
                temperature=float(0.5),  # randomness/diversity
                repetition_penalty=float(1.2),
                renormalize_logits=True
            )
            generated_text = text_generator[0]['generated_text']
            if model in ["galactica-6.7b", "galactica-30b"]:
                generated_text = generated_text.split('### Response:\n')[1]
            elif model in ['falcon-7b', 'falcon-40b']:
                pass
            print(generated_text)
            response_list.append(generated_text)

    else:
        raise NotImplementedError(f"""get_model_response() is not implemented for model {model}.""")
    return response_list


def main():
    if args.dataset in ["alpha", "c_v", "Delta_epsilon", "epsilon_HOMO",
                        "epsilon_LUMO", "G", "H", "mu", "R^2", "U_0", "U", "ZPVE"]:
        file_folder = os.path.join(args.input_folder, 'qm9')
    else:
        file_folder = os.path.join(args.input_folder, args.dataset)
    if args.subtask == "":
        train_file_name = args.dataset + '_train.csv'
    else:
        train_file_name = args.subtask + '_train.csv'
    train_file_path = os.path.join(file_folder, train_file_name)
    smile_label_list = load_dataset(train_file_path)

    tokenizer, pipeline = get_hf_tokenizer_pipeline(args.model)
    dk_prompt = get_inference_prompt()

    list_of_smile_label_lists = split_smile_list(smile_label_list, dk_prompt, tokenizer, args.list_num)
    print(f'Split into {len(list_of_smile_label_lists)} lists')

    if args.subtask:
        output_file_name = f"{args.model}_{args.dataset}_{args.subtask}_dk_response_sample_{args.list_num}.txt"
    else:
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
    parser.add_argument('--prompt_folder', type=str, default='prompt_file', help='Prompt file folder')
    parser.add_argument('--prompt_file', type=str, default='inference_prompt.json', help='Inference prompt json file')
    parser.add_argument('--input_folder', type=str, default='scaffold_datasets', help="load training dataset")
    parser.add_argument('--output_folder', type=str, default='inference_model_response')
    parser.add_argument('--dataset', type=str, default='bbbp', help='dataset name')
    parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider dataset')
    parser.add_argument('--list_num', type=int, default=30, help='number of lists for model inference')
    parser.add_argument('--model', type=str, default="galactica-6.7b", help='model for data knowledge')
    args = parser.parse_args()

    main()
