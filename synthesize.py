import argparse
import os
import json
import time
import torch
import transformers
from transformers import AutoTokenizer, BitsAndBytesConfig
from inference import get_token_limit


def get_hf_tokenizer_pipeline(model, is_8bit=False):
    """Return HF tokenizer for the model"""
    model = model.lower()
    if model == 'falcon-7b':
        hf_model = "tiiuae/falcon-7b-instruct"
    elif model == 'falcon-40b':
        hf_model = "tiiuae/falcon-40b-instruct"
    elif model == "galactica-6.7b":
        hf_model = "GeorgiaTechResearchInstitute/galactica-6.7b-evol-instruct-70k"
    elif model == "galactica-30b":
        hf_model = "GeorgiaTechResearchInstitute/galactica-30b-evol-instruct-70k"
    else:
        raise NotImplementedError(f"Cannot find Hugging Face tokenizer for model {model}.")
    model_kwargs = {}
    quantization_config = None
    if is_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=200.0,)
    model_kwargs['quantization_config'] = quantization_config
    tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=False, padding_side="left", trust_remote_code=True,)
    pipeline = transformers.pipeline("text-generation", model=hf_model, tokenizer=tokenizer, torch_dtype=torch.float16,
                                     trust_remote_code=True, use_fast=False, device_map='auto',
                                     model_kwargs=model_kwargs, )
    return tokenizer, pipeline


def get_synthesize_prompt():
    """
    Read prompt json file to load prompt for the task, return a task name list and a prompt list
    """
    prompt_file = os.path.join(args.input_folder, args.input_file)
    pk_prompt_list = []
    task_list = []
    with open(prompt_file, 'r') as f:
        prompt_dict = json.load(f)
    print(f'Extracting {args.dataset} task(s) synthesize prompt ....')
    if args.dataset in list(prompt_dict.keys()) and not args.subtask:
        task_list.append(args.dataset)
        pk_prompt_list.append(prompt_dict[args.dataset])
    elif args.subtask:
        task_list.append(args.dataset + "_" + args.subtask)
        pk_prompt_list.append(prompt_dict[args.dataset][args.subtask])
    else:
        raise NotImplementedError(f"""No synthesize prompt for task {args.dataset} {args.subtask}.""")
    return task_list, pk_prompt_list


def get_pk_model_response(model, tokenizer, model_run, pk_prompt_list):
    model = model.lower()
    # prompt format depends on the LLM, you can add the system prompt here
    if model in ["galactica-6.7b", "galactica-30b"]:
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n### Response:\n")
    elif model in ['falcon-7b', 'falcon-40b']:
        system_prompt = ("{instruction}\n")
    else:
        system_prompt = ("{instruction}\n")
    response_list = []
    if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b"]:
        for pk_prompt in pk_prompt_list:
            input_text = system_prompt.format_map({'instruction': pk_prompt.strip()})
            input_ids = tokenizer.encode(input_text, return_tensors='pt')
            input_ids = input_ids.to(model_run.device)
            len_input_text = len(tokenizer.tokenize(input_text))
            print(input_text)
            max_new_token = get_token_limit(model, for_response=True) - len_input_text
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
                pass
            print(generated_text)
            response_list.append(generated_text)
    else:
        raise NotImplementedError(f"""get_model_response() is not implemented for model {model}.""")
    return response_list


def main():
    tokenizer, pipeline = get_hf_tokenizer_pipeline(args.model)
    task_list, pk_prompt_list = get_synthesize_prompt()
    response_list = get_pk_model_response(args.model, tokenizer, pipeline, pk_prompt_list)
    output_file_folder = os.path.join(args.output_folder, args.dataset)
    if args.subtask != '':
        subtask_name = "_" + args.subtask
    else:
        subtask_name = ''
    output_file = os.path.join(output_file_folder, f'{args.model}{subtask_name}_pk_response.txt')
    if not os.path.exists(output_file_folder):
        os.makedirs(output_file_folder)
    with open(output_file, 'w') as f:
        for i in range(len(task_list)):
            f.write(f'task name: {task_list[i]}\n')
            f.write('Response from model: \n')
            f.write(response_list[i])
            f.write("\n\n================================\n\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbbp', help='dataset/task name')
    parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider dataset')
    parser.add_argument('--model', type=str, default='galactica-6.7b', help='LLM model name')
    parser.add_argument('--input_folder', type=str, default='prompt_file', help='Synthesize prompt file folder')
    parser.add_argument('--input_file', type=str, default='synthesize_prompt.json', help='synthesize prompt json file')
    parser.add_argument('--output_folder', type=str, default='synthesize_model_response', help='Synthesize output folder')

    args = parser.parse_args()

    start = time.time()
    main()
    end = time.time()
    print(f"Synthesize/Time elapsed: {end-start} seconds")