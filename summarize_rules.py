import time
import openai
import tiktoken
import os
import argparse


def query(message, api_key, model="gpt-4-turbo"):
    openai.api_key = api_key
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                request_timeout=180,
            )
            result = response["choices"][0]["message"]["content"].strip()
            return result
        except Exception as e:
            print(e)
            time.sleep(10)
            continue


def num_tokens_from_message(path_string, model='gpt-4-turbo'):
    """Returns the number of tokens used by a list of messages."""
    messages = [{"role": "user", "content": path_string}]
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if "gpt-3.5" in model:
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
    elif "gpt-4" in model:
        tokens_per_message = 3  # every reply is primed with <|start|>assistant<|message|>
    else:
        raise NotImplementedError(f"num_tokens_from_messages() is not implemented for model {model}.")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 3  
    return num_tokens


def get_token_limit(model='gpt-4-turbo'):
    """Returns the token limitation of provided model"""
    if model in ['gpt-4', 'gpt-4-0613']:
        num_tokens_limit = 8192 -1500 # 1500 for response
    elif model in ['gpt-3.5-turbo-16k', 'gpt-3.5-turbo-16k-0613']:
        num_tokens_limit = 16384 -1500
    elif model in ['gpt-3.5-turbo', 'gpt-3.5-turbo-0613', 'text-davinci-003', 'text-davinci-002']:
        num_tokens_limit = 4096 -1500
    elif model == "gpt-4-turbo":
        num_tokens_limit = 128000 -1500
    else:
        raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit


def split_rules_list(rule_list, token_limit, model='gpt-4-turbo'):
    """
    Split the rule list into several lists, each list can be fed into the model.
    """
    output_list = []
    current_list = []
    current_token_count = 4

    for rule in rule_list:
        rule += '\n'
        rule_token_count = num_tokens_from_message(rule, model) - 4
        if current_token_count + rule_token_count > token_limit:
            output_list.append(current_list)
            current_list = [rule]  # Start a new list.
            current_token_count = rule_token_count + 4
        else:  # The new path fits into the current list without exceeding the limit
            current_list.append(rule)  # Just add it there.
            current_token_count += rule_token_count
    # Add the last list of tokens, if it's non-empty.
    if current_list:  # The last list not exceed the limit but no more paths
        output_list.append(current_list)
    return output_list


def split_response_list(content_list, summarize_prompt, model='gpt-4-turbo'):
    token_limitation = get_token_limit(model)  # Get input token limitation for current model
    all_rules_content = '\n'.join(content_list)
    formatted_all_response = summarize_prompt.format_map({'instruction': all_rules_content.strip()})
    token_num_all_response = num_tokens_from_message(formatted_all_response, model)
    if token_num_all_response > token_limitation:
        current_len = num_tokens_from_message(summarize_prompt,model)
        token_limitation -= current_len
        list_of_response = split_rules_list(content_list, token_limitation, model)
    else:
        list_of_response = [[path.strip() + '\n' for path in content_list]]
    print('len of list_of_response', len(list_of_response))
    return list_of_response


def load_rule_file(input_file_path):
    
    with open(input_file_path, 'r') as f:
        content = f.read()
        
    rule_file_list = content.split("\n\n================================\n\n")
    print(f"Processing file {input_file_path}")
    print(f"Load {len(rule_file_list)-1} response ...")
    return rule_file_list


def main():
    if args.subtask != "":
        task_name = f"{args.dataset}_{args.subtask}"
    else:
        task_name = args.dataset
    input_file_name = f"{args.input_model_folder}_{task_name}_dk_response_sample_{args.list_num}.txt"
    input_file_path = os.path.join(args.input_folder, args.input_model_folder, args.dataset, input_file_name)
    rule_file_list = load_rule_file(input_file_path)
    summarize_prompt = (
        "please extract and summarise rules to the following format (You can exclude duplicate rules). :\n"
        "Rule X: .....\n"
        "Rule X+1: .......\n"
        "----------Please Summarise Based on the Context Below--------------"
        "\n{instruction}")
    splitted_response_list = split_response_list(rule_file_list, summarize_prompt)
    response_list = []
    for rule_list in splitted_response_list:
        rule_content = '\n'.join(rule_list)
        input_content = summarize_prompt.format_map({'instruction': rule_content.strip()})
        response = query(input_content, args.api_key, model='gpt-4-turbo')
        response_list.extend(response.split('\n'))
    
    output_folder = os.path.join(args.output_folder, args.input_model_folder, args.dataset)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    name, ext = os.path.splitext(input_file_name)
    output_file_name = name + '_summarized_rules.txt'
    output_file = os.path.join(output_folder, output_file_name)

    print(f"Writing file {output_file_name}")
    with open(output_file, 'w') as f:
        for res in response_list:
            f.write(res)
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='inference_model_response')
    parser.add_argument('--input_model_folder', type=str, default='galactica-30b')
    parser.add_argument('--dataset', type=str, default='sider')
    parser.add_argument('--subtask', type=str, default='', help='subtask for sider/tox21/qm9')
    parser.add_argument('--list_num', type=int, default=30, help='number of lists for model inference')
    parser.add_argument('--output_folder', type=str, default='summarized_inference_rules', help="summarized rules folder")
    parser.add_argument('--api_key', type=str, default="", help="Openai API Key")
    args = parser.parse_args()
    main()

