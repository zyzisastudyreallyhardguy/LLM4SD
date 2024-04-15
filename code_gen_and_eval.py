import openai
import time
import os
import json
import re
import multiprocessing
import numpy as np
import argparse
import math
from math import sqrt
import pandas as pd

import rdkit
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, Fragments
from rdkit.Chem.rdMolDescriptors import CalcNumAliphaticCarbocycles, CalcNumAromaticCarbocycles
from mordred import Weight, WienerIndex, RotatableBond, EccentricConnectivityIndex
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem


from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bbbp', help='dataset/task name')
parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider/qm9 dataset')
parser.add_argument('--model', type=str, default='galactica-6.7b', help='LLM model')
parser.add_argument('--knowledge_type', type=str, default='all', help='synthesize/inference/all')
parser.add_argument('--list_num', type=int, default=30, help='number of sample lists (30/50) for inference')
parser.add_argument('--output_dir', type=str, default='eval_result', help='output folder')
parser.add_argument('--code_gen_folder', type=str, default='eval_code_generation_repo')
parser.add_argument('--api_key', type=str, default='', help='openai key')
args = parser.parse_args()

prompt = '''Question: Please generate the following rules to code like the following examples. You can define function name by yourself. Please ensure the code is executable! Each rule can have multiple functions! Don't make mistakes like ''rdkit.Chem.rdMolDescriptors' has no attribute 'CalXXX'. Don't skip rules! All functions have to return numbers and takes one argument mol!:
# Rule 1: Molecule should have a minimum of two hydrogen bond donors
def rule55302_hb_donors1232143(mol):
    return rdMolDescriptors.CalcNumHBD(mol)

# Rule 2: The molecule should have at least three hydrogen bond acceptors
def rule950021_hb_acceptor35749(mol):
    return rdMolDescriptors.CalcNumHBA(mol)
---------------------'''

fix_prompt = """Please regenerate and ensure all the code are executable and correct!"""


# load csv datasets from a directory
def load_data(which='train'):
    if args.dataset in ["alpha", "c_v", "Delta_epsilon", "epsilon_HOMO",
                        "epsilon_LUMO", "G", "H", "mu", "R^2", "U_0", "U", "ZPVE"]:
        dataset_folder = os.path.join('scaffold_datasets', 'qm9')
    else:
        dataset_folder = os.path.join('scaffold_datasets', args.dataset)
    if args.subtask == "":
        file_name = args.dataset + '_' + which + '.csv'
    else:
        file_name = args.subtask + '_' + which + '.csv'
    file_path = os.path.join(dataset_folder, file_name)
    df = pd.read_csv(file_path)

    if args.dataset == 'bbbp':
        y = df['p_np'].tolist()
    elif args.dataset == 'clintox':
        y = df['CT_TOX'].tolist()
    elif args.dataset == 'hiv':
        y = df['HIV_active'].tolist()
    elif args.dataset == 'bace':
        y = df['Class'].tolist()
    elif args.dataset == 'lipophilicity':
        y = df['exp'].tolist()
    elif args.dataset == 'esol':
        y = df['ESOL predicted log solubility in mols per litre'].tolist()
    elif args.dataset == 'freesolv':
        y = df['calc'].tolist()
    elif args.dataset in ['tox21', 'sider', 'qm9']:
        y = df[args.subtask].tolist()
    else:
        raise NotImplementedError(f"Load Dataset Error")

    if args.dataset != 'bace':
        smiles_list = df['smiles'].tolist()
    else:
        smiles_list = df['mol'].tolist()
    return smiles_list, y


# automatically generate code based on prompt
def auto_gen_code(prompt):
    openai.api_key = args.api_key
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1,
            )
            # If no exception is raised, the request was successful, so we break out of the loop
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            time.sleep(2)
    
    generated_code = response.choices[0].message.content

    pattern = r"(def [a-zA-Z_][a-zA-Z0-9_]*\(.+\):(?:\n(?:    .+\n)+)+)"
    matches = re.findall(pattern, generated_code, re.MULTILINE)

    generated_code = '\n'.join(matches)

    return generated_code


def worker(content):
    temp_prompt = prompt + content
    return auto_gen_code(temp_prompt)


def get_function_code(generated_code, function_name):
    lines = generated_code.split('\n')
    start = -1
    for i in range(len(lines)):
        if lines[i].strip().startswith('def ' + function_name):
            start = i
            break
    if start == -1:
        return None  # Function not found
    end = start + 1
    while end < len(lines) and lines[end].startswith(' '):  # Continue until we reach a line not indented
        end += 1
    return '\n'.join(lines[start:end])


def update_temp_py_file(code_repo_dir, old_function_code, rectified_code):
    # read the original file
    with open(code_repo_dir, 'r') as f:
        file_content = f.read()

    # replace the old function definition with the new one
    file_content = file_content.replace(old_function_code, rectified_code)

    # write the content back to the file
    with open(code_repo_dir, 'w') as f:
        f.write(file_content)


def get_code_repo():
    if args.subtask != '':
        subtask_name = '_' + args.subtask
    else:
        subtask_name = ''
    if args.knowledge_type == 'inference':
        subfolder = f"sample_{args.list_num}"
    else:
        subfolder = ""
    output_folder = os.path.join(args.code_gen_folder, args.model, args.dataset, args.knowledge_type, subfolder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    file_name = f"{args.model}_{args.dataset}{subtask_name}_{args.knowledge_type}_rules.txt"
    code_repo_dir = os.path.join(output_folder, file_name)
    return code_repo_dir


def validate_functions(generated_code):
    exec(generated_code, globals())
    # Create a file using generated_code
    code_repo_dir = get_code_repo()
    with open(code_repo_dir, 'w') as f:
        f.write(generated_code)

    # Get a list of all the function names in the generated code
    function_names = [line.split()[1].split('(')[0] for line in generated_code.split('\n') if line.startswith('def ')]
    valid_function_names = function_names.copy()  # Make a copy of the function names

    MAX_RETRIES = 3
    # Create a dummy molecule for testing functions
    mol = Chem.MolFromSmiles('CC')
    skipped_indices = []  # A list to record the indices of the skipped functions
    features = []

    for j, function_name in enumerate(function_names):
        solve_flag = False
        try:
            feature = globals()[function_name](mol)
            if feature is None or not isinstance(feature, (int, float)):
                valid_function_names.remove(function_name)
            else:
                solve_flag = True
        except Exception as e:
            error_message = str(e)
            for retry in range(MAX_RETRIES):
                print(f"Error in function {function_name}: {error_message}")
                print("Attempting to rectify the code...")
                if retry == 0:
                    function_code = get_function_code(generated_code, function_name)
                    old_function_code = function_code
                rectify_prompt = ("The function is incorrect. Please rectify the code! Here is the function code and its corresponding error! Please keep the function name unchanged! The function cannot raise errors and can only return number!")
                rectify_prompt += f"\n\n### Function {function_name}:\n{function_code}\n\n### Error:\n{error_message}\n\n### Rectified Code:"
                rectified_code = auto_gen_code(rectify_prompt)

                # Update function name based on rectified_code
                try:  # Try executing the rectified function
                    exec(rectified_code, globals())  # Replace the erroneous function with the rectified one
                    # get the new function name in rectified_code
                    function_name = [line.split()[1].split('(')[0] for line in rectified_code.split('\n') if line.startswith('def ')][0]
                    feature = globals()[function_name](mol)
                    if feature is None:
                        function_code = rectified_code
                        continue
                    if feature is not None and isinstance(feature, (int, float)):
                        features.append(feature)
                        solve_flag = True
                except Exception as e:
                    print(f"Error in rectified function {function_name} after {retry + 1} retries: {str(e)}")
                    function_code = rectified_code
                    error_message = str(e)
                    solve_flag = False
                if solve_flag:
                    break
            if j not in skipped_indices and not solve_flag:
                skipped_indices.append(j)
                print(f"Skipping function {function_name} after {MAX_RETRIES} retries.")
            if solve_flag:
                update_temp_py_file(code_repo_dir, old_function_code, rectified_code)
            else:
                update_temp_py_file(code_repo_dir, old_function_code, '')
            if not solve_flag:
                valid_function_names.remove(function_name)  # Remove the function from the list
                print(f"Discarding function {function_name} after {MAX_RETRIES} retries.")

    return valid_function_names, skipped_indices


def exec_code(smiles_list, valid_function_names):
    smiles_feat = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        features = []
        for function_name in valid_function_names:  # Loop over the valid function names
            try:
                feature = globals()[function_name](mol)
                if feature is not None and isinstance(feature, (int, float)):
                    features.append(feature)
                else:
                    features.append(np.nan)
            except Exception as e:
                print(f"Unexpected error in function {function_name}: {str(e)}")
                features.append(np.nan)
        smiles_feat.append(features)

    return smiles_feat


def dropna(X):
    X = pd.DataFrame(X)
    X = X.dropna(axis=1)
    X = X.values.tolist()
    return X


def evaluation(generated_code, valid_function_names):
    smiles_list, y_train = load_data('train')
    X_train = exec_code(smiles_list, valid_function_names)

    smiles_list, y_valid = load_data('valid')
    X_valid = exec_code(smiles_list, valid_function_names)

    smiles_list, y_test = load_data('test')
    X_test = exec_code(smiles_list, valid_function_names)
    seeds = [0, 1, 2, 3, 4]

    if args.subtask != '':
        subtask_name = '_' + args.subtask
    else:
        subtask_name = ''

    if args.knowledge_type in ['inference', 'all']:
        subfolder = f"sample_{args.list_num}"
    else:
        subfolder = ''
    result_folder = os.path.join(args.output_dir, args.model, args.dataset, args.knowledge_type, subfolder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    X_train = dropna(X_train)
    X_valid = dropna(X_valid)
    X_test = dropna(X_test)

    if args.dataset in ['bbbp', 'clintox', 'hiv', 'bace', 'tox21', 'sider']:

        average_roc_auc_test = []
        average_roc_auc_valid = []

        # Normalize scaler
        scaler = StandardScaler()
        X_valid = [sublist[:len(X_train[0])] for sublist in X_valid]
        X_test = [sublist[:len(X_train[0])] for sublist in X_test]

        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        for seed in seeds:
            # random forest
            with open('llm4sd_models.json', 'r') as model_parms:
                best_models = json.load(model_parms)
            if args.dataset in ['sider', 'tox21']:
                best_params = best_models[args.dataset][args.subtask]
            else:
                best_params = best_models[args.dataset]

            best_params['random_state'] = seed
            clf = RandomForestClassifier(**best_params)
            clf.fit(X_train, y_train)
            y_valid_proba = clf.predict_proba(X_valid)[:, 1]
            y_test_proba = clf.predict_proba(X_test)[:, 1]
            average_roc_auc_test.append(roc_auc_score(y_test, y_test_proba))
            average_roc_auc_valid.append(roc_auc_score(y_valid, y_valid_proba))

        print('=================================================')
        print(f"Average test ROC-AUC: {np.mean(average_roc_auc_test)}")
        print(f"Average valid ROC-AUC: {np.mean(average_roc_auc_valid)}")
        print(f"Standard deviation of test ROC-AUC: {np.std(average_roc_auc_test)}")
        print(f"Standard deviation of valid ROC-AUC: {np.std(average_roc_auc_valid)}")
        print('===================================================')

        # store the results
        file_name = f'{args.model}_{args.dataset}{subtask_name}_{args.knowledge_type}_rules_test_roc_auc.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in average_roc_auc_test:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test ROC-AUC: {np.mean(average_roc_auc_test)} \n")
            f.write(f"Standard deviation of test ROC-AUC: {np.std(average_roc_auc_test)}")
    elif args.dataset in ['esol', 'lipophilicity', 'freesolv']:
        rmse_test_list = []
        rmse_valid_list = []
        for seed in seeds:
            with open('llm4sd_models.json', 'r') as model_parms:
                best_models = json.load(model_parms)
            best_params = best_models[args.dataset]
            best_params['random_state'] = seed
            clf = RandomForestRegressor(**best_params)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            y_test_pred = clf.predict(X_test)
            # Compute the RMSE
            rmse_test = sqrt(mean_squared_error(y_test, y_test_pred))
            rmse_valid = sqrt(mean_squared_error(y_valid, y_valid_pred))
            rmse_test_list.append(rmse_test)
            rmse_valid_list.append(rmse_valid)

        print('=================================================')
        print(f"Average test RMSE: {np.mean(rmse_test_list)}")
        print(f"Average valid RMSE: {np.mean(rmse_valid_list)}")
        print(f"Standard deviation of test RMSE: {np.std(rmse_test_list)}")
        print(f"Standard deviation of valid RMSE: {np.std(rmse_valid_list)}")
        print('=================================================')
        # store the results
        file_name = f'{args.model}_{args.dataset}_{args.knowledge_type}_rules_test_rmse.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in rmse_test_list:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test RMSE: {np.mean(rmse_test_list)} \n")
            f.write(f"Standard deviation of test RMSE: {np.std(rmse_test_list)}")
    elif args.dataset == 'qm9':
        mae_test_list = []
        mae_valid_list = []
        scaler = StandardScaler()

        np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        np.nan_to_num(X_valid, nan=0, posinf=0, neginf=0)
        np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        for seed in seeds:
            # Normalize scaler
            clf = RandomForestRegressor(random_state=seed)
            clf.fit(X_train, y_train)
            y_valid_pred = clf.predict(X_valid)
            y_test_pred = clf.predict(X_test)

            # Compute the MAE
            mae_test = mean_absolute_error(y_test, y_test_pred)
            mae_valid = mean_absolute_error(y_valid, y_valid_pred)
            mae_test_list.append(mae_test)
            mae_valid_list.append(mae_valid)
        print('=================================================')
        print(f"Average test MAE: {np.mean(mae_test_list)}")
        print(f"Average valid MAE: {np.mean(mae_valid_list)}")
        print(f"Standard deviation of test MAE: {np.std(mae_test_list)}")
        print(f"Standard deviation of valid MAE: {np.std(mae_valid_list)}")
        print('=================================================')
        # store the results
        file_name = f'{args.model}_{args.subtask}_{args.knowledge_type}_rules_test_mae.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in mae_test_list:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test MAE: {np.mean(mae_test_list)} \n")
            f.write(f"Standard deviation of test MAE: {np.std(mae_test_list)}")
    else:
        raise NotImplementedError(f"Dataset Name Error: {args.dataset}.")


def reward_calculation(contents):
    max_tries = 3
    for i in range(max_tries):
        solve_flag = False
        try:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            generated_code_list = pool.map(worker, contents)
            
            # obtain the first element of each tuple
            generated_code_list_result = [x for x in generated_code_list]
            generated_code = '\n'.join(generated_code_list_result)

            valid_function_names, skipped_indices = validate_functions(generated_code)
            solve_flag = True
            break
        except Exception as e:
            print(f"Error in reward_calculation(): {str(e)}")
            # retry the whole process
            continue
    
    if not solve_flag:
        print(f"Failed to generate code after {max_tries} retries.")
        return None

    # prine number of function names
    print('number of generated functions: ', len(valid_function_names))
    code_repo_dir = get_code_repo()
    with open(code_repo_dir, 'r') as f:
        generated_code = f.read()
    evaluation(generated_code, valid_function_names)
    return generated_code


def split_string_into_parts(s, max_lines_per_part=10):
    lines = s.splitlines()
    num_parts = math.ceil(len(lines) / max_lines_per_part)
    split_points = [len(lines) * i // num_parts for i in range(num_parts + 1)]
    return ['\n'.join(lines[split_points[i]:split_points[i + 1]]) for i in range(num_parts)]


def get_synthesize_file_path():
    dataset = args.dataset
    if args.dataset in ['sider', 'tox21']:
        subtask_name = "_" + args.subtask
    elif args.dataset == 'qm9':
        subtask_name = ""
        dataset = args.subtask
    else:
        subtask_name = ""
    synthesize_file_folder = os.path.join('synthesize_model_response', args.model, dataset)
    synthesize_file_name = f'{args.model}{subtask_name}_pk_response.txt'
    synthesize_file_path = os.path.join(synthesize_file_folder, synthesize_file_name)
    return synthesize_file_path


def get_inference_file_path():
    if args.subtask != '':
        subtask_name = args.subtask + '_'
    else:
        subtask_name = ""
    inference_file_folder = os.path.join('summarized_inference_rules', args.model, args.dataset)
    inference_file_name = f"{args.model}_{args.dataset}_{subtask_name}dk_response_sample_{args.list_num}_summarized_rules.txt"
    inference_file_path = os.path.join(inference_file_folder, inference_file_name)
    return inference_file_path


def get_synthesize_inference_code():
    if args.subtask != '':
        subtask_name = '_' + args.subtask
    else:
        subtask_name = ''
    synthesize_folder = os.path.join(args.code_gen_folder, args.model, args.dataset, "synthesize")
    inference_folder = os.path.join(args.code_gen_folder, args.model, args.dataset, "inference", f"sample_{args.list_num}")
    synthesize_filename = os.path.join(synthesize_folder, f"{args.model}_{args.dataset}{subtask_name}_synthesize_rules.txt")
    inference_filename = os.path.join(inference_folder, f"{args.model}_{args.dataset}{subtask_name}_inference_rules.txt")

    if not os.path.exists(synthesize_filename) or not os.path.exists(inference_filename):
        raise NotImplementedError(f"Please run synthesize code and inference code first")

    with open(synthesize_filename, 'r') as f:
        synthesize_code = f.read()

    with open(inference_filename, 'r') as f:
        inference_code = f.read()
    all_generated_code = synthesize_code + '\n' + inference_code
    return all_generated_code


if __name__ == '__main__':
    if args.knowledge_type == 'synthesize':
        syn_file_path = get_synthesize_file_path()
        with open(syn_file_path, 'r') as f:
            content = f.read()
        contents = split_string_into_parts(content)
        reward_calculation(contents)
    elif args.knowledge_type == 'inference':
        infer_file_path = get_inference_file_path()
        with open(infer_file_path, 'r') as f:
            content = f.read()
        contents = split_string_into_parts(content)
        reward_calculation(contents)
    elif args.knowledge_type == 'all':
        generated_code = get_synthesize_inference_code()
        exec(generated_code, globals())
        function_names = [line.split()[1].split('(')[0] for line in generated_code.split('\n') if line.startswith('def ')]
        evaluation(generated_code, function_names)
    else:
        raise NotImplementedError(f"knowledge_type error")
