import numpy as np
import argparse
import math
from math import sqrt
import pandas as pd
import os
import json
import warnings

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, Fragments
from rdkit.Chem.rdMolDescriptors import CalcNumAliphaticCarbocycles, CalcNumAromaticCarbocycles
from mordred import Weight, WienerIndex, RotatableBond, EccentricConnectivityIndex
from rdkit.Chem import rdchem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem
import rdkit

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score, mean_absolute_error

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bbbp', help='dataset name in lower case')
parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider/qm9 dataset')
parser.add_argument('--model', type=str, default='galactica-6.7b', help='LLM model')
parser.add_argument('--knowledge_type', type=str, default='synthesize', help='synthesize/inference/all')
parser.add_argument('--num_samples', type=int, default=50, help='number of sample lists (30/50) for inference')
parser.add_argument('--output_dir', type=str, default='eval_result', help='output folder')
args = parser.parse_args()


# load csv datasets from a directory
def load_data(which='train'):
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


def exec_code(generated_code, smiles_list, valid_function_names):
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
    X_train = exec_code(generated_code, smiles_list, valid_function_names)

    smiles_list, y_valid = load_data('valid')
    X_valid = exec_code(generated_code, smiles_list, valid_function_names)

    smiles_list, y_test = load_data('test')
    X_test = exec_code(generated_code, smiles_list, valid_function_names)
    seeds = [0, 1, 2, 3, 4]

    if args.subtask != '':
        subtask_name = '_' + args.subtask
    else:
        subtask_name = ''

    if args.knowledge_type in ['inference', 'all']:
        subfolder = f"sample_{args.num_samples}"
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
            y_train_proba = clf.predict_proba(X_train)[:, 1]
            y_valid_proba = clf.predict_proba(X_valid)[:, 1]
            y_test_proba = clf.predict_proba(X_test)[:, 1]

            average_roc_auc_test.append(roc_auc_score(y_test, y_test_proba))
            average_roc_auc_valid.append(roc_auc_score(y_valid, y_valid_proba))
        print('=================================================')
        print(f"Dataset: {args.dataset}, Sub Task: {args.subtask}, Knowledge Type: {args.knowledge_type}, Sample_number: {args.num_samples}")
        print(f"Average test ROC-AUC: {np.mean(average_roc_auc_test)}")
        print(f"Average valid ROC-AUC: {np.mean(average_roc_auc_valid)}")

        # standard deviation
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

        print(f"Average test RMSE: {np.mean(rmse_test_list)}")
        print(f"Average valid RMSE: {np.mean(rmse_valid_list)}")

        # standard deviation
        print(f"Standard deviation of test RMSE: {np.std(rmse_test_list)}")
        print(f"Standard deviation of valid RMSE: {np.std(rmse_valid_list)}")

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
        print(f"Average test MAE: {np.mean(mae_test_list)}")
        print(f"Average valid MAE: {np.mean(mae_valid_list)}")

        # standard deviation
        print(f"Standard deviation of test MAE: {np.std(mae_test_list)}")
        print(f"Standard deviation of valid MAE: {np.std(mae_valid_list)}")
        # store the results
        file_name = f'{args.model}_{args.subtask}_{args.knowledge_type}_rules_test_mae.txt'
        file_path = os.path.join(result_folder, file_name)
        with open(file_path, 'w') as f:
            for item in mae_test_list:
                f.write("%s\n" % item)
            f.write(f"\n\nAverage test MAE: {np.mean(mae_test_list)} \n")
            f.write(f"Standard deviation of test MAE: {np.std(mae_test_list)}")


def split_string_into_parts(s, max_lines_per_part=10):
    lines = s.splitlines()
    num_parts = math.ceil(len(lines) / max_lines_per_part)
    split_points = [len(lines) * i // num_parts for i in range(num_parts + 1)]
    return ['\n'.join(lines[split_points[i]:split_points[i + 1]]) for i in range(num_parts)]


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    dataset = args.dataset
    model = args.model

    if args.subtask != '':
        if args.dataset in ['tox21', 'sider']:
            subtask_name = f'{args.dataset}_{args.subtask}'
        elif args.dataset == 'qm9' and args.num_samples == 50:
            subtask_name = args.subtask
        else:
            raise NotImplementedError(f"Folder Name error")
    else:
        subtask_name = args.dataset

    file_folder = os.path.join('eval_code_generation_repo', args.model, args.dataset)
    synthesize_folder = os.path.join(file_folder, 'synthesize')
    synthesize_file_name = f'{args.model}_{subtask_name}_pk_rules.txt'
    synthesize_file_path = os.path.join(synthesize_folder, synthesize_file_name)

    if args.knowledge_type != 'synthesize' and args.num_samples not in [30, 50]:
        raise NotImplementedError(f"num_samples should be 30 or 50")

    inference_folder = os.path.join(file_folder, 'inference', f"sample_{args.num_samples}")
    inference_file_name = f'{args.model}_{subtask_name}_dk_rules.txt'
    inference_file_path = os.path.join(inference_folder, inference_file_name)

    if args.knowledge_type == 'synthesize':
        with open(synthesize_file_path, 'r') as f:
            generated_code = f.read()
    elif args.knowledge_type == 'inference':
        with open(inference_file_path, 'r') as f:
            generated_code = f.read()
    elif args.knowledge_type == 'all':
        with open(synthesize_file_path, 'r') as f:
            synthesize_code = f.read()
        with open(inference_file_path, 'r') as f:
            inference_code = f.read()
        generated_code = synthesize_code + '\n' + inference_code  # combine synthesize_code and inference_code
    else:
        raise NotImplementedError(f"Knowledge_type is wrong.(synthesize/inference/all)")

    exec(generated_code, globals())
    function_names = [line.split()[1].split('(')[0] for line in generated_code.split('\n') if line.startswith('def ')]
    evaluation(generated_code, function_names)
    