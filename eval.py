import openai
import multiprocessing
import numpy as np
import re
import time
import argparse
import math
from math import sqrt
import pandas as pd


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
# cross validation
from sklearn.model_selection import cross_val_score
# calculate the AUC-ROC
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='bbbp', help='dataset/task name')
parser.add_argument('--model', type=str, default='galactica-6.7b', help='model for prior knowledge')
parser.add_argument('--knowledge_type', type=str, default='test', help='synthesize/inference/all')
parser.add_argument('--num_samples', type=int, default=1, help='number of samples to generate')
parser.add_argument('--output_dir', type=str, default='/home/yzhe0006/az20_scratch/yzhe0006/LLM4SD_demo/', help='output folder')
args = parser.parse_args()

# load csv datasets from a directory
def load_data(dataset, which = 'train'):
    df = pd.read_csv(args.output_dir + 'scaffold_datasets/' + dataset + '/' +  dataset + '_' + which + '.csv')
    if args.dataset == 'bbbp':
        y = df['p_np'].tolist()
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
    dataset = args.dataset
    smiles_list, y_train = load_data(dataset, 'train')
    X_train = exec_code(generated_code, smiles_list, valid_function_names)
    # check whether the length of each element in X_train is the same
    for i in range(len(X_train)):
        if len(X_train[i]) != len(X_train[0]):
            print('length of each element in X_train is not the same')

    smiles_list, y_valid = load_data(dataset, 'valid')
    X_valid = exec_code(generated_code, smiles_list, valid_function_names)

    smiles_list, y_test = load_data(dataset, 'test')
    X_test = exec_code(generated_code, smiles_list, valid_function_names)
    seeds = [0, 1, 2, 3, 4]

    if args.dataset in ['bbbp', 'clintox', 'hiv', 'bace']:

        average_roc_auc_test = []
        average_roc_auc_valid = []

        X_train = dropna(X_train)
        X_valid = dropna(X_valid)
        X_test = dropna(X_test)

        # Normalize scaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_valid = scaler.transform(X_valid)
        X_test = scaler.transform(X_test)

        for seed in seeds:
            # random forest
            if dataset == 'bbbp':
                clf = RandomForestClassifier(n_estimators=500, max_depth=None, random_state=seed, 
                                            min_samples_split=20, bootstrap=True, class_weight='balanced', criterion='entropy', min_samples_leaf=4)
            clf.fit(X_train, y_train)

            y_train_proba = clf.predict_proba(X_train)[:, 1]
            y_train_pred = clf.predict(X_train)

            y_valid_proba = clf.predict_proba(X_valid)[:, 1]
            y_valid_pred = clf.predict(X_valid)

            y_test_proba = clf.predict_proba(X_test)[:, 1]
            y_test_pred = clf.predict(X_test)

            print(f"Training ROC-AUC: {roc_auc_score(y_train, y_train_proba)}")
            print(f"Validation ROC-AUC: {roc_auc_score(y_valid, y_valid_proba)}")
            print(f"Test ROC-AUC: {roc_auc_score(y_test, y_test_proba)}")
        
            average_roc_auc_test.append(roc_auc_score(y_test, y_test_proba))
            average_roc_auc_valid.append(roc_auc_score(y_valid, y_valid_proba))
        
        print(f"Average test ROC-AUC: {np.mean(average_roc_auc_test)}")
        print(f"Average valid ROC-AUC: {np.mean(average_roc_auc_valid)}")

        # standard deviation
        print(f"Standard deviation of test ROC-AUC: {np.std(average_roc_auc_test)}")
        print(f"Standard deviation of valid ROC-AUC: {np.std(average_roc_auc_valid)}")

        # scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc')

        #store the results 
        with open(args.output_dir + 'result/' + args.model + '_' + args.dataset + '_' + args.knowledge_type + '_' + str(args.num_samples) + '_rules' + '_test_roc_auc.txt', 'w') as f:
            # for item in average_roc_auc_test:
            #     f.write("%s\n" % item)
            f.write("%s\n" % np.mean(average_roc_auc_test))
            f.write("%s\n" % np.std(average_roc_auc_test))
    elif args.dataset in ['esol', 'lipophilicity', 'freesolv']:
        rmse_test_list = []
        rmse_valid_list = []
        for seed in seeds:
            if args.dataset == 'esol':
                clf = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=seed, criterion='absolute_error', min_samples_leaf=1, bootstrap = True, n_jobs=-1)
            elif args.dataset == 'lipophilicity':
                clf = RandomForestRegressor(n_estimators=600, max_depth=None, random_state=seed, criterion='squared_error', min_samples_leaf=1, bootstrap = True, n_jobs=-1)
            elif args.dataset == 'freesolv':
                clf = RandomForestRegressor(n_estimators=500, max_depth=None, random_state=seed, criterion='squared_error', min_samples_leaf=1, bootstrap = False, n_jobs=-1, min_samples_split=2)

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

        #store the results
        with open(args.output_dir + 'result/' + args.model + '_' + args.dataset + '_' + args.knowledge_type + '_' + str(args.num_samples) + '_rules' + '_test_rmse.txt', 'w') as f:
            # for item in rmse_test_list:
            #     f.write("%s\n" % item)
            f.write("%s\n" % np.mean(rmse_test_list))
            f.write("%s\n" % np.std(rmse_test_list))

def split_string_into_parts(s, max_lines_per_part=10):
    lines = s.splitlines()
    num_parts = math.ceil(len(lines) / max_lines_per_part)
    split_points = [len(lines) * i // num_parts for i in range(num_parts + 1)]
    return ['\n'.join(lines[split_points[i]:split_points[i + 1]]) for i in range(num_parts)]

if __name__ == '__main__':
    dataset = args.dataset
    model = args.model
    num_sample = str(args.num_samples)
    if args.knowledge_type == 'synthesize':
        with open(args.output_dir + 'code_generation_repo/' + args.model + '_' + args.dataset + '_synthesize' + '.txt', 'r') as f:
            generated_code = f.read()
        
        exec(generated_code, globals())
        function_names = [line.split()[1].split('(')[0] for line in generated_code.split('\n') if line.startswith('def ')]
        evaluation(generated_code, function_names)
    elif args.knowledge_type == 'inference':
        with open(args.output_dir + 'code_generation_repo/' + args.model + '_' + args.dataset + '_inference' + '.txt', 'r') as f:
            generated_code = f.read()
        
        exec(generated_code, globals())
        function_names = [line.split()[1].split('(')[0] for line in generated_code.split('\n') if line.startswith('def ')]
        evaluation(generated_code, function_names)
    elif args.knowledge_type == 'all':
        with open(args.output_dir + 'code_generation_repo/' + args.model + '_' + args.dataset + '_synthesize' + '.txt', 'r') as f:
            pk_code = f.read()
        with open(args.output_dir + 'code_generation_repo/' + args.model + '_' + args.dataset + '_inference' + '.txt', 'r') as f:
            dk_code = f.read()
        
        # combine pk_code and dk_code
        generated_code = pk_code + '\n' + dk_code
        exec(generated_code, globals())
        function_names = [line.split()[1].split('(')[0] for line in generated_code.split('\n') if line.startswith('def ')]
        evaluation(generated_code, function_names)
    