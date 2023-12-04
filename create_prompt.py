import json
import argparse
import os


def get_synthesize_task_prompt():
    prompt_dict = {}
    prompt_dict['bbbp_small'] = "Assumed you are an experienced chemist. Please come up with 20 rules that you think are very important to predict if a molecule is blood brain barrier permeable (BBBP). Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 5 words."
    prompt_dict['bbbp_big'] = "Assumed you are an experienced chemist. Please come up with 30 rules that you think are very important to predict if a molecule is blood brain barrier permeable (BBBP). Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 20 words."
    prompt_dict['clintox_small'] = "Assumed you are an experienced chemist. Please come up with 20 rules that you think are very important to predict if a molecule will be approved by the FDA or failed clinical trials for toxicity reasons.. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 5 words."
    prompt_dict['clintox_big'] = "Assumed you are an experienced chemist. Please come up with 30 rules that you think are very important to predict if a molecule will fail clinical trails due to toxicity reasons. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 20 words."
    prompt_dict['hiv_small'] = "Assumed you are an experienced chemist. Please come up with 20 rules that you think are very important to predict if a molecule can inhibit HIV replication. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 5 words."
    prompt_dict['hiv_big'] = "Assumed you are an experienced chemist. Please come up with 30 rules that you think are very important to predict if a molecule can inhibit HIV replication. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 20 words."
    prompt_dict['bace_small'] = "Assumed you are an experienced chemist. Please come up with 20 rules that you think are very important to predict if a molecule can inhibit human \u03b2-secretase 1(BACE-1). Each rule starts with 'Calculate....' and don't explain, be short and within 5 wrods."
    prompt_dict['bace_big'] = "Assumed you are an experienced chemist. Please come up with 30 rules that you think are very important to predict if a molecule can inhibit HIV replication. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 20 words."
    prompt_dict['esol_small'] = "Assumed you are an experienced chemist. Please come up with 20 rules that you think are very important to predict the water solubility data(log solubility in mols per litre) for a molecule. Each rule starts with 'Calculate....' and don't explain, be short and within 5 words."
    prompt_dict['esol_big'] = "Assumed you are an experienced chemist. Please come up with 30 rules that you think are very important to predict the water solubility data(log solubility in mols per litre) for a molecule. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 20 words."
    prompt_dict['lipophilicity_small'] = "Assumed you are an experienced chemist. Please come up with 20 rules that you think are very important to predict octanol/water distribution coefficient(logD at pH 7.4)for a molecule. Each rule starts with 'Calculate....' and don't explain, be short and within 5 words."
    prompt_dict['lipophilicity_big'] = "Assumed you are an experienced chemist. Please come up with 30 rules that you think are very important to predict octanol/water distribution coefficient(logD at pH 7.4) for a molecule. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 20 words."
    prompt_dict['freesolv_small'] = "Assumed you are an experienced chemist. Please come up with 20 rules that you think are very important to predict hydration free energy of a molecule in water. Each rule starts with 'Calculate....' and don't explain, be short and within 5 wrods. "
    prompt_dict['freesolv_big'] = "Assumed you are an experienced chemist. Please come up with 30 rules that you think are very important to predict hydration free energy of a molecule in water. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within 20 words."

    tox21 = ['nr-ar', 'nr-ar-lbd', 'nr-ahr', 'nr-aromatase', 'nr-er', 'nr-er-lbd', 'nr-ppar-gamma',
             'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']
    sider = ['hepatobiliary disorders', 'metabolism and nutrition disorders', 'product issues', 'eye disorders',
             'investigations', 'musculoskeletal and connective tissue disorders', 'gastrointestinal disorders',
             'social circumstances', 'immune system disorders', 'reproductive system and breast disorders',
             'neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'general disorders and administration site conditions', 'endocrine disorders',
             'surgical and medical procedures', 'vascular disorders', 'blood and lymphatic system disorders',
             'skin and subcutaneous tissue disorders', 'congenital, familial and genetic disorders',
             'infections and infestations', 'respiratory, thoracic and mediastinal disorders', 'psychiatric disorders',
             'renal and urinary disorders', 'pregnancy, puerperium and perinatal conditions',
             'ear and labyrinth disorders', 'cardiac disorders', 'nervous system disorders',
             'injury, poisoning and procedural complications']
    
    prompt_dict['tox21_small'] = {}
    prompt_dict['tox21_big'] = {}
    prompt_dict['sider_small'] = {}
    prompt_dict['sider_big'] = {}

    for item in tox21:
        if item == 'nr-ar':
            task_desc =  'toxicity activity of a molecule against the androgen receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-ar-lbd': 
            task_desc = 'toxicity activity of a molecule against the androgen receptor ligand-binding domain in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-ahr':
            task_desc = 'toxicity activity of a molecule against the aryl hydrocarbon receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-aromatase':
            task_desc = 'toxicity activity of a molecule against the aromatase in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-er':
            task_desc = 'toxicity activity of a molecule against the estrogen receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-er-lbd':
            task_desc = 'toxicity activity of a molecule against the estrogen receptor ligand-binding domain in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-ppar-gamma':
            task_desc =  'toxicity activity of a molecule against the peroxisome proliferator activated receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'sr-are':
            task_desc =  'toxicity activity of a molecule against the nuclear factor (erythroid- derived 2)-like 2 antioxidant responsive element in the stress response (SR) signaling pathway'
        elif item == 'sr-atad5':
            task_desc =  'toxicity activity of a molecule against the genotoxicity indicated by ATAD5 in the stress response (SR) signaling pathway'
        elif item == 'sr-hse':
            task_desc =  'toxicity activity of a molecule against the heat shock factor response element in the stress response (SR) signaling pathway'
        elif item == 'sr-mmp':
            task_desc =  'toxicity activity of a molecule against the mitochondrial membrane potential in the stress response (SR) signaling pathway'
        elif item == 'sr-p53':
            task_desc =  'toxicity activity of a molecule against the DNA damage p53-pathway in the stress response (SR) signaling pathway'

        tox21_prompt = ("Assumed you are an experienced chemist. Please come up with {k_rules} rules that you think are very important to predict {item}. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within {k_words} words.")
        prompt_dict['tox21_small'][item] = tox21_prompt.format_map({'k_rules': '20', 'item': task_desc, 'k_words': '5'})
        prompt_dict['tox21_big'][item] = tox21_prompt.format_map({'k_rules': '30', 'item': task_desc, 'k_words': '20'})


    for item in sider:
        
        sider_prompt = ("Assumed you are an experienced chemist. Please come up with {k_rules} rules that you think are very important to predict {item}. Each rule is either about the structure or property of molecules. Each rule starts with 'Calculate....' and don't explain, be short and within {k_words} words.")
        task_desc = "the toxicity activity of a molecule in causing " + item
        prompt_dict['sider_small'][item] = sider_prompt.format_map({'k_rules': '20', 'item': task_desc, 'k_words': '5'})
        prompt_dict['sider_big'][item] = sider_prompt.format_map({'k_rules': '30', 'item': task_desc, 'k_words': '20'})
        
    prompt_dict['mu_small'] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict dipole moment (Mu) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict['alpha_small'] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict Isotropic polarizability of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict['R^2_small'] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict electronic spatial extent (R2) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["epsilon_HOMO_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict Highest occupied molecular orbital (HOMO) energy of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["epsilon_LUMO_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict Lowest Unoccupied Molecular Orbital (LUMO) energy of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["Delta_epsilon_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict HUMO-LUMO gap of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["ZPVE_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict Zero-Point Vibrational Energy (ZPVE) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["U_0_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict U0 refers to the internal energy at absolute zero temperature (0 Kelvin) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["U_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict U (the internal energy of a molecule at a specific temperature, specifically at 298.15 Kelvin (approximately room temperature) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["H_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict H, the enthalpy of the molecule at a specific temperature, specifically 298.15 Kelvin (approximately room temperature) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["G_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict G, Gibbs free energy of the molecule at a specific temperature, specifically 298.15 Kelvin (approximately room temperature) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["c_v_small"] = "Assume you are an experienced chemist. please come up with 20 rules that you think are very important to predict Cv, the heat capacity at constant volume of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."

    prompt_dict['mu_big'] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict dipole moment (Mu) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict['alpha_big'] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict Isotropic polarizability of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict['R^2_big'] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict electronic spatial extent (R2) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["epsilon_HOMO_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict Highest occupied molecular orbital (HOMO) energy of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["epsilon_LUMO_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict Lowest Unoccupied Molecular Orbital (LUMO) energy of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["Delta_epsilon_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict HUMO-LUMO gap of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["ZPVE_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict Zero-Point Vibrational Energy (ZPVE) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["U_0_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict U0 refers to the internal energy at absolute zero temperature (0 Kelvin) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["U_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict U (the internal energy of a molecule at a specific temperature, specifically at 298.15 Kelvin (approximately room temperature) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["H_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict H, the enthalpy of the molecule at a specific temperature, specifically 298.15 Kelvin (approximately room temperature) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["G_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict G, Gibbs free energy of the molecule at a specific temperature, specifically 298.15 Kelvin (approximately room temperature) of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    prompt_dict["c_v_big"] = "Assume you are an experienced chemist. please come up with 50 rules that you think are very important to predict Cv, the heat capacity at constant volume of a molecule. Each rule is either about the structure or property of molecules without access to 3D information. and should be short (no more than 15 words). Each rule needs to be very specific. please start with 'The list of rules are....'."
    
    return prompt_dict


def get_inference_task_prompt():
    prompt_dict = {}

    prompt_dict['bbbp'] = 'Assume you are a very experienced Chemist. In the following data, with label 1, it means the smiles string is BBBP. With label 0, it means the smiles string is not BBBP. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict if it can be BBBP.'
    prompt_dict['clintox'] = 'Assume you are a very experienced Chemist. In the following data, with label 1, it means the molecule is approved by FDA, With label 0, it means the molecule failed clinical trails for toxicity reasons. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict if it can be approved by FDA.' 
    prompt_dict['hiv'] = 'Assume you are a very experienced Chemist. In the following data, with label 1, it means the molecule can inhibit HIV replication, With label 0, it means the molecule cannot inhibit HIV replication. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict if it can be inhibit HIV replication.'
    prompt_dict['bace'] = 'Assume you are a very experienced Chemist. In the following data, with label 1, it means the molecule can inhibit human β-secretase 1(BACE-1), With label 0, it means the molecule cannot inhibit human β-secretase 1(BACE-1). Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict if it can be inhibit human β-secretase 1(BACE-1).'
    prompt_dict['esol'] = 'Assume you are a very experienced chemist. The following data includes molecules and their corresponding value (the water solubility data/ ESOL value (log solubility in mols per litre). Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict molecules ESOL value.'
    prompt_dict['lipophilicity'] = 'Assume you are a very experienced chemist. The following data includes molecules and their corresponding value (octanol/water distribution coefficient (logD at pH 7.4)). Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict molecules lipophilicity.'
    prompt_dict['freesolv'] = 'Assume you are a very experienced chemist. The following data includes molecules and their corresponding value (hydration free energy of a molecule in water). Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict molecules hydration free energy in water.'

    tox21 = ['nr-ar', 'nr-ar-lbd', 'nr-ahr', 'nr-aromatase', 'nr-er', 'nr-er-lbd', 'nr-ppar-gamma',
             'sr-are', 'sr-atad5', 'sr-hse', 'sr-mmp', 'sr-p53']
    sider = ['hepatobiliary disorders', 'metabolism and nutrition disorders', 'product issues', 'eye disorders',
             'investigations', 'musculoskeletal and connective tissue disorders', 'gastrointestinal disorders',
             'social circumstances', 'immune system disorders', 'reproductive system and breast disorders',
             'neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'general disorders and administration site conditions', 'endocrine disorders',
             'surgical and medical procedures', 'vascular disorders', 'blood and lymphatic system disorders',
             'skin and subcutaneous tissue disorders', 'congenital, familial and genetic disorders',
             'infections and infestations', 'respiratory, thoracic and mediastinal disorders', 'psychiatric disorders',
             'renal and urinary disorders', 'pregnancy, puerperium and perinatal conditions',
             'ear and labyrinth disorders', 'cardiac disorders', 'nervous system disorders',
             'injury, poisoning and procedural complications']
    prompt_dict['tox21'] = {}
    prompt_dict['sider'] = {}

    for item in tox21:
        if item == 'nr-ar':
            task_desc =  'toxicity activity of a molecule against the androgen receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-ar-lbd': 
            task_desc = 'toxicity activity of a molecule against the androgen receptor ligand-binding domain in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-ahr':
            task_desc = 'toxicity activity of a molecule against the aryl hydrocarbon receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-aromatase':
            task_desc = 'toxicity activity of a molecule against the aromatase in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-er':
            task_desc = 'toxicity activity of a molecule against the estrogen receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-er-lbd':
            task_desc = 'toxicity activity of a molecule against the estrogen receptor ligand-binding domain in the nuclear receptor (NR) signaling pathway'
        elif item == 'nr-ppar-gamma':
            task_desc =  'toxicity activity of a molecule against the peroxisome proliferator activated receptor in the nuclear receptor (NR) signaling pathway'
        elif item == 'sr-are':
            task_desc =  'toxicity activity of a molecule against the nuclear factor (erythroid- derived 2)-like 2 antioxidant responsive element in the stress response (SR) signaling pathway'
        elif item == 'sr-atad5':
            task_desc =  'toxicity activity of a molecule against the genotoxicity indicated by ATAD5 in the stress response (SR) signaling pathway'
        elif item == 'sr-hse':
            task_desc =  'toxicity activity of a molecule against the heat shock factor response element in the stress response (SR) signaling pathway'
        elif item == 'sr-mmp':
            task_desc =  'toxicity activity of a molecule against the mitochondrial membrane potential in the stress response (SR) signaling pathway'
        elif item == 'sr-p53':
            task_desc =  'toxicity activity of a molecule against the DNA damage p53-pathway in the stress response (SR) signaling pathway'
        
        tox21_prompt = "Assume you are a very experienced Chemist. In the following data, with label 1, it means the smiles string is related to {}. With label 0, it means the smiles string is not. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict whether it is can cause adverse effect."
        prompt_dict['tox21'][item] = tox21_prompt.format(task_desc)

    for item in sider:
        sider_prompt = "Assume you are a very experienced Chemist. In the following data, with label 1, it means the smiles string is related to {}. With label 0, it means the smiles string is not. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict whether it is can cause adverse effect."
        task_desc = "the side-effect activity of a molecule related to " + item
        prompt_dict['sider'][item] = sider_prompt.format(task_desc)

    prompt_dict['mu'] = "Assume you are an experienced chemist. The following data include molecules and their corresponding Mu value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the Mu value."
    prompt_dict['alpha'] = "Assume you are an experienced chemist. The following data include molecules and their corresponding alpha value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the alpha value."
    prompt_dict['R^2'] = "Assume you are an experienced chemist. The following data include molecules and their corresponding epsilon R^2 value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the R^2 value."
    prompt_dict["epsilon_HOMO"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding HOMO value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the epsilon HOMO value."
    prompt_dict["epsilon_LUMO"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding epsilon LUMO value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the epsilon LUMO value."
    prompt_dict["Delta_epsilon"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding HOMO-LUMO gap value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the HOMO-LUMO gap value."
    prompt_dict["ZPVE"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding ZPVE value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the ZPVE value."
    prompt_dict["U_0"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding U_0 value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the U_0 value."
    prompt_dict["U"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding U value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the U value."
    prompt_dict["H"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding H value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the H value."
    prompt_dict["G"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding G value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the G value."
    prompt_dict["c_v"] = "Assume you are an experienced chemist. The following data include molecules and their corresponding Cv value. Please infer step-by-step to come up with 3 rules that directly relate the properties/structures of a molecule to predict the Cv value."

    return prompt_dict


def main():
    if args.task == 'synthesize':
        prompt_dict = get_synthesize_task_prompt()
    elif args.task == 'inference':
        prompt_dict = get_inference_task_prompt()
    else:
        raise NotImplementedError(f"""No implementation for task {args.task}.""")
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    output_filename = os.path.join(args.output_folder, f'{args.task}_prompt.json')
    with open(output_filename, 'w') as f:
        json.dump(prompt_dict, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='synthesize', help='synthesize/inference')
    parser.add_argument('--output_folder', type=str, default='prompt_file', help='prompt json file output folder')
    args = parser.parse_args()
    main()