# Codex Edit Brief: PAMPA_NCATS / pampa_ncats

- Analyze run id: `pampa_ncats_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__PAMPA_NCATS/galactica-6.7b__pampa_ncats__synthesize__seed_0.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/PAMPA_NCATS/pampa_ncats_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 154
- False positives: 133
- False negatives: 21

## Recurring Harmful Features

### false_positive

- mol_weight | samples=80 | affected_fraction=0.602 | mean_harmful_shap=0.0046 | description=Molecular weight (Descriptors.MolWt).
- estimated_logd_ph74 | samples=79 | affected_fraction=0.594 | mean_harmful_shap=0.0217 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- balaban_j | samples=65 | affected_fraction=0.489 | mean_harmful_shap=0.0050 | description=Balaban J topological index.
- num_acidic_sites | samples=64 | affected_fraction=0.481 | mean_harmful_shap=0.0035 | description=Number of predicted acidic sites from the MolGpKa helper.
- kappa3 | samples=62 | affected_fraction=0.466 | mean_harmful_shap=0.0064 | description=Third kappa shape index.
- labute_asa_per_heavy_atom | samples=61 | affected_fraction=0.459 | mean_harmful_shap=0.0030 | description=Labute ASA divided by heavy-atom count.
- tpsa | samples=59 | affected_fraction=0.444 | mean_harmful_shap=0.0138 | description=Topological polar surface area.
- hbd | samples=59 | affected_fraction=0.444 | mean_harmful_shap=0.0058 | description=Hydrogen bond donor count.
- acid_deprotonated_fraction_ph74 | samples=55 | affected_fraction=0.414 | mean_harmful_shap=0.0161 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- neutral_fraction_ph74 | samples=55 | affected_fraction=0.414 | mean_harmful_shap=0.0039 | description=Estimated neutral fraction at pH 7.4.

### false_negative

- charged_fraction_ph74 | samples=21 | affected_fraction=1.000 | mean_harmful_shap=0.0206 | description=1 - neutral_fraction_ph74.
- net_charge_proxy_ph74 | samples=21 | affected_fraction=1.000 | mean_harmful_shap=0.0198 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- neutral_fraction_ph74 | samples=21 | affected_fraction=1.000 | mean_harmful_shap=0.0195 | description=Estimated neutral fraction at pH 7.4.
- acid_deprotonated_fraction_ph74 | samples=19 | affected_fraction=0.905 | mean_harmful_shap=0.0514 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- estimated_logd_ph74 | samples=17 | affected_fraction=0.810 | mean_harmful_shap=0.3007 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- tpsa | samples=15 | affected_fraction=0.714 | mean_harmful_shap=0.0297 | description=Topological polar surface area.
- kappa1 | samples=15 | affected_fraction=0.714 | mean_harmful_shap=0.0045 | description=First kappa shape index.
- hbd | samples=13 | affected_fraction=0.619 | mean_harmful_shap=0.0128 | description=Hydrogen bond donor count.
- molar_refractivity | samples=13 | affected_fraction=0.619 | mean_harmful_shap=0.0063 | description=Wildman-Crippen molar refractivity.
- logp | samples=12 | affected_fraction=0.571 | mean_harmful_shap=0.0193 | description=Wildman-Crippen logP.

## Dropped Features

2 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
