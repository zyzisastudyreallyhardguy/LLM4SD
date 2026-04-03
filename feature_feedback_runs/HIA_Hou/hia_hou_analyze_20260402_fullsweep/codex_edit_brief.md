# Codex Edit Brief: HIA_Hou / hia_hou

- Analyze run id: `hia_hou_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__HIA_Hou/galactica-6.7b__hia_hou__synthesize__seed_0.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/HIA_Hou/hia_hou_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 32
- False positives: 25
- False negatives: 7

## Recurring Harmful Features

### false_positive

- hia_primary_rule_pass_count | samples=17 | affected_fraction=0.680 | mean_harmful_shap=0.0036 | description=Count of passed HIA composite rules among Lipinski, Veber, Egan, Palm static, Ghose, and Veber alternative.
- tpsa | samples=10 | affected_fraction=0.400 | mean_harmful_shap=0.0078 | description=Topological polar surface area.
- total_hbond_donors_acceptors | samples=9 | affected_fraction=0.360 | mean_harmful_shap=0.0044 | description=HBD + HBA total.
- mol_weight | samples=8 | affected_fraction=0.320 | mean_harmful_shap=0.0028 | description=Molecular weight (Descriptors.MolWt).
- egan_pass | samples=7 | affected_fraction=0.280 | mean_harmful_shap=0.0024 | description=Composite Egan pass flag.
- tpsa_le_140 | samples=7 | affected_fraction=0.280 | mean_harmful_shap=0.0024 | description=Rule flag: TPSA <= 140.
- estimated_logd_ph74 | samples=6 | affected_fraction=0.240 | mean_harmful_shap=0.0034 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- veber_pass | samples=6 | affected_fraction=0.240 | mean_harmful_shap=0.0023 | description=Composite Veber pass flag.
- estimated_logd_ph80 | samples=5 | affected_fraction=0.200 | mean_harmful_shap=0.0032 | description=Estimated logD at pH 8.0 from logP and neutral fraction.
- intramolecular_hbond_pair_count | samples=4 | affected_fraction=0.160 | mean_harmful_shap=0.0037 | description=Count of donor-acceptor atom pairs 4-8 bonds apart as a proxy for intramolecular H-bond potential.

### false_negative

- tpsa | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0921 | description=Topological polar surface area.
- total_hbond_donors_acceptors | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0469 | description=HBD + HBA total.
- hia_primary_rule_pass_count | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0416 | description=Count of passed HIA composite rules among Lipinski, Veber, Egan, Palm static, Ghose, and Veber alternative.
- tpsa_le_140 | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0318 | description=Rule flag: TPSA <= 140.
- egan_pass | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0284 | description=Composite Egan pass flag.
- veber_pass | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0263 | description=Composite Veber pass flag.
- tpsa_le_131p6 | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0253 | description=Rule flag: TPSA <= 131.6.
- intramolecular_hbond_pair_count | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0201 | description=Count of donor-acceptor atom pairs 4-8 bonds apart as a proxy for intramolecular H-bond potential.
- veber_alt_pass | samples=6 | affected_fraction=0.857 | mean_harmful_shap=0.0351 | description=Veber alternative pass flag using HBD + HBA <= 12.
- total_hbond_donors_acceptors_le_12 | samples=6 | affected_fraction=0.857 | mean_harmful_shap=0.0283 | description=Rule flag: HBD + HBA <= 12.

## Dropped Features

2 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
