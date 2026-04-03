# Codex Edit Brief: CYP3A4_Substrate_CarbonMangels / cyp3a4_substrate_carbonmangels

- Analyze run id: `cyp3a4_substrate_carbonmangels_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__CYP3A4_Substrate_CarbonMangels/galactica-6.7b__cyp3a4_substrate_carbonmangels__synthesize__seed_4.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/CYP3A4_Substrate_CarbonMangels/cyp3a4_substrate_carbonmangels_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 102
- False positives: 53
- False negatives: 49

## Recurring Harmful Features

### false_positive

- exact_mol_weight | samples=45 | affected_fraction=0.849 | mean_harmful_shap=0.0084 | description=Exact molecular weight (Descriptors.ExactMolWt).
- carbon_atom_count | samples=44 | affected_fraction=0.830 | mean_harmful_shap=0.0126 | description=Carbon atom count.
- estimated_logd_ph74 | samples=43 | affected_fraction=0.811 | mean_harmful_shap=0.0283 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- molar_refractivity | samples=43 | affected_fraction=0.811 | mean_harmful_shap=0.0104 | description=Wildman-Crippen molar refractivity.
- acid_deprotonated_fraction_ph74 | samples=42 | affected_fraction=0.792 | mean_harmful_shap=0.0091 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- logp | samples=41 | affected_fraction=0.774 | mean_harmful_shap=0.0160 | description=Wildman-Crippen logP.
- mol_weight | samples=41 | affected_fraction=0.774 | mean_harmful_shap=0.0103 | description=Molecular weight (Descriptors.MolWt).
- labute_asa | samples=33 | affected_fraction=0.623 | mean_harmful_shap=0.0142 | description=Labute approximate surface area as a stable size or volume proxy.
- heavy_atom_count | samples=30 | affected_fraction=0.566 | mean_harmful_shap=0.0101 | description=Heavy atom count.
- estimated_logd_gt_3 | samples=25 | affected_fraction=0.472 | mean_harmful_shap=0.0151 | description=Rule flag: estimated logD(7.4) > 3.

### false_negative

- estimated_logd_gt_3 | samples=48 | affected_fraction=0.980 | mean_harmful_shap=0.0089 | description=Rule flag: estimated logD(7.4) > 3.
- estimated_logd_ph74 | samples=41 | affected_fraction=0.837 | mean_harmful_shap=0.0280 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- carbon_atom_count | samples=37 | affected_fraction=0.755 | mean_harmful_shap=0.0182 | description=Carbon atom count.
- heavy_atom_count | samples=36 | affected_fraction=0.735 | mean_harmful_shap=0.0110 | description=Heavy atom count.
- neutral_fraction_ph74 | samples=30 | affected_fraction=0.612 | mean_harmful_shap=0.0179 | description=Estimated neutral fraction at pH 7.4.
- labute_asa | samples=30 | affected_fraction=0.612 | mean_harmful_shap=0.0166 | description=Labute approximate surface area as a stable size or volume proxy.
- hba | samples=30 | affected_fraction=0.612 | mean_harmful_shap=0.0060 | description=Hydrogen bond acceptor count.
- molar_refractivity | samples=29 | affected_fraction=0.592 | mean_harmful_shap=0.0153 | description=Wildman-Crippen molar refractivity.
- charged_fraction_ph74 | samples=28 | affected_fraction=0.571 | mean_harmful_shap=0.0136 | description=1 - neutral_fraction_ph74.
- logp | samples=27 | affected_fraction=0.551 | mean_harmful_shap=0.0263 | description=Wildman-Crippen logP.

## Dropped Features

2 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
