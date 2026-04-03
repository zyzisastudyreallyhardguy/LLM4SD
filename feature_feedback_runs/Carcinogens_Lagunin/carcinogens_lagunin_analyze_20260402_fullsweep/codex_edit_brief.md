# Codex Edit Brief: Carcinogens_Lagunin / carcinogens_lagunin

- Analyze run id: `carcinogens_lagunin_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__Carcinogens_Lagunin/galactica-6.7b__carcinogens_lagunin__synthesize__seed_0.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/Carcinogens_Lagunin/carcinogens_lagunin_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 8
- False positives: 1
- False negatives: 7

## Recurring Harmful Features

### false_positive

- acid_deprotonated_fraction_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.1069 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- charged_fraction_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0856 | description=1 - neutral_fraction_ph74.
- neutral_fraction_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0653 | description=Estimated neutral fraction at pH 7.4.
- azo_linkage_present | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0576 | description=1 if an azo linkage alert is present.
- azo_linkage_count | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0344 | description=Count of azo linkage alerts.
- logp | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0310 | description=Wildman-Crippen logP.
- fraction_csp3 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0253 | description=Fraction of sp3 carbons.
- aromatic_ring_count | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0244 | description=Aromatic ring count.
- aliphatic_ring_count | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0181 | description=Aliphatic ring count.
- high_aromatic_ring_burden_flag | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0176 | description=Heuristic flag: aromatic ring count >= 3.

### false_negative

- azo_linkage_present | samples=7 | affected_fraction=1.000 | mean_harmful_shap=0.0063 | description=1 if an azo linkage alert is present.
- acid_deprotonated_fraction_ph74 | samples=6 | affected_fraction=0.857 | mean_harmful_shap=0.0107 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- charged_fraction_ph74 | samples=4 | affected_fraction=0.571 | mean_harmful_shap=0.0071 | description=1 - neutral_fraction_ph74.
- heavy_atom_count | samples=3 | affected_fraction=0.429 | mean_harmful_shap=0.0056 | description=Heavy atom count.
- azo_linkage_count | samples=3 | affected_fraction=0.429 | mean_harmful_shap=0.0049 | description=Count of azo linkage alerts.
- aliphatic_ring_count | samples=2 | affected_fraction=0.286 | mean_harmful_shap=0.0169 | description=Aliphatic ring count.
- mol_weight | samples=2 | affected_fraction=0.286 | mean_harmful_shap=0.0050 | description=Molecular weight (Descriptors.MolWt).
- aromatic_ring_count | samples=2 | affected_fraction=0.286 | mean_harmful_shap=0.0050 | description=Aromatic ring count.
- ring_count | samples=1 | affected_fraction=0.143 | mean_harmful_shap=0.0065 | description=Total ring count.
- estimated_logd_ph74 | samples=1 | affected_fraction=0.143 | mean_harmful_shap=0.0063 | description=Estimated logD at pH 7.4 from logP and neutral fraction.

## Dropped Features

2 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
