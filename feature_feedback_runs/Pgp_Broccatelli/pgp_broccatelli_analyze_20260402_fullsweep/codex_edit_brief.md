# Codex Edit Brief: Pgp_Broccatelli / pgp_broccatelli

- Analyze run id: `pgp_broccatelli_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__Pgp_Broccatelli/galactica-6.7b__pgp_broccatelli__synthesize__seed_2.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/Pgp_Broccatelli/pgp_broccatelli_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 76
- False positives: 37
- False negatives: 39

## Recurring Harmful Features

### false_positive

- mw_ge_300 | samples=37 | affected_fraction=1.000 | mean_harmful_shap=0.0106 | description=Rule flag: molecular weight >= 300.
- hydrophobic_large_molecule | samples=34 | affected_fraction=0.919 | mean_harmful_shap=0.0278 | description=Composite flag: molecular weight >= 300 and logP >= 3.
- logp_ge_3 | samples=34 | affected_fraction=0.919 | mean_harmful_shap=0.0132 | description=Rule flag: logP >= 3.
- logp_le_3 | samples=34 | affected_fraction=0.919 | mean_harmful_shap=0.0116 | description=Rule flag: logP <= 3.
- pgp_flowchart_high_likelihood | samples=31 | affected_fraction=0.838 | mean_harmful_shap=0.0316 | description=Composite flag implementing the simplified DeepResearch flowchart high-likelihood branch.
- molar_refractivity | samples=29 | affected_fraction=0.784 | mean_harmful_shap=0.0386 | description=Wildman-Crippen molar refractivity.
- carbon_atom_count | samples=26 | affected_fraction=0.703 | mean_harmful_shap=0.0478 | description=Total carbon atom count.
- heavy_atom_count | samples=25 | affected_fraction=0.676 | mean_harmful_shap=0.0287 | description=Heavy atom count.
- exact_mol_weight | samples=25 | affected_fraction=0.676 | mean_harmful_shap=0.0194 | description=Exact molecular weight (Descriptors.ExactMolWt).
- mol_weight | samples=25 | affected_fraction=0.676 | mean_harmful_shap=0.0129 | description=Molecular weight (Descriptors.MolWt).

### false_negative

- carbon_atom_count | samples=35 | affected_fraction=0.897 | mean_harmful_shap=0.0554 | description=Total carbon atom count.
- pgp_flowchart_high_likelihood | samples=35 | affected_fraction=0.897 | mean_harmful_shap=0.0388 | description=Composite flag implementing the simplified DeepResearch flowchart high-likelihood branch.
- hydrophobic_large_molecule | samples=35 | affected_fraction=0.897 | mean_harmful_shap=0.0323 | description=Composite flag: molecular weight >= 300 and logP >= 3.
- aromatic_ring_count | samples=35 | affected_fraction=0.897 | mean_harmful_shap=0.0160 | description=Aromatic ring count.
- exact_mol_weight | samples=33 | affected_fraction=0.846 | mean_harmful_shap=0.0202 | description=Exact molecular weight (Descriptors.ExactMolWt).
- molar_refractivity | samples=32 | affected_fraction=0.821 | mean_harmful_shap=0.0574 | description=Wildman-Crippen molar refractivity.
- heavy_atom_count | samples=32 | affected_fraction=0.821 | mean_harmful_shap=0.0342 | description=Heavy atom count.
- mol_weight | samples=31 | affected_fraction=0.795 | mean_harmful_shap=0.0149 | description=Molecular weight (Descriptors.MolWt).
- logp_ge_3 | samples=28 | affected_fraction=0.718 | mean_harmful_shap=0.0171 | description=Rule flag: logP >= 3.
- logp_le_3 | samples=28 | affected_fraction=0.718 | mean_harmful_shap=0.0161 | description=Rule flag: logP <= 3.

## Dropped Features

3 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_basic_pka_gt_8 | most_basic_pka_gt_8
- most_basic_pka_lt_8 | most_basic_pka_lt_8

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
