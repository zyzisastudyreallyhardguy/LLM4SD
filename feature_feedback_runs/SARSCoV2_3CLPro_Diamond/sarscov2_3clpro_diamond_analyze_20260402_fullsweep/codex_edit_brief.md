# Codex Edit Brief: SARSCoV2_3CLPro_Diamond / sarscov2_3clpro_diamond

- Analyze run id: `sarscov2_3clpro_diamond_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__SARSCoV2_3CLPro_Diamond/galactica-6.7b__sarscov2_3clpro_diamond__synthesize__seed_0.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/SARSCoV2_3CLPro_Diamond/sarscov2_3clpro_diamond_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 48
- False positives: 35
- False negatives: 13

## Recurring Harmful Features

### false_positive

- mol_weight | samples=33 | affected_fraction=0.943 | mean_harmful_shap=0.0732 | description=Molecular weight (Descriptors.MolWt).
- p3_p4_polar_substituent_present | samples=33 | affected_fraction=0.943 | mean_harmful_shap=0.0068 | description=1 if amide-like polar substituents suggest a P3/P4 hydrogen-bonding handle.
- exact_mol_weight | samples=28 | affected_fraction=0.800 | mean_harmful_shap=0.0869 | description=Exact molecular weight (Descriptors.ExactMolWt).
- heteroatom_count | samples=25 | affected_fraction=0.714 | mean_harmful_shap=0.0168 | description=Heteroatom count.
- amide_bond_count | samples=24 | affected_fraction=0.686 | mean_harmful_shap=0.0090 | description=Amide-bond count.
- polar_amide_like_group_count | samples=23 | affected_fraction=0.657 | mean_harmful_shap=0.0192 | description=Sum of amide, carbamate, urea, and sulfonamide counts as a polar P3/P4 substituent proxy.
- rotatable_bonds | samples=23 | affected_fraction=0.657 | mean_harmful_shap=0.0141 | description=Rotatable bond count.
- fraction_csp3 | samples=22 | affected_fraction=0.629 | mean_harmful_shap=0.0072 | description=Fraction of sp3 carbons.
- ring_count | samples=17 | affected_fraction=0.486 | mean_harmful_shap=0.0129 | description=Total ring count.
- p2_hydrophobic_sidechain_present | samples=17 | affected_fraction=0.486 | mean_harmful_shap=0.0045 | description=1 if hydrophobic branched/cyclic P2-sidechain proxies are present.

### false_negative

- exact_mol_weight | samples=13 | affected_fraction=1.000 | mean_harmful_shap=0.0318 | description=Exact molecular weight (Descriptors.ExactMolWt).
- mol_weight | samples=13 | affected_fraction=1.000 | mean_harmful_shap=0.0306 | description=Molecular weight (Descriptors.MolWt).
- polar_amide_like_group_count | samples=13 | affected_fraction=1.000 | mean_harmful_shap=0.0097 | description=Sum of amide, carbamate, urea, and sulfonamide counts as a polar P3/P4 substituent proxy.
- sulfonamide_count | samples=12 | affected_fraction=0.923 | mean_harmful_shap=0.0093 | description=Sulfonamide count.
- heteroatom_count | samples=11 | affected_fraction=0.846 | mean_harmful_shap=0.0159 | description=Heteroatom count.
- heavy_atom_count | samples=11 | affected_fraction=0.846 | mean_harmful_shap=0.0121 | description=Heavy atom count.
- p3_p4_polar_substituent_present | samples=8 | affected_fraction=0.615 | mean_harmful_shap=0.0102 | description=1 if amide-like polar substituents suggest a P3/P4 hydrogen-bonding handle.
- rotatable_bonds | samples=7 | affected_fraction=0.538 | mean_harmful_shap=0.0082 | description=Rotatable bond count.
- amide_bond_count | samples=7 | affected_fraction=0.538 | mean_harmful_shap=0.0081 | description=Amide-bond count.
- mpro_physchem_window_pass | samples=6 | affected_fraction=0.462 | mean_harmful_shap=0.0036 | description=1 if all physicochemical windows from the source response are satisfied.

## Dropped Features

No features were dropped by the train-time preprocessing step.

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
