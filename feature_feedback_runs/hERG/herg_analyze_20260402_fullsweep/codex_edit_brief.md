# Codex Edit Brief: hERG / herg

- Analyze run id: `herg_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__hERG/galactica-6.7b__herg__synthesize__seed_1.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/hERG/herg_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 64
- False positives: 31
- False negatives: 33

## Recurring Harmful Features

### false_positive

- exact_mol_weight | samples=31 | affected_fraction=1.000 | mean_harmful_shap=0.0095 | description=Exact molecular weight (Descriptors.ExactMolWt).
- molar_refractivity | samples=30 | affected_fraction=0.968 | mean_harmful_shap=0.0198 | description=Wildman-Crippen molar refractivity.
- carbon_atom_count | samples=30 | affected_fraction=0.968 | mean_harmful_shap=0.0191 | description=Carbon atom count.
- heavy_atom_count | samples=29 | affected_fraction=0.935 | mean_harmful_shap=0.0176 | description=Heavy atom count.
- mol_weight | samples=29 | affected_fraction=0.935 | mean_harmful_shap=0.0105 | description=Molecular weight (Descriptors.MolWt).
- estimated_logd_ph74 | samples=24 | affected_fraction=0.774 | mean_harmful_shap=0.0160 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- aromatic_ring_count | samples=24 | affected_fraction=0.774 | mean_harmful_shap=0.0123 | description=Aromatic ring count.
- ring_count | samples=23 | affected_fraction=0.742 | mean_harmful_shap=0.0091 | description=Total ring count.
- base_protonated_fraction_ph74 | samples=22 | affected_fraction=0.710 | mean_harmful_shap=0.0167 | description=Estimated protonated fraction for the dominant basic site at pH 7.4.
- positive_charge_present_ph74 | samples=22 | affected_fraction=0.710 | mean_harmful_shap=0.0078 | description=Rule flag: positive charge is expected at pH 7.4 from formal charge or dominant basic-site protonation.

### false_negative

- logp | samples=26 | affected_fraction=0.788 | mean_harmful_shap=0.0232 | description=Wildman-Crippen logP.
- aromatic_ring_count | samples=25 | affected_fraction=0.758 | mean_harmful_shap=0.0222 | description=Aromatic ring count.
- tpsa | samples=24 | affected_fraction=0.727 | mean_harmful_shap=0.0111 | description=Topological polar surface area.
- herg_risk_alert_count | samples=23 | affected_fraction=0.697 | mean_harmful_shap=0.0055 | description=Count of matched discrete hERG-risk alert families.
- estimated_logd_ph74 | samples=16 | affected_fraction=0.485 | mean_harmful_shap=0.0338 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- hbd | samples=16 | affected_fraction=0.485 | mean_harmful_shap=0.0064 | description=Hydrogen bond donor count.
- carbon_atom_count | samples=15 | affected_fraction=0.455 | mean_harmful_shap=0.0463 | description=Carbon atom count.
- base_protonated_fraction_ph74 | samples=14 | affected_fraction=0.424 | mean_harmful_shap=0.0382 | description=Estimated protonated fraction for the dominant basic site at pH 7.4.
- positive_charge_present_ph74 | samples=14 | affected_fraction=0.424 | mean_harmful_shap=0.0164 | description=Rule flag: positive charge is expected at pH 7.4 from formal charge or dominant basic-site protonation.
- ring_count | samples=14 | affected_fraction=0.424 | mean_harmful_shap=0.0156 | description=Total ring count.

## Dropped Features

9 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- combined_logp_pka_sq | combined_logp_pka_sq
- min_basic_center_distance | min_basic_center_distance
- min_basic_to_aromatic_distance | min_basic_to_aromatic_distance
- combined_logp_pka_sq_lt_110 | combined_logp_pka_sq_lt_110
- most_basic_pka_le_8 | most_basic_pka_le_8
- cation_pi_proxy_present | cation_pi_proxy_present
- short_basic_center_distance_present | short_basic_center_distance_present
- long_basic_to_aromatic_distance_present | long_basic_to_aromatic_distance_present

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
