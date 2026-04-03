# Codex Edit Brief: ClinTox / clintox

- Analyze run id: `clintox_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__ClinTox/galactica-6.7b__clintox__synthesize__seed_0.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/ClinTox/clintox_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 32
- False positives: 30
- False negatives: 2

## Recurring Harmful Features

### false_positive

- oxygen_nitrogen_count | samples=30 | affected_fraction=1.000 | mean_harmful_shap=0.0578 | description=Count of O and N atoms.
- estimated_logd_ph74 | samples=28 | affected_fraction=0.933 | mean_harmful_shap=0.0765 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- aromatic_ring_count | samples=26 | affected_fraction=0.867 | mean_harmful_shap=0.0122 | description=Aromatic ring count.
- ring_count | samples=26 | affected_fraction=0.867 | mean_harmful_shap=0.0108 | description=Total ring count.
- tpsa | samples=23 | affected_fraction=0.767 | mean_harmful_shap=0.0109 | description=Topological polar surface area.
- unspecified_chiral_center_count | samples=23 | affected_fraction=0.767 | mean_harmful_shap=0.0109 | description=Count of unspecified atom stereo centers.
- logp | samples=18 | affected_fraction=0.600 | mean_harmful_shap=0.0377 | description=Wildman-Crippen logP.
- hba | samples=16 | affected_fraction=0.533 | mean_harmful_shap=0.0060 | description=Hydrogen bond acceptor count.
- chiral_center_count | samples=15 | affected_fraction=0.500 | mean_harmful_shap=0.0120 | description=Count of atom stereo centers.
- charged_fraction_ph74 | samples=15 | affected_fraction=0.500 | mean_harmful_shap=0.0081 | description=1 - neutral_fraction_ph74.

### false_negative

- oxygen_nitrogen_count | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0440 | description=Count of O and N atoms.
- estimated_logd_ph74 | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0193 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- num_basic_sites | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0157 | description=Number of predicted basic sites from the MolGpKa helper.
- base_protonated_fraction_ph74 | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0150 | description=Estimated protonated fraction for the dominant basic site at pH 7.4.
- heteroatom_count | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0142 | description=Total heteroatom count.
- ring_count | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0107 | description=Total ring count.
- hbd | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0069 | description=Hydrogen bond donor count.
- aromatic_ring_count | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0068 | description=Aromatic ring count.
- aromatic_amine_count | samples=2 | affected_fraction=1.000 | mean_harmful_shap=0.0064 | description=Count of primary or secondary aromatic amine nitrogens.
- acid_deprotonated_fraction_ph74 | samples=1 | affected_fraction=0.500 | mean_harmful_shap=0.0431 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.

## Dropped Features

5 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka
- most_basic_pka_lt_7 | most_basic_pka_lt_7
- most_basic_pka_ge_7 | most_basic_pka_ge_7
- basic_aryl_halide_risk_flag | basic_aryl_halide_risk_flag

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
