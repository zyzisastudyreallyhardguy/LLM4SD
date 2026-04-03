# Codex Edit Brief: BBB_Martins / bbb_martins

- Analyze run id: `bbb_martins_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__BBB_Martins/galactica-6.7b__bbb_martins__synthesize__seed_1.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/BBB_Martins/bbb_martins_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 139
- False positives: 41
- False negatives: 98

## Recurring Harmful Features

### false_positive

- heteroatom_count | samples=41 | affected_fraction=1.000 | mean_harmful_shap=0.0128 | description=Total heteroatom count.
- oxygen_nitrogen_count | samples=40 | affected_fraction=0.976 | mean_harmful_shap=0.0158 | description=Count of O and N atoms.
- hbd_lt_3 | samples=40 | affected_fraction=0.976 | mean_harmful_shap=0.0128 | description=Rule flag: HBD < 3.
- acid_deprotonated_fraction_ph74 | samples=38 | affected_fraction=0.927 | mean_harmful_shap=0.0225 | description=Estimated deprotonated fraction for the dominant acidic site.
- tpsa_lt_79 | samples=38 | affected_fraction=0.927 | mean_harmful_shap=0.0169 | description=Rule flag: TPSA < 79.
- boiled_egg_bbb_likely | samples=34 | affected_fraction=0.829 | mean_harmful_shap=0.0111 | description=Rule flag: BOILED-Egg BBB likely region.
- tpsa | samples=30 | affected_fraction=0.732 | mean_harmful_shap=0.0180 | description=Topological polar surface area.
- net_charge_proxy_ph74 | samples=30 | affected_fraction=0.732 | mean_harmful_shap=0.0108 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- total_hbond_donors_acceptors | samples=27 | affected_fraction=0.659 | mean_harmful_shap=0.0141 | description=HBD + HBA total.
- tpsa_lt_70 | samples=27 | affected_fraction=0.659 | mean_harmful_shap=0.0082 | description=Rule flag: TPSA < 70.

### false_negative

- boiled_egg_bbb_likely | samples=80 | affected_fraction=0.816 | mean_harmful_shap=0.0129 | description=Rule flag: BOILED-Egg BBB likely region.
- oxygen_nitrogen_le_5 | samples=78 | affected_fraction=0.796 | mean_harmful_shap=0.0108 | description=Rule flag: O + N <= 5.
- tpsa_lt_79 | samples=77 | affected_fraction=0.786 | mean_harmful_shap=0.0231 | description=Rule flag: TPSA < 79.
- tpsa_lt_70 | samples=72 | affected_fraction=0.735 | mean_harmful_shap=0.0099 | description=Rule flag: TPSA < 70.
- estimated_logd_ph74 | samples=64 | affected_fraction=0.653 | mean_harmful_shap=0.0253 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- oxygen_nitrogen_count | samples=63 | affected_fraction=0.643 | mean_harmful_shap=0.0270 | description=Count of O and N atoms.
- hbd | samples=56 | affected_fraction=0.571 | mean_harmful_shap=0.0212 | description=Hydrogen bond donor count.
- min_estate_index | samples=53 | affected_fraction=0.541 | mean_harmful_shap=0.0113 | description=Minimum E-State index.
- acid_deprotonated_fraction_ph74 | samples=52 | affected_fraction=0.531 | mean_harmful_shap=0.0526 | description=Estimated deprotonated fraction for the dominant acidic site.
- total_hbond_donors_acceptors | samples=52 | affected_fraction=0.531 | mean_harmful_shap=0.0187 | description=HBD + HBA total.

## Dropped Features

6 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka
- most_basic_pka_le_8 | most_basic_pka_le_8
- most_basic_pka_le_10 | most_basic_pka_le_10
- most_basic_pka_7p5_10p5 | most_basic_pka_7p5_10p5
- most_acidic_pka_ge_4 | most_acidic_pka_ge_4

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
