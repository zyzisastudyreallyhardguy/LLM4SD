# Codex Edit Brief: SARSCoV2_Vitro_Touret / sarscov2_vitro_touret

- Analyze run id: `sarscov2_vitro_touret_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__SARSCoV2_Vitro_Touret/galactica-6.7b__sarscov2_vitro_touret__synthesize__seed_3.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/SARSCoV2_Vitro_Touret/sarscov2_vitro_touret_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 175
- False positives: 159
- False negatives: 16

## Recurring Harmful Features

### false_positive

- logp | samples=136 | affected_fraction=0.855 | mean_harmful_shap=0.0157 | description=Wildman-Crippen logP.
- neutral_fraction_ph74 | samples=118 | affected_fraction=0.742 | mean_harmful_shap=0.0159 | description=Estimated neutral fraction at pH 7.4.
- charged_fraction_ph74 | samples=116 | affected_fraction=0.730 | mean_harmful_shap=0.0149 | description=1 - neutral_fraction_ph74.
- base_protonated_fraction_ph74 | samples=115 | affected_fraction=0.723 | mean_harmful_shap=0.0173 | description=Estimated protonated fraction for the dominant basic site.
- net_charge_proxy_ph74 | samples=111 | affected_fraction=0.698 | mean_harmful_shap=0.0170 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- estimated_logd_ph74 | samples=103 | affected_fraction=0.648 | mean_harmful_shap=0.0088 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- heteroatom_count | samples=102 | affected_fraction=0.642 | mean_harmful_shap=0.0111 | description=Total heteroatom count.
- stereocenter_count_ge_1 | samples=102 | affected_fraction=0.642 | mean_harmful_shap=0.0048 | description=Rule flag: at least one stereocenter is present.
- fraction_csp3 | samples=91 | affected_fraction=0.572 | mean_harmful_shap=0.0088 | description=Fraction of sp3 carbons.
- estimated_logd_in_1_3 | samples=90 | affected_fraction=0.566 | mean_harmful_shap=0.0088 | description=Rule flag: 1 <= estimated logD(7.4) <= 3.

### false_negative

- net_charge_proxy_ph74 | samples=15 | affected_fraction=0.938 | mean_harmful_shap=0.0084 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- base_protonated_fraction_ph74 | samples=13 | affected_fraction=0.812 | mean_harmful_shap=0.0110 | description=Estimated protonated fraction for the dominant basic site.
- rotatable_bonds | samples=12 | affected_fraction=0.750 | mean_harmful_shap=0.0061 | description=Rotatable bond count.
- heteroatom_count | samples=11 | affected_fraction=0.688 | mean_harmful_shap=0.0104 | description=Total heteroatom count.
- estimated_logd_in_1_3 | samples=11 | affected_fraction=0.688 | mean_harmful_shap=0.0048 | description=Rule flag: 1 <= estimated logD(7.4) <= 3.
- tpsa | samples=10 | affected_fraction=0.625 | mean_harmful_shap=0.0072 | description=Topological polar surface area.
- ring_count_4_7 | samples=9 | affected_fraction=0.562 | mean_harmful_shap=0.0035 | description=Rule flag: 4 <= total ring count <= 7.
- logp | samples=8 | affected_fraction=0.500 | mean_harmful_shap=0.0350 | description=Wildman-Crippen logP.
- fraction_csp3 | samples=8 | affected_fraction=0.500 | mean_harmful_shap=0.0144 | description=Fraction of sp3 carbons.
- metabolic_lability_alert_count | samples=8 | affected_fraction=0.500 | mean_harmful_shap=0.0031 | description=Combined count of simple metabolic-lability motifs such as esters and phenols.

## Dropped Features

2 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
