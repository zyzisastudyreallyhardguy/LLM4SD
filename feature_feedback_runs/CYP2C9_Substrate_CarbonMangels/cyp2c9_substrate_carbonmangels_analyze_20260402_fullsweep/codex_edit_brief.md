# Codex Edit Brief: CYP2C9_Substrate_CarbonMangels / cyp2c9_substrate_carbonmangels

- Analyze run id: `cyp2c9_substrate_carbonmangels_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__CYP2C9_Substrate_CarbonMangels/galactica-6.7b__cyp2c9_substrate_carbonmangels__synthesize__seed_1.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/CYP2C9_Substrate_CarbonMangels/cyp2c9_substrate_carbonmangels_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 11
- False positives: 10
- False negatives: 1

## Recurring Harmful Features

### false_positive

- net_charge_proxy_ph74 | samples=8 | affected_fraction=0.800 | mean_harmful_shap=0.0459 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- acid_deprotonated_fraction_ph74 | samples=8 | affected_fraction=0.800 | mean_harmful_shap=0.0238 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- base_protonated_fraction_le_0p5 | samples=8 | affected_fraction=0.800 | mean_harmful_shap=0.0085 | description=Rule flag: dominant basic site is at most half protonated at pH 7.4.
- base_protonated_fraction_ph74 | samples=6 | affected_fraction=0.600 | mean_harmful_shap=0.0239 | description=Estimated protonated fraction for the dominant basic site at pH 7.4.
- charged_fraction_ph74 | samples=6 | affected_fraction=0.600 | mean_harmful_shap=0.0137 | description=1 - neutral_fraction_ph74.
- acid_deprotonated_fraction_ge_0p5 | samples=5 | affected_fraction=0.500 | mean_harmful_shap=0.0144 | description=Rule flag: dominant acidic site is at least half deprotonated at pH 7.4.
- neutral_fraction_ph74 | samples=5 | affected_fraction=0.500 | mean_harmful_shap=0.0128 | description=Estimated neutral fraction at pH 7.4.
- fraction_csp3 | samples=5 | affected_fraction=0.500 | mean_harmful_shap=0.0095 | description=Fraction of sp3 carbons.
- aromatic_ring_count_ge_2 | samples=5 | affected_fraction=0.500 | mean_harmful_shap=0.0079 | description=Rule flag: aromatic ring count >= 2.
- allylic_ch_site_count | samples=5 | affected_fraction=0.500 | mean_harmful_shap=0.0060 | description=Count of non-aromatic carbons with at least one H adjacent to an unsaturated carbon.

### false_negative

- base_protonated_fraction_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0301 | description=Estimated protonated fraction for the dominant basic site at pH 7.4.
- net_charge_proxy_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0247 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- neutral_fraction_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0083 | description=Estimated neutral fraction at pH 7.4.
- alpha_heteroatom_ch_site_count | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0079 | description=Count of carbons with at least one H and a directly adjacent N/O/P/S atom.
- exact_mol_weight | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0058 | description=Exact molecular weight (Descriptors.ExactMolWt).
- charged_fraction_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0057 | description=1 - neutral_fraction_ph74.
- base_protonated_fraction_le_0p5 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0054 | description=Rule flag: dominant basic site is at most half protonated at pH 7.4.
- cyp2c9_primary_rule_pass_count | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0052 | description=Count of passed primary CYP2C9 heuristic checks among size, lipophilicity, polarity, acidity, aromaticity, and oxidation-site proxies.
- aromatic_heterocycle_ring_count | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0047 | description=Count of aromatic rings containing at least one heteroatom.
- acid_deprotonated_fraction_ph74 | samples=1 | affected_fraction=1.000 | mean_harmful_shap=0.0040 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.

## Dropped Features

9 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka
- oxidation_site_min_heavy_degree | oxidation_site_min_heavy_degree
- oxidation_site_mean_heavy_degree | oxidation_site_mean_heavy_degree
- acidic_to_oxidation_min_graph_distance | acidic_to_oxidation_min_graph_distance
- acidic_to_oxidation_distance_4_8 | acidic_to_oxidation_distance_4_8
- most_acidic_pka_in_3_8p5 | most_acidic_pka_in_3_8p5
- weak_acidic_lipophilic_profile | weak_acidic_lipophilic_profile
- cyp2c9_substrate_heuristic_pass | cyp2c9_substrate_heuristic_pass

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
