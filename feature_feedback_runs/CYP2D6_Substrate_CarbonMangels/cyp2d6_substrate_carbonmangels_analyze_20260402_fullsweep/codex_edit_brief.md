# Codex Edit Brief: CYP2D6_Substrate_CarbonMangels / cyp2d6_substrate_carbonmangels

- Analyze run id: `cyp2d6_substrate_carbonmangels_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__CYP2D6_Substrate_CarbonMangels/galactica-6.7b__cyp2d6_substrate_carbonmangels__synthesize__seed_0.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/CYP2D6_Substrate_CarbonMangels/cyp2d6_substrate_carbonmangels_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 61
- False positives: 18
- False negatives: 43

## Recurring Harmful Features

### false_positive

- net_charge_proxy_ph74 | samples=18 | affected_fraction=1.000 | mean_harmful_shap=0.0374 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- base_protonated_fraction_ph74 | samples=18 | affected_fraction=1.000 | mean_harmful_shap=0.0298 | description=Estimated protonated fraction for the dominant basic site at pH 7.4.
- tpsa | samples=18 | affected_fraction=1.000 | mean_harmful_shap=0.0173 | description=Topological polar surface area.
- positive_charge_present_ph74 | samples=18 | affected_fraction=1.000 | mean_harmful_shap=0.0162 | description=Rule flag: positive charge is expected at pH 7.4 from formal charge or dominant basic-site protonation.
- high_polarity_or_acidic_penalty | samples=18 | affected_fraction=1.000 | mean_harmful_shap=0.0120 | description=Heuristic penalty for high TPSA, excess heteroatom-driven polarity, or strong acidity.
- oxygen_nitrogen_count | samples=18 | affected_fraction=1.000 | mean_harmful_shap=0.0089 | description=Count of O and N atoms.
- cyp2d6_primary_rule_pass_count | samples=17 | affected_fraction=0.944 | mean_harmful_shap=0.0262 | description=Count of passed primary CYP2D6 heuristic checks across basicity, charge, lipophilicity, polarity, aromaticity, and soft-spot proxies.
- acid_deprotonated_fraction_ph74 | samples=17 | affected_fraction=0.944 | mean_harmful_shap=0.0261 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- oxidation_site_proxy_count | samples=16 | affected_fraction=0.889 | mean_harmful_shap=0.0130 | description=Number of unique benzylic, allylic, or alpha-heteroatom carbon atoms.
- alpha_heteroatom_ch_site_count | samples=16 | affected_fraction=0.889 | mean_harmful_shap=0.0115 | description=Count of carbons with at least one H and a directly adjacent N/O/P/S atom.

### false_negative

- net_charge_proxy_ph74 | samples=34 | affected_fraction=0.791 | mean_harmful_shap=0.0185 | description=Base protonation minus acid deprotonation proxy at pH 7.4.
- most_basic_pka_ge_8 | samples=33 | affected_fraction=0.767 | mean_harmful_shap=0.0087 | description=Rule flag: most basic predicted pKa >= 8.
- positive_charge_present_ph74 | samples=32 | affected_fraction=0.744 | mean_harmful_shap=0.0111 | description=Rule flag: positive charge is expected at pH 7.4 from formal charge or dominant basic-site protonation.
- base_protonated_fraction_ph74 | samples=30 | affected_fraction=0.698 | mean_harmful_shap=0.0196 | description=Estimated protonated fraction for the dominant basic site at pH 7.4.
- cyp2d6_primary_rule_pass_count | samples=29 | affected_fraction=0.674 | mean_harmful_shap=0.0173 | description=Count of passed primary CYP2D6 heuristic checks across basicity, charge, lipophilicity, polarity, aromaticity, and soft-spot proxies.
- cation_aromatic_pharmacophore_present | samples=26 | affected_fraction=0.605 | mean_harmful_shap=0.0100 | description=Heuristic flag: a likely cationic center co-occurs with aromaticity and a conservative 2D distance proxy to aromatic or oxidation regions.
- acid_deprotonated_fraction_ph74 | samples=18 | affected_fraction=0.419 | mean_harmful_shap=0.0212 | description=Estimated deprotonated fraction for the dominant acidic site at pH 7.4.
- high_polarity_or_acidic_penalty | samples=18 | affected_fraction=0.419 | mean_harmful_shap=0.0106 | description=Heuristic penalty for high TPSA, excess heteroatom-driven polarity, or strong acidity.
- tpsa | samples=17 | affected_fraction=0.395 | mean_harmful_shap=0.0172 | description=Topological polar surface area.
- oxidation_site_proxy_count | samples=16 | affected_fraction=0.372 | mean_harmful_shap=0.0152 | description=Number of unique benzylic, allylic, or alpha-heteroatom carbon atoms.

## Dropped Features

6 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka
- oxidation_site_min_heavy_degree | oxidation_site_min_heavy_degree
- oxidation_site_mean_heavy_degree | oxidation_site_mean_heavy_degree
- basic_to_oxidation_min_graph_distance | basic_to_oxidation_min_graph_distance
- basic_to_aromatic_min_graph_distance | basic_to_aromatic_min_graph_distance

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
