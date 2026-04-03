# Codex Edit Brief: Bioavailability_Ma / bioavailability_ma

- Analyze run id: `bioavailability_ma_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__Bioavailability_Ma/galactica-6.7b__bioavailability_ma__synthesize__seed_0.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/Bioavailability_Ma/bioavailability_ma_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 58
- False positives: 16
- False negatives: 42

## Recurring Harmful Features

### false_positive

- stereocenter_count | samples=16 | affected_fraction=1.000 | mean_harmful_shap=0.0260 | description=Count of assigned or potential tetrahedral stereocenters.
- total_atom_count | samples=14 | affected_fraction=0.875 | mean_harmful_shap=0.0286 | description=Total atom count after adding explicit hydrogens as a Ghose-style atom-count proxy.
- ghose_mw_in_160_480 | samples=14 | affected_fraction=0.875 | mean_harmful_shap=0.0077 | description=Rule flag: 160 < MW < 480.
- labute_asa | samples=11 | affected_fraction=0.688 | mean_harmful_shap=0.0117 | description=Labute approximate surface area.
- hbd | samples=11 | affected_fraction=0.688 | mean_harmful_shap=0.0060 | description=Hydrogen bond donor count.
- estimated_logd_ph74 | samples=10 | affected_fraction=0.625 | mean_harmful_shap=0.0253 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- intramolecular_hbond_pair_count | samples=10 | affected_fraction=0.625 | mean_harmful_shap=0.0055 | description=Count of donor-acceptor atom pairs 4-8 bonds apart as a proxy for intramolecular H-bond potential.
- oral_bioavailability_rule_pass_count | samples=9 | affected_fraction=0.562 | mean_harmful_shap=0.0083 | description=Count of passed composite rules among Lipinski, Veber, Egan, Ghose, Oprea, Rule-of-3, Muegge, and Pfizer 3/75-safe.
- aliphatic_ring_count | samples=9 | affected_fraction=0.562 | mean_harmful_shap=0.0066 | description=Aliphatic ring count.
- fraction_csp3 | samples=8 | affected_fraction=0.500 | mean_harmful_shap=0.0320 | description=Fraction of sp3 carbons.

### false_negative

- fraction_csp3 | samples=33 | affected_fraction=0.786 | mean_harmful_shap=0.0131 | description=Fraction of sp3 carbons.
- ghose_pass | samples=32 | affected_fraction=0.762 | mean_harmful_shap=0.0133 | description=Composite Ghose filter pass flag.
- estimated_logd_ph74 | samples=25 | affected_fraction=0.595 | mean_harmful_shap=0.0198 | description=Estimated logD at pH 7.4 from logP and neutral fraction.
- oral_bioavailability_rule_pass_count | samples=25 | affected_fraction=0.595 | mean_harmful_shap=0.0121 | description=Count of passed composite rules among Lipinski, Veber, Egan, Ghose, Oprea, Rule-of-3, Muegge, and Pfizer 3/75-safe.
- total_atom_count | samples=24 | affected_fraction=0.571 | mean_harmful_shap=0.0659 | description=Total atom count after adding explicit hydrogens as a Ghose-style atom-count proxy.
- stereocenter_count | samples=24 | affected_fraction=0.571 | mean_harmful_shap=0.0578 | description=Count of assigned or potential tetrahedral stereocenters.
- labute_asa | samples=21 | affected_fraction=0.500 | mean_harmful_shap=0.0167 | description=Labute approximate surface area.
- aromatic_heavy_atom_fraction | samples=21 | affected_fraction=0.500 | mean_harmful_shap=0.0111 | description=Aromatic atom fraction among heavy atoms.
- hbd | samples=20 | affected_fraction=0.476 | mean_harmful_shap=0.0162 | description=Hydrogen bond donor count.
- ghose_mw_in_160_480 | samples=18 | affected_fraction=0.429 | mean_harmful_shap=0.0261 | description=Rule flag: 160 < MW < 480.

## Dropped Features

2 features never survived the train preprocessor because their training columns contained NaN.
- most_basic_pka | most_basic_pka
- most_acidic_pka | most_acidic_pka

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
