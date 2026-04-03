# Codex Edit Brief: AMES / ames

- Analyze run id: `ames_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__AMES/galactica-6.7b__ames__synthesize__seed_2.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/AMES/ames_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 478
- False positives: 222
- False negatives: 256

## Recurring Harmful Features

### false_positive

- ames_structural_alert_present | samples=202 | affected_fraction=0.910 | mean_harmful_shap=0.0352 | description=1 if any DeepResearch Ames structural alert is present.
- ames_alert_family_count | samples=201 | affected_fraction=0.905 | mean_harmful_shap=0.0324 | description=Count of distinct DeepResearch Ames structural alert families matched.
- aromatic_alert_family_count | samples=147 | affected_fraction=0.662 | mean_harmful_shap=0.0334 | description=Count of aromatic or polycyclic Ames alert families matched.
- aromatic_ring_count | samples=122 | affected_fraction=0.550 | mean_harmful_shap=0.0269 | description=Aromatic ring count.
- fraction_csp3 | samples=115 | affected_fraction=0.518 | mean_harmful_shap=0.0150 | description=Fraction of sp3 carbons.
- ring_count | samples=96 | affected_fraction=0.432 | mean_harmful_shap=0.0269 | description=Total ring count.
- michael_acceptor_count | samples=95 | affected_fraction=0.428 | mean_harmful_shap=0.0084 | description=Count of alpha,beta-unsaturated carbonyl alerts.
- hba | samples=81 | affected_fraction=0.365 | mean_harmful_shap=0.0113 | description=Hydrogen bond acceptor count.
- mol_weight | samples=76 | affected_fraction=0.342 | mean_harmful_shap=0.0099 | description=Molecular weight (Descriptors.MolWt).
- negative_formal_atom_count | samples=67 | affected_fraction=0.302 | mean_harmful_shap=0.0424 | description=Number of atoms with negative formal charge.

### false_negative

- aromatic_alert_family_count | samples=233 | affected_fraction=0.910 | mean_harmful_shap=0.0160 | description=Count of aromatic or polycyclic Ames alert families matched.
- negative_formal_atom_count | samples=223 | affected_fraction=0.871 | mean_harmful_shap=0.0092 | description=Number of atoms with negative formal charge.
- reactive_alert_family_count | samples=205 | affected_fraction=0.801 | mean_harmful_shap=0.0115 | description=Count of reactive Ames alert families matched.
- ames_structural_alert_present | samples=195 | affected_fraction=0.762 | mean_harmful_shap=0.0355 | description=1 if any DeepResearch Ames structural alert is present.
- ames_alert_family_count | samples=195 | affected_fraction=0.762 | mean_harmful_shap=0.0342 | description=Count of distinct DeepResearch Ames structural alert families matched.
- positive_formal_atom_count | samples=164 | affected_fraction=0.641 | mean_harmful_shap=0.0083 | description=Number of atoms with positive formal charge.
- ring_count | samples=150 | affected_fraction=0.586 | mean_harmful_shap=0.0104 | description=Total ring count.
- polycyclic_alert_present | samples=149 | affected_fraction=0.582 | mean_harmful_shap=0.0061 | description=1 if a PAH or hetero-PAH alert is present.
- max_fused_aromatic_system_size | samples=134 | affected_fraction=0.523 | mean_harmful_shap=0.0100 | description=Largest fused aromatic system size measured in aromatic rings.
- aromatic_ring_count | samples=126 | affected_fraction=0.492 | mean_harmful_shap=0.0139 | description=Aromatic ring count.

## Dropped Features

No features were dropped by the train-time preprocessing step.

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
