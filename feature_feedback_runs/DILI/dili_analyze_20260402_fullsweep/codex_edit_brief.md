# Codex Edit Brief: DILI / dili

- Analyze run id: `dili_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__DILI/galactica-6.7b__dili__synthesize__seed_2.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/DILI/dili_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 50
- False positives: 17
- False negatives: 33

## Recurring Harmful Features

### false_positive

- ring_count | samples=15 | affected_fraction=0.882 | mean_harmful_shap=0.0142 | description=Total ring count.
- fraction_csp3 | samples=13 | affected_fraction=0.765 | mean_harmful_shap=0.1209 | description=Fraction of sp3 carbons.
- aromatic_ring_count | samples=13 | affected_fraction=0.765 | mean_harmful_shap=0.0660 | description=Aromatic ring count.
- tpsa | samples=13 | affected_fraction=0.765 | mean_harmful_shap=0.0215 | description=Topological polar surface area.
- phenol_count | samples=13 | affected_fraction=0.765 | mean_harmful_shap=0.0099 | description=Count of phenol or phenoxide substituents on aromatic rings.
- heteroatom_count | samples=12 | affected_fraction=0.706 | mean_harmful_shap=0.1239 | description=Total heteroatom count.
- logp | samples=11 | affected_fraction=0.647 | mean_harmful_shap=0.0076 | description=Wildman-Crippen logP.
- halogen_atom_count | samples=11 | affected_fraction=0.647 | mean_harmful_shap=0.0018 | description=Total halogen atom count.
- rotatable_bonds | samples=10 | affected_fraction=0.588 | mean_harmful_shap=0.0113 | description=Rotatable bond count.
- pains_alert_present | samples=10 | affected_fraction=0.588 | mean_harmful_shap=0.0012 | description=1 if any RDKit PAINS alert was matched.

### false_negative

- aromatic_ring_count | samples=29 | affected_fraction=0.879 | mean_harmful_shap=0.0500 | description=Aromatic ring count.
- lipophilic_structural_alert_flag | samples=28 | affected_fraction=0.848 | mean_harmful_shap=0.0021 | description=1 if logP >= 3 and at least one reactive alert family is present.
- reactive_metabolite_alert_count | samples=25 | affected_fraction=0.758 | mean_harmful_shap=0.0057 | description=Number of distinct reactive-metabolite-oriented alert families matched.
- fraction_csp3 | samples=24 | affected_fraction=0.727 | mean_harmful_shap=0.0929 | description=Fraction of sp3 carbons.
- structural_alert_count | samples=24 | affected_fraction=0.727 | mean_harmful_shap=0.0312 | description=Number of distinct DILI structural alert families matched.
- hba | samples=24 | affected_fraction=0.727 | mean_harmful_shap=0.0062 | description=Hydrogen bond acceptor count.
- heteroatom_count | samples=23 | affected_fraction=0.697 | mean_harmful_shap=0.0754 | description=Total heteroatom count.
- ring_count | samples=20 | affected_fraction=0.606 | mean_harmful_shap=0.0289 | description=Total ring count.
- molar_refractivity | samples=16 | affected_fraction=0.485 | mean_harmful_shap=0.0101 | description=Wildman-Crippen molar refractivity.
- logp | samples=15 | affected_fraction=0.455 | mean_harmful_shap=0.0164 | description=Wildman-Crippen logP.

## Dropped Features

No features were dropped by the train-time preprocessing step.

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
