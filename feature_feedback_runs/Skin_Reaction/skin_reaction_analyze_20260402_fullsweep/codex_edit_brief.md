# Codex Edit Brief: Skin_Reaction / skin_reaction

- Analyze run id: `skin_reaction_analyze_20260402_fullsweep`
- Reference bundle: `checkpoints/forest/TDC__Skin_Reaction/galactica-6.7b__skin_reaction__synthesize__seed_4.joblib`
- SHAP evidence file: `/data1/tianang/Projects/LLM4SD/feature_feedback_runs/Skin_Reaction/skin_reaction_analyze_20260402_fullsweep/train_shap_evidence.jsonl`
- This brief is train-only. Do not use valid-set errors, valid SHAP, or valid single-sample patterns in feature-edit proposals.
- Edit only a versioned backend copy under `codex_generated_code_variants/`; do not modify `codex_generated_code/`.
- A candidate is acceptable only if train macro F1 improves, valid macro F1 improves, and valid ROC-AUC does not decrease.

## Persistent Train Errors

- Persistent train errors: 35
- False positives: 29
- False negatives: 6

## Recurring Harmful Features

### false_positive

- chiral_center_count | samples=22 | affected_fraction=0.759 | mean_harmful_shap=0.0030 | description=Count of assigned or potential tetrahedral stereocenters.
- tpsa | samples=21 | affected_fraction=0.724 | mean_harmful_shap=0.0722 | description=Topological polar surface area.
- aromatic_halogen_count | samples=16 | affected_fraction=0.552 | mean_harmful_shap=0.0023 | description=Number of halogens directly attached to aromatic atoms.
- ring_count | samples=15 | affected_fraction=0.517 | mean_harmful_shap=0.0164 | description=Total ring count.
- heteroatom_count | samples=15 | affected_fraction=0.517 | mean_harmful_shap=0.0159 | description=Total heteroatom count.
- heavy_atom_count | samples=15 | affected_fraction=0.517 | mean_harmful_shap=0.0158 | description=Heavy atom count.
- aromatic_ring_count | samples=15 | affected_fraction=0.517 | mean_harmful_shap=0.0032 | description=Aromatic ring count.
- halogen_atom_count | samples=13 | affected_fraction=0.448 | mean_harmful_shap=0.0046 | description=Total halogen atom count.
- overall_alert_family_count | samples=12 | affected_fraction=0.414 | mean_harmful_shap=0.0048 | description=Number of distinct skin-sensitization alert families present.
- molar_refractivity | samples=10 | affected_fraction=0.345 | mean_harmful_shap=0.0143 | description=Wildman-Crippen molar refractivity.

### false_negative

- mol_weight | samples=6 | affected_fraction=1.000 | mean_harmful_shap=0.0080 | description=Molecular weight (Descriptors.MolWt).
- tpsa | samples=5 | affected_fraction=0.833 | mean_harmful_shap=0.2018 | description=Topological polar surface area.
- fraction_csp3 | samples=5 | affected_fraction=0.833 | mean_harmful_shap=0.0119 | description=Fraction of sp3 carbons.
- direct_electrophile_site_count | samples=5 | affected_fraction=0.833 | mean_harmful_shap=0.0047 | description=Combined count of direct electrophilic or acylating alert sites.
- molar_refractivity | samples=4 | affected_fraction=0.667 | mean_harmful_shap=0.0435 | description=Wildman-Crippen molar refractivity.
- heteroatom_count | samples=4 | affected_fraction=0.667 | mean_harmful_shap=0.0245 | description=Total heteroatom count.
- ring_count | samples=4 | affected_fraction=0.667 | mean_harmful_shap=0.0215 | description=Total ring count.
- rotatable_bonds | samples=4 | affected_fraction=0.667 | mean_harmful_shap=0.0074 | description=Rotatable bond count.
- aromatic_amine_count | samples=4 | affected_fraction=0.667 | mean_harmful_shap=0.0050 | description=Count of aniline-like aromatic amine substituents.
- multiple_reactive_sites_flag | samples=4 | affected_fraction=0.667 | mean_harmful_shap=0.0029 | description=1 if two or more direct electrophilic sites are present.

## Dropped Features

No features were dropped by the train-time preprocessing step.

## Editing Guidance

- Prefer changes that address recurring train-side patterns across multiple persistent errors.
- Do not justify an edit using any valid-set sample behavior.
- If you add, remove, or modify features, record the work in a new feedback variant package and keep the interface compatible with the original backend.
