#!/usr/bin/env python3
"""Deterministic HIA_Hou rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch HIA_Hou ruleset.

Implemented feature groups:
- core HIA-related physicochemical descriptors behind Lipinski, Veber, Egan,
  Ghose, Palm, and related oral-absorption screens
- aromatic ring count and intramolecular H-bond topology proxies for the
  non-threshold guidance in the source text
- pKa/logD-derived neutral-fraction proxies across intestinal pH values using
  the shared MolGpKa helper already used elsewhere in this repository

Intentionally skipped:
- dynamic 3D PSA because this repository does not ship a validated conformer
  ensemble workflow for Palm's dynamic PSA rule
- experimental or external-model aqueous solubility estimates
- any hard aromatic-ring or intramolecular-H-bond thresholds not stated in the
  source response
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

INTERN_S1_ROOT = Path("/data1/tianang/Projects/Intern-S1")
PHYSIOLOGICAL_PH = 7.4
INTESTINAL_PH_VALUES = {
    "ph65": 6.5,
    "ph74": PHYSIOLOGICAL_PH,
    "ph80": 8.0,
}


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("total_atom_count", "Total atom count after adding explicit hydrogens as a Ghose-style atom-count proxy."),
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count as a continuous proxy for lower aromatic burden."),
        (
            "intramolecular_hbond_pair_count",
            "Count of donor-acceptor atom pairs 4-8 bonds apart as a proxy for intramolecular H-bond potential.",
        ),
        ("intramolecular_hbond_potential_present", "1 if the intramolecular H-bond proxy count is non-zero."),
        ("most_basic_pka", "Most basic predicted pKa from the MolGpKa helper."),
        ("most_basic_pka_present", "1 if at least one basic pKa site was predicted."),
        ("most_acidic_pka", "Most acidic predicted pKa from the MolGpKa helper."),
        ("most_acidic_pka_present", "1 if at least one acidic pKa site was predicted."),
        ("num_basic_sites", "Number of predicted basic sites from the MolGpKa helper."),
        ("num_acidic_sites", "Number of predicted acidic sites from the MolGpKa helper."),
        ("neutral_fraction_ph65", "Estimated neutral fraction at pH 6.5."),
        ("neutral_fraction_ph74", "Estimated neutral fraction at pH 7.4."),
        ("neutral_fraction_ph80", "Estimated neutral fraction at pH 8.0."),
        ("charged_fraction_ph65", "1 - neutral_fraction_ph65."),
        ("charged_fraction_ph74", "1 - neutral_fraction_ph74."),
        ("charged_fraction_ph80", "1 - neutral_fraction_ph80."),
        ("base_protonated_fraction_ph65", "Estimated protonated fraction for the dominant basic site at pH 6.5."),
        ("base_protonated_fraction_ph74", "Estimated protonated fraction for the dominant basic site at pH 7.4."),
        ("base_protonated_fraction_ph80", "Estimated protonated fraction for the dominant basic site at pH 8.0."),
        ("acid_deprotonated_fraction_ph65", "Estimated deprotonated fraction for the dominant acidic site at pH 6.5."),
        ("acid_deprotonated_fraction_ph74", "Estimated deprotonated fraction for the dominant acidic site at pH 7.4."),
        ("acid_deprotonated_fraction_ph80", "Estimated deprotonated fraction for the dominant acidic site at pH 8.0."),
        ("net_charge_proxy_ph65", "Base protonation minus acid deprotonation proxy at pH 6.5."),
        ("net_charge_proxy_ph74", "Base protonation minus acid deprotonation proxy at pH 7.4."),
        ("net_charge_proxy_ph80", "Base protonation minus acid deprotonation proxy at pH 8.0."),
        ("estimated_logd_ph65", "Estimated logD at pH 6.5 from logP and neutral fraction."),
        ("estimated_logd_ph74", "Estimated logD at pH 7.4 from logP and neutral fraction."),
        ("estimated_logd_ph80", "Estimated logD at pH 8.0 from logP and neutral fraction."),
        ("min_neutral_fraction_intestinal_ph", "Minimum estimated neutral fraction across pH 6.5, 7.4, and 8.0."),
        ("mean_neutral_fraction_intestinal_ph", "Mean estimated neutral fraction across pH 6.5, 7.4, and 8.0."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
        ("mw_le_500", "Rule flag: MW <= 500."),
        ("logp_le_5", "Rule flag: logP <= 5."),
        ("hbd_le_5", "Rule flag: HBD <= 5."),
        ("hba_le_10", "Rule flag: HBA <= 10."),
        ("lipinski_pass", "Composite Lipinski Rule-of-5 pass flag."),
        ("lipinski_violation_count", "Number of Lipinski cutoffs violated among MW/logP/HBD/HBA."),
        ("rotatable_bonds_le_10", "Rule flag: rotatable bonds <= 10."),
        ("tpsa_le_140", "Rule flag: TPSA <= 140."),
        ("veber_pass", "Composite Veber pass flag."),
        ("logp_le_5p88", "Rule flag: logP <= 5.88."),
        ("tpsa_le_131p6", "Rule flag: TPSA <= 131.6."),
        ("egan_pass", "Composite Egan pass flag."),
        ("tpsa_le_63", "Rule flag: TPSA <= 63."),
        ("palm_static_pass", "Static Palm-style pass flag using TPSA <= 63."),
        ("ghose_mw_in_160_480", "Rule flag: 160 < MW < 480."),
        ("logp_ge_neg0p4", "Rule flag: logP >= -0.4."),
        ("logp_le_5p6", "Rule flag: logP <= 5.6."),
        ("ghose_logp_in_neg0p4_5p6", "Rule flag: -0.4 < logP < 5.6."),
        ("ghose_atom_count_in_20_70", "Rule flag: 20 < atom count < 70."),
        ("ghose_mr_in_40_130", "Rule flag: 40 < molar refractivity < 130."),
        ("ghose_pass", "Composite Ghose filter pass flag."),
        ("total_hbond_donors_acceptors_le_12", "Rule flag: HBD + HBA <= 12."),
        ("veber_alt_pass", "Veber alternative pass flag using HBD + HBA <= 12."),
        (
            "unionized_ph65_ge_0p5",
            "Heuristic flag: estimated neutral fraction >= 0.5 at pH 6.5 as a 'predominantly unionized' proxy.",
        ),
        (
            "unionized_ph74_ge_0p5",
            "Heuristic flag: estimated neutral fraction >= 0.5 at pH 7.4 as a 'predominantly unionized' proxy.",
        ),
        (
            "unionized_ph80_ge_0p5",
            "Heuristic flag: estimated neutral fraction >= 0.5 at pH 8.0 as a 'predominantly unionized' proxy.",
        ),
        (
            "unionized_across_intestinal_ph",
            "Heuristic flag: estimated neutral fraction >= 0.5 across pH 6.5, 7.4, and 8.0.",
        ),
        (
            "hia_primary_rule_pass_count",
            "Count of passed HIA composite rules among Lipinski, Veber, Egan, Palm static, Ghose, and Veber alternative.",
        ),
        (
            "hia_strict_absorption_screen",
            "Combined heuristic screen requiring Lipinski, Veber, Egan, Palm static, and Veber alternative passes.",
        ),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Palm's dynamic 3D PSA <= 60 rule is not hard-coded because this repository does not provide a validated conformer-based PSA workflow.",
    "High aqueous solubility is not hard-coded because the source response does not specify a threshold and the repository does not ship a validated local solubility predictor.",
    "Lower aromatic ring count and intramolecular H-bond guidance are represented as continuous topology proxies rather than unsupported hard thresholds.",
]


def get_feature_names() -> list[str]:
    return list(FEATURE_DESCRIPTIONS.keys())


def get_feature_descriptions() -> dict[str, str]:
    return dict(FEATURE_DESCRIPTIONS)


def get_skipped_rule_groups() -> list[str]:
    return list(SKIPPED_RULE_GROUPS)


def _empty_feature_template() -> OrderedDict[str, float]:
    return OrderedDict((name, math.nan) for name in FEATURE_DESCRIPTIONS)


def _as_float(value: float | int) -> float:
    return float(value)


def _flag(condition: bool | None) -> float:
    if condition is None:
        return math.nan
    return float(bool(condition))


def _is_missing(value: float | None) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def _maybe_le(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value <= threshold)


def _maybe_ge(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value >= threshold)


def _maybe_between(value: float | None, lower: float, upper: float, inclusive: bool = True) -> float:
    if _is_missing(value):
        return math.nan
    if inclusive:
        return _flag(lower <= value <= upper)
    return _flag(lower < value < upper)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem, RDConfig
        from rdkit.Chem import ChemicalFeatures, Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run HIA_Hou feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        RDConfig=RDConfig,
        ChemicalFeatures=ChemicalFeatures,
        Crippen=Crippen,
        Descriptors=Descriptors,
        Lipinski=Lipinski,
        rdMolDescriptors=rdMolDescriptors,
    )


@lru_cache(maxsize=1)
def _feature_factory():
    rdkit = _rdkit()
    feature_path = Path(rdkit.RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return rdkit.ChemicalFeatures.BuildFeatureFactory(str(feature_path))


@lru_cache(maxsize=1)
def _get_pka_predictor():
    if str(INTERN_S1_ROOT) not in sys.path:
        sys.path.insert(0, str(INTERN_S1_ROOT))

    from tools.pka_related_tools import _get_pka_predictor as _load_predictor

    return _load_predictor()


def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    rdkit = _rdkit()
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _count_total_atoms(mol) -> int:
    rdkit = _rdkit()
    return rdkit.Chem.AddHs(mol).GetNumAtoms()


def _count_intramolecular_hbond_pairs(mol, min_distance: int = 4, max_distance: int = 8) -> int:
    try:
        factory = _feature_factory()
    except Exception:
        return 0

    donor_atoms: set[int] = set()
    acceptor_atoms: set[int] = set()
    for feature in factory.GetFeaturesForMol(mol):
        atom_ids = feature.GetAtomIds()
        if feature.GetFamily() == "Donor":
            donor_atoms.update(atom_ids)
        elif feature.GetFamily() == "Acceptor":
            acceptor_atoms.update(atom_ids)

    if not donor_atoms or not acceptor_atoms:
        return 0

    rdkit = _rdkit()
    unique_pairs: set[tuple[int, int]] = set()
    for donor_idx in donor_atoms:
        for acceptor_idx in acceptor_atoms:
            if donor_idx == acceptor_idx:
                continue
            path = rdkit.Chem.GetShortestPath(mol, donor_idx, acceptor_idx)
            bond_distance = len(path) - 1
            if min_distance <= bond_distance <= max_distance:
                unique_pairs.add(tuple(sorted((donor_idx, acceptor_idx))))
    return len(unique_pairs)


def _compute_pka_summary(mol, logp: float) -> dict[str, float]:
    result = {
        "most_basic_pka": 0.0,
        "most_basic_pka_present": 0.0,
        "most_acidic_pka": 14.0,
        "most_acidic_pka_present": 0.0,
        "num_basic_sites": 0.0,
        "num_acidic_sites": 0.0,
        "neutral_fraction_ph65": math.nan,
        "neutral_fraction_ph74": math.nan,
        "neutral_fraction_ph80": math.nan,
        "charged_fraction_ph65": math.nan,
        "charged_fraction_ph74": math.nan,
        "charged_fraction_ph80": math.nan,
        "base_protonated_fraction_ph65": math.nan,
        "base_protonated_fraction_ph74": math.nan,
        "base_protonated_fraction_ph80": math.nan,
        "acid_deprotonated_fraction_ph65": math.nan,
        "acid_deprotonated_fraction_ph74": math.nan,
        "acid_deprotonated_fraction_ph80": math.nan,
        "net_charge_proxy_ph65": math.nan,
        "net_charge_proxy_ph74": math.nan,
        "net_charge_proxy_ph80": math.nan,
        "estimated_logd_ph65": math.nan,
        "estimated_logd_ph74": math.nan,
        "estimated_logd_ph80": math.nan,
        "min_neutral_fraction_intestinal_ph": math.nan,
        "mean_neutral_fraction_intestinal_ph": math.nan,
        "pka_features_available": 0.0,
    }

    try:
        predictor = _get_pka_predictor()
        prediction = predictor.predict(mol)
    except Exception:
        return result

    base_sites = getattr(prediction, "base_sites_1", {}) or {}
    acid_sites = getattr(prediction, "acid_sites_1", {}) or {}

    most_basic_pka = max(base_sites.values()) if base_sites else math.nan
    most_acidic_pka = min(acid_sites.values()) if acid_sites else math.nan
    result["most_basic_pka"] = _as_float(most_basic_pka) if base_sites else 0.0
    result["most_basic_pka_present"] = float(bool(base_sites))
    result["most_acidic_pka"] = _as_float(most_acidic_pka) if acid_sites else 14.0
    result["most_acidic_pka_present"] = float(bool(acid_sites))
    result["num_basic_sites"] = float(len(base_sites))
    result["num_acidic_sites"] = float(len(acid_sites))

    neutral_fractions: list[float] = []
    for label, ph_value in INTESTINAL_PH_VALUES.items():
        base_protonated_fraction = 0.0
        if base_sites:
            base_protonated_fraction = 1.0 / (1.0 + 10.0 ** (ph_value - most_basic_pka))

        acid_deprotonated_fraction = 0.0
        if acid_sites:
            acid_deprotonated_fraction = 1.0 / (1.0 + 10.0 ** (most_acidic_pka - ph_value))

        neutral_fraction = 1.0
        if base_sites:
            neutral_fraction *= 1.0 - base_protonated_fraction
        if acid_sites:
            neutral_fraction *= 1.0 - acid_deprotonated_fraction
        neutral_fraction = min(1.0, max(1e-12, neutral_fraction))

        neutral_fractions.append(neutral_fraction)
        result[f"neutral_fraction_{label}"] = neutral_fraction
        result[f"charged_fraction_{label}"] = 1.0 - neutral_fraction
        result[f"base_protonated_fraction_{label}"] = base_protonated_fraction
        result[f"acid_deprotonated_fraction_{label}"] = acid_deprotonated_fraction
        result[f"net_charge_proxy_{label}"] = base_protonated_fraction - acid_deprotonated_fraction
        result[f"estimated_logd_{label}"] = logp + math.log10(neutral_fraction)

    result["min_neutral_fraction_intestinal_ph"] = min(neutral_fractions)
    result["mean_neutral_fraction_intestinal_ph"] = sum(neutral_fractions) / len(neutral_fractions)
    result["pka_features_available"] = 1.0
    return result


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    total_atom_count = _as_float(_count_total_atoms(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    molar_refractivity = _as_float(rdkit.Crippen.MolMR(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    total_hbonds = hbd + hba
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    intramolecular_hbond_pair_count = _as_float(_count_intramolecular_hbond_pairs(mol))

    features.update(
        {
            "mol_weight": mol_weight,
            "total_atom_count": total_atom_count,
            "logp": logp,
            "molar_refractivity": molar_refractivity,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbonds,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "intramolecular_hbond_pair_count": intramolecular_hbond_pair_count,
            "intramolecular_hbond_potential_present": _flag(intramolecular_hbond_pair_count > 0.0),
        }
    )

    features.update(_compute_pka_summary(mol, logp))

    lipinski_pass = mol_weight <= 500.0 and logp <= 5.0 and hbd <= 5.0 and hba <= 10.0
    lipinski_violation_count = float(
        int(mol_weight > 500.0)
        + int(logp > 5.0)
        + int(hbd > 5.0)
        + int(hba > 10.0)
    )
    veber_pass = rotatable_bonds <= 10.0 and tpsa <= 140.0
    egan_pass = logp <= 5.88 and tpsa <= 131.6
    palm_static_pass = tpsa <= 63.0
    ghose_pass = (
        160.0 < mol_weight < 480.0
        and -0.4 < logp < 5.6
        and 20.0 < total_atom_count < 70.0
        and 40.0 < molar_refractivity < 130.0
    )
    veber_alt_pass = total_hbonds <= 12.0

    min_neutral_fraction = features["min_neutral_fraction_intestinal_ph"]
    hia_primary_rule_pass_count = float(
        sum(
            (
                lipinski_pass,
                veber_pass,
                egan_pass,
                palm_static_pass,
                ghose_pass,
                veber_alt_pass,
            )
        )
    )
    hia_strict_absorption_screen = (
        lipinski_pass
        and veber_pass
        and egan_pass
        and palm_static_pass
        and veber_alt_pass
    )

    features.update(
        {
            "mw_le_500": _maybe_le(mol_weight, 500.0),
            "logp_le_5": _maybe_le(logp, 5.0),
            "hbd_le_5": _maybe_le(hbd, 5.0),
            "hba_le_10": _maybe_le(hba, 10.0),
            "lipinski_pass": _flag(lipinski_pass),
            "lipinski_violation_count": lipinski_violation_count,
            "rotatable_bonds_le_10": _maybe_le(rotatable_bonds, 10.0),
            "tpsa_le_140": _maybe_le(tpsa, 140.0),
            "veber_pass": _flag(veber_pass),
            "logp_le_5p88": _maybe_le(logp, 5.88),
            "tpsa_le_131p6": _maybe_le(tpsa, 131.6),
            "egan_pass": _flag(egan_pass),
            "tpsa_le_63": _maybe_le(tpsa, 63.0),
            "palm_static_pass": _flag(palm_static_pass),
            "ghose_mw_in_160_480": _maybe_between(mol_weight, 160.0, 480.0, inclusive=False),
            "logp_ge_neg0p4": _maybe_ge(logp, -0.4),
            "logp_le_5p6": _maybe_le(logp, 5.6),
            "ghose_logp_in_neg0p4_5p6": _maybe_between(logp, -0.4, 5.6, inclusive=False),
            "ghose_atom_count_in_20_70": _maybe_between(total_atom_count, 20.0, 70.0, inclusive=False),
            "ghose_mr_in_40_130": _maybe_between(molar_refractivity, 40.0, 130.0, inclusive=False),
            "ghose_pass": _flag(ghose_pass),
            "total_hbond_donors_acceptors_le_12": _maybe_le(total_hbonds, 12.0),
            "veber_alt_pass": _flag(veber_alt_pass),
            "unionized_ph65_ge_0p5": _maybe_ge(features["neutral_fraction_ph65"], 0.5),
            "unionized_ph74_ge_0p5": _maybe_ge(features["neutral_fraction_ph74"], 0.5),
            "unionized_ph80_ge_0p5": _maybe_ge(features["neutral_fraction_ph80"], 0.5),
            "unionized_across_intestinal_ph": _maybe_ge(min_neutral_fraction, 0.5),
            "hia_primary_rule_pass_count": hia_primary_rule_pass_count,
            "hia_strict_absorption_screen": _flag(hia_strict_absorption_screen),
        }
    )

    return dict(features)


def featurize_smiles_list(
    smiles_list: Iterable[str],
    *,
    include_smiles: bool = False,
    on_error: str = "raise",
) -> "pd.DataFrame":
    import pandas as pd

    rows: list[dict[str, float | str]] = []

    for smiles in smiles_list:
        try:
            row: dict[str, float | str] = featurize_smiles(smiles)
        except Exception:
            if on_error == "raise":
                raise
            if on_error != "nan":
                raise ValueError("on_error must be either 'raise' or 'nan'")
            row = dict(_empty_feature_template())
            row["pka_features_available"] = 0.0
        if include_smiles:
            row = {"smiles": smiles, **row}
        rows.append(row)

    return pd.DataFrame(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate HIA_Hou DeepResearch features from SMILES."
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default="",
        help="Single SMILES string to featurize.",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="",
        help="CSV file containing a SMILES column for batch featurization.",
    )
    parser.add_argument(
        "--smiles_col",
        type=str,
        default="smiles",
        help="SMILES column name for --input_csv mode.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Optional output CSV path for batch mode.",
    )
    parser.add_argument(
        "--include_smiles",
        action="store_true",
        help="Keep the original SMILES column in the exported features.",
    )
    parser.add_argument(
        "--on_error",
        choices=["raise", "nan"],
        default="raise",
        help="How batch mode should handle invalid molecules.",
    )
    parser.add_argument(
        "--show_metadata",
        action="store_true",
        help="Print feature descriptions and skipped rule groups as JSON.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.show_metadata:
        payload = {
            "feature_names": get_feature_names(),
            "feature_descriptions": get_feature_descriptions(),
            "skipped_rule_groups": get_skipped_rule_groups(),
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    if bool(args.smiles) == bool(args.input_csv):
        parser.error("Provide exactly one of --smiles or --input_csv.")

    if args.smiles:
        feature_dict = featurize_smiles(args.smiles)
        print(json.dumps(feature_dict, indent=2, ensure_ascii=False))
        return 0

    import pandas as pd

    df = pd.read_csv(args.input_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"Column {args.smiles_col!r} was not found in {args.input_csv}")

    feature_df = featurize_smiles_list(
        df[args.smiles_col].astype(str).tolist(),
        include_smiles=args.include_smiles,
        on_error=args.on_error,
    )

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path, index=False)
        print(f"Wrote {len(feature_df)} rows to {output_path}")
    else:
        print(feature_df.to_csv(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
