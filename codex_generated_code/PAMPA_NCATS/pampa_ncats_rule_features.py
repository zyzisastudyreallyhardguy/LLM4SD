#!/usr/bin/env python3
"""Deterministic PAMPA_NCATS rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch PAMPA_NCATS ruleset. The implemented
features focus on:

- passive-permeability-related physicochemical windows such as MW, logP/logD,
  TPSA, HBD/HBA balance, flexibility, and formal charge
- acidic-functionality counts and pKa-derived ionization proxies
- local structure proxies for intramolecular H-bond potential and compactness

Intentionally skipped:
- experimental PAMPA values or measured conformer populations
- exact 3D globularity/sphericity and exact van der Waals surface area
- externally measured logD/pKa values when the local MolGpKa helper
  is unavailable
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

PHYSIOLOGICAL_PH = 7.4
INTERN_S1_ROOT = Path("/data1/tianang/Projects/Intern-S1")

CARBOXYLIC_ACID_PATTERNS = (
    "[CX3](=O)[OX2H1,O-]",
)
SULFONIC_ACID_PATTERNS = (
    "[SX4](=[OX1])(=[OX1])[OX2H1,O-]",
)
PHOSPHONIC_ACID_PATTERNS = (
    "[PX4](=[OX1])([OX2H1,O-])[OX2H1,O-]",
)


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("heteroatom_count", "Total heteroatom count."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("aliphatic_ring_count", "Aliphatic ring count."),
        ("fraction_csp3", "Fraction of sp3 carbons as a compactness proxy."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("abs_formal_charge", "Absolute formal charge magnitude."),
        ("labute_asa", "Labute approximate surface area as a local surface-area proxy."),
        ("labute_asa_per_heavy_atom", "Labute ASA divided by heavy-atom count."),
        ("balaban_j", "Balaban J topological index."),
        ("kappa1", "First kappa shape index."),
        ("kappa2", "Second kappa shape index."),
        ("kappa3", "Third kappa shape index."),
        ("aromatic_heavy_atom_fraction", "Aromatic atom fraction among heavy atoms."),
        ("most_basic_pka", "Most basic predicted pKa from the MolGpKa helper."),
        ("most_acidic_pka", "Most acidic predicted pKa from the MolGpKa helper."),
        ("num_basic_sites", "Number of predicted basic sites from the MolGpKa helper."),
        ("num_acidic_sites", "Number of predicted acidic sites from the MolGpKa helper."),
        ("neutral_fraction_ph74", "Estimated neutral fraction at pH 7.4."),
        ("charged_fraction_ph74", "1 - neutral_fraction_ph74."),
        ("base_protonated_fraction_ph74", "Estimated protonated fraction for the dominant basic site at pH 7.4."),
        ("acid_deprotonated_fraction_ph74", "Estimated deprotonated fraction for the dominant acidic site at pH 7.4."),
        ("net_charge_proxy_ph74", "Base protonation minus acid deprotonation proxy at pH 7.4."),
        ("estimated_logd_ph74", "Estimated logD at pH 7.4 from logP and neutral fraction."),
        ("has_amphoteric_sites", "1 if both acidic and basic sites are predicted."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
        ("carboxylic_acid_count", "Count of carboxylic acid or carboxylate groups."),
        ("sulfonic_acid_count", "Count of sulfonic acid or sulfonate groups."),
        ("phosphonic_acid_count", "Count of phosphonic/phosphonate-like acidic groups."),
        ("acidic_group_count", "Total acidic-group count from carboxylate/sulfonate/phosphonate motifs."),
        ("strong_acidic_group_count", "Count of strong-acid-like sulfonate/phosphonate motifs."),
        (
            "intramolecular_hbond_pair_count",
            "Count of donor-acceptor atom pairs 4-8 bonds apart as a proxy for intramolecular H-bond potential.",
        ),
        ("intramolecular_hbond_potential_present", "1 if the intramolecular H-bond proxy count is non-zero."),
        ("mw_le_500", "Rule flag: molecular weight <= 500."),
        ("hbd_le_5", "Rule flag: HBD <= 5."),
        ("hba_le_10", "Rule flag: HBA <= 10."),
        ("logp_le_5", "Rule flag: logP <= 5."),
        ("estimated_logd_between_neg0p5_4p5", "Rule flag: -0.5 <= estimated logD(7.4) <= 4.5."),
        ("tpsa_lt_60", "Rule flag: TPSA < 60."),
        ("tpsa_le_120", "Rule flag: TPSA <= 120."),
        ("rotatable_bonds_le_10", "Rule flag: rotatable bonds <= 10."),
        ("total_hbond_donors_acceptors_le_12", "Rule flag: HBD + HBA <= 12."),
        ("formal_charge_zero", "Rule flag: formal charge == 0."),
        ("neutral_fraction_ge_0p5", "Rule flag: estimated neutral fraction at pH 7.4 >= 0.5."),
        ("acidic_group_count_eq_0", "Rule flag: no acidic-group motif is present."),
        ("strong_acidic_group_count_eq_0", "Rule flag: no strong-acid-like motif is present."),
        (
            "passive_permeability_screen_pass",
            "Composite flag combining the main PAMPA_NCATS heuristic windows and avoiding strong-acid motifs.",
        ),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Exact 3D globularity or sphericity is not hard-coded; the module exposes 2D shape and compactness proxies such as kappa indices and fractionCSP3 instead.",
    "Exact van der Waals surface area is approximated with LabuteASA because no dedicated local vdw-surface calculator is wired into this repository.",
    "Closed-conformation and intramolecular H-bond behavior are represented only by a donor-acceptor topological proxy, not by conformer enumeration.",
    "Experimental logD/pKa measurements are not queried; pKa-derived features fall back to NaN when the local MolGpKa helper is unavailable.",
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
        from rdkit.Chem import ChemicalFeatures, Crippen, Descriptors, GraphDescriptors, Lipinski, MolSurf, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run PAMPA_NCATS feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        RDConfig=RDConfig,
        ChemicalFeatures=ChemicalFeatures,
        Crippen=Crippen,
        Descriptors=Descriptors,
        GraphDescriptors=GraphDescriptors,
        Lipinski=Lipinski,
        MolSurf=MolSurf,
        rdMolDescriptors=rdMolDescriptors,
    )


@lru_cache(maxsize=None)
def _smarts(pattern: str):
    mol = _rdkit().Chem.MolFromSmarts(pattern)
    if mol is None:
        raise ValueError(f"Invalid SMARTS pattern: {pattern}")
    return mol


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


def _count_formal_charge_atoms(mol) -> tuple[int, int]:
    positive = 0
    negative = 0
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            positive += 1
        elif charge < 0:
            negative += 1
    return positive, negative


def _count_aromatic_atoms(mol) -> int:
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms())


def _count_unique_matches(mol, pattern, atom_positions: tuple[int, ...] | None = None) -> int:
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if atom_positions is None:
        return len(matches)
    unique_keys = {
        tuple(sorted(match[position] for position in atom_positions))
        for match in matches
    }
    return len(unique_keys)


def _count_unique_matches_any(mol, patterns: Iterable[str], atom_positions: tuple[int, ...] | None = None) -> int:
    unique_keys: set[tuple[int, ...]] = set()
    for pattern in patterns:
        compiled = _smarts(pattern)
        matches = mol.GetSubstructMatches(compiled, uniquify=True)
        if atom_positions is None:
            unique_keys.update(tuple(sorted(match)) for match in matches)
        else:
            unique_keys.update(
                tuple(sorted(match[position] for position in atom_positions))
                for match in matches
            )
    return len(unique_keys)


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


def _compute_pka_summary(mol) -> dict[str, float]:
    result = {
        "most_basic_pka": math.nan,
        "most_acidic_pka": math.nan,
        "num_basic_sites": math.nan,
        "num_acidic_sites": math.nan,
        "neutral_fraction_ph74": math.nan,
        "charged_fraction_ph74": math.nan,
        "base_protonated_fraction_ph74": math.nan,
        "acid_deprotonated_fraction_ph74": math.nan,
        "net_charge_proxy_ph74": math.nan,
        "estimated_logd_ph74": math.nan,
        "has_amphoteric_sites": math.nan,
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
    num_basic_sites = float(len(base_sites))
    num_acidic_sites = float(len(acid_sites))

    base_protonated_fraction = 0.0
    if base_sites:
        base_protonated_fraction = 1.0 / (1.0 + 10.0 ** (PHYSIOLOGICAL_PH - most_basic_pka))

    acid_deprotonated_fraction = 0.0
    if acid_sites:
        acid_deprotonated_fraction = 1.0 / (1.0 + 10.0 ** (most_acidic_pka - PHYSIOLOGICAL_PH))

    neutral_fraction = 1.0
    if base_sites:
        neutral_fraction *= 1.0 - base_protonated_fraction
    if acid_sites:
        neutral_fraction *= 1.0 - acid_deprotonated_fraction
    neutral_fraction = min(1.0, max(1e-12, neutral_fraction))

    logp = _as_float(_rdkit().Crippen.MolLogP(mol))
    estimated_logd = logp + math.log10(neutral_fraction)

    result.update(
        {
            "most_basic_pka": _as_float(most_basic_pka) if base_sites else math.nan,
            "most_acidic_pka": _as_float(most_acidic_pka) if acid_sites else math.nan,
            "num_basic_sites": num_basic_sites,
            "num_acidic_sites": num_acidic_sites,
            "neutral_fraction_ph74": neutral_fraction,
            "charged_fraction_ph74": 1.0 - neutral_fraction,
            "base_protonated_fraction_ph74": base_protonated_fraction,
            "acid_deprotonated_fraction_ph74": acid_deprotonated_fraction,
            "net_charge_proxy_ph74": base_protonated_fraction - acid_deprotonated_fraction,
            "estimated_logd_ph74": estimated_logd,
            "has_amphoteric_sites": float(bool(base_sites and acid_sites)),
            "pka_features_available": 1.0,
        }
    )
    return result


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    exact_mol_weight = _as_float(rdkit.Descriptors.ExactMolWt(mol))
    heavy_atom_count = _as_float(mol.GetNumHeavyAtoms())
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    molar_refractivity = _as_float(rdkit.Crippen.MolMR(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    total_hbonds = hbd + hba
    heteroatom_count = _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol))
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.Lipinski.NumAliphaticRings(mol))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    abs_formal_charge = abs(formal_charge)
    labute_asa = _as_float(rdkit.MolSurf.LabuteASA(mol))
    labute_asa_per_heavy_atom = math.nan if heavy_atom_count == 0 else labute_asa / heavy_atom_count
    aromatic_heavy_atom_fraction = math.nan if heavy_atom_count == 0 else _count_aromatic_atoms(mol) / heavy_atom_count

    carboxylic_acid_count = _as_float(_count_unique_matches_any(mol, CARBOXYLIC_ACID_PATTERNS, (0,)))
    sulfonic_acid_count = _as_float(_count_unique_matches_any(mol, SULFONIC_ACID_PATTERNS, (0,)))
    phosphonic_acid_count = _as_float(_count_unique_matches_any(mol, PHOSPHONIC_ACID_PATTERNS, (0,)))
    acidic_group_count = carboxylic_acid_count + sulfonic_acid_count + phosphonic_acid_count
    strong_acidic_group_count = sulfonic_acid_count + phosphonic_acid_count
    intramolecular_hbond_pair_count = _as_float(_count_intramolecular_hbond_pairs(mol))

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
            "logp": logp,
            "molar_refractivity": molar_refractivity,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbonds,
            "heteroatom_count": heteroatom_count,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "fraction_csp3": fraction_csp3,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "abs_formal_charge": abs_formal_charge,
            "labute_asa": labute_asa,
            "labute_asa_per_heavy_atom": labute_asa_per_heavy_atom,
            "balaban_j": _as_float(rdkit.GraphDescriptors.BalabanJ(mol)),
            "kappa1": _as_float(rdkit.GraphDescriptors.Kappa1(mol)),
            "kappa2": _as_float(rdkit.GraphDescriptors.Kappa2(mol)),
            "kappa3": _as_float(rdkit.GraphDescriptors.Kappa3(mol)),
            "aromatic_heavy_atom_fraction": aromatic_heavy_atom_fraction,
            "carboxylic_acid_count": carboxylic_acid_count,
            "sulfonic_acid_count": sulfonic_acid_count,
            "phosphonic_acid_count": phosphonic_acid_count,
            "acidic_group_count": acidic_group_count,
            "strong_acidic_group_count": strong_acidic_group_count,
            "intramolecular_hbond_pair_count": intramolecular_hbond_pair_count,
            "intramolecular_hbond_potential_present": _flag(intramolecular_hbond_pair_count > 0.0),
        }
    )

    features.update(_compute_pka_summary(mol))

    passive_permeability_screen_pass = None
    if not any(
        _is_missing(features[key])
        for key in (
            "mol_weight",
            "hbd",
            "hba",
            "logp",
            "estimated_logd_ph74",
            "tpsa",
            "rotatable_bonds",
            "total_hbond_donors_acceptors",
            "formal_charge",
        )
    ):
        passive_permeability_screen_pass = (
            features["mol_weight"] <= 500.0
            and features["hbd"] <= 5.0
            and features["hba"] <= 10.0
            and features["logp"] <= 5.0
            and -0.5 <= features["estimated_logd_ph74"] <= 4.5
            and features["tpsa"] <= 120.0
            and features["rotatable_bonds"] <= 10.0
            and features["total_hbond_donors_acceptors"] <= 12.0
            and features["formal_charge"] == 0.0
            and strong_acidic_group_count == 0.0
        )

    features.update(
        {
            "mw_le_500": _maybe_le(mol_weight, 500.0),
            "hbd_le_5": _maybe_le(hbd, 5.0),
            "hba_le_10": _maybe_le(hba, 10.0),
            "logp_le_5": _maybe_le(logp, 5.0),
            "estimated_logd_between_neg0p5_4p5": _maybe_between(features["estimated_logd_ph74"], -0.5, 4.5),
            "tpsa_lt_60": _flag(tpsa < 60.0),
            "tpsa_le_120": _maybe_le(tpsa, 120.0),
            "rotatable_bonds_le_10": _maybe_le(rotatable_bonds, 10.0),
            "total_hbond_donors_acceptors_le_12": _maybe_le(total_hbonds, 12.0),
            "formal_charge_zero": _flag(formal_charge == 0.0),
            "neutral_fraction_ge_0p5": _maybe_ge(features["neutral_fraction_ph74"], 0.5),
            "acidic_group_count_eq_0": _flag(acidic_group_count == 0.0),
            "strong_acidic_group_count_eq_0": _flag(strong_acidic_group_count == 0.0),
            "passive_permeability_screen_pass": _flag(passive_permeability_screen_pass),
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
        description="Generate PAMPA_NCATS DeepResearch features from SMILES."
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
