#!/usr/bin/env python3
"""Deterministic SARSCoV2_3CLPro_Diamond rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch SARS-CoV-2 3CLpro ruleset.

Implemented feature groups:
- core physicochemical descriptors behind the reported MW/logP/TPSA/HBD/HBA/
  flexibility/aromaticity/Fsp3 windows
- electrophilic-warhead motif counts for nitriles, aldehydes, ketones, and
  Michael-acceptor-like enones
- peptidomimetic structural proxies for P1 glutamine mimics, hydrophobic P2
  substituents, and polar P3/P4 amide-like groups

Intentionally skipped:
- exact residue-level interactions with Cys145, His163, Glu166, and Gln189,
  because these require an aligned protein-ligand pose rather than a 2D graph
- exact P1/P2/P3/P4 positional assignment, because the repository does not
  ship a pose-aware peptidomimetic alignment workflow
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

CARBAMATE_PATTERN_SPECS = (
    ("[OX2][CX3](=[OX1])[NX3]", (1,)),
    ("[NX3][CX3](=[OX1])[OX2]", (1,)),
)
SULFONAMIDE_PATTERN_SPECS = (
    ("[SX4](=[OX1])(=[OX1])[NX3]", (0,)),
)
MICHAEL_ACCEPTOR_PATTERN_SPECS = (
    ("[C,c]=[C,c][CX3](=[OX1])[#6,#7,#8]", (2,)),
)


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("heteroatom_count", "Heteroatom count."),
        ("logp", "Wildman-Crippen logP."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("aliphatic_ring_count", "Aliphatic ring count."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("formal_charge", "Formal charge from the input graph."),
        ("amide_bond_count", "Amide-bond count."),
        ("primary_amide_count", "Primary amide count."),
        ("lactam_count", "Cyclic amide count."),
        ("gamma_lactam_count", "Five-membered lactam count as a P1 glutamine-mimic proxy."),
        ("nitrile_count", "Nitrile count."),
        ("aldehyde_count", "Aldehyde count."),
        ("ketone_count", "Ketone count."),
        ("michael_acceptor_count", "Alpha,beta-unsaturated carbonyl count as a Michael-acceptor proxy."),
        (
            "electrophilic_warhead_count",
            "Sum of nitrile, aldehyde, ketone, and Michael-acceptor motif counts as a warhead proxy.",
        ),
        ("carbamate_count", "Carbamate count."),
        ("urea_count", "Urea count."),
        ("sulfonamide_count", "Sulfonamide count."),
        (
            "polar_amide_like_group_count",
            "Sum of amide, carbamate, urea, and sulfonamide counts as a polar P3/P4 substituent proxy.",
        ),
        (
            "polar_amide_like_group_density",
            "Polar amide-like group count normalized by heavy atom count.",
        ),
        ("branched_sp3_carbon_count", "Count of non-ring sp3 carbons attached to at least two carbons."),
        (
            "small_aliphatic_carbocycle_count",
            "Count of non-aromatic carbocyclic rings of size 3-6 as a hydrophobic sidechain proxy.",
        ),
        ("bridgehead_atom_count", "Bridgehead atom count as a rigid bicyclic-sidechain proxy."),
        ("spiro_atom_count", "Spiro atom count as an additional rigid-sidechain proxy."),
        ("has_electrophilic_warhead", "1 if any electrophilic warhead proxy is present."),
        ("p1_glutamine_mimic_present", "1 if a five-membered lactam P1 glutamine-mimic proxy is present."),
        (
            "s1_hbond_proxy_present",
            "1 if gamma-lactam or donor/acceptor-rich amide motifs suggest the conserved S1 H-bond pattern.",
        ),
        ("p2_hydrophobic_sidechain_present", "1 if hydrophobic branched/cyclic P2-sidechain proxies are present."),
        ("p2_hydrophobic_proxy_score", "Count of satisfied hydrophobic P2 proxy conditions."),
        (
            "p2_hydrophobic_proxy_density",
            "Hydrophobic P2 proxy score normalized by heavy atom count.",
        ),
        (
            "p3_p4_polar_substituent_present",
            "1 if amide-like polar substituents suggest a P3/P4 hydrogen-bonding handle.",
        ),
        ("mpro_structural_motif_count", "Count of conserved structural motif proxies among warhead, P1, P2, and P3/P4."),
        ("rotatable_bonds_le_10", "Rule flag: rotatable bonds <= 10."),
        ("tpsa_le_140", "Rule flag: TPSA <= 140."),
        ("logp_lt_5", "Rule flag: logP < 5."),
        ("mol_weight_le_500", "Rule flag: molecular weight <= 500."),
        ("hbd_le_5", "Rule flag: HBD <= 5."),
        ("hba_le_10", "Rule flag: HBA <= 10."),
        ("total_hbond_donors_acceptors_le_12", "Rule flag: HBD + HBA <= 12."),
        ("aromatic_ring_count_le_3", "Rule flag: aromatic ring count <= 3."),
        ("fraction_csp3_ge_0p4", "Rule flag: fractionCSP3 >= 0.4."),
        ("mpro_physchem_rule_pass_count", "Count of passed physicochemical rules derived from the source response."),
        ("mpro_physchem_window_pass", "1 if all physicochemical windows from the source response are satisfied."),
        ("mpro_design_rule_pass_count", "Physicochemical pass count plus the four structural motif flags."),
        (
            "mpro_compactness_adjusted_design_score",
            "Design-rule pass count normalized by a simple heavy-atom compactness penalty.",
        ),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Exact covalent binding to Cys145 and exact His163/Glu166/Gln189 contacts are not hard-coded because they require a protein-ligand pose rather than a ligand-only SMILES graph.",
    "Exact P1/P2/P3/P4 residue-position assignment is not hard-coded because the repository does not provide an aligned peptidomimetic template or docking workflow.",
    "The source response mentions glutamine surrogates beyond gamma-lactam, but only stable 2D motif proxies such as gamma-lactam and amide-like groups are encoded here.",
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


def _maybe_lt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value < threshold)


def _maybe_ge(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value >= threshold)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Fragments, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run SARSCoV2_3CLPro_Diamond feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        Fragments=Fragments,
        Lipinski=Lipinski,
        rdMolDescriptors=rdMolDescriptors,
    )


@lru_cache(maxsize=None)
def _smarts(pattern: str):
    mol = _rdkit().Chem.MolFromSmarts(pattern)
    if mol is None:
        raise ValueError(f"Invalid SMARTS pattern: {pattern}")
    return mol


def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    rdkit = _rdkit()
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _count_unique_matches_any(
    mol,
    pattern_specs: Iterable[tuple[str, tuple[int, ...] | None]],
) -> int:
    unique_keys: set[tuple[int, ...]] = set()
    for pattern, atom_positions in pattern_specs:
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


def _is_lactam_ring(mol, ring_atoms: tuple[int, ...]) -> bool:
    ring_set = set(ring_atoms)
    if any(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_atoms):
        return False

    has_ring_nitrogen = any(
        mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 7 for atom_idx in ring_atoms
    )
    if not has_ring_nitrogen:
        return False

    for atom_idx in ring_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetAtomicNum() != 6:
            continue

        has_exocyclic_carbonyl_oxygen = False
        has_ring_nitrogen_neighbor = False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            other_idx = other.GetIdx()
            if (
                bond.GetBondTypeAsDouble() == 2.0
                and other.GetAtomicNum() == 8
                and other_idx not in ring_set
            ):
                has_exocyclic_carbonyl_oxygen = True
            if other_idx in ring_set and other.GetAtomicNum() == 7:
                has_ring_nitrogen_neighbor = True

        if has_exocyclic_carbonyl_oxygen and has_ring_nitrogen_neighbor:
            return True

    return False


def _count_lactam_rings(mol, ring_size: int | None = None) -> int:
    ring_info = mol.GetRingInfo()
    count = 0
    for ring_atoms in ring_info.AtomRings():
        if ring_size is not None and len(ring_atoms) != ring_size:
            continue
        if _is_lactam_ring(mol, ring_atoms):
            count += 1
    return count


def _count_branched_sp3_carbons(mol) -> int:
    rdkit = _rdkit()
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic() or atom.IsInRing():
            continue
        if atom.GetHybridization() != rdkit.Chem.HybridizationType.SP3:
            continue
        carbon_neighbor_count = sum(
            neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()
        )
        if carbon_neighbor_count >= 2:
            count += 1
    return count


def _count_small_aliphatic_carbocycles(mol, min_size: int = 3, max_size: int = 6) -> int:
    ring_info = mol.GetRingInfo()
    count = 0
    for ring_atoms in ring_info.AtomRings():
        if not min_size <= len(ring_atoms) <= max_size:
            continue
        if any(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_atoms):
            continue
        if all(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6 for atom_idx in ring_atoms):
            count += 1
    return count


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    exact_mol_weight = _as_float(rdkit.Descriptors.ExactMolWt(mol))
    heavy_atom_count = _as_float(mol.GetNumHeavyAtoms())
    heteroatom_count = _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    total_hbonds = hbd + hba
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.Lipinski.NumAliphaticRings(mol))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))

    amide_bond_count = _as_float(rdkit.rdMolDescriptors.CalcNumAmideBonds(mol))
    primary_amide_count = _as_float(rdkit.Fragments.fr_priamide(mol))
    lactam_count = _as_float(rdkit.Fragments.fr_lactam(mol))
    gamma_lactam_count = _as_float(_count_lactam_rings(mol, ring_size=5))

    nitrile_count = _as_float(rdkit.Fragments.fr_nitrile(mol))
    aldehyde_count = _as_float(rdkit.Fragments.fr_aldehyde(mol))
    ketone_count = _as_float(rdkit.Fragments.fr_ketone(mol))
    michael_acceptor_count = _as_float(_count_unique_matches_any(mol, MICHAEL_ACCEPTOR_PATTERN_SPECS))
    electrophilic_warhead_count = nitrile_count + aldehyde_count + ketone_count + michael_acceptor_count

    carbamate_count = _as_float(_count_unique_matches_any(mol, CARBAMATE_PATTERN_SPECS))
    urea_count = _as_float(rdkit.Fragments.fr_urea(mol))
    sulfonamide_count = _as_float(_count_unique_matches_any(mol, SULFONAMIDE_PATTERN_SPECS))
    polar_amide_like_group_count = amide_bond_count + carbamate_count + urea_count + sulfonamide_count

    branched_sp3_carbon_count = _as_float(_count_branched_sp3_carbons(mol))
    small_aliphatic_carbocycle_count = _as_float(_count_small_aliphatic_carbocycles(mol))
    bridgehead_atom_count = _as_float(rdkit.rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    spiro_atom_count = _as_float(rdkit.rdMolDescriptors.CalcNumSpiroAtoms(mol))

    has_electrophilic_warhead = electrophilic_warhead_count > 0.0
    p1_glutamine_mimic_present = gamma_lactam_count > 0.0
    s1_hbond_proxy_present = (
        gamma_lactam_count > 0.0
        or (amide_bond_count >= 2.0 and hbd >= 1.0 and hba >= 2.0)
    )
    p2_hydrophobic_sidechain_present = (
        branched_sp3_carbon_count > 0.0
        or small_aliphatic_carbocycle_count > 0.0
        or ((bridgehead_atom_count > 0.0 or spiro_atom_count > 0.0) and fraction_csp3 >= 0.4)
    )
    p2_hydrophobic_proxy_score = float(
        sum(
            (
                branched_sp3_carbon_count > 0.0,
                small_aliphatic_carbocycle_count > 0.0,
                (bridgehead_atom_count > 0.0 and fraction_csp3 >= 0.4),
                (spiro_atom_count > 0.0 and fraction_csp3 >= 0.4),
                fraction_csp3 >= 0.4,
            )
        )
    )
    p3_p4_polar_substituent_present = polar_amide_like_group_count > 0.0
    mpro_structural_motif_count = float(
        sum(
            (
                has_electrophilic_warhead,
                p1_glutamine_mimic_present,
                p2_hydrophobic_sidechain_present,
                p3_p4_polar_substituent_present,
            )
        )
    )

    physchem_flags = {
        "rotatable_bonds_le_10": rotatable_bonds <= 10.0,
        "tpsa_le_140": tpsa <= 140.0,
        "logp_lt_5": logp < 5.0,
        "mol_weight_le_500": mol_weight <= 500.0,
        "hbd_le_5": hbd <= 5.0,
        "hba_le_10": hba <= 10.0,
        "total_hbond_donors_acceptors_le_12": total_hbonds <= 12.0,
        "aromatic_ring_count_le_3": aromatic_ring_count <= 3.0,
        "fraction_csp3_ge_0p4": fraction_csp3 >= 0.4,
    }
    mpro_physchem_rule_pass_count = float(sum(physchem_flags.values()))
    mpro_physchem_window_pass = all(physchem_flags.values())
    mpro_design_rule_pass_count = mpro_physchem_rule_pass_count + mpro_structural_motif_count
    polar_amide_like_group_density = _as_float(polar_amide_like_group_count / max(heavy_atom_count, 1.0))
    p2_hydrophobic_proxy_density = _as_float(p2_hydrophobic_proxy_score / max(heavy_atom_count, 1.0))
    mpro_compactness_adjusted_design_score = _as_float(
        mpro_design_rule_pass_count / (1.0 + heavy_atom_count / 20.0)
    )

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
            "heteroatom_count": heteroatom_count,
            "logp": logp,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbonds,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "fraction_csp3": fraction_csp3,
            "formal_charge": formal_charge,
            "amide_bond_count": amide_bond_count,
            "primary_amide_count": primary_amide_count,
            "lactam_count": lactam_count,
            "gamma_lactam_count": gamma_lactam_count,
            "nitrile_count": nitrile_count,
            "aldehyde_count": aldehyde_count,
            "ketone_count": ketone_count,
            "michael_acceptor_count": michael_acceptor_count,
            "electrophilic_warhead_count": electrophilic_warhead_count,
            "carbamate_count": carbamate_count,
            "urea_count": urea_count,
            "sulfonamide_count": sulfonamide_count,
            "polar_amide_like_group_count": polar_amide_like_group_count,
            "polar_amide_like_group_density": polar_amide_like_group_density,
            "branched_sp3_carbon_count": branched_sp3_carbon_count,
            "small_aliphatic_carbocycle_count": small_aliphatic_carbocycle_count,
            "bridgehead_atom_count": bridgehead_atom_count,
            "spiro_atom_count": spiro_atom_count,
            "has_electrophilic_warhead": _flag(has_electrophilic_warhead),
            "p1_glutamine_mimic_present": _flag(p1_glutamine_mimic_present),
            "s1_hbond_proxy_present": _flag(s1_hbond_proxy_present),
            "p2_hydrophobic_sidechain_present": _flag(p2_hydrophobic_sidechain_present),
            "p2_hydrophobic_proxy_score": p2_hydrophobic_proxy_score,
            "p2_hydrophobic_proxy_density": p2_hydrophobic_proxy_density,
            "p3_p4_polar_substituent_present": _flag(p3_p4_polar_substituent_present),
            "mpro_structural_motif_count": mpro_structural_motif_count,
            "rotatable_bonds_le_10": _flag(physchem_flags["rotatable_bonds_le_10"]),
            "tpsa_le_140": _flag(physchem_flags["tpsa_le_140"]),
            "logp_lt_5": _flag(physchem_flags["logp_lt_5"]),
            "mol_weight_le_500": _flag(physchem_flags["mol_weight_le_500"]),
            "hbd_le_5": _flag(physchem_flags["hbd_le_5"]),
            "hba_le_10": _flag(physchem_flags["hba_le_10"]),
            "total_hbond_donors_acceptors_le_12": _flag(physchem_flags["total_hbond_donors_acceptors_le_12"]),
            "aromatic_ring_count_le_3": _flag(physchem_flags["aromatic_ring_count_le_3"]),
            "fraction_csp3_ge_0p4": _flag(physchem_flags["fraction_csp3_ge_0p4"]),
            "mpro_physchem_rule_pass_count": mpro_physchem_rule_pass_count,
            "mpro_physchem_window_pass": _flag(mpro_physchem_window_pass),
            "mpro_design_rule_pass_count": mpro_design_rule_pass_count,
            "mpro_compactness_adjusted_design_score": mpro_compactness_adjusted_design_score,
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
        if include_smiles:
            row = {"smiles": smiles, **row}
        rows.append(row)

    return pd.DataFrame(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SARSCoV2_3CLPro_Diamond DeepResearch features from SMILES."
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
