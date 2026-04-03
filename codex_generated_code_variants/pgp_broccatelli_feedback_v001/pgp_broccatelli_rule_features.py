#!/usr/bin/env python3
"""Deterministic Pgp_Broccatelli rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch P-gp inhibition ruleset. The implemented
features emphasize:

- size, lipophilicity, polarity, and H-bond balance
- aromaticity and bulky hydrophobic motif proxies
- basic/cationic center features plus MolGpKa-derived pKa descriptors

Intentionally skipped:
- assay-specific transporter kinetics such as IC50, ATPase, and efflux ratios
- exact experimental pKa or logD values when the local MolGpKa helper is
  unavailable
- explicit 3D planarity or docking features that need conformer generation and
  a validated transporter model
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

PHYSIOLOGICAL_PH = 7.4
INTERN_S1_ROOT = Path("/data1/tianang/Projects/Intern-S1")

BASIC_CENTER_PATTERNS = (
    "[NX4+]",
    "[NX3;H2,H1,H0;+0;!$([N]-[C,S,P]=[O,S,N])]",
)
TERTIARY_AMINE_PATTERNS = (
    "[NX3;H0;+0;!$([N]-[C,S,P]=[O,S,N])]([#6])([#6])[#6]",
)
QUATERNARY_AMMONIUM_PATTERNS = (
    "[NX4+]([#6])([#6])([#6])[#6]",
)
ADAMANTANE_SMARTS = "C1C2CC3CC1CC(C2)C3"


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("carbon_atom_count", "Total carbon atom count."),
        ("heteroatom_count", "Total heteroatom count."),
        ("oxygen_nitrogen_count", "Count of O and N atoms."),
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("aliphatic_ring_count", "Aliphatic ring count."),
        ("fused_aromatic_ring_count", "Count of aromatic rings participating in fused aromatic systems."),
        ("aromatic_atom_fraction", "Aromatic atom fraction among heavy atoms."),
        ("carbon_heavy_atom_fraction", "Carbon atom fraction among heavy atoms."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("basic_center_count", "Count of amine-like basic or permanently cationic nitrogen centers."),
        ("tertiary_amine_count", "Count of tertiary amine motifs excluding simple amide-like nitrogens."),
        ("quaternary_ammonium_count", "Count of quaternary ammonium centers."),
        ("aromatic_linker_count", "Count of non-ring bonds directly connecting two aromatic atoms."),
        ("longest_alkyl_chain", "Longest contiguous non-ring sp3 carbon chain length."),
        ("adamantane_like_present", "1 if an adamantane-like cage motif is present."),
        ("most_basic_pka", "Most basic predicted pKa from the MolGpKa helper."),
        ("num_basic_sites", "Number of predicted basic sites from the MolGpKa helper."),
        ("base_protonated_fraction_ph74", "Estimated protonated fraction for the dominant basic site at pH 7.4."),
        ("pka_features_available", "1 if MolGpKa-derived pKa features were computed."),
        ("mw_gt_500", "Rule flag: molecular weight > 500."),
        ("mw_ge_300", "Rule flag: molecular weight >= 300."),
        ("mw_lt_300", "Rule flag: molecular weight < 300."),
        ("logp_ge_3", "Rule flag: logP >= 3."),
        ("logp_le_3", "Rule flag: logP <= 3."),
        ("tpsa_gt_75", "Rule flag: TPSA > 75."),
        ("tpsa_le_75", "Rule flag: TPSA <= 75."),
        ("hba_ge_3", "Rule flag: HBA >= 3."),
        ("hba_le_2", "Rule flag: HBA <= 2."),
        ("hbd_le_3", "Rule flag: HBD <= 3."),
        ("hbd_ge_4", "Rule flag: HBD >= 4."),
        ("oxygen_nitrogen_ge_4", "Rule flag: O + N >= 4."),
        ("oxygen_nitrogen_le_3", "Rule flag: O + N <= 3."),
        ("basic_center_present", "Rule flag: at least one amine-like basic or cationic center is present."),
        ("cationic_center_present", "Rule flag: at least one atom carries positive formal charge."),
        ("tertiary_amine_present", "Rule flag: at least one tertiary amine motif is present."),
        ("most_basic_pka_gt_8", "Rule flag: most basic predicted pKa > 8."),
        ("most_basic_pka_lt_8", "Rule flag: most basic predicted pKa < 8."),
        ("aromatic_ring_count_ge_2", "Rule flag: aromatic ring count >= 2."),
        ("aromatic_linker_present", "Rule flag: at least one direct aromatic linker bond is present."),
        ("long_alkyl_chain_ge_6", "Rule flag: longest non-ring sp3 carbon chain length >= 6."),
        (
            "bulky_hydrophobic_moiety_present",
            "Heuristic flag for adamantane-like, long-alkyl, or multi-aromatic bulky hydrophobes.",
        ),
        ("hydrophobic_large_molecule", "Composite flag: molecular weight >= 300 and logP >= 3."),
        ("aromatic_basic_combo_present", "Composite flag: >=2 aromatic rings and at least one basic center."),
        (
            "pgp_flowchart_high_likelihood",
            "Composite flag implementing the simplified DeepResearch flowchart high-likelihood branch.",
        ),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Direct transporter inhibition measurements such as IC50, Ki, ATPase, and bidirectional efflux ratios are skipped because they require assay data.",
    "Exact 3D planarity, conformer preferences, and transporter docking interactions are reduced to 2D aromaticity proxies rather than modeled explicitly.",
    "Base pKa thresholds are exposed through the local MolGpKa helper when it is available; no external API is queried.",
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


def _maybe_ge(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value >= threshold)


def _maybe_gt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value > threshold)


def _maybe_le(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value <= threshold)


def _maybe_lt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value < threshold)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run Pgp_Broccatelli feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
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

    mol = _rdkit().Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _count_atomic_number(mol, atomic_num: int) -> int:
    return sum(atom.GetAtomicNum() == atomic_num for atom in mol.GetAtoms())


def _count_oxygen_and_nitrogen(mol) -> int:
    return sum(atom.GetAtomicNum() in (7, 8) for atom in mol.GetAtoms())


def _count_aromatic_atoms(mol) -> int:
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms())


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


def _count_unique_atom_matches_any(mol, patterns: Iterable[str]) -> int:
    unique_atom_indices: set[int] = set()
    for pattern in patterns:
        compiled = _smarts(pattern)
        for match in mol.GetSubstructMatches(compiled, uniquify=True):
            if match:
                unique_atom_indices.add(match[0])
    return len(unique_atom_indices)


def _count_fused_aromatic_rings(mol) -> int:
    aromatic_rings = []
    for ring in mol.GetRingInfo().AtomRings():
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring):
            aromatic_rings.append(set(ring))

    if not aromatic_rings:
        return 0

    fused_indices: set[int] = set()
    for i, ring_i in enumerate(aromatic_rings):
        for j in range(i + 1, len(aromatic_rings)):
            if len(ring_i.intersection(aromatic_rings[j])) >= 2:
                fused_indices.add(i)
                fused_indices.add(j)
    return len(fused_indices)


def _count_aromatic_linkers(mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetIsAromatic() and end.GetIsAromatic():
            count += 1
    return count


def _count_longest_alkyl_chain(mol) -> int:
    rdkit = _rdkit()
    eligible_atoms = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
        and not atom.GetIsAromatic()
        and not atom.IsInRing()
        and atom.GetHybridization() == rdkit.Chem.HybridizationType.SP3
    }
    if not eligible_atoms:
        return 0

    adjacency = {atom_idx: [] for atom_idx in eligible_atoms}
    for bond in mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        if begin_idx not in eligible_atoms or end_idx not in eligible_atoms:
            continue
        if bond.GetBondType() != rdkit.Chem.BondType.SINGLE:
            continue
        adjacency[begin_idx].append(end_idx)
        adjacency[end_idx].append(begin_idx)

    def bfs_farthest(start_idx: int) -> tuple[int, dict[int, int]]:
        queue = deque([start_idx])
        distances = {start_idx: 0}
        farthest = start_idx
        while queue:
            current = queue.popleft()
            if distances[current] > distances[farthest]:
                farthest = current
            for neighbor in adjacency[current]:
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
        return farthest, distances

    longest_chain = 1
    visited: set[int] = set()
    for atom_idx in eligible_atoms:
        if atom_idx in visited:
            continue

        component_queue = deque([atom_idx])
        component = {atom_idx}
        visited.add(atom_idx)
        while component_queue:
            current = component_queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in component:
                    continue
                component.add(neighbor)
                visited.add(neighbor)
                component_queue.append(neighbor)

        start = next(iter(component))
        farthest, _ = bfs_farthest(start)
        _, distances = bfs_farthest(farthest)
        diameter_edges = max(distances.values()) if distances else 0
        longest_chain = max(longest_chain, diameter_edges + 1)

    return longest_chain


@lru_cache(maxsize=1)
def _get_pka_predictor():
    if str(INTERN_S1_ROOT) not in sys.path:
        sys.path.insert(0, str(INTERN_S1_ROOT))

    from tools.pka_related_tools import _get_pka_predictor as _load_predictor

    return _load_predictor()


def _compute_basic_pka_summary(mol) -> dict[str, float]:
    result = {
        "most_basic_pka": 0.0,
        "num_basic_sites": 0.0,
        "base_protonated_fraction_ph74": 0.0,
        "pka_features_available": 0.0,
    }

    try:
        predictor = _get_pka_predictor()
        prediction = predictor.predict(mol)
    except Exception:
        return result

    base_sites = getattr(prediction, "base_sites_1", {}) or {}
    if not base_sites:
        result["pka_features_available"] = 1.0
        return result

    most_basic_pka = max(base_sites.values())
    protonated_fraction = 1.0 / (1.0 + 10.0 ** (PHYSIOLOGICAL_PH - most_basic_pka))

    result.update(
        {
            "most_basic_pka": _as_float(most_basic_pka),
            "num_basic_sites": _as_float(len(base_sites)),
            "base_protonated_fraction_ph74": protonated_fraction,
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
    carbon_atom_count = _as_float(_count_atomic_number(mol, 6))
    heteroatom_count = _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol))
    oxygen_nitrogen_count = _as_float(_count_oxygen_and_nitrogen(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    molar_refractivity = _as_float(rdkit.Crippen.MolMR(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    total_hbonds = hbd + hba
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.Lipinski.NumAliphaticRings(mol))
    fused_aromatic_ring_count = _as_float(_count_fused_aromatic_rings(mol))
    aromatic_atom_fraction = math.nan if heavy_atom_count == 0 else _count_aromatic_atoms(mol) / heavy_atom_count
    carbon_heavy_atom_fraction = math.nan if heavy_atom_count == 0 else carbon_atom_count / heavy_atom_count
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    basic_center_count = _as_float(_count_unique_atom_matches_any(mol, BASIC_CENTER_PATTERNS))
    tertiary_amine_count = _as_float(_count_unique_atom_matches_any(mol, TERTIARY_AMINE_PATTERNS))
    quaternary_ammonium_count = _as_float(_count_unique_atom_matches_any(mol, QUATERNARY_AMMONIUM_PATTERNS))
    aromatic_linker_count = _as_float(_count_aromatic_linkers(mol))
    longest_alkyl_chain = _as_float(_count_longest_alkyl_chain(mol))
    adamantane_like_present = _flag(mol.HasSubstructMatch(_smarts(ADAMANTANE_SMARTS)))

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
            "carbon_atom_count": carbon_atom_count,
            "heteroatom_count": heteroatom_count,
            "oxygen_nitrogen_count": oxygen_nitrogen_count,
            "logp": logp,
            "molar_refractivity": molar_refractivity,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbonds,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "fused_aromatic_ring_count": fused_aromatic_ring_count,
            "aromatic_atom_fraction": aromatic_atom_fraction,
            "carbon_heavy_atom_fraction": carbon_heavy_atom_fraction,
            "fraction_csp3": fraction_csp3,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "basic_center_count": basic_center_count,
            "tertiary_amine_count": tertiary_amine_count,
            "quaternary_ammonium_count": quaternary_ammonium_count,
            "aromatic_linker_count": aromatic_linker_count,
            "longest_alkyl_chain": longest_alkyl_chain,
            "adamantane_like_present": adamantane_like_present,
        }
    )

    features.update(_compute_basic_pka_summary(mol))

    basic_center_present = basic_center_count > 0.0
    cationic_center_present = positive_formal_atom_count > 0
    tertiary_amine_present = tertiary_amine_count > 0.0
    aromatic_ring_count_ge_2 = aromatic_ring_count >= 2.0
    aromatic_linker_present = aromatic_linker_count > 0.0
    long_alkyl_chain_ge_6 = longest_alkyl_chain >= 6.0
    bulky_hydrophobic_moiety_present = bool(
        adamantane_like_present
        or long_alkyl_chain_ge_6
        or aromatic_ring_count >= 3.0
        or fused_aromatic_ring_count >= 2.0
    )
    hydrophobic_large_molecule = bool(mol_weight >= 300.0 and logp >= 3.0)
    aromatic_basic_combo_present = bool(aromatic_ring_count_ge_2 and basic_center_present)
    pgp_flowchart_high_likelihood = bool(
        logp >= 3.0
        and mol_weight >= 300.0
        and hbd <= 3.0
        and (hba >= 3.0 or aromatic_basic_combo_present)
    )

    features.update(
        {
            "mw_gt_500": _maybe_gt(mol_weight, 500.0),
            "mw_ge_300": _maybe_ge(mol_weight, 300.0),
            "mw_lt_300": _maybe_lt(mol_weight, 300.0),
            "logp_ge_3": _maybe_ge(logp, 3.0),
            "logp_le_3": _maybe_le(logp, 3.0),
            "tpsa_gt_75": _maybe_gt(tpsa, 75.0),
            "tpsa_le_75": _maybe_le(tpsa, 75.0),
            "hba_ge_3": _maybe_ge(hba, 3.0),
            "hba_le_2": _maybe_le(hba, 2.0),
            "hbd_le_3": _maybe_le(hbd, 3.0),
            "hbd_ge_4": _maybe_ge(hbd, 4.0),
            "oxygen_nitrogen_ge_4": _maybe_ge(oxygen_nitrogen_count, 4.0),
            "oxygen_nitrogen_le_3": _maybe_le(oxygen_nitrogen_count, 3.0),
            "basic_center_present": _flag(basic_center_present),
            "cationic_center_present": _flag(cationic_center_present),
            "tertiary_amine_present": _flag(tertiary_amine_present),
            "most_basic_pka_gt_8": _maybe_gt(features["most_basic_pka"], 8.0),
            "most_basic_pka_lt_8": _maybe_lt(features["most_basic_pka"], 8.0),
            "aromatic_ring_count_ge_2": _flag(aromatic_ring_count_ge_2),
            "aromatic_linker_present": _flag(aromatic_linker_present),
            "long_alkyl_chain_ge_6": _flag(long_alkyl_chain_ge_6),
            "bulky_hydrophobic_moiety_present": _flag(bulky_hydrophobic_moiety_present),
            "hydrophobic_large_molecule": _flag(hydrophobic_large_molecule),
            "aromatic_basic_combo_present": _flag(aromatic_basic_combo_present),
            "pgp_flowchart_high_likelihood": _flag(pgp_flowchart_high_likelihood),
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
        description="Generate Pgp_Broccatelli DeepResearch features from SMILES."
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
