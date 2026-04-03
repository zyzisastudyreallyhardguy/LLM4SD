#!/usr/bin/env python3
"""Deterministic AMES rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch AMES mutagenicity ruleset. The implemented
features focus on:

- canonical mutagenicity structural alerts such as aromatic nitro, aromatic
  amine, azo, N-nitroso, epoxide, alkyl halide, Michael acceptor, quinone,
  hydrazine, isocyanate, and alkyl O-N esters
- fused polycyclic aromatic and heteroaromatic systems that proxy intercalative
  or metabolically activated mutagenicity risk
- basic graph-level physicochemical descriptors, including the continuous logP
  signal mentioned in the source rules

Intentionally skipped:
- experimental AMES outcomes, strain-specific assay conditions, and metabolic
  activation settings such as S9 mix requirements
- exact metabolic bioactivation likelihood for aromatic amines and PAHs beyond
  direct local structural proxies
- literature-derived rule priority weights as model coefficients; this module
  exposes counts and flags instead of hard-coded endpoint predictions
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

HALOGEN_ATOMIC_NUMBERS = {9, 17, 35, 53}


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("carbon_atom_count", "Total carbon atom count."),
        ("heteroatom_count", "Total heteroatom count."),
        ("oxygen_nitrogen_count", "Count of O and N atoms."),
        ("halogen_atom_count", "Total halogen atom count."),
        ("logp", "Wildman-Crippen logP kept as a continuous Ames-related descriptor."),
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
        ("max_fused_aromatic_system_size", "Largest fused aromatic system size measured in aromatic rings."),
        ("aromatic_n_heterocycle_ring_count", "Count of aromatic rings containing at least one nitrogen atom."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("aromatic_nitro_count", "Count of aromatic nitro alerts (Ar-NO2)."),
        ("aromatic_nitro_present", "1 if an aromatic nitro alert is present."),
        ("aromatic_amine_count", "Count of primary or secondary aromatic amine nitrogens."),
        ("aromatic_amine_present", "1 if a primary or secondary aromatic amine alert is present."),
        ("azo_linkage_count", "Count of azo linkage alerts (R-N=N-R)."),
        ("azo_linkage_present", "1 if an azo linkage alert is present."),
        ("n_nitroso_count", "Count of conservative N-nitroso alerts (R2N-N=O)."),
        ("n_nitroso_present", "1 if an N-nitroso alert is present."),
        ("epoxide_ring_count", "Count of epoxide-like three-membered O-containing rings."),
        ("epoxide_present", "1 if an epoxide alert is present."),
        ("alkyl_halide_count", "Count of non-aromatic sp3 carbon-halogen alkylating motifs."),
        ("alkyl_halide_present", "1 if an alkyl halide alert is present."),
        ("michael_acceptor_count", "Count of alpha,beta-unsaturated carbonyl alerts."),
        ("michael_acceptor_present", "1 if an alpha,beta-unsaturated carbonyl alert is present."),
        (
            "alkyl_oxy_nitrogen_ester_count",
            "Count of conservative alkyl O-N ester alerts covering alkyl nitrite or nitrate-like motifs.",
        ),
        ("alkyl_oxy_nitrogen_ester_present", "1 if an alkyl O-N ester alert is present."),
        ("hydrazine_count", "Count of hydrazine-like single-bonded N-N alerts with hydrogens."),
        ("hydrazine_present", "1 if a hydrazine-like alert is present."),
        ("isocyanate_count", "Count of isocyanate alerts (R-N=C=O)."),
        ("isocyanate_present", "1 if an isocyanate alert is present."),
        ("quinone_ring_count", "Count of ortho- or para-quinone ring alerts."),
        ("quinone_present", "1 if a quinone-like alert is present."),
        ("pah_system_count", "Count of fused aromatic hydrocarbon systems with at least three aromatic rings."),
        ("pah_present", "1 if a 3+ fused-ring polycyclic aromatic hydrocarbon alert is present."),
        ("hetero_pah_system_count", "Count of fused aromatic heteroatom-containing systems with at least three aromatic rings."),
        ("hetero_pah_present", "1 if a 3+ fused-ring hetero-polyaromatic alert is present."),
        ("polycyclic_alert_present", "1 if a PAH or hetero-PAH alert is present."),
        (
            "aromatic_alert_family_count",
            "Count of direct aromatic Ames alert families matched, excluding generic polycyclic scaffold-only signals.",
        ),
        ("reactive_alert_family_count", "Count of reactive Ames alert families matched."),
        (
            "ames_alert_family_count",
            "Count of direct DeepResearch Ames structural alert families matched, with polycyclic scaffold-only signals tracked separately.",
        ),
        ("ames_structural_alert_present", "1 if any direct DeepResearch Ames structural alert is present."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Experimental AMES assay outcomes, strain-specific behavior, and S9 metabolic activation conditions are skipped because they are not recoverable from SMILES alone.",
    "Metabolic activation likelihood for aromatic amines, azo compounds, and polycyclic aromatics is reduced to direct local structural alerts instead of a separate metabolism model.",
    "The source text mentions logP as relevant but gives no validated threshold, so this module keeps logP as a continuous descriptor rather than inventing a hard cutoff.",
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


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
        from rdkit.Chem.rdchem import HybridizationType
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run AMES feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        HybridizationType=HybridizationType,
        LipinskiL=Lipinski,
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


def _count_halogen_atoms(mol) -> int:
    return sum(atom.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS for atom in mol.GetAtoms())


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


def _count_unique_matches(mol, pattern, atom_positions: tuple[int, ...] | None = None) -> int:
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if atom_positions is None:
        return len(matches)
    unique_keys = {
        tuple(sorted(match[position] for position in atom_positions))
        for match in matches
    }
    return len(unique_keys)


def _cyclic_distance(i: int, j: int, ring_size: int) -> int:
    distance = abs(i - j)
    return min(distance, ring_size - distance)


def _aromatic_ring_system_summary(mol) -> dict[str, int]:
    aromatic_rings = [
        set(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
    ]
    if not aromatic_rings:
        return {
            "fused_aromatic_ring_count": 0,
            "max_fused_aromatic_system_size": 0,
            "pah_system_count": 0,
            "hetero_pah_system_count": 0,
        }

    adjacency = [set() for _ in aromatic_rings]
    for i, ring_i in enumerate(aromatic_rings):
        for j in range(i + 1, len(aromatic_rings)):
            if len(ring_i.intersection(aromatic_rings[j])) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)

    visited: set[int] = set()
    fused_aromatic_ring_count = 0
    max_fused_aromatic_system_size = 0
    pah_system_count = 0
    hetero_pah_system_count = 0

    for start in range(len(aromatic_rings)):
        if start in visited:
            continue

        stack = [start]
        ring_indices: list[int] = []
        component_atoms: set[int] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            ring_indices.append(current)
            component_atoms.update(aromatic_rings[current])
            stack.extend(adjacency[current] - visited)

        system_size = len(ring_indices)
        max_fused_aromatic_system_size = max(max_fused_aromatic_system_size, system_size)
        if system_size >= 2:
            fused_aromatic_ring_count += system_size
        if system_size >= 3:
            if all(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6 for atom_idx in component_atoms):
                pah_system_count += 1
            elif any(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 6 for atom_idx in component_atoms):
                hetero_pah_system_count += 1

    return {
        "fused_aromatic_ring_count": fused_aromatic_ring_count,
        "max_fused_aromatic_system_size": max_fused_aromatic_system_size,
        "pah_system_count": pah_system_count,
        "hetero_pah_system_count": hetero_pah_system_count,
    }


def _count_aromatic_n_heterocycle_rings(mol) -> int:
    count = 0
    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if ring_atoms and all(atom.GetIsAromatic() for atom in ring_atoms) and any(atom.GetAtomicNum() == 7 for atom in ring_atoms):
            count += 1
    return count


def _count_epoxides(mol) -> int:
    epoxide_count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 3:
            continue
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        atomic_nums = sorted(atom.GetAtomicNum() for atom in ring_atoms)
        if atomic_nums != [6, 6, 8]:
            continue
        if all(
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % 3]).GetBondTypeAsDouble() == 1.0
            for i in range(3)
        ):
            epoxide_count += 1
    return epoxide_count


def _is_sp3_alkyl_carbon(atom) -> bool:
    rdkit = _rdkit()
    if atom.GetAtomicNum() != 6 or atom.GetIsAromatic():
        return False
    if atom.GetHybridization() != rdkit.HybridizationType.SP3:
        return False
    return all(bond.GetBondTypeAsDouble() == 1.0 for bond in atom.GetBonds())


def _count_alkyl_halides(mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS and _is_sp3_alkyl_carbon(end):
            count += 1
        elif end.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS and _is_sp3_alkyl_carbon(begin):
            count += 1
    return count


def _is_amide_like_nitrogen(atom) -> bool:
    for neighbor in atom.GetNeighbors():
        if neighbor.GetAtomicNum() != 6:
            continue
        bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            continue
        for other_neighbor in neighbor.GetNeighbors():
            if other_neighbor.GetIdx() == atom.GetIdx():
                continue
            other_bond = atom.GetOwningMol().GetBondBetweenAtoms(neighbor.GetIdx(), other_neighbor.GetIdx())
            if other_bond is None or other_bond.GetBondTypeAsDouble() != 2.0:
                continue
            if other_neighbor.GetAtomicNum() in (7, 8, 16):
                return True
    return False


def _count_aromatic_amines(mol) -> int:
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue
        if atom.GetFormalCharge() != 0:
            continue
        if atom.GetTotalNumHs() < 1:
            continue
        if _is_amide_like_nitrogen(atom):
            continue
        if any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()):
            count += 1
    return count


def _count_hydrazines(mol) -> int:
    count = 0
    seen_pairs: set[tuple[int, int]] = set()
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0:
            continue
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetAtomicNum() != 7 or end.GetAtomicNum() != 7:
            continue
        if begin.GetFormalCharge() != 0 or end.GetFormalCharge() != 0:
            continue
        if begin.GetTotalNumHs() < 1 or end.GetTotalNumHs() < 1:
            continue
        if _is_amide_like_nitrogen(begin) or _is_amide_like_nitrogen(end):
            continue
        pair = tuple(sorted((begin.GetIdx(), end.GetIdx())))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            count += 1
    return count


def _count_alkyl_oxy_nitrogen_esters(mol) -> int:
    matched_nitrogen_indices: set[int] = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 7:
            continue

        has_double_oxygen = False
        single_bond_oxygen_neighbors = []
        for neighbor in atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 8:
                continue
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            if bond is None:
                continue
            bond_order = bond.GetBondTypeAsDouble()
            if bond_order == 2.0:
                has_double_oxygen = True
            elif bond_order == 1.0:
                single_bond_oxygen_neighbors.append(neighbor)

        if not has_double_oxygen or not single_bond_oxygen_neighbors:
            continue

        for oxygen in single_bond_oxygen_neighbors:
            for oxygen_neighbor in oxygen.GetNeighbors():
                if oxygen_neighbor.GetIdx() == atom.GetIdx():
                    continue
                if _is_sp3_alkyl_carbon(oxygen_neighbor):
                    matched_nitrogen_indices.add(atom.GetIdx())
                    break
            if atom.GetIdx() in matched_nitrogen_indices:
                break

    return len(matched_nitrogen_indices)


def _count_quinone_like_rings(mol) -> int:
    rdkit = _rdkit()
    quinone_count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 6:
            continue
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if any(atom.GetAtomicNum() != 6 for atom in ring_atoms):
            continue
        carbonyl_positions: list[int] = []
        for position, atom_idx in enumerate(ring):
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetHybridization() not in (rdkit.HybridizationType.SP2, rdkit.HybridizationType.SP):
                continue
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in ring or neighbor.GetAtomicNum() != 8:
                    continue
                bond = mol.GetBondBetweenAtoms(atom_idx, neighbor.GetIdx())
                if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                    carbonyl_positions.append(position)
                    break

        if len(carbonyl_positions) != 2:
            continue
        distance = _cyclic_distance(carbonyl_positions[0], carbonyl_positions[1], 6)
        if distance not in (1, 3):
            continue
        quinone_count += 1
    return quinone_count


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.LipinskiL.NumHDonors(mol))
    hba = _as_float(rdkit.LipinskiL.NumHAcceptors(mol))
    total_hbond = hbd + hba
    ring_count = _as_float(rdkit.LipinskiL.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.LipinskiL.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.LipinskiL.NumAliphaticRings(mol))
    fraction_csp3 = _as_float(rdkit.LipinskiL.FractionCSP3(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    aromatic_ring_summary = _aromatic_ring_system_summary(mol)

    aromatic_nitro_count = _count_unique_matches(mol, _smarts("[a]-[NX3+](=O)[O-]"), (1,))
    aromatic_amine_count = _count_aromatic_amines(mol)
    azo_linkage_count = _count_unique_matches(mol, _smarts("[#6,a]-[N]=[N]-[#6,a]"), (1, 2))
    n_nitroso_count = _count_unique_matches(
        mol,
        _smarts("[NX3;!$([N]-[C,S,P]=[O,S,N])]-[N;X2]=O"),
        (1,),
    )
    epoxide_ring_count = _count_epoxides(mol)
    alkyl_halide_count = _count_alkyl_halides(mol)
    michael_acceptor_count = _count_unique_matches(
        mol,
        _smarts("[C,c]=[C,c]-[C](=O)[#6,#7,#8]"),
        (2,),
    )
    alkyl_oxy_nitrogen_ester_count = _count_alkyl_oxy_nitrogen_esters(mol)
    hydrazine_count = _count_hydrazines(mol)
    isocyanate_count = _count_unique_matches(mol, _smarts("[N]=[C]=O"), (1,))
    quinone_ring_count = _count_quinone_like_rings(mol)
    pah_system_count = aromatic_ring_summary["pah_system_count"]
    hetero_pah_system_count = aromatic_ring_summary["hetero_pah_system_count"]

    polycyclic_alert_present = (pah_system_count + hetero_pah_system_count) > 0
    aromatic_alert_family_count = float(
        sum(
            [
                aromatic_nitro_count > 0,
                aromatic_amine_count > 0,
                azo_linkage_count > 0,
            ]
        )
    )
    reactive_alert_family_count = float(
        sum(
            [
                n_nitroso_count > 0,
                epoxide_ring_count > 0,
                alkyl_halide_count > 0,
                michael_acceptor_count > 0,
                alkyl_oxy_nitrogen_ester_count > 0,
                hydrazine_count > 0,
                isocyanate_count > 0,
                quinone_ring_count > 0,
            ]
        )
    )
    ames_alert_family_count = float(
        sum(
            [
                aromatic_nitro_count > 0,
                aromatic_amine_count > 0,
                azo_linkage_count > 0,
                n_nitroso_count > 0,
                epoxide_ring_count > 0,
                alkyl_halide_count > 0,
                michael_acceptor_count > 0,
                alkyl_oxy_nitrogen_ester_count > 0,
                hydrazine_count > 0,
                isocyanate_count > 0,
                quinone_ring_count > 0,
            ]
        )
    )

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": _as_float(rdkit.Descriptors.ExactMolWt(mol)),
            "heavy_atom_count": _as_float(mol.GetNumHeavyAtoms()),
            "carbon_atom_count": _as_float(_count_atomic_number(mol, 6)),
            "heteroatom_count": _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "oxygen_nitrogen_count": _as_float(_count_oxygen_and_nitrogen(mol)),
            "halogen_atom_count": _as_float(_count_halogen_atoms(mol)),
            "logp": logp,
            "molar_refractivity": _as_float(rdkit.Crippen.MolMR(mol)),
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbond,
            "rotatable_bonds": _as_float(rdkit.LipinskiL.NumRotatableBonds(mol)),
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "fused_aromatic_ring_count": _as_float(aromatic_ring_summary["fused_aromatic_ring_count"]),
            "max_fused_aromatic_system_size": _as_float(aromatic_ring_summary["max_fused_aromatic_system_size"]),
            "aromatic_n_heterocycle_ring_count": _as_float(_count_aromatic_n_heterocycle_rings(mol)),
            "fraction_csp3": fraction_csp3,
            "formal_charge": _as_float(rdkit.Chem.GetFormalCharge(mol)),
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "aromatic_nitro_count": _as_float(aromatic_nitro_count),
            "aromatic_nitro_present": _flag(aromatic_nitro_count > 0),
            "aromatic_amine_count": _as_float(aromatic_amine_count),
            "aromatic_amine_present": _flag(aromatic_amine_count > 0),
            "azo_linkage_count": _as_float(azo_linkage_count),
            "azo_linkage_present": _flag(azo_linkage_count > 0),
            "n_nitroso_count": _as_float(n_nitroso_count),
            "n_nitroso_present": _flag(n_nitroso_count > 0),
            "epoxide_ring_count": _as_float(epoxide_ring_count),
            "epoxide_present": _flag(epoxide_ring_count > 0),
            "alkyl_halide_count": _as_float(alkyl_halide_count),
            "alkyl_halide_present": _flag(alkyl_halide_count > 0),
            "michael_acceptor_count": _as_float(michael_acceptor_count),
            "michael_acceptor_present": _flag(michael_acceptor_count > 0),
            "alkyl_oxy_nitrogen_ester_count": _as_float(alkyl_oxy_nitrogen_ester_count),
            "alkyl_oxy_nitrogen_ester_present": _flag(alkyl_oxy_nitrogen_ester_count > 0),
            "hydrazine_count": _as_float(hydrazine_count),
            "hydrazine_present": _flag(hydrazine_count > 0),
            "isocyanate_count": _as_float(isocyanate_count),
            "isocyanate_present": _flag(isocyanate_count > 0),
            "quinone_ring_count": _as_float(quinone_ring_count),
            "quinone_present": _flag(quinone_ring_count > 0),
            "pah_system_count": _as_float(pah_system_count),
            "pah_present": _flag(pah_system_count > 0),
            "hetero_pah_system_count": _as_float(hetero_pah_system_count),
            "hetero_pah_present": _flag(hetero_pah_system_count > 0),
            "polycyclic_alert_present": _flag(polycyclic_alert_present),
            "aromatic_alert_family_count": aromatic_alert_family_count,
            "reactive_alert_family_count": reactive_alert_family_count,
            "ames_alert_family_count": ames_alert_family_count,
            "ames_structural_alert_present": _flag(ames_alert_family_count > 0),
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
        description="Generate AMES DeepResearch features from SMILES."
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
