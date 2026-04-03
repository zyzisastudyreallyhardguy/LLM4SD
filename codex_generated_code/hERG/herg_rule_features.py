#!/usr/bin/env python3
"""Deterministic hERG rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch hERG ruleset. The implemented features
focus on:

- lipophilicity, polarity, size, aromaticity, flexibility, and formal charge
- amine/basic-center features plus MolGpKa-derived pKa/logD proxies
- structural alerts and 2D topological proxies for cation-pi and planar
  aromatic binding motifs associated with hERG blockade

Intentionally skipped:
- exact dipole moments, docking scores, and explicit 3D cation-pi interaction
  energies that require conformer generation or a validated channel model
- lipophilic efficiency / LLE values that require measured potency
- experimental hERG IC50 annotations or medicinal-chemistry transform effects
  that are not derivable from the molecular graph alone
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
HALOGEN_ATOMIC_NUMBERS = {9, 17, 35, 53}

BASIC_CENTER_PATTERNS = (
    "[NX4+]",
    "[NX3;H2,H1,H0;+0;!$([N]-[C,S,P]=[O,S,N])]",
    "[nH0;+0]",
)
TERTIARY_AMINE_PATTERNS = (
    "[NX3;H0;+0;!$([N]-[C,S,P]=[O,S,N])]([#6])([#6])[#6]",
)
QUATERNARY_AMMONIUM_PATTERNS = (
    "[NX4+]([#6])([#6])([#6])[#6]",
)
METHANESULFONAMIDE_PATTERNS = (
    "[CH3][SX4](=[OX1])(=[OX1])[NX3;H2,H1,H0]",
)
ARYL_GUANIDINE_PATTERNS = (
    "c[NX3][CX3](=[NX2])[NX3]",
    "c[NX3][CX3](=[NX2+])[NX3]",
)
PHENYL_PIPERAZINE_PATTERNS = (
    "c1ccccc1N1CCNCC1",
    "c1ccccc1CN1CCNCC1",
    "c1ccccc1CCN1CCNCC1",
)


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("carbon_atom_count", "Carbon atom count."),
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
        ("fused_ring_count", "Count of rings participating in fused ring systems."),
        ("fused_aromatic_ring_count", "Count of aromatic rings participating in fused aromatic systems."),
        ("aromatic_atom_fraction", "Fraction of heavy atoms that are aromatic."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("halogen_count", "Total halogen atom count (F/Cl/Br/I)."),
        ("fluorine_count", "Fluorine atom count."),
        ("chlorine_count", "Chlorine atom count."),
        ("bromine_count", "Bromine atom count."),
        ("iodine_count", "Iodine atom count."),
        ("basic_center_count", "Count of amine-like or aromatic-basic nitrogen centers from local SMARTS heuristics."),
        ("basic_heterocycle_count", "Count of aromatic heterocycles containing at least one pyridine-like aromatic nitrogen."),
        ("tertiary_amine_count", "Count of tertiary amine motifs excluding amide-like nitrogens."),
        ("quaternary_ammonium_count", "Count of quaternary ammonium centers."),
        ("most_basic_pka", "Most basic predicted pKa from the MolGpKa helper."),
        ("num_basic_sites", "Number of predicted basic sites from the MolGpKa helper."),
        ("neutral_fraction_ph74", "Estimated neutral fraction at pH 7.4."),
        ("charged_fraction_ph74", "1 - neutral_fraction_ph74."),
        ("base_protonated_fraction_ph74", "Estimated protonated fraction for the dominant basic site at pH 7.4."),
        ("estimated_logd_ph74", "Estimated logD at pH 7.4 from logP and neutral fraction."),
        ("combined_logp_pka_sq", "Combined hERG heuristic metric: logP^2 + pKa^2."),
        ("min_basic_center_distance", "Minimum graph distance between two basic-center atoms."),
        ("min_basic_to_aromatic_distance", "Minimum graph distance from a basic-center atom to a different aromatic atom."),
        ("methanesulfonamide_count", "Count of methanesulfonamide motifs."),
        ("thiazole_count", "Count of aromatic 5-membered rings containing both S and a pyridine-like aromatic N."),
        ("quinoline_isoquinoline_like_count", "Count of fused bicyclic aza-aromatic systems with one ring nitrogen and two fused aromatic rings."),
        ("aryl_guanidine_count", "Count of aryl-guanidine alert motifs."),
        ("phenyl_piperazine_count", "Count of conservative phenyl-piperazine alert motifs."),
        ("structural_alert_count", "Number of distinct hERG-oriented structural alert families matched."),
        ("property_window_violation_count", "Count of violated core hERG-mitigation property windows."),
        ("herg_risk_alert_count", "Count of matched discrete hERG-risk alert families."),
        ("logp_lt_4", "Rule flag: logP < 4."),
        ("estimated_logd_lt_3", "Rule flag: estimated logD(7.4) < 3."),
        ("combined_logp_pka_sq_lt_110", "Rule flag: logP^2 + pKa^2 < 110."),
        ("most_basic_pka_le_8", "Rule flag: most basic predicted pKa <= 8."),
        ("tpsa_gt_75", "Rule flag: TPSA > 75."),
        ("hbd_ge_2", "Rule flag: H-bond donor count >= 2."),
        ("hba_ge_5", "Rule flag: H-bond acceptor count >= 5."),
        ("mw_le_500", "Rule flag: molecular weight <= 500."),
        ("aromatic_ring_count_le_3", "Rule flag: aromatic ring count <= 3."),
        ("fused_bicyclic_aromatic_absent", "Rule flag: no fused aromatic ring system is present."),
        ("rotatable_bonds_gt_5", "Rule flag: rotatable bond count > 5."),
        ("num_basic_sites_le_2", "Rule flag: predicted basic-site count <= 2."),
        ("basic_center_count_le_2", "Rule flag: heuristic basic-center count <= 2."),
        ("multiple_halogens_present", "Rule flag: at least two F/Cl/Br/I atoms are present."),
        ("positive_charge_present_ph74", "Rule flag: positive charge is expected at pH 7.4 from formal charge or dominant basic-site protonation."),
        ("tertiary_amine_present", "Rule flag: at least one tertiary amine motif is present."),
        ("basic_heterocycle_present", "Rule flag: at least one aromatic basic heterocycle is present."),
        ("quaternary_ammonium_present", "Rule flag: at least one quaternary ammonium center is present."),
        ("cation_pi_proxy_present", "Heuristic flag: a basic center lies within four graph bonds of a different aromatic atom."),
        ("short_basic_center_distance_present", "Heuristic flag: two basic centers are separated by six or fewer graph bonds."),
        ("long_basic_to_aromatic_distance_present", "Heuristic flag: the nearest basic-center to aromatic-atom distance is greater than five graph bonds."),
        ("methanesulfonamide_present", "Rule flag: at least one methanesulfonamide motif is present."),
        ("thiazole_present", "Rule flag: at least one thiazole-like aromatic ring is present."),
        ("quinoline_isoquinoline_like_present", "Rule flag: at least one quinoline/isoquinoline-like fused aza-aromatic system is present."),
        ("aryl_guanidine_present", "Rule flag: at least one aryl-guanidine alert is present."),
        ("phenyl_piperazine_present", "Rule flag: at least one phenyl-piperazine alert is present."),
        ("formal_charge_abs_le_1", "Rule flag: absolute formal charge <= 1."),
        ("low_planarity_proxy", "Heuristic flag favoring less planar structures: fraction_csp3 >= 0.3, no fused aromatic rings, and <=3 aromatic rings."),
        ("herg_property_window_pass", "Composite flag: all core hERG-mitigation property windows are satisfied."),
        ("herg_structural_alert_present", "Composite flag: at least one explicit hERG-oriented structural alert family is present."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Exact dipole moments and other quantum-chemical descriptors are skipped because they require a validated conformer or QM workflow not present here.",
    "Lipophilic efficiency / LLE and other potency-normalized metrics are skipped because they require measured activity values such as hERG IC50.",
    "Explicit docking, cation-pi energy, and channel-pore geometry calculations are reduced to conservative 2D topological proxies rather than modeled directly.",
    "Medicinal-chemistry transform effects such as matched-pair changes after adding O, sp2 N, or beta-hydroxyl groups are not hard-coded as pseudo-labels.",
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


def _maybe_lt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value < threshold)


def _maybe_le(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value <= threshold)


def _maybe_gt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value > threshold)


def _maybe_ge(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value >= threshold)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run hERG feature generation. "
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


@lru_cache(maxsize=1)
def _get_pka_predictor():
    if str(INTERN_S1_ROOT) not in sys.path:
        sys.path.insert(0, str(INTERN_S1_ROOT))

    from tools.pka_related_tools import _get_pka_predictor as _load_predictor

    return _load_predictor()


def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    mol = _rdkit().Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _count_atomic_number(mol, atomic_num: int) -> int:
    return sum(atom.GetAtomicNum() == atomic_num for atom in mol.GetAtoms())


def _count_atomic_numbers(mol, atomic_numbers: set[int]) -> int:
    return sum(atom.GetAtomicNum() in atomic_numbers for atom in mol.GetAtoms())


def _count_oxygen_and_nitrogen(mol) -> int:
    return sum(atom.GetAtomicNum() in (7, 8) for atom in mol.GetAtoms())


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


def _ring_atom_sets(mol) -> list[set[int]]:
    return [set(ring) for ring in mol.GetRingInfo().AtomRings()]


def _is_aromatic_ring(mol, ring_atom_indices: set[int]) -> bool:
    return bool(ring_atom_indices) and all(
        mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_atom_indices
    )


def _count_fused_rings(mol) -> int:
    atom_rings = _ring_atom_sets(mol)
    if not atom_rings:
        return 0

    fused_ring_indices = set()
    for i, ring_i in enumerate(atom_rings):
        for j in range(i + 1, len(atom_rings)):
            if len(ring_i.intersection(ring_j := atom_rings[j])) >= 2:
                fused_ring_indices.add(i)
                fused_ring_indices.add(j)
    return len(fused_ring_indices)


def _count_fused_aromatic_rings(mol) -> int:
    aromatic_rings = [ring for ring in _ring_atom_sets(mol) if _is_aromatic_ring(mol, ring)]
    if not aromatic_rings:
        return 0

    fused_ring_indices = set()
    for i, ring_i in enumerate(aromatic_rings):
        for j in range(i + 1, len(aromatic_rings)):
            if len(ring_i.intersection(ring_j := aromatic_rings[j])) >= 2:
                fused_ring_indices.add(i)
                fused_ring_indices.add(j)
    return len(fused_ring_indices)


def _aromatic_atom_fraction(mol) -> float:
    heavy_atoms = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return aromatic_atoms / heavy_atoms


def _match_atom_indices(mol, patterns: Iterable[str]) -> list[int]:
    atom_indices: set[int] = set()
    for pattern in patterns:
        for match in mol.GetSubstructMatches(_smarts(pattern), uniquify=True):
            atom_indices.add(match[0])
    return sorted(atom_indices)


def _count_union_matches(
    mol,
    patterns: Iterable[str],
    atom_positions: tuple[int, ...] | None = None,
) -> int:
    unique_keys: set[tuple[int, ...]] = set()
    for pattern in patterns:
        for match in mol.GetSubstructMatches(_smarts(pattern), uniquify=True):
            if atom_positions is None:
                key = tuple(sorted(match))
            else:
                key = tuple(sorted(match[position] for position in atom_positions))
            unique_keys.add(key)
    return len(unique_keys)


def _count_basic_heterocycles(mol) -> int:
    count = 0
    for ring in _ring_atom_sets(mol):
        if not _is_aromatic_ring(mol, ring):
            continue
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if (
                atom.GetAtomicNum() == 7
                and atom.GetIsAromatic()
                and atom.GetFormalCharge() == 0
                and atom.GetTotalNumHs() == 0
            ):
                count += 1
                break
    return count


def _count_thiazole_like_rings(mol) -> int:
    count = 0
    for ring in _ring_atom_sets(mol):
        if len(ring) != 5 or not _is_aromatic_ring(mol, ring):
            continue
        sulfur_count = 0
        pyridine_like_n_count = 0
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetAtomicNum() == 16 and atom.GetIsAromatic():
                sulfur_count += 1
            if (
                atom.GetAtomicNum() == 7
                and atom.GetIsAromatic()
                and atom.GetFormalCharge() == 0
                and atom.GetTotalNumHs() == 0
            ):
                pyridine_like_n_count += 1
        if sulfur_count >= 1 and pyridine_like_n_count >= 1:
            count += 1
    return count


def _count_quinoline_isoquinoline_like_systems(mol) -> int:
    aromatic_rings = [ring for ring in _ring_atom_sets(mol) if _is_aromatic_ring(mol, ring)]
    unique_systems: set[tuple[int, ...]] = set()
    for i, ring_i in enumerate(aromatic_rings):
        for j in range(i + 1, len(aromatic_rings)):
            ring_j = aromatic_rings[j]
            if len(ring_i.intersection(ring_j)) < 2:
                continue
            union = ring_i.union(ring_j)
            if len(union) not in (9, 10):
                continue
            aromatic_n_count = sum(
                1
                for atom_idx in union
                if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 7
                and mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
            )
            aromatic_s_count = sum(
                1
                for atom_idx in union
                if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 16
                and mol.GetAtomWithIdx(atom_idx).GetIsAromatic()
            )
            if aromatic_n_count == 1 and aromatic_s_count == 0:
                unique_systems.add(tuple(sorted(union)))
    return len(unique_systems)


def _min_pairwise_distance(mol, atom_indices: list[int]) -> float:
    if len(atom_indices) < 2:
        return math.nan
    distance_matrix = _rdkit().Chem.GetDistanceMatrix(mol)
    min_distance = math.inf
    for i, atom_idx_i in enumerate(atom_indices):
        for atom_idx_j in atom_indices[i + 1 :]:
            min_distance = min(min_distance, float(distance_matrix[atom_idx_i][atom_idx_j]))
    return math.nan if math.isinf(min_distance) else float(min_distance)


def _min_distance_between_sets(
    mol,
    source_indices: list[int],
    target_indices: list[int],
) -> float:
    if not source_indices or not target_indices:
        return math.nan
    distance_matrix = _rdkit().Chem.GetDistanceMatrix(mol)
    min_distance = math.inf
    for source_idx in source_indices:
        for target_idx in target_indices:
            if source_idx == target_idx:
                continue
            min_distance = min(min_distance, float(distance_matrix[source_idx][target_idx]))
    return math.nan if math.isinf(min_distance) else float(min_distance)


def _compute_pka_summary(mol, logp: float) -> dict[str, float]:
    result = {
        "most_basic_pka": math.nan,
        "num_basic_sites": math.nan,
        "neutral_fraction_ph74": math.nan,
        "charged_fraction_ph74": math.nan,
        "base_protonated_fraction_ph74": math.nan,
        "estimated_logd_ph74": math.nan,
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
    num_basic_sites = float(len(base_sites))

    base_protonated_fraction = 0.0
    if base_sites:
        base_protonated_fraction = 1.0 / (1.0 + 10.0 ** (PHYSIOLOGICAL_PH - most_basic_pka))

    acid_deprotonated_fraction = 0.0
    if acid_sites:
        most_acidic_pka = min(acid_sites.values())
        acid_deprotonated_fraction = 1.0 / (1.0 + 10.0 ** (most_acidic_pka - PHYSIOLOGICAL_PH))

    neutral_fraction = 1.0
    if base_sites:
        neutral_fraction *= 1.0 - base_protonated_fraction
    if acid_sites:
        neutral_fraction *= 1.0 - acid_deprotonated_fraction
    neutral_fraction = min(1.0, max(1e-12, neutral_fraction))

    result.update(
        {
            "most_basic_pka": _as_float(most_basic_pka) if base_sites else math.nan,
            "num_basic_sites": num_basic_sites,
            "neutral_fraction_ph74": neutral_fraction,
            "charged_fraction_ph74": 1.0 - neutral_fraction,
            "base_protonated_fraction_ph74": base_protonated_fraction,
            "estimated_logd_ph74": logp + math.log10(neutral_fraction),
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
    total_hbond = hbd + hba
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.Lipinski.NumAliphaticRings(mol))
    fused_ring_count = _as_float(_count_fused_rings(mol))
    fused_aromatic_ring_count = _as_float(_count_fused_aromatic_rings(mol))
    aromatic_atom_fraction = _as_float(_aromatic_atom_fraction(mol))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    halogen_count = _as_float(_count_atomic_numbers(mol, HALOGEN_ATOMIC_NUMBERS))
    fluorine_count = _as_float(_count_atomic_number(mol, 9))
    chlorine_count = _as_float(_count_atomic_number(mol, 17))
    bromine_count = _as_float(_count_atomic_number(mol, 35))
    iodine_count = _as_float(_count_atomic_number(mol, 53))

    basic_center_atom_indices = _match_atom_indices(mol, BASIC_CENTER_PATTERNS)
    tertiary_amine_count = _as_float(_count_union_matches(mol, TERTIARY_AMINE_PATTERNS, atom_positions=(0,)))
    quaternary_ammonium_count = _as_float(
        _count_union_matches(mol, QUATERNARY_AMMONIUM_PATTERNS, atom_positions=(0,))
    )
    basic_heterocycle_count = _as_float(_count_basic_heterocycles(mol))
    min_basic_center_distance = _min_pairwise_distance(mol, basic_center_atom_indices)

    aromatic_atom_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic() and atom.GetAtomicNum() > 1
    ]
    min_basic_to_aromatic_distance = _min_distance_between_sets(
        mol,
        basic_center_atom_indices,
        aromatic_atom_indices,
    )

    methanesulfonamide_count = _as_float(_count_union_matches(mol, METHANESULFONAMIDE_PATTERNS))
    thiazole_count = _as_float(_count_thiazole_like_rings(mol))
    quinoline_isoquinoline_like_count = _as_float(_count_quinoline_isoquinoline_like_systems(mol))
    aryl_guanidine_count = _as_float(_count_union_matches(mol, ARYL_GUANIDINE_PATTERNS))
    phenyl_piperazine_count = _as_float(_count_union_matches(mol, PHENYL_PIPERAZINE_PATTERNS))

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
            "total_hbond_donors_acceptors": total_hbond,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "fused_ring_count": fused_ring_count,
            "fused_aromatic_ring_count": fused_aromatic_ring_count,
            "aromatic_atom_fraction": aromatic_atom_fraction,
            "fraction_csp3": fraction_csp3,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "halogen_count": halogen_count,
            "fluorine_count": fluorine_count,
            "chlorine_count": chlorine_count,
            "bromine_count": bromine_count,
            "iodine_count": iodine_count,
            "basic_center_count": _as_float(len(basic_center_atom_indices)),
            "basic_heterocycle_count": basic_heterocycle_count,
            "tertiary_amine_count": tertiary_amine_count,
            "quaternary_ammonium_count": quaternary_ammonium_count,
            "min_basic_center_distance": min_basic_center_distance,
            "min_basic_to_aromatic_distance": min_basic_to_aromatic_distance,
            "methanesulfonamide_count": methanesulfonamide_count,
            "thiazole_count": thiazole_count,
            "quinoline_isoquinoline_like_count": quinoline_isoquinoline_like_count,
            "aryl_guanidine_count": aryl_guanidine_count,
            "phenyl_piperazine_count": phenyl_piperazine_count,
        }
    )

    features.update(_compute_pka_summary(mol, logp))

    combined_logp_pka_sq = math.nan
    if not _is_missing(features["most_basic_pka"]):
        combined_logp_pka_sq = logp ** 2 + features["most_basic_pka"] ** 2

    positive_charge_present_ph74 = None
    if not _is_missing(features["base_protonated_fraction_ph74"]):
        positive_charge_present_ph74 = (
            features["base_protonated_fraction_ph74"] >= 0.5
            or positive_formal_atom_count > 0
        )
    elif positive_formal_atom_count >= 1:
        positive_charge_present_ph74 = True

    cation_pi_proxy_present = None
    if not _is_missing(min_basic_to_aromatic_distance):
        cation_pi_proxy_present = min_basic_to_aromatic_distance <= 4.0

    short_basic_center_distance_present = None
    if not _is_missing(min_basic_center_distance):
        short_basic_center_distance_present = min_basic_center_distance <= 6.0

    long_basic_to_aromatic_distance_present = None
    if not _is_missing(min_basic_to_aromatic_distance):
        long_basic_to_aromatic_distance_present = min_basic_to_aromatic_distance > 5.0

    structural_alert_families = [
        methanesulfonamide_count > 0,
        thiazole_count > 0,
        quinoline_isoquinoline_like_count > 0,
        aryl_guanidine_count > 0,
        phenyl_piperazine_count > 0,
        quaternary_ammonium_count > 0,
    ]
    structural_alert_count = float(sum(structural_alert_families))

    property_checks = [
        logp < 4.0,
        tpsa > 75.0,
        hbd >= 2.0,
        hba >= 5.0,
        mol_weight <= 500.0,
        aromatic_ring_count <= 3.0,
        fused_aromatic_ring_count == 0.0,
        rotatable_bonds > 5.0,
        len(basic_center_atom_indices) <= 2,
        halogen_count < 2.0,
        abs(formal_charge) <= 1.0,
    ]
    property_window_violation_count = float(sum(not check for check in property_checks))
    herg_property_window_pass = all(property_checks)

    herg_risk_alert_count = float(
        sum(
            bool(flag)
            for flag in [
                halogen_count >= 2.0,
                positive_charge_present_ph74,
                tertiary_amine_count > 0,
                basic_heterocycle_count > 0,
                quaternary_ammonium_count > 0,
                cation_pi_proxy_present,
                methanesulfonamide_count > 0,
                thiazole_count > 0,
                quinoline_isoquinoline_like_count > 0,
                aryl_guanidine_count > 0,
                phenyl_piperazine_count > 0,
                fused_aromatic_ring_count > 0,
            ]
        )
    )

    features.update(
        {
            "combined_logp_pka_sq": combined_logp_pka_sq,
            "structural_alert_count": structural_alert_count,
            "property_window_violation_count": property_window_violation_count,
            "herg_risk_alert_count": herg_risk_alert_count,
            "logp_lt_4": _maybe_lt(logp, 4.0),
            "estimated_logd_lt_3": _maybe_lt(features["estimated_logd_ph74"], 3.0),
            "combined_logp_pka_sq_lt_110": _maybe_lt(combined_logp_pka_sq, 110.0),
            "most_basic_pka_le_8": _maybe_le(features["most_basic_pka"], 8.0),
            "tpsa_gt_75": _maybe_gt(tpsa, 75.0),
            "hbd_ge_2": _maybe_ge(hbd, 2.0),
            "hba_ge_5": _maybe_ge(hba, 5.0),
            "mw_le_500": _maybe_le(mol_weight, 500.0),
            "aromatic_ring_count_le_3": _maybe_le(aromatic_ring_count, 3.0),
            "fused_bicyclic_aromatic_absent": _flag(fused_aromatic_ring_count == 0.0),
            "rotatable_bonds_gt_5": _maybe_gt(rotatable_bonds, 5.0),
            "num_basic_sites_le_2": _maybe_le(features["num_basic_sites"], 2.0),
            "basic_center_count_le_2": _maybe_le(float(len(basic_center_atom_indices)), 2.0),
            "multiple_halogens_present": _flag(halogen_count >= 2.0),
            "positive_charge_present_ph74": _flag(positive_charge_present_ph74),
            "tertiary_amine_present": _flag(tertiary_amine_count > 0.0),
            "basic_heterocycle_present": _flag(basic_heterocycle_count > 0.0),
            "quaternary_ammonium_present": _flag(quaternary_ammonium_count > 0.0),
            "cation_pi_proxy_present": _flag(cation_pi_proxy_present),
            "short_basic_center_distance_present": _flag(short_basic_center_distance_present),
            "long_basic_to_aromatic_distance_present": _flag(long_basic_to_aromatic_distance_present),
            "methanesulfonamide_present": _flag(methanesulfonamide_count > 0.0),
            "thiazole_present": _flag(thiazole_count > 0.0),
            "quinoline_isoquinoline_like_present": _flag(quinoline_isoquinoline_like_count > 0.0),
            "aryl_guanidine_present": _flag(aryl_guanidine_count > 0.0),
            "phenyl_piperazine_present": _flag(phenyl_piperazine_count > 0.0),
            "formal_charge_abs_le_1": _flag(abs(formal_charge) <= 1.0),
            "low_planarity_proxy": _flag(
                fraction_csp3 >= 0.3
                and fused_aromatic_ring_count == 0.0
                and aromatic_ring_count <= 3.0
            ),
            "herg_property_window_pass": _flag(herg_property_window_pass),
            "herg_structural_alert_present": _flag(structural_alert_count > 0.0),
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
        description="Generate hERG DeepResearch features from SMILES."
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
