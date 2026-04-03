#!/usr/bin/env python3
"""Deterministic CYP3A4 substrate rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch CYP3A4 substrate ruleset. The implemented
features focus on:

- large, lipophilic, aromatic, and flexible physchem windows highlighted in
  the source response
- heterocycle, tertiary-amine/basic-center, charge, and MolGpKa-derived pKa
  or logD proxies that capture the "neutral or weak base" guidance
- halogen and fused-ring scaffold proxies, including a conservative
  steroid-like fused-ring heuristic to represent the stated CYP3A7-like
  counterexample

Intentionally skipped:
- exact active-site contacts to Phe108, Ser119, Leu211, Phe304, and Thr309,
  because this repository does not ship a validated CYP3A4 docking workflow
- quantum-mechanical polarizability or explicit 3D flatness calculations;
  these are approximated with heterocycle, halogen, aromaticity, and simple
  planarity proxies instead
- experimental turnover, affinity, and curated substrate labels beyond what
  can be derived deterministically from the molecular graph
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
MISSING_PKA_SENTINEL = PHYSIOLOGICAL_PH

BASIC_CENTER_PATTERNS = (
    "[NX4+]",
    "[NX3;H2,H1,H0;+0;!$([N]-[C,S,P]=[O,S,N])]",
    "[nH0;+0]",
)
PRIMARY_AMINE_PATTERNS = (
    "[NX3;H2;+0;!$([N]-[C,S,P]=[O,S,N])][#6]",
)
SECONDARY_AMINE_PATTERNS = (
    "[NX3;H1;+0;!$([N]-[C,S,P]=[O,S,N])]([#6])[#6]",
)
TERTIARY_AMINE_PATTERNS = (
    "[NX3;H0;+0;!$([N]-[C,S,P]=[O,S,N])]([#6])([#6])[#6]",
)
QUATERNARY_AMMONIUM_PATTERNS = (
    "[NX4+]([#6])([#6])([#6])[#6]",
)
CARBOXYLIC_ACID_PATTERNS = (
    "[CX3](=O)[OX2H1]",
    "[CX3](=O)[O-]",
)
SULFONIC_ACID_PATTERNS = (
    "[SX4](=[OX1])(=[OX1])[OX2H1]",
    "[SX4](=[OX1])(=[OX1])[O-]",
)


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("carbon_atom_count", "Carbon atom count."),
        ("heteroatom_count", "Total heteroatom count."),
        ("oxygen_nitrogen_count", "Count of O and N atoms."),
        ("halogen_atom_count", "Count of F, Cl, Br, and I atoms."),
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("labute_asa", "Labute approximate surface area as a stable size or volume proxy."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("aliphatic_ring_count", "Aliphatic ring count."),
        ("heterocycle_ring_count", "Count of rings containing at least one heteroatom."),
        ("aromatic_heterocycle_ring_count", "Count of aromatic rings containing at least one heteroatom."),
        ("fused_ring_count", "Count of rings participating in fused ring systems."),
        ("largest_fused_ring_system_ring_count", "Number of rings in the largest fused ring system."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("aromatic_atom_fraction", "Fraction of heavy atoms that are aromatic."),
        ("ring_to_rotatable_ratio", "Ring count divided by max(rotatable bonds, 1) as a semi-rigidity proxy."),
        ("planarity_proxy_score", "Aromatic atom fraction minus fractionCSP3 as a simple flatness proxy."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("basic_center_count", "Count of local amine-like or aromatic-basic nitrogen centers."),
        ("primary_amine_count", "Count of primary amine motifs excluding amide-like nitrogens."),
        ("secondary_amine_count", "Count of secondary amine motifs excluding amide-like nitrogens."),
        ("tertiary_amine_count", "Count of tertiary amine motifs excluding amide-like nitrogens."),
        ("quaternary_ammonium_count", "Count of quaternary ammonium centers."),
        ("aromatic_basic_heterocycle_count", "Count of aromatic heterocycles containing at least one pyridine-like aromatic nitrogen."),
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
        ("carboxylic_acid_count", "Count of carboxylic acid or carboxylate motifs."),
        ("sulfonic_acid_count", "Count of sulfonic acid or sulfonate motifs."),
        ("strong_acidic_group_family_count", "Number of strong acidic motif families present among carboxylates and sulfonates."),
        ("aryl_halide_count", "Count of halogen atoms directly attached to aromatic atoms."),
        ("halogen_substituent_present", "1 if any halogen atom is present."),
        ("aryl_halide_present", "1 if any halogen is directly attached to an aromatic atom."),
        ("heterocycle_present", "1 if any heterocycle ring is present."),
        ("tertiary_amine_or_basic_center_present", "1 if a tertiary amine, aromatic basic heterocycle, heuristic basic center, or predicted basic site is present."),
        ("steroid_like_ring_system_count", "Count of conservative fused 6-6-6-5 steroid-like ring systems."),
        ("steroid_like_scaffold_present", "1 if a conservative fused 6-6-6-5 steroid-like ring system is present."),
        ("mw_gt_400", "Rule flag: molecular weight > 400."),
        ("logp_gt_3", "Rule flag: logP > 3."),
        ("estimated_logd_gt_3", "Rule flag: estimated logD(7.4) > 3."),
        ("tpsa_lt_100", "Rule flag: TPSA < 100."),
        ("hbd_le_2", "Rule flag: HBD <= 2."),
        ("hba_in_4_6", "Rule flag: 4 <= HBA <= 6."),
        ("rotatable_bonds_gt_6", "Rule flag: rotatable bonds > 6."),
        ("aromatic_ring_count_ge_2", "Rule flag: aromatic ring count >= 2."),
        ("aromatic_ring_count_ge_3", "Rule flag: aromatic ring count >= 3."),
        ("lipophilic_pi_scaffold_present", "Heuristic flag for a lipophilic multi-aromatic scaffold with a non-sp3-biased flatness proxy."),
        ("large_flexible_scaffold_present", "Heuristic flag for a relatively large and flexible scaffold using MW > 350 and rotatable bonds > 6."),
        ("non_strong_acidic_profile", "1 if the molecule lacks strong acidic or anionic signatures that are uncommon for CYP3A4 substrates."),
        ("neutral_or_weak_base_profile", "1 if the molecule appears neutral or at most weakly basic rather than strongly acidic or permanently cationic."),
        ("high_polarity_penalty", "Heuristic penalty for excessive polarity beyond the stated CYP3A4 substrate windows."),
        ("cyp3a4_primary_rule_pass_count", "Count of passed primary CYP3A4 heuristic checks across size, lipophilicity, polarity, flexibility, aromaticity, heterocycles, charge, and steroid-like penalties."),
        ("cyp3a4_substrate_heuristic_pass", "Composite heuristic screen for large, lipophilic, aromatic, flexible, non-acidic CYP3A4 substrate-like molecules."),
        ("pka_features_available", "1 if MolGpKa-derived pKa or logD features were computed."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Exact CYP3A4 residue-contact geometry and docking interactions are skipped because this repository does not provide a validated CYP3A4 structural workflow.",
    "Polarizability and 3D flatness are not computed with quantum or conformer models; aromaticity, halogenation, heterocycles, and simple 2D planarity proxies are used instead.",
    "Experimental turnover, inhibition, and curated substrate probabilities are skipped because they require assay data or external trained models beyond deterministic graph features.",
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


def _maybe_gt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value > threshold)


def _maybe_ge(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value >= threshold)


def _maybe_lt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value < threshold)


def _maybe_le(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value <= threshold)


def _maybe_between(value: float | None, lower: float, upper: float, inclusive: bool = True) -> float:
    if _is_missing(value):
        return math.nan
    if inclusive:
        return _flag(lower <= value <= upper)
    return _flag(lower < value < upper)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run CYP3A4 feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        LipinskiL=Lipinski,
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


def _count_halogen_atoms(mol) -> int:
    return sum(atom.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS for atom in mol.GetAtoms())


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


def _count_heterocycle_rings(mol, *, aromatic_only: bool = False) -> int:
    count = 0
    for ring in _ring_atom_sets(mol):
        if aromatic_only and not _is_aromatic_ring(mol, ring):
            continue
        if any(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() != 6 for atom_idx in ring):
            count += 1
    return count


def _count_fused_rings(mol) -> int:
    atom_rings = _ring_atom_sets(mol)
    if not atom_rings:
        return 0

    fused_ring_indices = set()
    for i, ring_i in enumerate(atom_rings):
        for j in range(i + 1, len(atom_rings)):
            if len(ring_i.intersection(atom_rings[j])) >= 2:
                fused_ring_indices.add(i)
                fused_ring_indices.add(j)
    return len(fused_ring_indices)


def _fused_ring_components(mol) -> list[list[set[int]]]:
    atom_rings = _ring_atom_sets(mol)
    if not atom_rings:
        return []

    adjacency = {idx: set() for idx in range(len(atom_rings))}
    for i, ring_i in enumerate(atom_rings):
        for j in range(i + 1, len(atom_rings)):
            if len(ring_i.intersection(atom_rings[j])) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)

    components: list[list[set[int]]] = []
    seen: set[int] = set()
    for start_idx in range(len(atom_rings)):
        if start_idx in seen:
            continue
        stack = [start_idx]
        component_indices: list[int] = []
        while stack:
            idx = stack.pop()
            if idx in seen:
                continue
            seen.add(idx)
            component_indices.append(idx)
            stack.extend(sorted(adjacency[idx] - seen))
        if len(component_indices) > 1:
            components.append([atom_rings[idx] for idx in component_indices])
    return components


def _largest_fused_ring_system_ring_count(mol) -> int:
    components = _fused_ring_components(mol)
    if not components:
        return 0
    return max(len(component) for component in components)


def _is_steroid_like_component(mol, component: list[set[int]]) -> bool:
    if len(component) != 4:
        return False

    ring_sizes = sorted(len(ring) for ring in component)
    if ring_sizes != [5, 6, 6, 6]:
        return False

    aromatic_ring_count = sum(_is_aromatic_ring(mol, ring) for ring in component)
    if aromatic_ring_count > 1:
        return False

    component_atom_indices = set().union(*component)
    carbon_indices = [
        atom_idx
        for atom_idx in component_atom_indices
        if mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6
    ]
    if not carbon_indices:
        return False

    sp3_fraction = sum(
        str(mol.GetAtomWithIdx(atom_idx).GetHybridization()) == "SP3"
        for atom_idx in carbon_indices
    ) / len(carbon_indices)
    return sp3_fraction >= 0.45


def _count_steroid_like_ring_systems(mol) -> int:
    return sum(_is_steroid_like_component(mol, component) for component in _fused_ring_components(mol))


def _aromatic_atom_fraction(mol) -> float:
    heavy_atoms = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return aromatic_atoms / heavy_atoms


def _match_atom_indices(mol, patterns: Iterable[str]) -> list[int]:
    atom_indices: set[int] = set()
    for pattern in patterns:
        for match in mol.GetSubstructMatches(_smarts(pattern), uniquify=True):
            if match:
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


def _count_aromatic_basic_heterocycles(mol) -> int:
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


def _count_aryl_halides(mol) -> int:
    count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in HALOGEN_ATOMIC_NUMBERS:
            continue
        if any(neighbor.GetIsAromatic() for neighbor in atom.GetNeighbors()):
            count += 1
    return count


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float(numerator)
    return float(numerator / denominator)


def _compute_pka_summary(mol, logp: float) -> dict[str, float]:
    result = {
        "most_basic_pka": MISSING_PKA_SENTINEL,
        "most_acidic_pka": MISSING_PKA_SENTINEL,
        "num_basic_sites": math.nan,
        "num_acidic_sites": math.nan,
        "neutral_fraction_ph74": math.nan,
        "charged_fraction_ph74": math.nan,
        "base_protonated_fraction_ph74": math.nan,
        "acid_deprotonated_fraction_ph74": math.nan,
        "net_charge_proxy_ph74": math.nan,
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

    result.update(
        {
            "most_basic_pka": _as_float(most_basic_pka) if base_sites else MISSING_PKA_SENTINEL,
            "most_acidic_pka": _as_float(most_acidic_pka) if acid_sites else MISSING_PKA_SENTINEL,
            "num_basic_sites": num_basic_sites,
            "num_acidic_sites": num_acidic_sites,
            "neutral_fraction_ph74": neutral_fraction,
            "charged_fraction_ph74": 1.0 - neutral_fraction,
            "base_protonated_fraction_ph74": base_protonated_fraction,
            "acid_deprotonated_fraction_ph74": acid_deprotonated_fraction,
            "net_charge_proxy_ph74": base_protonated_fraction - acid_deprotonated_fraction,
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
    halogen_atom_count = _as_float(_count_halogen_atoms(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    molar_refractivity = _as_float(rdkit.Crippen.MolMR(mol))
    labute_asa = _as_float(rdkit.rdMolDescriptors.CalcLabuteASA(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.LipinskiL.NumHDonors(mol))
    hba = _as_float(rdkit.LipinskiL.NumHAcceptors(mol))
    total_hbond = hbd + hba
    rotatable_bonds = _as_float(rdkit.LipinskiL.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.LipinskiL.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.LipinskiL.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.LipinskiL.NumAliphaticRings(mol))
    heterocycle_ring_count = _as_float(_count_heterocycle_rings(mol))
    aromatic_heterocycle_ring_count = _as_float(_count_heterocycle_rings(mol, aromatic_only=True))
    fused_ring_count = _as_float(_count_fused_rings(mol))
    largest_fused_ring_system_ring_count = _as_float(_largest_fused_ring_system_ring_count(mol))
    fraction_csp3 = _as_float(rdkit.LipinskiL.FractionCSP3(mol))
    aromatic_atom_fraction = _as_float(_aromatic_atom_fraction(mol))
    ring_to_rotatable_ratio = _safe_ratio(ring_count, rotatable_bonds if rotatable_bonds > 0 else 1.0)
    planarity_proxy_score = aromatic_atom_fraction - fraction_csp3
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)

    basic_center_atom_indices = _match_atom_indices(mol, BASIC_CENTER_PATTERNS)
    primary_amine_count = _as_float(_count_union_matches(mol, PRIMARY_AMINE_PATTERNS, atom_positions=(0,)))
    secondary_amine_count = _as_float(_count_union_matches(mol, SECONDARY_AMINE_PATTERNS, atom_positions=(0,)))
    tertiary_amine_count = _as_float(_count_union_matches(mol, TERTIARY_AMINE_PATTERNS, atom_positions=(0,)))
    quaternary_ammonium_count = _as_float(_count_union_matches(mol, QUATERNARY_AMMONIUM_PATTERNS, atom_positions=(0,)))
    aromatic_basic_heterocycle_count = _as_float(_count_aromatic_basic_heterocycles(mol))

    carboxylic_acid_count = _as_float(_count_union_matches(mol, CARBOXYLIC_ACID_PATTERNS, atom_positions=(0,)))
    sulfonic_acid_count = _as_float(_count_union_matches(mol, SULFONIC_ACID_PATTERNS, atom_positions=(0,)))
    strong_acidic_group_family_count = float(
        int(carboxylic_acid_count > 0.0)
        + int(sulfonic_acid_count > 0.0)
    )
    aryl_halide_count = _as_float(_count_aryl_halides(mol))
    steroid_like_ring_system_count = _as_float(_count_steroid_like_ring_systems(mol))

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
            "carbon_atom_count": carbon_atom_count,
            "heteroatom_count": heteroatom_count,
            "oxygen_nitrogen_count": oxygen_nitrogen_count,
            "halogen_atom_count": halogen_atom_count,
            "logp": logp,
            "molar_refractivity": molar_refractivity,
            "labute_asa": labute_asa,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbond,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "heterocycle_ring_count": heterocycle_ring_count,
            "aromatic_heterocycle_ring_count": aromatic_heterocycle_ring_count,
            "fused_ring_count": fused_ring_count,
            "largest_fused_ring_system_ring_count": largest_fused_ring_system_ring_count,
            "fraction_csp3": fraction_csp3,
            "aromatic_atom_fraction": aromatic_atom_fraction,
            "ring_to_rotatable_ratio": ring_to_rotatable_ratio,
            "planarity_proxy_score": planarity_proxy_score,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "basic_center_count": _as_float(len(basic_center_atom_indices)),
            "primary_amine_count": primary_amine_count,
            "secondary_amine_count": secondary_amine_count,
            "tertiary_amine_count": tertiary_amine_count,
            "quaternary_ammonium_count": quaternary_ammonium_count,
            "aromatic_basic_heterocycle_count": aromatic_basic_heterocycle_count,
            "carboxylic_acid_count": carboxylic_acid_count,
            "sulfonic_acid_count": sulfonic_acid_count,
            "strong_acidic_group_family_count": strong_acidic_group_family_count,
            "aryl_halide_count": aryl_halide_count,
            "steroid_like_ring_system_count": steroid_like_ring_system_count,
        }
    )

    features.update(_compute_pka_summary(mol, logp))

    heterocycle_present = heterocycle_ring_count > 0.0

    tertiary_amine_or_basic_center_present = (
        tertiary_amine_count > 0.0
        or aromatic_basic_heterocycle_count > 0.0
        or len(basic_center_atom_indices) > 0
    )
    if (
        features["pka_features_available"] == 1.0
        and not _is_missing(features["num_basic_sites"])
        and features["num_basic_sites"] > 0.0
    ):
        tertiary_amine_or_basic_center_present = True

    strong_acidic_signature = (
        strong_acidic_group_family_count > 0.0
        or negative_formal_atom_count > 0
    )
    if (
        not strong_acidic_signature
        and features["pka_features_available"] == 1.0
        and not _is_missing(features["num_acidic_sites"])
        and features["num_acidic_sites"] > 0.0
        and not _is_missing(features["most_acidic_pka"])
        and not _is_missing(features["acid_deprotonated_fraction_ph74"])
    ):
        strong_acidic_signature = (
            features["most_acidic_pka"] <= 6.5
            and features["acid_deprotonated_fraction_ph74"] >= 0.5
        )

    weak_base_pka_ok = True
    if (
        features["pka_features_available"] == 1.0
        and not _is_missing(features["num_basic_sites"])
        and features["num_basic_sites"] > 0.0
        and not _is_missing(features["most_basic_pka"])
    ):
        weak_base_pka_ok = features["most_basic_pka"] <= 10.5

    neutral_or_weak_base_profile = (
        not strong_acidic_signature
        and negative_formal_atom_count == 0
        and formal_charge >= 0.0
        and formal_charge <= 1.0
        and quaternary_ammonium_count == 0.0
        and weak_base_pka_ok
    )
    if (
        neutral_or_weak_base_profile
        and features["pka_features_available"] == 1.0
        and not _is_missing(features["net_charge_proxy_ph74"])
    ):
        neutral_or_weak_base_profile = features["net_charge_proxy_ph74"] >= -0.25

    lipophilic_pi_scaffold_present = (
        aromatic_ring_count >= 2.0
        and logp > 3.0
        and planarity_proxy_score >= -0.05
    )
    large_flexible_scaffold_present = (
        mol_weight > 350.0
        and rotatable_bonds > 6.0
    )
    high_polarity_penalty = (
        tpsa > 100.0
        or hbd > 2.0
        or hba > 8.0
        or total_hbond > 10.0
    )
    steroid_like_scaffold_present = steroid_like_ring_system_count > 0.0
    non_strong_acidic_profile = not strong_acidic_signature

    lipophilicity_ok = logp > 3.0
    if not _is_missing(features["estimated_logd_ph74"]):
        lipophilicity_ok = lipophilicity_ok or features["estimated_logd_ph74"] > 3.0

    hba_moderate_profile = 3.0 <= hba <= 8.0

    primary_rule_flags = [
        _maybe_gt(mol_weight, 400.0),
        _maybe_gt(logp, 3.0),
        _maybe_lt(tpsa, 100.0),
        _maybe_le(hbd, 2.0),
        _maybe_between(hba, 4.0, 6.0, inclusive=True),
        _maybe_gt(rotatable_bonds, 6.0),
        _maybe_ge(aromatic_ring_count, 2.0),
        _flag(heterocycle_present),
        _flag(tertiary_amine_or_basic_center_present),
        _flag(neutral_or_weak_base_profile),
        _flag(not steroid_like_scaffold_present),
    ]
    cyp3a4_primary_rule_pass_count = float(sum(flag == 1.0 for flag in primary_rule_flags))

    cyp3a4_substrate_heuristic_pass = (
        large_flexible_scaffold_present
        and lipophilicity_ok
        and tpsa < 100.0
        and hbd <= 2.0
        and hba_moderate_profile
        and aromatic_ring_count >= 2.0
        and lipophilic_pi_scaffold_present
        and (heterocycle_present or tertiary_amine_or_basic_center_present)
        and neutral_or_weak_base_profile
        and not steroid_like_scaffold_present
        and not high_polarity_penalty
    )

    features.update(
        {
            "halogen_substituent_present": _flag(halogen_atom_count > 0.0),
            "aryl_halide_present": _flag(aryl_halide_count > 0.0),
            "heterocycle_present": _flag(heterocycle_present),
            "tertiary_amine_or_basic_center_present": _flag(tertiary_amine_or_basic_center_present),
            "steroid_like_scaffold_present": _flag(steroid_like_scaffold_present),
            "mw_gt_400": _maybe_gt(mol_weight, 400.0),
            "logp_gt_3": _maybe_gt(logp, 3.0),
            "estimated_logd_gt_3": _maybe_gt(features["estimated_logd_ph74"], 3.0),
            "tpsa_lt_100": _maybe_lt(tpsa, 100.0),
            "hbd_le_2": _maybe_le(hbd, 2.0),
            "hba_in_4_6": _maybe_between(hba, 4.0, 6.0, inclusive=True),
            "rotatable_bonds_gt_6": _maybe_gt(rotatable_bonds, 6.0),
            "aromatic_ring_count_ge_2": _maybe_ge(aromatic_ring_count, 2.0),
            "aromatic_ring_count_ge_3": _maybe_ge(aromatic_ring_count, 3.0),
            "lipophilic_pi_scaffold_present": _flag(lipophilic_pi_scaffold_present),
            "large_flexible_scaffold_present": _flag(large_flexible_scaffold_present),
            "non_strong_acidic_profile": _flag(non_strong_acidic_profile),
            "neutral_or_weak_base_profile": _flag(neutral_or_weak_base_profile),
            "high_polarity_penalty": _flag(high_polarity_penalty),
            "cyp3a4_primary_rule_pass_count": cyp3a4_primary_rule_pass_count,
            "cyp3a4_substrate_heuristic_pass": _flag(cyp3a4_substrate_heuristic_pass),
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
        description="Generate CYP3A4 substrate DeepResearch features from SMILES."
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
