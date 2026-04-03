#!/usr/bin/env python3
"""Deterministic CYP2C9 substrate rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch CYP2C9 substrate ruleset. The implemented
features focus on:

- core physicochemical descriptors around size, lipophilicity, polarity,
  aromaticity, and flexibility
- MolGpKa-derived acidity/basicity, ionization, and logD proxies at pH 7.4
- acidic motif and aromatic-scaffold features inspired by common CYP2C9
  substrates such as aryl acids, tetrazoles, and sulfonylureas
- 2D oxidation-site proxies for benzylic, allylic, and alpha-heteroatom C-H
  positions together with simple graph-distance and steric proxies

Intentionally skipped:
- exact 3D acid-to-site distances in angstroms, docking poses, and active-site
  complementarity because the repository does not ship a validated CYP2C9
  structural modeling workflow
- experimental or external-model endpoints such as aqueous solubility, plasma
  protein binding, and confirmed substrate labels for other CYP isoforms
- specialized inhibitor/confounder fragment libraries such as
  sulphinpyrazole-like motifs because there is no curated local reference set
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
HETERO_ADJACENT_ATOMIC_NUMBERS = {7, 8, 15, 16}

CARBOXYLIC_ACID_PATTERNS = (
    "[CX3](=O)[OX2H1]",
    "[CX3](=O)[O-]",
)
ACIDIC_SULFONAMIDE_PATTERNS = (
    "[SX4](=[OX1])(=[OX1])[NX3;H1]",
    "[NX3;H1][SX4](=[OX1])(=[OX1])[#6]",
)
SULFONYLUREA_PATTERNS = (
    "[NX3;H1][SX4](=[OX1])(=[OX1])[NX3][CX3](=[OX1])[#6,#7]",
    "[NX3][SX4](=[OX1])(=[OX1])[NX3][CX3](=[OX1])[#6,#7]",
)


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
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
        ("heterocycle_ring_count", "Count of rings containing at least one heteroatom."),
        ("aromatic_heterocycle_ring_count", "Count of aromatic rings containing at least one heteroatom."),
        ("fused_ring_count", "Count of rings participating in fused ring systems."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("aromatic_atom_fraction", "Fraction of heavy atoms that are aromatic."),
        ("ring_to_rotatable_ratio", "Ring count divided by max(rotatable bonds, 1) as a semi-rigidity proxy."),
        ("planarity_proxy_score", "Aromatic atom fraction minus fractionCSP3 as a simple flatness proxy."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
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
        ("tetrazole_ring_count", "Count of five-membered rings containing four nitrogens and one carbon."),
        ("acidic_sulfonamide_count", "Count of acidic sulfonamide motifs with an N-H sulfonamide nitrogen."),
        ("sulfonylurea_count", "Count of sulfonylurea motifs."),
        (
            "acidic_group_attached_to_aromatic_scaffold_count",
            "Count of acidic groups whose atoms lie on or within three graph bonds of an aromatic atom.",
        ),
        ("anionic_motif_family_count", "Number of acidic motif families present among carboxylate, tetrazole, sulfonamide, and sulfonylurea."),
        ("anionic_motif_present", "1 if any acidic motif family associated with CYP2C9 binding is present."),
        ("benzylic_ch_site_count", "Count of non-aromatic carbons with at least one H and a directly adjacent aromatic atom."),
        ("allylic_ch_site_count", "Count of non-aromatic carbons with at least one H adjacent to an unsaturated carbon."),
        ("alpha_heteroatom_ch_site_count", "Count of carbons with at least one H and a directly adjacent N/O/P/S atom."),
        ("oxidation_site_proxy_count", "Number of unique benzylic, allylic, or alpha-heteroatom carbon atoms."),
        ("oxidation_site_present", "1 if at least one oxidation-site proxy atom is present."),
        ("oxidation_site_min_heavy_degree", "Minimum heavy-atom degree among oxidation-site proxy carbons as a low-hindrance proxy."),
        ("oxidation_site_mean_heavy_degree", "Mean heavy-atom degree among oxidation-site proxy carbons."),
        ("acidic_to_oxidation_min_graph_distance", "Minimum graph distance between any acidic-group atom and oxidation-site proxy carbon."),
        ("acidic_to_oxidation_distance_4_8", "1 if the acidic-group to oxidation-site graph distance proxy falls between 4 and 8 bonds."),
        (
            "known_cyp2c9_motif_present",
            "1 if an acidic group lies on/near an aromatic scaffold or a sulfonylurea motif is present.",
        ),
        ("mw_gt_300", "Rule flag: molecular weight > 300."),
        ("logp_ge_2", "Rule flag: logP >= 2."),
        ("logp_in_2_5", "Rule flag: 2 <= logP <= 5."),
        ("estimated_logd_ge_2", "Rule flag: estimated logD(7.4) >= 2."),
        ("estimated_logd_in_2_5", "Rule flag: 2 <= estimated logD(7.4) <= 5."),
        ("tpsa_40_150", "Rule flag: 40 <= TPSA <= 150."),
        ("hba_ge_3", "Rule flag: HBA >= 3."),
        ("hba_ge_5", "Rule flag: HBA >= 5."),
        ("hbd_le_3", "Rule flag: HBD <= 3."),
        ("rotatable_bonds_le_7", "Rule flag: rotatable bonds <= 7."),
        ("rotatable_bonds_gt_10", "Rule flag: rotatable bonds > 10."),
        ("aromatic_ring_count_ge_1", "Rule flag: aromatic ring count >= 1."),
        ("aromatic_ring_count_ge_2", "Rule flag: aromatic ring count >= 2."),
        ("heterocycle_present", "Rule flag: at least one heterocycle ring is present."),
        ("most_acidic_pka_in_3_8p5", "Rule flag: most acidic predicted pKa lies between 3 and 8.5."),
        ("most_basic_pka_lt_8", "Rule flag: most basic predicted pKa < 8."),
        ("acid_deprotonated_fraction_ge_0p5", "Rule flag: dominant acidic site is at least half deprotonated at pH 7.4."),
        ("base_protonated_fraction_le_0p5", "Rule flag: dominant basic site is at most half protonated at pH 7.4."),
        ("predicted_neutral_or_anionic_ph74", "Rule flag: net charge proxy at pH 7.4 is neutral to non-positive."),
        ("semi_rigid_aromatic_scaffold", "Rule flag: aromatic ring count >= 1 and rotatable bonds <= 7."),
        (
            "weak_acidic_lipophilic_profile",
            "Composite heuristic: acidic signature plus aromaticity and at least moderate lipophilicity without strong basicity.",
        ),
        (
            "cyp2c9_primary_rule_pass_count",
            "Count of passed primary CYP2C9 heuristic checks among size, lipophilicity, polarity, acidity, aromaticity, and oxidation-site proxies.",
        ),
        (
            "cyp2c9_substrate_heuristic_pass",
            "Composite heuristic screen requiring weak-acidic/lipophilic profile, aromaticity, HBA/HBD window, limited flexibility, and an oxidation-site proxy.",
        ),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Exact 6-8 angstrom acidic-group to metabolic-site geometry is reduced to a conservative graph-distance proxy because this repository does not provide a validated 3D CYP2C9 docking workflow.",
    "Aqueous solubility and plasma protein binding are skipped because the source text does not provide a stable local predictor or assay data in this repository.",
    "Steric and polar environment around the exact oxidation site are represented only by simple 2D oxidation-site degree and acidity-distance proxies, not by explicit active-site modeling.",
    "Known CYP2C9 inhibitor or confounder motifs such as sulphinpyrazole-like fragments are skipped because there is no curated local fragment library to implement them reliably.",
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
            "RDKit is required to run CYP2C9 feature generation. "
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
            ring_j = atom_rings[j]
            if len(ring_i.intersection(ring_j)) >= 2:
                fused_ring_indices.add(i)
                fused_ring_indices.add(j)
    return len(fused_ring_indices)


def _aromatic_atom_fraction(mol) -> float:
    heavy_atoms = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return aromatic_atoms / heavy_atoms


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


def _collect_match_atom_sets(
    mol,
    patterns: Iterable[str],
) -> list[set[int]]:
    atom_sets: list[set[int]] = []
    seen: set[tuple[int, ...]] = set()
    for pattern in patterns:
        for match in mol.GetSubstructMatches(_smarts(pattern), uniquify=True):
            key = tuple(sorted(match))
            if key in seen:
                continue
            seen.add(key)
            atom_sets.append(set(match))
    return atom_sets


def _count_tetrazole_rings(mol) -> int:
    count = 0
    for ring in _ring_atom_sets(mol):
        if len(ring) != 5:
            continue
        atomic_nums = [mol.GetAtomWithIdx(atom_idx).GetAtomicNum() for atom_idx in ring]
        if atomic_nums.count(7) == 4 and atomic_nums.count(6) == 1:
            count += 1
    return count


def _collect_tetrazole_ring_sets(mol) -> list[set[int]]:
    ring_sets: list[set[int]] = []
    for ring in _ring_atom_sets(mol):
        if len(ring) != 5:
            continue
        atomic_nums = [mol.GetAtomWithIdx(atom_idx).GetAtomicNum() for atom_idx in ring]
        if atomic_nums.count(7) == 4 and atomic_nums.count(6) == 1:
            ring_sets.append(set(ring))
    return ring_sets


def _collect_acidic_group_atom_sets(mol) -> list[set[int]]:
    atom_sets: list[set[int]] = []
    atom_sets.extend(_collect_match_atom_sets(mol, CARBOXYLIC_ACID_PATTERNS))
    atom_sets.extend(_collect_match_atom_sets(mol, ACIDIC_SULFONAMIDE_PATTERNS))
    atom_sets.extend(_collect_match_atom_sets(mol, SULFONYLUREA_PATTERNS))
    atom_sets.extend(_collect_tetrazole_ring_sets(mol))
    return atom_sets


def _count_acidic_groups_near_aromatic_atoms(
    mol,
    acidic_group_atom_sets: list[set[int]],
    max_distance: int = 3,
) -> int:
    if not acidic_group_atom_sets:
        return 0

    aromatic_atom_indices = {
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic()
    }
    if not aromatic_atom_indices:
        return 0

    distance_matrix = _rdkit().Chem.GetDistanceMatrix(mol)
    count = 0
    for atom_set in acidic_group_atom_sets:
        min_distance = math.inf
        for acid_idx in atom_set:
            for aromatic_idx in aromatic_atom_indices:
                min_distance = min(min_distance, float(distance_matrix[acid_idx][aromatic_idx]))
        if min_distance <= max_distance:
            count += 1
    return count


def _atom_is_allylic_candidate(atom, mol) -> bool:
    atom_idx = atom.GetIdx()
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == atom_idx:
            continue
        if neighbor.GetAtomicNum() != 6:
            continue
        for bond in neighbor.GetBonds():
            other = bond.GetOtherAtom(neighbor)
            if other.GetIdx() == atom_idx:
                continue
            if bond.GetBondTypeAsDouble() == 2.0:
                return True
    return False


def _collect_oxidation_site_atom_sets(mol) -> tuple[set[int], set[int], set[int]]:
    benzylic: set[int] = set()
    allylic: set[int] = set()
    alpha_hetero: set[int] = set()

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6 or atom.GetIsAromatic() or atom.GetTotalNumHs() == 0:
            continue

        atom_idx = atom.GetIdx()
        neighbors = list(atom.GetNeighbors())

        if any(neighbor.GetIsAromatic() for neighbor in neighbors):
            benzylic.add(atom_idx)

        if _atom_is_allylic_candidate(atom, mol):
            allylic.add(atom_idx)

        if any(neighbor.GetAtomicNum() in HETERO_ADJACENT_ATOMIC_NUMBERS for neighbor in neighbors):
            alpha_hetero.add(atom_idx)

    return benzylic, allylic, alpha_hetero


def _heavy_degree(mol, atom_idx: int) -> int:
    atom = mol.GetAtomWithIdx(atom_idx)
    return sum(neighbor.GetAtomicNum() > 1 for neighbor in atom.GetNeighbors())


def _oxidation_site_degree_stats(mol, oxidation_atom_indices: set[int]) -> tuple[float, float]:
    if not oxidation_atom_indices:
        return math.nan, math.nan
    degrees = [_heavy_degree(mol, atom_idx) for atom_idx in sorted(oxidation_atom_indices)]
    return float(min(degrees)), float(sum(degrees) / len(degrees))


def _min_distance_between_atom_sets(
    mol,
    source_sets: list[set[int]],
    target_indices: set[int],
) -> float:
    if not source_sets or not target_indices:
        return math.nan

    distance_matrix = _rdkit().Chem.GetDistanceMatrix(mol)
    min_distance = math.inf
    for source_set in source_sets:
        for source_idx in source_set:
            for target_idx in target_indices:
                min_distance = min(min_distance, float(distance_matrix[source_idx][target_idx]))
    return math.nan if math.isinf(min_distance) else float(min_distance)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float(numerator)
    return float(numerator / denominator)


def _compute_pka_summary(mol, logp: float) -> dict[str, float]:
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
            "most_basic_pka": _as_float(most_basic_pka) if base_sites else math.nan,
            "most_acidic_pka": _as_float(most_acidic_pka) if acid_sites else math.nan,
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
    heterocycle_ring_count = _as_float(_count_heterocycle_rings(mol))
    aromatic_heterocycle_ring_count = _as_float(_count_heterocycle_rings(mol, aromatic_only=True))
    fused_ring_count = _as_float(_count_fused_rings(mol))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    aromatic_atom_fraction = _as_float(_aromatic_atom_fraction(mol))
    ring_to_rotatable_ratio = _safe_ratio(ring_count, rotatable_bonds if rotatable_bonds > 0 else 1.0)
    planarity_proxy_score = aromatic_atom_fraction - fraction_csp3
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
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
            "heterocycle_ring_count": heterocycle_ring_count,
            "aromatic_heterocycle_ring_count": aromatic_heterocycle_ring_count,
            "fused_ring_count": fused_ring_count,
            "fraction_csp3": fraction_csp3,
            "aromatic_atom_fraction": aromatic_atom_fraction,
            "ring_to_rotatable_ratio": ring_to_rotatable_ratio,
            "planarity_proxy_score": planarity_proxy_score,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
        }
    )

    features.update(_compute_pka_summary(mol, logp))

    carboxylic_acid_count = _as_float(_count_union_matches(mol, CARBOXYLIC_ACID_PATTERNS, atom_positions=(0,)))
    tetrazole_ring_count = _as_float(_count_tetrazole_rings(mol))
    acidic_sulfonamide_count = _as_float(_count_union_matches(mol, ACIDIC_SULFONAMIDE_PATTERNS))
    sulfonylurea_count = _as_float(_count_union_matches(mol, SULFONYLUREA_PATTERNS))

    acidic_group_atom_sets = _collect_acidic_group_atom_sets(mol)
    acidic_group_attached_to_aromatic_scaffold_count = _as_float(
        _count_acidic_groups_near_aromatic_atoms(mol, acidic_group_atom_sets, max_distance=3)
    )

    benzylic_atoms, allylic_atoms, alpha_hetero_atoms = _collect_oxidation_site_atom_sets(mol)
    oxidation_atom_indices = benzylic_atoms | allylic_atoms | alpha_hetero_atoms
    oxidation_site_min_heavy_degree, oxidation_site_mean_heavy_degree = _oxidation_site_degree_stats(
        mol,
        oxidation_atom_indices,
    )
    acidic_to_oxidation_min_graph_distance = _min_distance_between_atom_sets(
        mol,
        acidic_group_atom_sets,
        oxidation_atom_indices,
    )

    anionic_motif_family_count = float(
        int(carboxylic_acid_count > 0.0)
        + int(tetrazole_ring_count > 0.0)
        + int(acidic_sulfonamide_count > 0.0)
        + int(sulfonylurea_count > 0.0)
    )
    anionic_motif_present = _flag(anionic_motif_family_count > 0.0)
    oxidation_site_present = _flag(bool(oxidation_atom_indices))
    known_cyp2c9_motif_present = _flag(
        acidic_group_attached_to_aromatic_scaffold_count > 0.0 or sulfonylurea_count > 0.0
    )

    if (
        features["pka_features_available"] == 1.0
        and not _is_missing(features["num_basic_sites"])
        and features["num_basic_sites"] == 0.0
    ):
        most_basic_pka_lt_8 = 1.0
    else:
        most_basic_pka_lt_8 = _maybe_lt(features["most_basic_pka"], 8.0)
    most_acidic_pka_in_3_8p5 = _maybe_between(features["most_acidic_pka"], 3.0, 8.5, inclusive=True)
    acid_deprotonated_fraction_ge_0p5 = _maybe_ge(features["acid_deprotonated_fraction_ph74"], 0.5)
    base_protonated_fraction_le_0p5 = _maybe_le(features["base_protonated_fraction_ph74"], 0.5)
    predicted_neutral_or_anionic_ph74 = _maybe_le(features["net_charge_proxy_ph74"], 0.05)

    weak_acidic_lipophilic_profile = None
    basicity_ok = None
    if (
        features["pka_features_available"] == 1.0
        and not _is_missing(features["num_basic_sites"])
        and features["num_basic_sites"] == 0.0
    ):
        basicity_ok = True
    elif not _is_missing(features["most_basic_pka"]):
        basicity_ok = features["most_basic_pka"] < 8.0

    acidity_signature = None
    if anionic_motif_family_count > 0.0:
        acidity_signature = True
    elif not any(
        _is_missing(value)
        for value in (
            features["most_acidic_pka"],
            features["acid_deprotonated_fraction_ph74"],
        )
    ):
        acidity_signature = (
            3.0 <= features["most_acidic_pka"] <= 8.5
            and features["acid_deprotonated_fraction_ph74"] >= 0.5
        )

    lipophilicity_ok = None
    if not _is_missing(logp):
        lipophilicity_ok = logp >= 2.0
    if not _is_missing(features["estimated_logd_ph74"]):
        lipophilicity_ok = bool(lipophilicity_ok) or features["estimated_logd_ph74"] >= 2.0

    if basicity_ok is not None and acidity_signature is not None and lipophilicity_ok is not None:
        weak_acidic_lipophilic_profile = (
            aromatic_ring_count >= 1.0
            and lipophilicity_ok
            and acidity_signature
            and basicity_ok
        )

    primary_rule_flags = [
        _maybe_gt(mol_weight, 300.0),
        _maybe_ge(logp, 2.0),
        _maybe_between(tpsa, 40.0, 150.0, inclusive=True),
        _maybe_ge(hba, 3.0),
        _maybe_le(hbd, 3.0),
        _maybe_le(rotatable_bonds, 7.0),
        _maybe_ge(aromatic_ring_count, 1.0),
        most_basic_pka_lt_8,
        predicted_neutral_or_anionic_ph74,
        anionic_motif_present,
        oxidation_site_present,
    ]
    cyp2c9_primary_rule_pass_count = float(sum(flag == 1.0 for flag in primary_rule_flags))

    cyp2c9_substrate_heuristic_pass = None
    if weak_acidic_lipophilic_profile is not None:
        cyp2c9_substrate_heuristic_pass = (
            weak_acidic_lipophilic_profile
            and hba >= 3.0
            and hbd <= 3.0
            and rotatable_bonds <= 7.0
            and bool(oxidation_atom_indices)
        )

    features.update(
        {
            "carboxylic_acid_count": carboxylic_acid_count,
            "tetrazole_ring_count": tetrazole_ring_count,
            "acidic_sulfonamide_count": acidic_sulfonamide_count,
            "sulfonylurea_count": sulfonylurea_count,
            "acidic_group_attached_to_aromatic_scaffold_count": acidic_group_attached_to_aromatic_scaffold_count,
            "anionic_motif_family_count": anionic_motif_family_count,
            "anionic_motif_present": anionic_motif_present,
            "benzylic_ch_site_count": _as_float(len(benzylic_atoms)),
            "allylic_ch_site_count": _as_float(len(allylic_atoms)),
            "alpha_heteroatom_ch_site_count": _as_float(len(alpha_hetero_atoms)),
            "oxidation_site_proxy_count": _as_float(len(oxidation_atom_indices)),
            "oxidation_site_present": oxidation_site_present,
            "oxidation_site_min_heavy_degree": oxidation_site_min_heavy_degree,
            "oxidation_site_mean_heavy_degree": oxidation_site_mean_heavy_degree,
            "acidic_to_oxidation_min_graph_distance": acidic_to_oxidation_min_graph_distance,
            "acidic_to_oxidation_distance_4_8": _maybe_between(
                acidic_to_oxidation_min_graph_distance,
                4.0,
                8.0,
                inclusive=True,
            ),
            "known_cyp2c9_motif_present": known_cyp2c9_motif_present,
            "mw_gt_300": _maybe_gt(mol_weight, 300.0),
            "logp_ge_2": _maybe_ge(logp, 2.0),
            "logp_in_2_5": _maybe_between(logp, 2.0, 5.0, inclusive=True),
            "estimated_logd_ge_2": _maybe_ge(features["estimated_logd_ph74"], 2.0),
            "estimated_logd_in_2_5": _maybe_between(features["estimated_logd_ph74"], 2.0, 5.0, inclusive=True),
            "tpsa_40_150": _maybe_between(tpsa, 40.0, 150.0, inclusive=True),
            "hba_ge_3": _maybe_ge(hba, 3.0),
            "hba_ge_5": _maybe_ge(hba, 5.0),
            "hbd_le_3": _maybe_le(hbd, 3.0),
            "rotatable_bonds_le_7": _maybe_le(rotatable_bonds, 7.0),
            "rotatable_bonds_gt_10": _maybe_gt(rotatable_bonds, 10.0),
            "aromatic_ring_count_ge_1": _maybe_ge(aromatic_ring_count, 1.0),
            "aromatic_ring_count_ge_2": _maybe_ge(aromatic_ring_count, 2.0),
            "heterocycle_present": _flag(heterocycle_ring_count > 0.0),
            "most_acidic_pka_in_3_8p5": most_acidic_pka_in_3_8p5,
            "most_basic_pka_lt_8": most_basic_pka_lt_8,
            "acid_deprotonated_fraction_ge_0p5": acid_deprotonated_fraction_ge_0p5,
            "base_protonated_fraction_le_0p5": base_protonated_fraction_le_0p5,
            "predicted_neutral_or_anionic_ph74": predicted_neutral_or_anionic_ph74,
            "semi_rigid_aromatic_scaffold": _flag(aromatic_ring_count >= 1.0 and rotatable_bonds <= 7.0),
            "weak_acidic_lipophilic_profile": _flag(weak_acidic_lipophilic_profile),
            "cyp2c9_primary_rule_pass_count": cyp2c9_primary_rule_pass_count,
            "cyp2c9_substrate_heuristic_pass": _flag(cyp2c9_substrate_heuristic_pass),
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
        description="Generate CYP2C9 substrate DeepResearch features from SMILES."
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
