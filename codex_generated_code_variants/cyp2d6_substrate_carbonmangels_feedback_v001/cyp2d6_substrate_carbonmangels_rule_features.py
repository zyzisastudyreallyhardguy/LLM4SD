#!/usr/bin/env python3
"""Deterministic CYP2D6 substrate rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch CYP2D6 substrate ruleset. The implemented
features focus on:

- lipophilicity, polarity, size, aromaticity, and moderate flexibility
- amine/basic-center features plus MolGpKa-derived pKa, charge, and logD
  proxies at pH 7.4
- planar aromatic, basic-heterocycle, and cationic-center proxies associated
  with classical CYP2D6 substrates
- benzylic, allylic, alpha-heteroatom, and aromatic-alkoxy soft-spot proxies
  that approximate common CYP2D6 oxidation handles

Intentionally skipped:
- exact 5-7 angstrom pharmacophore geometry, docking poses, and explicit
  active-site interactions because the repository does not ship a validated
  CYP2D6 structural modeling workflow
- experimental turnover, kinetic constants, and confirmed metabolite data that
  are not derivable from the molecular graph alone
- highly specific 3D shape descriptors; instead this module exposes stable 2D
  flatness, ring, and sp3-complexity proxies
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
MISSING_PKA_SENTINEL = 0.0
MISSING_DEGREE_SENTINEL = 0.0
MISSING_DISTANCE_SENTINEL = 99.0

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
AROMATIC_ALKOXY_PATTERNS = (
    "[a][OX2][CX4]",
)
AROMATIC_METHOXY_PATTERNS = (
    "[a][OX2][CH3]",
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
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("labute_asa", "Labute approximate surface area as a stable size/volume proxy."),
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
        ("fused_aromatic_ring_count", "Count of aromatic rings participating in fused aromatic systems."),
        ("bridgehead_atom_count", "Bridgehead atom count."),
        ("spiro_atom_count", "Spiro atom count."),
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
        ("basic_ring_count", "Count of rings containing at least one heuristic basic-center atom."),
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
        ("benzylic_ch_site_count", "Count of non-aromatic carbons with at least one H and a directly adjacent aromatic atom."),
        ("allylic_ch_site_count", "Count of non-aromatic carbons with at least one H adjacent to an unsaturated carbon."),
        ("alpha_heteroatom_ch_site_count", "Count of carbons with at least one H and a directly adjacent N/O/P/S atom."),
        ("aromatic_alkoxy_count", "Count of aromatic alkoxy substituents using a local SMARTS proxy."),
        ("aromatic_methoxy_count", "Count of aromatic methoxy substituents."),
        ("oxidation_site_proxy_count", "Number of unique benzylic, allylic, or alpha-heteroatom carbon atoms."),
        ("oxidation_site_min_heavy_degree", "Minimum heavy-atom degree among oxidation-site proxy carbons as a low-hindrance proxy."),
        ("oxidation_site_mean_heavy_degree", "Mean heavy-atom degree among oxidation-site proxy carbons."),
        ("basic_to_oxidation_min_graph_distance", "Minimum graph distance between a basic-center atom and an oxidation-site proxy carbon."),
        ("basic_to_aromatic_min_graph_distance", "Minimum graph distance from a basic-center atom to a different aromatic atom."),
        ("carboxylic_acid_count", "Count of carboxylic acid or carboxylate motifs."),
        ("sulfonic_acid_count", "Count of sulfonic acid or sulfonate motifs."),
        ("acidic_group_family_count", "Number of strong acidic motif families present among carboxylates and sulfonates."),
        ("basic_center_present", "Rule flag: at least one heuristic or predicted basic center is present."),
        ("most_basic_pka_ge_8", "Rule flag: most basic predicted pKa >= 8."),
        ("most_basic_pka_ge_9", "Rule flag: most basic predicted pKa >= 9."),
        ("positive_charge_present_ph74", "Rule flag: positive charge is expected at pH 7.4 from formal charge or dominant basic-site protonation."),
        ("logp_ge_2", "Rule flag: logP >= 2."),
        ("logp_ge_3", "Rule flag: logP >= 3."),
        ("estimated_logd_ge_1p5", "Rule flag: estimated logD(7.4) >= 1.5."),
        ("tpsa_lt_50", "Rule flag: TPSA < 50."),
        ("hbd_le_5", "Rule flag: H-bond donor count <= 5."),
        ("hba_le_5", "Rule flag: H-bond acceptor count <= 5."),
        ("rotatable_bonds_le_10", "Rule flag: rotatable bond count <= 10."),
        ("aromatic_ring_count_ge_1", "Rule flag: aromatic ring count >= 1."),
        ("aromatic_ring_count_in_1_3", "Rule flag: aromatic ring count lies between 1 and 3."),
        ("aromatic_planar_scaffold_present", "Heuristic flag: at least one aromatic ring plus a non-sp3-biased flatness proxy."),
        ("basic_ring_present", "Rule flag: at least one ring contains a heuristic basic center."),
        ("oxidation_site_present", "Rule flag: at least one benzylic, allylic, or alpha-heteroatom soft-spot proxy is present."),
        ("aromatic_alkoxy_present", "Rule flag: at least one aromatic alkoxy motif is present."),
        ("mw_in_200_500", "Rule flag: molecular weight lies between 200 and 500."),
        ("low_3d_complexity_proxy", "Heuristic flag favoring relatively planar or elongated molecules: low sp3 content with little spiro or bridgehead complexity."),
        ("acidic_group_absent", "Rule flag: no strong acidic motif or dominant acidic-ionization signature is present."),
        ("high_polarity_or_acidic_penalty", "Heuristic penalty for high TPSA, excess heteroatom-driven polarity, or strong acidity."),
        ("cation_aromatic_pharmacophore_present", "Heuristic flag: a likely cationic center co-occurs with aromaticity and a conservative 2D distance proxy to aromatic or oxidation regions."),
        ("cyp2d6_primary_rule_pass_count", "Count of passed primary CYP2D6 heuristic checks across basicity, charge, lipophilicity, polarity, aromaticity, and soft-spot proxies."),
        ("cyp2d6_substrate_heuristic_pass", "Composite heuristic screen for lipophilic weak-base, low-PSA, aromatic, non-acidic CYP2D6 substrate-like molecules."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Exact CYP2D6 pharmacophore geometry in angstroms is reduced to conservative 2D graph-distance proxies because this repository does not provide a validated CYP2D6 docking workflow.",
    "True 3D shape, globularity, and conformer-specific aromatic coplanarity are approximated with ring, sp3, bridgehead, and spiro proxies instead of generated conformers.",
    "Experimental kinetic or turnover measurements such as Km, Vmax, or metabolite ratios are skipped because they require assay data rather than graph-derived features.",
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
        from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run CYP2D6 feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        GraphDescriptors=GraphDescriptors,
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


def _count_basic_center_rings(mol, basic_atom_indices: list[int]) -> int:
    basic_atom_set = set(basic_atom_indices)
    if not basic_atom_set:
        return 0
    return sum(bool(ring.intersection(basic_atom_set)) for ring in _ring_atom_sets(mol))


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


def _atom_is_allylic_candidate(atom, mol) -> bool:
    atom_idx = atom.GetIdx()
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == atom_idx or neighbor.GetAtomicNum() != 6:
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
        return MISSING_DEGREE_SENTINEL, MISSING_DEGREE_SENTINEL
    degrees = [_heavy_degree(mol, atom_idx) for atom_idx in sorted(oxidation_atom_indices)]
    return float(min(degrees)), float(sum(degrees) / len(degrees))


def _min_distance_between_sets(
    mol,
    source_indices: list[int] | set[int],
    target_indices: list[int] | set[int],
) -> float:
    if not source_indices or not target_indices:
        return MISSING_DISTANCE_SENTINEL
    distance_matrix = _rdkit().Chem.GetDistanceMatrix(mol)
    min_distance = math.inf
    for source_idx in source_indices:
        for target_idx in target_indices:
            if source_idx == target_idx:
                continue
            min_distance = min(min_distance, float(distance_matrix[source_idx][target_idx]))
    return MISSING_DISTANCE_SENTINEL if math.isinf(min_distance) else float(min_distance)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float(numerator)
    return float(numerator / denominator)


def _compute_pka_summary(mol, logp: float) -> dict[str, float]:
    result = {
        "most_basic_pka": MISSING_PKA_SENTINEL,
        "most_acidic_pka": MISSING_PKA_SENTINEL,
        "num_basic_sites": 0.0,
        "num_acidic_sites": 0.0,
        "neutral_fraction_ph74": 1.0,
        "charged_fraction_ph74": 0.0,
        "base_protonated_fraction_ph74": 0.0,
        "acid_deprotonated_fraction_ph74": 0.0,
        "net_charge_proxy_ph74": 0.0,
        "estimated_logd_ph74": logp,
        "pka_features_available": 0.0,
    }

    try:
        predictor = _get_pka_predictor()
        prediction = predictor.predict(mol)
    except Exception:
        return result

    base_sites = getattr(prediction, "base_sites_1", {}) or {}
    acid_sites = getattr(prediction, "acid_sites_1", {}) or {}

    most_basic_pka = max(base_sites.values()) if base_sites else MISSING_PKA_SENTINEL
    most_acidic_pka = min(acid_sites.values()) if acid_sites else MISSING_PKA_SENTINEL
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
            "most_basic_pka": _as_float(most_basic_pka),
            "most_acidic_pka": _as_float(most_acidic_pka),
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
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    molar_refractivity = _as_float(rdkit.Crippen.MolMR(mol))
    labute_asa = _as_float(rdkit.rdMolDescriptors.CalcLabuteASA(mol))
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
    fused_aromatic_ring_count = _as_float(_count_fused_aromatic_rings(mol))
    bridgehead_atom_count = _as_float(rdkit.rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
    spiro_atom_count = _as_float(rdkit.rdMolDescriptors.CalcNumSpiroAtoms(mol))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
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
    basic_ring_count = _as_float(_count_basic_center_rings(mol, basic_center_atom_indices))
    aromatic_basic_heterocycle_count = _as_float(_count_aromatic_basic_heterocycles(mol))

    aromatic_atom_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetIsAromatic() and atom.GetAtomicNum() > 1
    ]

    benzylic_atoms, allylic_atoms, alpha_hetero_atoms = _collect_oxidation_site_atom_sets(mol)
    oxidation_atom_indices = benzylic_atoms | allylic_atoms | alpha_hetero_atoms
    oxidation_site_min_heavy_degree, oxidation_site_mean_heavy_degree = _oxidation_site_degree_stats(
        mol,
        oxidation_atom_indices,
    )
    basic_to_oxidation_min_graph_distance = _min_distance_between_sets(
        mol,
        basic_center_atom_indices,
        oxidation_atom_indices,
    )
    basic_to_aromatic_min_graph_distance = _min_distance_between_sets(
        mol,
        basic_center_atom_indices,
        aromatic_atom_indices,
    )

    aromatic_alkoxy_count = _as_float(_count_union_matches(mol, AROMATIC_ALKOXY_PATTERNS, atom_positions=(1,)))
    aromatic_methoxy_count = _as_float(_count_union_matches(mol, AROMATIC_METHOXY_PATTERNS, atom_positions=(1,)))
    carboxylic_acid_count = _as_float(_count_union_matches(mol, CARBOXYLIC_ACID_PATTERNS, atom_positions=(0,)))
    sulfonic_acid_count = _as_float(_count_union_matches(mol, SULFONIC_ACID_PATTERNS, atom_positions=(0,)))

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
            "fused_aromatic_ring_count": fused_aromatic_ring_count,
            "bridgehead_atom_count": bridgehead_atom_count,
            "spiro_atom_count": spiro_atom_count,
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
            "basic_ring_count": basic_ring_count,
            "aromatic_basic_heterocycle_count": aromatic_basic_heterocycle_count,
            "benzylic_ch_site_count": _as_float(len(benzylic_atoms)),
            "allylic_ch_site_count": _as_float(len(allylic_atoms)),
            "alpha_heteroatom_ch_site_count": _as_float(len(alpha_hetero_atoms)),
            "aromatic_alkoxy_count": aromatic_alkoxy_count,
            "aromatic_methoxy_count": aromatic_methoxy_count,
            "oxidation_site_proxy_count": _as_float(len(oxidation_atom_indices)),
            "oxidation_site_min_heavy_degree": oxidation_site_min_heavy_degree,
            "oxidation_site_mean_heavy_degree": oxidation_site_mean_heavy_degree,
            "basic_to_oxidation_min_graph_distance": basic_to_oxidation_min_graph_distance,
            "basic_to_aromatic_min_graph_distance": basic_to_aromatic_min_graph_distance,
            "carboxylic_acid_count": carboxylic_acid_count,
            "sulfonic_acid_count": sulfonic_acid_count,
            "acidic_group_family_count": float(
                int(carboxylic_acid_count > 0.0)
                + int(sulfonic_acid_count > 0.0)
            ),
        }
    )

    features.update(_compute_pka_summary(mol, logp))

    basic_center_present = len(basic_center_atom_indices) > 0
    if features["pka_features_available"] == 1.0 and not _is_missing(features["num_basic_sites"]):
        basic_center_present = basic_center_present or features["num_basic_sites"] > 0.0

    if (
        features["pka_features_available"] == 1.0
        and not _is_missing(features["num_basic_sites"])
        and features["num_basic_sites"] == 0.0
    ):
        most_basic_pka_ge_8 = 0.0
        most_basic_pka_ge_9 = 0.0
    else:
        most_basic_pka_ge_8 = _maybe_ge(features["most_basic_pka"], 8.0)
        most_basic_pka_ge_9 = _maybe_ge(features["most_basic_pka"], 9.0)

    positive_charge_present_ph74 = positive_formal_atom_count > 0
    if features["pka_features_available"] == 1.0 and not _is_missing(features["base_protonated_fraction_ph74"]):
        positive_charge_present_ph74 = (
            positive_charge_present_ph74
            or features["base_protonated_fraction_ph74"] >= 0.5
        )
    elif tertiary_amine_count > 0.0 or secondary_amine_count > 0.0 or primary_amine_count > 0.0 or quaternary_ammonium_count > 0.0:
        positive_charge_present_ph74 = True

    acidic_signature = (
        features["acidic_group_family_count"] > 0.0
        or negative_formal_atom_count > 0
    )
    if (
        not acidic_signature
        and features["pka_features_available"] == 1.0
        and not _is_missing(features["num_acidic_sites"])
        and features["num_acidic_sites"] > 0.0
        and not _is_missing(features["acid_deprotonated_fraction_ph74"])
        and not _is_missing(features["most_acidic_pka"])
    ):
        acidic_signature = (
            features["acid_deprotonated_fraction_ph74"] >= 0.5
            and features["most_acidic_pka"] <= PHYSIOLOGICAL_PH
        )

    aromatic_planar_scaffold_present = (
        aromatic_ring_count >= 1.0
        and planarity_proxy_score >= 0.0
        and fused_aromatic_ring_count <= 2.0
    )
    low_3d_complexity_proxy = (
        fraction_csp3 <= 0.45
        and spiro_atom_count == 0.0
        and bridgehead_atom_count <= 1.0
    )

    high_polarity_or_acidic_penalty = (
        acidic_signature
        or tpsa > 70.0
        or hba > 6.0
        or oxygen_nitrogen_count >= 6.0
    )

    cationic_center_likely = (
        positive_charge_present_ph74
        or most_basic_pka_ge_8 == 1.0
        or quaternary_ammonium_count > 0.0
        or tertiary_amine_count > 0.0
        or secondary_amine_count > 0.0
        or primary_amine_count > 0.0
    )

    distance_proxy_ok = False
    if not _is_missing(basic_to_oxidation_min_graph_distance):
        distance_proxy_ok = 3.0 <= basic_to_oxidation_min_graph_distance <= 7.0
    if not distance_proxy_ok and not _is_missing(basic_to_aromatic_min_graph_distance):
        distance_proxy_ok = basic_to_aromatic_min_graph_distance <= 5.0

    cation_aromatic_pharmacophore_present = (
        basic_center_present
        and cationic_center_likely
        and aromatic_ring_count >= 1.0
        and distance_proxy_ok
    )

    oxidation_site_present = bool(oxidation_atom_indices)
    aromatic_alkoxy_present = aromatic_alkoxy_count > 0.0
    basic_ring_present = basic_ring_count > 0.0

    primary_rule_flags = [
        _flag(basic_center_present),
        most_basic_pka_ge_8,
        _flag(positive_charge_present_ph74),
        _maybe_ge(logp, 2.0),
        _maybe_ge(features["estimated_logd_ph74"], 1.5),
        _maybe_lt(tpsa, 50.0),
        _maybe_le(hbd, 5.0),
        _maybe_le(hba, 5.0),
        _maybe_le(rotatable_bonds, 10.0),
        _maybe_ge(aromatic_ring_count, 1.0),
        _flag(aromatic_planar_scaffold_present),
        _maybe_between(mol_weight, 200.0, 500.0, inclusive=True),
        _flag(not acidic_signature),
        _flag(oxidation_site_present),
    ]
    cyp2d6_primary_rule_pass_count = float(sum(flag == 1.0 for flag in primary_rule_flags))

    cyp2d6_substrate_heuristic_pass = (
        basic_center_present
        and cationic_center_likely
        and logp >= 2.0
        and tpsa < 60.0
        and hba <= 5.0
        and hbd <= 5.0
        and rotatable_bonds <= 10.0
        and aromatic_ring_count >= 1.0
        and aromatic_planar_scaffold_present
        and 200.0 <= mol_weight <= 500.0
        and not high_polarity_or_acidic_penalty
        and (oxidation_site_present or aromatic_alkoxy_present)
    )

    features.update(
        {
            "basic_center_present": _flag(basic_center_present),
            "most_basic_pka_ge_8": most_basic_pka_ge_8,
            "most_basic_pka_ge_9": most_basic_pka_ge_9,
            "positive_charge_present_ph74": _flag(positive_charge_present_ph74),
            "logp_ge_2": _maybe_ge(logp, 2.0),
            "logp_ge_3": _maybe_ge(logp, 3.0),
            "estimated_logd_ge_1p5": _maybe_ge(features["estimated_logd_ph74"], 1.5),
            "tpsa_lt_50": _maybe_lt(tpsa, 50.0),
            "hbd_le_5": _maybe_le(hbd, 5.0),
            "hba_le_5": _maybe_le(hba, 5.0),
            "rotatable_bonds_le_10": _maybe_le(rotatable_bonds, 10.0),
            "aromatic_ring_count_ge_1": _maybe_ge(aromatic_ring_count, 1.0),
            "aromatic_ring_count_in_1_3": _maybe_between(aromatic_ring_count, 1.0, 3.0, inclusive=True),
            "aromatic_planar_scaffold_present": _flag(aromatic_planar_scaffold_present),
            "basic_ring_present": _flag(basic_ring_present),
            "oxidation_site_present": _flag(oxidation_site_present),
            "aromatic_alkoxy_present": _flag(aromatic_alkoxy_present),
            "mw_in_200_500": _maybe_between(mol_weight, 200.0, 500.0, inclusive=True),
            "low_3d_complexity_proxy": _flag(low_3d_complexity_proxy),
            "acidic_group_absent": _flag(not acidic_signature),
            "high_polarity_or_acidic_penalty": _flag(high_polarity_or_acidic_penalty),
            "cation_aromatic_pharmacophore_present": _flag(cation_aromatic_pharmacophore_present),
            "cyp2d6_primary_rule_pass_count": cyp2d6_primary_rule_pass_count,
            "cyp2d6_substrate_heuristic_pass": _flag(cyp2d6_substrate_heuristic_pass),
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
        description="Generate CYP2D6 substrate DeepResearch features from SMILES."
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
