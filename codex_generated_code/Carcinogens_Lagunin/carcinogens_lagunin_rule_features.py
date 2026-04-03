#!/usr/bin/env python3
"""Deterministic Carcinogens_Lagunin rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch carcinogenicity ruleset. The implemented
features emphasize:

- structural alerts tied to electrophilicity, DNA reactivity, and redox cycling
- polyaromatic / intercalation proxies and halogenated persistent scaffolds
- core physicochemical descriptors plus MolGpKa-derived pKa/logD proxies

Intentionally skipped:
- experimental carcinogenicity measurements, exposure, and persistence assays
- exact 3D planarity, DNA docking, and kinetic reactivity measurements
- dedicated CYP-bioactivation or metabolism models beyond simple ring proxies
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
        ("max_fused_aromatic_system_size", "Largest fused aromatic ring-system size measured in aromatic rings."),
        ("aromatic_n_heterocycle_ring_count", "Count of aromatic rings containing at least one nitrogen atom."),
        ("aromatic_linker_count", "Count of non-ring bonds directly connecting two aromatic atoms."),
        ("halogen_atom_count", "Total halogen atom count."),
        ("fluorine_atom_count", "Fluorine atom count."),
        ("chlorine_atom_count", "Chlorine atom count."),
        ("bromine_atom_count", "Bromine atom count."),
        ("iodine_atom_count", "Iodine atom count."),
        ("aromatic_halogen_count", "Number of halogens directly attached to aromatic atoms."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
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
        ("estimated_logd_ph74", "Estimated logD at pH 7.4 from logP and neutral fraction."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
        ("pah_system_count", "Count of fused aromatic hydrocarbon systems with at least three aromatic rings."),
        ("pah_present", "1 if a polycyclic aromatic hydrocarbon alert is present."),
        ("planar_aromatic_proxy", "2D proxy for a flat multi-aromatic pi system with intercalation potential."),
        ("epoxide_ring_count", "Count of epoxide-like three-membered O-containing rings."),
        ("epoxide_present", "1 if an epoxide ring is present."),
        ("aziridine_ring_count", "Count of aziridine-like three-membered N-containing rings."),
        ("aziridine_present", "1 if an aziridine ring is present."),
        ("alkyl_halide_count", "Count of non-aromatic sp3 carbon-halogen alkylating motifs."),
        ("alkyl_sulfate_or_sulfonate_count", "Count of alkyl sulfate or sulfonate ester alkylator proxies."),
        ("alkylating_group_present", "1 if an alkyl halide or alkyl sulfate/sulfonate alert is present."),
        ("michael_acceptor_count", "Count of alpha,beta-unsaturated carbonyl alerts."),
        ("michael_acceptor_present", "1 if an alpha,beta-unsaturated carbonyl alert is present."),
        ("aromatic_amine_count", "Count of primary or secondary aromatic amine nitrogens."),
        ("aromatic_amine_present", "1 if a primary or secondary aromatic amine is present."),
        ("nitro_group_count", "Count of nitro groups."),
        ("nitroso_group_count", "Count of nitroso groups."),
        ("aromatic_nitro_or_nitroso_present", "1 if a nitro or nitroso group is attached to an aromatic atom."),
        ("azo_linkage_count", "Count of azo linkage alerts."),
        ("azo_linkage_present", "1 if an azo linkage alert is present."),
        ("quinone_ring_count", "Count of ortho- or para-quinone ring alerts."),
        ("quinone_present", "1 if a quinone-like ring alert is present."),
        ("aldehyde_count", "Count of aldehyde groups."),
        ("aldehyde_present", "1 if an aldehyde alert is present."),
        ("n_mustard_count", "Count of nitrogen mustard motif alerts."),
        ("s_mustard_count", "Count of sulfur mustard motif alerts."),
        ("mustard_present", "1 if an N- or S-mustard motif is present."),
        ("polyhalogenated_aromatic_ring_count", "Count of aromatic rings bearing two or more halogen substituents."),
        ("polyhalogenated_aromatic_present", "1 if a polyhalogenated aromatic ring alert is present."),
        ("catechol_ring_count", "Count of aromatic rings with adjacent phenolic substituents."),
        ("hydroquinone_ring_count", "Count of aromatic rings with para phenolic substituents."),
        ("redox_cycling_alert_count", "Count of catechol, hydroquinone, and quinone ring alerts."),
        ("redox_cycling_present", "1 if any redox-cycling alert is present."),
        ("metabolic_stability_proxy", "Heuristic persistent-scaffold flag based on ring burden, halogens, and lipophilicity."),
        ("cyp_bioactivation_proxy", "1 if an aromatic nitrogen heterocycle proxy for CYP bioactivation is present."),
        ("dna_binding_propensity_flag", "1 if a PAH or planar multi-aromatic DNA-binding proxy is present."),
        ("electrophilic_alert_family_count", "Count of distinct electrophilic or alkylating alert families."),
        ("genotoxic_alert_count", "Count of distinct carcinogenicity alert families matched."),
        ("genotoxic_alert_present", "1 if any major structural carcinogenicity alert is present."),
        ("high_logp_flag", "Heuristic flag: logP >= 5."),
        ("very_large_molecular_weight_flag", "Heuristic flag: molecular weight > 500."),
        ("low_tpsa_flag", "Heuristic flag: TPSA < 60."),
        ("low_hbond_balance_flag", "Heuristic flag: HBD + HBA <= 5."),
        ("high_aromatic_ring_burden_flag", "Heuristic flag: aromatic ring count >= 3."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Experimental carcinogenicity readouts, exposure burden, and clearance data are skipped because they are not recoverable from SMILES alone.",
    "Exact 3D planarity, DNA intercalation energetics, and bioactivation kinetics are reduced to 2D graph proxies rather than modeled explicitly.",
    "Dedicated CYP, ROS, or metabolic-stability predictors are not embedded here; only structural proxy alerts are exposed.",
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


def _maybe_lt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value < threshold)


def _maybe_le(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value <= threshold)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
        from rdkit.Chem.rdchem import HybridizationType
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run Carcinogens_Lagunin feature generation. "
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


def _count_aromatic_halogen_substituents(mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetIsAromatic() and end.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS:
            count += 1
        elif end.GetIsAromatic() and begin.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS:
            count += 1
    return count


def _count_aromatic_linkers(mol) -> int:
    count = 0
    for bond in mol.GetBonds():
        if bond.IsInRing():
            continue
        if bond.GetBeginAtom().GetIsAromatic() and bond.GetEndAtom().GetIsAromatic():
            count += 1
    return count


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


def _is_phenol_like_substituent(mol, ring_atom_idx: int, neighbor_idx: int) -> bool:
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    if neighbor.GetAtomicNum() != 8:
        return False
    bond = mol.GetBondBetweenAtoms(ring_atom_idx, neighbor_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return neighbor.GetDegree() == 1 and (neighbor.GetTotalNumHs() > 0 or neighbor.GetFormalCharge() == -1)


def _collect_aromatic_ring_substituent_alerts(mol) -> dict[str, int]:
    catechol_ring_count = 0
    hydroquinone_ring_count = 0
    polyhalogenated_aromatic_ring_count = 0

    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if not ring_atoms or not all(atom.GetIsAromatic() for atom in ring_atoms):
            continue

        phenol_positions: set[int] = set()
        halogen_positions: set[int] = set()

        for position, atom_idx in enumerate(ring):
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in ring:
                    continue
                if neighbor.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS:
                    halogen_positions.add(position)
                if _is_phenol_like_substituent(mol, atom_idx, neighbor_idx):
                    phenol_positions.add(position)

        if len(halogen_positions) >= 2:
            polyhalogenated_aromatic_ring_count += 1

        ring_size = len(ring)
        phenol_positions_sorted = sorted(phenol_positions)
        found_catechol = False
        found_hydroquinone = False
        for i, first in enumerate(phenol_positions_sorted):
            for second in phenol_positions_sorted[i + 1 :]:
                distance = _cyclic_distance(first, second, ring_size)
                if distance == 1:
                    found_catechol = True
                if ring_size == 6 and distance == 3:
                    found_hydroquinone = True

        if found_catechol:
            catechol_ring_count += 1
        if found_hydroquinone:
            hydroquinone_ring_count += 1

    return {
        "catechol_ring_count": catechol_ring_count,
        "hydroquinone_ring_count": hydroquinone_ring_count,
        "polyhalogenated_aromatic_ring_count": polyhalogenated_aromatic_ring_count,
    }


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
        if system_size >= 3 and all(mol.GetAtomWithIdx(atom_idx).GetAtomicNum() == 6 for atom_idx in component_atoms):
            pah_system_count += 1

    return {
        "fused_aromatic_ring_count": fused_aromatic_ring_count,
        "max_fused_aromatic_system_size": max_fused_aromatic_system_size,
        "pah_system_count": pah_system_count,
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


def _count_aziridines(mol) -> int:
    aziridine_count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 3:
            continue
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        atomic_nums = sorted(atom.GetAtomicNum() for atom in ring_atoms)
        if atomic_nums != [6, 6, 7]:
            continue
        if all(
            mol.GetBondBetweenAtoms(ring[i], ring[(i + 1) % 3]).GetBondTypeAsDouble() == 1.0
            for i in range(3)
        ):
            aziridine_count += 1
    return aziridine_count


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


@lru_cache(maxsize=1)
def _get_pka_predictor():
    if str(INTERN_S1_ROOT) not in sys.path:
        sys.path.insert(0, str(INTERN_S1_ROOT))

    from tools.pka_related_tools import _get_pka_predictor as _load_predictor

    return _load_predictor()


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
            "estimated_logd_ph74": estimated_logd,
            "pka_features_available": 1.0,
        }
    )
    return result


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
    aromatic_ring_count = _as_float(rdkit.LipinskiL.NumAromaticRings(mol))
    ring_count = _as_float(rdkit.LipinskiL.RingCount(mol))
    aliphatic_ring_count = _as_float(rdkit.LipinskiL.NumAliphaticRings(mol))
    fraction_csp3 = _as_float(rdkit.LipinskiL.FractionCSP3(mol))
    aromatic_linker_count = _count_aromatic_linkers(mol)
    halogen_atom_count = _count_halogen_atoms(mol)
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)

    aromatic_ring_summary = _aromatic_ring_system_summary(mol)
    ring_alerts = _collect_aromatic_ring_substituent_alerts(mol)

    epoxide_ring_count = _count_epoxides(mol)
    aziridine_ring_count = _count_aziridines(mol)
    alkyl_halide_count = _count_alkyl_halides(mol)
    alkyl_sulfate_or_sulfonate_count = _count_unique_matches(
        mol,
        _smarts("[CX4]-[OX2]-[SX4](=[OX1])(=[OX1])-[#6,#7,#8]"),
        (2,),
    )
    michael_acceptor_count = _count_unique_matches(mol, _smarts("[C,c]=[C,c]-[C](=O)[#6,#7,#8]"))
    aromatic_amine_count = _count_aromatic_amines(mol)
    nitro_group_count = _count_unique_matches(mol, _smarts("[NX3+](=O)[O-]"), (0,))
    nitroso_group_count = _count_unique_matches(mol, _smarts("[#6,#7,#8,#16,a]-[N;X2]=O"), (1,))
    aromatic_nitro_or_nitroso_present = (
        _count_unique_matches(mol, _smarts("[a]-[NX3+](=O)[O-]"), (1,)) > 0
        or _count_unique_matches(mol, _smarts("[a]-[N;X2]=O"), (1,)) > 0
    )
    azo_linkage_count = _count_unique_matches(mol, _smarts("[a,#6]-[N]=[N]-[a,#6]"), (1, 2))
    quinone_ring_count = _count_quinone_like_rings(mol)
    aldehyde_count = _count_unique_matches(mol, _smarts("[CX3H1](=O)[#6]"), (0,))
    n_mustard_count = _count_unique_matches(
        mol,
        _smarts("[NX3;!$([N]-[C,S,P]=[O,S,N])]([CH2][CH2][F,Cl,Br,I])[CH2][CH2][F,Cl,Br,I]"),
        (0,),
    )
    s_mustard_count = _count_unique_matches(
        mol,
        _smarts("[SX2]([CH2][CH2][F,Cl,Br,I])[CH2][CH2][F,Cl,Br,I]"),
        (0,),
    )

    pah_present = aromatic_ring_summary["pah_system_count"] > 0
    planar_aromatic_proxy = (
        pah_present
        or (
            aromatic_ring_count >= 2.0
            and fraction_csp3 <= 0.35
            and (
                aromatic_ring_summary["max_fused_aromatic_system_size"] >= 2
                or aromatic_linker_count >= 1
            )
        )
    )
    alkylating_group_present = (alkyl_halide_count + alkyl_sulfate_or_sulfonate_count) > 0
    quinone_present = quinone_ring_count > 0
    mustard_present = (n_mustard_count + s_mustard_count) > 0
    polyhalogenated_aromatic_present = ring_alerts["polyhalogenated_aromatic_ring_count"] > 0
    redox_cycling_alert_count = (
        ring_alerts["catechol_ring_count"] + ring_alerts["hydroquinone_ring_count"] + quinone_ring_count
    )
    redox_cycling_present = redox_cycling_alert_count > 0
    dna_binding_propensity_flag = pah_present or planar_aromatic_proxy
    aromatic_n_heterocycle_ring_count = _count_aromatic_n_heterocycle_rings(mol)
    cyp_bioactivation_proxy = aromatic_n_heterocycle_ring_count > 0

    electrophilic_alert_family_count = float(
        sum(
            [
                epoxide_ring_count > 0,
                aziridine_ring_count > 0,
                alkylating_group_present,
                michael_acceptor_count > 0,
                quinone_present,
                aldehyde_count > 0,
                mustard_present,
            ]
        )
    )
    genotoxic_alert_count = float(
        sum(
            [
                pah_present,
                planar_aromatic_proxy,
                epoxide_ring_count > 0,
                aziridine_ring_count > 0,
                alkylating_group_present,
                michael_acceptor_count > 0,
                aromatic_amine_count > 0,
                aromatic_nitro_or_nitroso_present,
                azo_linkage_count > 0,
                quinone_present,
                aldehyde_count > 0,
                mustard_present,
                polyhalogenated_aromatic_present,
                redox_cycling_present,
                dna_binding_propensity_flag,
            ]
        )
    )
    metabolic_stability_proxy = (
        polyhalogenated_aromatic_present
        or (ring_count >= 3.0 and halogen_atom_count >= 1)
        or (aromatic_ring_count >= 3.0 and logp >= 4.0)
    )

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": _as_float(rdkit.Descriptors.ExactMolWt(mol)),
            "heavy_atom_count": _as_float(mol.GetNumHeavyAtoms()),
            "carbon_atom_count": _as_float(_count_atomic_number(mol, 6)),
            "heteroatom_count": _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "oxygen_nitrogen_count": _as_float(_count_oxygen_and_nitrogen(mol)),
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
            "aromatic_n_heterocycle_ring_count": _as_float(aromatic_n_heterocycle_ring_count),
            "aromatic_linker_count": _as_float(aromatic_linker_count),
            "halogen_atom_count": _as_float(halogen_atom_count),
            "fluorine_atom_count": _as_float(_count_atomic_number(mol, 9)),
            "chlorine_atom_count": _as_float(_count_atomic_number(mol, 17)),
            "bromine_atom_count": _as_float(_count_atomic_number(mol, 35)),
            "iodine_atom_count": _as_float(_count_atomic_number(mol, 53)),
            "aromatic_halogen_count": _as_float(_count_aromatic_halogen_substituents(mol)),
            "fraction_csp3": fraction_csp3,
            "formal_charge": _as_float(rdkit.Chem.GetFormalCharge(mol)),
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
        }
    )

    features.update(_compute_pka_summary(mol))

    features.update(
        {
            "pah_system_count": _as_float(aromatic_ring_summary["pah_system_count"]),
            "pah_present": _flag(pah_present),
            "planar_aromatic_proxy": _flag(planar_aromatic_proxy),
            "epoxide_ring_count": _as_float(epoxide_ring_count),
            "epoxide_present": _flag(epoxide_ring_count > 0),
            "aziridine_ring_count": _as_float(aziridine_ring_count),
            "aziridine_present": _flag(aziridine_ring_count > 0),
            "alkyl_halide_count": _as_float(alkyl_halide_count),
            "alkyl_sulfate_or_sulfonate_count": _as_float(alkyl_sulfate_or_sulfonate_count),
            "alkylating_group_present": _flag(alkylating_group_present),
            "michael_acceptor_count": _as_float(michael_acceptor_count),
            "michael_acceptor_present": _flag(michael_acceptor_count > 0),
            "aromatic_amine_count": _as_float(aromatic_amine_count),
            "aromatic_amine_present": _flag(aromatic_amine_count > 0),
            "nitro_group_count": _as_float(nitro_group_count),
            "nitroso_group_count": _as_float(nitroso_group_count),
            "aromatic_nitro_or_nitroso_present": _flag(aromatic_nitro_or_nitroso_present),
            "azo_linkage_count": _as_float(azo_linkage_count),
            "azo_linkage_present": _flag(azo_linkage_count > 0),
            "quinone_ring_count": _as_float(quinone_ring_count),
            "quinone_present": _flag(quinone_present),
            "aldehyde_count": _as_float(aldehyde_count),
            "aldehyde_present": _flag(aldehyde_count > 0),
            "n_mustard_count": _as_float(n_mustard_count),
            "s_mustard_count": _as_float(s_mustard_count),
            "mustard_present": _flag(mustard_present),
            "polyhalogenated_aromatic_ring_count": _as_float(ring_alerts["polyhalogenated_aromatic_ring_count"]),
            "polyhalogenated_aromatic_present": _flag(polyhalogenated_aromatic_present),
            "catechol_ring_count": _as_float(ring_alerts["catechol_ring_count"]),
            "hydroquinone_ring_count": _as_float(ring_alerts["hydroquinone_ring_count"]),
            "redox_cycling_alert_count": _as_float(redox_cycling_alert_count),
            "redox_cycling_present": _flag(redox_cycling_present),
            "metabolic_stability_proxy": _flag(metabolic_stability_proxy),
            "cyp_bioactivation_proxy": _flag(cyp_bioactivation_proxy),
            "dna_binding_propensity_flag": _flag(dna_binding_propensity_flag),
            "electrophilic_alert_family_count": electrophilic_alert_family_count,
            "genotoxic_alert_count": genotoxic_alert_count,
            "genotoxic_alert_present": _flag(genotoxic_alert_count > 0),
            "high_logp_flag": _maybe_ge(logp, 5.0),
            "very_large_molecular_weight_flag": _maybe_gt(mol_weight, 500.0),
            "low_tpsa_flag": _maybe_lt(tpsa, 60.0),
            "low_hbond_balance_flag": _maybe_le(total_hbond, 5.0),
            "high_aromatic_ring_burden_flag": _maybe_ge(aromatic_ring_count, 3.0),
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
        description="Generate Carcinogens_Lagunin DeepResearch features from SMILES."
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
