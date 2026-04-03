#!/usr/bin/env python3
"""Deterministic Skin Reaction rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch Skin_Reaction ruleset. The implementation
emphasizes:

- direct electrophilic alerts that can covalently modify skin proteins
- bioactivation and prehapten proxy alerts that can increase sensitization risk
- physicochemical descriptors related to skin uptake and exposure

Intentionally skipped:
- experimental volatility, boiling point, and melting point measurements
- full pKa/speciation modeling of salt vs neutral parent forms
- definitive metabolic activation, phototoxicity, or GHS classification labels
  that require external data or a validated predictive model
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
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("heteroatom_count", "Total heteroatom count."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("aliphatic_ring_count", "Aliphatic ring count."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("charged_atom_count", "Number of atoms carrying non-zero formal charge."),
        ("chiral_center_count", "Count of assigned or potential tetrahedral stereocenters."),
        ("halogen_atom_count", "Total halogen atom count."),
        ("aromatic_halogen_count", "Number of halogens directly attached to aromatic atoms."),
        ("estimated_log_kp", "Potts-Guy style estimated skin permeability logKp."),
        ("estimated_kp_cm_per_h", "Skin permeability estimate in cm/h derived from logKp."),
        ("michael_acceptor_count", "Count of alpha,beta-unsaturated carbonyl alerts."),
        ("michael_acceptor_present", "1 if an alpha,beta-unsaturated carbonyl is present."),
        ("epoxide_ring_count", "Count of epoxide-like three-membered O-containing rings."),
        ("epoxide_present", "1 if an epoxide ring is present."),
        ("aziridine_ring_count", "Count of aziridine-like three-membered N-containing rings."),
        ("aziridine_present", "1 if an aziridine ring is present."),
        ("aldehyde_count", "Count of aldehyde alerts."),
        ("aldehyde_present", "1 if an aldehyde alert is present."),
        ("schiff_base_carbonyl_count", "Aldehydes plus unsaturated carbonyls that can form Schiff-base or related adducts."),
        ("schiff_base_carbonyl_present", "1 if an aldehyde or unsaturated carbonyl alert is present."),
        ("activated_aromatic_halide_count", "Count of halogenated aromatic positions with same-ring electron-withdrawing activation."),
        ("activated_aromatic_halide_present", "1 if an activated aromatic halide alert is present."),
        ("alkyl_halide_count", "Count of sp3 carbon-halogen SN2 alkylation alerts."),
        ("alkyl_halide_present", "1 if an alkyl halide alert is present."),
        ("alkyl_sulfonate_count", "Count of alkyl sulfonate ester leaving-group alerts."),
        ("alkyl_sulfonate_present", "1 if an alkyl sulfonate alert is present."),
        ("acyl_halide_count", "Count of acyl halide alerts."),
        ("sulfonyl_halide_count", "Count of sulfonyl halide alerts."),
        ("anhydride_count", "Count of acid anhydride alerts."),
        ("acyl_imidazole_count", "Count of N-acyl imidazole-like acyl transfer alerts."),
        ("acylating_alert_count", "Count of acylating or sulfuryl-transfer alert sites."),
        ("acylating_alert_present", "1 if any acylating alert is present."),
        ("aromatic_amine_count", "Count of aniline-like aromatic amine substituents."),
        ("aromatic_amine_present", "1 if an aromatic amine alert is present."),
        ("nitro_group_count", "Count of nitro groups."),
        ("nitroaromatic_present", "1 if a nitro group is attached to an aromatic ring."),
        ("quinone_like_ring_count", "Count of conjugated cyclic dione or imine-dione quinone-like ring systems."),
        ("quinone_like_present", "1 if a quinone-like ring system is present."),
        ("thiol_count", "Count of free thiol groups."),
        ("disulfide_count", "Count of disulfide motifs."),
        ("thiol_disulfide_alert_count", "Combined count of thiol and disulfide alerts."),
        ("thiol_disulfide_present", "1 if a thiol or disulfide alert is present."),
        ("hydrazine_count", "Count of hydrazine alerts."),
        ("hydrazide_count", "Count of hydrazide alerts."),
        ("hydrazine_hydrazide_present", "1 if a hydrazine or hydrazide alert is present."),
        ("phenol_count", "Count of phenol or phenoxide substituents on aromatic rings."),
        ("catechol_ring_count", "Count of aromatic rings with adjacent phenolic OH substituents."),
        ("hydroquinone_ring_count", "Count of aromatic rings with para phenolic OH substituents."),
        ("photoreactive_aromatic_carbonyl_count", "Count of aromatic ketone chromophores such as benzophenone-like motifs."),
        ("photoreactive_chromophore_present", "1 if a photoreactive aromatic carbonyl chromophore is present."),
        ("fused_aromatic_ring_count", "Number of aromatic rings participating in fused aromatic systems."),
        ("pah_like_present", "1 if a fused polyaromatic system is present."),
        ("prehapten_alert_count", "Count of simple prehapten proxy alert families such as phenols, quinone-forming phenols, photoreactive chromophores, and PAHs."),
        ("prehapten_alert_present", "1 if any prehapten proxy alert family is present."),
        ("leaving_group_count", "Combined count of aromatic/alkyl halides and sulfonate/acyl leaving-group alerts."),
        ("leaving_group_present", "1 if any strong leaving-group alert is present."),
        ("direct_electrophile_site_count", "Combined count of direct electrophilic or acylating alert sites."),
        ("bioactivation_alert_family_count", "Number of distinct bioactivation-oriented alert families present."),
        ("overall_alert_family_count", "Number of distinct skin-sensitization alert families present."),
        ("multiple_reactive_sites_flag", "1 if two or more direct electrophilic sites are present."),
        ("any_known_alert_present", "1 if any structural skin-sensitization alert family is present."),
        ("neutral_formal_charge_flag", "1 if the formal charge is zero."),
        ("charged_species_flag", "1 if any atom carries a non-zero formal charge."),
        ("quaternary_ammonium_count", "Count of quaternary ammonium atoms."),
        ("quaternary_ammonium_present", "1 if a quaternary ammonium atom is present."),
        ("mw_le_500", "Rule flag: molecular weight <= 500."),
        ("logp_in_1_4", "Rule flag: 1 <= logP <= 4."),
        ("logp_le_0", "Rule flag: logP <= 0."),
        ("logp_ge_5", "Rule flag: logP >= 5."),
        ("high_tpsa_flag", "Rule flag: TPSA > 140."),
        ("high_hbd_flag", "Rule flag: HBD > 5."),
        ("high_hba_flag", "Rule flag: HBA > 10."),
        ("high_polarity_low_uptake_flag", "Rule flag: TPSA > 140 or HBD > 5 or HBA > 10."),
        ("estimated_log_kp_ge_minus3", "Rule flag: estimated logKp >= -3."),
        ("estimated_log_kp_ge_minus2p5", "Rule flag: estimated logKp >= -2.5."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Boiling point, melting point, and measured volatility are skipped because they require dedicated property predictors or experimental data.",
    "Detailed salt-versus-neutral-parent prioritization is only approximated by graph formal charge and quaternary ammonium features; no full pKa/speciation model is used.",
    "Metabolic activation of terpenes, phenols, and other prohaptens is represented only by structural proxy alerts, not a validated metabolism simulator.",
    "Stereochemical effects are reduced to a chiral-center count; no 3D binding or enantioselective reactivity model is used.",
    "GHS category assignment is not hard-coded; this module only exposes structural and physicochemical proxy features for downstream models.",
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


def _maybe_le(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value <= threshold)


def _maybe_gt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value > threshold)


def _maybe_between(value: float | None, lower: float, upper: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(lower <= value <= upper)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run Skin_Reaction feature generation. "
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


def _count_epoxide_like_rings(mol, hetero_atomic_num: int) -> int:
    ring_count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 3:
            continue

        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        atomic_nums = sorted(atom.GetAtomicNum() for atom in ring_atoms)
        if atomic_nums != [6, 6, hetero_atomic_num]:
            continue

        all_single = True
        for index, atom_idx in enumerate(ring):
            next_idx = ring[(index + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                all_single = False
                break
        if all_single:
            ring_count += 1
    return ring_count


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


def _is_carbonyl_like_substituent(atom, ring_atom_idx: int) -> bool:
    if atom.GetAtomicNum() not in (6, 7):
        return False
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == ring_atom_idx:
            continue
        bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0 and neighbor.GetAtomicNum() in (7, 8, 16):
            return True
    return False


def _is_sulfuryl_like_substituent(atom, ring_atom_idx: int) -> bool:
    if atom.GetAtomicNum() != 16:
        return False
    oxygen_double_bonds = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == ring_atom_idx:
            continue
        bond = atom.GetOwningMol().GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
        if bond is not None and bond.GetBondTypeAsDouble() == 2.0 and neighbor.GetAtomicNum() == 8:
            oxygen_double_bonds += 1
    return oxygen_double_bonds >= 2


def _is_trifluoromethyl_substituent(atom, ring_atom_idx: int) -> bool:
    if atom.GetAtomicNum() != 6:
        return False
    fluorine_neighbors = 0
    heavy_neighbors = 0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIdx() == ring_atom_idx:
            heavy_neighbors += 1
            continue
        if neighbor.GetAtomicNum() == 9:
            fluorine_neighbors += 1
        elif neighbor.GetAtomicNum() > 1:
            heavy_neighbors += 1
    return fluorine_neighbors >= 3 and heavy_neighbors == 1


def _is_ewg_substituent(mol, ring_atom_idx: int, neighbor_idx: int) -> bool:
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    bond = mol.GetBondBetweenAtoms(ring_atom_idx, neighbor_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False

    if _is_sulfuryl_like_substituent(neighbor, ring_atom_idx):
        return True
    if _is_trifluoromethyl_substituent(neighbor, ring_atom_idx):
        return True
    if _is_carbonyl_like_substituent(neighbor, ring_atom_idx):
        return True

    if neighbor.GetAtomicNum() == 7:
        oxygen_double_bonds = 0
        for other in neighbor.GetNeighbors():
            if other.GetIdx() == ring_atom_idx:
                continue
            other_bond = mol.GetBondBetweenAtoms(neighbor_idx, other.GetIdx())
            if other_bond is not None and other_bond.GetBondTypeAsDouble() == 2.0 and other.GetAtomicNum() == 8:
                oxygen_double_bonds += 1
        if oxygen_double_bonds >= 1:
            return True

    if neighbor.GetAtomicNum() == 6:
        for other in neighbor.GetNeighbors():
            if other.GetIdx() == ring_atom_idx:
                continue
            other_bond = mol.GetBondBetweenAtoms(neighbor_idx, other.GetIdx())
            if other_bond is not None and other_bond.GetBondTypeAsDouble() == 3.0 and other.GetAtomicNum() == 7:
                return True

    return False


def _is_aromatic_amine_substituent(mol, ring_atom_idx: int, neighbor_idx: int) -> bool:
    atom = mol.GetAtomWithIdx(neighbor_idx)
    if atom.GetAtomicNum() != 7:
        return False
    bond = mol.GetBondBetweenAtoms(ring_atom_idx, neighbor_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    if atom.GetFormalCharge() > 0 or atom.GetDegree() > 3:
        return False

    for other in atom.GetNeighbors():
        if other.GetIdx() == ring_atom_idx:
            continue
        other_bond = mol.GetBondBetweenAtoms(neighbor_idx, other.GetIdx())
        if other_bond is not None and other_bond.GetBondTypeAsDouble() == 2.0 and other.GetAtomicNum() in (6, 7, 8, 16):
            return False
        if _is_sulfuryl_like_substituent(other, neighbor_idx):
            return False
    return True


def _collect_aromatic_ring_alerts(mol) -> dict[str, int]:
    phenol_atom_indices: set[int] = set()
    aromatic_amine_indices: set[int] = set()
    aromatic_halogen_indices: set[int] = set()
    activated_aromatic_halide_indices: set[int] = set()
    catechol_ring_count = 0
    hydroquinone_ring_count = 0

    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if not ring_atoms or not all(atom.GetIsAromatic() for atom in ring_atoms):
            continue

        halogen_positions: set[int] = set()
        halogen_neighbor_indices: set[int] = set()
        ewg_positions: set[int] = set()
        phenol_positions: set[int] = set()

        for position, atom_idx in enumerate(ring):
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in ring:
                    continue
                if neighbor.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS:
                    aromatic_halogen_indices.add(neighbor_idx)
                    halogen_positions.add(position)
                    halogen_neighbor_indices.add(neighbor_idx)
                if _is_ewg_substituent(mol, atom_idx, neighbor_idx):
                    ewg_positions.add(position)
                if _is_phenol_like_substituent(mol, atom_idx, neighbor_idx):
                    phenol_positions.add(position)
                    phenol_atom_indices.add(neighbor_idx)
                if _is_aromatic_amine_substituent(mol, atom_idx, neighbor_idx):
                    aromatic_amine_indices.add(neighbor_idx)

        if halogen_positions and ewg_positions:
            activated_aromatic_halide_indices.update(halogen_neighbor_indices)

        ring_size = len(ring)
        sorted_positions = sorted(phenol_positions)
        found_catechol = False
        found_hydroquinone = False
        for index, first in enumerate(sorted_positions):
            for second in sorted_positions[index + 1 :]:
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
        "phenol_count": len(phenol_atom_indices),
        "aromatic_amine_count": len(aromatic_amine_indices),
        "aromatic_halogen_count": len(aromatic_halogen_indices),
        "activated_aromatic_halide_count": len(activated_aromatic_halide_indices),
        "catechol_ring_count": catechol_ring_count,
        "hydroquinone_ring_count": hydroquinone_ring_count,
    }


def _count_fused_aromatic_rings(mol) -> int:
    aromatic_rings = [
        set(ring)
        for ring in mol.GetRingInfo().AtomRings()
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring)
    ]
    if not aromatic_rings:
        return 0

    fused_indices: set[int] = set()
    for i, ring_i in enumerate(aromatic_rings):
        for j in range(i + 1, len(aromatic_rings)):
            if len(ring_i.intersection(aromatic_rings[j])) >= 2:
                fused_indices.add(i)
                fused_indices.add(j)
    return len(fused_indices)


def _count_quinone_like_rings(mol) -> int:
    count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) not in (5, 6):
            continue

        exocyclic_double_hetero_positions: set[int] = set()
        conjugated_bond_count = 0

        for index, atom_idx in enumerate(ring):
            atom = mol.GetAtomWithIdx(atom_idx)
            next_idx = ring[(index + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)
            if bond is not None and (bond.GetIsAromatic() or bond.GetBondTypeAsDouble() == 2.0):
                conjugated_bond_count += 1

            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in ring:
                    continue
                other_bond = mol.GetBondBetweenAtoms(atom_idx, neighbor_idx)
                if (
                    other_bond is not None
                    and other_bond.GetBondTypeAsDouble() == 2.0
                    and neighbor.GetAtomicNum() in (7, 8)
                ):
                    exocyclic_double_hetero_positions.add(atom_idx)

        if len(exocyclic_double_hetero_positions) >= 2 and conjugated_bond_count >= len(ring) - 2:
            count += 1
    return count


def _estimate_log_kp(logp: float, mol_weight: float) -> float:
    return -2.72 + 0.71 * logp - 0.0061 * mol_weight


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    charged_atom_count = positive_formal_atom_count + negative_formal_atom_count
    aromatic_alerts = _collect_aromatic_ring_alerts(mol)

    michael_acceptor_count = _count_unique_matches(mol, _smarts("[C,c]=[C,c]-[C](=O)[#6,#7,#8]"), (2,))
    epoxide_ring_count = _count_epoxide_like_rings(mol, hetero_atomic_num=8)
    aziridine_ring_count = _count_epoxide_like_rings(mol, hetero_atomic_num=7)
    aldehyde_count = _count_unique_matches(mol, _smarts("[CX3H1](=O)[#6,#1]"), (0,))
    alkyl_halide_count = _count_unique_matches(mol, _smarts("[CX4]-[F,Cl,Br,I]"), (0,))
    alkyl_sulfonate_count = _count_unique_matches(mol, _smarts("[CX4]-[OX2]-[SX4](=[OX1])(=[OX1])[#6,#7,#8]"), (0,))
    acyl_halide_count = _count_unique_matches(mol, _smarts("[CX3](=[OX1])[F,Cl,Br,I]"), (0,))
    sulfonyl_halide_count = _count_unique_matches(mol, _smarts("[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]"), (0,))
    anhydride_count = _count_unique_matches(mol, _smarts("[CX3](=[OX1])[OX2][CX3](=[OX1])"), (0, 3))
    acyl_imidazole_count = _count_unique_matches_any(
        mol,
        [
            "[CX3](=[OX1])[n]1ccnc1",
            "[CX3](=[OX1])[n]1cncc1",
        ],
        (0,),
    )
    nitro_group_count = _count_unique_matches(mol, _smarts("[NX3+](=O)[O-]"), (0,))
    nitroaromatic_present = _count_unique_matches(mol, _smarts("[a]-[NX3+](=O)[O-]"), (1,)) > 0
    quinone_like_ring_count = _count_quinone_like_rings(mol)
    thiol_count = _count_unique_matches(mol, _smarts("[SX2H]"), (0,))
    disulfide_count = _count_unique_matches(mol, _smarts("[#16X2]-[#16X2]"), (0, 1))
    hydrazine_count = _count_unique_matches(mol, _smarts("[NX3;H1,H2,H3]-[NX3;H1,H2,H3]"), (0, 1))
    hydrazide_count = _count_unique_matches(mol, _smarts("[CX3](=[OX1])[NX3][NX3]"), (1, 2))
    photoreactive_aromatic_carbonyl_count = _count_unique_matches(mol, _smarts("[a]-[CX3](=[OX1])-[a]"), (1,))
    fused_aromatic_ring_count = _count_fused_aromatic_rings(mol)
    quaternary_ammonium_count = _count_unique_matches(mol, _smarts("[NX4+]"), (0,))
    estimated_log_kp = _estimate_log_kp(logp, mol_weight)
    estimated_kp_cm_per_h = 10.0 ** estimated_log_kp

    schiff_base_carbonyl_count = aldehyde_count + michael_acceptor_count
    acylating_alert_count = acyl_halide_count + sulfonyl_halide_count + anhydride_count + acyl_imidazole_count
    thiol_disulfide_alert_count = thiol_count + disulfide_count
    leaving_group_count = (
        aromatic_alerts["activated_aromatic_halide_count"]
        + alkyl_halide_count
        + alkyl_sulfonate_count
        + acyl_halide_count
        + sulfonyl_halide_count
    )
    direct_electrophile_site_count = (
        michael_acceptor_count
        + epoxide_ring_count
        + aziridine_ring_count
        + schiff_base_carbonyl_count
        + aromatic_alerts["activated_aromatic_halide_count"]
        + alkyl_halide_count
        + alkyl_sulfonate_count
        + acylating_alert_count
        + quinone_like_ring_count
    )

    prehapten_alert_flags = {
        "phenol": int(aromatic_alerts["phenol_count"] > 0),
        "quinone_forming_phenol": int(
            aromatic_alerts["catechol_ring_count"] > 0 or aromatic_alerts["hydroquinone_ring_count"] > 0
        ),
        "photoreactive_chromophore": int(photoreactive_aromatic_carbonyl_count > 0),
        "pah_like": int(fused_aromatic_ring_count >= 2),
    }
    prehapten_alert_count = float(sum(prehapten_alert_flags.values()))

    bioactivation_alert_family_flags = {
        "aromatic_amine": int(aromatic_alerts["aromatic_amine_count"] > 0),
        "nitroaromatic": int(nitroaromatic_present),
        "hydrazine_hydrazide": int((hydrazine_count + hydrazide_count) > 0),
        "prehapten_proxy": int(prehapten_alert_count > 0),
        "photoreactive_chromophore": int(photoreactive_aromatic_carbonyl_count > 0),
    }
    bioactivation_alert_family_count = float(sum(bioactivation_alert_family_flags.values()))

    overall_alert_family_flags = {
        "michael_acceptor": int(michael_acceptor_count > 0),
        "epoxide": int(epoxide_ring_count > 0),
        "aziridine": int(aziridine_ring_count > 0),
        "schiff_base_carbonyl": int(schiff_base_carbonyl_count > 0),
        "activated_aromatic_halide": int(aromatic_alerts["activated_aromatic_halide_count"] > 0),
        "alkyl_halide": int(alkyl_halide_count > 0),
        "alkyl_sulfonate": int(alkyl_sulfonate_count > 0),
        "acylating_alert": int(acylating_alert_count > 0),
        "aromatic_amine": int(aromatic_alerts["aromatic_amine_count"] > 0),
        "nitroaromatic": int(nitroaromatic_present),
        "quinone_like": int(quinone_like_ring_count > 0),
        "thiol_disulfide": int(thiol_disulfide_alert_count > 0),
        "hydrazine_hydrazide": int((hydrazine_count + hydrazide_count) > 0),
        "photoreactive_chromophore": int(photoreactive_aromatic_carbonyl_count > 0),
        "pah_like": int(fused_aromatic_ring_count >= 2),
        "prehapten_proxy": int(prehapten_alert_count > 0),
    }
    overall_alert_family_count = float(sum(overall_alert_family_flags.values()))

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": _as_float(rdkit.Descriptors.ExactMolWt(mol)),
            "heavy_atom_count": _as_float(mol.GetNumHeavyAtoms()),
            "logp": logp,
            "molar_refractivity": _as_float(rdkit.Crippen.MolMR(mol)),
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "heteroatom_count": _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "rotatable_bonds": _as_float(rdkit.Lipinski.NumRotatableBonds(mol)),
            "ring_count": _as_float(rdkit.Lipinski.RingCount(mol)),
            "aromatic_ring_count": _as_float(rdkit.Lipinski.NumAromaticRings(mol)),
            "aliphatic_ring_count": _as_float(rdkit.Lipinski.NumAliphaticRings(mol)),
            "fraction_csp3": _as_float(rdkit.Lipinski.FractionCSP3(mol)),
            "formal_charge": _as_float(rdkit.Chem.GetFormalCharge(mol)),
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "charged_atom_count": _as_float(charged_atom_count),
            "chiral_center_count": _as_float(
                len(rdkit.Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))
            ),
            "halogen_atom_count": _as_float(_count_halogen_atoms(mol)),
            "aromatic_halogen_count": _as_float(aromatic_alerts["aromatic_halogen_count"]),
            "estimated_log_kp": estimated_log_kp,
            "estimated_kp_cm_per_h": estimated_kp_cm_per_h,
            "michael_acceptor_count": _as_float(michael_acceptor_count),
            "michael_acceptor_present": _flag(michael_acceptor_count > 0),
            "epoxide_ring_count": _as_float(epoxide_ring_count),
            "epoxide_present": _flag(epoxide_ring_count > 0),
            "aziridine_ring_count": _as_float(aziridine_ring_count),
            "aziridine_present": _flag(aziridine_ring_count > 0),
            "aldehyde_count": _as_float(aldehyde_count),
            "aldehyde_present": _flag(aldehyde_count > 0),
            "schiff_base_carbonyl_count": _as_float(schiff_base_carbonyl_count),
            "schiff_base_carbonyl_present": _flag(schiff_base_carbonyl_count > 0),
            "activated_aromatic_halide_count": _as_float(aromatic_alerts["activated_aromatic_halide_count"]),
            "activated_aromatic_halide_present": _flag(aromatic_alerts["activated_aromatic_halide_count"] > 0),
            "alkyl_halide_count": _as_float(alkyl_halide_count),
            "alkyl_halide_present": _flag(alkyl_halide_count > 0),
            "alkyl_sulfonate_count": _as_float(alkyl_sulfonate_count),
            "alkyl_sulfonate_present": _flag(alkyl_sulfonate_count > 0),
            "acyl_halide_count": _as_float(acyl_halide_count),
            "sulfonyl_halide_count": _as_float(sulfonyl_halide_count),
            "anhydride_count": _as_float(anhydride_count),
            "acyl_imidazole_count": _as_float(acyl_imidazole_count),
            "acylating_alert_count": _as_float(acylating_alert_count),
            "acylating_alert_present": _flag(acylating_alert_count > 0),
            "aromatic_amine_count": _as_float(aromatic_alerts["aromatic_amine_count"]),
            "aromatic_amine_present": _flag(aromatic_alerts["aromatic_amine_count"] > 0),
            "nitro_group_count": _as_float(nitro_group_count),
            "nitroaromatic_present": _flag(nitroaromatic_present),
            "quinone_like_ring_count": _as_float(quinone_like_ring_count),
            "quinone_like_present": _flag(quinone_like_ring_count > 0),
            "thiol_count": _as_float(thiol_count),
            "disulfide_count": _as_float(disulfide_count),
            "thiol_disulfide_alert_count": _as_float(thiol_disulfide_alert_count),
            "thiol_disulfide_present": _flag(thiol_disulfide_alert_count > 0),
            "hydrazine_count": _as_float(hydrazine_count),
            "hydrazide_count": _as_float(hydrazide_count),
            "hydrazine_hydrazide_present": _flag((hydrazine_count + hydrazide_count) > 0),
            "phenol_count": _as_float(aromatic_alerts["phenol_count"]),
            "catechol_ring_count": _as_float(aromatic_alerts["catechol_ring_count"]),
            "hydroquinone_ring_count": _as_float(aromatic_alerts["hydroquinone_ring_count"]),
            "photoreactive_aromatic_carbonyl_count": _as_float(photoreactive_aromatic_carbonyl_count),
            "photoreactive_chromophore_present": _flag(photoreactive_aromatic_carbonyl_count > 0),
            "fused_aromatic_ring_count": _as_float(fused_aromatic_ring_count),
            "pah_like_present": _flag(fused_aromatic_ring_count >= 2),
            "prehapten_alert_count": prehapten_alert_count,
            "prehapten_alert_present": _flag(prehapten_alert_count > 0),
            "leaving_group_count": _as_float(leaving_group_count),
            "leaving_group_present": _flag(leaving_group_count > 0),
            "direct_electrophile_site_count": _as_float(direct_electrophile_site_count),
            "bioactivation_alert_family_count": bioactivation_alert_family_count,
            "overall_alert_family_count": overall_alert_family_count,
            "multiple_reactive_sites_flag": _flag(direct_electrophile_site_count >= 2),
            "any_known_alert_present": _flag(overall_alert_family_count > 0),
            "neutral_formal_charge_flag": _flag(rdkit.Chem.GetFormalCharge(mol) == 0),
            "charged_species_flag": _flag(charged_atom_count > 0),
            "quaternary_ammonium_count": _as_float(quaternary_ammonium_count),
            "quaternary_ammonium_present": _flag(quaternary_ammonium_count > 0),
            "mw_le_500": _maybe_le(mol_weight, 500.0),
            "logp_in_1_4": _maybe_between(logp, 1.0, 4.0),
            "logp_le_0": _maybe_le(logp, 0.0),
            "logp_ge_5": _maybe_ge(logp, 5.0),
            "high_tpsa_flag": _maybe_gt(tpsa, 140.0),
            "high_hbd_flag": _maybe_gt(hbd, 5.0),
            "high_hba_flag": _maybe_gt(hba, 10.0),
            "high_polarity_low_uptake_flag": _flag(tpsa > 140.0 or hbd > 5.0 or hba > 10.0),
            "estimated_log_kp_ge_minus3": _maybe_ge(estimated_log_kp, -3.0),
            "estimated_log_kp_ge_minus2p5": _maybe_ge(estimated_log_kp, -2.5),
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
        description="Generate Skin_Reaction DeepResearch features from SMILES."
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
