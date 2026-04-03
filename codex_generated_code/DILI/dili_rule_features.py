#!/usr/bin/env python3
"""Deterministic DILI rule-to-feature code for downstream ML.

This module turns SMILES strings into numeric features derived from the
computable parts of the DeepResearch DILI ruleset. The implementation focuses
on two categories:

- RDKit descriptors that capture size, lipophilicity, polarity, and flexibility.
- Structural alerts that can be detected locally from the molecular graph.

Intentionally skipped:
- Clinical or assay-dependent factors such as daily dose, plasma protein
  binding, solubility measurements, transporter inhibition, metabolic fraction,
  CYP substrate status, half-life, and HLA genotype.
- Mechanistic liabilities that need dedicated predictive models or experimental
  data, such as confirmed reactive metabolite formation, glutathione depletion,
  mitochondrial toxicity, and direct oxidative stress readouts.
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
        ("radical_electrons", "Total radical electron count."),
        ("halogen_atom_count", "Total halogen atom count."),
        ("fluorine_atom_count", "Fluorine atom count."),
        ("chlorine_atom_count", "Chlorine atom count."),
        ("bromine_atom_count", "Bromine atom count."),
        ("iodine_atom_count", "Iodine atom count."),
        ("aromatic_halogen_count", "Number of halogens directly attached to aromatic atoms."),
        ("nitro_group_count", "Count of nitro groups."),
        ("nitroaromatic_present", "1 if a nitro group is attached to an aromatic atom."),
        ("furan_ring_count", "Count of aromatic five-membered rings containing one oxygen."),
        ("furan_present", "1 if a furan ring is present."),
        ("thiophene_ring_count", "Count of aromatic five-membered rings containing one sulfur."),
        ("thiophene_present", "1 if a thiophene ring is present."),
        ("aromatic_primary_amine_count", "Count of primary aniline-like amines attached to aromatic rings."),
        ("aromatic_primary_amine_present", "1 if an aromatic primary amine is present."),
        ("hydrazine_alert_count", "Count of hydrazine-like N-N alerts."),
        ("hydrazide_alert_count", "Count of hydrazide alerts."),
        ("hydrazine_or_hydrazide_present", "1 if hydrazine or hydrazide chemistry is present."),
        ("michael_acceptor_count", "Count of alpha,beta-unsaturated carbonyl alerts."),
        ("michael_acceptor_present", "1 if an alpha,beta-unsaturated carbonyl is present."),
        ("epoxide_ring_count", "Count of epoxide-like three-membered O-containing rings."),
        ("epoxide_present", "1 if an epoxide ring is present."),
        ("acyl_halide_count", "Count of acyl halide alerts."),
        ("sulfonyl_halide_count", "Count of sulfonyl halide alerts."),
        ("anhydride_count", "Count of acid anhydride alerts."),
        ("isocyanate_count", "Count of isocyanate alerts."),
        ("isothiocyanate_count", "Count of isothiocyanate alerts."),
        ("strong_electrophile_count", "Count of strong electrophilic or acylating alerts."),
        ("strong_electrophile_present", "1 if any strong electrophilic or acylating alert is present."),
        ("phenol_count", "Count of phenol or phenoxide substituents on aromatic rings."),
        ("catechol_ring_count", "Count of aromatic rings with adjacent phenolic OH substituents."),
        ("hydroquinone_ring_count", "Count of aromatic rings with para phenolic OH substituents."),
        ("aminophenol_ring_count", "Count of aromatic rings with ortho/para phenol plus primary arylamine alerts."),
        ("quinone_forming_phenol_alert_count", "Count of quinone-forming phenol motif alerts."),
        ("quinone_forming_phenol_present", "1 if a quinone-forming phenol alert is present."),
        ("halogenated_aromatic_ring_present", "1 if any aromatic ring bears a halogen substituent."),
        ("polyhalogenated_aromatic_ring_present", "1 if any aromatic ring bears two or more halogen substituents."),
        ("pains_filter_available", "1 if RDKit PAINS filters were available during featurization."),
        ("pains_alert_count", "Count of matched RDKit PAINS alerts."),
        ("pains_alert_present", "1 if any RDKit PAINS alert was matched."),
        ("structural_alert_count", "Number of distinct DILI structural alert families matched."),
        ("reactive_metabolite_alert_count", "Number of distinct reactive-metabolite-oriented alert families matched."),
        ("high_lipophilicity_flag", "1 if logP >= 3."),
        ("very_high_lipophilicity_flag", "1 if logP >= 5."),
        ("large_molecular_weight_flag", "1 if molecular weight > 500."),
        ("multiple_rotatable_bonds_flag", "1 if rotatable bonds >= 8."),
        ("lipophilic_structural_alert_flag", "1 if logP >= 3 and at least one reactive alert family is present."),
        ("large_lipophilic_molecule_flag", "1 if molecular weight > 500 and logP >= 3."),
        ("polyhalogenated_lipophilic_alert_flag", "1 if an aromatic ring is polyhalogenated and logP >= 3."),
        ("reactive_alert_burden_ge_2", "1 if two or more reactive alert families are present."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Daily dose, Rule-of-Two dose term, and long half-life or accumulation risk require exposure metadata beyond SMILES.",
    "Plasma protein binding, aqueous solubility, and extensive hepatic metabolism are skipped as direct endpoints because they require assay data or a separate validated predictor.",
    "CYP2D6/CYP3A4 substrate status, BSEP inhibition, and MRP4 inhibition are skipped because they need dedicated ADME models or experimental annotations.",
    "Confirmed reactive metabolite formation, glutathione depletion, mitochondrial toxicity, and oxidative stress are not hard-coded; this module only exposes structural proxy alerts.",
    "Immune and patient-specific factors such as hapten-confirmed immunogenicity and HLA risk alleles are skipped because they cannot be recovered reliably from SMILES alone.",
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


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run DILI feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        FilterCatalog=FilterCatalog,
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
def _pains_catalog():
    rdkit = _rdkit()
    try:
        params = rdkit.FilterCatalog.FilterCatalogParams()
        params.AddCatalog(rdkit.FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        params.AddCatalog(rdkit.FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        params.AddCatalog(rdkit.FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        return rdkit.FilterCatalog.FilterCatalog(params)
    except Exception:
        return None


def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    rdkit = _rdkit()
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _count_atomic_number(mol, atomic_num: int) -> int:
    return sum(atom.GetAtomicNum() == atomic_num for atom in mol.GetAtoms())


def _count_halogen_atoms(mol) -> int:
    return sum(atom.GetAtomicNum() in HALOGEN_ATOMIC_NUMBERS for atom in mol.GetAtoms())


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


def _count_ring_types(mol) -> tuple[int, int]:
    furan_count = 0
    thiophene_count = 0

    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if len(ring_atoms) != 5 or not all(atom.GetIsAromatic() for atom in ring_atoms):
            continue
        oxygen_count = sum(atom.GetAtomicNum() == 8 for atom in ring_atoms)
        sulfur_count = sum(atom.GetAtomicNum() == 16 for atom in ring_atoms)
        carbon_count = sum(atom.GetAtomicNum() == 6 for atom in ring_atoms)
        if oxygen_count == 1 and sulfur_count == 0 and carbon_count == 4:
            furan_count += 1
        if sulfur_count == 1 and oxygen_count == 0 and carbon_count == 4:
            thiophene_count += 1

    return furan_count, thiophene_count


def _count_epoxides(mol) -> int:
    epoxide_count = 0
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) != 3:
            continue
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        atomic_nums = [atom.GetAtomicNum() for atom in ring_atoms]
        if sorted(atomic_nums) != [6, 6, 8]:
            continue

        ring_bonds_are_single = True
        for i, atom_idx in enumerate(ring):
            next_idx = ring[(i + 1) % len(ring)]
            bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                ring_bonds_are_single = False
                break
        if ring_bonds_are_single:
            epoxide_count += 1
    return epoxide_count


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


def _is_primary_arylamine_substituent(mol, ring_atom_idx: int, neighbor_idx: int) -> bool:
    neighbor = mol.GetAtomWithIdx(neighbor_idx)
    if neighbor.GetAtomicNum() != 7:
        return False
    bond = mol.GetBondBetweenAtoms(ring_atom_idx, neighbor_idx)
    if bond is None or bond.GetBondTypeAsDouble() != 1.0:
        return False
    return neighbor.GetDegree() == 1 and neighbor.GetTotalNumHs() >= 1


def _collect_ring_substituent_alerts(mol) -> dict[str, int]:
    phenol_atom_indices: set[int] = set()
    primary_amine_atom_indices: set[int] = set()
    halogenated_ring_present = False
    polyhalogenated_ring_present = False
    catechol_ring_count = 0
    hydroquinone_ring_count = 0
    aminophenol_ring_count = 0

    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if not ring_atoms or not all(atom.GetIsAromatic() for atom in ring_atoms):
            continue

        phenol_positions: set[int] = set()
        primary_amine_positions: set[int] = set()
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
                    phenol_atom_indices.add(neighbor_idx)
                if _is_primary_arylamine_substituent(mol, atom_idx, neighbor_idx):
                    primary_amine_positions.add(position)
                    primary_amine_atom_indices.add(neighbor_idx)

        if halogen_positions:
            halogenated_ring_present = True
        if len(halogen_positions) >= 2:
            polyhalogenated_ring_present = True

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
            if found_catechol and (found_hydroquinone or ring_size != 6):
                break

        if found_catechol:
            catechol_ring_count += 1
        if found_hydroquinone:
            hydroquinone_ring_count += 1

        if ring_size == 6 and phenol_positions and primary_amine_positions:
            found_aminophenol = False
            for phenol_position in phenol_positions:
                for amine_position in primary_amine_positions:
                    if _cyclic_distance(phenol_position, amine_position, ring_size) in (1, 3):
                        aminophenol_ring_count += 1
                        found_aminophenol = True
                        break
                if found_aminophenol:
                    break

    return {
        "phenol_count": len(phenol_atom_indices),
        "aromatic_primary_amine_count": len(primary_amine_atom_indices),
        "catechol_ring_count": catechol_ring_count,
        "hydroquinone_ring_count": hydroquinone_ring_count,
        "aminophenol_ring_count": aminophenol_ring_count,
        "halogenated_aromatic_ring_present": int(halogenated_ring_present),
        "polyhalogenated_aromatic_ring_present": int(polyhalogenated_ring_present),
    }


def _count_pains_alerts(mol) -> tuple[float, float]:
    catalog = _pains_catalog()
    if catalog is None:
        return 0.0, math.nan
    matches = catalog.GetMatches(mol)
    return 1.0, float(len(matches))


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    rotatable_bonds = _as_float(rdkit.LipinskiL.NumRotatableBonds(mol))

    nitro_group_count = _count_unique_matches(mol, _smarts("[NX3+](=O)[O-]"), (0,))
    nitroaromatic_present = _count_unique_matches(mol, _smarts("[a]-[NX3+](=O)[O-]"), (1,)) > 0
    furan_ring_count, thiophene_ring_count = _count_ring_types(mol)
    hydrazine_alert_count = _count_unique_matches(mol, _smarts("[NX3;H1,H2,H3]-[NX3;H1,H2,H3]"))
    hydrazide_alert_count = _count_unique_matches(mol, _smarts("[CX3](=[OX1])[NX3][NX3]"))
    michael_acceptor_count = _count_unique_matches(mol, _smarts("[C,c]=[C,c]-[C](=O)[#6,#7,#8]"))
    epoxide_ring_count = _count_epoxides(mol)
    acyl_halide_count = _count_unique_matches(mol, _smarts("[CX3](=[OX1])[F,Cl,Br,I]"), (0,))
    sulfonyl_halide_count = _count_unique_matches(mol, _smarts("[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]"), (0,))
    anhydride_count = _count_unique_matches(mol, _smarts("[CX3](=[OX1])[OX2][CX3](=[OX1])"), (0, 3))
    isocyanate_count = _count_unique_matches(mol, _smarts("[NX2]=[CX2]=[OX1]"), (1,))
    isothiocyanate_count = _count_unique_matches(mol, _smarts("[NX2]=[CX2]=[SX1]"), (1,))
    ring_substituent_alerts = _collect_ring_substituent_alerts(mol)
    pains_filter_available, pains_alert_count = _count_pains_alerts(mol)

    strong_electrophile_count = (
        acyl_halide_count
        + sulfonyl_halide_count
        + anhydride_count
        + isocyanate_count
        + isothiocyanate_count
    )
    quinone_forming_phenol_alert_count = (
        ring_substituent_alerts["catechol_ring_count"]
        + ring_substituent_alerts["hydroquinone_ring_count"]
        + ring_substituent_alerts["aminophenol_ring_count"]
    )

    reactive_alert_present_flags = {
        "nitroaromatic": int(nitroaromatic_present),
        "furan": int(furan_ring_count > 0),
        "thiophene": int(thiophene_ring_count > 0),
        "aniline": int(ring_substituent_alerts["aromatic_primary_amine_count"] > 0),
        "hydrazine_or_hydrazide": int((hydrazine_alert_count + hydrazide_alert_count) > 0),
        "michael_acceptor": int(michael_acceptor_count > 0),
        "epoxide": int(epoxide_ring_count > 0),
        "strong_electrophile": int(strong_electrophile_count > 0),
        "quinone_forming_phenol": int(quinone_forming_phenol_alert_count > 0),
    }
    structural_alert_present_flags = {
        **reactive_alert_present_flags,
        "halogenated_aromatic": ring_substituent_alerts["halogenated_aromatic_ring_present"],
        "pains": 0 if _is_missing(pains_alert_count) else int(pains_alert_count > 0),
    }

    reactive_metabolite_alert_count = float(sum(reactive_alert_present_flags.values()))
    structural_alert_count = float(sum(structural_alert_present_flags.values()))

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": _as_float(rdkit.Descriptors.ExactMolWt(mol)),
            "heavy_atom_count": _as_float(mol.GetNumHeavyAtoms()),
            "logp": logp,
            "molar_refractivity": _as_float(rdkit.Crippen.MolMR(mol)),
            "tpsa": _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol)),
            "hbd": _as_float(rdkit.LipinskiL.NumHDonors(mol)),
            "hba": _as_float(rdkit.LipinskiL.NumHAcceptors(mol)),
            "heteroatom_count": _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "rotatable_bonds": rotatable_bonds,
            "ring_count": _as_float(rdkit.LipinskiL.RingCount(mol)),
            "aromatic_ring_count": _as_float(rdkit.LipinskiL.NumAromaticRings(mol)),
            "aliphatic_ring_count": _as_float(rdkit.LipinskiL.NumAliphaticRings(mol)),
            "fraction_csp3": _as_float(rdkit.LipinskiL.FractionCSP3(mol)),
            "formal_charge": _as_float(rdkit.Chem.GetFormalCharge(mol)),
            "radical_electrons": _as_float(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())),
            "halogen_atom_count": _as_float(_count_halogen_atoms(mol)),
            "fluorine_atom_count": _as_float(_count_atomic_number(mol, 9)),
            "chlorine_atom_count": _as_float(_count_atomic_number(mol, 17)),
            "bromine_atom_count": _as_float(_count_atomic_number(mol, 35)),
            "iodine_atom_count": _as_float(_count_atomic_number(mol, 53)),
            "aromatic_halogen_count": _as_float(_count_aromatic_halogen_substituents(mol)),
            "nitro_group_count": _as_float(nitro_group_count),
            "nitroaromatic_present": _flag(nitroaromatic_present),
            "furan_ring_count": _as_float(furan_ring_count),
            "furan_present": _flag(furan_ring_count > 0),
            "thiophene_ring_count": _as_float(thiophene_ring_count),
            "thiophene_present": _flag(thiophene_ring_count > 0),
            "aromatic_primary_amine_count": _as_float(ring_substituent_alerts["aromatic_primary_amine_count"]),
            "aromatic_primary_amine_present": _flag(ring_substituent_alerts["aromatic_primary_amine_count"] > 0),
            "hydrazine_alert_count": _as_float(hydrazine_alert_count),
            "hydrazide_alert_count": _as_float(hydrazide_alert_count),
            "hydrazine_or_hydrazide_present": _flag((hydrazine_alert_count + hydrazide_alert_count) > 0),
            "michael_acceptor_count": _as_float(michael_acceptor_count),
            "michael_acceptor_present": _flag(michael_acceptor_count > 0),
            "epoxide_ring_count": _as_float(epoxide_ring_count),
            "epoxide_present": _flag(epoxide_ring_count > 0),
            "acyl_halide_count": _as_float(acyl_halide_count),
            "sulfonyl_halide_count": _as_float(sulfonyl_halide_count),
            "anhydride_count": _as_float(anhydride_count),
            "isocyanate_count": _as_float(isocyanate_count),
            "isothiocyanate_count": _as_float(isothiocyanate_count),
            "strong_electrophile_count": _as_float(strong_electrophile_count),
            "strong_electrophile_present": _flag(strong_electrophile_count > 0),
            "phenol_count": _as_float(ring_substituent_alerts["phenol_count"]),
            "catechol_ring_count": _as_float(ring_substituent_alerts["catechol_ring_count"]),
            "hydroquinone_ring_count": _as_float(ring_substituent_alerts["hydroquinone_ring_count"]),
            "aminophenol_ring_count": _as_float(ring_substituent_alerts["aminophenol_ring_count"]),
            "quinone_forming_phenol_alert_count": _as_float(quinone_forming_phenol_alert_count),
            "quinone_forming_phenol_present": _flag(quinone_forming_phenol_alert_count > 0),
            "halogenated_aromatic_ring_present": _flag(ring_substituent_alerts["halogenated_aromatic_ring_present"] > 0),
            "polyhalogenated_aromatic_ring_present": _flag(ring_substituent_alerts["polyhalogenated_aromatic_ring_present"] > 0),
            "pains_filter_available": pains_filter_available,
            "pains_alert_count": pains_alert_count,
            "pains_alert_present": _flag(None if _is_missing(pains_alert_count) else pains_alert_count > 0),
            "structural_alert_count": structural_alert_count,
            "reactive_metabolite_alert_count": reactive_metabolite_alert_count,
            "high_lipophilicity_flag": _maybe_ge(logp, 3.0),
            "very_high_lipophilicity_flag": _maybe_ge(logp, 5.0),
            "large_molecular_weight_flag": _maybe_gt(mol_weight, 500.0),
            "multiple_rotatable_bonds_flag": _maybe_ge(rotatable_bonds, 8.0),
            "lipophilic_structural_alert_flag": _flag(logp >= 3.0 and reactive_metabolite_alert_count >= 1.0),
            "large_lipophilic_molecule_flag": _flag(mol_weight > 500.0 and logp >= 3.0),
            "polyhalogenated_lipophilic_alert_flag": _flag(
                ring_substituent_alerts["polyhalogenated_aromatic_ring_present"] > 0 and logp >= 3.0
            ),
            "reactive_alert_burden_ge_2": _flag(reactive_metabolite_alert_count >= 2.0),
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
            row["pains_filter_available"] = 1.0 if _pains_catalog() is not None else 0.0
        if include_smiles:
            row = {"smiles": smiles, **row}
        rows.append(row)

    return pd.DataFrame(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate DILI DeepResearch features from SMILES."
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
