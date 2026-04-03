#!/usr/bin/env python3
"""Deterministic ClinTox rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch ClinTox ruleset. The implemented features
focus on:

- structural alerts associated with reactive metabolites, electrophilicity,
  redox cycling, promiscuity, and metal-chelation risk
- physicochemical windows such as MW, logP, TPSA, HBD/HBA balance,
  flexibility, and aromatic ring burden
- MolGpKa-derived pKa/logD proxies for the "strong base / hERG-risk" rule

Intentionally skipped:
- clearance, half-life, and plasma protein binding endpoints that require
  assay or PK metadata beyond the molecular graph
- exact quantum-chemical electrophilicity indices and dedicated hERG models;
  this module exposes local structural proxies instead
- exhaustive metal-chelation coverage; only conservative catechol and
  hydroxamic-acid motifs are represented here
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
HYDROXAMIC_ACID_PATTERNS = (
    "[CX3](=[OX1])[NX3][OX2H1]",
    "[CX3](=[OX1])[NX3][OX1-]",
)


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("heteroatom_count", "Total heteroatom count."),
        ("oxygen_nitrogen_count", "Count of O and N atoms."),
        ("logp", "Wildman-Crippen logP."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("chiral_center_count", "Count of atom stereo centers."),
        ("unspecified_chiral_center_count", "Count of unspecified atom stereo centers."),
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
        ("nitroso_group_count", "Count of nitroso groups."),
        ("nitroso_present", "1 if a nitroso group is present."),
        ("aromatic_amine_count", "Count of primary or secondary aromatic amine nitrogens."),
        ("aromatic_amine_present", "1 if an aromatic amine alert is present."),
        ("michael_acceptor_count", "Count of alpha,beta-unsaturated carbonyl alerts."),
        ("michael_acceptor_present", "1 if an alpha,beta-unsaturated carbonyl alert is present."),
        ("catechol_ring_count", "Count of aromatic rings with adjacent phenolic substituents."),
        ("hydroquinone_ring_count", "Count of aromatic rings with para phenolic substituents."),
        ("quinone_ring_count", "Count of ortho- or para-quinone ring alerts."),
        ("quinone_or_catechol_alert_count", "Count of quinone, catechol, or hydroquinone redox-alert motifs."),
        ("quinone_or_catechol_present", "1 if any quinone/catechol/hydroquinone alert is present."),
        ("epoxide_ring_count", "Count of epoxide-like three-membered O-containing rings."),
        ("epoxide_present", "1 if an epoxide ring is present."),
        ("alkyl_halide_count", "Count of non-aromatic sp3 carbon-halogen alkylating motifs."),
        ("alkyl_halide_present", "1 if an alkyl halide alert is present."),
        ("aryl_halide_count", "Count of halogens directly attached to aromatic atoms."),
        ("aryl_chloride_count", "Count of chlorines directly attached to aromatic atoms."),
        ("aryl_halide_present", "1 if an aryl halide alert is present."),
        ("hydroxamic_acid_count", "Count of hydroxamic-acid metal-chelation motifs."),
        ("metal_chelator_alert_count", "Count of conservative metal-chelating alerts from catechol and hydroxamic-acid motifs."),
        ("metal_chelator_present", "1 if a conservative metal-chelation alert is present."),
        ("pains_filter_available", "1 if RDKit PAINS filters were available during featurization."),
        ("pains_alert_count", "Count of matched RDKit PAINS alerts."),
        ("pains_alert_present", "1 if any RDKit PAINS alert was matched."),
        ("electrophilic_alert_count", "Number of distinct electrophilic alert families matched."),
        ("electrophilic_alert_present", "1 if any major electrophilic alert family is present."),
        ("structural_alert_family_count", "Number of distinct ClinTox-oriented structural alert families matched."),
        ("mw_le_500", "Rule flag: molecular weight <= 500."),
        ("logp_le_5", "Rule flag: logP <= 5."),
        ("logp_lt_3", "Rule flag: logP < 3."),
        ("tpsa_gt_75", "Rule flag: TPSA > 75."),
        ("tpsa_le_140", "Rule flag: TPSA <= 140."),
        ("hbd_le_5", "Rule flag: HBD <= 5."),
        ("hba_le_10", "Rule flag: HBA <= 10."),
        ("rotatable_bonds_le_10", "Rule flag: rotatable bonds <= 10."),
        ("aromatic_ring_count_le_3", "Rule flag: aromatic ring count <= 3."),
        ("most_basic_pka_lt_7", "Rule flag: most basic predicted pKa < 7."),
        ("most_basic_pka_ge_7", "Rule flag: most basic predicted pKa >= 7."),
        ("property_alert_count", "Count of violated ClinTox physchem windows excluding pKa."),
        ("property_window_pass", "1 if the main ClinTox physchem windows are all satisfied."),
        ("basic_aryl_halide_risk_flag", "Heuristic alert combining an aryl halide with predicted basic pKa >= 7."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Clearance, half-life, and plasma protein binding are skipped because they require PK or assay metadata beyond SMILES.",
    "Exact quantum-chemical electrophilicity indices are not hard-coded; this module exposes local structural electrophile proxies instead.",
    "Dedicated hERG prediction is not embedded here; aryl-halide and basic-pKa features are provided as conservative proxy signals only.",
    "Metal-chelation coverage is intentionally conservative and limited to catechol and hydroxamic-acid motifs, not an exhaustive chelator catalog.",
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


def _maybe_gt(value: float | None, threshold: float) -> float:
    if _is_missing(value):
        return math.nan
    return _flag(value > threshold)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, rdMolDescriptors
        from rdkit.Chem.rdchem import HybridizationType
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run ClinTox feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        FilterCatalog=FilterCatalog,
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


def _count_unique_matches(mol, pattern, atom_positions: tuple[int, ...] | None = None) -> int:
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if atom_positions is None:
        return len(matches)
    unique_keys = {
        tuple(sorted(match[position] for position in atom_positions))
        for match in matches
    }
    return len(unique_keys)


def _count_union_matches(mol, patterns: Iterable[str], atom_positions: tuple[int, ...] | None = None) -> int:
    unique_keys: set[tuple[int, ...]] = set()
    raw_match_count = 0
    for pattern in patterns:
        matches = mol.GetSubstructMatches(_smarts(pattern), uniquify=True)
        if atom_positions is None:
            raw_match_count += len(matches)
            continue
        for match in matches:
            unique_keys.add(tuple(sorted(match[position] for position in atom_positions)))
    if atom_positions is None:
        return raw_match_count
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


def _collect_aromatic_redox_alerts(mol) -> dict[str, int]:
    catechol_ring_count = 0
    hydroquinone_ring_count = 0

    for ring in mol.GetRingInfo().AtomRings():
        ring_atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if not ring_atoms or not all(atom.GetIsAromatic() for atom in ring_atoms):
            continue

        phenol_positions: set[int] = set()
        for position, atom_idx in enumerate(ring):
            atom = mol.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                if neighbor_idx in ring:
                    continue
                if _is_phenol_like_substituent(mol, atom_idx, neighbor_idx):
                    phenol_positions.add(position)

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
    }


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
        if _cyclic_distance(carbonyl_positions[0], carbonyl_positions[1], 6) in (1, 3):
            quinone_count += 1
    return quinone_count


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


def _count_aromatic_halides(mol, halogens: set[int] | None = None) -> int:
    allowed_halogens = HALOGEN_ATOMIC_NUMBERS if halogens is None else halogens
    count = 0
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if begin.GetIsAromatic() and end.GetAtomicNum() in allowed_halogens:
            count += 1
        elif end.GetIsAromatic() and begin.GetAtomicNum() in allowed_halogens:
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
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.LipinskiL.NumHDonors(mol))
    hba = _as_float(rdkit.LipinskiL.NumHAcceptors(mol))
    total_hbond_donors_acceptors = hbd + hba
    rotatable_bonds = _as_float(rdkit.LipinskiL.NumRotatableBonds(mol))
    aromatic_ring_count = _as_float(rdkit.LipinskiL.NumAromaticRings(mol))

    ring_alerts = _collect_aromatic_redox_alerts(mol)
    nitroso_group_count = _count_unique_matches(mol, _smarts("[#6,#7,#8,#16,a]-[N;X2]=O"), (1,))
    aromatic_amine_count = _count_aromatic_amines(mol)
    michael_acceptor_count = _count_unique_matches(mol, _smarts("[C,c]=[C,c]-[C](=O)[#6,#7,#8]"))
    quinone_ring_count = _count_quinone_like_rings(mol)
    quinone_or_catechol_alert_count = (
        ring_alerts["catechol_ring_count"]
        + ring_alerts["hydroquinone_ring_count"]
        + quinone_ring_count
    )
    epoxide_ring_count = _count_epoxides(mol)
    alkyl_halide_count = _count_alkyl_halides(mol)
    aryl_halide_count = _count_aromatic_halides(mol)
    aryl_chloride_count = _count_aromatic_halides(mol, halogens={17})
    hydroxamic_acid_count = _count_union_matches(mol, HYDROXAMIC_ACID_PATTERNS, (0,))
    metal_chelator_alert_count = ring_alerts["catechol_ring_count"] + hydroxamic_acid_count
    pains_filter_available, pains_alert_count = _count_pains_alerts(mol)

    electrophilic_alert_flags = {
        "nitroso": int(nitroso_group_count > 0),
        "michael_acceptor": int(michael_acceptor_count > 0),
        "quinone_or_catechol": int(quinone_or_catechol_alert_count > 0),
        "epoxide": int(epoxide_ring_count > 0),
        "alkyl_halide": int(alkyl_halide_count > 0),
    }
    electrophilic_alert_count = float(sum(electrophilic_alert_flags.values()))
    structural_alert_family_count = float(
        electrophilic_alert_count
        + int(aromatic_amine_count > 0)
        + int(aryl_halide_count > 0)
        + int(metal_chelator_alert_count > 0)
        + (0 if _is_missing(pains_alert_count) else int(pains_alert_count > 0))
    )

    property_alert_count = float(
        sum(
            [
                int(mol_weight > 500.0),
                int(logp > 5.0),
                int(not (tpsa > 75.0 and tpsa <= 140.0)),
                int(hbd > 5.0),
                int(hba > 10.0),
                int(rotatable_bonds > 10.0),
                int(aromatic_ring_count > 3.0),
            ]
        )
    )

    features.update(_compute_pka_summary(mol))

    most_basic_pka = features["most_basic_pka"]
    basic_aryl_halide_risk_flag = (
        math.nan
        if _is_missing(most_basic_pka)
        else _flag(aryl_halide_count > 0 and most_basic_pka >= 7.0)
    )

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": _as_float(rdkit.Descriptors.ExactMolWt(mol)),
            "heavy_atom_count": _as_float(mol.GetNumHeavyAtoms()),
            "heteroatom_count": _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "oxygen_nitrogen_count": _as_float(_count_oxygen_and_nitrogen(mol)),
            "logp": logp,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbond_donors_acceptors,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": _as_float(rdkit.LipinskiL.RingCount(mol)),
            "aromatic_ring_count": aromatic_ring_count,
            "chiral_center_count": _as_float(rdkit.rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
            "unspecified_chiral_center_count": _as_float(rdkit.rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)),
            "nitroso_group_count": _as_float(nitroso_group_count),
            "nitroso_present": _flag(nitroso_group_count > 0),
            "aromatic_amine_count": _as_float(aromatic_amine_count),
            "aromatic_amine_present": _flag(aromatic_amine_count > 0),
            "michael_acceptor_count": _as_float(michael_acceptor_count),
            "michael_acceptor_present": _flag(michael_acceptor_count > 0),
            "catechol_ring_count": _as_float(ring_alerts["catechol_ring_count"]),
            "hydroquinone_ring_count": _as_float(ring_alerts["hydroquinone_ring_count"]),
            "quinone_ring_count": _as_float(quinone_ring_count),
            "quinone_or_catechol_alert_count": _as_float(quinone_or_catechol_alert_count),
            "quinone_or_catechol_present": _flag(quinone_or_catechol_alert_count > 0),
            "epoxide_ring_count": _as_float(epoxide_ring_count),
            "epoxide_present": _flag(epoxide_ring_count > 0),
            "alkyl_halide_count": _as_float(alkyl_halide_count),
            "alkyl_halide_present": _flag(alkyl_halide_count > 0),
            "aryl_halide_count": _as_float(aryl_halide_count),
            "aryl_chloride_count": _as_float(aryl_chloride_count),
            "aryl_halide_present": _flag(aryl_halide_count > 0),
            "hydroxamic_acid_count": _as_float(hydroxamic_acid_count),
            "metal_chelator_alert_count": _as_float(metal_chelator_alert_count),
            "metal_chelator_present": _flag(metal_chelator_alert_count > 0),
            "pains_filter_available": pains_filter_available,
            "pains_alert_count": pains_alert_count,
            "pains_alert_present": _flag(None if _is_missing(pains_alert_count) else pains_alert_count > 0),
            "electrophilic_alert_count": electrophilic_alert_count,
            "electrophilic_alert_present": _flag(electrophilic_alert_count > 0),
            "structural_alert_family_count": structural_alert_family_count,
            "mw_le_500": _maybe_le(mol_weight, 500.0),
            "logp_le_5": _maybe_le(logp, 5.0),
            "logp_lt_3": _maybe_lt(logp, 3.0),
            "tpsa_gt_75": _maybe_gt(tpsa, 75.0),
            "tpsa_le_140": _maybe_le(tpsa, 140.0),
            "hbd_le_5": _maybe_le(hbd, 5.0),
            "hba_le_10": _maybe_le(hba, 10.0),
            "rotatable_bonds_le_10": _maybe_le(rotatable_bonds, 10.0),
            "aromatic_ring_count_le_3": _maybe_le(aromatic_ring_count, 3.0),
            "most_basic_pka_lt_7": _maybe_lt(most_basic_pka, 7.0),
            "most_basic_pka_ge_7": _maybe_ge(most_basic_pka, 7.0),
            "property_alert_count": property_alert_count,
            "property_window_pass": _flag(property_alert_count == 0.0),
            "basic_aryl_halide_risk_flag": basic_aryl_halide_risk_flag,
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
            row["pka_features_available"] = 0.0
        if include_smiles:
            row = {"smiles": smiles, **row}
        rows.append(row)

    return pd.DataFrame(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate ClinTox DeepResearch features from SMILES."
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
