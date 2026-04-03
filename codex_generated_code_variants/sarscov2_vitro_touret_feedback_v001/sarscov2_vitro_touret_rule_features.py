#!/usr/bin/env python3
"""Deterministic SARSCoV2_Vitro_Touret rule-to-feature code for downstream ML.

This module converts SMILES strings into numeric features distilled from the
computable parts of the DeepResearch SARS-CoV-2 in vitro activity ruleset.

Implemented feature groups:
- core drug-like descriptors behind the MW/logP/TPSA/HBD/HBA/rotatable-bond
  windows highlighted in the source response
- pKa/logD-derived ionization features using the local MolGpKa helper
- aromatic/heterocycle, ring-topology, and stereochemistry proxies for the
  medicinal-chemistry design guidance in the source response
- liability filters for PAINS, reactive electrophiles, metal-chelating motifs,
  and simple metabolic-lability motifs

Intentionally skipped:
- potency-dependent metrics such as LipE and ligand efficiency, because they
  require assay activity values rather than ligand-only structure
- experimental solubility and permeability endpoints, because the repository
  does not ship validated local predictors for those assay properties
- context-specific target-mechanism rules for when covalent warheads are
  acceptable, because the source response is for a broad in vitro antiviral
  screen rather than a single covalent target program
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

MICHAEL_ACCEPTOR_PATTERN_SPECS = (
    ("[C,c]=[C,c][CX3](=[OX1])[#6,#7,#8]", (2,)),
)
ALKYL_HALIDE_PATTERN_SPECS = (
    ("[CX4][Cl,Br,I]", (0,)),
)
ACYL_HALIDE_PATTERN_SPECS = (
    ("[CX3](=[OX1])[F,Cl,Br,I]", (0,)),
)
SULFONYL_HALIDE_PATTERN_SPECS = (
    ("[SX4](=[OX1])(=[OX1])[F,Cl,Br,I]", (0,)),
)
ISOCYANATE_PATTERN_SPECS = (
    ("[NX2]=[CX2]=[OX1]", (1,)),
)
ISOTHIOCYANATE_PATTERN_SPECS = (
    ("[NX2]=[CX2]=[SX1]", (1,)),
)
ANHYDRIDE_PATTERN_SPECS = (
    ("[CX3](=[OX1])[OX2][CX3](=[OX1])", (0, 3)),
)
HYDROXAMATE_PATTERN_SPECS = (
    ("[CX3](=[OX1])[NX3][OX2H,OX1-]", (0,)),
)
BETA_DICARBONYL_PATTERN_SPECS = (
    ("[CX3](=[OX1])[#6][CX3](=[OX1])", (0, 3)),
)


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("heteroatom_count", "Total heteroatom count."),
        ("logp", "Wildman-Crippen logP."),
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
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("abs_formal_charge", "Absolute formal charge magnitude."),
        ("stereocenter_count", "Count of assigned or potential tetrahedral stereocenters."),
        ("most_basic_pka", "Most basic predicted pKa from the MolGpKa helper."),
        ("most_acidic_pka", "Most acidic predicted pKa from the MolGpKa helper."),
        ("num_basic_sites", "Number of predicted basic sites."),
        ("num_acidic_sites", "Number of predicted acidic sites."),
        ("neutral_fraction_ph74", "Estimated neutral fraction at pH 7.4."),
        ("charged_fraction_ph74", "1 - neutral_fraction_ph74."),
        ("base_protonated_fraction_ph74", "Estimated protonated fraction for the dominant basic site."),
        ("acid_deprotonated_fraction_ph74", "Estimated deprotonated fraction for the dominant acidic site."),
        ("net_charge_proxy_ph74", "Base protonation minus acid deprotonation proxy at pH 7.4."),
        ("estimated_logd_ph74", "Estimated logD at pH 7.4 from logP and neutral fraction."),
        ("has_amphoteric_sites", "1 if both acidic and basic sites are predicted."),
        ("ionizable_center_present", "1 if at least one acidic or basic site is predicted."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
        ("pains_filter_available", "1 if RDKit PAINS filters were available during featurization."),
        ("pains_alert_count", "Count of matched RDKit PAINS alerts."),
        ("pains_alert_present", "1 if any RDKit PAINS alert was matched."),
        ("michael_acceptor_count", "Count of alpha,beta-unsaturated carbonyl alerts."),
        ("aldehyde_count", "Count of aldehyde alerts."),
        ("epoxide_ring_count", "Count of epoxide-like three-membered O-containing rings."),
        ("aziridine_ring_count", "Count of aziridine-like three-membered N-containing rings."),
        ("alkyl_halide_count", "Count of sp3 carbon-halogen SN2 alkylation alerts."),
        ("acyl_halide_count", "Count of acyl halide alerts."),
        ("sulfonyl_halide_count", "Count of sulfonyl halide alerts."),
        ("isocyanate_count", "Count of isocyanate alerts."),
        ("isothiocyanate_count", "Count of isothiocyanate alerts."),
        ("anhydride_count", "Count of acid anhydride alerts."),
        ("reactive_alert_site_count", "Combined count of electrophilic or covalent-liability alert sites."),
        ("reactive_alert_present", "1 if any reactive electrophile alert is present."),
        ("catechol_ring_count", "Count of aromatic rings with adjacent phenolic OH substituents."),
        ("hydroxamate_count", "Count of hydroxamate metal-chelation alerts."),
        ("beta_dicarbonyl_count", "Count of 1,3-dicarbonyl metal-chelation proxy motifs."),
        ("metal_chelator_alert_count", "Combined count of common metal-chelation liability motifs."),
        ("metal_chelator_alert_present", "1 if any common metal-chelator alert is present."),
        ("ester_count", "Count of ester motifs."),
        ("phenol_count", "Count of phenol or phenoxide substituents on aromatic rings."),
        ("metabolic_lability_alert_count", "Combined count of simple metabolic-lability motifs such as esters and phenols."),
        ("metabolic_lability_alert_present", "1 if a simple metabolic-lability motif is present."),
        ("aromatic_or_heterocycle_present", "1 if at least one aromatic ring or heterocycle ring is present."),
        ("mol_weight_lt_500", "Rule flag: molecular weight < 500."),
        ("logp_le_5", "Rule flag: logP <= 5."),
        ("logp_in_1_3", "Rule flag: 1 <= logP <= 3."),
        ("estimated_logd_le_5", "Rule flag: estimated logD(7.4) <= 5."),
        ("estimated_logd_in_1_3", "Rule flag: 1 <= estimated logD(7.4) <= 3."),
        ("tpsa_le_140", "Rule flag: TPSA <= 140."),
        ("hbd_le_5", "Rule flag: HBD <= 5."),
        ("hba_le_10", "Rule flag: HBA <= 10."),
        ("rotatable_bonds_le_10", "Rule flag: rotatable bonds <= 10."),
        ("basic_ionization_ok", "Rule flag: no basic site is predicted or the most basic pKa is < 10."),
        ("acidic_ionization_ok", "Rule flag: no acidic site is predicted or the most acidic pKa is > 3."),
        ("formal_charge_zero_or_plus_one", "Rule flag: formal charge is 0 or +1."),
        ("formal_charge_abs_le_1", "Rule flag: abs(formal charge) <= 1."),
        ("fraction_csp3_ge_0p3", "Rule flag: fractionCSP3 >= 0.3."),
        ("stereocenter_count_ge_1", "Rule flag: at least one stereocenter is present."),
        ("aromatic_ring_count_ge_1", "Rule flag: aromatic ring count >= 1."),
        ("aromatic_ring_count_1_3", "Rule flag: 1 <= aromatic ring count <= 3."),
        ("ring_count_4_7", "Rule flag: 4 <= total ring count <= 7."),
        ("pains_alert_absent", "Rule flag: no RDKit PAINS alert is matched."),
        ("reactive_alert_absent", "Rule flag: no reactive electrophile alert is matched."),
        ("metal_chelator_alert_absent", "Rule flag: no common metal-chelator alert is matched."),
        ("metabolic_lability_alert_absent", "Rule flag: no simple metabolic-lability motif is matched."),
        ("druglike_physchem_pass_count", "Count of passed MW/logP/TPSA/HBD/HBA/rotatable-bond rules."),
        ("druglike_physchem_window_pass", "1 if all six main drug-like physicochemical rules pass."),
        ("ionization_window_pass", "1 if charge and pKa-derived ionization heuristics all pass."),
        ("liability_alert_absent_pass", "1 if PAINS, reactive-electrophile, and metal-chelator alerts are all absent."),
        (
            "sarscov2_vitro_design_rule_pass_count",
            "Count of passed physicochemical, scaffold, saturation, and liability heuristics distilled from the source response.",
        ),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "LipE and ligand efficiency are skipped because they require potency values such as IC50 or pIC50, not just SMILES.",
    "Measured aqueous solubility and permeability endpoints are skipped because the repository does not ship validated local predictors for those assay properties.",
    "The source response only conditionally allows covalent warheads for target-specific mechanisms; this module encodes general reactive-alert counts but not assay-context exceptions.",
    "Metabolic stability is only approximated with simple ester and phenol liabilities; no clearance, microsomal stability, or metabolism simulator is hard-coded.",
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


def _maybe_between(value: float | None, lower: float, upper: float, *, inclusive: bool = True) -> float:
    if _is_missing(value):
        return math.nan
    if inclusive:
        return _flag(lower <= value <= upper)
    return _flag(lower < value < upper)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Fragments, Lipinski, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run SARSCoV2_Vitro_Touret feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        FilterCatalog=FilterCatalog,
        Fragments=Fragments,
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


def _count_unique_matches(mol, pattern, atom_positions: tuple[int, ...] | None = None) -> int:
    matches = mol.GetSubstructMatches(pattern, uniquify=True)
    if atom_positions is None:
        return len(matches)
    unique_keys = {
        tuple(sorted(match[position] for position in atom_positions))
        for match in matches
    }
    return len(unique_keys)


def _count_unique_matches_any(
    mol,
    pattern_specs: Iterable[tuple[str, tuple[int, ...] | None]],
) -> int:
    unique_keys: set[tuple[int, ...]] = set()
    for pattern, atom_positions in pattern_specs:
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


def _count_stereocenters(mol) -> int:
    chiral_centers = _rdkit().Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return len(chiral_centers)


def _count_heterocycle_rings(mol, *, aromatic_only: bool = False) -> int:
    count = 0
    for ring_atoms in mol.GetRingInfo().AtomRings():
        ring_atom_objs = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring_atoms]
        if aromatic_only and not all(atom.GetIsAromatic() for atom in ring_atom_objs):
            continue
        if any(atom.GetAtomicNum() != 6 for atom in ring_atom_objs):
            count += 1
    return count


def _count_epoxide_rings(mol) -> int:
    count = 0
    for ring_atoms in mol.GetRingInfo().AtomRings():
        if len(ring_atoms) != 3:
            continue
        ring_atom_objs = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring_atoms]
        atomic_nums = sorted(atom.GetAtomicNum() for atom in ring_atom_objs)
        if atomic_nums != [6, 6, 8]:
            continue
        ring_is_all_single = True
        for index, atom_idx in enumerate(ring_atoms):
            next_idx = ring_atoms[(index + 1) % len(ring_atoms)]
            bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                ring_is_all_single = False
                break
        if ring_is_all_single:
            count += 1
    return count


def _count_aziridine_rings(mol) -> int:
    count = 0
    for ring_atoms in mol.GetRingInfo().AtomRings():
        if len(ring_atoms) != 3:
            continue
        ring_atom_objs = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring_atoms]
        atomic_nums = sorted(atom.GetAtomicNum() for atom in ring_atom_objs)
        if atomic_nums != [6, 6, 7]:
            continue
        ring_is_all_single = True
        for index, atom_idx in enumerate(ring_atoms):
            next_idx = ring_atoms[(index + 1) % len(ring_atoms)]
            bond = mol.GetBondBetweenAtoms(atom_idx, next_idx)
            if bond is None or bond.GetBondTypeAsDouble() != 1.0:
                ring_is_all_single = False
                break
        if ring_is_all_single:
            count += 1
    return count


def _has_phenolic_oxygen_substituent(mol, ring_atom_idx: int, ring_atom_set: set[int]) -> bool:
    ring_atom = mol.GetAtomWithIdx(ring_atom_idx)
    for neighbor in ring_atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx in ring_atom_set:
            continue
        if neighbor.GetAtomicNum() != 8:
            continue
        bond = mol.GetBondBetweenAtoms(ring_atom_idx, neighbor_idx)
        if bond is None or bond.GetBondTypeAsDouble() != 1.0:
            continue
        if neighbor.GetFormalCharge() < 0 or neighbor.GetTotalNumHs() > 0:
            return True
    return False


def _count_catechol_rings(mol) -> int:
    count = 0
    for ring_atoms in mol.GetRingInfo().AtomRings():
        if len(ring_atoms) != 6:
            continue
        ring_atom_objs = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring_atoms]
        if not all(atom.GetIsAromatic() for atom in ring_atom_objs):
            continue

        ring_atom_set = set(ring_atoms)
        phenolic_positions = [
            index
            for index, atom_idx in enumerate(ring_atoms)
            if _has_phenolic_oxygen_substituent(mol, atom_idx, ring_atom_set)
        ]
        if len(phenolic_positions) < 2:
            continue

        position_set = set(phenolic_positions)
        if any(((position + 1) % len(ring_atoms)) in position_set for position in phenolic_positions):
            count += 1
    return count


def _count_pains_alerts(mol) -> tuple[float, float]:
    catalog = _pains_catalog()
    if catalog is None:
        return 0.0, math.nan
    matches = catalog.GetMatches(mol)
    return 1.0, float(len(matches))


def _compute_pka_summary(logp: float, mol) -> dict[str, float]:
    result = {
        "most_basic_pka": 0.0,
        "most_acidic_pka": 14.0,
        "num_basic_sites": math.nan,
        "num_acidic_sites": math.nan,
        "neutral_fraction_ph74": math.nan,
        "charged_fraction_ph74": math.nan,
        "base_protonated_fraction_ph74": math.nan,
        "acid_deprotonated_fraction_ph74": math.nan,
        "net_charge_proxy_ph74": math.nan,
        "estimated_logd_ph74": math.nan,
        "has_amphoteric_sites": math.nan,
        "ionizable_center_present": math.nan,
        "pka_features_available": 0.0,
    }

    try:
        predictor = _get_pka_predictor()
        prediction = predictor.predict(mol)
    except Exception:
        return result

    base_sites = getattr(prediction, "base_sites_1", {}) or {}
    acid_sites = getattr(prediction, "acid_sites_1", {}) or {}

    most_basic_pka = max(base_sites.values()) if base_sites else 0.0
    most_acidic_pka = min(acid_sites.values()) if acid_sites else 14.0
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
    estimated_logd = logp + math.log10(neutral_fraction)

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
            "estimated_logd_ph74": estimated_logd,
            "has_amphoteric_sites": float(bool(base_sites and acid_sites)),
            "ionizable_center_present": float(bool(base_sites or acid_sites)),
            "pka_features_available": 1.0,
        }
    )
    return result


def _ionization_rule_flag(
    *,
    num_sites: float,
    value: float,
    comparator,
) -> float:
    if _is_missing(num_sites):
        return math.nan
    if num_sites == 0.0:
        return 1.0
    if _is_missing(value):
        return math.nan
    return _flag(comparator(value))


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    exact_mol_weight = _as_float(rdkit.Descriptors.ExactMolWt(mol))
    heavy_atom_count = _as_float(mol.GetNumHeavyAtoms())
    heteroatom_count = _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    total_hbonds = hbd + hba
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.Lipinski.NumAliphaticRings(mol))
    heterocycle_ring_count = _as_float(_count_heterocycle_rings(mol))
    aromatic_heterocycle_ring_count = _as_float(_count_heterocycle_rings(mol, aromatic_only=True))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    abs_formal_charge = _as_float(abs(formal_charge))
    stereocenter_count = _as_float(_count_stereocenters(mol))

    pains_filter_available, pains_alert_count = _count_pains_alerts(mol)

    michael_acceptor_count = _as_float(_count_unique_matches_any(mol, MICHAEL_ACCEPTOR_PATTERN_SPECS))
    aldehyde_count = _as_float(rdkit.Fragments.fr_aldehyde(mol))
    epoxide_ring_count = _as_float(_count_epoxide_rings(mol))
    aziridine_ring_count = _as_float(_count_aziridine_rings(mol))
    alkyl_halide_count = _as_float(_count_unique_matches_any(mol, ALKYL_HALIDE_PATTERN_SPECS))
    acyl_halide_count = _as_float(_count_unique_matches_any(mol, ACYL_HALIDE_PATTERN_SPECS))
    sulfonyl_halide_count = _as_float(_count_unique_matches_any(mol, SULFONYL_HALIDE_PATTERN_SPECS))
    isocyanate_count = _as_float(_count_unique_matches_any(mol, ISOCYANATE_PATTERN_SPECS))
    isothiocyanate_count = _as_float(_count_unique_matches_any(mol, ISOTHIOCYANATE_PATTERN_SPECS))
    anhydride_count = _as_float(_count_unique_matches_any(mol, ANHYDRIDE_PATTERN_SPECS))
    reactive_alert_site_count = (
        michael_acceptor_count
        + aldehyde_count
        + epoxide_ring_count
        + aziridine_ring_count
        + alkyl_halide_count
        + acyl_halide_count
        + sulfonyl_halide_count
        + isocyanate_count
        + isothiocyanate_count
        + anhydride_count
    )

    catechol_ring_count = _as_float(_count_catechol_rings(mol))
    hydroxamate_count = _as_float(_count_unique_matches_any(mol, HYDROXAMATE_PATTERN_SPECS))
    beta_dicarbonyl_count = _as_float(_count_unique_matches_any(mol, BETA_DICARBONYL_PATTERN_SPECS))
    metal_chelator_alert_count = catechol_ring_count + hydroxamate_count + beta_dicarbonyl_count

    ester_count = _as_float(rdkit.Fragments.fr_ester(mol))
    phenol_count = _as_float(rdkit.Fragments.fr_phenol(mol))
    metabolic_lability_alert_count = ester_count + phenol_count

    features.update(_compute_pka_summary(logp, mol))

    basic_ionization_ok = _ionization_rule_flag(
        num_sites=features["num_basic_sites"],
        value=features["most_basic_pka"],
        comparator=lambda value: value < 10.0,
    )
    acidic_ionization_ok = _ionization_rule_flag(
        num_sites=features["num_acidic_sites"],
        value=features["most_acidic_pka"],
        comparator=lambda value: value > 3.0,
    )

    pains_alert_present = _flag(None if _is_missing(pains_alert_count) else pains_alert_count > 0.0)
    pains_alert_absent = _flag(None if _is_missing(pains_alert_count) else pains_alert_count == 0.0)
    reactive_alert_present = _flag(reactive_alert_site_count > 0.0)
    metal_chelator_alert_present = _flag(metal_chelator_alert_count > 0.0)
    metabolic_lability_alert_present = _flag(metabolic_lability_alert_count > 0.0)
    aromatic_or_heterocycle_present = _flag(aromatic_ring_count > 0.0 or heterocycle_ring_count > 0.0)

    physchem_flags = {
        "mol_weight_lt_500": mol_weight < 500.0,
        "logp_le_5": logp <= 5.0,
        "tpsa_le_140": tpsa <= 140.0,
        "hbd_le_5": hbd <= 5.0,
        "hba_le_10": hba <= 10.0,
        "rotatable_bonds_le_10": rotatable_bonds <= 10.0,
    }
    druglike_physchem_pass_count = float(sum(physchem_flags.values()))
    druglike_physchem_window_pass = _flag(all(physchem_flags.values()))

    formal_charge_zero_or_plus_one = _flag(formal_charge in (0.0, 1.0))
    formal_charge_abs_le_1 = _flag(abs_formal_charge <= 1.0)
    ionization_window_pass = _flag(
        None
        if any(
            _is_missing(value)
            for value in (basic_ionization_ok, acidic_ionization_ok, formal_charge_zero_or_plus_one)
        )
        else bool(basic_ionization_ok and acidic_ionization_ok and formal_charge_zero_or_plus_one)
    )
    liability_alert_absent_pass = _flag(
        None
        if _is_missing(pains_alert_count)
        else pains_alert_count == 0.0 and reactive_alert_site_count == 0.0 and metal_chelator_alert_count == 0.0
    )

    design_rule_flags = [
        float(physchem_flags["mol_weight_lt_500"]),
        float(physchem_flags["logp_le_5"]),
        float(physchem_flags["tpsa_le_140"]),
        float(physchem_flags["hbd_le_5"]),
        float(physchem_flags["hba_le_10"]),
        float(physchem_flags["rotatable_bonds_le_10"]),
        aromatic_or_heterocycle_present,
        _maybe_between(aromatic_ring_count, 1.0, 3.0),
        _maybe_between(ring_count, 4.0, 7.0),
        _maybe_ge(fraction_csp3, 0.3),
        pains_alert_absent,
        _flag(reactive_alert_site_count == 0.0),
        _flag(metal_chelator_alert_count == 0.0),
        _flag(metabolic_lability_alert_count == 0.0),
    ]
    sarscov2_vitro_design_rule_pass_count = float(
        sum(1 for value in design_rule_flags if not _is_missing(value) and value > 0.0)
    )

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
            "heteroatom_count": heteroatom_count,
            "logp": logp,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbonds,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "heterocycle_ring_count": heterocycle_ring_count,
            "aromatic_heterocycle_ring_count": aromatic_heterocycle_ring_count,
            "fraction_csp3": fraction_csp3,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "abs_formal_charge": abs_formal_charge,
            "stereocenter_count": stereocenter_count,
            "pains_filter_available": pains_filter_available,
            "pains_alert_count": pains_alert_count,
            "pains_alert_present": pains_alert_present,
            "michael_acceptor_count": michael_acceptor_count,
            "aldehyde_count": aldehyde_count,
            "epoxide_ring_count": epoxide_ring_count,
            "aziridine_ring_count": aziridine_ring_count,
            "alkyl_halide_count": alkyl_halide_count,
            "acyl_halide_count": acyl_halide_count,
            "sulfonyl_halide_count": sulfonyl_halide_count,
            "isocyanate_count": isocyanate_count,
            "isothiocyanate_count": isothiocyanate_count,
            "anhydride_count": anhydride_count,
            "reactive_alert_site_count": reactive_alert_site_count,
            "reactive_alert_present": reactive_alert_present,
            "catechol_ring_count": catechol_ring_count,
            "hydroxamate_count": hydroxamate_count,
            "beta_dicarbonyl_count": beta_dicarbonyl_count,
            "metal_chelator_alert_count": metal_chelator_alert_count,
            "metal_chelator_alert_present": metal_chelator_alert_present,
            "ester_count": ester_count,
            "phenol_count": phenol_count,
            "metabolic_lability_alert_count": metabolic_lability_alert_count,
            "metabolic_lability_alert_present": metabolic_lability_alert_present,
            "aromatic_or_heterocycle_present": aromatic_or_heterocycle_present,
            "mol_weight_lt_500": _flag(physchem_flags["mol_weight_lt_500"]),
            "logp_le_5": _flag(physchem_flags["logp_le_5"]),
            "logp_in_1_3": _maybe_between(logp, 1.0, 3.0),
            "estimated_logd_le_5": _maybe_le(features["estimated_logd_ph74"], 5.0),
            "estimated_logd_in_1_3": _maybe_between(features["estimated_logd_ph74"], 1.0, 3.0),
            "tpsa_le_140": _flag(physchem_flags["tpsa_le_140"]),
            "hbd_le_5": _flag(physchem_flags["hbd_le_5"]),
            "hba_le_10": _flag(physchem_flags["hba_le_10"]),
            "rotatable_bonds_le_10": _flag(physchem_flags["rotatable_bonds_le_10"]),
            "basic_ionization_ok": basic_ionization_ok,
            "acidic_ionization_ok": acidic_ionization_ok,
            "formal_charge_zero_or_plus_one": formal_charge_zero_or_plus_one,
            "formal_charge_abs_le_1": formal_charge_abs_le_1,
            "fraction_csp3_ge_0p3": _maybe_ge(fraction_csp3, 0.3),
            "stereocenter_count_ge_1": _maybe_ge(stereocenter_count, 1.0),
            "aromatic_ring_count_ge_1": _maybe_ge(aromatic_ring_count, 1.0),
            "aromatic_ring_count_1_3": _maybe_between(aromatic_ring_count, 1.0, 3.0),
            "ring_count_4_7": _maybe_between(ring_count, 4.0, 7.0),
            "pains_alert_absent": pains_alert_absent,
            "reactive_alert_absent": _flag(reactive_alert_site_count == 0.0),
            "metal_chelator_alert_absent": _flag(metal_chelator_alert_count == 0.0),
            "metabolic_lability_alert_absent": _flag(metabolic_lability_alert_count == 0.0),
            "druglike_physchem_pass_count": druglike_physchem_pass_count,
            "druglike_physchem_window_pass": druglike_physchem_window_pass,
            "ionization_window_pass": ionization_window_pass,
            "liability_alert_absent_pass": liability_alert_absent_pass,
            "sarscov2_vitro_design_rule_pass_count": sarscov2_vitro_design_rule_pass_count,
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
            row["pains_filter_available"] = 1.0 if _pains_catalog() is not None else 0.0
        if include_smiles:
            row = {"smiles": smiles, **row}
        rows.append(row)

    return pd.DataFrame(rows)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SARSCoV2_Vitro_Touret DeepResearch features from SMILES."
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
