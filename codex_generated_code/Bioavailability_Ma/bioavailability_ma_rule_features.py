#!/usr/bin/env python3
"""Deterministic Bioavailability_Ma rule-to-feature code for downstream ML.

This module turns SMILES strings into numeric features distilled from the
computable parts of the DeepResearch oral bioavailability ruleset.

Implemented feature groups:
- direct physicochemical descriptors behind Lipinski, Veber, Egan, Ghose,
  Oprea, Rule-of-3, and Muegge filters
- aromaticity, saturation, stereochemistry, and intramolecular H-bond proxies
- pKa/logD-derived ionization and salt-formability proxies via the shared
  MolGpKa helper used elsewhere in this repository

Intentionally skipped:
- experimental oral exposure endpoints such as measured F, Fa, Caco-2, or
  formulation-dependent absorption values
- exact crystal packing, conformer ensembles, or 3D intramolecular H-bond
  energetics that would require validated external modeling
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


def _feature_specs() -> list[tuple[str, str]]:
    return [
        ("mol_weight", "Molecular weight (Descriptors.MolWt)."),
        ("exact_mol_weight", "Exact molecular weight (Descriptors.ExactMolWt)."),
        ("heavy_atom_count", "Heavy atom count."),
        ("total_atom_count", "Total atom count after adding explicit hydrogens as a Ghose-style atom-count proxy."),
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("heteroatom_count", "Total heteroatom count."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("aliphatic_ring_count", "Aliphatic ring count."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("stereocenter_count", "Count of assigned or potential tetrahedral stereocenters."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("abs_formal_charge", "Absolute formal charge magnitude."),
        ("labute_asa", "Labute approximate surface area."),
        ("labute_asa_per_heavy_atom", "Labute ASA divided by heavy-atom count."),
        ("aromatic_heavy_atom_fraction", "Aromatic atom fraction among heavy atoms."),
        ("fused_aromatic_ring_count", "Count of aromatic rings participating in fused aromatic systems."),
        (
            "intramolecular_hbond_pair_count",
            "Count of donor-acceptor atom pairs 4-8 bonds apart as a proxy for intramolecular H-bond potential.",
        ),
        ("intramolecular_hbond_potential_present", "1 if the intramolecular H-bond proxy count is non-zero."),
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
        ("has_amphoteric_sites", "1 if both acidic and basic sites are predicted."),
        ("ionizable_center_present", "1 if at least one acidic or basic site is predicted."),
        ("moderate_basic_pka_8_12", "Heuristic flag: a predicted basic site exists and its most basic pKa is between 8 and 12."),
        ("moderate_acidic_pka_2_6", "Heuristic flag: a predicted acidic site exists and its most acidic pKa is between 2 and 6."),
        ("any_moderate_pka_2_12", "Heuristic flag: at least one predicted acidic/basic site falls in the broad 2-12 oral-ionization window."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
        ("mw_le_500", "Rule flag: MW <= 500."),
        ("logp_le_5", "Rule flag: logP <= 5."),
        ("hbd_le_5", "Rule flag: HBD <= 5."),
        ("hba_le_10", "Rule flag: HBA <= 10."),
        ("lipinski_pass", "Composite Lipinski Rule-of-5 pass flag."),
        ("lipinski_violation_count", "Number of Lipinski cutoffs violated among MW/logP/HBD/HBA."),
        ("rotatable_bonds_le_10", "Rule flag: rotatable bonds <= 10."),
        ("tpsa_le_140", "Rule flag: TPSA <= 140."),
        ("veber_pass", "Composite Veber pass flag."),
        ("logp_le_5p88", "Rule flag: logP <= 5.88."),
        ("tpsa_le_131p6", "Rule flag: TPSA <= 131.6."),
        ("egan_pass", "Composite Egan pass flag."),
        ("ghose_mw_in_160_480", "Rule flag: 160 < MW < 480."),
        ("ghose_logp_in_neg0p4_5p6", "Rule flag: -0.4 < logP < 5.6."),
        ("ghose_atom_count_in_20_70", "Rule flag: 20 < atom count < 70."),
        ("ghose_mr_in_40_130", "Rule flag: 40 < molar refractivity < 130."),
        ("ghose_pass", "Composite Ghose filter pass flag."),
        ("oprea_hbd_lt_2", "Rule flag: HBD < 2."),
        ("oprea_hba_in_2_10", "Rule flag: 2 <= HBA <= 10."),
        ("oprea_rotatable_bonds_in_2_8", "Rule flag: 2 < rotatable bonds < 8."),
        ("oprea_ring_count_in_1_4", "Rule flag: 1 <= ring count <= 4."),
        ("oprea_pass", "Composite Oprea lead-like pass flag."),
        ("mw_le_300", "Rule flag: MW <= 300."),
        ("logp_le_3", "Rule flag: logP <= 3."),
        ("hbd_le_3", "Rule flag: HBD <= 3."),
        ("hba_le_3", "Rule flag: HBA <= 3."),
        ("rule_of_3_pass", "Composite fragment Rule-of-3 pass flag."),
        ("muegge_mw_in_200_600", "Rule flag: 200 <= MW <= 600."),
        ("muegge_logp_in_neg2_5", "Rule flag: -2 <= logP <= 5."),
        ("tpsa_le_150", "Rule flag: TPSA <= 150."),
        ("rotatable_bonds_le_15", "Rule flag: rotatable bonds <= 15."),
        ("ring_count_le_7", "Rule flag: ring count <= 7."),
        ("muegge_pass", "Composite Muegge pass flag."),
        ("pfizer_3_75_risk", "Rule flag: logP > 3 and TPSA < 75."),
        ("pfizer_3_75_safe", "Rule flag: not(logP > 3 and TPSA < 75)."),
        ("aromatic_ring_count_le_3", "Heuristic flag: aromatic ring count <= 3."),
        ("fraction_csp3_ge_0p4", "Heuristic flag: fractionCSP3 >= 0.4."),
        ("fraction_csp3_distance_to_0p45", "Absolute distance from the oral-bioavailability saturation target 0.45."),
        ("stereocenter_count_ge_1", "Heuristic flag: at least one stereocenter is present."),
        (
            "crystal_packing_risk_proxy",
            "Empirical proxy for excessive planar aromatic surface using aromaticity, fused aromaticity, and low saturation.",
        ),
        ("oral_bioavailability_rule_pass_count", "Count of passed composite rules among Lipinski, Veber, Egan, Ghose, Oprea, Rule-of-3, Muegge, and Pfizer 3/75-safe."),
        (
            "oral_bioavailability_screen_pass",
            "Combined heuristic screen requiring Lipinski, Veber, Egan, Pfizer-safe, aromatic ring <= 3, and either stereochemistry or fractionCSP3 >= 0.4.",
        ),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "Measured oral bioavailability, permeability, exposure, and formulation effects are not hard-coded because they require experimental data.",
    "Exact crystal packing and pi-stacking behavior are represented only by aromaticity and saturation proxies, not by solid-state simulation.",
    "Intramolecular H-bonding is represented by a donor-acceptor topological proxy rather than conformer-dependent energetics.",
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


def _maybe_between(value: float | None, lower: float, upper: float, inclusive: bool = True) -> float:
    if _is_missing(value):
        return math.nan
    if inclusive:
        return _flag(lower <= value <= upper)
    return _flag(lower < value < upper)


@lru_cache(maxsize=1)
def _rdkit() -> SimpleNamespace:
    try:
        from rdkit import Chem, RDConfig
        from rdkit.Chem import ChemicalFeatures, Crippen, Descriptors, Lipinski, MolSurf, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run Bioavailability_Ma feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        RDConfig=RDConfig,
        ChemicalFeatures=ChemicalFeatures,
        Crippen=Crippen,
        Descriptors=Descriptors,
        Lipinski=Lipinski,
        MolSurf=MolSurf,
        rdMolDescriptors=rdMolDescriptors,
    )


@lru_cache(maxsize=1)
def _feature_factory():
    rdkit = _rdkit()
    feature_path = Path(rdkit.RDConfig.RDDataDir) / "BaseFeatures.fdef"
    return rdkit.ChemicalFeatures.BuildFeatureFactory(str(feature_path))


@lru_cache(maxsize=1)
def _get_pka_predictor():
    if str(INTERN_S1_ROOT) not in sys.path:
        sys.path.insert(0, str(INTERN_S1_ROOT))

    from tools.pka_related_tools import _get_pka_predictor as _load_predictor

    return _load_predictor()


def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    rdkit = _rdkit()
    mol = rdkit.Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return mol


def _count_total_atoms(mol) -> int:
    rdkit = _rdkit()
    return rdkit.Chem.AddHs(mol).GetNumAtoms()


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


def _count_aromatic_atoms(mol) -> int:
    return sum(atom.GetIsAromatic() for atom in mol.GetAtoms())


def _count_stereocenters(mol) -> int:
    chiral_centers = _rdkit().Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    return len(chiral_centers)


def _count_fused_aromatic_rings(mol) -> int:
    ring_info = mol.GetRingInfo()
    aromatic_rings = []
    for ring in ring_info.AtomRings():
        ring_atoms = set(ring)
        if all(mol.GetAtomWithIdx(atom_idx).GetIsAromatic() for atom_idx in ring_atoms):
            aromatic_rings.append(ring_atoms)

    if not aromatic_rings:
        return 0

    fused_ring_indices = set()
    for i, ring_i in enumerate(aromatic_rings):
        for j in range(i + 1, len(aromatic_rings)):
            if len(ring_i.intersection(aromatic_rings[j])) >= 2:
                fused_ring_indices.add(i)
                fused_ring_indices.add(j)
    return len(fused_ring_indices)


def _count_intramolecular_hbond_pairs(mol, min_distance: int = 4, max_distance: int = 8) -> int:
    try:
        factory = _feature_factory()
    except Exception:
        return 0

    donor_atoms: set[int] = set()
    acceptor_atoms: set[int] = set()
    for feature in factory.GetFeaturesForMol(mol):
        atom_ids = feature.GetAtomIds()
        if feature.GetFamily() == "Donor":
            donor_atoms.update(atom_ids)
        elif feature.GetFamily() == "Acceptor":
            acceptor_atoms.update(atom_ids)

    if not donor_atoms or not acceptor_atoms:
        return 0

    rdkit = _rdkit()
    unique_pairs: set[tuple[int, int]] = set()
    for donor_idx in donor_atoms:
        for acceptor_idx in acceptor_atoms:
            if donor_idx == acceptor_idx:
                continue
            path = rdkit.Chem.GetShortestPath(mol, donor_idx, acceptor_idx)
            bond_distance = len(path) - 1
            if min_distance <= bond_distance <= max_distance:
                unique_pairs.add(tuple(sorted((donor_idx, acceptor_idx))))
    return len(unique_pairs)


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
        "has_amphoteric_sites": math.nan,
        "ionizable_center_present": math.nan,
        "moderate_basic_pka_8_12": math.nan,
        "moderate_acidic_pka_2_6": math.nan,
        "any_moderate_pka_2_12": math.nan,
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
    estimated_logd = logp + math.log10(neutral_fraction)

    moderate_basic = bool(base_sites and 8.0 <= most_basic_pka <= 12.0)
    moderate_acidic = bool(acid_sites and 2.0 <= most_acidic_pka <= 6.0)

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
            "estimated_logd_ph74": estimated_logd,
            "has_amphoteric_sites": float(bool(base_sites and acid_sites)),
            "ionizable_center_present": float(bool(base_sites or acid_sites)),
            "moderate_basic_pka_8_12": float(moderate_basic),
            "moderate_acidic_pka_2_6": float(moderate_acidic),
            "any_moderate_pka_2_12": float(moderate_basic or moderate_acidic),
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
    total_atom_count = _as_float(_count_total_atoms(mol))
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    molar_refractivity = _as_float(rdkit.Crippen.MolMR(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    total_hbonds = hbd + hba
    heteroatom_count = _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol))
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.Lipinski.NumAliphaticRings(mol))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    stereocenter_count = _as_float(_count_stereocenters(mol))
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    abs_formal_charge = abs(formal_charge)
    labute_asa = _as_float(rdkit.MolSurf.LabuteASA(mol))
    labute_asa_per_heavy_atom = math.nan if heavy_atom_count == 0 else labute_asa / heavy_atom_count
    aromatic_heavy_atom_fraction = math.nan if heavy_atom_count == 0 else _count_aromatic_atoms(mol) / heavy_atom_count
    fused_aromatic_ring_count = _as_float(_count_fused_aromatic_rings(mol))
    intramolecular_hbond_pair_count = _as_float(_count_intramolecular_hbond_pairs(mol))

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
            "total_atom_count": total_atom_count,
            "logp": logp,
            "molar_refractivity": molar_refractivity,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbonds,
            "heteroatom_count": heteroatom_count,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "fraction_csp3": fraction_csp3,
            "stereocenter_count": stereocenter_count,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "abs_formal_charge": abs_formal_charge,
            "labute_asa": labute_asa,
            "labute_asa_per_heavy_atom": labute_asa_per_heavy_atom,
            "aromatic_heavy_atom_fraction": aromatic_heavy_atom_fraction,
            "fused_aromatic_ring_count": fused_aromatic_ring_count,
            "intramolecular_hbond_pair_count": intramolecular_hbond_pair_count,
            "intramolecular_hbond_potential_present": _flag(intramolecular_hbond_pair_count > 0.0),
        }
    )

    features.update(_compute_pka_summary(mol, logp))

    lipinski_pass = mol_weight <= 500.0 and logp <= 5.0 and hbd <= 5.0 and hba <= 10.0
    lipinski_violation_count = float(
        int(mol_weight > 500.0)
        + int(logp > 5.0)
        + int(hbd > 5.0)
        + int(hba > 10.0)
    )
    veber_pass = rotatable_bonds <= 10.0 and tpsa <= 140.0
    egan_pass = logp <= 5.88 and tpsa <= 131.6
    ghose_pass = 160.0 < mol_weight < 480.0 and -0.4 < logp < 5.6 and 20.0 < total_atom_count < 70.0 and 40.0 < molar_refractivity < 130.0
    oprea_pass = hbd < 2.0 and 2.0 <= hba <= 10.0 and 2.0 < rotatable_bonds < 8.0 and 1.0 <= ring_count <= 4.0
    rule_of_3_pass = mol_weight <= 300.0 and logp <= 3.0 and hbd <= 3.0 and hba <= 3.0
    muegge_pass = 200.0 <= mol_weight <= 600.0 and -2.0 <= logp <= 5.0 and tpsa <= 150.0 and hbd <= 5.0 and hba <= 10.0 and rotatable_bonds <= 15.0 and ring_count <= 7.0
    pfizer_3_75_risk = logp > 3.0 and tpsa < 75.0
    pfizer_3_75_safe = not pfizer_3_75_risk
    crystal_packing_risk_proxy = aromatic_ring_count > 3.0 or (
        not _is_missing(aromatic_heavy_atom_fraction)
        and aromatic_heavy_atom_fraction >= 0.5
        and fraction_csp3 < 0.25
        and fused_aromatic_ring_count >= 2.0
    )

    oral_bioavailability_rule_pass_count = float(
        sum(
            (
                lipinski_pass,
                veber_pass,
                egan_pass,
                ghose_pass,
                oprea_pass,
                rule_of_3_pass,
                muegge_pass,
                pfizer_3_75_safe,
            )
        )
    )
    oral_bioavailability_screen_pass = (
        lipinski_pass
        and veber_pass
        and egan_pass
        and pfizer_3_75_safe
        and aromatic_ring_count <= 3.0
        and (fraction_csp3 >= 0.4 or stereocenter_count >= 1.0)
    )

    features.update(
        {
            "mw_le_500": _maybe_le(mol_weight, 500.0),
            "logp_le_5": _maybe_le(logp, 5.0),
            "hbd_le_5": _maybe_le(hbd, 5.0),
            "hba_le_10": _maybe_le(hba, 10.0),
            "lipinski_pass": _flag(lipinski_pass),
            "lipinski_violation_count": lipinski_violation_count,
            "rotatable_bonds_le_10": _maybe_le(rotatable_bonds, 10.0),
            "tpsa_le_140": _maybe_le(tpsa, 140.0),
            "veber_pass": _flag(veber_pass),
            "logp_le_5p88": _maybe_le(logp, 5.88),
            "tpsa_le_131p6": _maybe_le(tpsa, 131.6),
            "egan_pass": _flag(egan_pass),
            "ghose_mw_in_160_480": _maybe_between(mol_weight, 160.0, 480.0, inclusive=False),
            "ghose_logp_in_neg0p4_5p6": _maybe_between(logp, -0.4, 5.6, inclusive=False),
            "ghose_atom_count_in_20_70": _maybe_between(total_atom_count, 20.0, 70.0, inclusive=False),
            "ghose_mr_in_40_130": _maybe_between(molar_refractivity, 40.0, 130.0, inclusive=False),
            "ghose_pass": _flag(ghose_pass),
            "oprea_hbd_lt_2": _flag(hbd < 2.0),
            "oprea_hba_in_2_10": _maybe_between(hba, 2.0, 10.0, inclusive=True),
            "oprea_rotatable_bonds_in_2_8": _maybe_between(rotatable_bonds, 2.0, 8.0, inclusive=False),
            "oprea_ring_count_in_1_4": _maybe_between(ring_count, 1.0, 4.0, inclusive=True),
            "oprea_pass": _flag(oprea_pass),
            "mw_le_300": _maybe_le(mol_weight, 300.0),
            "logp_le_3": _maybe_le(logp, 3.0),
            "hbd_le_3": _maybe_le(hbd, 3.0),
            "hba_le_3": _maybe_le(hba, 3.0),
            "rule_of_3_pass": _flag(rule_of_3_pass),
            "muegge_mw_in_200_600": _maybe_between(mol_weight, 200.0, 600.0, inclusive=True),
            "muegge_logp_in_neg2_5": _maybe_between(logp, -2.0, 5.0, inclusive=True),
            "tpsa_le_150": _maybe_le(tpsa, 150.0),
            "rotatable_bonds_le_15": _maybe_le(rotatable_bonds, 15.0),
            "ring_count_le_7": _maybe_le(ring_count, 7.0),
            "muegge_pass": _flag(muegge_pass),
            "pfizer_3_75_risk": _flag(pfizer_3_75_risk),
            "pfizer_3_75_safe": _flag(pfizer_3_75_safe),
            "aromatic_ring_count_le_3": _maybe_le(aromatic_ring_count, 3.0),
            "fraction_csp3_ge_0p4": _flag(fraction_csp3 >= 0.4),
            "fraction_csp3_distance_to_0p45": abs(fraction_csp3 - 0.45),
            "stereocenter_count_ge_1": _flag(stereocenter_count >= 1.0),
            "crystal_packing_risk_proxy": _flag(crystal_packing_risk_proxy),
            "oral_bioavailability_rule_pass_count": oral_bioavailability_rule_pass_count,
            "oral_bioavailability_screen_pass": _flag(oral_bioavailability_screen_pass),
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
        description="Generate Bioavailability_Ma DeepResearch features from SMILES."
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
