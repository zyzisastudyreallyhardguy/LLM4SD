#!/usr/bin/env python3
"""Deterministic BBB Martins rule-to-feature code for downstream ML.

This module converts SMILES strings into a numeric feature vector grounded in the
computable parts of the DeepResearch BBB Martins ruleset. The output is designed
to plug directly into sklearn models such as RandomForest or LinearRegression.

Supported feature groups:
- Core physicochemical descriptors: MW, logP, TPSA, HBD/HBA, heteroatoms, rings.
- Topological descriptors: Balaban J, kappa, chi, BCUT, E-State, Wiener index.
- BBB rule-derived features: BOILED-Egg flag, CNS MPO approximation, threshold flags.
- pKa/logD-derived features using the local MolGpKa-based helper when available.

Intentionally skipped for now:
- Experimental or external-assay properties such as P-gp ER, BCRP, Kp,uu,brain.
- Descriptors that require a validated local implementation not present here,
  such as IAM/logkw, membrane-water interaction terms, and AlphaQ-style ESP terms.
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
        ("logp", "Wildman-Crippen logP."),
        ("molar_refractivity", "Wildman-Crippen molar refractivity."),
        ("tpsa", "Topological polar surface area."),
        ("hbd", "Hydrogen bond donor count."),
        ("hba", "Hydrogen bond acceptor count."),
        ("total_hbond_donors_acceptors", "HBD + HBA total."),
        ("heteroatom_count", "Total heteroatom count."),
        ("oxygen_nitrogen_count", "Count of O and N atoms."),
        ("rotatable_bonds", "Rotatable bond count."),
        ("ring_count", "Total ring count."),
        ("aromatic_ring_count", "Aromatic ring count."),
        ("aliphatic_ring_count", "Aliphatic ring count."),
        ("fused_ring_count", "Count of rings participating in fused ring systems."),
        ("fraction_csp3", "Fraction of sp3 carbons."),
        ("formal_charge", "Formal charge from the input graph."),
        ("positive_formal_atom_count", "Number of atoms with positive formal charge."),
        ("negative_formal_atom_count", "Number of atoms with negative formal charge."),
        ("labute_asa", "Labute approximate surface area."),
        ("balaban_j", "Balaban J topological index."),
        ("kappa1", "First kappa shape index."),
        ("kappa2", "Second kappa shape index."),
        ("chi0v", "Valence chi connectivity index order 0."),
        ("chi1v", "Valence chi connectivity index order 1."),
        ("chi2v", "Valence chi connectivity index order 2."),
        ("chi3v", "Valence chi connectivity index order 3."),
        ("chi4v", "Valence chi connectivity index order 4."),
        ("chi0n", "Simple chi connectivity index order 0."),
        ("chi1n", "Simple chi connectivity index order 1."),
        ("chi2n", "Simple chi connectivity index order 2."),
        ("chi3n", "Simple chi connectivity index order 3."),
        ("chi4n", "Simple chi connectivity index order 4."),
        ("max_estate_index", "Maximum E-State index."),
        ("min_estate_index", "Minimum E-State index."),
        ("max_abs_estate_index", "Maximum absolute E-State index."),
        ("min_abs_estate_index", "Minimum absolute E-State index."),
        ("bcut2d_mwhi", "BCUT2D MW high."),
        ("bcut2d_mwlow", "BCUT2D MW low."),
        ("bcut2d_chghi", "BCUT2D charge high."),
        ("bcut2d_chglo", "BCUT2D charge low."),
        ("bcut2d_logphi", "BCUT2D logP high."),
        ("bcut2d_logplow", "BCUT2D logP low."),
        ("bcut2d_mrhi", "BCUT2D MR high."),
        ("bcut2d_mrlow", "BCUT2D MR low."),
        ("wiener_index", "Wiener index from topological distances."),
        ("eccentric_connectivity_index", "Eccentric connectivity index."),
        ("most_basic_pka", "Most basic predicted pKa from MolGpKa helper."),
        ("most_acidic_pka", "Most acidic predicted pKa from MolGpKa helper."),
        ("num_basic_sites", "Number of predicted basic sites."),
        ("num_acidic_sites", "Number of predicted acidic sites."),
        ("neutral_fraction_ph74", "Estimated neutral fraction at pH 7.4."),
        ("charged_fraction_ph74", "1 - neutral_fraction_ph74."),
        ("base_protonated_fraction_ph74", "Estimated protonated fraction for the dominant basic site."),
        ("acid_deprotonated_fraction_ph74", "Estimated deprotonated fraction for the dominant acidic site."),
        ("net_charge_proxy_ph74", "Base protonation minus acid deprotonation proxy at pH 7.4."),
        ("estimated_logd_ph74", "Estimated logD at pH 7.4 from logP and neutral fraction."),
        ("has_amphoteric_sites", "1 if both acidic and basic sites are predicted."),
        ("mw_le_360", "Rule flag: MW <= 360."),
        ("mw_le_450", "Rule flag: MW <= 450."),
        ("mw_lt_400", "Rule flag: MW < 400."),
        ("mw_between_400_600", "Rule flag: 400 <= MW <= 600."),
        ("logp_le_3", "Rule flag: logP <= 3."),
        ("logp_lt_5", "Rule flag: logP < 5."),
        ("logp_in_1p5_2p7", "Rule flag: 1.5 <= logP <= 2.7."),
        ("logp_distance_to_3p4", "Absolute distance from the heuristic logP target 3.4."),
        ("tpsa_40_90", "Rule flag: 40 <= TPSA <= 90."),
        ("tpsa_lt_70", "Rule flag: TPSA < 70."),
        ("tpsa_le_90", "Rule flag: TPSA <= 90."),
        ("tpsa_lt_79", "Rule flag: TPSA < 79."),
        ("hbd_le_0p5", "Rule flag: HBD <= 0.5."),
        ("hbd_lt_3", "Rule flag: HBD < 3."),
        ("hba_lt_7", "Rule flag: HBA < 7."),
        ("total_hbond_lt_8", "Rule flag: HBD + HBA < 8."),
        ("oxygen_nitrogen_le_5", "Rule flag: O + N <= 5."),
        ("rotatable_bonds_lt_8", "Rule flag: rotatable bonds < 8."),
        ("rotatable_bonds_le_5", "Rule flag: rotatable bonds <= 5."),
        ("most_basic_pka_le_8", "Rule flag: most basic pKa <= 8."),
        ("most_basic_pka_le_10", "Rule flag: most basic pKa <= 10."),
        ("most_basic_pka_7p5_10p5", "Rule flag: 7.5 <= most basic pKa <= 10.5."),
        ("most_acidic_pka_ge_4", "Rule flag: most acidic pKa >= 4."),
        ("estimated_logd_0_3", "Rule flag: 0 < estimated logD(7.4) < 3."),
        ("estimated_logd_le_3", "Rule flag: estimated logD(7.4) <= 3."),
        ("boiled_egg_bbb_likely", "Rule flag: BOILED-Egg BBB likely region."),
        ("formal_charge_abs_le_1", "Rule flag: abs(formal charge) <= 1."),
        ("formal_charge_zero_or_plus_one", "Rule flag: formal charge in {0, +1}."),
        ("cns_mpo_score", "Approximate six-parameter CNS MPO score."),
        ("cns_mpo_ge_4", "Rule flag: CNS MPO score >= 4."),
        ("passive_bbb_screen_pass", "Combined passive BBB screen using MW/TPSA/logD/formal charge."),
        ("pka_features_available", "1 if MolGpKa-derived pKa/logD features were computed."),
    ]


FEATURE_DESCRIPTIONS = OrderedDict(_feature_specs())
SKIPPED_RULE_GROUPS = [
    "P-gp and BCRP transport features requiring a dedicated transporter model or assay data.",
    "Experimental permeability and exposure endpoints such as Papp, Kp,uu,brain, fu,p, and fu,brain.",
    "3D ESP, AlphaQ, amphiphilic-axis CSA, and membrane-water interaction terms without a validated local calculator.",
    "Chromatographic descriptors such as IAM/logkw and fingerprint-to-BBB+ reference similarity without a curated reference set.",
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
        from rdkit.Chem import Crippen, Descriptors, GraphDescriptors, Lipinski, MolSurf, rdMolDescriptors
    except ImportError as exc:
        raise ImportError(
            "RDKit is required to run BBB Martins feature generation. "
            "Please use the project environment that provides rdkit."
        ) from exc

    return SimpleNamespace(
        Chem=Chem,
        Crippen=Crippen,
        Descriptors=Descriptors,
        GraphDescriptors=GraphDescriptors,
        Lipinski=Lipinski,
        MolSurf=MolSurf,
        rdMolDescriptors=rdMolDescriptors,
    )


def _mol_from_smiles(smiles: str):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("smiles must be a non-empty string")

    rdkit = _rdkit()
    mol = rdkit.Chem.MolFromSmiles(smiles)
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


def _count_fused_rings(mol) -> int:
    ring_info = mol.GetRingInfo()
    atom_rings = [set(ring) for ring in ring_info.AtomRings()]
    if not atom_rings:
        return 0

    fused_ring_indices = set()
    for i, ring_i in enumerate(atom_rings):
        for j in range(i + 1, len(atom_rings)):
            if len(ring_i.intersection(atom_rings[j])) >= 2:
                fused_ring_indices.add(i)
                fused_ring_indices.add(j)
    return len(fused_ring_indices)


def _wiener_index(mol) -> float:
    rdkit = _rdkit()
    distance_matrix = rdkit.Chem.GetDistanceMatrix(mol)
    total = 0.0
    num_atoms = len(distance_matrix)
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            total += float(distance_matrix[i][j])
    return total


def _eccentric_connectivity_index(mol) -> float:
    rdkit = _rdkit()
    distance_matrix = rdkit.Chem.GetDistanceMatrix(mol)
    total = 0.0
    for atom_index, atom in enumerate(mol.GetAtoms()):
        eccentricity = max(float(distance) for distance in distance_matrix[atom_index])
        total += float(atom.GetDegree()) * eccentricity
    return total


def _piecewise_decreasing(value: float, low_good: float, high_bad: float) -> float:
    if value <= low_good:
        return 1.0
    if value >= high_bad:
        return 0.0
    return (high_bad - value) / (high_bad - low_good)


def _piecewise_hump(
    value: float,
    low_zero: float,
    low_one: float,
    high_one: float,
    high_zero: float,
) -> float:
    if value <= low_zero or value >= high_zero:
        return 0.0
    if low_one <= value <= high_one:
        return 1.0
    if value < low_one:
        return (value - low_zero) / (low_one - low_zero)
    return (high_zero - value) / (high_zero - high_one)


@lru_cache(maxsize=1)
def _get_pka_predictor():
    if str(INTERN_S1_ROOT) not in sys.path:
        sys.path.insert(0, str(INTERN_S1_ROOT))

    from tools.pka_related_tools import _get_pka_predictor as _load_predictor

    return _load_predictor()


def _compute_pka_summary(smiles: str, mol) -> dict[str, float]:
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
            "net_charge_proxy_ph74": base_protonated_fraction - acid_deprotonated_fraction,
            "estimated_logd_ph74": estimated_logd,
            "has_amphoteric_sites": float(bool(base_sites and acid_sites)),
            "pka_features_available": 1.0,
        }
    )
    return result


def _compute_cns_mpo_score(features: dict[str, float]) -> float:
    logp = features["logp"]
    logd = features["estimated_logd_ph74"]
    mw = features["mol_weight"]
    tpsa = features["tpsa"]
    hbd = features["hbd"]
    most_basic_pka = features["most_basic_pka"]
    num_basic_sites = features["num_basic_sites"]

    if any(_is_missing(value) for value in (logp, logd, mw, tpsa, hbd)):
        return math.nan

    if _is_missing(most_basic_pka) and not _is_missing(num_basic_sites) and num_basic_sites == 0:
        pka_component = 1.0
    elif _is_missing(most_basic_pka):
        return math.nan
    else:
        pka_component = _piecewise_decreasing(most_basic_pka, 8.0, 10.0)

    components = [
        _piecewise_decreasing(logp, 3.0, 5.0),
        _piecewise_decreasing(logd, 2.0, 4.0),
        _piecewise_decreasing(mw, 360.0, 500.0),
        _piecewise_hump(tpsa, 20.0, 40.0, 90.0, 120.0),
        _piecewise_decreasing(hbd, 0.5, 3.5),
        pka_component,
    ]
    return float(sum(components))


def featurize_smiles(smiles: str) -> dict[str, float]:
    rdkit = _rdkit()
    mol = _mol_from_smiles(smiles)
    features = _empty_feature_template()

    mol_weight = _as_float(rdkit.Descriptors.MolWt(mol))
    exact_mol_weight = _as_float(rdkit.Descriptors.ExactMolWt(mol))
    heavy_atom_count = _as_float(mol.GetNumHeavyAtoms())
    logp = _as_float(rdkit.Crippen.MolLogP(mol))
    molar_refractivity = _as_float(rdkit.Crippen.MolMR(mol))
    tpsa = _as_float(rdkit.rdMolDescriptors.CalcTPSA(mol))
    hbd = _as_float(rdkit.Lipinski.NumHDonors(mol))
    hba = _as_float(rdkit.Lipinski.NumHAcceptors(mol))
    heteroatom_count = _as_float(rdkit.rdMolDescriptors.CalcNumHeteroatoms(mol))
    oxygen_nitrogen_count = _as_float(_count_oxygen_and_nitrogen(mol))
    rotatable_bonds = _as_float(rdkit.Lipinski.NumRotatableBonds(mol))
    ring_count = _as_float(rdkit.Lipinski.RingCount(mol))
    aromatic_ring_count = _as_float(rdkit.Lipinski.NumAromaticRings(mol))
    aliphatic_ring_count = _as_float(rdkit.Lipinski.NumAliphaticRings(mol))
    fused_ring_count = _as_float(_count_fused_rings(mol))
    fraction_csp3 = _as_float(rdkit.Lipinski.FractionCSP3(mol))
    formal_charge = _as_float(rdkit.Chem.GetFormalCharge(mol))
    positive_formal_atom_count, negative_formal_atom_count = _count_formal_charge_atoms(mol)
    labute_asa = _as_float(rdkit.MolSurf.LabuteASA(mol))
    total_hbonds = hbd + hba

    features.update(
        {
            "mol_weight": mol_weight,
            "exact_mol_weight": exact_mol_weight,
            "heavy_atom_count": heavy_atom_count,
            "logp": logp,
            "molar_refractivity": molar_refractivity,
            "tpsa": tpsa,
            "hbd": hbd,
            "hba": hba,
            "total_hbond_donors_acceptors": total_hbonds,
            "heteroatom_count": heteroatom_count,
            "oxygen_nitrogen_count": oxygen_nitrogen_count,
            "rotatable_bonds": rotatable_bonds,
            "ring_count": ring_count,
            "aromatic_ring_count": aromatic_ring_count,
            "aliphatic_ring_count": aliphatic_ring_count,
            "fused_ring_count": fused_ring_count,
            "fraction_csp3": fraction_csp3,
            "formal_charge": formal_charge,
            "positive_formal_atom_count": _as_float(positive_formal_atom_count),
            "negative_formal_atom_count": _as_float(negative_formal_atom_count),
            "labute_asa": labute_asa,
            "balaban_j": _as_float(rdkit.GraphDescriptors.BalabanJ(mol)),
            "kappa1": _as_float(rdkit.GraphDescriptors.Kappa1(mol)),
            "kappa2": _as_float(rdkit.GraphDescriptors.Kappa2(mol)),
            "chi0v": _as_float(rdkit.GraphDescriptors.Chi0v(mol)),
            "chi1v": _as_float(rdkit.GraphDescriptors.Chi1v(mol)),
            "chi2v": _as_float(rdkit.GraphDescriptors.Chi2v(mol)),
            "chi3v": _as_float(rdkit.GraphDescriptors.Chi3v(mol)),
            "chi4v": _as_float(rdkit.GraphDescriptors.Chi4v(mol)),
            "chi0n": _as_float(rdkit.GraphDescriptors.Chi0n(mol)),
            "chi1n": _as_float(rdkit.GraphDescriptors.Chi1n(mol)),
            "chi2n": _as_float(rdkit.GraphDescriptors.Chi2n(mol)),
            "chi3n": _as_float(rdkit.GraphDescriptors.Chi3n(mol)),
            "chi4n": _as_float(rdkit.GraphDescriptors.Chi4n(mol)),
            "max_estate_index": _as_float(rdkit.Descriptors.MaxEStateIndex(mol)),
            "min_estate_index": _as_float(rdkit.Descriptors.MinEStateIndex(mol)),
            "max_abs_estate_index": _as_float(rdkit.Descriptors.MaxAbsEStateIndex(mol)),
            "min_abs_estate_index": _as_float(rdkit.Descriptors.MinAbsEStateIndex(mol)),
            "bcut2d_mwhi": _as_float(rdkit.Descriptors.BCUT2D_MWHI(mol)),
            "bcut2d_mwlow": _as_float(rdkit.Descriptors.BCUT2D_MWLOW(mol)),
            "bcut2d_chghi": _as_float(rdkit.Descriptors.BCUT2D_CHGHI(mol)),
            "bcut2d_chglo": _as_float(rdkit.Descriptors.BCUT2D_CHGLO(mol)),
            "bcut2d_logphi": _as_float(rdkit.Descriptors.BCUT2D_LOGPHI(mol)),
            "bcut2d_logplow": _as_float(rdkit.Descriptors.BCUT2D_LOGPLOW(mol)),
            "bcut2d_mrhi": _as_float(rdkit.Descriptors.BCUT2D_MRHI(mol)),
            "bcut2d_mrlow": _as_float(rdkit.Descriptors.BCUT2D_MRLOW(mol)),
            "wiener_index": _wiener_index(mol),
            "eccentric_connectivity_index": _eccentric_connectivity_index(mol),
        }
    )

    features.update(_compute_pka_summary(smiles, mol))

    cns_mpo_score = _compute_cns_mpo_score(features)
    passive_bbb_screen_pass = None
    if not any(
        _is_missing(features[key])
        for key in ("mol_weight", "tpsa", "estimated_logd_ph74", "formal_charge")
    ):
        passive_bbb_screen_pass = (
            features["mol_weight"] <= 450.0
            and features["tpsa"] <= 90.0
            and features["estimated_logd_ph74"] <= 3.0
            and features["formal_charge"] in (0.0, 1.0)
        )

    features.update(
        {
            "mw_le_360": _maybe_le(mol_weight, 360.0),
            "mw_le_450": _maybe_le(mol_weight, 450.0),
            "mw_lt_400": _maybe_lt(mol_weight, 400.0),
            "mw_between_400_600": _maybe_between(mol_weight, 400.0, 600.0, inclusive=True),
            "logp_le_3": _maybe_le(logp, 3.0),
            "logp_lt_5": _maybe_lt(logp, 5.0),
            "logp_in_1p5_2p7": _maybe_between(logp, 1.5, 2.7, inclusive=True),
            "logp_distance_to_3p4": abs(logp - 3.4),
            "tpsa_40_90": _maybe_between(tpsa, 40.0, 90.0, inclusive=True),
            "tpsa_lt_70": _maybe_lt(tpsa, 70.0),
            "tpsa_le_90": _maybe_le(tpsa, 90.0),
            "tpsa_lt_79": _maybe_lt(tpsa, 79.0),
            "hbd_le_0p5": _maybe_le(hbd, 0.5),
            "hbd_lt_3": _maybe_lt(hbd, 3.0),
            "hba_lt_7": _maybe_lt(hba, 7.0),
            "total_hbond_lt_8": _maybe_lt(total_hbonds, 8.0),
            "oxygen_nitrogen_le_5": _maybe_le(oxygen_nitrogen_count, 5.0),
            "rotatable_bonds_lt_8": _maybe_lt(rotatable_bonds, 8.0),
            "rotatable_bonds_le_5": _maybe_le(rotatable_bonds, 5.0),
            "most_basic_pka_le_8": _maybe_le(features["most_basic_pka"], 8.0),
            "most_basic_pka_le_10": _maybe_le(features["most_basic_pka"], 10.0),
            "most_basic_pka_7p5_10p5": _maybe_between(features["most_basic_pka"], 7.5, 10.5, inclusive=True),
            "most_acidic_pka_ge_4": _maybe_ge(features["most_acidic_pka"], 4.0),
            "estimated_logd_0_3": _maybe_between(features["estimated_logd_ph74"], 0.0, 3.0, inclusive=False),
            "estimated_logd_le_3": _maybe_le(features["estimated_logd_ph74"], 3.0),
            "boiled_egg_bbb_likely": _flag(0.4 <= logp <= 6.0 and tpsa < 79.0),
            "formal_charge_abs_le_1": _flag(abs(formal_charge) <= 1.0),
            "formal_charge_zero_or_plus_one": _flag(formal_charge in (0.0, 1.0)),
            "cns_mpo_score": cns_mpo_score,
            "cns_mpo_ge_4": _maybe_ge(cns_mpo_score, 4.0),
            "passive_bbb_screen_pass": _flag(passive_bbb_screen_pass),
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
        description="Generate BBB Martins DeepResearch features from SMILES."
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
