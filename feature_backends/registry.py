from __future__ import annotations

import importlib

from feature_feedback import is_feedback_backend_name


def load_feature_backend(name: str, *, function_names: list[str] | None = None, namespace: dict | None = None):
    if name == "generated_rules":
        from .generated_rules import GeneratedRulesBackend

        if function_names is None or namespace is None:
            raise ValueError("generated_rules backend requires function_names and namespace")
        return GeneratedRulesBackend(function_names=function_names, namespace=namespace)
    if is_feedback_backend_name(name):
        from .module_backed import ModuleBackedFeatureBackend

        module = importlib.import_module(f"codex_generated_code_variants.{name}")
        return ModuleBackedFeatureBackend(name=name, module=module)
    if name == "bbb_martins":
        from .bbb_martins import BBBMartinsBackend

        return BBBMartinsBackend()
    if name == "dili":
        from .dili import DILIBackend

        return DILIBackend()
    if name == "clintox":
        from .clintox import ClinToxBackend

        return ClinToxBackend()
    if name == "herg":
        from .herg import HERGBackend

        return HERGBackend()
    if name == "pampa_ncats":
        from .pampa_ncats import PAMPANCATSBackend

        return PAMPANCATSBackend()
    if name == "skin_reaction":
        from .skin_reaction import SkinReactionBackend

        return SkinReactionBackend()
    if name == "pgp_broccatelli":
        from .pgp_broccatelli import PgpBroccatelliBackend

        return PgpBroccatelliBackend()
    if name == "carcinogens_lagunin":
        from .carcinogens_lagunin import CarcinogensLaguninBackend

        return CarcinogensLaguninBackend()
    if name == "ames":
        from .ames import AMESBackend

        return AMESBackend()
    if name == "bioavailability_ma":
        from .bioavailability_ma import BioavailabilityMABackend

        return BioavailabilityMABackend()
    if name == "hia_hou":
        from .hia_hou import HIAHouBackend

        return HIAHouBackend()
    if name == "cyp2c9_substrate_carbonmangels":
        from .cyp2c9_substrate_carbonmangels import CYP2C9SubstrateCarbonMangelsBackend

        return CYP2C9SubstrateCarbonMangelsBackend()
    if name == "cyp2d6_substrate_carbonmangels":
        from .cyp2d6_substrate_carbonmangels import CYP2D6SubstrateCarbonMangelsBackend

        return CYP2D6SubstrateCarbonMangelsBackend()
    if name == "cyp3a4_substrate_carbonmangels":
        from .cyp3a4_substrate_carbonmangels import CYP3A4SubstrateCarbonMangelsBackend

        return CYP3A4SubstrateCarbonMangelsBackend()
    if name == "sarscov2_3clpro_diamond":
        from .sarscov2_3clpro_diamond import SARSCoV23CLProDiamondBackend

        return SARSCoV23CLProDiamondBackend()
    if name == "sarscov2_vitro_touret":
        from .sarscov2_vitro_touret import SARSCoV2VitroTouretBackend

        return SARSCoV2VitroTouretBackend()
    raise ValueError(f"Unknown feature backend: {name}")
