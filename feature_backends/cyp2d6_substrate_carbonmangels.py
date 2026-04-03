from __future__ import annotations

from codex_generated_code.CYP2D6_Substrate_CarbonMangels import (
    cyp2d6_substrate_carbonmangels_rule_features,
)


class CYP2D6SubstrateCarbonMangelsBackend:
    @property
    def name(self) -> str:
        return "cyp2d6_substrate_carbonmangels"

    def get_feature_names(self) -> list[str]:
        return cyp2d6_substrate_carbonmangels_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return cyp2d6_substrate_carbonmangels_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return cyp2d6_substrate_carbonmangels_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return cyp2d6_substrate_carbonmangels_rule_features.featurize_smiles_list(
            smiles_list,
            on_error=on_error,
        )
