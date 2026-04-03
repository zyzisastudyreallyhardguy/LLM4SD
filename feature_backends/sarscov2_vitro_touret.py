from __future__ import annotations

from codex_generated_code.SARSCoV2_Vitro_Touret import sarscov2_vitro_touret_rule_features


class SARSCoV2VitroTouretBackend:
    @property
    def name(self) -> str:
        return "sarscov2_vitro_touret"

    def get_feature_names(self) -> list[str]:
        return sarscov2_vitro_touret_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return sarscov2_vitro_touret_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return sarscov2_vitro_touret_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return sarscov2_vitro_touret_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
 