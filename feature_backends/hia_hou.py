from __future__ import annotations

from codex_generated_code.HIA_Hou import hia_hou_rule_features


class HIAHouBackend:
    @property
    def name(self) -> str:
        return "hia_hou"

    def get_feature_names(self) -> list[str]:
        return hia_hou_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return hia_hou_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return hia_hou_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return hia_hou_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
