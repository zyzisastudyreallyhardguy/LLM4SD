from __future__ import annotations

from codex_generated_code.AMES import ames_rule_features


class AMESBackend:
    @property
    def name(self) -> str:
        return "ames"

    def get_feature_names(self) -> list[str]:
        return ames_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return ames_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return ames_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return ames_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
