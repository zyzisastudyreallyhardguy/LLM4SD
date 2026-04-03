from __future__ import annotations

from codex_generated_code.Carcinogens_Lagunin import carcinogens_lagunin_rule_features


class CarcinogensLaguninBackend:
    @property
    def name(self) -> str:
        return "carcinogens_lagunin"

    def get_feature_names(self) -> list[str]:
        return carcinogens_lagunin_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return carcinogens_lagunin_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return carcinogens_lagunin_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return carcinogens_lagunin_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
