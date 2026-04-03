from __future__ import annotations

from codex_generated_code.BBB_Martins import bbb_martins_rule_features


class BBBMartinsBackend:
    @property
    def name(self) -> str:
        return "bbb_martins"

    def get_feature_names(self) -> list[str]:
        return bbb_martins_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return bbb_martins_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return bbb_martins_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return bbb_martins_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
