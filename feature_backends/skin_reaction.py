from __future__ import annotations

from codex_generated_code.Skin_Reaction import skin_reaction_rule_features


class SkinReactionBackend:
    @property
    def name(self) -> str:
        return "skin_reaction"

    def get_feature_names(self) -> list[str]:
        return skin_reaction_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return skin_reaction_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return skin_reaction_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return skin_reaction_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
