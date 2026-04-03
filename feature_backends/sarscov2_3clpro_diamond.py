from __future__ import annotations

from codex_generated_code.SARSCoV2_3CLPro_Diamond import sarscov2_3clpro_diamond_rule_features


class SARSCoV23CLProDiamondBackend:
    @property
    def name(self) -> str:
        return "sarscov2_3clpro_diamond"

    def get_feature_names(self) -> list[str]:
        return sarscov2_3clpro_diamond_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return sarscov2_3clpro_diamond_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return sarscov2_3clpro_diamond_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return sarscov2_3clpro_diamond_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
