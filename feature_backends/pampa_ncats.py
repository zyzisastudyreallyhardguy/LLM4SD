from __future__ import annotations

from codex_generated_code.PAMPA_NCATS import pampa_ncats_rule_features


class PAMPANCATSBackend:
    @property
    def name(self) -> str:
        return "pampa_ncats"

    def get_feature_names(self) -> list[str]:
        return pampa_ncats_rule_features.get_feature_names()

    def get_feature_descriptions(self) -> dict[str, str]:
        return pampa_ncats_rule_features.get_feature_descriptions()

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return pampa_ncats_rule_features.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return pampa_ncats_rule_features.featurize_smiles_list(smiles_list, on_error=on_error)
