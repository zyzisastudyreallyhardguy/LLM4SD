from __future__ import annotations


class ModuleBackedFeatureBackend:
    def __init__(self, *, name: str, module):
        self._name = name
        self._module = module

    @property
    def name(self) -> str:
        return self._name

    def get_feature_names(self) -> list[str]:
        return list(self._module.get_feature_names())

    def get_feature_descriptions(self) -> dict[str, str]:
        if hasattr(self._module, "get_feature_descriptions"):
            return dict(self._module.get_feature_descriptions())
        return {name: name for name in self.get_feature_names()}

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        return self._module.featurize_smiles(smiles)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise"):
        return self._module.featurize_smiles_list(smiles_list, on_error=on_error)
