from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd
from rdkit import Chem


class GeneratedRulesBackend:
    def __init__(self, function_names: list[str], namespace: dict[str, Callable]):
        self._function_names = list(function_names)
        self._namespace = namespace

    @property
    def name(self) -> str:
        return "generated_rules"

    def get_feature_names(self) -> list[str]:
        return list(self._function_names)

    def featurize_smiles(self, smiles: str) -> dict[str, float]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles!r}")

        row: OrderedDict[str, float] = OrderedDict()
        for function_name in self._function_names:
            try:
                feature = self._namespace[function_name](mol)
                if feature is not None and isinstance(feature, (int, float)):
                    row[function_name] = float(feature)
                else:
                    row[function_name] = np.nan
            except Exception as exc:
                print(f"Unexpected error in function {function_name}: {str(exc)}")
                row[function_name] = np.nan
        return dict(row)

    def featurize_smiles_list(self, smiles_list: list[str], on_error: str = "raise") -> pd.DataFrame:
        rows = []
        for smiles in smiles_list:
            try:
                rows.append(self.featurize_smiles(smiles))
            except Exception:
                if on_error == "raise":
                    raise
                if on_error != "nan":
                    raise ValueError("on_error must be either 'raise' or 'nan'")
                rows.append(OrderedDict((name, np.nan) for name in self._function_names))
        return pd.DataFrame(rows, columns=self._function_names)
