from __future__ import annotations


def featurize_smiles(*args, **kwargs):
    from .dili_rule_features import featurize_smiles as _featurize_smiles

    return _featurize_smiles(*args, **kwargs)


def featurize_smiles_list(*args, **kwargs):
    from .dili_rule_features import featurize_smiles_list as _featurize_smiles_list

    return _featurize_smiles_list(*args, **kwargs)


def get_feature_descriptions(*args, **kwargs):
    from .dili_rule_features import get_feature_descriptions as _get_feature_descriptions

    return _get_feature_descriptions(*args, **kwargs)


def get_feature_names(*args, **kwargs):
    from .dili_rule_features import get_feature_names as _get_feature_names

    return _get_feature_names(*args, **kwargs)


def get_skipped_rule_groups(*args, **kwargs):
    from .dili_rule_features import get_skipped_rule_groups as _get_skipped_rule_groups

    return _get_skipped_rule_groups(*args, **kwargs)


__all__ = [
    "featurize_smiles",
    "featurize_smiles_list",
    "get_feature_descriptions",
    "get_feature_names",
    "get_skipped_rule_groups",
]
