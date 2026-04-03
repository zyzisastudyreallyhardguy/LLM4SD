from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def _coerce_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = frame.copy()
    for column in numeric_frame.columns:
        numeric_frame[column] = pd.to_numeric(numeric_frame[column], errors="coerce")
    return numeric_frame


def fit_feature_preprocessor(
    train_df: pd.DataFrame,
    *,
    scale_features: bool = False,
):
    train_df = _coerce_numeric_frame(train_df)
    input_columns = train_df.columns.tolist()
    surviving_columns = train_df.columns[~train_df.isna().any(axis=0)].tolist()
    if not surviving_columns:
        raise ValueError("No usable feature columns remain after dropping NaN columns from training data.")

    train_df = train_df[surviving_columns]

    imputer = SimpleImputer(strategy="median")
    imputer.fit(train_df)

    scaler = None
    if scale_features:
        scaler = StandardScaler()
        scaler.fit(imputer.transform(train_df))

    return {
        "input_columns": input_columns,
        "surviving_columns": surviving_columns,
        "imputer": imputer,
        "scaler": scaler,
        "scale_features": scale_features,
    }


def transform_feature_frame(
    frame: pd.DataFrame,
    preprocessor: dict[str, object],
):
    numeric_frame = _coerce_numeric_frame(frame)
    surviving_columns = list(preprocessor["surviving_columns"])
    aligned_frame = numeric_frame.reindex(columns=surviving_columns)
    transformed = preprocessor["imputer"].transform(aligned_frame)

    scaler = preprocessor.get("scaler")
    if scaler is not None:
        transformed = scaler.transform(transformed)

    transformed_df = pd.DataFrame(
        transformed,
        columns=surviving_columns,
        index=frame.index,
    )
    return aligned_frame, transformed_df


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    scale_features: bool = False,
    return_preprocessor: bool = False,
):
    train_numeric_df = _coerce_numeric_frame(train_df)
    preprocessor = fit_feature_preprocessor(
        train_numeric_df,
        scale_features=scale_features,
    )
    surviving_columns = list(preprocessor["surviving_columns"])

    _, X_train_df = transform_feature_frame(train_df, preprocessor)
    _, X_valid_df = transform_feature_frame(valid_df, preprocessor)
    _, X_test_df = transform_feature_frame(test_df, preprocessor)

    result = (
        X_train_df.to_numpy(),
        X_valid_df.to_numpy(),
        X_test_df.to_numpy(),
        surviving_columns,
    )
    if return_preprocessor:
        return result + (preprocessor,)
    return result
