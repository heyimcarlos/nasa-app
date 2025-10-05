from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred) -> float:
    """Compute RMSE compatible with older scikit-learn (no squared= kw)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def get_feature_columns(climo_cols) -> list:
    """Return ordered feature column names shared by all models."""
    return ["lat", "lon", "doy_sin", "doy_cos"] + list(climo_cols)


def make_XY(df: pd.DataFrame, X_cols: list, y_cls: str = "rain_flag") -> Tuple[pd.DataFrame, pd.Series]:
    """Classification inputs: X and integer y."""
    return df[X_cols], df[y_cls].astype(int)


def make_XR(
    df: pd.DataFrame,
    X_cols: list,
    y_reg: str = "y_log1p",
    y_flag: str = "rain_flag",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Rain-only regression inputs on log1p scale."""
    mask_rain = df[y_flag] == 1
    return df.loc[mask_rain, X_cols], df.loc[mask_rain, y_reg]


def build_climo_baseline(train_df: pd.DataFrame, target_param: str) -> pd.DataFrame:
    """Build climatology baseline table keyed by (lat, lon, doy)."""
    return (
        train_df.groupby(["lat", "lon", "doy"])
        .agg(
            p_rain=("rain_flag", "mean"),
            med_mm=(target_param, lambda s: s[s > 0].median()),
        )
        .reset_index()
        .fillna(0.0)
    )


def baseline_expected_mm(df: pd.DataFrame, climo_baseline: pd.DataFrame) -> np.ndarray:
    """Compute baseline expected mm for df from a precomputed climatology table."""
    key = df[["lat", "lon", "doy"]]
    base = key.merge(climo_baseline, on=["lat", "lon", "doy"], how="left").fillna(0.0)
    return base["p_rain"].to_numpy() * base["med_mm"].to_numpy()


def combined_expected_mm(clf, reg, df: pd.DataFrame, X_cols: list) -> np.ndarray:
    """p(rain) * amount_if_rain using classifier proba and regressor on log1p scale."""
    p = clf.predict_proba(df[X_cols])[:, 1]
    amt = np.expm1(reg.predict(df[X_cols])).clip(min=0.0)
    return p * amt


