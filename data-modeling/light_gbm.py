# lgbm_model.py
# -----------------------------------------------------------------------------
# A 2-stage model using LightGBM.
# This version uses a simplified feature set identified by Boruta.
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from power_data import TARGET_PARAM, get_dataset, split_by_years
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    """Compute RMSE."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    # === Load the dataset ===
    bundle = get_dataset(years=5)
    data = bundle.data

    # === Split data by year ===
    train_df, val_df, test_df = split_by_years(
        data, train=0.6, val=0.2, test=0.2, chronological=True
    )

    # === Define feature set using ONLY Boruta's recommendations ===
    # This tests if a simpler model with high-impact features is more effective.
    X_cols = [
        "lat",
        "lon",
        "doy_sin",
        "doy_cos",
        "RH2M_clim",
        "T2M_MAX_clim",
        "SLP_clim",
    ]

    y_cls = "rain_flag"
    y_reg = "y_log1p"

    print(
        f"Training with a simplified feature set of {len(X_cols)} features selected by Boruta."
    )

    # --- Prepare data splits ---
    X_train, y_train = train_df[X_cols], train_df[y_cls]
    X_val, y_val = val_df[X_cols], val_df[y_cls]
    X_test, y_test = test_df[X_cols], test_df[y_cls]

    rain_train_df = train_df[train_df[y_cls] == 1]
    X_train_r, y_train_r = rain_train_df[X_cols], rain_train_df[y_reg]

    # === Stage 1 — LightGBM Classifier ===
    print("\n--- Training LGBM Classifier ---")
    clf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", lgb.LGBMClassifier(objective="binary", random_state=42)),
        ]
    ).fit(X_train, y_train)

    p_val = clf.predict_proba(X_val)[:, 1]
    p_test = clf.predict_proba(X_test)[:, 1]
    print(
        f"Classifier — Val AUC: {roc_auc_score(y_val, p_val):.4f}, Brier: {brier_score_loss(y_val, p_val):.4f}"
    )
    print(
        f"Classifier — Test AUC: {roc_auc_score(y_test, p_test):.4f}, Brier: {brier_score_loss(y_test, p_test):.4f}"
    )

    # === Stage 2 — LightGBM Regressor on rainy days ===
    print("\n--- Training LGBM Regressor ---")
    reg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "lgbm",
                lgb.LGBMRegressor(objective="regression_l1", random_state=42),
            ),  # L1 = MAE
        ]
    ).fit(X_train_r, y_train_r)

    # === Evaluate combined model on test set ===
    y_true = test_df[TARGET_PARAM].to_numpy()

    p_rain = clf.predict_proba(test_df[X_cols])[:, 1]
    amount_log = reg.predict(test_df[X_cols])
    y_model = p_rain * np.expm1(amount_log).clip(min=0.0)

    print("\n== Combined Model Performance (Test Set) ==")
    print(
        f"Model (2-stage LGBM) — MAE={mean_absolute_error(y_true, y_model):.3f} mm, RMSE={rmse(y_true, y_model):.3f} mm"
    )

    # === Save model bundle for serving ===
    bundle_to_save = {"clf": clf, "reg": reg, "X_cols": X_cols, "bundle_info": bundle}
    out_path = ARTIFACTS_DIR / "rain_model_lgbm_boruta_v1.joblib"
    joblib.dump(bundle_to_save, out_path)
    print(f"\nSaved model bundle -> {out_path.resolve()}")


if __name__ == "__main__":
    main()
