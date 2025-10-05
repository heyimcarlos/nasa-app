

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boruta import BorutaPy
import shap
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

from power_data import (
    TARGET_PARAM,
    EXTRA_PARAMS,
    BBOX_TORONTO,
    get_dataset,
    split_by_years,
)

BASE_DIR = Path(__file__).resolve().parent              # → nasa-app/data-modeling
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # === Load the shared dataset (one-liner) ===
    bundle = get_dataset(years=5, bbox=BBOX_TORONTO, target_param=TARGET_PARAM, extra_params=EXTRA_PARAMS)

    data = bundle.data
    climo_cols = bundle.climo_cols

    # === Split: proportions by years (chronological) ===
    train_df, val_df, test_df = split_by_years(data, train=0.6, val=0.2, test=0.2, chronological=True)

    # === Feature set ===
    X_cols = ["lat", "lon", "doy_sin", "doy_cos"] + climo_cols
    y_cls = "rain_flag"
    y_reg = "y_log1p"

    def XY(df):
        return df[X_cols], df[y_cls].astype(int)

    def XR(df):
        m = df[y_cls] == 1
        return df.loc[m, X_cols], df.loc[m, y_reg]

    Xtr, ytr = XY(train_df)
    Xva, yva = XY(val_df)
    Xte, yte = XY(test_df)

    Xtr_r, ytr_r = XR(train_df)
    Xva_r, yva_r = XR(val_df)
    Xte_r, yte_r = XR(test_df)

    print(f"Features: {len(X_cols)}  | Train rows: {len(train_df):,}  | Rain-only rows: {len(Xtr_r):,}")

    # === Impute & scale common ===
    imp_cls = SimpleImputer(strategy="median")
    scaler_cls = StandardScaler()

    Xtr_imp = imp_cls.fit_transform(Xtr)
    Xva_imp = imp_cls.transform(Xva)
    Xte_imp = imp_cls.transform(Xte)

    Xtr_scaled = scaler_cls.fit_transform(Xtr_imp)
    Xva_scaled = scaler_cls.transform(Xva_imp)
    Xte_scaled = scaler_cls.transform(Xte_imp)


    # -------------------------------------------------------------------------
    # 1️⃣ BORUTA  —  CLASSIFICATION (rain/no‑rain)
    # -------------------------------------------------------------------------
    print("\n=== BORUTA: Classification (rain / no‑rain) ===")
    rf_boruta_cls = RandomForestClassifier(
        n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"
    )
    boruta_cls = BorutaPy(
        rf_boruta_cls, n_estimators="auto", random_state=42, verbose=2, perc=90
    )
    boruta_cls.fit(Xtr_scaled, ytr)

    selected_cls_mask = boruta_cls.support_
    boruta_cls_features = list(np.array(X_cols)[selected_cls_mask])
    print("\n✅ Boruta CLASSIFICATION selected features:")
    print(boruta_cls_features)


    


    # -------------------------------------------------------------------------
    # 3️⃣ BORUTA  —  REGRESSION (rainfall amount, rain‑only subset)
    # -------------------------------------------------------------------------
    print("\n=== BORUTA: Regression (rainfall amount) ===")
    imp_reg = SimpleImputer(strategy="median")
    Xtr_r_imp = imp_reg.fit_transform(Xtr_r)

    rf_boruta_reg = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    boruta_reg = BorutaPy(rf_boruta_reg, n_estimators="auto", random_state=42, verbose=2)
    boruta_reg.fit(Xtr_r_imp, ytr_r)

    selected_reg_mask = boruta_reg.support_
    boruta_reg_features = list(np.array(Xtr_r.columns)[selected_reg_mask])
    print("\n✅ Boruta REGRESSION selected features:")
    print(boruta_reg_features)
    
    print("\n=== SHAP: Classification model (LightGBM) ===")

    lgb_cls = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgb_cls.fit(Xtr_scaled, ytr)

    # Use LightGBM's built‑in TreeExplainer (fast)
    explainer_cls = shap.TreeExplainer(lgb_cls)
    shap_vals_cls = explainer_cls.shap_values(Xtr_scaled)[1]  # class=1 (rain)
    mean_abs_shap_cls = np.abs(shap_vals_cls).mean(axis=0)
    shap_imp_cls = pd.Series(mean_abs_shap_cls, index=X_cols).sort_values(ascending=False)

    print("✅ SHAP (classification, LightGBM) feature importances:")
    for feat, val in shap_imp_cls.head(10).items():
        print(f"{feat:<35} {val:.4f}")


    # ===========================================================================
    # 4️⃣ SHAP — REGRESSION (LightGBM)
    # ===========================================================================
    print("\n=== SHAP: Regression model (LightGBM) ===")

    lgb_reg = LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    lgb_reg.fit(Xtr_r_imp, ytr_r)

    explainer_reg = shap.TreeExplainer(lgb_reg)
    shap_vals_reg = explainer_reg.shap_values(Xtr_r_imp)
    mean_abs_shap_reg = np.abs(shap_vals_reg).mean(axis=0)
    shap_imp_reg = pd.Series(mean_abs_shap_reg, index=Xtr_r.columns).sort_values(ascending=False)

    print("✅ SHAP (regression, LightGBM) feature importances:")
    for feat, val in shap_imp_reg.head(10).items():
        print(f"{feat:<35} {val:.4f}")
    
   



if __name__ == "__main__":
    main()
