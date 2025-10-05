# logreg_randomforest.py
# -----------------------------------------------------------------------------
# Baseline 2-stage model:
#   1) Logistic Regression -> rain / no-rain probability
#   2) Random Forest Regressor on log1p(rain mm) for rainy days
# Consumes the shared dataset via power_data.get_dataset(...), evaluates,
# and writes a single .pkl artifact with everything needed for serving.
#
# Dependencies:
#   pip install scikit-learn joblib
# -----------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from power_data import (TARGET_PARAM, EXTRA_PARAMS, BBOX_TORONTO, get_dataset, split_by_years)
from common import (
    rmse,
    get_feature_columns,
    make_XY,
    make_XR,
    build_climo_baseline,
    baseline_expected_mm as baseline_expected_mm_common,
    combined_expected_mm as combined_expected_mm_common,
)
from inference import predict_precip_mm as shared_predict_precip_mm  # noqa: F401 (imported for external use)

BASE_DIR = Path(__file__).resolve().parent              # → nasa-app/data-modeling
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    # === Load the shared dataset (one-liner) ===
    bundle = get_dataset(years=5, bbox=BBOX_TORONTO, target_param=TARGET_PARAM, extra_params=EXTRA_PARAMS)

    data = bundle.data
    climo_cols = bundle.climo_cols
    ds = bundle.ds
    ds_climo = bundle.ds_climo

    # === Split: proportions by years (chronological) ===
    train_df, val_df, test_df = split_by_years(data, train=0.6, val=0.2, test=0.2, chronological=True)

    # === Feature set ===
    X_cols = get_feature_columns(climo_cols)
    y_cls = "rain_flag"
    y_reg = "y_log1p"

    Xtr, ytr = make_XY(train_df, X_cols, y_cls)
    Xva, yva = make_XY(val_df, X_cols, y_cls)
    Xte, yte = make_XY(test_df, X_cols, y_cls)

    Xtr_r, ytr_r = make_XR(train_df, X_cols, y_reg, y_cls)
    Xva_r, yva_r = make_XR(val_df, X_cols, y_reg, y_cls)
    Xte_r, yte_r = make_XR(test_df, X_cols, y_reg, y_cls)

    print(f"Features: {len(X_cols)}  | Train rows: {len(train_df):,}  | Rain-only rows: {len(Xtr_r):,}")

    # === Stage 1 — rain/no-rain classifier ===
    clf = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scal", StandardScaler(with_mean=False)),
            ("logit", LogisticRegression(max_iter=5000, class_weight="balanced")),
        ]
    ).fit(Xtr, ytr)

    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    print("Classifier — Val AUC:", round(roc_auc_score(yva, p_va), 3))
    print("Classifier — Val Brier:", round(brier_score_loss(yva, p_va), 4))
    print("Classifier — Test AUC:", round(roc_auc_score(yte, p_te), 3))
    print("Classifier — Test Brier:", round(brier_score_loss(yte, p_te), 4))

    # === Stage 2 — regressor on rainy rows (log1p scale) ===
    reg = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
        ]
    ).fit(Xtr_r, ytr_r)

    pred_va_log = reg.predict(Xva_r)
    pred_te_log = reg.predict(Xte_r)

    print(
        f"Regressor (rain-only) — Val MAE: {mean_absolute_error(np.expm1(yva_r), np.expm1(pred_va_log)):.3f} mm, "
        f"RMSE: {rmse(np.expm1(yva_r), np.expm1(pred_va_log)):.3f} mm"
    )
    print(
        f"Regressor (rain-only) — Test MAE: {mean_absolute_error(np.expm1(yte_r), np.expm1(pred_te_log)):.3f} mm, "
        f"RMSE: {rmse(np.expm1(yte_r), np.expm1(pred_te_log)):.3f} mm"
    )

    # === Combined expected rainfall vs climatology baseline ===
    climo_baseline = build_climo_baseline(train_df, TARGET_PARAM)

    y_true = test_df[TARGET_PARAM].to_numpy()
    y_model = combined_expected_mm_common(clf, reg, test_df, X_cols)
    y_base = baseline_expected_mm_common(test_df, climo_baseline)

    print("\n== Test (all days) ==")
    print(f"Model (2-stage)  — MAE={mean_absolute_error(y_true, y_model):.3f} mm   RMSE={rmse(y_true, y_model):.3f} mm")
    print(f"Baseline (climo) — MAE={mean_absolute_error(y_true, y_base):.3f} mm   RMSE={rmse(y_true, y_base):.3f} mm")

    # === Save model bundle for serving (pickle) ===
    bundle_to_save = {
        "clf": clf,
        "reg": reg,
        "X_cols": X_cols,
        "climo_cols": climo_cols,
        "target_param": TARGET_PARAM,
        "bbox": BBOX_TORONTO,
        "feature_version": 1,  # bump when feature engineering changes
    }
    out_path = ARTIFACTS_DIR / "rain_model_v1.pkl"
    joblib.dump(bundle_to_save, out_path)
    print(f"\nSaved model bundle → {out_path.resolve()}")

    # Optional: one-off inference helper (kept here for dev use)
    # from logreg_randomforest import predict_precip_mm
    # print(predict_precip_mm(bundle_to_save, ds, ds_climo, 43.65, -79.38, "2025-11-15"))

# ---- Small utility for Flask-side inference (kept here for convenience) ----
def predict_precip_mm(
    model_bundle: Dict[str, Any],
    ds,           # xarray Dataset for TARGET_PARAM (grid reference)
    ds_climo,     # xarray Dataset with *_clim vars aligned to ds grid
    lat: float,
    lon: float,
    date_str: str,
) -> Dict[str, float]:
    """Given (lat, lon, date), return predictions using a saved model bundle.

    Returns dict with:
      - p_rain: probability of precipitation - X% chance of rain
      - amount_if_rain_mm: predicted amount conditional on rain - Y mm total if it rains
      - expected_mm: p_rain * amount_if_rain_mm - average you'd plan for ≈ Z mm
    """
    clf = model_bundle["clf"]
    reg = model_bundle["reg"]
    X_cols = model_bundle["X_cols"]

    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not locate lat/lon coords in ds.")

    dt = pd.to_datetime(date_str)
    doy = int(dt.strftime("%j"))
    row = {
        "lat": float(ds[lat_name].sel({lat_name: lat}, method="nearest").values),
        "lon": float(ds[lon_name].sel({lon_name: lon}, method="nearest").values),
        "doy": doy,
        "doy_sin": np.sin(2 * np.pi * doy / 365.0),
        "doy_cos": np.cos(2 * np.pi * doy / 365.0),
    }

    cl = ds_climo.sel(
        doy=doy,
        **{lat_name: row["lat"], lon_name: row["lon"]},
        method="nearest",
    )
    for v in ds_climo.data_vars:
        row[v] = float(cl[v].values)

    X = pd.DataFrame([row])[X_cols]
    p = float(clf.predict_proba(X)[:, 1])
    amt = float(np.expm1(reg.predict(X)))

    return {
        "p_rain": max(0.0, min(1.0, p)),
        "amount_if_rain_mm": max(0.0, amt),
        "expected_mm": max(0.0, p * amt),
    }

if __name__ == "__main__":
    main()
