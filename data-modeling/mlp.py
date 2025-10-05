# mlp.py
# -----------------------------------------------------------------------------
# Multi-layer Perceptron (MLP) classifier for rain/no-rain prediction.
# -----------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import (brier_score_loss, mean_absolute_error, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
from inference import predict_precip_mm as shared_predict_precip_mm 

BASE_DIR = Path(__file__).resolve().parent              # → nasa-app/data-modeling
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Optional knobs (default off)
DO_TUNE_CLF = False
DO_TUNE_REG = False
DO_CALIBRATE = False
DO_BALANCE = True

def build_mlp_classifier(
    hidden_layer_sizes=(64, 32),
    alpha=1e-4,
    learning_rate_init=1e-3,
    max_iter=10000,
    random_state=42,
):
    """Construct a readable MLP classifier pipeline.

    Pipeline layout:
      - SimpleImputer: fill missing values with feature-wise median
      - StandardScaler: zero-mean/unit-variance features (benefits neural nets)
      - MLPClassifier: small MLP suitable for tabular classification
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        shuffle=True,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        verbose=False,
    )

    return Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scal", StandardScaler(with_mean=True)),
            ("mlp", mlp),
        ]
    )

def downsample_majority(
    X: pd.DataFrame,
    y: pd.Series,
    ratio: float = 1.0,
    random_state: int = 42,
):
    """Downsample majority class so majority ~= ratio * minority.

    If classes are balanced or single-class, returns inputs unchanged.
    """
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2:
        return X, y
    min_class = classes[int(np.argmin(counts))]
    max_class = classes[int(np.argmax(counts))]
    n_min = int(counts.min())
    n_max_keep = max(n_min, int(round(n_min * ratio)))
    idx_min = y[y == min_class].index
    idx_max_all = y[y == max_class].index
    if len(idx_max_all) <= n_max_keep:
        return X, y
    idx_max = idx_max_all.to_series().sample(n=n_max_keep, random_state=random_state).index
    keep_idx = idx_min.union(idx_max)
    Xb = X.loc[keep_idx]
    yb = y.loc[keep_idx]
    return Xb, yb

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

    # === Stage 1 — rain/no-rain classifier (Multi-layer Perceptron) ===
    base_clf = build_mlp_classifier()
    Xtr_fit, ytr_fit = (downsample_majority(Xtr, ytr) if DO_BALANCE else (Xtr, ytr))
    if DO_TUNE_CLF:
        clf_grid = {
            "mlp__hidden_layer_sizes": [(64, 32), (128, 64)],
            "mlp__alpha": [1e-5, 1e-4, 1e-3],
            "mlp__learning_rate_init": [1e-3, 5e-4],
        }
        gs = GridSearchCV(
            estimator=base_clf,
            param_grid=clf_grid,
            scoring="roc_auc",
            cv=3,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs.fit(Xtr_fit, ytr_fit)
        clf = gs.best_estimator_
        print("[TUNE] Best classifier params:", gs.best_params_)
    else:
        clf = base_clf.fit(Xtr_fit, ytr_fit)

    if DO_CALIBRATE:
        calibrator = CalibratedClassifierCV(base_estimator=clf, method="isotonic", cv="prefit")
        calibrator.fit(Xva, yva)
        clf = calibrator

    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    print("NN Classifier — Val AUC:", round(roc_auc_score(yva, p_va), 3))
    print("NN Classifier — Val Brier:", round(brier_score_loss(yva, p_va), 4))
    print("NN Classifier — Test AUC:", round(roc_auc_score(yte, p_te), 3))
    print("NN Classifier — Test Brier:", round(brier_score_loss(yte, p_te), 4))

    # === Stage 2 — regressor on rainy rows (log1p scale) ===
    reg_pipeline = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("scal", StandardScaler(with_mean=True)),
            ("mlpr", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate="constant",
                learning_rate_init=1e-3,
                max_iter=10000,
                shuffle=True,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                verbose=False,
            )),
        ]
    )
    if DO_TUNE_REG:
        reg_grid = {
            "mlpr__hidden_layer_sizes": [(64, 32), (128, 64)],
            "mlpr__alpha": [1e-5, 1e-4, 1e-3],
            "mlpr__learning_rate_init": [1e-3, 5e-4],
        }
        gs_r = GridSearchCV(
            estimator=reg_pipeline,
            param_grid=reg_grid,
            scoring="neg_mean_squared_error",
            cv=3,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        gs_r.fit(Xtr_r, ytr_r)
        reg = gs_r.best_estimator_
        print("[TUNE] Best regressor params:", gs_r.best_params_)
    else:
        reg = reg_pipeline.fit(Xtr_r, ytr_r)

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
    out_path = ARTIFACTS_DIR / "mlp_rain_model.pkl"
    joblib.dump(bundle_to_save, out_path)
    print(f"\nSaved model bundle → {out_path.resolve()}")

if __name__ == "__main__":
    # Optional CLI flags without adding a dependency: set via env or edit booleans above.
    # If needed later, we can add argparse here to toggle DO_* flags.
    main()
