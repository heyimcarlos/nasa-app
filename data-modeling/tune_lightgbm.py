# tune_lgbm_model.py
# -----------------------------------------------------------------------------
# Uses RandomizedSearchCV to find optimal hyperparameters for the 2-stage
# LightGBM model. This is a separate, potentially long-running process.
# -----------------------------------------------------------------------------

import time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.stats import randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from power_data import get_dataset, split_by_years


# --- Main Script ---
def main():
    # === Load the dataset ===
    bundle = get_dataset(years=5)
    data = bundle.data
    climo_cols = bundle.climo_cols

    # === Split data by year ===
    train_df, _, _ = split_by_years(data, train=0.8, val=0.0, test=0.2)
    # Note: We combine train and val for tuning, as CV will create its own internal splits.

    # === Define feature set ===
    lag_cols = [c for c in data.columns if "_lag_" in c]
    roll_cols = [c for c in data.columns if "_day_avg" in c]
    X_cols = ["lat", "lon", "doy_sin", "doy_cos"] + climo_cols + lag_cols + roll_cols
    y_cls = "rain_flag"
    y_reg = "y_log1p"

    X_train, y_train = train_df[X_cols], train_df[y_cls]

    rain_train_df = train_df[train_df[y_cls] == 1]
    X_train_r, y_train_r = rain_train_df[X_cols], rain_train_df[y_reg]

    # =========================================================================
    # === 1. Tune the LGBM Classifier ===
    # =========================================================================
    print("\n--- Tuning LGBM Classifier ---")

    clf_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", lgb.LGBMClassifier(objective="binary", random_state=42)),
        ]
    )

    clf_param_dist = {
        "lgbm__n_estimators": randint(100, 500),
        "lgbm__learning_rate": uniform(0.01, 0.1),
        "lgbm__num_leaves": randint(20, 50),
        "lgbm__reg_alpha": uniform(0, 1),
        "lgbm__reg_lambda": uniform(0, 1),
        "lgbm__colsample_bytree": uniform(0.6, 0.4),
    }

    clf_tuner = RandomizedSearchCV(
        estimator=clf_pipe,
        param_distributions=clf_param_dist,
        n_iter=25,
        cv=3,
        scoring="roc_auc",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    start_time = time.time()
    clf_tuner.fit(X_train, y_train)
    print(f"Classifier tuning finished in {time.time() - start_time:.2f} seconds.")

    print("\nBest Classifier Parameters:")
    print(clf_tuner.best_params_)
    print(f"\nBest Classifier AUC Score: {clf_tuner.best_score_:.4f}")

    # =========================================================================
    # === 2. Tune the LGBM Regressor ===
    # =========================================================================
    print("\n\n--- Tuning LGBM Regressor ---")

    reg_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("lgbm", lgb.LGBMRegressor(objective="regression_l1", random_state=42)),
        ]
    )

    reg_param_dist = clf_param_dist

    reg_tuner = RandomizedSearchCV(
        estimator=reg_pipe,
        param_distributions=reg_param_dist,
        n_iter=25,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )

    start_time = time.time()
    reg_tuner.fit(X_train_r, y_train_r)
    print(f"Regressor tuning finished in {time.time() - start_time:.2f} seconds.")

    print("\nBest Regressor Parameters:")
    print(reg_tuner.best_params_)
    print(f"\nBest Regressor MAE Score: {-reg_tuner.best_score_:.4f}")


if __name__ == "__main__":
    main()
