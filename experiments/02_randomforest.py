"""
Random Forest Regressor - 5-Fold Cross Validation (with A/B/C configs per fold)
Reactor Yield Prediction (ML Hackathon: Predictive Modeling Optimization Challenge)
====================================================================================

Predicts 'overall_yield' of Product B from a non-isothermal continuous flow
reactor using: flow_rate_L_min, concentration_mol_L, inlet_temperature_K,
length_m, jacket_temperature_K.

Within EACH of the 5 folds, three Random Forest hyperparameter configurations
(A, B, C) are trained and scored on that fold's validation split, printed
under their own sub-headings. After all folds, the mean RMSE/MAE/R2 of each
config (across the 5 folds) is compared and the best-performing config is
retrained on the full training data to produce the final model + predictions.

Run this file directly (e.g. in IDLE: Run > Run Module, or `python train_rf_model.py`).
It expects train_dataset.csv and test_dataset.csv to be in the SAME folder as
this script. All outputs (model + predictions) are also written to this folder.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# ----------------------------------------------------------------------------
# 0. SETUP — resolve paths relative to this script so it runs anywhere
# ----------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
TRAIN_PATH = os.path.join(BASE_DIR, "train_dataset.csv")
TEST_PATH = os.path.join(BASE_DIR, "test_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "final_model.joblib")
OUTPUT_PATH = os.path.join(BASE_DIR, "output.csv")

RANDOM_STATE = 42
N_SPLITS = 5

FEATURE_COLS = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]
TARGET_COL = "overall_yield"

# ----------------------------------------------------------------------------
# Three Random Forest hyperparameter configurations compared inside every fold
# ----------------------------------------------------------------------------
RF_CONFIGS = {
    "A": dict(n_estimators=300, max_depth=None, min_samples_leaf=2),   # baseline
    "B": dict(n_estimators=500, max_depth=10,   min_samples_leaf=1),   # deeper / more trees
    "C": dict(n_estimators=200, max_depth=5,    min_samples_leaf=4),   # shallower / regularized
}


def main():
    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)

    print(f"Train shape: {train_data.shape}")
    print(f"Test shape:  {test_data.shape}")

    X = train_data[FEATURE_COLS].values
    y = train_data[TARGET_COL].values
    X_test = test_data[FEATURE_COLS].values

    # ------------------------------------------------------------------
    # 2. 5-FOLD CROSS VALIDATION -- CONFIGS A / B / C COMPARED PER FOLD
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: 5-FOLD CROSS VALIDATION (Configs A / B / C per fold)")
    print("=" * 70)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # scores[config_name] = {"rmse": [...], "mae": [...], "r2": [...]}  (one entry per fold)
    scores = {name: {"rmse": [], "mae": [], "r2": []} for name in RF_CONFIGS}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        print(f"\n--- Fold {fold_idx} ---")

        for config_name, params in RF_CONFIGS.items():
            print(f"  Sub-config {config_name}: {params}")

            fold_model = RandomForestRegressor(
                **params,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            fold_model.fit(X_tr, y_tr)
            val_pred = fold_model.predict(X_val)

            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            mae = mean_absolute_error(y_val, val_pred)
            r2 = r2_score(y_val, val_pred)

            scores[config_name]["rmse"].append(rmse)
            scores[config_name]["mae"].append(mae)
            scores[config_name]["r2"].append(r2)

            print(f"    Fold {fold_idx}{config_name}: RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    print("\n" + "-" * 70)
    print("5-Fold CV Summary (mean +/- std across the 5 folds, per config)")
    print("-" * 70)
    for config_name in RF_CONFIGS:
        r = scores[config_name]["rmse"]
        m = scores[config_name]["mae"]
        r2 = scores[config_name]["r2"]
        print(f"Config {config_name}:")
        print(f"  Mean RMSE: {np.mean(r):.4f}  (+/- {np.std(r):.4f})")
        print(f"  Mean MAE:  {np.mean(m):.4f}  (+/- {np.std(m):.4f})")
        print(f"  Mean R2:   {np.mean(r2):.4f}  (+/- {np.std(r2):.4f})")

    # Pick the best config = lowest mean RMSE across the 5 folds
    best_config_name = min(RF_CONFIGS, key=lambda name: np.mean(scores[name]["rmse"]))
    best_params = RF_CONFIGS[best_config_name]
    print(f"\nBest config across folds: {best_config_name}  {best_params}")

    # ------------------------------------------------------------------
    # 3. TRAIN FINAL MODEL ON FULL TRAINING DATA (using best config)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 3: TRAINING FINAL MODEL ON FULL TRAIN DATA (Config {best_config_name})")
    print("=" * 70)

    final_model = RandomForestRegressor(
        **best_params,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    final_model.fit(X, y)
    print(f"Final model trained on all {X.shape[0]} rows using Config {best_config_name}.")

    # Feature importance (useful for the "process insight" judging criterion)
    print("\nFeature Importances:")
    for name, imp in sorted(
        zip(FEATURE_COLS, final_model.feature_importances_),
        key=lambda t: t[1],
        reverse=True,
    ):
        print(f"  {name}: {imp:.4f}")

    # ------------------------------------------------------------------
    # 4. PREDICT ON TEST DATA
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: PREDICTING ON TEST DATA")
    print("=" * 70)

    test_pred = final_model.predict(X_test)
    test_pred = np.round(test_pred, 3)

    print(f"Generated {len(test_pred)} predictions.")
    print("First 10 predictions:")
    for i in range(min(10, len(test_pred))):
        print(f"  Sample {i + 1}: {test_pred[i]}")

    # ------------------------------------------------------------------
    # 5. SAVE OUTPUTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: SAVING OUTPUTS")
    print("=" * 70)

    output_df = pd.DataFrame({TARGET_COL: test_pred})
    output_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to: {OUTPUT_PATH}")

    joblib.dump(final_model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
