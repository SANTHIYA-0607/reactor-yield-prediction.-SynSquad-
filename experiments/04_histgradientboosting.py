"""
HistGradientBoostingRegressor - 5-Fold Cross Validation (Configs A / B / C per fold)
======================================================================================

Within EACH of the 5 folds, three HistGradientBoostingRegressor hyperparameter
configurations (A, B, C) are trained and scored on that fold's validation split,
printed under their own sub-headings. After all folds, the mean RMSE/MAE/R2 of
each config (across the 5 folds) is compared and the best-performing config is
retrained on the full (feature-engineered) training data to produce the final
model + predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Import preprocessing
def engineer_features(df):
    df_fe = df.copy()
    df_fe['residence_time'] = df_fe['length_m'] / (df_fe['flow_rate_L_min'] + 1e-5)
    df_fe['delta_T'] = df_fe['jacket_temperature_K'] - df_fe['inlet_temperature_K']
    df_fe['inv_inlet_T'] = 1.0 / df_fe['inlet_temperature_K']
    df_fe['inv_jacket_T'] = 1.0 / df_fe['jacket_temperature_K']
    df_fe['mean_T'] = (df_fe['inlet_temperature_K'] + df_fe['jacket_temperature_K']) / 2.0
    df_fe['reaction_capacity'] = df_fe['concentration_mol_L'] * df_fe['residence_time']
    return df_fe

RANDOM_STATE = 42
N_SPLITS = 5
TARGET_COL = "overall_yield"

# ----------------------------------------------------------------------------
# Three HistGradientBoostingRegressor hyperparameter configs compared per fold
# ----------------------------------------------------------------------------
HGB_CONFIGS = {
    "A": dict(max_iter=100, learning_rate=0.10, max_depth=None, min_samples_leaf=20, l2_regularization=0.0),  # baseline
    "B": dict(max_iter=200, learning_rate=0.05, max_depth=5,    min_samples_leaf=10, l2_regularization=0.1),  # deeper / slower LR
    "C": dict(max_iter=50,  learning_rate=0.01, max_depth=3,    min_samples_leaf=5,  l2_regularization=1.0),  # shallow / heavily regularized
}


def main():
    base_dir = os.path.dirname(os.path.dirname(__file__))

    # ------------------------------------------------------------------
    # 1. LOAD DATA
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: LOADING DATA")
    print("=" * 70)

    train = pd.read_csv(os.path.join(base_dir, 'data', 'train_dataset.csv'))
    test = pd.read_csv(os.path.join(base_dir, 'data', 'test_dataset.csv'))

    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")

    # Feature Engineering
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    X = train_fe.drop(columns=[TARGET_COL])
    y = train_fe[TARGET_COL].values
    X_test = test_fe.copy()

    print(f"Engineered feature columns: {X.columns.tolist()}")

    X_values = X.values

    # ------------------------------------------------------------------
    # 2. 5-FOLD CROSS VALIDATION -- CONFIGS A / B / C COMPARED PER FOLD
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: 5-FOLD CROSS VALIDATION (Configs A / B / C per fold)")
    print("=" * 70)

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = {name: {"rmse": [], "mae": [], "r2": []} for name in HGB_CONFIGS}

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_values), start=1):
        X_tr, X_val = X_values[train_idx], X_values[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        print(f"\n--- Fold {fold_idx} ---")

        for config_name, params in HGB_CONFIGS.items():
            print(f"  Sub-config {config_name}: {params}")

            fold_model = HistGradientBoostingRegressor(**params, random_state=RANDOM_STATE)
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
    for config_name in HGB_CONFIGS:
        r = scores[config_name]["rmse"]
        m = scores[config_name]["mae"]
        r2v = scores[config_name]["r2"]
        print(f"Config {config_name}:")
        print(f"  Mean RMSE: {np.mean(r):.4f}  (+/- {np.std(r):.4f})")
        print(f"  Mean MAE:  {np.mean(m):.4f}  (+/- {np.std(m):.4f})")
        print(f"  Mean R2:   {np.mean(r2v):.4f}  (+/- {np.std(r2v):.4f})")

    # Pick the best config = lowest mean RMSE across the 5 folds
    best_config_name = min(HGB_CONFIGS, key=lambda name: np.mean(scores[name]["rmse"]))
    best_params = HGB_CONFIGS[best_config_name]
    print(f"\nBest config across folds: {best_config_name}  {best_params}")

    # ------------------------------------------------------------------
    # 3. TRAIN FINAL MODEL ON FULL TRAINING DATA (using best config)
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"STEP 3: TRAINING FINAL MODEL ON FULL TRAIN DATA (Config {best_config_name})")
    print("=" * 70)

    best_model = HistGradientBoostingRegressor(**best_params, random_state=RANDOM_STATE)
    best_model.fit(X, y)
    print(f"Final model trained on all {X.shape[0]} rows using Config {best_config_name}.")

    # ------------------------------------------------------------------
    # 4. PREDICT ON TEST DATA
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4: PREDICTING ON TEST DATA")
    print("=" * 70)

    predictions = best_model.predict(X_test)

    # Post-processing: physically yield cannot be negative
    predictions = np.clip(predictions, a_min=0.0, a_max=None)

    print(f"Generated {len(predictions)} predictions.")
    print("First 10 predictions:")
    for i in range(min(10, len(predictions))):
        print(f"  Sample {i + 1}: {predictions[i]:.3f}")

    # ------------------------------------------------------------------
    # 5. SAVE OUTPUTS
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 5: SAVING OUTPUTS")
    print("=" * 70)

    # Save Model
    model_path = os.path.join(base_dir, 'model', 'final_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")

    # Save Predictions
    sub = pd.DataFrame({'overall_yield': np.round(predictions, 3)})
    out_path = os.path.join(base_dir, 'outputs', 'TeamName.csv')
    sub.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")

    # Validation checks
    print(f"--- Submission Validation ---")
    print(f"Shape: {sub.shape} (Expected: (50, 1))")
    print(f"Missing Values: {sub.isna().sum().sum()} (Expected: 0)")
    print(f"Column Name: {sub.columns.tolist()[0]} (Expected: 'overall_yield')")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
