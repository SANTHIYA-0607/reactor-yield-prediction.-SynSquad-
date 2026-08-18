"""
train_predict.py
-----------------
End-to-end Gradient Boosting training + prediction.

5-Fold Cross Validation with Configs A / B / C per fold
---------------------------------------------------------
Within EACH of the 5 folds, three GradientBoostingRegressor hyperparameter
configurations (A, B, C) are trained and scored on that fold's validation
split, printed under their own sub-headings. After all folds, the mean
RMSE/STD of each config (across the 5 folds) is compared, the best config
is selected, and that config is retrained on the full training data to
produce the final model + submission.
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def engineer_features(df):
    df = df.copy()
    df["res_time_proxy"] = df["length_m"] / df["flow_rate_L_min"]
    df["avg_temperature_K"] = (df["inlet_temperature_K"] + df["jacket_temperature_K"]) / 2.0
    df["temp_diff_jacket_inlet_K"] = df["jacket_temperature_K"] - df["inlet_temperature_K"]
    df["arrhenius_proxy_inlet"] = np.exp(-1000.0 / df["inlet_temperature_K"])
    df["arrhenius_proxy_jacket"] = np.exp(-1000.0 / df["jacket_temperature_K"])
    df["temp_x_restime"] = df["avg_temperature_K"] * df["res_time_proxy"]
    df["arrhenius_x_restime"] = df["arrhenius_proxy_jacket"] * df["res_time_proxy"]
    df["damkohler_proxy"] = df["concentration_mol_L"] * df["res_time_proxy"]
    return df

def load_and_engineer(train_path, test_path):
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)
    train_eng = engineer_features(train_raw)
    test_eng = engineer_features(test_raw)
    features = [c for c in train_eng.columns if c != "overall_yield"]
    return train_eng[features], train_eng["overall_yield"], test_eng[features], features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_dataset.csv")
TEST_PATH = os.path.join(BASE_DIR, "data", "test_dataset.csv")

MODEL_OUT = os.path.join(BASE_DIR, "experiments", "gradientboosting_model.joblib")
SUBMISSION_OUT = os.path.join(BASE_DIR, "experiments", "gradientboosting_submission.csv")


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
N_SPLITS = 5

EXTRATREES_BENCHMARK_MEAN = 19.02
EXTRATREES_BENCHMARK_STD = 0.49

# ------------------------------------------------------------
# Three Gradient Boosting hyperparameter configs compared per fold
# ------------------------------------------------------------
GB_CONFIGS = {
    "A": dict(n_estimators=100, learning_rate=0.10, max_depth=2, subsample=1.0, min_samples_leaf=1),  # baseline
    "B": dict(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, min_samples_leaf=3),  # mid-complexity
    "C": dict(n_estimators=400, learning_rate=0.02, max_depth=2, subsample=0.9, min_samples_leaf=5),  # slow-learning / regularized
}


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("GRADIENT BOOSTING REACTOR YIELD MODEL")
    print("=" * 70)

    # --------------------------------------------------------
    # Load + engineer
    # --------------------------------------------------------

    X_train, y_train, X_test, feature_cols = load_and_engineer(
        TRAIN_PATH,
        TEST_PATH
    )

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    if len(X_train) != 150:
        print(f"WARNING: Expected 150 train rows, found {len(X_train)}")

    if len(X_test) != 50:
        print(f"WARNING: Expected 50 test rows, found {len(X_test)}")

    X_values = X_train.values
    y_values = y_train.values

    # --------------------------------------------------------
    # 5-Fold Cross-validation -- Configs A / B / C per fold
    # --------------------------------------------------------

    print("\nUsing:")
    print("5-fold CV (KFold, shuffled)")
    print(f"random_state = {RANDOM_STATE}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    scores = {name: {"rmse": [], "mae": [], "r2": []} for name in GB_CONFIGS}

    print("\n" + "=" * 70)
    print("STEP: 5-FOLD CROSS VALIDATION (Configs A / B / C per fold)")
    print("=" * 70)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_values), start=1):
        X_tr, X_val = X_values[train_idx], X_values[val_idx]
        y_tr, y_val = y_values[train_idx], y_values[val_idx]

        print(f"\n--- Fold {fold_idx} ---")

        for config_name, params in GB_CONFIGS.items():
            print(f"  Sub-config {config_name}: {params}")

            fold_model = GradientBoostingRegressor(random_state=RANDOM_STATE, **params)
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
    for config_name in GB_CONFIGS:
        r = scores[config_name]["rmse"]
        m = scores[config_name]["mae"]
        r2v = scores[config_name]["r2"]
        print(f"Config {config_name}:")
        print(f"  Mean RMSE: {np.mean(r):.4f}  (+/- {np.std(r):.4f})")
        print(f"  Mean MAE:  {np.mean(m):.4f}  (+/- {np.std(m):.4f})")
        print(f"  Mean R2:   {np.mean(r2v):.4f}  (+/- {np.std(r2v):.4f})")

    # --------------------------------------------------------
    # Select best config = lowest mean RMSE across the 5 folds
    # --------------------------------------------------------

    best_config_name = min(GB_CONFIGS, key=lambda name: np.mean(scores[name]["rmse"]))
    best_params = GB_CONFIGS[best_config_name]
    best_mean = float(np.mean(scores[best_config_name]["rmse"]))
    best_std = float(np.std(scores[best_config_name]["rmse"]))

    print("\n" + "=" * 70)
    print("BEST GRADIENT BOOSTING MODEL")
    print("=" * 70)

    print(f"Best config : {best_config_name}")
    print(f"CV RMSE : {best_mean:.4f} +/- {best_std:.4f}")
    print(f"Params  : {best_params}")

    # --------------------------------------------------------
    # Compare with ExtraTrees
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMPARISON WITH EXTRATREES")
    print("=" * 70)

    print(
        f"ExtraTrees      : "
        f"{EXTRATREES_BENCHMARK_MEAN:.2f} +/- "
        f"{EXTRATREES_BENCHMARK_STD:.2f}"
    )

    print(
        f"Gradient Boosting: "
        f"{best_mean:.2f} +/- "
        f"{best_std:.2f}"
    )

    if best_mean < EXTRATREES_BENCHMARK_MEAN:
        print("\nGradient Boosting BEATS the ExtraTrees benchmark.")
    else:
        print("\nGradient Boosting DOES NOT beat the ExtraTrees benchmark.")

    # --------------------------------------------------------
    # Train final model
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"TRAINING FINAL MODEL (Config {best_config_name})")
    print("=" * 70)

    final_model = GradientBoostingRegressor(
        random_state=RANDOM_STATE,
        **best_params
    )

    final_model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        final_model,
        MODEL_OUT
    )

    print(
        f"Saved model -> {MODEL_OUT}"
    )

    # --------------------------------------------------------
    # Predict test
    # --------------------------------------------------------

    preds = final_model.predict(
        X_test
    )

    # Yield must be within physical percentage range.
    preds = np.clip(
        preds,
        0.0,
        100.0
    )

    preds = np.round(
        preds,
        3
    )

    # --------------------------------------------------------
    # Submission
    # --------------------------------------------------------

    submission = pd.DataFrame({
        "overall_yield": preds
    })

    assert submission.shape == (
        50,
        1
    ), "Submission must contain exactly 50 rows and 1 column."

    assert not submission[
        "overall_yield"
    ].isna().any(), "Submission contains NaN."

    assert submission[
        "overall_yield"
    ].between(0, 100).all(), "Prediction outside 0-100."

    submission.to_csv(
        SUBMISSION_OUT,
        index=False
    )

    print(
        f"Saved submission -> {SUBMISSION_OUT}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"Final CV RMSE : {best_mean:.4f}"
    )

    print(
        f"CV Std        : {best_std:.4f}"
    )

    print(
        f"Prediction min : {preds.min():.3f}"
    )

    print(
        f"Prediction max : {preds.max():.3f}"
    )

    print(
        f"Prediction mean: {preds.mean():.3f}"
    )


if __name__ == "__main__":
    main()
