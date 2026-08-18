from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor

try:
    from preprocessing import engineer_features, FINAL_FEATURES, TARGET_COLUMN, project_root
except ImportError:
    from src.preprocessing import engineer_features, FINAL_FEATURES, TARGET_COLUMN, project_root

RANDOM_STATE = 42
MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 3,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def main() -> None:
    root = project_root()
    train_path = root / "data" / "train_dataset.csv"
    test_path = root / "data" / "test_dataset.csv"
    model_path = root / "model" / "final_extratrees_model.joblib"
    submission_path = root / "outputs" / "submission.csv"
    report_path = root / "outputs" / "final_extratrees_report.json"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if len(train_df) != 150:
        raise ValueError(f"Expected 150 training rows, found {len(train_df)}")
    if len(test_df) != 50:
        raise ValueError(f"Expected 50 test rows, found {len(test_df)}")
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Training target '{TARGET_COLUMN}' not found")

    train_fe = engineer_features(train_df)
    test_fe = engineer_features(test_df)

    X = train_fe[FINAL_FEATURES]
    y = train_fe[TARGET_COLUMN]
    X_test = test_fe[FINAL_FEATURES]

    if X.isna().any().any() or X_test.isna().any().any() or y.isna().any():
        raise ValueError("NaN values detected in training/test data after feature engineering")

    model = ExtraTreesRegressor(**MODEL_PARAMS)
    model.fit(X, y)

    raw_preds = model.predict(X_test)
    preds = np.clip(raw_preds, 0.0, 100.0)
    clipped_rows = int(np.sum(~np.isclose(raw_preds, preds)))

    joblib.dump(model, model_path)
    pd.DataFrame({TARGET_COLUMN: np.round(preds, 3)}).to_csv(submission_path, index=False)

    importances = {
        feature: float(importance)
        for feature, importance in zip(FINAL_FEATURES, model.feature_importances_)
    }
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    report = {
        "final_model": "ExtraTreesRegressor",
        "final_feature_set": "Feature Set B",
        "n_features": len(FINAL_FEATURES),
        "features": FINAL_FEATURES,
        "validation_metric": "RMSE",
        "validation_rmse": 16.2273,
        "validation_std": 2.1423,
        "validation_note": "Local validation result supplied in the project report; not the hidden competition score.",
        "model_parameters": MODEL_PARAMS,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "prediction_min": float(preds.min()),
        "prediction_max": float(preds.max()),
        "prediction_mean": float(preds.mean()),
        "rows_clipped_to_0_100": clipped_rows,
        "feature_importance": importances,
        "submission_file": "outputs/submission.csv",
        "model_file": "model/final_extratrees_model.joblib",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Final model: {type(model).__name__}")
    print(f"Features: {len(FINAL_FEATURES)}")
    print(f"Submission rows: {len(preds)}")
    print(f"Saved: {model_path}")
    print(f"Saved: {submission_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
