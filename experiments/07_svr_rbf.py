import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import RepeatedKFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "data"))

TRAIN_PATH = os.path.join(DATA_DIR, "train_dataset.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_dataset.csv")

MODEL_PATH = os.path.join(SCRIPT_DIR, "final_svr_model.joblib")
SCALER_X_PATH = os.path.join(SCRIPT_DIR, "scaler_X.joblib")
SCALER_Y_PATH = os.path.join(SCRIPT_DIR, "scaler_y.joblib")
SUBMISSION_PATH = os.path.join(SCRIPT_DIR, "submission_svr.csv")
RESULTS_PATH = os.path.join(SCRIPT_DIR, "svr_cv_results.csv")
INFO_PATH = os.path.join(SCRIPT_DIR, "svr_model_info.json")


# ============================================================
# CONSTANTS
# ============================================================

RANDOM_STATE = 42
TARGET = "overall_yield"

RAW_FEATURES = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]

ENGINEERED_FEATURES = [
    "res_time_proxy",
    "temp_avg",
    "jacket_x_restime",
    "inlet_x_restime",
]

FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df):
    out = df.copy()

    out["res_time_proxy"] = (
        out["length_m"] / out["flow_rate_L_min"]
    )

    out["temp_avg"] = (
        out["inlet_temperature_K"]
        + out["jacket_temperature_K"]
    ) / 2.0

    out["jacket_x_restime"] = (
        out["jacket_temperature_K"]
        * out["res_time_proxy"]
    )

    out["inlet_x_restime"] = (
        out["inlet_temperature_K"]
        * out["res_time_proxy"]
    )

    return out


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("SVR REACTOR YIELD MODEL")
print("=" * 70)

print("\nCode folder:")
print(SCRIPT_DIR)

print("\nData folder:")
print(DATA_DIR)

if not os.path.isfile(TRAIN_PATH):
    raise FileNotFoundError(
        f"\nTraining file not found:\n{TRAIN_PATH}"
    )

if not os.path.isfile(TEST_PATH):
    raise FileNotFoundError(
        f"\nTest file not found:\n{TEST_PATH}"
    )

print("\nTraining file: FOUND")
print("Test file    : FOUND")


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("\nTrain shape:", train.shape)
print("Test shape :", test.shape)

if TARGET not in train.columns:
    raise ValueError(
        f"Training data does not contain '{TARGET}'"
    )

if len(train) != 150:
    print(
        f"WARNING: expected 150 training rows, found {len(train)}"
    )

if len(test) != 50:
    print(
        f"WARNING: expected 50 test rows, found {len(test)}"
    )


# ============================================================
# CHECK REQUIRED FEATURES
# ============================================================

missing_train = [
    c for c in RAW_FEATURES
    if c not in train.columns
]

missing_test = [
    c for c in RAW_FEATURES
    if c not in test.columns
]

if missing_train:
    raise ValueError(
        f"Missing training features: {missing_train}"
    )

if missing_test:
    raise ValueError(
        f"Missing test features: {missing_test}"
    )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

train_fe = engineer_features(train)
test_fe = engineer_features(test)

X = train_fe[FEATURES].copy()
y = train[TARGET].copy()

X_test = test_fe[FEATURES].copy()

print("\nFeatures used:")
for feature in FEATURES:
    print(" -", feature)

print("\nNumber of features:", len(FEATURES))


# ============================================================
# SVR PIPELINE
# ============================================================

# X scaling happens inside the pipeline.
# y scaling happens inside TransformedTargetRegressor.
# This prevents scaling leakage during CV.

svr_pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "svr",
        SVR(kernel="rbf")
    )
])

model = TransformedTargetRegressor(
    regressor=svr_pipeline,
    transformer=StandardScaler()
)


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "regressor__svr__C": [10, 100, 300],
    "regressor__svr__epsilon": [0.05, 0.1],
    "regressor__svr__gamma": ["scale", 0.01],
}


# ============================================================
# CROSS VALIDATION
# ============================================================

cv = RepeatedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=RANDOM_STATE
)

print("\n" + "=" * 70)
print("SVR HYPERPARAMETER SEARCH")
print("=" * 70)

print("Validation: Repeated 5-Fold CV")
print("Repeats   : 5")
print("Workers   : 1")
print("Metric    : RMSE")


# ============================================================
# GRID SEARCH
# ============================================================

search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=1,
    refit=False,
    return_train_score=False
)

search.fit(X, y)


# ============================================================
# RESULTS
# ============================================================

best_index = search.best_index_

cv_results = pd.DataFrame(
    search.cv_results_
)

cv_results["mean_RMSE"] = (
    -cv_results["mean_test_score"]
)

cv_results["std_RMSE"] = (
    cv_results["std_test_score"]
)

results_to_save = cv_results[
    [
        "params",
        "mean_RMSE",
        "std_RMSE"
    ]
].sort_values(
    "mean_RMSE"
)

results_to_save.to_csv(
    RESULTS_PATH,
    index=False
)

best_params = search.best_params_
best_rmse = -search.best_score_
best_std = search.cv_results_[
    "std_test_score"
][best_index]

print("\nBest parameters:")
print(best_params)

print(
    f"\nBest CV RMSE : {best_rmse:.4f}"
)

print(
    f"CV Std       : {best_std:.4f}"
)


# ============================================================
# TRAIN FINAL MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL SVR")
print("=" * 70)

final_model = TransformedTargetRegressor(
    regressor=Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "svr",
            SVR(
                kernel="rbf",
                C=best_params[
                    "regressor__svr__C"
                ],
                epsilon=best_params[
                    "regressor__svr__epsilon"
                ],
                gamma=best_params[
                    "regressor__svr__gamma"
                ]
            )
        )
    ]),
    transformer=StandardScaler()
)

final_model.fit(X, y)

print("Final SVR trained on all 150 training rows.")


# ============================================================
# TRAINING PERFORMANCE
# ============================================================

train_pred = final_model.predict(X)

train_rmse = np.sqrt(
    mean_squared_error(
        y,
        train_pred
    )
)

train_mae = mean_absolute_error(
    y,
    train_pred
)

train_r2 = r2_score(
    y,
    train_pred
)

print("\nTraining metrics:")
print(
    f"RMSE: {train_rmse:.4f}"
)

print(
    f"MAE : {train_mae:.4f}"
)

print(
    f"R²  : {train_r2:.4f}"
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    final_model,
    MODEL_PATH
)

print("\nSaved model:")
print(MODEL_PATH)


# ============================================================
# SAVE OPTIONAL SCALERS
# ============================================================

# These are saved separately for reference.
# The final model itself already contains the scaling pipeline.

x_scaler = StandardScaler()
x_scaler.fit(X)

y_scaler = StandardScaler()
y_scaler.fit(
    y.to_numpy().reshape(-1, 1)
)

joblib.dump(
    x_scaler,
    SCALER_X_PATH
)

joblib.dump(
    y_scaler,
    SCALER_Y_PATH
)

print("Saved X scaler:")
print(SCALER_X_PATH)

print("Saved y scaler:")
print(SCALER_Y_PATH)


# ============================================================
# TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

predictions = final_model.predict(
    X_test
)

# Physical yield percentage range
predictions = np.clip(
    predictions,
    0.0,
    100.0
)

predictions = np.round(
    predictions,
    3
)


# ============================================================
# CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    TARGET: predictions
})


# ============================================================
# VALIDATION
# ============================================================

assert len(submission) == 50
assert list(submission.columns) == [TARGET]
assert not submission[TARGET].isna().any()
assert submission[TARGET].between(
    0,
    100
).all()

submission.to_csv(
    SUBMISSION_PATH,
    index=False
)

print("\nSubmission saved:")
print(SUBMISSION_PATH)


# ============================================================
# SAVE MODEL INFORMATION
# ============================================================

model_info = {
    "model": "SVR",
    "kernel": "RBF",
    "features": FEATURES,
    "best_parameters": best_params,
    "cv_method": "RepeatedKFold",
    "n_splits": 5,
    "n_repeats": 5,
    "random_state": RANDOM_STATE,
    "cv_rmse": float(best_rmse),
    "cv_rmse_std": float(best_std),
    "training_rmse": float(train_rmse),
    "training_mae": float(train_mae),
    "training_r2": float(train_r2),
    "test_prediction_min": float(predictions.min()),
    "test_prediction_max": float(predictions.max()),
    "test_prediction_mean": float(predictions.mean()),
    "test_rows": int(len(predictions))
}

with open(
    INFO_PATH,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        model_info,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SVR PROJECT COMPLETE")
print("=" * 70)

print(
    f"Best CV RMSE : {best_rmse:.4f}"
)

print(
    f"CV Std       : {best_std:.4f}"
)

print(
    f"Train RMSE   : {train_rmse:.4f}"
)

print(
    f"Prediction Min: {predictions.min():.3f}"
)

print(
    f"Prediction Max: {predictions.max():.3f}"
)

print(
    f"Prediction Mean: {predictions.mean():.3f}"
)

print("\nFiles created:")

print("1.", MODEL_PATH)
print("2.", SCALER_X_PATH)
print("3.", SCALER_Y_PATH)
print("4.", SUBMISSION_PATH)
print("5.", RESULTS_PATH)
print("6.", INFO_PATH)

print("\nFirst 10 predictions:")
print(
    submission.head(10).to_string(
        index=False
    )
)

print("\nSUCCESS - SVR pipeline finished.")
