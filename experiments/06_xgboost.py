"""
xgb_experiment.py  (flat-folder version)

Put this file in the SAME folder as:
  - preprocessing.py
  - train_dataset.csv
  - test_dataset.csv

Run with:  python xgb_experiment.py

Compares three feature sets under IDENTICAL 5-fold CV splits:
  Experiment A: 5 raw features
  Experiment B: existing 9-feature physics-informed set
  Experiment C: expanded 11-feature physics-informed set

Writes cv_experiment_results.csv and best_experiment.json into this same folder.
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

RAW_FEATURES = [
    "flow_rate_L_min", "concentration_mol_L", "inlet_temperature_K", "length_m", "jacket_temperature_K"
]
FINAL_FEATURES = RAW_FEATURES + ["res_time_proxy", "temp_avg", "jacket_x_restime", "inlet_x_restime"]
EXPANDED_FEATURES = FINAL_FEATURES + ["thermal_diff", "conc_x_restime"]
TARGET_COLUMN = "overall_yield"

def engineer_features(df):
    df = df.copy()
    df["res_time_proxy"] = df["length_m"] / df["flow_rate_L_min"]
    df["temp_avg"] = (df["inlet_temperature_K"] + df["jacket_temperature_K"]) / 2.0
    df["jacket_x_restime"] = df["jacket_temperature_K"] * df["res_time_proxy"]
    df["inlet_x_restime"] = df["inlet_temperature_K"] * df["res_time_proxy"]
    return df

def engineer_features_expanded(df):
    df = engineer_features(df)
    df["thermal_diff"] = df["jacket_temperature_K"] - df["inlet_temperature_K"]
    df["conc_x_restime"] = df["concentration_mol_L"] * df["res_time_proxy"]
    return df

RANDOM_STATE = 42
N_SPLITS = 5
N_ITER_SEARCH = 40

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

train = pd.read_csv(os.path.join(ROOT, 'data', 'train_dataset.csv'))
test = pd.read_csv(os.path.join(ROOT, 'data', 'test_dataset.csv'))

train_eng = engineer_features_expanded(train)  # superset; slice per experiment
test_eng = engineer_features_expanded(test)

y_train = train_eng[TARGET_COLUMN].values

# The SAME folds are reused for every experiment / every candidate config.
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

param_dist = {
    'n_estimators': randint(100, 600),
    'learning_rate': uniform(0.01, 0.19),       # 0.01 - 0.20
    'max_depth': randint(2, 6),
    'min_child_weight': randint(1, 8),
    'subsample': uniform(0.6, 0.4),              # 0.6 - 1.0
    'colsample_bytree': uniform(0.6, 0.4),        # 0.6 - 1.0
    'gamma': uniform(0.0, 0.5),
    'reg_alpha': uniform(0.0, 1.0),
    'reg_lambda': uniform(0.5, 2.5),
}

def run_experiment(name, feature_list):
    X = train_eng[feature_list].values
    base_est = XGBRegressor(
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    search = RandomizedSearchCV(
        estimator=base_est,
        param_distributions=param_dist,
        n_iter=N_ITER_SEARCH,
        scoring='neg_root_mean_squared_error',
        cv=kf,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y_train)

    best_idx = search.best_index_
    cvres = search.cv_results_
    fold_rmses = [-cvres[f'split{i}_test_score'][best_idx] for i in range(N_SPLITS)]
    mean_rmse = -cvres['mean_test_score'][best_idx]
    std_rmse = cvres['std_test_score'][best_idx]

    train_pred = search.best_estimator_.predict(X)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))

    result = {
        'experiment': name,
        'n_features': len(feature_list),
        'features': feature_list,
        'best_params': search.best_params_,
        'fold_rmses': fold_rmses,
        'mean_cv_rmse': mean_rmse,
        'std_cv_rmse': std_rmse,
        'train_rmse_full_fit': train_rmse,
    }
    print(f"\n=== {name} ({len(feature_list)} features) ===")
    print("Best params:", search.best_params_)
    for i, r in enumerate(fold_rmses, 1):
        print(f"  Fold {i} RMSE: {r:.4f}")
    print(f"  Mean CV RMSE: {mean_rmse:.4f}  Std: {std_rmse:.4f}")
    print(f"  Train RMSE (full fit): {train_rmse:.4f}  (gap = {mean_rmse - train_rmse:.4f})")
    return result, search.best_estimator_


results = []
res_A, model_A = run_experiment('A: raw 5 features', RAW_FEATURES)
results.append(res_A)
res_B, model_B = run_experiment('B: existing 9 features', FINAL_FEATURES)
results.append(res_B)
res_C, model_C = run_experiment('C: expanded 11 features', EXPANDED_FEATURES)
results.append(res_C)

# ---- Reference: original ExtraTrees spec on the SAME folds / feature set B ----
et_ref = ExtraTreesRegressor(
    n_estimators=300, max_depth=8, min_samples_leaf=3,
    max_features=1.0, random_state=RANDOM_STATE,
)
X_B = train_eng[FINAL_FEATURES].values
et_scores = cross_val_score(et_ref, X_B, y_train, scoring='neg_root_mean_squared_error', cv=kf)
et_fold_rmses = list(-et_scores)
et_mean, et_std = -et_scores.mean(), et_scores.std()
print(f"\n=== Reference: existing ExtraTreesRegressor (9 features, same folds) ===")
for i, r in enumerate(et_fold_rmses, 1):
    print(f"  Fold {i} RMSE: {r:.4f}")
print(f"  Mean CV RMSE: {et_mean:.4f}  Std: {et_std:.4f}")

results.append({
    'experiment': 'Reference: ExtraTrees (9 features)',
    'n_features': 9,
    'features': FINAL_FEATURES,
    'best_params': {'n_estimators': 300, 'max_depth': 8, 'min_samples_leaf': 3, 'max_features': 1.0},
    'fold_rmses': et_fold_rmses,
    'mean_cv_rmse': et_mean,
    'std_cv_rmse': et_std,
    'train_rmse_full_fit': None,
})

# ---- Save comparison table ----
rows = []
for r in results:
    row = {'Experiment': r['experiment'], 'N_Features': r['n_features']}
    for i, v in enumerate(r['fold_rmses'], 1):
        row[f'Fold{i}'] = round(v, 4)
    row['Mean_RMSE'] = round(r['mean_cv_rmse'], 4)
    row['Std_RMSE'] = round(r['std_cv_rmse'], 4)
    rows.append(row)
comp_df = pd.DataFrame(rows)
comp_path = os.path.join(HERE, 'cv_experiment_results.csv')
comp_df.to_csv(comp_path, index=False)
print(f"\nSaved comparison table to {comp_path}")
print(comp_df.to_string(index=False))

xgb_results = [r for r in results if r['experiment'].startswith(('A', 'B', 'C'))]
best = min(xgb_results, key=lambda r: r['mean_cv_rmse'])
print(f"\nBest XGBoost experiment: {best['experiment']}  (mean CV RMSE={best['mean_cv_rmse']:.4f})")

with open(os.path.join(HERE, 'best_experiment.json'), 'w') as f:
    json.dump({
        'experiment': best['experiment'],
        'features': best['features'],
        'best_params': best['best_params'],
        'mean_cv_rmse': best['mean_cv_rmse'],
        'std_cv_rmse': best['std_cv_rmse'],
        'fold_rmses': best['fold_rmses'],
        'train_rmse_full_fit': best['train_rmse_full_fit'],
        'et_reference_mean_cv_rmse': et_mean,
        'et_reference_std_cv_rmse': et_std,
    }, f, indent=2)

with open(os.path.join(HERE, 'all_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("Done.")
