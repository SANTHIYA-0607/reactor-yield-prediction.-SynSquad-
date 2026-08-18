# Reactor Yield Prediction

A physics-informed machine-learning workflow for predicting **Overall Yield (%) of Product B** in a non-isothermal, continuous-flow chemical reactor.

## Problem summary

The reactor follows a series-parallel reaction network:

- Desired reaction: `A → B`
- Side reaction: `B → C`

The ML model is used as a surrogate predictor for reactor yield, avoiding the need to directly solve the underlying differential equations for every prediction.

**Dataset:** 150 training rows with target + 50 test rows without target.

**Target:** `overall_yield`

**Competition metric:** blind RMSE on the hidden test solution. The validation results reported below are **local validation results only**, not the hidden competition score.

## Input features

| Feature | Description |
|---|---|
| `flow_rate_L_min` | Volumetric flow rate of the reactant mixture (L/min) |
| `concentration_mol_L` | Inlet concentration of Reactant A (mol/L) |
| `inlet_temperature_K` | Feed temperature entering the reactor (K) |
| `length_m` | Reactor length (m) |
| `jacket_temperature_K` | External heating jacket temperature (K) |

## Feature engineering

The final model uses **Feature Set B — Physics-informed**, consisting of the five raw inputs plus four engineered variables:

- `res_time_proxy = length_m / flow_rate_L_min`
- `temp_avg = (inlet_temperature_K + jacket_temperature_K) / 2`
- `jacket_x_restime = jacket_temperature_K × res_time_proxy`
- `inlet_x_restime = inlet_temperature_K × res_time_proxy`

These transformations are applied identically to training and test data in `src/preprocessing.py`.

### Feature sets

**Feature Set A — Raw (5 features)**

The five original reactor inputs.

**Feature Set B — Physics-informed (9 features) — FINAL**

Feature Set A + `res_time_proxy`, `temp_avg`, `jacket_x_restime`, `inlet_x_restime`.

**Feature Set C — Expanded (11 features)**

Feature Set B + `thermal_diff` and `conc_x_restime`.

Feature Set C is documented for comparison only and is **not** the final feature set.

## Models experimented with

The project experimentation covered:

1. ExtraTreesRegressor — **FINAL MODEL**
2. RandomForestRegressor
3. GradientBoostingRegressor
4. HistGradientBoostingRegressor
5. Random Forest + Boosting alternatives
6. XGBoost
7. SVR with RBF kernel

## Model comparison

The final experiment report supplied with the project provides the following best validation results:

| Model | Best Validation RMSE | Status |
|---|---:|---|
| **ExtraTreesRegressor** | **16.2273 ± 2.1423** | 🏆 **Selected** |
| RandomForestRegressor | 21.2544 ± 2.0361 | Tested |
| GradientBoostingRegressor | 17.0097 ± 2.3465 | Tested |
| HistGradientBoostingRegressor | 19.5199 | Tested |
| Random Forest + Boosting Alternatives | 17.0097 ± 2.3465 | Tested |
| XGBoost | 17.5208 ± 3.1676 | Tested |
| SVR — RBF Kernel | 22.6413 ± 1.6937 | Tested |

The **Random Forest + Boosting Alternatives** row represents the best result reported within that experimental category; it is not one single combined algorithm.

### ExtraTrees configurations

From the supplied report:

- Feature Set A: `18.7312 ± 2.1141`
- **Feature Set B: `16.2273 ± 2.1423`**
- Feature Set C: `25.1084 ± 1.2799`

Therefore the repository uses:

> **Final model = ExtraTreesRegressor**
>
> **Final feature set = Feature Set B**

These are local validation results and must not be presented as the official hidden leaderboard score.

## Why ExtraTreesRegressor was selected

ExtraTreesRegressor with Feature Set B achieved the strongest reported validation performance among the tested approaches. The selection is therefore based on the supplied local cross-validation report, not on an invented or assumed hidden score.

## Final model configuration

The supplied experiment code specifies the final ExtraTrees configuration as:

```text
n_estimators = 300
max_depth = 6
min_samples_leaf = 3
random_state = 42
n_jobs = -1
```

The final artifact is stored at:

`model/final_extratrees_model.joblib`

The original uploaded `final_model.joblib` workflow was CatBoost-based; because the submission instructions explicitly require ExtraTreesRegressor, that artifact was **not** used as the final model.

## Evaluation metric

The competition uses RMSE (Root Mean Squared Error). Lower is better.

## Reproduction

From the project root:

```bash
pip install -r requirements.txt
python src/final_extratrees_model.py
```

The pipeline:

1. Loads `data/train_dataset.csv` and `data/test_dataset.csv`.
2. Applies the shared Feature Set B preprocessing.
3. Trains the final ExtraTreesRegressor on all 150 training rows.
4. Generates 50 test predictions.
5. Saves the fitted model to `model/final_extratrees_model.joblib`.
6. Saves the only final submission file to `outputs/submission.csv`.
7. Writes `outputs/final_extratrees_report.json`.

## Final submission

The final prediction file is:

`outputs/submission.csv`

Before uploading to Unstop, rename it to:

`SynSquad.csv`

Only this single final submission CSV is included under `outputs/`.

## Feature-importance / process insight

The final pipeline records ExtraTrees feature importances in `outputs/final_extratrees_report.json`. These importances are model-derived and should be interpreted as predictive-process insights rather than causal proof of reactor chemistry.

## Validation caveats

- The reported RMSE values are local validation results from the supplied experiment report.
- No hidden leaderboard score is claimed.
- No test labels are used.
- No fold-level values are fabricated where only mean ± standard deviation was supplied.
- The supplied original CatBoost artifact was inspected as source evidence during development and is not included as the final model artifact.

## Repository structure

```text
reactor-yield-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── train_dataset.csv
│   └── test_dataset.csv
├── experiments/
│   ├── 01_extratrees.py
│   ├── 02_randomforest.py
│   ├── 03_gradientboosting.py
│   ├── 04_histgradientboosting.py
│   ├── 05_randomforest_boosting_alternatives.py
│   ├── 06_xgboost.py
│   ├── 07_svr_rbf.py
│   └── results/
│       └── cv_results_all_models.csv
├── src/
│   ├── preprocessing.py
│   └── final_extratrees_model.py
├── model/
│   └── final_extratrees_model.joblib
├── outputs/
│   ├── submission.csv
│   └── final_extratrees_report.json
├── notebooks/
│   └── final_workflow.ipynb
└── docs/
    └── ML_Hackathon_Problem_Statement_Final.pdf
```

## Team information

**Team:** SynSquad


