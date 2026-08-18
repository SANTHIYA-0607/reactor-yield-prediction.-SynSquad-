from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import engineer_features, FINAL_FEATURES, TARGET_COLUMN

root = Path(__file__).resolve().parents[1]
df = engineer_features(pd.read_csv(root / "data" / "train_dataset.csv"))
X, y = df[FINAL_FEATURES], df[TARGET_COLUMN]
cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
model = ExtraTreesRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=-1)
scores = np.sqrt(-cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=None))
print(f"ExtraTrees / Feature Set B RMSE: {scores.mean():.4f} ± {scores.std():.4f}")
