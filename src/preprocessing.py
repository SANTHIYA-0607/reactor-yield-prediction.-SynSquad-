from __future__ import annotations

from pathlib import Path
import pandas as pd

RAW_FEATURES = [
    "flow_rate_L_min",
    "concentration_mol_L",
    "inlet_temperature_K",
    "length_m",
    "jacket_temperature_K",
]
FINAL_FEATURES = RAW_FEATURES + [
    "res_time_proxy",
    "temp_avg",
    "jacket_x_restime",
    "inlet_x_restime",
]
TARGET_COLUMN = "overall_yield"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the final Feature Set B engineering consistently to any split."""
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw features: {missing}")
    result = df.copy()
    if (result["flow_rate_L_min"] == 0).any():
        raise ValueError("flow_rate_L_min contains zero; residence-time proxy is undefined.")
    result["res_time_proxy"] = result["length_m"] / result["flow_rate_L_min"]
    result["temp_avg"] = (result["inlet_temperature_K"] + result["jacket_temperature_K"]) / 2.0
    result["jacket_x_restime"] = result["jacket_temperature_K"] * result["res_time_proxy"]
    result["inlet_x_restime"] = result["inlet_temperature_K"] * result["res_time_proxy"]
    return result


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]
