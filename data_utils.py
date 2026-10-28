"""
data_utils.py
-------------
Everything about the data: loading the AI4I 2020 CSV, cleaning column names,
creating extra "physics" features, and a small demo-data generator that is used
only when the real CSV is not found.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

# Features that go into the neural network (order matters!)
FEATURES = [
    "type_code", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
    "temp_diff", "power_w", "strain",
]

INPUT_COLUMNS = ["type", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]
REQUIRED_COLUMNS = INPUT_COLUMNS + ["failure"]

LABELS = {
    "air_temp": "Air temperature [K]",
    "process_temp": "Process temperature [K]",
    "rpm": "Rotational speed [rpm]",
    "torque": "Torque [Nm]",
    "tool_wear": "Tool wear [min]",
    "temp_diff": "Temperature difference [K]",
    "power_w": "Mechanical power [W]",
    "strain": "Strain (wear x torque)",
    "type_code": "Product quality type",
}


# ----------------------------------------------------------------------------
# Column names: "Air temperature [K]"  ->  "air_temp"
# ----------------------------------------------------------------------------
def standardize_columns(df):
    rename = {}
    for col in df.columns:
        key = re.sub(r"\[.*?\]", "", str(col)).strip().lower().replace(" ", "").replace("_", "")
        if key in ("type", "producttype"):
            rename[col] = "type"
        elif key.startswith("airtemp"):
            rename[col] = "air_temp"
        elif key.startswith("processtemp"):
            rename[col] = "process_temp"
        elif key.startswith("rotational") or key in ("rpm", "speed"):
            rename[col] = "rpm"
        elif key == "torque":
            rename[col] = "torque"
        elif key.startswith("toolwear"):
            rename[col] = "tool_wear"
        elif key in ("machinefailure", "failure"):
            rename[col] = "failure"
    return df.rename(columns=rename)


def missing_columns(df, needed):
    return [c for c in needed if c not in df.columns]


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
def add_features(df):
    out = df.copy()
    out["type_code"] = out["type"].astype(str).str.strip().str.upper().map(
        {"L": 0, "M": 1, "H": 2}).fillna(0).astype(float)
    out["temp_diff"] = out["process_temp"] - out["air_temp"]
    out["power_w"] = out["torque"] * out["rpm"] * 2 * np.pi / 60.0
    out["strain"] = out["tool_wear"] * out["torque"]
    return out


# ----------------------------------------------------------------------------
# Demo data (only used when data/ai4i2020.csv is missing)
# ----------------------------------------------------------------------------
def make_demo_dataset(n=10000, seed=7):
    rng = np.random.default_rng(seed)
    kind = rng.choice(["L", "M", "H"], size=n, p=[0.5, 0.3, 0.2])
    air = 300 + rng.normal(0, 2.0, n)
    proc = air + 10 + rng.normal(0, 0.9, n)
    torque = np.clip(rng.normal(40, 10, n), 4, 77).round(1)
    rpm = np.clip(1538 - 15 * (torque - 40) + rng.normal(0, 98, n), 1170, 2880).round()
    wear = np.clip(rng.normal(108, 64, n), 0, 253).astype(int)

    power = torque * rpm * 2 * np.pi / 60
    strain = wear * torque
    limit = np.where(kind == "L", 11000, np.where(kind == "M", 12000, 13000))

    twf = (wear >= 200) & (wear <= 240) & (rng.random(n) < 0.09)
    hdf = ((proc - air) < 8.6) & (rpm < 1380)
    pwf = (power < 3500) | (power > 9000)
    osf = strain > limit
    rnf = rng.random(n) < 0.001
    failure = (twf | hdf | pwf | osf | rnf).astype(int)

    return pd.DataFrame({
        "UDI": np.arange(1, n + 1),
        "type": kind,
        "air_temp": air.round(1),
        "process_temp": proc.round(1),
        "rpm": rpm.astype(int),
        "torque": torque,
        "tool_wear": wear,
        "failure": failure,
        "TWF": twf.astype(int), "HDF": hdf.astype(int), "PWF": pwf.astype(int),
        "OSF": osf.astype(int), "RNF": rnf.astype(int),
    })


def load_dataset():
    """Returns (dataframe, is_demo). Uses the real CSV if it exists."""
    if DATA_PATH.exists():
        df = standardize_columns(pd.read_csv(DATA_PATH))
        missing = missing_columns(df, REQUIRED_COLUMNS)
        if missing:
            raise ValueError(f"ai4i2020.csv is missing these columns: {missing}")
        return df, False
    return make_demo_dataset(), True