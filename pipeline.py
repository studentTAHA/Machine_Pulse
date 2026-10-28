"""
pipeline.py
-----------
Glue code between the data and the neural network:
train / evaluate / save / load / predict / explain.
"""
import pickle

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ann_engine import NeuralNetwork
from data_utils import BASE_DIR, FEATURES, add_features

MODEL_PATH = BASE_DIR / "models" / "machinepulse_model.pkl"

WARNING_LEVEL = 0.25     # below  -> Healthy
CRITICAL_LEVEL = 0.50    # above  -> Critical (predicted failure)


def risk_level(p):
    if p < WARNING_LEVEL:
        return "Healthy"
    if p < CRITICAL_LEVEL:
        return "Warning"
    return "Critical"


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train_model(df, hidden_layers, learning_rate, epochs, batch_size,
                pos_weight, is_demo=False, progress_callback=None):
    data = add_features(df)
    X = data[FEATURES].to_numpy(dtype=float)
    y = data["failure"].to_numpy(dtype=int)

    # 80 % train+validation, 20 % untouched test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.15, stratify=y_trainval, random_state=42)

    scaler = StandardScaler().fit(X_train)           # learn mean/std on TRAIN only
    net = NeuralNetwork([len(FEATURES), *hidden_layers, 1], learning_rate=learning_rate)
    history = net.fit(
        scaler.transform(X_train), y_train,
        scaler.transform(X_val), y_val,
        epochs=epochs, batch_size=batch_size, pos_weight=pos_weight,
        progress_callback=progress_callback)

    proba = net.predict_proba(scaler.transform(X_test))
    pred = (proba >= CRITICAL_LEVEL).astype(int)

    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba),
    }
    bundle = {
        "net": net.to_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "features": FEATURES,
        "hidden_layers": list(hidden_layers),
        "parameters": net.count_parameters(),
        "metrics": metrics,
        "confusion": confusion_matrix(y_test, pred),
        "history": history,
        "test_size": int(len(y_test)),
        "trained_on_demo": bool(is_demo),
    }
    save_bundle(bundle)
    return bundle


# ----------------------------------------------------------------------------
# Save / load
# ----------------------------------------------------------------------------
def save_bundle(bundle):
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)


def load_bundle():
    if MODEL_PATH.exists():
        try:
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
def _scaled_matrix(bundle, df):
    data = add_features(df)
    X = data[bundle["features"]].to_numpy(dtype=float)
    return (X - bundle["scaler_mean"]) / bundle["scaler_scale"]


def predict_proba_df(bundle, df):
    net = NeuralNetwork.from_dict(bundle["net"])
    return net.predict_proba(_scaled_matrix(bundle, df))


def feature_influence(bundle, row_df):
    """How much does each feature push the risk up or down?
    We set one feature at a time to the dataset average and see how the
    failure probability changes."""
    net = NeuralNetwork.from_dict(bundle["net"])
    x = _scaled_matrix(bundle, row_df)[:1]
    base = float(net.predict_proba(x)[0])
    result = {}
    for i, name in enumerate(bundle["features"]):
        x_avg = x.copy()
        x_avg[0, i] = 0.0                       # 0 == average after scaling
        result[name] = base - float(net.predict_proba(x_avg)[0])
    return base, result