"""
Load shared artifacts and trained models for the Streamlit app.

Uses Streamlit cache so models / test data are not reloaded on every click.
"""

import glob
import json
import os
import pickle

import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from prototype.config import (
    MODELS_DIR,
    NON_MODEL_PICKLES,
    PROCESSED_DIR,
    RESULTS_DIR,
)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_first_existing(paths):
    """Return (object, path) for the first path that exists."""
    for path in paths:
        if os.path.exists(path):
            return load_pickle(path), path
    raise FileNotFoundError(f"None of these files exist: {paths}")


def load_scaler():
    return load_first_existing(
        [
            os.path.join(PROCESSED_DIR, "scaler.pkl"),
            os.path.join(MODELS_DIR, "scaler.pkl"),
        ]
    )[0]


def load_feature_columns():
    return load_first_existing(
        [
            os.path.join(PROCESSED_DIR, "feature_columns.pkl"),
            os.path.join(MODELS_DIR, "feature_columns.pkl"),
        ]
    )[0]


def load_best_model_name():
    path = os.path.join(MODELS_DIR, "best_model_name.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return f.read().strip()


def list_model_names():
    """Names of trained member models in models/ (without .pkl)."""
    files = sorted(
        p
        for p in glob.glob(os.path.join(MODELS_DIR, "*.pkl"))
        if os.path.basename(p) not in NON_MODEL_PICKLES
    )
    return [os.path.splitext(os.path.basename(p))[0] for p in files]


def has_test_set() -> bool:
    return os.path.exists(os.path.join(PROCESSED_DIR, "X_test.csv"))


@st.cache_resource
def load_model(name: str):
    return load_pickle(os.path.join(MODELS_DIR, f"{name}.pkl"))


@st.cache_data
def load_model_threshold(name: str) -> float:
    """Custom decision threshold from training metrics JSON, else 0.5."""
    path = os.path.join(RESULTS_DIR, f"{name}_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f).get("threshold", 0.5)
    return 0.5


@st.cache_data
def load_test_set():
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).squeeze()
    return X_test, y_test


@st.cache_data(show_spinner=False)
def score_model(name: str, _model, X_test, y_test):
    """Score one model on the held-out test set (live metrics for Insights tab)."""
    y_prob = _model.predict_proba(X_test.astype(float))[:, 1]
    threshold = load_model_threshold(name)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }
    return y_pred, y_prob, metrics
