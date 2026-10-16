"""
common.py
=========
Shared utilities used by every script in this package:

* loading the NTHRI tea yield table (CSV)
* stratified 10-fold cross-validation for a continuous target
  (folds are stratified on yield deciles)
* per-fold standardisation (fitted on the training part only)
* the five evaluation metrics of the paper (R2, RMSE, MSE, MAE, MAPE)
* inference-speed measurement in observations per second (OPS)
* a generic cross-validation runner for the scikit-learn style baselines

Expected CSV columns (one row = one monthly yield observation of one plot)
-------------------------------------------------------------------------
farm_id, date (YYYY-MM or YYYY-MM-DD),
min_temp, max_temp, mean_temp, humidity, rainfall, solar_radiation,
ndvi_mean, ndvi_std, ndvi_min, ndvi_max, yield
Optional: l7_ndvi_mean, l7_ndvi_std, l7_ndvi_min, l7_ndvi_max (Landsat-7 NDVI,
only needed by ndvi_source_comparison.py).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

AGRO_FEATURES = ["min_temp", "max_temp", "mean_temp", "humidity", "rainfall", "solar_radiation"]
NDVI_FEATURES = ["ndvi_mean", "ndvi_std", "ndvi_min", "ndvi_max"]
FEATURES = AGRO_FEATURES + NDVI_FEATURES
TARGET = "yield"
FEATURE_LABELS = {
    "min_temp": "Min. temperature", "max_temp": "Max. temperature", "mean_temp": "Mean temperature",
    "humidity": "Relative humidity", "rainfall": "Rainfall", "solar_radiation": "Solar radiation",
    "ndvi_mean": "NDVI mean", "ndvi_std": "NDVI std dev", "ndvi_min": "NDVI min", "ndvi_max": "NDVI max",
}


# ----------------------------------------------------------------------------- reproducibility
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ----------------------------------------------------------------------------- data
def load_data(path: str) -> pd.DataFrame:
    """Read the CSV, parse dates, sort by plot and date, drop incomplete rows."""
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Data file '{path}' not found. Create it with your NTHRI data (see README) or run "
            f"'python make_synthetic_dataset.py' to generate a test file.")
    df = pd.read_csv(path)
    missing = [c for c in ["farm_id", "date"] + FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df = df.dropna(subset=FEATURES + [TARGET]).sort_values(["farm_id", "date"]).reset_index(drop=True)
    return df


def stratified_folds(y: np.ndarray, n_splits: int = 10, seed: int = 42, n_bins: int = 10):
    """Stratified K-fold for a continuous target: strata are yield quantile bins."""
    bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), bins))


# ----------------------------------------------------------------------------- metrics
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    err = y_true - y_pred
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    nz = y_true != 0
    mape = float(100.0 * np.mean(np.abs(err[nz] / y_true[nz])))
    return {"R2": r2, "RMSE": float(np.sqrt(mse)), "MSE": mse, "MAE": float(np.mean(np.abs(err))), "MAPE": mape}


def mean_metrics(per_fold: List[Dict[str, float]]) -> Dict[str, float]:
    """The paper reports the mean of the ten fold-level metrics."""
    keys = per_fold[0].keys()
    return {k: float(np.mean([m[k] for m in per_fold])) for k in keys}


def format_metrics(m: Dict[str, float]) -> str:
    return (f"R2={m['R2']:.4f}  RMSE={m['RMSE']:.2f}  MSE={m['MSE']:.2f}  "
            f"MAE={m['MAE']:.2f}  MAPE={m['MAPE']:.2f}%")


def measure_ops(predict_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, repeats: int = 5) -> float:
    """Inference speed in observations per second (median of several runs)."""
    predict_fn(X[: min(len(X), 32)])  # warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_fn(X)
        times.append(time.perf_counter() - t0)
    return float(len(X) / max(np.median(times), 1e-9))


# ----------------------------------------------------------------------------- output
def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_predictions(df: pd.DataFrame, y_pred: np.ndarray, fold_id: np.ndarray, path: str) -> None:
    out = df[["farm_id", "date", "month", TARGET]].copy()
    out["predicted"] = y_pred
    out["fold"] = fold_id
    out["residual"] = out[TARGET] - out["predicted"]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out.to_csv(path, index=False)


# ----------------------------------------------------------------------------- CLI
def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--data", default="data/nthri_tea_yield.csv", help="CSV file with the dataset")
    p.add_argument("--out-dir", default="results", help="output folder")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p


# ----------------------------------------------------------------------------- baseline runner
def run_cv_baseline(name: str, model_factory: Callable[[], object], args,
                    features: Sequence[str] = FEATURES, scale_y: bool = False,
                    fit_kwargs: Optional[dict] = None) -> Dict[str, float]:
    """10-fold CV for a scikit-learn style regressor on the flat 10-feature vector.

    Features are standardised inside each fold (mean/std of the training part only),
    exactly as described in the Methods section.
    """
    set_seed(args.seed)
    df = load_data(args.data)
    X = df[list(features)].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    folds = stratified_folds(y, args.folds, args.seed)
    oof = np.zeros(len(y))
    fold_id = np.zeros(len(y), dtype=int)
    per_fold, ops, t_train = [], [], 0.0
    for k, (tr, te) in enumerate(folds):
        sx = StandardScaler().fit(X[tr])
        Xtr, Xte = sx.transform(X[tr]), sx.transform(X[te])
        ytr = y[tr]
        sy = None
        if scale_y:
            sy = StandardScaler().fit(ytr.reshape(-1, 1))
            ytr = sy.transform(ytr.reshape(-1, 1)).ravel()
        model = model_factory()
        t0 = time.perf_counter()
        model.fit(Xtr, ytr, **(fit_kwargs or {}))
        t_train += time.perf_counter() - t0

        def predict(Z, model=model, sy=sy):
            p = np.asarray(model.predict(Z)).ravel()
            return sy.inverse_transform(p.reshape(-1, 1)).ravel() if sy is not None else p

        pred = predict(Xte)
        oof[te], fold_id[te] = pred, k
        m = regression_metrics(y[te], pred)
        per_fold.append(m)
        ops.append(measure_ops(predict, Xte))
        print(f"[{name}] fold {k + 1:2d}/{len(folds)}  {format_metrics(m)}")
    summary = mean_metrics(per_fold)
    summary.update({"OPS": float(np.median(ops)), "train_time_s": t_train})
    print(f"[{name}] MEAN  {format_metrics(summary)}  OPS={summary['OPS']:.0f}  train={t_train:.1f}s")
    out = os.path.join(args.out_dir, name)
    save_json({"model": name, "mean": summary, "folds": per_fold}, os.path.join(out, "metrics.json"))
    save_predictions(df, oof, fold_id, os.path.join(out, "oof_predictions.csv"))
    return summary
