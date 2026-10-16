"""
relieff_feature_selection.py - Feature ranking and selection (Methods, 'Feature ranking and selection').

1. RReliefF (the regression version of ReliefF; Robnik-Sikonja and Kononenko, 2003)
   scores every standardised feature from its k = 10 nearest neighbours of m sampled
   instances (default: all instances), with distance-weighted neighbours (sigma = 20).
2. Sequential forward selection: features are added one at a time (the one giving the
   largest RMSE reduction under 10-fold CV) until the RMSE stops decreasing.

Usage: python relieff_feature_selection.py --data data/nthri_tea_yield.csv [--wrapper rf|mlr]
"""
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from common import FEATURE_LABELS, FEATURES, TARGET, base_parser, load_data, stratified_folds


def rrelieff(X, y, k=10, sigma=20.0, m=None, seed=42):
    """Returns RReliefF weights (one per column of X)."""
    rng = np.random.default_rng(seed)
    n, p = X.shape
    Xn = (X - X.min(0)) / np.maximum(X.max(0) - X.min(0), 1e-12)
    yn = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    idx = np.arange(n) if m is None else rng.choice(n, size=min(m, n), replace=False)
    rank_w = np.exp(-((np.arange(1, k + 1) / sigma) ** 2))
    rank_w /= rank_w.sum()
    N_dC, N_dA, N_dCdA = 0.0, np.zeros(p), np.zeros(p)
    for i in idx:
        d = np.abs(Xn - Xn[i]).sum(1)
        d[i] = np.inf
        nb = np.argsort(d)[:k]
        dy = np.abs(yn[nb] - yn[i])
        dA = np.abs(Xn[nb] - Xn[i])
        N_dC += float((dy * rank_w).sum())
        N_dA += (dA * rank_w[:, None]).sum(0)
        N_dCdA += (dA * (dy * rank_w)[:, None]).sum(0)
    mm = len(idx)
    return N_dCdA / max(N_dC, 1e-12) - (N_dA - N_dCdA) / max(mm - N_dC, 1e-12)


def cv_rmse(X, y, folds, wrapper, seed):
    errs = []
    for tr, te in folds:
        s = StandardScaler().fit(X[tr])
        mdl = (RandomForestRegressor(200, n_jobs=-1, random_state=seed) if wrapper == "rf" else LinearRegression())
        mdl.fit(s.transform(X[tr]), y[tr])
        errs.append(np.sqrt(np.mean((mdl.predict(s.transform(X[te])) - y[te]) ** 2)))
    return float(np.mean(errs))


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--wrapper", choices=["rf", "mlr"], default="rf", help="model used for forward selection")
    p.add_argument("--k", type=int, default=10)
    args = p.parse_args()
    df = load_data(args.data)
    X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)
    w = rrelieff(X, y, k=args.k, seed=args.seed)
    rank = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in FEATURES], "relieff_weight": w}).sort_values(
        "relieff_weight", ascending=False)
    print(rank.to_string(index=False))
    folds = stratified_folds(y, args.folds, args.seed)
    chosen, remaining, history, best = [], list(range(len(FEATURES))), [], np.inf
    while remaining:
        scores = [(cv_rmse(X[:, chosen + [j]], y, folds, args.wrapper, args.seed), j) for j in remaining]
        r, j = min(scores)
        if r >= best - 1e-9:
            break
        best = r
        chosen.append(j)
        remaining.remove(j)
        history.append({"step": len(chosen), "added": FEATURE_LABELS[FEATURES[j]], "cv_rmse": r})
        print(f"step {len(chosen)}: + {FEATURES[j]:16s} CV RMSE = {r:.2f}")
    os.makedirs(args.out_dir, exist_ok=True)
    rank.to_csv(os.path.join(args.out_dir, "relieff_ranking.csv"), index=False)
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, "forward_selection.csv"), index=False)
    print("Selected features:", [FEATURES[j] for j in chosen])
