"""
permutation_importance.py - Permutation feature importance of TASTFN (Fig. 9 of the paper).

TASTFN is trained in each of the 10 folds; then, for every feature, its values are
shuffled among the held-out observations (50 repeats), the windows are rebuilt and
the increase in RMSE is recorded. The score is the mean RMSE increase (kg ha-1).

Usage: python permutation_importance.py --data data/nthri_tea_yield.csv [--repeats 50]
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common import FEATURE_LABELS, base_parser, load_data, set_seed, stratified_folds
from tastfn import TASTFNConfig, add_common_args, predict, prepare_data, train_fold, window_tensors

if __name__ == "__main__":
    p = add_common_args(base_parser(__doc__))
    p.add_argument("--repeats", type=int, default=50)
    args = p.parse_args()
    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg = TASTFNConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
    feats = cfg.agro_features + cfg.ndvi_features
    df = load_data(args.data)
    X, y, months, win = prepare_data(df, cfg)
    rng = np.random.default_rng(args.seed)
    inc = {f: [] for f in feats}
    for k, (tr_all, te) in enumerate(stratified_folds(y, args.folds, args.seed)):
        sx = StandardScaler().fit(X[tr_all])
        Xs = sx.transform(X)
        sy = StandardScaler().fit(y[tr_all].reshape(-1, 1))
        y_s = {"raw": y, "scaled": sy.transform(y.reshape(-1, 1)).ravel(),
               "inverse": lambda q, sy=sy: sy.inverse_transform(np.asarray(q).reshape(-1, 1)).ravel()}
        tr, val = (tr_all, te) if args.early_stop_on_test else train_test_split(tr_all, test_size=0.1, random_state=args.seed + k)
        model, _, (xte, mte, vte) = train_fold(cfg, Xs, months, win, y_s, tr, val, te, device, args.seed + k, f"fold {k + 1}")
        base = np.sqrt(np.mean((y_s["inverse"](predict(model, xte, mte, vte)["y"]) - y[te]) ** 2))
        for j, f in enumerate(feats):
            for _ in range(args.repeats):
                Xp = Xs.copy()
                Xp[te, j] = Xs[rng.permutation(te), j]
                xp, mp, vp = window_tensors(Xp, months, win, te, device)
                r = np.sqrt(np.mean((y_s["inverse"](predict(model, xp, mp, vp)["y"]) - y[te]) ** 2))
                inc[f].append(r - base)
        print(f"fold {k + 1}: base RMSE={base:.2f}")
    t = pd.DataFrame({"feature": [FEATURE_LABELS[f] for f in feats],
                      "mean_rmse_increase": [np.mean(inc[f]) for f in feats],
                      "std_rmse_increase": [np.std(inc[f]) for f in feats]}).sort_values("mean_rmse_increase", ascending=False)
    os.makedirs(args.out_dir, exist_ok=True)
    t.to_csv(os.path.join(args.out_dir, "permutation_importance.csv"), index=False)
    print(t.to_string(index=False))
