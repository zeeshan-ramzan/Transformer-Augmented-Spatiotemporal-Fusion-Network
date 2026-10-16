"""
nas_dnn.py - Deep regression network designed by neural architecture search (NAS-DNN),
the previously published benchmark on the NTHRI data (Ramzan et al., IEEE Access 2023).

Two modes
---------
1. Default: train the published architecture with 10-fold CV
   (3 fully connected layers of 188, 29 and 77 units, ReLU, L2 strength 2.24e-6,
   standardised inputs, linear output, L-BFGS solver, max. 1000 iterations,
   loss and step tolerance 1e-6).
2. --search: run the neural architecture search first. The search space is
   number of layers (1-3), layer sizes (1-300, log scale), activation
   (relu / tanh / sigmoid / none) and L2 strength (log scale), explored by
   Bayesian optimisation (Gaussian-process surrogate + expected improvement).
   The objective is log(1 + 10-fold CV MSE), as in the original study, evaluated
   for --evals configurations (100 in the paper). The best architecture is saved
   to results/nas_dnn/best_architecture.json and then evaluated with 10-fold CV.

Usage:
  python nas_dnn.py --data data/nthri_tea_yield.csv
  python nas_dnn.py --data data/nthri_tea_yield.csv --search --evals 100
"""
import json
import os
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from common import FEATURES, TARGET, base_parser, load_data, run_cv_baseline, save_json, set_seed, stratified_folds

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PUBLISHED = {"layers": [188, 29, 77], "activation": "relu", "lambda": 2.24e-6}
ACTIVATIONS = ["relu", "tanh", "logistic", "identity"]          # logistic = sigmoid, identity = none


def build_model(arch=PUBLISHED, n_train=1620, seed=42, max_iter=1000):
    # MATLAB's Lambda penalises the mean loss; scikit-learn's alpha is divided by n,
    # hence alpha = lambda * n_train gives the same effective penalty.
    return MLPRegressor(hidden_layer_sizes=tuple(arch["layers"]), activation=arch["activation"],
                        solver="lbfgs", alpha=arch["lambda"] * n_train, max_iter=max_iter, tol=1e-6,
                        max_fun=15000 * 4, random_state=seed)


def count_parameters(layers, n_in=len(FEATURES)):
    sizes = [n_in] + list(layers) + [1]
    return int(sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:])))


def count_flops(layers, n_in=len(FEATURES)):
    sizes = [n_in] + list(layers) + [1]
    return int(sum(2 * a * b for a, b in zip(sizes[:-1], sizes[1:])))   # 1 MAC = 2 FLOPs


# ----------------------------------------------------------------------------- search space
def decode(u):
    """u in [0,1]^6 -> architecture."""
    n_layers = 1 + min(int(u[0] * 3), 2)
    sizes = [int(round(np.exp(np.log(1) + u[1 + i] * (np.log(300) - np.log(1))))) for i in range(3)]
    act = ACTIVATIONS[min(int(u[4] * 4), 3)]
    lam = float(10 ** (-8 + u[5] * 7))                   # 1e-8 ... 1e-1
    return {"layers": [max(1, s) for s in sizes[:n_layers]], "activation": act, "lambda": lam}


def cv_objective(arch, X, y, folds, seed):
    mses = []
    for tr, te in folds:
        sx = StandardScaler().fit(X[tr])
        sy = StandardScaler().fit(y[tr].reshape(-1, 1))
        m = build_model(arch, len(tr), seed, max_iter=400)
        m.fit(sx.transform(X[tr]), sy.transform(y[tr].reshape(-1, 1)).ravel())
        pred = sy.inverse_transform(m.predict(sx.transform(X[te])).reshape(-1, 1)).ravel()
        mses.append(np.mean((y[te] - pred) ** 2))
    return float(np.log1p(np.mean(mses)))


def bayesian_nas(X, y, folds, n_evals=100, n_init=10, seed=42):
    rng = np.random.default_rng(seed)
    U, F = [], []
    for i in range(n_evals):
        if i < n_init:
            u = rng.random(6)
        else:
            gp = GaussianProcessRegressor(ConstantKernel() * Matern(nu=2.5, length_scale=np.ones(6)) + WhiteKernel(1e-3),
                                          normalize_y=True, random_state=seed).fit(np.array(U), np.array(F))
            cand = rng.random((3000, 6))
            mu, sd = gp.predict(cand, return_std=True)
            best = min(F)
            z = (best - mu) / np.maximum(sd, 1e-9)
            ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
            u = cand[int(np.argmax(ei))]
        arch = decode(u)
        f = cv_objective(arch, X, y, folds, seed)
        U.append(u)
        F.append(f)
        print(f"[NAS] eval {i + 1:3d}/{n_evals}  objective={f:.4f}  best={min(F):.4f}  arch={arch}")
    best = decode(U[int(np.argmin(F))])
    return best, float(min(F)), [{"objective": f, **decode(u)} for u, f in zip(U, F)]


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--search", action="store_true", help="run Bayesian-optimisation NAS first")
    p.add_argument("--evals", type=int, default=100)
    args = p.parse_args()
    set_seed(args.seed)
    arch = PUBLISHED
    out = os.path.join(args.out_dir, "nas_dnn")
    if args.search:
        df = load_data(args.data)
        X, y = df[FEATURES].to_numpy(float), df[TARGET].to_numpy(float)
        arch, best_obj, history = bayesian_nas(X, y, stratified_folds(y, args.folds, args.seed), args.evals, seed=args.seed)
        save_json({"best_architecture": arch, "min_objective_log1p_mse": best_obj, "history": history},
                  os.path.join(out, "best_architecture.json"))
        print("Best architecture:", json.dumps(arch))
    n_train = int(len(load_data(args.data)) * (args.folds - 1) / args.folds)
    summary = run_cv_baseline("nas_dnn", lambda: build_model(arch, n_train, args.seed), args, scale_y=True)
    info = {"architecture": arch, "parameters": count_parameters(arch["layers"]),
            "flops_per_observation": count_flops(arch["layers"])}
    save_json(info, os.path.join(out, "complexity.json"))
    print(f"[nas_dnn] parameters={info['parameters']}  FLOPs/observation={info['flops_per_observation']}")
