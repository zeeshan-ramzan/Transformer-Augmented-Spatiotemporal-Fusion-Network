"""
hyperparameter_sensitivity.py - One-at-a-time sensitivity of TASTFN (Table 11 of the paper).

Varied: context window T (3, 4, 5, 6, 8, 12), attention heads N_h (1, 2, 4, 8, 16),
encoder layers L (1, 2, 3, 4) and model dimension d_model (32, 64, 128, 256; d_ff = 4 d_model).
All other hyperparameters stay at their optimal values.

Usage: python hyperparameter_sensitivity.py --data data/nthri_tea_yield.csv [--only window]
"""
import os

import pandas as pd

from common import base_parser
from tastfn import TASTFNConfig, add_common_args, run_cv

GRID = {
    "window": [3, 4, 5, 6, 8, 12],
    "n_heads": [1, 2, 4, 8, 16],
    "n_layers": [1, 2, 3, 4],
    "d_model": [32, 64, 128, 256],
}

if __name__ == "__main__":
    p = add_common_args(base_parser(__doc__))
    p.add_argument("--only", choices=list(GRID), default=None, help="run a single hyperparameter")
    args = p.parse_args()
    rows = []
    for hp, values in GRID.items():
        if args.only and hp != args.only:
            continue
        for val in values:
            kw = {hp: val}
            if hp == "d_model":
                kw["d_ff"] = 4 * val
            cfg = TASTFNConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, **kw)
            s = run_cv(cfg, args, f"sensitivity/{hp}_{val}", save_extras=False, verbose=False)["summary"]
            rows.append({"hyperparameter": hp, "value": val, "RMSE": s["RMSE"], "R2": s["R2"]})
            print(f"{hp}={val}: RMSE={s['RMSE']:.2f}")
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "hyperparameter_sensitivity.csv"), index=False)
