"""
ablation_study.py - Cumulative ablation of TASTFN (Table 5 of the paper).

Configurations (each adds one component to the previous one):
  1. Baseline MLP (3 layers)        - current month only
  2. + Temporal sequencing          - MLP on the flattened 6-month window
  3. + MHSA encoder (L = 1)         - no positional encoding, concatenation fusion, plain decoder
  4. + MHSA encoder (L = 2)
  5. + Phenological PE
  6. + CMAG module                  - replaces concatenation fusion
  7. + Residual decoder             - full TASTFN

Usage: python ablation_study.py --data data/nthri_tea_yield.csv [--epochs 500]
"""
import os

import pandas as pd

from common import base_parser
from tastfn import TASTFNConfig, add_common_args, run_cv

STEPS = [
    ("Baseline MLP (3 layers)", dict(model="mlp_flat")),
    ("+ Temporal sequencing", dict(model="mlp_seq")),
    ("+ MHSA encoder (L=1)", dict(model="tastfn", n_layers=1, pe="none", fusion="concat", residual_decoder=False)),
    ("+ MHSA encoder (L=2)", dict(model="tastfn", n_layers=2, pe="none", fusion="concat", residual_decoder=False)),
    ("+ Phenological PE", dict(model="tastfn", n_layers=2, pe="ppe", fusion="concat", residual_decoder=False)),
    ("+ CMAG module", dict(model="tastfn", n_layers=2, pe="ppe", fusion="cmag", residual_decoder=False)),
    ("+ Residual decoder", dict(model="tastfn", n_layers=2, pe="ppe", fusion="cmag", residual_decoder=True)),
]

if __name__ == "__main__":
    args = add_common_args(base_parser(__doc__)).parse_args()
    rows, prev = [], None
    for i, (label, kw) in enumerate(STEPS):
        cfg = TASTFNConfig(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size, **kw)
        s = run_cv(cfg, args, f"ablation/step{i + 1}", save_extras=False, verbose=False)["summary"]
        gain = None if prev is None else 100 * (prev - s["RMSE"]) / prev
        rows.append({"Configuration": label, "R2": s["R2"], "RMSE": s["RMSE"], "MAE": s["MAE"],
                     "MAPE": s["MAPE"], "RMSE_reduction_vs_previous_%": gain})
        prev = s["RMSE"]
    table = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "ablation_study.csv"), index=False)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
