"""
ndvi_source_comparison.py - Sensitivity of TASTFN to the NDVI source (Table 10 of the paper).

  1. Landsat-8 OLI only     : ndvi_mean, ndvi_std, ndvi_min, ndvi_max
  2. Landsat-7/8 combined   : Landsat-7 ETM+ NDVI (columns l7_ndvi_*) before 2017, Landsat-8 from 2017
  3. NDVI mean only (1D)    : the NDVI stream contains ndvi_mean only

Usage: python ndvi_source_comparison.py --data data/nthri_tea_yield.csv
"""
import os

import pandas as pd

from common import NDVI_FEATURES, base_parser, load_data
from tastfn import TASTFNConfig, add_common_args, run_cv

if __name__ == "__main__":
    args = add_common_args(base_parser(__doc__)).parse_args()
    df = load_data(args.data)
    rows = []
    base = dict(epochs=args.epochs, patience=args.patience, batch_size=args.batch_size)
    s = run_cv(TASTFNConfig(**base), args, "ndvi_source/landsat8_only", df=df, save_extras=False, verbose=False)["summary"]
    rows.append({"configuration": "Landsat-8 OLI only", **s})
    l7 = ["l7_" + c for c in NDVI_FEATURES]
    if all(c in df.columns for c in l7):
        d2 = df.copy()
        early = d2["date"] < "2017-01-01"
        for c in NDVI_FEATURES:
            d2.loc[early, c] = d2.loc[early, "l7_" + c]
        s = run_cv(TASTFNConfig(**base), args, "ndvi_source/landsat7_8", df=d2, save_extras=False, verbose=False)["summary"]
        rows.append({"configuration": "Landsat-7/8 combined", **s})
    else:
        print("Columns l7_ndvi_* not found: skipping the Landsat-7/8 configuration.")
    s = run_cv(TASTFNConfig(ndvi_features=["ndvi_mean"], **base), args, "ndvi_source/mean_only", df=df,
               save_extras=False, verbose=False)["summary"]
    rows.append({"configuration": "NDVI mean only (1D)", **s})
    t = pd.DataFrame(rows)[["configuration", "R2", "RMSE", "MAE", "MAPE"]]
    os.makedirs(args.out_dir, exist_ok=True)
    t.to_csv(os.path.join(args.out_dir, "ndvi_source_comparison.csv"), index=False)
    print(t.to_string(index=False))
