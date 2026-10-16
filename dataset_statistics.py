"""
dataset_statistics.py - Descriptive statistics of the dataset (Table 2 of the paper):
minimum, maximum, mean, standard deviation and coefficient of variation of every feature.

Usage: python dataset_statistics.py --data data/nthri_tea_yield.csv
"""
import os

import pandas as pd

from common import FEATURE_LABELS, FEATURES, TARGET, base_parser, load_data

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    df = load_data(args.data)
    rows = []
    for c in [TARGET] + FEATURES:
        s = df[c]
        rows.append({"feature": FEATURE_LABELS.get(c, "Yield (kg ha-1)"), "min": s.min(), "max": s.max(),
                     "mean": s.mean(), "std": s.std(), "cv": s.std() / s.mean() if s.mean() else float("nan")})
    t = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    t.to_csv(os.path.join(args.out_dir, "dataset_statistics.csv"), index=False)
    print(f"n = {len(df)} observations, {df['farm_id'].nunique()} plots, "
          f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
