"""
seasonal_analysis.py - TASTFN performance by growing season (Table 9 of the paper).

Reads the out-of-fold predictions written by tastfn.py and computes the metrics for
spring flush (Mar-May), summer season (Jun-Sep), autumn flush (Oct-Nov) and, if present,
winter months (Dec-Feb), plus the overall values.

Usage: python seasonal_analysis.py --pred results/tastfn/oof_predictions.csv
"""
import argparse
import os

import pandas as pd

from common import TARGET, regression_metrics

SEASONS = {"Spring flush (Mar-May)": [3, 4, 5], "Summer season (Jun-Sep)": [6, 7, 8, 9],
           "Autumn flush (Oct-Nov)": [10, 11], "Winter (Dec-Feb)": [12, 1, 2]}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", default="results/tastfn/oof_predictions.csv")
    p.add_argument("--out", default="results/seasonal_analysis.csv")
    a = p.parse_args()
    d = pd.read_csv(a.pred)
    rows = []
    for name, months in SEASONS.items():
        s = d[d["month"].isin(months)]
        if len(s) > 1:
            rows.append({"season": name, "n": len(s), **regression_metrics(s[TARGET], s["predicted"])})
    rows.append({"season": "Overall (pooled)", "n": len(d), **regression_metrics(d[TARGET], d["predicted"])})
    t = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t.to_csv(a.out, index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
