"""
make_synthetic_dataset.py
=========================
Writes a SYNTHETIC table with exactly the same columns as the NTHRI dataset so that
every script can be tested end to end without the real (restricted) data.
The numbers it produces are NOT the NTHRI data and must not be reported.

Yield is generated from lagged weather (one- and two-month lags), NDVI and season,
so the temporal models have a real signal to learn.

Usage:  python make_synthetic_dataset.py --out data/nthri_tea_yield.csv --n 1800
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/nthri_tea_yield.csv")
    p.add_argument("--n", type=int, default=1800, help="number of observations to keep")
    p.add_argument("--plots", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    dates = pd.date_range("2015-01-01", "2025-12-01", freq="MS")
    rows = []
    for plot in range(1, a.plots + 1):
        plot_eff = rng.normal(0, 25)
        prev = [None, None]
        for d in dates:
            m = d.month
            season = np.cos(2 * np.pi * (m - 7) / 12)                 # +1 in July, -1 in January
            mean_t = 19.6 + 7.5 * season + rng.normal(0, 1.2)
            min_t = mean_t - 5.8 + rng.normal(0, 0.8)
            max_t = mean_t + 5.8 + rng.normal(0, 0.8)
            hum = np.clip(68 + 14 * season + rng.normal(0, 6), 41, 92)
            rain = max(0.0, rng.gamma(2.0, 49.0) * (1 + 0.8 * season))
            solar = np.clip(196 + 70 * np.cos(2 * np.pi * (m - 6) / 12) + rng.normal(0, 15), 82, 318)
            ndvi_mean = np.clip(0.48 + 0.15 * season + 0.0004 * (rain - 98) + rng.normal(0, 0.03), 0.21, 0.74)
            ndvi_std = np.clip(rng.normal(0.07, 0.02), 0.02, 0.18)
            ndvi_min = np.clip(ndvi_mean - 2.2 * ndvi_std + rng.normal(0, 0.02), 0.08, 0.61)
            ndvi_max = np.clip(ndvi_mean + 2.1 * ndvi_std + rng.normal(0, 0.02), 0.34, 0.88)
            lag1 = prev[0] if prev[0] is not None else (rain, mean_t)
            lag2 = prev[1] if prev[1] is not None else (rain, mean_t)
            yld = (60 + 1.05 * rain + 0.45 * lag1[0] + 0.20 * lag2[0] + 520 * (ndvi_mean - 0.3)
                   + 6.0 * (min_t - 10) + 1.2 * (hum - 60) + plot_eff + rng.normal(0, 6))
            yld = float(np.clip(yld, 85, 1240))
            rows.append({"farm_id": f"P{plot:02d}", "date": d.strftime("%Y-%m"),
                         "min_temp": round(min_t, 1), "max_temp": round(max_t, 1), "mean_temp": round(mean_t, 1),
                         "humidity": round(hum, 1), "rainfall": round(rain, 1), "solar_radiation": round(solar, 1),
                         "ndvi_mean": round(ndvi_mean, 3), "ndvi_std": round(ndvi_std, 3),
                         "ndvi_min": round(ndvi_min, 3), "ndvi_max": round(ndvi_max, 3),
                         # Landsat-7 NDVI (slightly different radiometry), for ndvi_source_comparison.py
                         "l7_ndvi_mean": round(ndvi_mean * 0.97 + rng.normal(0, 0.015), 3),
                         "l7_ndvi_std": round(ndvi_std * 1.05, 3),
                         "l7_ndvi_min": round(ndvi_min * 0.97 + rng.normal(0, 0.015), 3),
                         "l7_ndvi_max": round(ndvi_max * 0.97 + rng.normal(0, 0.015), 3),
                         "yield": round(yld, 1)})
            prev = [(rain, mean_t), lag1]
    df = pd.DataFrame(rows)
    if a.n < len(df):
        df = df.sample(n=a.n, random_state=a.seed).sort_values(["farm_id", "date"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Synthetic dataset with {len(df)} rows written to {a.out}")


if __name__ == "__main__":
    main()
