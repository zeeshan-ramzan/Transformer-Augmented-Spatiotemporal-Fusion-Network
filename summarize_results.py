"""
summarize_results.py - Collects the mean 10-fold CV metrics of all models into one
table (Table 4 of the paper) and writes results/comparison_table.csv.

Usage: python summarize_results.py --results results
"""
import argparse
import json
import os

import pandas as pd

ORDER = [("Conventional regression", "MLR", "mlr"), ("Conventional regression", "MLR with interactions", "mlr_interactions"),
         ("Conventional regression", "Stepwise MLR", "stepwise_mlr"), ("Conventional regression", "Decision tree", "decision_tree"),
         ("Conventional regression", "SVR (linear)", "svr_linear"), ("Conventional regression", "SVR (quadratic)", "svr_quadratic"),
         ("Conventional regression", "SVR (cubic)", "svr_cubic"), ("Conventional regression", "SVR (Gaussian)", "svr_gaussian"),
         ("GPR variants", "GPR (rational quadratic)", "gpr_rational_quadratic"),
         ("GPR variants", "GPR (squared exponential)", "gpr_squared_exponential"),
         ("GPR variants", "GPR (exponential)", "gpr_exponential"), ("Ensemble methods", "Random forest", "random_forest"),
         ("Ensemble methods", "Gradient boosting", "gradient_boosting"), ("Ensemble methods", "XGBoost", "xgboost"),
         ("NAS-DNN", "Deep regression", "nas_dnn"), ("Proposed", "TASTFN", "tastfn")]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    a = p.parse_args()
    rows = []
    for cat, lab, folder in ORDER:
        f = os.path.join(a.results, folder, "metrics.json")
        if os.path.isfile(f):
            m = json.load(open(f))["mean"]
            rows.append({"category": cat, "model": lab, **{k: m[k] for k in ["R2", "RMSE", "MSE", "MAE", "MAPE", "OPS"]}})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.results, "comparison_table.csv"), index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
