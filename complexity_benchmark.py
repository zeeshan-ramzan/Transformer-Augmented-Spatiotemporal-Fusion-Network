"""
complexity_benchmark.py - Computational complexity table (Table 8 of the paper).

Collects parameters, FLOPs per observation, training time (one 10-fold CV run) and
inference speed (OPS) from the metrics.json files written by the model scripts, and
prints the analytic parameter/FLOP counts of TASTFN and NAS-DNN.

Usage: python complexity_benchmark.py --results results
"""
import argparse
import json
import os

import pandas as pd

from nas_dnn import PUBLISHED, count_flops as nas_flops, count_parameters as nas_params
from tastfn import TASTFNConfig, build_model, count_flops, count_parameters

MODELS = [("Decision tree", "decision_tree"), ("SVR (Gaussian)", "svr_gaussian"), ("GPR (exponential)", "gpr_exponential"),
          ("Random forest", "random_forest"), ("XGBoost", "xgboost"), ("NAS-DNN", "nas_dnn"), ("TASTFN", "tastfn")]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    a = p.parse_args()
    cfg = TASTFNConfig()
    analytic = {"NAS-DNN": (nas_params(PUBLISHED["layers"]), nas_flops(PUBLISHED["layers"])),
                "TASTFN": (count_parameters(build_model(cfg)), count_flops(cfg))}
    rows = []
    for label, folder in MODELS:
        f = os.path.join(a.results, folder, "metrics.json")
        m = json.load(open(f))["mean"] if os.path.isfile(f) else {}
        par, fl = analytic.get(label, (None, None))
        rows.append({"model": label, "parameters": par, "FLOPs": fl,
                     "train_time_s": m.get("train_time_s"), "OPS": m.get("OPS")})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(a.results, "complexity_table.csv"), index=False)
    print(t.to_string(index=False))
