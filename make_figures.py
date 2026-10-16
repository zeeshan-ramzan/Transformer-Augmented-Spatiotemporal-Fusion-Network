"""
make_figures.py - Draws Figures 1-10 of the paper from the result files.

  fig1_bar_comparison.png      RMSE / MAE / MAPE of the top methods
  fig2_boxplot.png             per-fold RMSE of the top models
  fig3_histogram.png           observed vs predicted yield distributions (bin width 150 kg ha-1)
  fig4_scatter.png             predicted vs observed, best-fit and 1:1 lines
  fig5_residuals.png           residuals vs observation index
  fig6_residual_hist.png       residual distributions, TASTFN vs NAS-DNN
  fig7_learning_curves.png     mean RMSE per epoch over the 10 folds
  fig8_attention_heatmap.png   mean self-attention matrix of encoder layer 2
  fig9_feature_importance.png  permutation importance
  fig10_gate_activity.png      mean CMAG gate value by calendar month

Usage: python make_figures.py --results results --out figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TOP = [("gpr_rational_quadratic", "GPR (RQ)"), ("xgboost", "XGBoost"), ("gpr_exponential", "GPR (exp.)"),
       ("nas_dnn", "NAS-DNN"), ("tastfn", "TASTFN")]
UNIT = "kg ha$^{-1}$"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def load(res, name):
    f = os.path.join(res, name, "metrics.json")
    return json.load(open(f)) if os.path.isfile(f) else None


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=300)
    plt.close(fig)
    print("written", name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    avail = [(k, lab, load(a.results, k)) for k, lab in TOP]
    avail = [(k, lab, m) for k, lab, m in avail if m is not None]

    if avail:                                                            # Fig. 1
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
        for ax, met, unit in zip(axes, ["RMSE", "MAE", "MAPE"], [UNIT, UNIT, "%"]):
            vals = [m["mean"][met] for _, _, m in avail]
            ax.bar([lab for _, lab, _ in avail], vals, color=["#9aa5b1"] * (len(vals) - 1) + ["#1f5aa6"])
            ax.set_title(met)
            ax.set_ylabel(f"{met} ({unit})")
            ax.tick_params(axis="x", rotation=40)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        save(fig, a.out, "fig1_bar_comparison.png")

        fig, ax = plt.subplots(figsize=(6.5, 4))                         # Fig. 2
        ax.boxplot([[f["RMSE"] for f in m["folds"]] for _, _, m in avail])
        ax.set_xticks(range(1, len(avail) + 1), [lab for _, lab, _ in avail])
        ax.set_ylabel(f"RMSE ({UNIT})")
        ax.set_xlabel("Model")
        save(fig, a.out, "fig2_boxplot.png")

    pred_f = os.path.join(a.results, "tastfn", "oof_predictions.csv")
    if os.path.isfile(pred_f):
        d = pd.read_csv(pred_f)
        y, yh = d["yield"].to_numpy(), d["predicted"].to_numpy()
        bins = np.arange(np.floor(min(y.min(), yh.min()) / 150) * 150, max(y.max(), yh.max()) + 150, 150)
        fig, ax = plt.subplots(figsize=(6.5, 4))                         # Fig. 3
        ax.hist(y, bins=bins, alpha=0.6, label="Observed", color="#555555")
        ax.hist(yh, bins=bins, alpha=0.6, label="TASTFN predicted", color="#1f5aa6")
        ax.set_xlabel(f"Yield ({UNIT})")
        ax.set_ylabel("Frequency")
        ax.legend(frameon=False)
        save(fig, a.out, "fig3_histogram.png")

        slope, icpt = np.polyfit(y, yh, 1)                               # Fig. 4
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(y, yh, s=8, alpha=0.5, color="#1f5aa6")
        lim = [min(y.min(), yh.min()), max(y.max(), yh.max())]
        ax.plot(lim, lim, "k--", lw=1, label="1:1 line")
        ax.plot(lim, [slope * v + icpt for v in lim], color="#c0392b", lw=1.2,
                label=f"Fit: y = {slope:.4f}x + {icpt:.2f}")
        ax.set_xlabel(f"Observed yield ({UNIT})")
        ax.set_ylabel(f"Predicted yield ({UNIT})")
        ax.legend(frameon=False)
        save(fig, a.out, "fig4_scatter.png")

        fig, ax = plt.subplots(figsize=(7, 3.5))                         # Fig. 5
        ax.scatter(np.arange(len(d)), d["residual"], s=5, color="#1f5aa6")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("Observation index")
        ax.set_ylabel(f"Residual ({UNIT})")
        save(fig, a.out, "fig5_residuals.png")

        nas_f = os.path.join(a.results, "nas_dnn", "oof_predictions.csv")    # Fig. 6
        fig, ax = plt.subplots(figsize=(6.5, 4))
        r = d["residual"].to_numpy()
        series = [("TASTFN", r, "#1f5aa6")]
        if os.path.isfile(nas_f):
            series.append(("NAS-DNN", pd.read_csv(nas_f)["residual"].to_numpy(), "#c0392b"))
        lo = min(s[1].min() for s in series)
        hi = max(s[1].max() for s in series)
        for lab, rr, col in series:
            ax.hist(rr, bins=np.linspace(lo, hi, 40), alpha=0.55, label=lab, color=col)
        ax.set_xlabel(f"Residual ({UNIT})")
        ax.set_ylabel("Frequency")
        ax.legend(frameon=False)
        save(fig, a.out, "fig6_residual_hist.png")

    lc_f = os.path.join(a.results, "tastfn", "learning_curves.csv")         # Fig. 7
    if os.path.isfile(lc_f):
        lc = pd.read_csv(lc_f).groupby("epoch")[["test_fold_rmse", "val_rmse"]].mean()
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(lc.index, lc["test_fold_rmse"], color="#1f5aa6", label="TASTFN (held-out fold)")
        ax.plot(lc.index, lc["val_rmse"], color="#1f5aa6", ls=":", label="TASTFN (inner validation)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(f"Mean RMSE ({UNIT})")
        ax.set_yscale("log")
        ax.legend(frameon=False)
        save(fig, a.out, "fig7_learning_curves.png")

    att_f = os.path.join(a.results, "tastfn", "attention_layer2.npy")       # Fig. 8
    if os.path.isfile(att_f):
        A = np.load(att_f)
        T = A.shape[0]
        lab = [f"t-{T - 1 - i}" if i < T - 1 else "t" for i in range(T)]
        fig, ax = plt.subplots(figsize=(5, 4.3))
        im = ax.imshow(A, cmap="Blues", vmin=0)
        ax.set_xticks(range(T), lab)
        ax.set_yticks(range(T), lab)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        for i in range(T):
            for j in range(T):
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if A[i, j] > A.max() / 2 else "black")
        fig.colorbar(im, ax=ax, label="Attention weight")
        save(fig, a.out, "fig8_attention_heatmap.png")

    pi_f = os.path.join(a.results, "permutation_importance.csv")            # Fig. 9
    if os.path.isfile(pi_f):
        t = pd.read_csv(pi_f).sort_values("mean_rmse_increase")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.barh(t["feature"], t["mean_rmse_increase"], xerr=t["std_rmse_increase"], color="#1f5aa6")
        ax.set_xlabel(f"Mean RMSE increase ({UNIT})")
        save(fig, a.out, "fig9_feature_importance.png")

    g_f = os.path.join(a.results, "tastfn", "gates_by_month.csv")           # Fig. 10
    if os.path.isfile(g_f):
        g = pd.read_csv(g_f).sort_values("month")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.errorbar(g["month"], g["mean"], yerr=g["std"], marker="o", color="#1f5aa6", capsize=3)
        ax.set_xticks(range(1, 13), MONTHS)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Calendar month")
        ax.set_ylabel(r"Mean gate value $\alpha$")
        save(fig, a.out, "fig10_gate_activity.png")


if __name__ == "__main__":
    main()
