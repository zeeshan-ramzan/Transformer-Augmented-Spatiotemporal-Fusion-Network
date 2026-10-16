"""
correlation_analysis.py - Pearson correlation analysis (Methods, 'Correlation analysis'):
feature-feature correlation matrix and correlation of each feature with yield.
Writes CSV files and a heatmap (correlation_matrix.png).

Usage: python correlation_analysis.py --data data/nthri_tea_yield.csv
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FEATURE_LABELS, FEATURES, TARGET, base_parser, load_data

if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    df = load_data(args.data)
    cols = FEATURES + [TARGET]
    C = df[cols].corr(method="pearson")
    labels = [FEATURE_LABELS.get(c, "Yield") for c in cols]
    C.index, C.columns = labels, labels
    os.makedirs(args.out_dir, exist_ok=True)
    C.to_csv(os.path.join(args.out_dir, "correlation_matrix.csv"))
    with_y = C["Yield"].drop("Yield").sort_values(ascending=False)
    with_y.to_csv(os.path.join(args.out_dir, "correlation_with_yield.csv"), header=["r"])
    print(with_y.to_string(float_format=lambda v: f"{v:.2f}"))
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(C.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{C.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "correlation_matrix.png"), dpi=300)
