"""
gradient_boosting.py - Gradient boosting regression trees (least-squares loss).

Usage: python gradient_boosting.py --data data/nthri_tea_yield.csv
"""
from sklearn.ensemble import GradientBoostingRegressor

from common import base_parser, run_cv_baseline


def build_model(n_estimators=500, lr=0.05, depth=3, seed=42):
    return GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=depth,
                                     subsample=0.8, random_state=seed)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=3)
    args = p.parse_args()
    run_cv_baseline("gradient_boosting", lambda: build_model(args.n_estimators, args.lr, args.depth, args.seed), args)
