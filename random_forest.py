"""
random_forest.py - Random forest regression (500 trees, bootstrap aggregation).

Usage: python random_forest.py --data data/nthri_tea_yield.csv [--trees 500]
"""
from sklearn.ensemble import RandomForestRegressor

from common import base_parser, run_cv_baseline


def build_model(n_trees=500, seed=42):
    return RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=1, max_features=1.0,
                                 n_jobs=-1, random_state=seed)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--trees", type=int, default=500)
    args = p.parse_args()
    run_cv_baseline("random_forest", lambda: build_model(args.trees, args.seed), args)
