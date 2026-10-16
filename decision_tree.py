"""
decision_tree.py - Regression tree baseline (CART, minimum leaf size 4).

Usage: python decision_tree.py --data data/nthri_tea_yield.csv
"""
from sklearn.tree import DecisionTreeRegressor

from common import base_parser, run_cv_baseline


def build_model(seed=42):
    return DecisionTreeRegressor(min_samples_leaf=4, random_state=seed)


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_cv_baseline("decision_tree", lambda: build_model(args.seed), args)
