"""
xgboost_model.py - Extreme gradient boosting (XGBoost) regression.

Usage: python xgboost_model.py --data data/nthri_tea_yield.csv
"""
from xgboost import XGBRegressor

from common import base_parser, run_cv_baseline


def build_model(n_estimators=600, lr=0.05, depth=4, seed=42):
    return XGBRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=depth, subsample=0.8,
                        colsample_bytree=0.9, reg_lambda=1.0, objective="reg:squarederror",
                        tree_method="hist", n_jobs=-1, random_state=seed)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--n-estimators", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--depth", type=int, default=4)
    args = p.parse_args()
    run_cv_baseline("xgboost", lambda: build_model(args.n_estimators, args.lr, args.depth, args.seed), args)
