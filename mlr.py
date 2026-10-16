"""
mlr.py - Multiple linear regression (MLR) baseline.
Ordinary least squares on the ten standardised features, 10-fold CV.

Usage: python mlr.py --data data/nthri_tea_yield.csv --out-dir results
"""
from sklearn.linear_model import LinearRegression

from common import base_parser, run_cv_baseline


def build_model():
    return LinearRegression()


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_cv_baseline("mlr", build_model, args)
