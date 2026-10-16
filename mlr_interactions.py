"""
mlr_interactions.py - Multiple linear regression with pairwise interaction terms.
The design matrix contains the ten features and all 45 pairwise products.

Usage: python mlr_interactions.py --data data/nthri_tea_yield.csv
"""
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from common import base_parser, run_cv_baseline


def build_model():
    return make_pipeline(PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                         LinearRegression())


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_cv_baseline("mlr_interactions", build_model, args)
