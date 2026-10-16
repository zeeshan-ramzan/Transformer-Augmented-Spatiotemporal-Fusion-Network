"""
svr_linear.py - Support vector regression with a linear kernel.

The target is standardised inside each fold so that C and epsilon are scale free.
Usage: python svr_linear.py --data data/nthri_tea_yield.csv [--C 10 --epsilon 0.05]
"""
from sklearn.svm import SVR

from common import base_parser, run_cv_baseline


def build_model(C=10.0, epsilon=0.05):
    return SVR(kernel="linear", C=C, epsilon=epsilon)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--C", type=float, default=10.0)
    p.add_argument("--epsilon", type=float, default=0.05)
    args = p.parse_args()
    run_cv_baseline("svr_linear", lambda: build_model(args.C, args.epsilon), args, scale_y=True)
