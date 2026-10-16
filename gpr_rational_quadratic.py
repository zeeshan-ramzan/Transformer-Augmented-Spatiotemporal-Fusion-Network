"""
gpr_rational_quadratic.py - Gaussian process regression with a rational quadratic kernel.

Kernel = constant * base kernel + white noise; hyperparameters are fitted by
maximising the marginal likelihood (3 optimiser restarts). The target is
standardised inside each fold.
Usage: python gpr_rational_quadratic.py --data data/nthri_tea_yield.csv
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, WhiteKernel, RationalQuadratic

from common import FEATURES, base_parser, run_cv_baseline


def build_model(n_features=len(FEATURES), seed=42, restarts=3):
    kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0, alpha=1.0) + WhiteKernel(noise_level=0.01)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=restarts,
                                    random_state=seed)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--restarts", type=int, default=3)
    args = p.parse_args()
    run_cv_baseline("gpr_rational_quadratic", lambda: build_model(seed=args.seed, restarts=args.restarts), args, scale_y=True)
