"""
gpr_matern52.py - Gaussian process regression with a Matern 5/2 kernel (nu = 5/2) and
automatic relevance determination. In the published NTHRI benchmark this was the best
model after the NAS-DNN.

Kernel = constant * Matern(nu = 2.5) + white noise; hyperparameters are fitted by
maximising the marginal likelihood (3 optimiser restarts). The target is
standardised inside each fold.
Usage: python gpr_matern52.py --data data/nthri_tea_yield.csv
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

from common import FEATURES, base_parser, run_cv_baseline


def build_model(n_features=len(FEATURES), seed=42, restarts=3):
    kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(n_features), nu=2.5) + WhiteKernel(noise_level=0.01)
    return GaussianProcessRegressor(kernel=kernel, normalize_y=False, n_restarts_optimizer=restarts,
                                    random_state=seed)


if __name__ == "__main__":
    p = base_parser(__doc__)
    p.add_argument("--restarts", type=int, default=3)
    args = p.parse_args()
    run_cv_baseline("gpr_matern52", lambda: build_model(seed=args.seed, restarts=args.restarts), args, scale_y=True)
