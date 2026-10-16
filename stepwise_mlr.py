"""
stepwise_mlr.py - Stepwise multiple linear regression.

Bidirectional stepwise selection over linear and pairwise interaction terms
(as in MATLAB stepwiselm): a term enters when its partial F-test p-value < 0.05 and
leaves when it rises above 0.10. Selection is repeated inside every training fold.

Usage: python stepwise_mlr.py --data data/nthri_tea_yield.csv
"""
import itertools

import numpy as np
from scipy import stats

from common import base_parser, run_cv_baseline


class StepwiseLinearRegression:
    def __init__(self, p_enter=0.05, p_remove=0.10, interactions=True, max_steps=200):
        self.p_enter, self.p_remove = p_enter, p_remove
        self.interactions, self.max_steps = interactions, max_steps

    def _terms(self, X):
        cols = [X[:, i] for i in range(X.shape[1])]
        names = [(i,) for i in range(X.shape[1])]
        if self.interactions:
            for i, j in itertools.combinations(range(X.shape[1]), 2):
                cols.append(X[:, i] * X[:, j])
                names.append((i, j))
        return np.column_stack(cols), names

    @staticmethod
    def _rss(A, y):
        A1 = np.column_stack([np.ones(len(y)), A]) if A.size else np.ones((len(y), 1))
        coef, *_ = np.linalg.lstsq(A1, y, rcond=None)
        r = y - A1 @ coef
        return float(r @ r), A1.shape[1]

    def _p_value(self, rss_small, rss_big, df_big, n):
        f = (rss_small - rss_big) / (rss_big / max(n - df_big, 1))
        return float(1 - stats.f.cdf(max(f, 0.0), 1, max(n - df_big, 1)))

    def fit(self, X, y):
        T, names = self._terms(X)
        n = len(y)
        sel = []
        for _ in range(self.max_steps):
            changed = False
            rss_cur, _ = self._rss(T[:, sel], y)
            best_p, best_j = 1.0, None                      # forward step
            for j in range(T.shape[1]):
                if j in sel:
                    continue
                rss_new, df_new = self._rss(T[:, sel + [j]], y)
                p = self._p_value(rss_cur, rss_new, df_new, n)
                if p < best_p:
                    best_p, best_j = p, j
            if best_j is not None and best_p < self.p_enter:
                sel.append(best_j)
                changed = True
            if len(sel) > 1:                                 # backward step
                rss_full, df_full = self._rss(T[:, sel], y)
                worst_p, worst_j = 0.0, None
                for j in sel:
                    rest = [k for k in sel if k != j]
                    rss_red, _ = self._rss(T[:, rest], y)
                    p = self._p_value(rss_red, rss_full, df_full, n)
                    if p > worst_p:
                        worst_p, worst_j = p, j
                if worst_j is not None and worst_p > self.p_remove:
                    sel.remove(worst_j)
                    changed = True
            if not changed:
                break
        self.selected_ = sel
        self.term_names_ = [names[j] for j in sel]
        A1 = np.column_stack([np.ones(n), T[:, sel]])
        self.coef_, *_ = np.linalg.lstsq(A1, y, rcond=None)
        return self

    def predict(self, X):
        T, _ = self._terms(X)
        A1 = np.column_stack([np.ones(len(X)), T[:, self.selected_]])
        return A1 @ self.coef_


def build_model():
    return StepwiseLinearRegression()


if __name__ == "__main__":
    args = base_parser(__doc__).parse_args()
    run_cv_baseline("stepwise_mlr", build_model, args)
