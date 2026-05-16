"""
Bivariate ANM prototype.

Fits a GP in each direction, takes HSIC of the input against the residual, picks
the direction with smaller HSIC. Standard recipe; see Hoyer et al. (2009) and
the GP-pHSIC entry in Mooij et al. (2016, Tab. 1).

Standalone -- does not import pgmpy. Used for the GSoC proposal to check the
API surface before writing the real PR.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


def hsic(x, y):
    # V-statistic, RBF kernel, median bandwidth.
    # (Gretton et al. 2008, eq. 4; here we return Tr(KHLH)/n^2, not the
    # gamma p-value -- monotonic, sufficient for ranking the two directions.)
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = x.shape[0]

    def gram(a):
        d = squareform(pdist(a, "sqeuclidean"))
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        return np.exp(-d / med)   # length-scale^2 = med / 2, factored in

    K, L = gram(x), gram(y)
    H = np.eye(n) - 1.0 / n
    return float(np.trace(K @ H @ L @ H)) / (n * n)


class PairwiseBase(BaseEstimator):
    # Mirrors the proposed _BasePairwiseDiscovery in pgmpy/causal_discovery/_base.py.
    # Subclasses implement _score(cause, effect). Lower = cause -> effect.
    def fit(self, X):
        a, b = X.columns
        x, y = X[a].to_numpy(), X[b].to_numpy()
        self.s_fwd_ = self._score(x, y)
        self.s_bwd_ = self._score(y, x)
        if self.s_fwd_ < self.s_bwd_:
            self.direction_ = (a, b)
        else:
            self.direction_ = (b, a)
        self.confidence_ = abs(self.s_fwd_ - self.s_bwd_) / (self.s_fwd_ + self.s_bwd_)
        return self

    def _score(self, cause, effect):
        raise NotImplementedError


class ANM(PairwiseBase):
    def __init__(self, regressor=None, random_state=None):
        self.regressor = regressor
        self.random_state = random_state

    def _score(self, cause, effect):
        gp = self.regressor or GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(),
            normalize_y=True,
            random_state=self.random_state,
        )
        gp = clone(gp) if self.regressor is not None else gp
        gp.fit(cause.reshape(-1, 1), effect)
        r = effect - gp.predict(cause.reshape(-1, 1))
        return hsic(cause, r)


# ---- experiments ---------------------------------------------------------

def sim_anm(n, seed):
    # Y = tanh(1.5 X) + E, X uniform, E gaussian; true direction x -> y.
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, n)
    y = np.tanh(1.5 * x) + 0.3 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y})


def sim_linear_gaussian(n, seed):
    # Identifiability fails here (Peters 14, Thm 1 exclusion).
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 0.7 * x + 0.5 * rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y})


if __name__ == "__main__":
    nseeds, n = 50, 500

    hits = sum(
        ANM(random_state=s).fit(sim_anm(n, s)).direction_ == ("x", "y")
        for s in range(nseeds)
    )
    print("ANM-identifiable:    %d / %d" % (hits, nseeds))

    confs = [ANM(random_state=s).fit(sim_anm(n, s)).confidence_ for s in range(nseeds)]
    print("                     mean confidence = %.3f" % np.mean(confs))

    confs_lg = [
        ANM(random_state=s).fit(sim_linear_gaussian(n, s)).confidence_
        for s in range(nseeds)
    ]
    print("linear-Gaussian:     mean confidence = %.3f  (should be << ANM)" % np.mean(confs_lg))
