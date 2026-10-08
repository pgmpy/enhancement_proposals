"""Prototype of pgmpy v2.0 parameterization refactor.

Two user journeys:
  Section A — Classic BN user (structure → parameters → inference with CIs).
  Section B — Causal inference user (interventions + counterfactuals + diagnostics).

Accessors map to Pearl's rungs plus tooling:
  dag.inference.query(evidence=..., query=...)         — rung 1
  dag.intervene.query(do=..., query=...)               — rung 2
  dag.counterfactual.query(observed=..., do=..., query=...)  — rung 3
                                                        (multi-world if do is list)
  dag.diagnostics.identifiability_report()
  dag.transforms.{ancestors, descendants, d_separated, ...}
  dag.bootstrap.query(data, query_fn, n_bootstrap=...)

All query() methods return QueryResult.

Design simplifications:
  - Unified NoiseDistribution-typed noise (Delta, NormalNoise, Empirical, TruncatedUniform).
  - Single WrappedRegressor for ANM (link=None) and PNL (link/link_inv).
  - CPDAdapter auto-wraps third-party CPDs at dag.parameters.add().
  - DAG is not a sklearn/skbase estimator; refit uses dag.copy_template(...).
  - Reduced 7-tag set (dropped noise_invertible, parameter_uncertainty, is_mixture, supports_analytic_conditioning).
  - dag.diagnostics for identifiability checks.

Run: PYTHONPATH=/path/to/skpro python prototype.py
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Callable, Hashable, Literal

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import ClassifierMixin

from skpro.regression.base import BaseProbaRegressor


# --- 1. Internal infrastructure ------------------------------------------

class CPDContractError(TypeError):
    """Raised when an object does not satisfy the pgmpy CPD contract."""


class IncompatibleCPDError(TypeError):
    """Raised when an operation requires a capability the CPD doesn't expose."""


def _get_tag(cpd, name, default=None):
    """Read a tag from a CPD, falling back to default if absent."""
    if hasattr(cpd, "get_tag"):
        try:
            val = cpd.get_tag(name)
            if val is not None:
                return val
        except (KeyError, ValueError):
            pass
    tags = getattr(cpd, "_tags", {})
    if name in tags:
        return tags[name]
    wrapped = getattr(cpd, "wrapped", cpd)
    if name == "variable_type":
        if isinstance(wrapped, ClassifierMixin):
            return "discrete"
        if isinstance(wrapped, BaseProbaRegressor):
            return "continuous"
    return default


def require_fittable(obj: Any) -> None:
    """Validate the per-node fitting protocol: fit(X, y, ...)."""
    if not callable(getattr(obj, "fit", None)):
        raise CPDContractError(f"{type(obj).__name__} must define fit(...)")
    fit_sig = inspect.signature(obj.fit)
    if "X" not in fit_sig.parameters or "y" not in fit_sig.parameters:
        raise CPDContractError(
            f"{type(obj).__name__}.fit must accept (X, y, ...); got {fit_sig}"
        )


def require_predictive(obj: Any) -> None:
    """Validate the predict_proba protocol used by CPDAdapter."""
    if not callable(getattr(obj, "predict_proba", None)):
        raise CPDContractError(
            f"{type(obj).__name__} must define predict_proba(...)"
        )


def require_sampleable(obj: Any) -> None:
    """Validate the sampling protocol used by simulate/intervene."""
    if not callable(getattr(obj, "sample", None)):
        raise CPDContractError(f"{type(obj).__name__} must define sample(...)")


def require_scorable(obj: Any) -> None:
    """Validate the scoring protocol used by likelihood weighting."""
    if not callable(getattr(obj, "log_prob", None)):
        raise CPDContractError(
            f"{type(obj).__name__} must define log_prob(...)"
        )


def check_parameterization(obj: Any) -> None:
    """Compatibility validator for third-party predictive CPDs.

    Built-in and simulation-native CPDs can satisfy narrower protocols;
    CPDAdapter specifically needs fit(X, y) plus predict_proba(X).
    """
    require_fittable(obj)
    require_predictive(obj)


def _validate_full_runtime_cpd(obj: Any) -> None:
    """Validate the runtime surface stored in DAG.parameters."""
    for check in (require_fittable, require_sampleable, require_scorable):
        check(obj)


class CPDAdapter:
    """Adapt any (fit, predict_proba)-style object to the full pgmpy CPD
    interface. Used for third-party estimators (sklearn / sklearn.Pipeline /
    skpro) without native sample/log_prob. Built-in CPDs pass through
    unwrapped; `_DAGParameters.add()` auto-wraps as needed.
    """

    def __init__(self, wrapped):
        check_parameterization(wrapped)
        self._wrapped = wrapped

    @property
    def wrapped(self):
        return self._wrapped

    def __getattr__(self, name):
        # Forward everything not explicitly defined here to the wrapped object.
        return getattr(self._wrapped, name)

    def fit(self, X, y, **kwargs):
        # Try with kwargs first; fall back if the wrapped object doesn't accept them.
        try:
            return self._wrapped.fit(X, y, **kwargs)
        except TypeError:
            return self._wrapped.fit(X, y)

    def predict_proba(self, X):
        return self._wrapped.predict_proba(X)

    def sample(self, X, n_samples=None):
        if n_samples is None:
            n_samples = len(X)
        proba = self._wrapped.predict_proba(X)
        # skpro distribution path
        if hasattr(proba, "sample") and callable(proba.sample):
            s = proba.sample()
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            idx = getattr(X, "index", pd.RangeIndex(n_samples))
            return pd.Series(np.asarray(s).ravel(), index=idx)
        # sklearn classifier path: 2-D probability matrix
        if isinstance(proba, pd.DataFrame):
            classes, probs = list(proba.columns), proba.values
        else:
            proba = np.asarray(proba)
            classes = list(getattr(self._wrapped, "classes_",
                                    range(proba.shape[1])))
            probs = proba
        rng = np.random.default_rng()
        return pd.Series(
            [rng.choice(classes, p=probs[i]) for i in range(probs.shape[0])]
        )

    def log_prob(self, y, X):
        proba = self._wrapped.predict_proba(X)
        if hasattr(proba, "log_pdf") and callable(proba.log_pdf):
            idx = y.index if hasattr(y, "index") else pd.RangeIndex(len(y))
            y_df = pd.DataFrame({"value": np.asarray(y).ravel()}, index=idx)
            lp = proba.log_pdf(y_df)
            if isinstance(lp, pd.DataFrame):
                lp = lp.iloc[:, 0]
            return pd.Series(np.asarray(lp).ravel(), index=idx)
        if isinstance(proba, pd.DataFrame):
            classes, probs = list(proba.columns), proba.values
        else:
            proba = np.asarray(proba)
            classes = list(getattr(self._wrapped, "classes_",
                                    range(proba.shape[1])))
            probs = proba
        cols = [classes.index(v) for v in np.asarray(y)]
        return pd.Series(np.log(probs[np.arange(len(y)), cols]))

    def get_tag(self, name, default=None):
        if hasattr(self._wrapped, "get_tag"):
            try:
                val = self._wrapped.get_tag(name)
                if val is not None:
                    return val
            except (KeyError, ValueError):
                pass
        return getattr(self._wrapped, "_tags", {}).get(name, default)

    def clone(self):
        """Return an unfit CPDAdapter wrapping a clone of the underlying object."""
        if hasattr(self._wrapped, "clone"):
            return CPDAdapter(self._wrapped.clone())
        try:
            from sklearn.base import clone as sk_clone
            return CPDAdapter(sk_clone(self._wrapped))
        except Exception:
            return CPDAdapter(type(self._wrapped)())


def _natively_supports_cpd_contract(obj):
    """True iff obj implements sample(X, n_samples) and log_prob(y, X) directly."""
    return (
        callable(getattr(obj, "sample", None))
        and callable(getattr(obj, "log_prob", None))
        and "X" in inspect.signature(obj.sample).parameters
    )


# --- 2. Unified noise distributions --------------------------------------
# All StructuralCPDs return one of these from noise_prior() and abduct().
# Uniform interface: .sample(n, random_state) -> ndarray; .point() -> value.

@dataclass
class Delta:
    """Degenerate point distribution. Returned by abduct() for invertible CPDs."""
    value: np.ndarray

    def sample(self, n: int = 1, random_state=None) -> np.ndarray:
        val = np.atleast_1d(np.asarray(self.value, dtype=float))
        return np.tile(val, (n, 1)).squeeze(-1) if val.size > 0 else val

    def point(self):
        return self.value


@dataclass
class NormalNoise:
    """Gaussian noise prior; lightweight stand-in for skpro.distributions.Normal."""
    mu: float = 0.0
    sigma: float = 1.0

    def sample(self, n: int = 1, random_state=None) -> np.ndarray:
        return np.random.default_rng(random_state).normal(self.mu, self.sigma, n)

    def point(self):
        return self.mu


@dataclass
class Empirical:
    """Empirical noise distribution; resamples training residuals."""
    samples: np.ndarray

    def sample(self, n: int = 1, random_state=None) -> np.ndarray:
        return np.random.default_rng(random_state).choice(self.samples, size=n)

    def point(self):
        return float(np.median(self.samples))


@dataclass
class TruncatedUniform:
    """1-D truncated uniform per row. Returned by TabularCPD.abduct()."""
    low: np.ndarray
    high: np.ndarray

    def sample(self, n: int = 1, random_state=None) -> np.ndarray:
        rng = np.random.default_rng(random_state)
        # Shape (n_samples, n_rows)
        return self.low[None, :] + rng.uniform(size=(n, len(self.low))) * (
            self.high - self.low
        )[None, :]

    def point(self):
        return (self.low + self.high) / 2.0


# --- 3. QueryResult — unified distribution-valued result type ------------

@dataclass
class QueryResult:
    """Result of any query that returns a distribution over a single variable.

    Distribution-valued by construction. For point-invertible cases, samples
    is shape (1,) and .point() returns that single value. For Monte Carlo
    cases or Bayesian regressors, samples is (n,).
    """
    samples: np.ndarray
    query: Any
    operation: str  # "intervene", "counterfact", "predict"
    operation_args: dict
    meta: dict = field(default_factory=dict)

    def point(self) -> float:
        return float(np.mean(self.samples))

    def distribution(self) -> pd.Series:
        return pd.Series(self.samples, name=str(self.query))

    def expectation(self, fn: Callable[[float], float]) -> float:
        return float(np.mean([fn(s) for s in self.samples]))

    def credible_interval(self, level: float = 0.95) -> tuple[float, float]:
        alpha = (1.0 - level) / 2.0
        lo, hi = np.quantile(self.samples, [alpha, 1.0 - alpha])
        return float(lo), float(hi)

    def compare_to(self, other: "QueryResult") -> dict:
        from scipy.stats import wasserstein_distance
        return {
            "wasserstein": float(wasserstein_distance(self.samples, other.samples)),
            "abs_mean_diff": abs(self.point() - other.point()),
        }


# --- 4. CPD classes ------------------------------------------------------
# Reduced 7-tag set; dropped noise_invertible / parameter_uncertainty /
# is_mixture / supports_analytic_conditioning.

class TabularCPD(ClassifierMixin, SklearnBaseEstimator):
    """Discrete CPT with inverse-CDF SCM encoding for counterfactuals."""

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "supports_counterfactual": True,
        "noise_type": "inverse_cdf",
    }

    def __init__(self, variable_card, evidence_card=None, state_names=None,
                 prior_type=None, equivalent_sample_size=10,
                 pseudo_counts=None, noise_repr="inverse_cdf"):
        # Hyperparameters (sklearn convention — stored unchanged).
        self.variable_card = variable_card
        self.evidence_card = evidence_card
        self.state_names = state_names
        # Bayesian fitting hyperparameters (None == MLE in v2.0).
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size
        self.pseudo_counts = pseudo_counts
        # Discrete SCM noise representation. Currently only inverse_cdf
        # is implemented; gumbel_max is reserved for v2.x.
        if noise_repr not in ("inverse_cdf", "gumbel_max"):
            raise ValueError(
                f"noise_repr must be 'inverse_cdf' or 'gumbel_max'; "
                f"got {noise_repr!r}"
            )
        if noise_repr == "gumbel_max":
            raise NotImplementedError(
                "noise_repr='gumbel_max' is reserved for v2.x; "
                "v2.0 ships inverse-CDF semantics."
            )
        self.noise_repr = noise_repr

    @classmethod
    def from_values(cls, variable_card, values, evidence_card=None,
                    state_names=None):
        instance = cls(variable_card=variable_card, evidence_card=evidence_card,
                       state_names=state_names)
        arr = np.asarray(values, dtype=float)
        n_combos = int(np.prod(evidence_card)) if evidence_card else 1
        if arr.shape != (variable_card, n_combos):
            raise ValueError(
                f"values shape {arr.shape} != expected ({variable_card}, "
                f"{n_combos})"
            )
        instance.values_ = arr / arr.sum(axis=0, keepdims=True)
        if state_names is not None:
            instance.classes_ = np.array(state_names[0])
            instance._fitted_parent_states_ = [list(s) for s in state_names[1:]]
        else:
            instance.classes_ = np.arange(variable_card)
            instance._fitted_parent_states_ = (
                [list(range(c)) for c in evidence_card] if evidence_card else []
            )
        instance.is_fitted_ = True
        return instance

    def fit(self, X, y, sample_weight=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y
        sw = np.ones(len(y)) if sample_weight is None else np.asarray(
            sample_weight, dtype=float
        )
        if self.state_names is None:
            child_states = sorted(y.unique().tolist())
            parent_states = [sorted(X[c].unique().tolist()) for c in X.columns]
        else:
            child_states = list(self.state_names[0])
            parent_states = [list(s) for s in self.state_names[1:]]
        n_combos = (int(np.prod([len(s) for s in parent_states]))
                    if parent_states else 1)
        counts = np.zeros((self.variable_card, n_combos), dtype=float)
        child_idx = {v: i for i, v in enumerate(child_states)}
        if X.shape[1] == 0:
            parent_flat = np.zeros(len(y), dtype=int)
        else:
            lookups = [{v: i for i, v in enumerate(ps)} for ps in parent_states]
            mults = [1]
            for s in parent_states[:-1]:
                mults.append(mults[-1] * len(s))
            mults = mults[::-1]
            parent_flat = np.zeros(len(y), dtype=int)
            for k, col in enumerate(X.columns):
                parent_flat += X[col].map(lookups[k]).to_numpy() * mults[k]
        for row in range(len(y)):
            counts[child_idx[y.iat[row]], parent_flat[row]] += sw[row]
        col_sums = counts.sum(axis=0, keepdims=True)
        zero = col_sums == 0
        col_sums[zero] = 1.0
        normalized = counts / col_sums
        normalized[:, zero[0]] = 1.0 / self.variable_card
        self.values_ = normalized
        self.classes_ = np.array(child_states)
        self._fitted_parent_states_ = parent_states
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        parent_states = self._fitted_parent_states_
        if X.shape[1] == 0:
            row_idx = np.zeros(len(X), dtype=int)
        else:
            mults = [1]
            for s in parent_states[:-1]:
                mults.append(mults[-1] * len(s))
            mults = mults[::-1]
            row_idx = np.zeros(len(X), dtype=int)
            for k, col in enumerate(X.columns):
                lookup = {v: i for i, v in enumerate(parent_states[k])}
                row_idx += X[col].map(lookup).to_numpy() * mults[k]
        probs = self.values_[:, row_idx].T
        return pd.DataFrame(probs, columns=list(self.classes_), index=X.index)

    def sample(self, X, n_samples=None):
        proba = self.predict_proba(X)
        n = n_samples or len(X)
        classes = list(proba.columns)
        rng = np.random.default_rng()
        draws = [rng.choice(classes, p=proba.iloc[i].values)
                 for i in range(min(n, len(proba)))]
        return pd.Series(draws, index=proba.index[:len(draws)])

    def log_prob(self, y, X):
        proba = self.predict_proba(X)
        cols = [proba.columns.get_loc(v) for v in y.values]
        return pd.Series(np.log(proba.values[np.arange(len(y)), cols]),
                          index=y.index)

    def get_tag(self, name, default=None):
        return self._tags.get(name, default)

    # ---- StructuralCPD protocol (inverse-CDF encoding) -------------------
    def noise_prior(self):
        # Uniform on [0, 1]; expressed via the unified Distribution interface.
        return TruncatedUniform(low=np.array([0.0]), high=np.array([1.0]))

    def structural_predict(self, parents, noise):
        proba = self.predict_proba(parents).values  # (n, K)
        u = np.asarray(noise, dtype=float).ravel()
        cum = np.cumsum(proba, axis=1)
        chosen = (cum >= u[:, None]).argmax(axis=1)
        return pd.Series([self.classes_[i] for i in chosen],
                         index=parents.index if hasattr(parents, "index") else None)

    def abduct(self, x, parents):
        proba = self.predict_proba(parents).values
        cum = np.cumsum(proba, axis=1)
        cum_lower = np.concatenate(
            [np.zeros((cum.shape[0], 1)), cum[:, :-1]], axis=1
        )
        k_obs = np.array([list(self.classes_).index(v) for v in x])
        rows = np.arange(len(x))
        return TruncatedUniform(
            low=cum_lower[rows, k_obs], high=cum[rows, k_obs]
        )


class LinearGaussianCPD(BaseProbaRegressor):
    """LG CPD; canonical additive noise model with invertible abduction."""

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": True,
        "supports_counterfactual": True,
        "noise_type": "additive",
    }

    def __init__(self):
        super().__init__()

    @classmethod
    def from_values(cls, beta, std):
        instance = cls()
        instance.beta_ = np.asarray(beta, dtype=float)
        instance.std_ = float(std)
        instance._is_fitted = True
        return instance

    def _fit(self, X, y, C=None):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()
        n = len(y_arr)
        X_design = (np.ones((n, 1)) if X_arr.size == 0
                    else np.hstack([np.ones((n, 1)), X_arr.reshape(n, -1)]))
        beta, *_ = np.linalg.lstsq(X_design, y_arr, rcond=None)
        residuals = y_arr - X_design @ beta
        self.beta_ = beta
        self.std_ = float(np.sqrt(np.mean(residuals ** 2)))
        return self

    def _predict_proba(self, X):
        from skpro.distributions import Normal
        X_arr = np.asarray(X, dtype=float)
        n = X_arr.shape[0]
        if X_arr.size == 0 or (X_arr.ndim == 2 and X_arr.shape[1] == 0):
            mean = np.full(n, self.beta_[0])
        else:
            mean = (self.beta_[0]
                    + (X_arr.reshape(n, -1) * self.beta_[1:]).sum(axis=1))
        idx = X.index if hasattr(X, "index") else pd.RangeIndex(n)
        return Normal(mu=pd.DataFrame({"value": mean}, index=idx),
                       sigma=pd.DataFrame({"value": np.full(n, self.std_)},
                                          index=idx))

    def sample(self, X, n_samples=None):
        dist = self.predict_proba(X)
        s = dist.sample()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return pd.Series(np.asarray(s).ravel(),
                         index=X.index if hasattr(X, "index") else None)

    def log_prob(self, y, X):
        dist = self.predict_proba(X)
        idx = y.index if hasattr(y, "index") else pd.RangeIndex(len(y))
        y_df = pd.DataFrame({"value": np.asarray(y).ravel()}, index=idx)
        log_pdf = dist.log_pdf(y_df)
        if isinstance(log_pdf, pd.DataFrame):
            log_pdf = log_pdf.iloc[:, 0]
        return pd.Series(np.asarray(log_pdf).ravel(), index=idx)

    # ---- StructuralCPD protocol ------------------------------------------
    def noise_prior(self):
        return NormalNoise(mu=0.0, sigma=self.std_)

    def structural_predict(self, parents, noise):
        u = np.asarray(noise, dtype=float).ravel()
        if parents is None or (hasattr(parents, "shape")
                                and parents.shape[1] == 0):
            f_pa = np.full(len(u), self.beta_[0])
        else:
            pa = np.asarray(parents, dtype=float).reshape(len(u), -1)
            f_pa = self.beta_[0] + pa @ self.beta_[1:]
        return pd.Series(f_pa + u,
                         index=parents.index if hasattr(parents, "index") else None)

    def abduct(self, x, parents):
        x_arr = np.asarray(x, dtype=float).ravel()
        n = len(x_arr)
        if parents is None or (hasattr(parents, "shape")
                                and parents.shape[1] == 0):
            f_pa = np.full(n, self.beta_[0])
        else:
            pa = np.asarray(parents, dtype=float).reshape(n, -1)
            f_pa = self.beta_[0] + pa @ self.beta_[1:]
        return Delta(value=x_arr - f_pa)


# --- 5. WrappedRegressor — merged ANM + PNL ------------------------------

class WrappedRegressor(BaseProbaRegressor):
    """Wrap any regressor as an SCM CPD.

    link=None      → additive noise model: X = f(pa) + U
    link, link_inv → post-nonlinear:       X = link(f(pa) + U)

    Composing PNL via a wrapper rather than a new class is the
    "protocol over hierarchy" payoff.
    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "supports_counterfactual": True,
        # noise_type is "additive" or "post_nonlinear" depending on link;
        # set per-instance in __init__.
        "noise_type": "additive",
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "capability:multioutput": False,
        "capability:missing": False,
    }

    def __init__(self, regressor, *, link=None, link_inv=None, noise_dist=None):
        self.regressor = regressor
        self.link = link
        self.link_inv = link_inv
        self.noise_dist = noise_dist
        super().__init__()
        # Per-instance tag override for diagnostics.
        self._tags = dict(self.__class__._tags)
        if link is not None:
            self._tags["noise_type"] = "post_nonlinear"
        if (link is None) ^ (link_inv is None):
            raise ValueError("link and link_inv must be provided together.")

    def _fit(self, X, y, C=None):
        from sklearn.base import clone as sk_clone
        y_arr = np.asarray(y, dtype=float).ravel()
        # If a link is given, fit on the inv-link scale.
        y_target = self.link_inv(y_arr) if self.link_inv else y_arr
        self.regressor_ = sk_clone(self.regressor).fit(X, y_target)
        residuals = y_target - self.regressor_.predict(X)
        # Resolve noise distribution.
        if self.noise_dist is None or self.noise_dist == "empirical":
            self.noise_dist_ = Empirical(samples=residuals)
        else:
            self.noise_dist_ = self.noise_dist
        self.noise_residuals_ = residuals
        return self

    def _predict_proba(self, X):
        from skpro.distributions import Normal
        mean_inv = self.regressor_.predict(X)
        sigma = float(np.std(self.noise_residuals_))
        idx = X.index if hasattr(X, "index") else pd.RangeIndex(len(mean_inv))
        return Normal(mu=pd.DataFrame({"value": mean_inv}, index=idx),
                       sigma=pd.DataFrame({"value": np.full(len(mean_inv), sigma)},
                                          index=idx))

    def sample(self, X, n_samples=None):
        """Native sampling: f(pa) + noise drawn from noise_dist_."""
        n = n_samples or len(X)
        u = self.noise_dist_.sample(n=n)
        u = np.asarray(u, dtype=float).ravel()
        return self.structural_predict(X, u[:n])

    def log_prob(self, y, X):
        """Native log-prob via predict_proba's log_pdf."""
        dist = self.predict_proba(X)
        idx = y.index if hasattr(y, "index") else pd.RangeIndex(len(y))
        y_df = pd.DataFrame({"value": np.asarray(y).ravel()}, index=idx)
        lp = dist.log_pdf(y_df)
        if isinstance(lp, pd.DataFrame):
            lp = lp.iloc[:, 0]
        return pd.Series(np.asarray(lp).ravel(), index=idx)

    # ---- StructuralCPD protocol ------------------------------------------
    def noise_prior(self):
        return self.noise_dist_

    def structural_predict(self, parents, noise):
        u = np.asarray(noise, dtype=float).ravel()
        f_pa = self.regressor_.predict(parents)
        z = f_pa + u
        return pd.Series(self.link(z) if self.link else z,
                         index=parents.index if hasattr(parents, "index") else None)

    def abduct(self, x, parents):
        x_arr = np.asarray(x, dtype=float).ravel()
        f_pa = self.regressor_.predict(parents)
        z = self.link_inv(x_arr) if self.link_inv else x_arr
        return Delta(value=z - f_pa)

    def get_tag(self, name, default=None):
        return self._tags.get(name, default)


# --- 6. DAG-owned schema -------------------------------------------------

@dataclass(frozen=True)
class VariableSchema:
    variable: Hashable
    variable_type: str
    states: tuple | None = None
    dtype: Any | None = None
    ordered: bool = False
    encoder: Any | None = None
    decoder: Any | None = None


class _DAGSchema:
    """Internal variable metadata registry.

    Users normally don't populate this manually. It is inferred from CPDs,
    pandas categoricals, observed fitting data, and read/write formats.
    """

    def __init__(self, dag):
        self._dag = dag
        self._schemas: dict[Hashable, VariableSchema] = {}

    def __getitem__(self, variable):
        return self._schemas[variable]

    def get(self, variable, default=None):
        return self._schemas.get(variable, default)

    def items(self):
        return self._schemas.items()

    def set(
        self,
        variable,
        *,
        variable_type,
        states=None,
        dtype=None,
        ordered=False,
        encoder=None,
        decoder=None,
    ):
        states_tuple = None if states is None else tuple(states)
        new = VariableSchema(
            variable=variable,
            variable_type=variable_type,
            states=states_tuple,
            dtype=dtype,
            ordered=ordered,
            encoder=encoder,
            decoder=decoder,
        )
        old = self._schemas.get(variable)
        if old is not None:
            if old.variable_type != new.variable_type:
                raise ValueError(
                    f"Schema type conflict for {variable!r}: "
                    f"{old.variable_type!r} vs {new.variable_type!r}"
                )
            if old.states is not None and new.states is not None and old.states != new.states:
                raise ValueError(
                    f"State conflict for {variable!r}: "
                    f"{old.states!r} vs {new.states!r}"
                )
            if new.states is None or old.states is not None:
                new = old
        self._schemas[variable] = new
        return self

    def infer_from_cpd(self, variable, cpd, parent_order):
        var_type = _get_tag(cpd, "variable_type", None) or "continuous"
        state_names = getattr(cpd, "state_names", None)
        if state_names is not None and len(state_names) > 0:
            self.set(variable, variable_type=var_type, states=state_names[0])
            for parent, states in zip(parent_order, state_names[1:]):
                self.set(parent, variable_type="discrete", states=states)
            return self

        if var_type == "discrete":
            card = getattr(cpd, "variable_card", None)
            states = tuple(range(card)) if card is not None else None
            self.set(variable, variable_type="discrete", states=states)
        else:
            self.set(variable, variable_type="continuous")
        return self

    def infer_from_data(self, data: pd.DataFrame, cpds: dict):
        """Fill in schema metadata from observed data.

        Precedence policy: CPD-authored state ordering wins. If a variable
        already has `states` set (typically by `infer_from_cpd`), data is
        only used to (a) update `dtype` and (b) validate that observed
        values lie within the declared state set. The order is preserved
        as the CPD declared it — alphabetical re-sorting from
        `sorted(unique)` would be a hostile reinterpretation of the user's
        explicit choice.
        """
        for variable, cpd in cpds.items():
            if variable not in data:
                continue
            series = data[variable]
            existing = self._schemas.get(variable)
            var_type = (existing.variable_type if existing is not None
                        else _get_tag(cpd, "variable_type", None))

            data_is_discrete = (
                isinstance(series.dtype, pd.CategoricalDtype)
                or var_type == "discrete"
                or series.dtype == object
                or series.dtype == bool
            )

            if not data_is_discrete:
                self.set(variable, variable_type="continuous",
                          dtype=series.dtype)
                continue

            if existing is not None and existing.states is not None:
                # CPD-authored states are authoritative. Validate consistency.
                observed = set(series.dropna().unique())
                declared = set(existing.states)
                unexpected = observed - declared
                if unexpected:
                    raise ValueError(
                        f"Variable {variable!r} has observed values "
                        f"{sorted(unexpected, key=str)!r} not in declared "
                        f"states {existing.states!r}."
                    )
                # Preserve CPD order; update dtype only.
                self.set(
                    variable,
                    variable_type="discrete",
                    states=existing.states,
                    ordered=existing.ordered,
                    dtype=series.dtype,
                )
            elif isinstance(series.dtype, pd.CategoricalDtype):
                # Pandas categorical metadata is the next source of truth.
                self.set(
                    variable,
                    variable_type="discrete",
                    states=series.cat.categories,
                    ordered=series.cat.ordered,
                    dtype=series.dtype,
                )
            else:
                # Fall back to sorted unique values from the data.
                self.set(
                    variable,
                    variable_type="discrete",
                    states=sorted(series.dropna().unique(), key=str),
                    dtype=series.dtype,
                )
        return self

    def copy_to(self, other):
        other.schema._schemas = dict(self._schemas)


# --- 7. DAG + accessors --------------------------------------------------

class _DAGParameters:
    """CPD-registry management."""

    def __init__(self, dag):
        self._dag = dag

    def add(self, *, variable, cpd, parent_order=None):
        if variable not in self._dag.nodes():
            raise ValueError(f"Variable {variable!r} not in DAG.")
        expected = set(self._dag.predecessors(variable))
        if parent_order is None:
            parent_order = list(self._dag.predecessors(variable))
        elif set(parent_order) != expected:
            raise ValueError(
                f"parent_order {parent_order!r} != graph parents "
                f"{sorted(expected, key=str)!r}"
            )
        # Auto-wrap third-party objects that don't natively implement
        # the full CPD contract (sample(X, n_samples) + log_prob(y, X)).
        # Built-in CPDs (TabularCPD, LinearGaussianCPD, WrappedRegressor)
        # pass through unwrapped.
        if not _natively_supports_cpd_contract(cpd):
            check_parameterization(cpd)
            cpd = CPDAdapter(cpd)
        _validate_full_runtime_cpd(cpd)
        self._dag.schema.infer_from_cpd(variable, cpd, parent_order)
        self._dag._cpds[variable] = cpd
        self._dag._parent_order[variable] = list(parent_order)
        return self

    def get(self, node=None):
        return (self._dag._cpds[node] if node is not None
                else [self._dag._cpds[n] for n in self._dag.nodes()
                      if n in self._dag._cpds])

    def keys(self):     return self._dag._cpds.keys()
    def values(self):   return self._dag._cpds.values()
    def items(self):    return self._dag._cpds.items()
    def __len__(self):  return len(self._dag._cpds)
    def __iter__(self): return iter(self._dag._cpds)
    def __contains__(self, n): return n in self._dag._cpds
    def __getitem__(self, n):
        if n not in self._dag._cpds:
            raise KeyError(f"No CPD for node {n!r}")
        return self._dag._cpds[n]


class _DAGDiagnostics:
    """Static SCM diagnostics — simplification #4 promoted from counterfactual.

    Identifiability isn't only a counterfactual concern (do-calculus / ID
    algorithm applies to interventional estimands too). Future home for
    residual analysis, structure adequacy, fit quality.
    """

    def __init__(self, dag):
        self._dag = dag

    def identifiability_report(self, query_type: str = "counterfactual") -> dict:
        """Flag known non-identification patterns.

        - Pure linear-Gaussian sub-graphs (Hoyer et al., NIPS 2009).
        - Discrete nodes with non-invertible noise representations
          (Oberst & Sontag, ICML 2019; Nasr-Esfahany et al., 2023).
        """
        warnings = []
        # Check 1: LG sub-graph closure.
        lg_nodes = {n for n in self._dag.nodes()
                    if _get_tag(self._dag.parameters[n], "is_linear_gaussian",
                                False)}
        if len(lg_nodes) >= 2:
            sub = self._dag.subgraph(lg_nodes)
            if len(sub.edges()) >= 1:
                warnings.append({
                    "type": "linear_gaussian_path",
                    "nodes": sorted(lg_nodes, key=str),
                    "ref": "Hoyer et al., NIPS 2009",
                    "message": (
                        f"Nodes {sorted(lg_nodes, key=str)} form a "
                        f"linear-Gaussian sub-graph. Counterfactuals within "
                        f"it are not identified by the observational "
                        f"distribution alone."
                    ),
                })
        # Check 2: discrete nodes with non-invertible noise.
        for n in self._dag.nodes():
            cpd = self._dag.parameters[n]
            noise_type = _get_tag(cpd, "noise_type", None)
            if (_get_tag(cpd, "variable_type") == "discrete"
                    and _get_tag(cpd, "supports_counterfactual", False)
                    and noise_type in ("inverse_cdf", "gumbel_max", "custom")):
                warnings.append({
                    "type": "discrete_non_identified",
                    "node": n,
                    "noise_type": noise_type,
                    "ref": "Oberst & Sontag, ICML 2019",
                    "message": (
                        f"Node {n!r} uses a non-invertible noise "
                        f"representation ({noise_type!r}). The "
                        f"counterfactual depends on the noise_repr choice."
                    ),
                })
        return {"warnings": warnings, "n_warnings": len(warnings)}


class _DAGCounterfactual:
    """Pearl's abduction-action-prediction over CPDs that opt into the protocol."""

    def __init__(self, dag):
        self._dag = dag

    def _check_capability(self):
        offenders = [n for n in self._dag.nodes()
                     if not _get_tag(self._dag.parameters[n],
                                     "supports_counterfactual", False)]
        if offenders:
            raise IncompatibleCPDError(
                f"CPDs on {offenders} do not implement the StructuralCPD "
                f"protocol (supports_counterfactual=False)."
            )

    def _validate_noise_overrides(self, noise_overrides):
        """Allow per-node noise_overrides only if they match the CPD's
        declared noise_type. Alternative encodings (e.g. gumbel_max for
        TabularCPD) are reserved for v2.x."""
        if not noise_overrides:
            return
        for node, requested in noise_overrides.items():
            if node not in self._dag.nodes():
                raise ValueError(
                    f"noise_overrides references unknown node {node!r}."
                )
            cpd = self._dag.parameters[node]
            current = _get_tag(cpd, "noise_type", None)
            if requested != current:
                raise NotImplementedError(
                    f"noise_overrides[{node!r}] = {requested!r} is not yet "
                    f"supported (CPD declares noise_type={current!r}). "
                    f"Cross-noise-representation robustness ships in v2.x."
                )

    def query(self, observed: dict, do, query, n_samples: int = 1,
              noise_overrides: dict | None = None,
              seed: int | None = None):
        """Abduction → action → prediction.

        Parameters
        ----------
        observed : dict
            Factual observation `{node: value, ...}`.
        do : dict OR list[dict]
            Intervention. If a single dict, returns a `QueryResult`.
            If a list of dicts, returns a list of `QueryResult`s sharing
            the same abducted noise — Pearl's "twin/multi-world" semantics
            (after ChiRho's `MultiWorldCounterfactual`).
        query : str
            Variable name to query under each intervention.
        noise_overrides : dict, optional
            Per-node override of the noise representation (e.g.
            ``{"Y": "gumbel_max"}``). In v2.0 only the CPD's declared
            ``noise_type`` is honoured; passing a value other than the
            CPD's declared `noise_type` raises `NotImplementedError`.
            The plumbing is in place so the cross-noise-representation
            robustness workflow (compare counterfactuals across encodings)
            is available the day alternative encodings ship in v2.x.
        """
        self._check_capability()
        self._validate_noise_overrides(noise_overrides)
        multi_world = isinstance(do, list)
        do_list = do if multi_world else [do]

        noise_samples, n = self._abduce(observed, n_samples=n_samples, seed=seed)

        results = []
        for do_dict in do_list:
            cf_samples = self._propagate(noise_samples, do_dict, n)
            results.append(QueryResult(
                samples=cf_samples[query], query=query,
                operation="counterfact",
                operation_args={"observed": observed, "do": do_dict},
                meta={"abducted_noise": {k: float(np.mean(noise_samples[k]))
                                           for k in noise_samples}},
            ))
        return results if multi_world else results[0]

    def _abduce(self, observed, n_samples=1, seed=None):
        """Step 1 — abduction. Returns (noise_samples: dict, n: int)."""
        rng = np.random.default_rng(seed)
        noise_dists = {}
        for node in nx.topological_sort(self._dag):
            cpd = self._dag.parameters[node]
            parents = self._dag._parent_order.get(node, [])
            X_pa = (pd.DataFrame({p: [observed[p]] for p in parents})
                    if parents else pd.DataFrame(index=[0]))
            x_obs = pd.Series([observed[node]])
            noise_dists[node] = cpd.abduct(x_obs, X_pa)
        all_delta = all(isinstance(d, Delta) for d in noise_dists.values())
        n = 1 if all_delta else max(n_samples, 200)
        noise_samples = {}
        for node, dist in noise_dists.items():
            seed_k = int(rng.integers(0, 2**32 - 1))
            s = dist.sample(n=n, random_state=seed_k)
            noise_samples[node] = np.atleast_1d(np.asarray(s, dtype=float).ravel())
        return noise_samples, n

    def _propagate(self, noise_samples, do_dict, n):
        """Steps 2+3 — action + prediction. Returns {node: samples}."""
        cf_samples = {node: None for node in self._dag.nodes()}
        for node in nx.topological_sort(self._dag):
            if node in do_dict:
                cf_samples[node] = np.full(n, float(do_dict[node]))
                continue
            cpd = self._dag.parameters[node]
            parents = self._dag._parent_order.get(node, [])
            if parents:
                pa_df = pd.DataFrame({p: cf_samples[p] for p in parents})
            else:
                pa_df = pd.DataFrame(index=range(n))
            x_cf = cpd.structural_predict(pa_df, noise_samples[node][:n])
            cf_samples[node] = np.asarray(x_cf.values, dtype=float).ravel()
        return cf_samples


class _DAGInference:
    """Rung 1 — associational queries P(query | evidence).

    Uses likelihood weighting under the hood (the generic algorithm that
    works on any CPD type implementing log_prob). Pure-discrete networks
    will dispatch to variable elimination in v2.0 implementation (a
    performance optimisation invisible at the API level).
    """

    def __init__(self, dag):
        self._dag = dag

    def query(self, evidence: dict | None = None, *, query: str,
              n_samples: int = 10000, seed: int | None = None) -> QueryResult:
        """P(query | evidence) via likelihood weighting.

        Algorithm: walk topological order. At each non-evidence node, sample
        from its CPD conditional on the already-sampled parents. At each
        evidence node, *clamp* the value to the evidence and accumulate
        log P(evidence | sampled parents) into the row's weight. Finally
        resample rows by exp(weights).
        """
        evidence = evidence or {}
        rng = np.random.default_rng(seed)

        samples = pd.DataFrame(index=range(n_samples))
        log_w = np.zeros(n_samples)

        for node in nx.topological_sort(self._dag):
            cpd = self._dag.parameters[node]
            parents = self._dag._parent_order.get(node, [])
            X_pa = (samples[parents].copy() if parents
                    else pd.DataFrame(index=samples.index))

            if node in evidence:
                # Clamp; weight by P(evidence | parents in sample).
                ev_val = evidence[node]
                y_obs = pd.Series([ev_val] * n_samples, index=samples.index)
                lp = cpd.log_prob(y_obs, X_pa)
                log_w = log_w + np.asarray(lp).ravel()
                samples[node] = [ev_val] * n_samples
            else:
                # Sample as usual.
                samples[node] = cpd.sample(X_pa, n_samples=n_samples).values

        # Resample query column by normalized weights.
        max_log = log_w.max()
        w = np.exp(log_w - max_log)
        w /= w.sum()
        idx = rng.choice(n_samples, size=n_samples, replace=True, p=w)
        query_col = samples[query].values[idx]

        return QueryResult(
            samples=(np.asarray(query_col, dtype=float).ravel()
                     if pd.api.types.is_numeric_dtype(samples[query])
                     else np.asarray(query_col)),
            query=query, operation="predict",
            operation_args={"evidence": evidence},
            meta={"method": "likelihood_weighting", "n_samples": n_samples},
        )


class _DAGIntervene:
    """Rung 2 — interventional queries P(query | do(...)).

    Severs incoming edges to do-variables and forward-samples. No
    reweighting is needed (intervention is a structural modification).
    """

    def __init__(self, dag):
        self._dag = dag

    def simulate(self, do: dict, n_samples: int = 1000,
                 seed: int | None = None) -> pd.DataFrame:
        """Forward sample under intervention. Thin alias for dag.simulate(do=...)."""
        return self._dag.simulate(n_samples=n_samples, do=do, seed=seed)

    def query(self, do: dict, *, query: str, n_samples: int = 10000,
              seed: int | None = None) -> QueryResult:
        """P(query | do(...)) via forward sampling under intervention."""
        samples = self.simulate(do=do, n_samples=n_samples, seed=seed)
        col = samples[query]
        return QueryResult(
            samples=(np.asarray(col, dtype=float).ravel()
                     if pd.api.types.is_numeric_dtype(col)
                     else np.asarray(col)),
            query=query, operation="intervene",
            operation_args={"do": do},
            meta={"n_samples": n_samples},
        )


class _DAGTransforms:
    """Graph operations and transformations exposed as standalone primitives.

    Borrowed from causal-learn / R6causal: these primitives are useful
    outside the inference pipeline (causal discovery, manual analysis,
    debugging). Most are thin wrappers around networkx + pgmpy.base
    operations.
    """

    def __init__(self, dag):
        self._dag = dag

    def ancestors(self, node) -> set:
        return nx.ancestors(self._dag, node)

    def descendants(self, node) -> set:
        return nx.descendants(self._dag, node)

    def topological_order(self) -> list:
        return list(nx.topological_sort(self._dag))

    def markov_blanket(self, node) -> set:
        """Parents ∪ Children ∪ Children's-parents."""
        parents = set(self._dag.predecessors(node))
        children = set(self._dag.successors(node))
        co_parents = set()
        for c in children:
            co_parents.update(self._dag.predecessors(c))
        return (parents | children | co_parents) - {node}

    def d_separated(self, x, y, given: set | None = None) -> bool:
        """Standard graphical d-separation test.

        Wraps networkx 3.x's `is_d_separator`; pgmpy.base.DAG also has
        `active_trail_nodes` for more nuanced queries.
        """
        given = given or set()
        return nx.is_d_separator(self._dag, {x}, {y}, given)


class _DAGBootstrap:
    """Bootstrap-over-fit confidence intervals for any query.

    Inspired by dowhy.gcm.bootstrap_sampling: resample the training data,
    refit a fresh DAG of the same structure + CPD types, run the supplied
    query, and aggregate the resulting samples. The result is a
    QueryResult whose `.credible_interval(...)` reflects fit-time
    uncertainty.
    """

    def __init__(self, dag):
        self._dag = dag

    def query(self, data: pd.DataFrame, query_fn, n_bootstrap: int = 200,
              seed: int | None = None) -> QueryResult:
        """
        Parameters
        ----------
        data : pd.DataFrame
            Training data used for each bootstrap fit.
        query_fn : callable(dag) -> QueryResult
            Lambda or function applying any query to a freshly-fit DAG.
        n_bootstrap : int
            Number of bootstrap iterations.
        """
        rng = np.random.default_rng(seed)
        all_samples = []
        template = None
        for _ in range(n_bootstrap):
            seed_i = int(rng.integers(0, 2**32 - 1))
            resampled = data.sample(frac=1, replace=True, random_state=seed_i)
            fresh = self._fresh_copy()
            fresh.fit(resampled)
            r = query_fn(fresh)
            all_samples.append(np.atleast_1d(r.samples).ravel())
            if template is None:
                template = r
        combined = np.concatenate(all_samples)
        return QueryResult(
            samples=combined, query=template.query,
            operation=f"bootstrap_{template.operation}",
            operation_args=template.operation_args,
            meta={"n_bootstrap": n_bootstrap, **template.meta},
        )

    def _fresh_copy(self):
        """Same structure + same CPD types, all unfit. Ready for fit()."""
        return self._dag.copy_template(parameters="unfit")


class DAG(nx.DiGraph):
    """DAG with parameterization + Pearl-ladder query accessors.

    The graph is deliberately not a sklearn/skbase estimator. CPDs keep the
    estimator-style API; the DAG is a pgmpy-native mutable model container.

    Accessors map directly to Pearl's three rungs:
      - dag.inference      (rung 1 — associational)
      - dag.intervene      (rung 2 — interventional)
      - dag.counterfactual (rung 3 — counterfactual)
    Plus diagnostics, transforms, bootstrap, and parameters management.
    All query methods return QueryResult uniformly.
    """

    def __init__(self, ebunch=None, latents=None):
        nx.DiGraph.__init__(self, incoming_graph_data=ebunch)
        self._cpds = {}
        self._parent_order = {}
        self._latents = set(latents) if latents is not None else set()

    @cached_property
    def parameters(self):
        return _DAGParameters(self)

    @cached_property
    def schema(self):
        return _DAGSchema(self)

    @cached_property
    def inference(self):
        return _DAGInference(self)

    @cached_property
    def intervene(self):
        return _DAGIntervene(self)

    @cached_property
    def counterfactual(self):
        return _DAGCounterfactual(self)

    @cached_property
    def diagnostics(self):
        return _DAGDiagnostics(self)

    @cached_property
    def transforms(self):
        return _DAGTransforms(self)

    @cached_property
    def bootstrap(self):
        return _DAGBootstrap(self)

    def copy_template(self, *, parameters: Literal["none", "unfit", "fitted"] = "unfit"):
        """Copy graph + schema, optionally also copying CPDs.

        Modes: "none" (graph+schema only), "unfit" (cpd.clone() →
        sklearn.base.clone → type(cpd)() fallback), "fitted" (deep-copy).
        CPDs without a working clone (e.g. FunctionalCPD with captured
        Pyro callable) can't bootstrap in v2.0.
        """
        if parameters not in {"none", "unfit", "fitted"}:
            raise ValueError("parameters must be one of: none, unfit, fitted")
        new_dag = DAG(self.edges(), latents=set(self._latents))
        new_dag.add_nodes_from(self.nodes())
        self.schema.copy_to(new_dag)
        if parameters == "none":
            return new_dag

        import copy
        for var in nx.topological_sort(self):
            cpd = self.parameters[var]
            if parameters == "fitted":
                fresh = copy.deepcopy(cpd)
            elif hasattr(cpd, "clone") and callable(cpd.clone):
                # skpro / skbase / CPDAdapter style
                fresh = cpd.clone()
            else:
                # sklearn-style — clone is a function, not a method
                try:
                    from sklearn.base import clone as sk_clone
                    fresh = sk_clone(cpd)
                except Exception:
                    fresh = type(cpd)()
            new_dag.parameters.add(
                variable=var,
                cpd=fresh,
                parent_order=self._parent_order.get(var),
            )
        return new_dag

    def fit(self, data, estimator=None):
        if estimator is None:
            estimator = MLEEstimator()
        return estimator.fit(self, data)

    def simulate(self, n_samples=1000, do=None, seed=None):
        """Forward sample (optionally under intervention).

        For Pearl-rung-2 semantics prefer `dag.intervene.simulate(do=...)`;
        `do=` here is back-compat.
        """
        do = do or {}
        samples = pd.DataFrame(index=range(n_samples))
        for node in nx.topological_sort(self):
            if node in do:
                samples[node] = [do[node]] * n_samples
                continue
            cpd = self._cpds[node]
            parents = self._parent_order.get(node, [])
            X = (samples[parents].copy() if parents
                 else pd.DataFrame(index=range(n_samples)))
            samples[node] = cpd.sample(X, n_samples=n_samples).values
        return samples


class MLEEstimator:
    """Per-node MLE; walks topological order, calls cpd.fit(X, y)."""

    def fit(self, model, data, sample_weight=None):
        model.schema.infer_from_data(data, model._cpds)
        for node in nx.topological_sort(model):
            cpd = model._cpds[node]
            parents = model._parent_order.get(node, [])
            X = data[parents] if parents else pd.DataFrame(index=data.index)
            cpd.fit(X, data[node])
        return model


# --- Demos: Section A — Classic Bayesian Network user --------------------
# Structure → parameters → inference with confidence statistics.

def demo_a1_discrete_bn_lifecycle():
    print("\n" + "=" * 70)
    print("Demo A1: Discrete BN — full lifecycle (build, fit, infer)")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n = 2000
    diff = rng.choice(["easy", "hard"], n, p=[0.6, 0.4])
    intel = rng.choice(["low", "high"], n, p=[0.7, 0.3])
    probs = {
        ("easy", "low"):  [0.3, 0.4, 0.3],
        ("hard", "low"):  [0.05, 0.25, 0.7],
        ("easy", "high"): [0.9, 0.08, 0.02],
        ("hard", "high"): [0.5, 0.3, 0.2],
    }
    grade = np.array([rng.choice(["A", "B", "C"], p=probs[(d, i)])
                       for d, i in zip(diff, intel)])
    data = pd.DataFrame({"diff": diff, "intel": intel, "grade": grade})

    # Build + fit.
    dag = DAG([("diff", "grade"), ("intel", "grade")])
    dag.parameters.add(variable="diff",
                        cpd=TabularCPD(variable_card=2,
                                        state_names=[["easy", "hard"]]))
    dag.parameters.add(variable="intel",
                        cpd=TabularCPD(variable_card=2,
                                        state_names=[["low", "high"]]))
    dag.parameters.add(variable="grade",
                        cpd=TabularCPD(variable_card=3, evidence_card=[2, 2],
                                        state_names=[["A", "B", "C"],
                                                       ["easy", "hard"],
                                                       ["low", "high"]]),
                        parent_order=["diff", "intel"])
    dag.fit(data)

    # Verify recovered parameters match truth.
    grade_table = dag.parameters["grade"].values_
    # Columns are flattened parent combinations (diff outer, intel inner):
    # 0 = (easy, low), 1 = (easy, high), 2 = (hard, low), 3 = (hard, high).
    # Rows are grade states: 0=A, 1=B, 2=C.
    print(f"  grade CPT rows=(A,B,C), cols=(easy_low, easy_high, hard_low, hard_high):")
    print(f"    {grade_table.round(2)}")
    p_a_easy_high = grade_table[0, 1]
    print(f"  P(grade=A | diff=easy, intel=high) = {p_a_easy_high:.3f}  "
          f"(truth: 0.90, error: {abs(p_a_easy_high - 0.9):.3f})")

    # Forward sampling for a marginal query.
    sims = dag.simulate(n_samples=10000, seed=0)
    p_grade = sims["grade"].value_counts(normalize=True).to_dict()
    print(f"  Marginal P(grade): {dict(sorted(p_grade.items()))}")

    # Verify dag.parameters and pgmpy-native template copy work.
    assert dag.parameters is dag.parameters     # cached_property
    cloned = dag.copy_template(parameters="unfit")
    assert cloned is not dag and set(cloned.nodes()) == set(dag.nodes())
    assert cloned.schema["grade"].states == ("A", "B", "C")
    print(f"  Lifecycle: build → fit → sample → copy_template all functional.")


def demo_a2_hybrid_bn_third_party():
    print("\n" + "=" * 70)
    print("Demo A2: Hybrid BN — third-party regressor + classifier as CPDs")
    print("=" * 70)
    print("  A network with continuous and discrete nodes, using skpro's")
    print("  GLMRegressor and sklearn's RandomForestClassifier as CPDs")
    print("  — drop-in without any pgmpy-specific wrapping.")

    from skpro.regression.linear import GLMRegressor
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 1500
    # SCM: age ~ N(40, 10); income ~ N(2*age, 5); risk = f(age, income) binary.
    age = rng.normal(40, 10, n)
    income = 2 * age + rng.normal(0, 5, n)
    logit = -0.05 * income + 0.5
    risk = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    data = pd.DataFrame({"age": age, "income": income, "risk": risk})

    dag = DAG([("age", "income"), ("age", "risk"), ("income", "risk")])
    dag.parameters.add(variable="age",  cpd=LinearGaussianCPD())
    dag.parameters.add(variable="income", cpd=GLMRegressor(),
                        parent_order=["age"])
    dag.parameters.add(variable="risk",
                        cpd=RandomForestClassifier(n_estimators=30,
                                                     random_state=0),
                        parent_order=["age", "income"])

    print(f"\n  CPD types: age={type(dag.parameters['age']).__name__}, "
          f"income={type(dag.parameters['income']).__name__}, "
          f"risk={type(dag.parameters['risk']).__name__}")
    dag.fit(data)
    print(f"  All three CPDs fit via dag.fit() through the same MLEEstimator.")

    # Forward sample from the fitted DAG.
    sims = dag.simulate(n_samples=2000, seed=0)
    print(f"\n  Forward sample (n=2000):")
    print(f"    age   mean = {sims['age'].mean():.2f}     (truth ≈ 40)")
    print(f"    income mean = {sims['income'].mean():.2f}  (truth ≈ 80)")
    print(f"    P(risk=1)   = {sims['risk'].mean():.3f}  "
          f"(truth ≈ {(risk == 1).mean():.3f})")

    # Bonus — sklearn.Pipeline as a CPD. Any Pipeline whose final step is a
    # classifier/regressor satisfies the contract. Useful for mixed-type
    # parent preprocessing (one-hot, scale, etc.) without leaving the
    # CPD abstraction.
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("scale", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=20,
                                                     random_state=0))])
    dag2 = DAG([("age", "income"), ("age", "risk"), ("income", "risk")])
    dag2.parameters.add(variable="age",  cpd=LinearGaussianCPD())
    dag2.parameters.add(variable="income", cpd=LinearGaussianCPD(),
                         parent_order=["age"])
    dag2.parameters.add(variable="risk", cpd=pipe,
                         parent_order=["age", "income"])
    dag2.fit(data)
    print(f"\n  sklearn.Pipeline (StandardScaler → RandomForestClassifier) "
          f"as a CPD: works through the same contract. CPDAdapter auto-wraps "
          f"third-party objects on .parameters.add().")
    print(f"    fitted risk CPD type: "
          f"{type(dag2.parameters['risk']).__name__}"
          f"(wrapped: {type(dag2.parameters['risk'].wrapped).__name__})")


def demo_a3_inference_credible_intervals():
    print("\n" + "=" * 70)
    print("Demo A3: Inference with credible intervals — dag.inference.query()")
    print("=" * 70)
    print("  Discrete BN (same as A1). Query P(grade | diff=easy, intel=high)")
    print("  via dag.inference.query(). Result is a QueryResult — provides")
    print("  .point() AND .credible_interval() AND .compare_to().")

    dag = DAG([("diff", "grade"), ("intel", "grade")])
    dag.parameters.add(variable="diff", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.6], [0.4]],
        state_names=[["easy", "hard"]],
    ))
    dag.parameters.add(variable="intel", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.7], [0.3]],
        state_names=[["low", "high"]],
    ))
    dag.parameters.add(variable="grade", cpd=TabularCPD.from_values(
        variable_card=3, evidence_card=[2, 2],
        values=[[0.3, 0.9, 0.05, 0.5],   # P(A | diff, intel)
                [0.4, 0.08, 0.25, 0.3],
                [0.3, 0.02, 0.7, 0.2]],
        state_names=[["A", "B", "C"], ["easy", "hard"], ["low", "high"]],
    ), parent_order=["diff", "intel"])

    # Query: P(grade | diff=easy, intel=high). Analytic: from the CPT,
    # P(grade=A) = 0.9, P(grade=B) = 0.08, P(grade=C) = 0.02.
    result = dag.inference.query(
        evidence={"diff": "easy", "intel": "high"}, query="grade",
        n_samples=10000, seed=0,
    )
    print(f"\n  P(grade | diff=easy, intel=high) via dag.inference.query():")
    samples = pd.Series(result.samples)
    freqs = samples.value_counts(normalize=True).to_dict()
    print(f"    estimated freqs: {dict(sorted(freqs.items()))}")
    print(f"    truth:           {{'A': 0.90, 'B': 0.08, 'C': 0.02}}")
    p_A = freqs.get("A", 0.0)
    err = abs(p_A - 0.9)
    print(f"    P(grade=A) error: {err:.3f}  "
          f"({'PASS' if err < 0.05 else 'FAIL'} at 0.05)")

    # Compare to a different evidence — .compare_to() works on discrete too.
    result2 = dag.inference.query(
        evidence={"diff": "hard", "intel": "low"}, query="grade",
        n_samples=10000, seed=0,
    )
    print(f"\n  Comparison: P(grade | diff=hard, intel=low):")
    freqs2 = pd.Series(result2.samples).value_counts(normalize=True).to_dict()
    print(f"    estimated freqs: {dict(sorted(freqs2.items()))}")
    print(f"    truth:           {{'A': 0.05, 'B': 0.25, 'C': 0.70}}")
    print(f"  Same QueryResult API for any inference query — composable.")


# --- Demos: Section B — Causal inference user ----------------------------
# SCMs → interventions → counterfactuals → diagnostics.

def demo_b1_scm_intervention():
    print("\n" + "=" * 70)
    print("Demo B1: SCM — associational vs interventional distributions")
    print("=" * 70)
    print("  Build a linear-Gaussian SCM and contrast E[Z] with E[Z | do(X=0)].")

    dag = DAG([("X", "Y"), ("Y", "Z")])
    dag.parameters.add(variable="X",
                        cpd=LinearGaussianCPD.from_values(beta=[2.0], std=1.0))
    dag.parameters.add(variable="Y",
                        cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                        parent_order=["X"])
    dag.parameters.add(variable="Z",
                        cpd=LinearGaussianCPD.from_values(beta=[0.0, 3.0], std=0.3),
                        parent_order=["Y"])

    # Rung 1 (associational) via dag.inference (no evidence → marginal).
    r_assoc = dag.inference.query(query="Z", n_samples=20000, seed=0)
    # Rung 2 (interventional) via dag.intervene.
    r_intv = dag.intervene.query(do={"X": 0.0}, query="Z",
                                  n_samples=20000, seed=1)

    print(f"\n  E[Z]              = {r_assoc.point():.3f}    "
          f"(prior:  3 * 2 * 2 = 12.000)")
    print(f"  E[Z | do(X=0)]    = {r_intv.point():.3f}    "
          f"(intervention zeroes the X-path: E[Z]=0)")
    print(f"  .compare_to() — wasserstein = "
          f"{r_assoc.compare_to(r_intv)['wasserstein']:.3f}  "
          f"(confirms two distinct distributions)")
    print(f"  Both return QueryResult — same vocabulary as Demo A3 and B2.")


def demo_b2_counterfactual_end_to_end():
    print("\n" + "=" * 70)
    print("Demo B2: Counterfactual reasoning — abduction-action-prediction")
    print("=" * 70)
    print("  Same SCM as B1. Observe specific values, ask what would have")
    print("  happened under a different intervention. QueryResult unifies the")
    print("  API with predictive queries (Demo A3).")

    dag = DAG([("X", "Y"), ("Y", "Z")])
    dag.parameters.add(variable="X",
                        cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    dag.parameters.add(variable="Y",
                        cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                        parent_order=["X"])
    dag.parameters.add(variable="Z",
                        cpd=LinearGaussianCPD.from_values(beta=[0.0, 3.0], std=0.3),
                        parent_order=["Y"])

    observed = {"X": 1.0, "Y": 3.0, "Z": 10.0}
    do = {"X": 0.0}

    # Analytic abduction: U_X=1, U_Y=1, U_Z=1.
    # Counterfactual: Y_cf = 2*0 + 1 = 1; Z_cf = 3*1 + 1 = 4.
    result = dag.counterfactual.query(observed=observed, do=do, query="Z")

    print(f"\n  Observed:        {observed}")
    print(f"  Intervention:    do(X=0)")
    print(f"  QueryResult.point() = {result.point():.4f}  "
          f"(analytic Z_cf = 4.0)")
    print(f"  .meta['abducted_noise'] = "
          f"{ {k: round(v, 3) for k, v in result.meta['abducted_noise'].items()} }")
    err = abs(result.point() - 4.0)
    print(f"  |error|:         {err:.6f}  "
          f"({'PASS' if err < 1e-9 else 'FAIL'} — exact closed form)")

    # Compare to an unrelated intervention to show .compare_to.
    other = dag.counterfactual.query(observed=observed, do={"X": 1.0},
                                       query="Z")
    cmp = result.compare_to(other)
    print(f"\n  result.compare_to(do(X=1)): "
          f"wasserstein={cmp['wasserstein']:.3f}, "
          f"|Δmean|={cmp['abs_mean_diff']:.3f}")

    # Multi-world counterfactual: pass list of do dicts, share abducted
    # noise across worlds. After ChiRho's MultiWorldCounterfactual.
    worlds = dag.counterfactual.query(
        observed=observed,
        do=[{"X": -1.0}, {"X": 0.0}, {"X": 1.0}],   # three worlds
        query="Z",
    )
    print(f"\n  Multi-world counterfactual P(Z | observed, do(X=x)):")
    for r, x_val in zip(worlds, [-1.0, 0.0, 1.0]):
        print(f"    do(X={x_val:>4.1f}) → Z_cf = {r.point():.3f}   "
              f"(analytic: {3 * (2 * x_val + 1) + 1:.3f})")
    print(f"  All three worlds share the SAME abducted noise — Pearl's "
          f"twin/parallel-world semantics.")


def demo_b3_wrapped_regressor_anm_pnl():
    print("\n" + "=" * 70)
    print("Demo B3: WrappedRegressor — single class for ANM and PNL")
    print("=" * 70)
    print("  Replaces ANMWrapper + PNLWrapper with one composable class:")
    print("    WrappedRegressor(regressor)                    → ANM")
    print("    WrappedRegressor(reg, link=tanh, link_inv=arctanh) → PNL")
    print("  Both implement StructuralCPD via composition.")

    from sklearn.linear_model import LinearRegression

    # 12.b.i — ANM mode (link=None).
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(0, 1, n)
    y_anm = 2 * x + rng.normal(0, 0.5, n)
    data_anm = pd.DataFrame({"X": x, "Y": y_anm})

    anm_cpd = WrappedRegressor(LinearRegression())
    dag_anm = DAG([("X", "Y")])
    dag_anm.parameters.add(variable="X",
                            cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    dag_anm.parameters.add(variable="Y", cpd=anm_cpd, parent_order=["X"])
    dag_anm.fit(data_anm)
    print(f"\n  ANM mode (link=None):")
    print(f"    noise_type tag:        {anm_cpd.get_tag('noise_type')}")
    print(f"    regressor coef:        {anm_cpd.regressor_.coef_.round(3)} "
          f"(truth: ~2.0)")

    # 12.b.ii — PNL mode (link=tanh).
    y_pnl = np.tanh(2 * x + rng.normal(0, 0.5, n))
    data_pnl = pd.DataFrame({"X": x, "Y": y_pnl})
    pnl_cpd = WrappedRegressor(LinearRegression(),
                                  link=np.tanh, link_inv=np.arctanh)
    dag_pnl = DAG([("X", "Y")])
    dag_pnl.parameters.add(variable="X",
                            cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    dag_pnl.parameters.add(variable="Y", cpd=pnl_cpd, parent_order=["X"])
    dag_pnl.fit(data_pnl)
    print(f"\n  PNL mode (link=tanh, link_inv=arctanh):")
    print(f"    noise_type tag:        {pnl_cpd.get_tag('noise_type')}")
    print(f"    regressor coef:        {pnl_cpd.regressor_.coef_.round(3)} "
          f"(truth: ~2.0)")

    # Counterfactual on PNL: observe (X=1, Y=tanh(2.3)), do(X=0). Abduction
    # invertible via link_inv. Round-trip check.
    x_obs, y_obs = 1.0, float(np.tanh(2.3))
    u = pnl_cpd.abduct(pd.Series([y_obs]),
                       pd.DataFrame({"X": [x_obs]}))
    y_back = pnl_cpd.structural_predict(pd.DataFrame({"X": [x_obs]}),
                                          u.point().reshape(-1))
    print(f"\n  PNL roundtrip (abduct → structural_predict):")
    print(f"    observed y         = {y_obs:.4f}")
    print(f"    recovered y         = {y_back.iloc[0]:.4f}")
    print(f"    |error|             = {abs(y_back.iloc[0] - y_obs):.2e}  "
          f"({'PASS' if abs(y_back.iloc[0] - y_obs) < 1e-10 else 'FAIL'})")
    print(f"\n  Same class, two SCM shapes — protocol-over-hierarchy wins.")


def demo_b4_diagnostics_identifiability():
    print("\n" + "=" * 70)
    print("Demo B4: dag.diagnostics.identifiability_report()")
    print("=" * 70)
    print("  Static check that flags configurations where counterfactuals are")
    print("  known to be non-identified by the observational distribution.")
    print("  Promoted from dag.counterfactual.* to dag.diagnostics.* —")
    print("  identifiability isn't only a counterfactual concern (do-calculus).")

    # Pure LG chain — should be flagged (Hoyer et al. 2009).
    lg = DAG([("X", "Y"), ("Y", "Z")])
    lg.parameters.add(variable="X",
                      cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    lg.parameters.add(variable="Y",
                      cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                      parent_order=["X"])
    lg.parameters.add(variable="Z",
                      cpd=LinearGaussianCPD.from_values(beta=[0.0, 3.0], std=0.3),
                      parent_order=["Y"])

    report_lg = lg.diagnostics.identifiability_report()
    print(f"\n  Pure LG chain  → {report_lg['n_warnings']} warning(s)")
    if report_lg["warnings"]:
        w = report_lg["warnings"][0]
        print(f"    type:  {w['type']}")
        print(f"    nodes: {w['nodes']}")
        print(f"    ref:   {w['ref']}")

    # PNL SCM — should NOT be flagged (Zhang & Hyvärinen 2009 identifiable).
    from sklearn.linear_model import LinearRegression
    pnl = DAG([("X", "Y")])
    pnl.parameters.add(variable="X",
                       cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    pnl_cpd = WrappedRegressor(LinearRegression(),
                                  link=np.tanh, link_inv=np.arctanh)
    pnl_cpd.regressor_ = type("M", (), {"predict": lambda self, X: 2 * X.iloc[:, 0].values})()
    pnl_cpd.noise_residuals_ = np.zeros(100)
    pnl_cpd.noise_dist_ = Empirical(samples=pnl_cpd.noise_residuals_)
    pnl_cpd._is_fitted = True
    pnl.parameters.add(variable="Y", cpd=pnl_cpd, parent_order=["X"])
    report_pnl = pnl.diagnostics.identifiability_report()
    print(f"\n  PNL SCM       → {report_pnl['n_warnings']} warning(s)  "
          f"(expected 0 — Zhang-Hyvärinen 2009 identifiability)")

    summary_pass = (report_lg["n_warnings"] >= 1
                    and report_pnl["n_warnings"] == 0)
    print(f"\n  identifiability_report PASS — "
          f"{'OK' if summary_pass else 'FAILED'} "
          f"(LG flagged, PNL not flagged).")

    # Standalone graph primitives under dag.transforms (borrowed from
    # causal-learn / R6causal — useful outside the inference pipeline).
    print(f"\n  Standalone primitives via dag.transforms:")
    print(f"    topological_order():     {lg.transforms.topological_order()}")
    print(f"    ancestors('Z'):          {sorted(lg.transforms.ancestors('Z'), key=str)}")
    print(f"    descendants('X'):        {sorted(lg.transforms.descendants('X'), key=str)}")
    print(f"    markov_blanket('Y'):     {sorted(lg.transforms.markov_blanket('Y'), key=str)}")
    print(f"    d_separated('X', 'Z', {{'Y'}}):  {lg.transforms.d_separated('X', 'Z', {'Y'})}  "
          f"(X ⊥ Z | Y in this chain)")


def demo_b5_bootstrap_fit_uncertainty():
    print("\n" + "=" * 70)
    print("Demo B5: dag.bootstrap — fit-time confidence intervals")
    print("=" * 70)
    print("  Resample the training data, refit the DAG, run the query.")
    print("  Aggregate resulting samples into one QueryResult whose")
    print("  .credible_interval() reflects parameter uncertainty from fit.")

    # Generate data from a known LG chain.
    rng = np.random.default_rng(0)
    n = 300         # small data → meaningful fit-time uncertainty
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.5, n)
    z = 3.0 * y + rng.normal(0, 0.3, n)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})

    dag = DAG([("X", "Y"), ("Y", "Z")])
    dag.parameters.add(variable="X",  cpd=LinearGaussianCPD())
    dag.parameters.add(variable="Y",  cpd=LinearGaussianCPD(),
                        parent_order=["X"])
    dag.parameters.add(variable="Z",  cpd=LinearGaussianCPD(),
                        parent_order=["Y"])
    dag.fit(data)

    # Single counterfactual on the point-fit.
    observed = {"X": 1.0, "Y": 3.0, "Z": 10.0}
    single = dag.counterfactual.query(observed=observed,
                                        do={"X": 0.0}, query="Z")
    print(f"\n  Single-fit counterfactual (no bootstrap):")
    print(f"    point()        = {single.point():.3f}   (closed form, no CI)")

    # Bootstrap: 100 refits → distribution of Z_cf | (observed, do).
    bootstrapped = dag.bootstrap.query(
        data=data,
        query_fn=lambda d: d.counterfactual.query(
            observed=observed, do={"X": 0.0}, query="Z",
        ),
        n_bootstrap=100,
        seed=42,
    )
    lo, hi = bootstrapped.credible_interval(0.95)
    print(f"\n  Bootstrapped (n=100 refits):")
    print(f"    point()             = {bootstrapped.point():.3f}")
    print(f"    credible_interval(95%) = ({lo:.3f}, {hi:.3f})")
    print(f"    samples shape       = {bootstrapped.samples.shape}")
    print(f"  Confidence over the fitted parameters; complements the "
          f"abduction-noise uncertainty exposed in Demo B2.")


# --- Run -----------------------------------------------------------------

if __name__ == "__main__":
    # Section A — classic BN user.
    demo_a1_discrete_bn_lifecycle()
    demo_a2_hybrid_bn_third_party()
    demo_a3_inference_credible_intervals()
    # Section B — causal inference user.
    demo_b1_scm_intervention()
    demo_b2_counterfactual_end_to_end()
    demo_b3_wrapped_regressor_anm_pnl()
    demo_b4_diagnostics_identifiability()
    demo_b5_bootstrap_fit_uncertainty()
    print("\n" + "=" * 70)
    print("All demos completed.")
    print("=" * 70)
