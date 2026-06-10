"""Prototype of pgmpy v2.0 parameterization refactor.

Standalone script — DOES NOT touch the actual pgmpy source tree. Exercises:

  - Identity-free CPD classes (TabularCPD, LinearGaussianCPD).
  - DAG enrichment with parameters accessor (cached_property + skbase + nx.DiGraph
    multi-inheritance).
  - MLEEstimator orchestrating per-node cpd.fit(X, y).
  - cpd_sample / cpd_log_prob dispatch helpers.
  - Verifies a third-party skpro regressor (GLMRegressor) drops in as a CPD.

Run: python prototype.py
"""

from __future__ import annotations

import inspect
from functools import cached_property
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator as SklearnBaseEstimator
from sklearn.base import ClassifierMixin
from skbase.base import BaseObject as SkbaseBaseObject

from skpro.regression.base import BaseProbaRegressor


# =========================================================================
# 1. Module-level helpers
# =========================================================================

class CPDContractError(TypeError):
    """Raised when a CPD-like object does not satisfy the contract."""


def check_parameterization(obj: Any) -> None:
    """Validate that *obj* satisfies the pgmpy CPD contract."""
    for required in ("fit", "predict_proba"):
        if not callable(getattr(obj, required, None)):
            raise CPDContractError(
                f"{type(obj).__name__} must define {required}(...)"
            )
    fit_sig = inspect.signature(obj.fit)
    if "X" not in fit_sig.parameters or "y" not in fit_sig.parameters:
        raise CPDContractError(
            f"{type(obj).__name__}.fit must have signature fit(X, y, ...); "
            f"got {fit_sig}"
        )
    pp_sig = inspect.signature(obj.predict_proba)
    if "X" not in pp_sig.parameters:
        raise CPDContractError(
            f"{type(obj).__name__}.predict_proba must accept X; got {pp_sig}"
        )


def _get_tag(cpd, name, default=None):
    """Read a tag from a CPD, returning *default* if absent."""
    if hasattr(cpd, "get_tag"):
        try:
            val = cpd.get_tag(name)
            if val is not None:
                return val
        except (KeyError, ValueError):
            pass
    # Inline _tags dict fallback (our own CPDs).
    tags = getattr(cpd, "_tags", {})
    if name in tags:
        return tags[name]
    return default


def cpd_sample(cpd, X, n_samples=None, random_state=None):
    """Ancestral-sample from cpd given parent rows X. Returns pd.Series."""
    if n_samples is None:
        n_samples = len(X)
    # 1. native sample()
    if hasattr(cpd, "sample") and callable(cpd.sample):
        try:
            sig = inspect.signature(cpd.sample)
            # Our CPDs accept (X, n_samples=...). skpro distributions take
            # only (n_samples,). The cpd is the CPD, not a distribution.
            if "X" in sig.parameters:
                return cpd.sample(X, n_samples)
        except (ValueError, TypeError):
            pass
    # 2. Through predict_proba
    proba = cpd.predict_proba(X)
    # 2a. skpro distribution: has sample() with no required args
    if hasattr(proba, "sample") and callable(proba.sample):
        samples = proba.sample()
        if isinstance(samples, pd.DataFrame):
            samples = samples.iloc[:, 0]
        idx = getattr(X, "index", pd.RangeIndex(n_samples))
        return pd.Series(np.asarray(samples).ravel(), index=idx)
    # 2b. sklearn classifier: 2-D probability array / DataFrame
    if isinstance(proba, pd.DataFrame):
        classes = list(proba.columns)
        probs = proba.values
    else:
        proba = np.asarray(proba)
        classes = list(getattr(cpd, "classes_", range(proba.shape[1])))
        probs = proba
    rng = np.random.default_rng(random_state)
    samples = [rng.choice(classes, p=probs[i]) for i in range(probs.shape[0])]
    return pd.Series(samples)


def cpd_log_prob(cpd, y, X):
    """Compute log P(y | X) row-wise. Returns pd.Series."""
    if hasattr(cpd, "log_prob") and callable(cpd.log_prob):
        try:
            return cpd.log_prob(y, X)
        except (TypeError, ValueError):
            pass
    proba = cpd.predict_proba(X)
    # skpro continuous distribution
    if hasattr(proba, "log_pdf") and callable(proba.log_pdf):
        idx = y.index if hasattr(y, "index") else pd.RangeIndex(len(y))
        y_df = pd.DataFrame({"value": np.asarray(y).ravel()}, index=idx)
        log_pdf = proba.log_pdf(y_df)
        if isinstance(log_pdf, pd.DataFrame):
            log_pdf = log_pdf.iloc[:, 0]
        return pd.Series(np.asarray(log_pdf).ravel(), index=idx)
    # sklearn classifier
    if isinstance(proba, pd.DataFrame):
        classes = list(proba.columns)
        probs = proba.values
    else:
        proba = np.asarray(proba)
        classes = list(getattr(cpd, "classes_", range(proba.shape[1])))
        probs = proba
    class_to_col = {c: i for i, c in enumerate(classes)}
    cols = [class_to_col[v] for v in np.asarray(y)]
    row_probs = probs[np.arange(len(y)), cols]
    return pd.Series(np.log(row_probs))


# =========================================================================
# 2. TabularCPD — sklearn-classifier-style
# =========================================================================

class TabularCPD(SklearnBaseEstimator, ClassifierMixin):
    """Conditional probability table for a discrete child given discrete parents."""

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "is_linear_gaussian": False,
    }

    def __init__(self, variable_card, evidence_card=None, state_names=None):
        self.variable_card = variable_card
        self.evidence_card = evidence_card
        self.state_names = state_names

    @classmethod
    def from_values(cls, variable_card, values, evidence_card=None,
                    state_names=None):
        instance = cls(variable_card=variable_card,
                       evidence_card=evidence_card,
                       state_names=state_names)
        arr = np.asarray(values, dtype=float)
        n_parent_combos = int(np.prod(evidence_card)) if evidence_card else 1
        if arr.shape != (variable_card, n_parent_combos):
            raise ValueError(
                f"values shape {arr.shape} != expected ({variable_card}, "
                f"{n_parent_combos})"
            )
        col_sums = arr.sum(axis=0, keepdims=True)
        if (col_sums == 0).any():
            raise ValueError("Each column must have a positive sum.")
        instance.values_ = arr / col_sums
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
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight = np.asarray(sample_weight, dtype=float)

        if self.state_names is None:
            child_states = sorted(y.unique().tolist())
            parent_states = [sorted(X[c].unique().tolist()) for c in X.columns]
        else:
            child_states = list(self.state_names[0])
            parent_states = [list(s) for s in self.state_names[1:]]

        n_parent_combos = (
            int(np.prod([len(s) for s in parent_states]))
            if parent_states else 1
        )
        counts = np.zeros((self.variable_card, n_parent_combos), dtype=float)
        child_idx = {v: i for i, v in enumerate(child_states)}

        if X.shape[1] == 0:
            parent_flat = np.zeros(len(y), dtype=int)
        else:
            parent_index_lookups = [
                {v: i for i, v in enumerate(parent_states[k])}
                for k in range(X.shape[1])
            ]
            mults_list = [1]
            for s in parent_states[:-1]:
                mults_list.append(mults_list[-1] * len(s))
            mults_list = mults_list[::-1]
            parent_flat = np.zeros(len(y), dtype=int)
            for k, col in enumerate(X.columns):
                col_idx = X[col].map(parent_index_lookups[k]).to_numpy()
                parent_flat = parent_flat + col_idx * mults_list[k]

        for row in range(len(y)):
            c = child_idx[y.iat[row]]
            p = parent_flat[row]
            counts[c, p] += sample_weight[row]

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
        child_states = list(self.classes_)
        n_parent_combos = (
            int(np.prod([len(s) for s in parent_states]))
            if parent_states else 1
        )
        if X.shape[1] == 0:
            row_idx = np.zeros(len(X), dtype=int)
        else:
            mults_list = [1]
            for s in parent_states[:-1]:
                mults_list.append(mults_list[-1] * len(s))
            mults_list = mults_list[::-1]
            row_idx = np.zeros(len(X), dtype=int)
            for k, col in enumerate(X.columns):
                col_lookup = {v: i for i, v in enumerate(parent_states[k])}
                row_idx = row_idx + X[col].map(col_lookup).to_numpy() * mults_list[k]
        probs = self.values_[:, row_idx].T
        return pd.DataFrame(probs, columns=child_states, index=X.index)

    def sample(self, X, n_samples=None):
        proba = self.predict_proba(X)
        if n_samples is None:
            n_samples = len(X)
        classes = list(proba.columns)
        rng = np.random.default_rng()
        draws = [rng.choice(classes, p=proba.iloc[i].values)
                 for i in range(min(n_samples, len(proba)))]
        return pd.Series(draws, index=proba.index[:len(draws)])

    def log_prob(self, y, X):
        proba = self.predict_proba(X)
        cols = [proba.columns.get_loc(v) for v in y.values]
        row_probs = proba.values[np.arange(len(y)), cols]
        return pd.Series(np.log(row_probs), index=y.index)

    def get_tag(self, name, default=None):
        return self._tags.get(name, default)


# =========================================================================
# 3. LinearGaussianCPD — skpro-proba-regressor-style
# =========================================================================

class LinearGaussianCPD(BaseProbaRegressor):
    """P(y | X) = N(beta_[0] + sum(beta_[k+1] * X[:, k]), std_).

    Inherits from skpro.BaseProbaRegressor so it works with skpro's
    fit / predict_proba contract.
    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": True,
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
        if X_arr.size == 0:
            X_design = np.ones((n, 1))
        else:
            X_design = np.hstack([np.ones((n, 1)), X_arr.reshape(n, -1)])
        beta, *_ = np.linalg.lstsq(X_design, y_arr, rcond=None)
        residuals = y_arr - X_design @ beta
        std = float(np.sqrt(np.mean(residuals ** 2)))
        self.beta_ = beta
        self.std_ = std
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
        index = X.index if hasattr(X, "index") else pd.RangeIndex(n)
        mu = pd.DataFrame({"value": mean}, index=index)
        sigma = pd.DataFrame({"value": np.full(n, self.std_)}, index=index)
        return Normal(mu=mu, sigma=sigma)

    def sample(self, X, n_samples=None):
        dist = self.predict_proba(X)
        samples = dist.sample()
        if isinstance(samples, pd.DataFrame):
            samples = samples.iloc[:, 0]
        idx = X.index if hasattr(X, "index") else None
        return pd.Series(np.asarray(samples).ravel(), index=idx)

    def log_prob(self, y, X):
        dist = self.predict_proba(X)
        idx = y.index if hasattr(y, "index") else pd.RangeIndex(len(y))
        y_df = pd.DataFrame({"value": np.asarray(y).ravel()}, index=idx)
        log_pdf = dist.log_pdf(y_df)
        if isinstance(log_pdf, pd.DataFrame):
            log_pdf = log_pdf.iloc[:, 0]
        return pd.Series(np.asarray(log_pdf).ravel(), index=idx)

    def get_linear_gaussian_params(self):
        return self.beta_, self.std_


# =========================================================================
# 4. DAG class
# =========================================================================

class _DAGParameters:
    """CPD-registry management on a DAG."""

    def __init__(self, dag):
        self._dag = dag

    def add(self, *, variable, cpd, parent_order=None):
        check_parameterization(cpd)
        if variable not in self._dag.nodes():
            raise ValueError(f"Variable {variable!r} not in DAG.")
        expected_parents = set(self._dag.predecessors(variable))
        if parent_order is None:
            parent_order = list(self._dag.predecessors(variable))
        else:
            if set(parent_order) != expected_parents:
                raise ValueError(
                    f"parent_order {parent_order!r} doesn't match graph "
                    f"parents {sorted(expected_parents, key=str)!r}"
                )
        self._dag._cpds[variable] = cpd
        self._dag._parent_order[variable] = list(parent_order)
        return self

    def remove(self, *variables):
        for v in variables:
            self._dag._cpds.pop(v, None)
            self._dag._parent_order.pop(v, None)
        return self

    def get(self, node=None):
        if node is None:
            return [self._dag._cpds[n] for n in self._dag.nodes()
                    if n in self._dag._cpds]
        return self._dag._cpds[node]

    def keys(self):     return self._dag._cpds.keys()
    def values(self):   return self._dag._cpds.values()
    def items(self):    return self._dag._cpds.items()
    def __len__(self):  return len(self._dag._cpds)
    def __iter__(self): return iter(self._dag._cpds)
    def __contains__(self, node): return node in self._dag._cpds
    def __getitem__(self, node):
        if node not in self._dag._cpds:
            raise KeyError(f"No CPD registered for node {node!r}")
        return self._dag._cpds[node]


class DAG(nx.DiGraph, SkbaseBaseObject):
    """Directed acyclic graph with CPD parameterization.

    Inherits graph machinery from nx.DiGraph and tags/clone() infrastructure
    from skbase.BaseObject.
    """

    _tags = {"object_type": "dag"}

    def __init__(self, ebunch=None, latents=None):
        # Store init args UNCHANGED on self so skbase get_params() / clone()
        # round-trips identically. nx.DiGraph doesn't mutate incoming_graph_data.
        self.ebunch = ebunch
        self.latents = latents
        # nx.DiGraph init
        nx.DiGraph.__init__(self, incoming_graph_data=ebunch)
        # skbase init (BaseObject has __init__ that needs to run for tags)
        SkbaseBaseObject.__init__(self)
        # pgmpy-specific runtime state (not init params)
        self._cpds = {}
        self._parent_order = {}
        self._latents = set(latents) if latents is not None else set()

    @cached_property
    def parameters(self):
        return _DAGParameters(self)

    def fit(self, data, estimator=None):
        if estimator is None:
            estimator = MLEEstimator()
        return estimator.fit(self, data)

    def simulate(self, n_samples=1000, do=None, seed=None):
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
            sampled = cpd_sample(cpd, X, n_samples=n_samples, random_state=seed)
            samples[node] = sampled.values
        return samples


# =========================================================================
# 5. MLEEstimator
# =========================================================================

class MLEEstimator:
    """Generic per-node MLE. Walks topological order, calls cpd.fit(X, y)."""

    def fit(self, model, data, sample_weight=None):
        for node in nx.topological_sort(model):
            cpd = model._cpds[node]
            parents = model._parent_order.get(node, [])
            X = data[parents] if parents else pd.DataFrame(index=data.index)
            y = data[node]
            cpd.fit(X, y)
        return model


# =========================================================================
# 6. Demos
# =========================================================================

def demo_1_discrete_from_values():
    print("\n" + "=" * 70)
    print("Demo 1: Discrete BN built from values, sampled forward")
    print("=" * 70)

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
        values=[[0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7,  0.02, 0.2]],
        state_names=[["A", "B", "C"], ["easy", "hard"], ["low", "high"]],
    ), parent_order=["diff", "intel"])

    print(f"  Nodes:     {sorted(dag.nodes(), key=str)}")
    print(f"  Edges:     {sorted(dag.edges(), key=str)}")
    print(f"  CPDs registered: {len(dag.parameters)}")
    print(f"  dag.parameters['grade'] is the CPD: "
          f"{'grade' in dag.parameters}")

    samples = dag.simulate(n_samples=2000, seed=0)
    print(f"\n  Sample (head):\n{samples.head().to_string(index=False)}")
    print(f"\n  Grade marginal: "
          f"{samples['grade'].value_counts(normalize=True).round(3).to_dict()}")


def demo_2_discrete_fit_from_data():
    print("\n" + "=" * 70)
    print("Demo 2: Discrete BN fit from data via MLEEstimator")
    print("=" * 70)

    rng = np.random.default_rng(0)
    n = 2000
    diff = rng.choice(["easy", "hard"], n, p=[0.6, 0.4])
    intel = rng.choice(["low", "high"], n, p=[0.7, 0.3])
    grade_probs = {
        ("easy", "low"):  [0.3, 0.4, 0.3],
        ("hard", "low"):  [0.05, 0.25, 0.7],
        ("easy", "high"): [0.9, 0.08, 0.02],
        ("hard", "high"): [0.5, 0.3, 0.2],
    }
    grade = np.array([
        rng.choice(["A", "B", "C"], p=grade_probs[(d, i)])
        for d, i in zip(diff, intel)
    ])
    data = pd.DataFrame({"diff": diff, "intel": intel, "grade": grade})

    dag = DAG([("diff", "grade"), ("intel", "grade")])
    dag.parameters.add(variable="diff", cpd=TabularCPD(
        variable_card=2, state_names=[["easy", "hard"]]))
    dag.parameters.add(variable="intel", cpd=TabularCPD(
        variable_card=2, state_names=[["low", "high"]]))
    dag.parameters.add(variable="grade", cpd=TabularCPD(
        variable_card=3, evidence_card=[2, 2],
        state_names=[["A", "B", "C"], ["easy", "hard"], ["low", "high"]],
    ), parent_order=["diff", "intel"])

    dag.fit(data)

    print(f"  diff CPD values_ (≈ [0.6, 0.4]):\n"
          f"{dag.parameters['diff'].values_}")
    print(f"\n  grade CPD values_ (3 × 4 table):\n"
          f"{dag.parameters['grade'].values_.round(2)}")


def demo_3_hybrid_with_skpro_glm():
    print("\n" + "=" * 70)
    print("Demo 3: Hybrid BN with a real skpro GLMRegressor as a CPD")
    print("=" * 70)

    from skpro.regression.linear import GLMRegressor

    dag = DAG([("x1", "x2")])

    # x1 ~ N(0, 1)
    dag.parameters.add(
        variable="x1",
        cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0),
    )
    # x2 | x1 — fit via skpro GLMRegressor (gaussian family, identity link)
    skpro_cpd = GLMRegressor()
    dag.parameters.add(variable="x2", cpd=skpro_cpd, parent_order=["x1"])

    print(f"  x2 CPD type: {type(skpro_cpd).__name__}")
    print(f"  check_parameterization passed: True (or we'd have errored)")
    print(f"  isinstance(x2 CPD, BaseProbaRegressor): "
          f"{isinstance(skpro_cpd, BaseProbaRegressor)}")

    # Synthetic data: x2 = 2*x1 + N(0, 0.5)
    rng = np.random.default_rng(42)
    n = 500
    x1 = rng.normal(0, 1, n)
    x2 = 2 * x1 + rng.normal(0, 0.5, n)
    data = pd.DataFrame({"x1": x1, "x2": x2})

    print(f"\n  Fitting (per-node MLEEstimator)...")
    dag.fit(data)

    print(f"  x1 CPD beta_: {dag.parameters['x1'].beta_}, "
          f"std_: {dag.parameters['x1'].std_:.4f}")
    print(f"  x2 (GLMRegressor) is fitted: "
          f"{dag.parameters['x2'].is_fitted}")

    # Sample forward
    print(f"\n  Forward sampling (n=10)...")
    samples = dag.simulate(n_samples=10, seed=0)
    print(samples.round(3).to_string(index=False))


def demo_4_log_prob_dispatch():
    print("\n" + "=" * 70)
    print("Demo 4: cpd_log_prob dispatch on both Tabular and LinearGaussian")
    print("=" * 70)

    tab = TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["yes", "no"], ["lo", "hi"]],
    )
    X = pd.DataFrame({"p": ["lo", "hi"]})
    y = pd.Series(["yes", "no"])
    lp = cpd_log_prob(tab, y, X)
    print(f"  TabularCPD log_prob(['yes', 'no'] | ['lo', 'hi']) = "
          f"{lp.values.round(4).tolist()}")
    print(f"  Expected:                                          "
          f"[{np.log(0.8):.4f}, {np.log(0.8):.4f}]")

    lg = LinearGaussianCPD.from_values(beta=[1.0, 2.0], std=0.5)
    X = pd.DataFrame({"p": [0.0, 1.0]})
    y = pd.Series([1.0, 3.0])  # exactly mean
    lp = cpd_log_prob(lg, y, X)
    print(f"\n  LinearGaussianCPD log_prob (y exactly at mean):    "
          f"{lp.values.round(4).tolist()}")
    from scipy.stats import norm
    expected = norm.logpdf([1.0, 3.0], loc=[1.0, 3.0], scale=0.5)
    print(f"  Expected (scipy):                                   "
          f"{expected.round(4).tolist()}")


def demo_5_skbase_clone():
    print("\n" + "=" * 70)
    print("Demo 5: skbase clone() / get_params() on DAG")
    print("=" * 70)

    dag = DAG([("a", "b")])
    print(f"  type(dag).__mro__ (first 5): "
          f"{[c.__name__ for c in type(dag).__mro__[:5]]}")
    print(f"  hasattr(dag, 'clone'):       {hasattr(dag, 'clone')}")
    print(f"  hasattr(dag, 'get_params'):  {hasattr(dag, 'get_params')}")
    print(f"  hasattr(dag, '_tags'):       {hasattr(dag, '_tags')}")

    if hasattr(dag, "get_params"):
        try:
            params = dag.get_params()
            print(f"  dag.get_params():            {params}")
        except Exception as e:
            print(f"  dag.get_params() raised:     {type(e).__name__}: {e}")

    if hasattr(dag, "clone"):
        try:
            cloned = dag.clone()
            print(f"  dag.clone() returned:        {type(cloned).__name__}")
            print(f"  cloned is dag:               {cloned is dag}")
            print(f"  cloned has nodes:            {sorted(cloned.nodes(), key=str)}")
        except Exception as e:
            print(f"  dag.clone() raised:          {type(e).__name__}: {e}")


def demo_6_parameter_accessor_introspection():
    print("\n" + "=" * 70)
    print("Demo 6: parameters accessor dict-like behavior")
    print("=" * 70)

    dag = DAG([("a", "b"), ("c", "b")])
    dag.parameters.add(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    dag.parameters.add(variable="c", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.3], [0.7]],
        state_names=[["p", "q"]],
    ))

    print(f"  len(dag.parameters):     {len(dag.parameters)}")
    print(f"  list(dag.parameters):    {list(dag.parameters)}")
    print(f"  'a' in dag.parameters:   {'a' in dag.parameters}")
    print(f"  'b' in dag.parameters:   {'b' in dag.parameters}  (no CPD yet)")
    print(f"  dag.parameters is dag.parameters: "
          f"{dag.parameters is dag.parameters}  (cached_property)")


def demo_7_sklearn_classifier_as_cpd():
    print("\n" + "=" * 70)
    print("Demo 7: sklearn RandomForestClassifier as a discrete CPD")
    print("=" * 70)
    print("  NOTE: sklearn estimators require numeric features. The spec")
    print("  documents this as an open issue (Mixed-type parent preprocessing).")
    print("  Users must numerically encode categorical parents themselves")
    print("  until the BN provides default one-hot encoding.")

    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(0)
    n = 500
    # Numeric-encoded parents (would normally come from a preprocessor)
    age = rng.choice([0, 1], n)        # 0=young, 1=old
    education = rng.choice([0, 1], n)  # 0=low, 1=high
    # outcome correlated with parents
    risk = (age == 0) & (education == 0)
    outcome = np.where(
        risk, rng.choice([0, 1], n, p=[0.8, 0.2]),   # 0=bad, 1=good
        rng.choice([0, 1], n, p=[0.2, 0.8])
    )
    data = pd.DataFrame({"age": age, "education": education,
                         "outcome": outcome})

    dag = DAG([("age", "outcome"), ("education", "outcome")])
    dag.parameters.add(variable="age", cpd=TabularCPD(
        variable_card=2, state_names=[[0, 1]],
    ))
    dag.parameters.add(variable="education", cpd=TabularCPD(
        variable_card=2, state_names=[[0, 1]],
    ))
    rf = RandomForestClassifier(n_estimators=20, random_state=0)
    dag.parameters.add(variable="outcome", cpd=rf,
                        parent_order=["age", "education"])

    print(f"\n  outcome CPD type: {type(rf).__name__}")
    print(f"  check_parameterization passed (else would have errored)")

    dag.fit(data)
    print(f"  After fit: rf.classes_ = {rf.classes_.tolist()}")

    samples = dag.simulate(n_samples=500, seed=0)
    risky = (samples["age"] == 0) & (samples["education"] == 0)
    p_bad_when_risky = (samples.loc[risky, "outcome"] == 0).mean()
    p_bad_when_safe = (samples.loc[~risky, "outcome"] == 0).mean()
    print(f"  P(outcome=0 | age=0 & edu=0):  {p_bad_when_risky:.3f}  (expected ~0.8)")
    print(f"  P(outcome=0 | other):           {p_bad_when_safe:.3f}  (expected ~0.2)")


def demo_8_likelihood_weighting_with_skpro():
    print("\n" + "=" * 70)
    print("Demo 8: LikelihoodWeighting inference on the hybrid skpro network")
    print("=" * 70)

    from skpro.regression.linear import GLMRegressor

    # x1 ~ N(0, 1); x2 = 2*x1 + N(0, 0.5)
    dag = DAG([("x1", "x2")])
    dag.parameters.add(variable="x1",
                        cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    skpro_cpd = GLMRegressor()
    dag.parameters.add(variable="x2", cpd=skpro_cpd, parent_order=["x1"])

    rng = np.random.default_rng(7)
    n = 1000
    x1_data = rng.normal(0, 1, n)
    x2_data = 2 * x1_data + rng.normal(0, 0.5, n)
    data = pd.DataFrame({"x1": x1_data, "x2": x2_data})
    dag.fit(data)

    # Posterior over x1 given x2=4 should concentrate near x1=2.
    # Use likelihood weighting: sample x1 from prior, weight by P(x2=4 | x1).
    n_samples = 20_000
    evidence_x2 = 4.0

    # Sample x1 from its CPD (root, no parents).
    x1_samples = cpd_sample(
        dag.parameters["x1"],
        pd.DataFrame(index=range(n_samples)),
        n_samples=n_samples,
    ).values

    # Compute log-weights: log P(x2=4 | x1=sample) via the skpro CPD.
    X_for_x2 = pd.DataFrame({"x1": x1_samples})
    y_obs = pd.Series([evidence_x2] * n_samples)
    log_weights = cpd_log_prob(dag.parameters["x2"], y_obs, X_for_x2)

    # Normalize weights (log-sum-exp).
    max_log = log_weights.max()
    weights = np.exp(log_weights - max_log)
    weights /= weights.sum()

    posterior_mean = float(np.average(x1_samples, weights=weights))
    posterior_var = float(
        np.average((x1_samples - posterior_mean) ** 2, weights=weights)
    )

    # Analytic Gaussian: P(x1) = N(0, 1), P(x2|x1) = N(2*x1, 0.5).
    # P(x1 | x2=4) ∝ exp(-x1²/2) * exp(-(4-2*x1)²/(2*0.25))
    # = N(8/(2 + 4/0.25), 1/(1 + 4/0.25)) = N(8/18, 1/17) ≈ N(0.4444, ?)
    # Wait — let me recompute. Conjugate update:
    # prior precision = 1, likelihood precision (per parent unit) = (2/0.5)² = 16
    # posterior precision = 1 + 16 = 17, posterior mean = 16/17 * (4/2) = 32/17 ≈ 1.882
    # The unit-precision conjugate update for x1 given x2:
    # x2 ~ N(2*x1, 0.5²) → x1 enters with effective precision (2/0.5)² = 16
    # and "observation" (4/2) = 2.
    # Posterior: precision 1+16=17, mean (0*1 + 2*16)/17 = 32/17 ≈ 1.882
    posterior_var_analytic = 1.0 / 17.0
    posterior_mean_analytic = 32.0 / 17.0

    print(f"  Evidence: x2 = {evidence_x2}")
    print(f"  Posterior P(x1 | x2={evidence_x2}):")
    print(f"    LW estimate:       mean={posterior_mean:.3f}, "
          f"var={posterior_var:.3f}")
    print(f"    Analytic conjugate: mean={posterior_mean_analytic:.3f}, "
          f"var={posterior_var_analytic:.3f}")
    err_mean = abs(posterior_mean - posterior_mean_analytic)
    print(f"    |mean error|:      {err_mean:.3f}  "
          f"({'PASS' if err_mean < 0.1 else 'FAIL'} at threshold 0.1)")


if __name__ == "__main__":
    demo_1_discrete_from_values()
    demo_2_discrete_fit_from_data()
    demo_3_hybrid_with_skpro_glm()
    demo_4_log_prob_dispatch()
    demo_5_skbase_clone()
    demo_6_parameter_accessor_introspection()
    demo_7_sklearn_classifier_as_cpd()
    demo_8_likelihood_weighting_with_skpro()
    print("\n" + "=" * 70)
    print("All demos completed.")
    print("=" * 70)
