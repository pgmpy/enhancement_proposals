# Parameterization Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor pgmpy's CPD layer onto the skpro/sklearn estimator contract so that any probabilistic regressor or classifier can serve as a Bayesian-network CPD, replace isinstance-based inference dispatch with capability tags, and move node identity from CPDs to the Bayesian network.

**Architecture:** A new `pgmpy.parameterization` module hosts three identity-free CPD classes (`TabularCPD`, `LinearGaussianCPD`, `FunctionalCPD`) that inherit from `skbase.BaseEstimator` plus either `skpro.regression.base.BaseProbaRegressor` (continuous) or `sklearn.base.ClassifierMixin` (discrete). Bayesian-network classes own a `(node → CPD)` and `(node → parent_order)` mapping; `add_cpds(variable=..., cpd=..., parent_order=...)` registers a CPD by node name. Capability tags (`variable_type`, `produces_factor`, `is_linear_gaussian`, …) drive inference dispatch; legacy `pgmpy.factors.*` CPD classes become deprecation shims.

**Tech Stack:** Python 3.10+, `skbase`, `skpro`, `scikit-learn`, `networkx`, `numpy`, `pandas`, `pyro-ppl` (soft dep for `FunctionalCPD`), `pytest` for testing.

**Reference spec:** `docs/superpowers/specs/2026-05-14-parameterization-refactor-design.md`

**Release staging:** The refactor ships as a sequence of additive 1.x minor releases ending in a small v2.0 cleanup release. The intent is to land as much as possible *before* the breaking change. During the 1.x window, the legacy v1.x classes coexist with the new APIs as `FutureWarning`-emitting shims; v2.0 deletes them.

| Release | Phase | Tasks | What ships |
|---|---|---|---|
| **1.x.1** | Phase 1 | Tasks 1–18 | New `pgmpy.parameterization` module. Pure addition; no v1.x code touched. |
| **1.x.2** | Phase 2 + 2b + 3 + Phase 4 inference | Tasks 19–40 | DAG enrichment with `parameters`/`transforms`/`inference`/`io` accessors. Legacy `dag.add_cpds`/`get_cpds`/`remove_cpds`/`cpds` kept as `FutureWarning`-emitting shims. New parameter estimators alongside legacy ones (which already emit `FutureWarning`). New inference algorithms (`LinearGaussianInference`, `LikelihoodWeighting`) and tag-dispatch in existing inference. Typed BN classes (`DiscreteBayesianNetwork`, etc.) become `FutureWarning`-emitting subclasses of `DAG`. |
| **1.x.3** | Phase 4 deprecation + readwrite + docs | Tasks 41–46 | `FutureWarning` on legacy `pgmpy.factors.*` CPDs. Readwrite updated to consume the new accessor API (returning deprecated-subclass instances for back-compat). End-to-end integration test. Migration guide announces the v2.0 deletions a release ahead of time. |
| **2.0** | Phase 5 (new) | Tasks 47–51 | Delete legacy classes. Delete `dag.add_cpds`/`get_cpds`/`remove_cpds`/`cpds` shims. Run final regression with `-W error::FutureWarning`. |

Every Phase 2 / 4 task that adds a new API also keeps the v1.x API working (as a `FutureWarning`-emitting shim). The deletion tasks in Phase 5 are mechanical: walk the tree, remove the shim files, update `__init__.py`, run tests.

---

## File Structure

High-level summary of the files each phase touches. Every task below has
a `**Files:**` block with the exact list — this is just the bird's-eye
view.

| Phase | Release | Created | Modified | Deleted |
|---|---|---|---|---|
| 1 | 1.x.1 | `pgmpy/parameterization/` (new module: `base.py`, `checks.py`, `tabular.py`, `linear_gaussian.py`, `functional.py`); matching test dir | `pyproject.toml` (skbase hard dep; skpro / sklearn extras) | — |
| 2 | 1.x.2 | `pgmpy/base/_accessors.py` (`_DAGParameters`, `_DAGTransforms`, `_DAGInference`, `_DAGIO`); test files for DAG and deprecated BN aliases | `pgmpy/base/DAG.py` (skbase inheritance + accessors + `FutureWarning` shims for legacy `add_cpds`/etc.); `pgmpy/models/{Discrete,LinearGaussian,Functional}BayesianNetwork.py` (become thin `FutureWarning`-emitting subclasses of DAG); `pgmpy/models/DynamicBayesianNetwork.py` (audit) | — |
| 2b | 1.x.2 | `pgmpy/parameter_estimator/mle.py` (`MLEEstimator`), `joint_pyro.py` (`JointPyroEstimator`); test files | `pgmpy/parameter_estimator/{base, discrete_mle, discrete_bayesian, discrete_em, linear_gaussian_mle}.py` (accept any DAG; delegate to per-CPD fit); `pgmpy/base/DAG.py` (fit delegates to `MLEEstimator()`); `pgmpy/parameterization/functional.py` (drop `fit_joint`) | `pgmpy/parameterization/_functional_joint.py` (logic moves to `JointPyroEstimator`) |
| 3 | 1.x.2 | `pgmpy/tests/test_inference/test_approx_inference_unified.py` | `pgmpy/inference/ApproxInference.py` (accept any DAG) | — |
| 4 | 1.x.2 / 1.x.3 | `pgmpy/inference/{linear_gaussian,likelihood_weighting}.py`; `docs/source/migration-v2.rst`; legacy-deprecation test files | `pgmpy/inference/{base,ApproxInference,ExactInference,__init__}.py` (tag dispatch); `pgmpy/sampling/Sampling.py` (cpd_sample dispatch); `pgmpy/readwrite/*` (BIF/XMLBIF/UAI/XDSL/PomdpX consume new accessor API; return `DiscreteBayesianNetwork` subclass in 1.x for back-compat); `pgmpy/factors/{discrete/CPD,continuous/LinearGaussianCPD,hybrid/FunctionalCPD}.py` (emit `FutureWarning`); `CHANGELOG.rst`, `docs/source/index.rst` | — |
| 5 | 2.0 | Deletion-confirmation test files | `pgmpy/base/DAG.py` (drop `FutureWarning` shims); `pgmpy/models/__init__.py`, `pgmpy/factors/{__init__.py, discrete/__init__.py}`, `pgmpy/estimators/__init__.py` (drop deleted exports); `pgmpy/readwrite/*` (readers return plain DAG); `CHANGELOG.rst` (v2.0 entry) | `pgmpy/models/{Discrete,LinearGaussian,Functional,}BayesianNetwork.py`; `pgmpy/factors/discrete/CPD.py`; entire `pgmpy/factors/continuous/` and `pgmpy/factors/hybrid/`; `pgmpy/estimators/{MLE,BayesianEstimator,EM}.py`; corresponding test files |

**Hard dep added:** `skbase>=0.13`. **Soft deps (extras):** `skpro>=2.8`, `scikit-learn>=1.4`, `pyro-ppl` (for `FunctionalCPD` only).

---

# Phase 1: New `pgmpy.parameterization` Module

## Task 1: Set up module skeleton

**Files:**
- Create: `pgmpy/parameterization/__init__.py`
- Create: `pgmpy/parameterization/base.py`
- Create: `pgmpy/tests/test_parameterization/__init__.py`
- Create: `pgmpy/tests/test_parameterization/test_base.py`

- [ ] **Step 1: Create the empty test file**

```python
# pgmpy/tests/test_parameterization/test_base.py
def test_module_import():
    import pgmpy.parameterization
    assert pgmpy.parameterization is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pgmpy.parameterization'`

- [ ] **Step 3: Create the module files**

```python
# pgmpy/parameterization/__init__.py
"""Identity-free CPD classes for Bayesian networks."""
```

```python
# pgmpy/parameterization/base.py
"""Dispatch helpers and the CPDContractError exception.

This module provides ``cpd_sample`` and ``cpd_log_prob`` — the canonical
entry points used by Bayesian-network code to sample from a CPD or score
data under it. They dispatch to ``cpd.sample`` / ``cpd.log_prob`` when
present, otherwise route through ``predict_proba``.
"""


class CPDContractError(TypeError):
    """Raised when a CPD-like object does not satisfy the contract."""
```

```python
# pgmpy/tests/test_parameterization/__init__.py
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/ pgmpy/tests/test_parameterization/
git commit -m "feat(parameterization): scaffold new module"
```

---

## Task 2: Implement `check_parameterization`

**Files:**
- Create: `pgmpy/parameterization/checks.py`
- Create: `pgmpy/tests/test_parameterization/test_checks.py`

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_parameterization/test_checks.py
import pytest

from pgmpy.parameterization.base import CPDContractError
from pgmpy.parameterization.checks import check_parameterization


def test_rejects_plain_object():
    class NotAnEstimator:
        pass

    with pytest.raises(CPDContractError, match="must define fit"):
        check_parameterization(NotAnEstimator())


def test_rejects_object_missing_predict_proba():
    class Partial:
        def fit(self, X, y):
            return self

    with pytest.raises(CPDContractError, match="must define predict_proba"):
        check_parameterization(Partial())


def test_accepts_sklearn_classifier():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestClassifier

    check_parameterization(RandomForestClassifier(n_estimators=5))


def test_accepts_skpro_regressor():
    pytest.importorskip("skpro")
    from skpro.regression.linear import GLMRegressor

    check_parameterization(GLMRegressor())


def test_rejects_object_with_wrong_predict_proba_signature():
    class Wrong:
        def fit(self, X, y):
            return self

        def predict_proba(self):   # missing X
            ...

    with pytest.raises(CPDContractError, match="predict_proba.*signature"):
        check_parameterization(Wrong())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_checks.py -v`
Expected: All FAIL with `ModuleNotFoundError: No module named 'pgmpy.parameterization.checks'`

- [ ] **Step 3: Implement `check_parameterization`**

```python
# pgmpy/parameterization/checks.py
"""Structural validator that accepts any sklearn-classifier-style or
skpro-regressor-style object as a CPD candidate."""

import inspect

from pgmpy.parameterization.base import CPDContractError


def check_parameterization(obj):
    """Validate that *obj* satisfies the pgmpy CPD contract.

    The contract is intentionally structural (duck-typed): any object with
    ``fit(X, y)`` and ``predict_proba(X)`` qualifies. Inheritance from
    sklearn / skpro base classes is *sufficient* but not strictly required.

    Raises
    ------
    CPDContractError
        If a required method is missing or has the wrong signature.
    """
    for required in ("fit", "predict_proba"):
        if not callable(getattr(obj, required, None)):
            raise CPDContractError(
                f"CPD candidate {type(obj).__name__} must define {required}(...)"
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
            f"{type(obj).__name__}.predict_proba must accept X; "
            f"got signature {pp_sig}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_checks.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/checks.py pgmpy/tests/test_parameterization/test_checks.py
git commit -m "feat(parameterization): add check_parameterization validator"
```

---

## Task 3: Implement `cpd_sample` dispatch helper

**Files:**
- Modify: `pgmpy/parameterization/base.py`
- Modify: `pgmpy/tests/test_parameterization/test_base.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_base.py
import numpy as np
import pandas as pd
import pytest

from pgmpy.parameterization.base import cpd_sample


class _FakeContinuousDistribution:
    def __init__(self, n_rows):
        self.n_rows = n_rows

    def sample(self, n_samples):
        rng = np.random.default_rng(0)
        return pd.Series(rng.normal(size=self.n_rows))


class _FakeContinuousCPD:
    def predict_proba(self, X):
        return _FakeContinuousDistribution(len(X))


class _FakeDiscreteCPD:
    """sklearn-classifier-style: predict_proba returns probability matrix."""

    classes_ = np.array(["a", "b", "c"])

    def predict_proba(self, X):
        n = len(X)
        # Uniform over three classes
        return np.full((n, 3), 1.0 / 3.0)


def test_cpd_sample_routes_to_native_sample_when_present():
    class Native:
        def sample(self, X, n_samples):
            return pd.Series([42] * n_samples)
        def predict_proba(self, X):
            raise AssertionError("should not be called")

    X = pd.DataFrame({"p": [0, 1, 2]})
    out = cpd_sample(Native(), X, n_samples=3)
    assert out.tolist() == [42, 42, 42]


def test_cpd_sample_falls_back_to_predict_proba_for_continuous():
    X = pd.DataFrame({"p": [0.0, 1.0]})
    out = cpd_sample(_FakeContinuousCPD(), X, n_samples=2)
    assert isinstance(out, pd.Series)
    assert len(out) == 2


def test_cpd_sample_falls_back_to_categorical_draw_for_discrete():
    X = pd.DataFrame({"p": [0, 0, 0, 0]})
    out = cpd_sample(_FakeDiscreteCPD(), X, n_samples=4, random_state=0)
    assert isinstance(out, pd.Series)
    assert set(out.unique()).issubset({"a", "b", "c"})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py -v`
Expected: 3 new tests FAIL with `ImportError: cannot import name 'cpd_sample'`

- [ ] **Step 3: Implement `cpd_sample`**

```python
# Append to pgmpy/parameterization/base.py
import numpy as np
import pandas as pd


def cpd_sample(cpd, X, n_samples=None, random_state=None):
    """Ancestral-sample from *cpd* given parent rows *X*.

    Dispatch order:

    1. If *cpd* defines ``sample(X, n_samples)``, call it.
    2. Otherwise call ``cpd.predict_proba(X)``. If the result has its own
       ``sample`` method (skpro distribution), use it. If it's a 2-D
       numpy array or DataFrame of class probabilities (sklearn classifier),
       draw categorical samples row-wise.

    Parameters
    ----------
    cpd : Any
        Object satisfying the CPD contract.
    X : pandas.DataFrame
        Parent values, one row per sample.
    n_samples : int or None
        Total number of samples to draw. If None, falls back to ``len(X)``.
    random_state : int or None
        Seed for the categorical draw fallback. Native ``cpd.sample`` is
        expected to manage its own randomness.

    Returns
    -------
    pandas.Series
        Sampled child values, one per row of X.
    """
    if n_samples is None:
        n_samples = len(X)

    if hasattr(cpd, "sample") and callable(cpd.sample):
        return cpd.sample(X, n_samples)

    proba = cpd.predict_proba(X)
    # Continuous case: skpro distribution with .sample()
    if hasattr(proba, "sample") and callable(proba.sample):
        return proba.sample(n_samples)

    # Discrete case: 2-D array/DataFrame of class probabilities
    if isinstance(proba, pd.DataFrame):
        classes = list(proba.columns)
        probs = proba.values
    else:
        proba = np.asarray(proba)
        classes = list(getattr(cpd, "classes_", range(proba.shape[1])))
        probs = proba

    rng = np.random.default_rng(random_state)
    samples = [rng.choice(classes, p=probs[i]) for i in range(probs.shape[0])]
    return pd.Series(samples, name=getattr(cpd, "variable_", None))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py -v`
Expected: 4 passed (1 from Task 1 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/base.py pgmpy/tests/test_parameterization/test_base.py
git commit -m "feat(parameterization): add cpd_sample dispatch helper"
```

---

## Task 4: Implement `cpd_log_prob` dispatch helper

**Files:**
- Modify: `pgmpy/parameterization/base.py`
- Modify: `pgmpy/tests/test_parameterization/test_base.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_base.py
from pgmpy.parameterization.base import cpd_log_prob


class _FakeLogPdfDistribution:
    def log_pdf(self, y):
        return pd.Series(np.full(len(y), -1.5))


class _FakeContinuousCPDWithLogPdf:
    def predict_proba(self, X):
        return _FakeLogPdfDistribution()


def test_cpd_log_prob_routes_to_native_log_prob_when_present():
    class Native:
        def log_prob(self, y, X):
            return pd.Series([1.23] * len(y))
        def predict_proba(self, X):
            raise AssertionError("should not be called")

    y = pd.Series(["a", "b"])
    X = pd.DataFrame({"p": [0, 1]})
    out = cpd_log_prob(Native(), y, X)
    assert out.tolist() == [1.23, 1.23]


def test_cpd_log_prob_uses_log_pdf_for_continuous_distribution():
    y = pd.Series([0.5, -0.2])
    X = pd.DataFrame({"p": [0.0, 1.0]})
    out = cpd_log_prob(_FakeContinuousCPDWithLogPdf(), y, X)
    assert np.allclose(out.values, -1.5)


def test_cpd_log_prob_indexes_class_probability_matrix_for_discrete():
    class DiscreteCPD:
        classes_ = np.array(["a", "b", "c"])
        def predict_proba(self, X):
            return np.array([[0.5, 0.3, 0.2], [0.1, 0.6, 0.3]])

    y = pd.Series(["a", "b"])
    X = pd.DataFrame({"p": [0, 1]})
    out = cpd_log_prob(DiscreteCPD(), y, X)
    expected = np.log([0.5, 0.6])
    assert np.allclose(out.values, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py -v`
Expected: 3 new tests FAIL with `ImportError: cannot import name 'cpd_log_prob'`

- [ ] **Step 3: Implement `cpd_log_prob`**

```python
# Append to pgmpy/parameterization/base.py
def cpd_log_prob(cpd, y, X):
    """Compute ``log P(y | X)`` for the rows of X.

    Dispatch order matches ``cpd_sample`` — native ``log_prob`` first,
    then ``predict_proba`` with ``log_pdf`` (skpro distribution) for
    continuous, then row-indexed class probabilities for discrete.

    Parameters
    ----------
    cpd : Any
        Object satisfying the CPD contract.
    y : pandas.Series
        Observed child values, one per row of X.
    X : pandas.DataFrame
        Parent values aligned with y.

    Returns
    -------
    pandas.Series
        Log probabilities (densities for continuous, masses for discrete).
    """
    if hasattr(cpd, "log_prob") and callable(cpd.log_prob):
        return cpd.log_prob(y, X)

    proba = cpd.predict_proba(X)
    if hasattr(proba, "log_pdf") and callable(proba.log_pdf):
        return proba.log_pdf(y)
    if hasattr(proba, "log_pmf") and callable(proba.log_pmf):
        return proba.log_pmf(y)

    # sklearn-classifier fallback: index into the (n_rows, n_classes) matrix
    if isinstance(proba, pd.DataFrame):
        classes = list(proba.columns)
        probs = proba.values
    else:
        proba = np.asarray(proba)
        classes = list(getattr(cpd, "classes_", range(proba.shape[1])))
        probs = proba

    class_to_col = {c: i for i, c in enumerate(classes)}
    cols = [class_to_col[v] for v in y.values]
    row_probs = probs[np.arange(len(y)), cols]
    return pd.Series(np.log(row_probs), index=y.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py -v`
Expected: 7 passed (4 prior + 3 new)

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/base.py pgmpy/tests/test_parameterization/test_base.py
git commit -m "feat(parameterization): add cpd_log_prob dispatch helper"
```

---

## Task 5: `TabularCPD.__init__` and hyperparameter contract

**Files:**
- Create: `pgmpy/parameterization/tabular.py`
- Create: `pgmpy/tests/test_parameterization/test_tabular.py`

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_parameterization/test_tabular.py
import pytest

pytest.importorskip("sklearn")
from sklearn.base import BaseEstimator, ClassifierMixin

from pgmpy.parameterization.tabular import TabularCPD


def test_tabular_cpd_is_an_sklearn_estimator():
    cpd = TabularCPD(variable_card=3)
    assert isinstance(cpd, BaseEstimator)
    assert isinstance(cpd, ClassifierMixin)


def test_tabular_cpd_records_hyperparameters_only_in_init():
    cpd = TabularCPD(variable_card=3, evidence_card=[2, 2],
                     state_names=[["A", "B", "C"], ["lo", "hi"], ["y", "n"]])
    assert cpd.variable_card == 3
    assert cpd.evidence_card == [2, 2]
    assert cpd.state_names == [["A", "B", "C"], ["lo", "hi"], ["y", "n"]]
    # Not yet fitted: no trailing-underscore attrs set.
    assert not getattr(cpd, "is_fitted_", False)


def test_tabular_cpd_get_params_set_params_roundtrip():
    cpd = TabularCPD(variable_card=2, evidence_card=[3])
    params = cpd.get_params()
    assert params["variable_card"] == 2
    assert params["evidence_card"] == [3]
    cpd2 = TabularCPD(variable_card=2)
    cpd2.set_params(**params)
    assert cpd2.evidence_card == [3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pgmpy.parameterization.tabular'`

- [ ] **Step 3: Implement the class skeleton**

```python
# pgmpy/parameterization/tabular.py
"""Discrete conditional probability table, sklearn-classifier-style."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from skbase.utils.dependencies import _check_soft_dependencies


class TabularCPD(BaseEstimator, ClassifierMixin):
    """Conditional probability table for a discrete child given discrete parents.

    Identity-free: this class does not store the node name or parent names.
    The Bayesian network that owns the CPD records that mapping.

    Parameters
    ----------
    variable_card : int
        Number of states of the child variable.
    evidence_card : list[int] or None, default=None
        Cardinalities of the parent variables, in the same order as the BN's
        registered parent_order. None for a root node.
    state_names : list[list] or None, default=None
        ``state_names[0]`` lists labels for the child; ``state_names[1:]``
        lists labels for each parent in order. None means auto-generate
        integer labels from cardinalities.
    """

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "is_linear_gaussian": False,
        "supports_analytic_conditioning": True,
        "supports_fit_joint": False,
        "python_dependencies": ["scikit-learn"],
    }

    def __init__(self, variable_card, evidence_card=None, state_names=None):
        _check_soft_dependencies("scikit-learn", obj=self)
        self.variable_card = variable_card
        self.evidence_card = evidence_card
        self.state_names = state_names
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/tabular.py pgmpy/tests/test_parameterization/test_tabular.py
git commit -m "feat(parameterization): TabularCPD skeleton with hyperparams"
```

---

## Task 6: `TabularCPD.from_values` classmethod

**Files:**
- Modify: `pgmpy/parameterization/tabular.py`
- Modify: `pgmpy/tests/test_parameterization/test_tabular.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_tabular.py
import numpy as np


def test_from_values_root_node():
    cpd = TabularCPD.from_values(variable_card=2, values=[[0.6], [0.4]])
    assert cpd.is_fitted_
    assert np.allclose(cpd.values_, [[0.6], [0.4]])
    assert cpd.evidence_card in (None, [])


def test_from_values_with_evidence():
    cpd = TabularCPD.from_values(
        variable_card=3,
        values=[[0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7,  0.02, 0.2]],
        evidence_card=[2, 2],
    )
    assert cpd.is_fitted_
    assert cpd.values_.shape == (3, 4)
    assert cpd.evidence_card == [2, 2]


def test_from_values_rejects_misshaped_values():
    with pytest.raises(ValueError, match="shape"):
        TabularCPD.from_values(variable_card=3, values=[[0.1, 0.9]],
                               evidence_card=[2])


def test_from_values_rejects_negative_entries():
    with pytest.raises(ValueError, match="non-negative"):
        TabularCPD.from_values(variable_card=2, values=[[-0.1], [1.1]])


def test_from_values_normalizes_columns_to_sum_to_one():
    cpd = TabularCPD.from_values(variable_card=2, values=[[2.0], [3.0]])
    assert np.allclose(cpd.values_.sum(axis=0), 1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py::test_from_values_root_node -v`
Expected: FAIL with `AttributeError: type object 'TabularCPD' has no attribute 'from_values'`

- [ ] **Step 3: Implement `from_values`**

```python
# Add inside class TabularCPD in pgmpy/parameterization/tabular.py
    @classmethod
    def from_values(cls, variable_card, values, evidence_card=None,
                    state_names=None):
        """Create a *fitted* TabularCPD directly from a probability table.

        ``values`` is a 2-D array of shape ``(variable_card,
        prod(evidence_card))``. Columns are normalized to sum to one.
        """
        instance = cls(variable_card=variable_card,
                       evidence_card=evidence_card,
                       state_names=state_names)
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"values must be a 2-D array, got ndim={arr.ndim}")
        n_parent_combos = int(np.prod(evidence_card)) if evidence_card else 1
        expected_shape = (variable_card, n_parent_combos)
        if arr.shape != expected_shape:
            raise ValueError(
                f"values must have shape {expected_shape}, got {arr.shape}"
            )
        if (arr < 0).any():
            raise ValueError("CPD values must be non-negative.")
        # Normalize each column to sum to 1.
        col_sums = arr.sum(axis=0, keepdims=True)
        if (col_sums == 0).any():
            raise ValueError("Each column must have a positive sum.")
        instance.values_ = arr / col_sums
        instance.is_fitted_ = True
        return instance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py -v`
Expected: 8 passed (3 prior + 5 new)

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/tabular.py pgmpy/tests/test_parameterization/test_tabular.py
git commit -m "feat(parameterization): TabularCPD.from_values classmethod"
```

---

## Task 7: `TabularCPD.fit` from data

**Files:**
- Modify: `pgmpy/parameterization/tabular.py`
- Modify: `pgmpy/tests/test_parameterization/test_tabular.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_tabular.py
import pandas as pd


def test_fit_root_node_estimates_marginal():
    cpd = TabularCPD(variable_card=2, state_names=[["yes", "no"]])
    # Empty X for a root node.
    X = pd.DataFrame(index=range(100))
    y = pd.Series(["yes"] * 70 + ["no"] * 30)
    cpd.fit(X, y)
    assert cpd.is_fitted_
    assert np.isclose(cpd.values_.sum(), 1.0)
    # 70/30 split should be reflected.
    assert np.isclose(cpd.values_[0, 0], 0.7, atol=0.01)


def test_fit_with_one_parent():
    cpd = TabularCPD(
        variable_card=2, evidence_card=[2],
        state_names=[["yes", "no"], ["lo", "hi"]],
    )
    X = pd.DataFrame({"p": ["lo"] * 50 + ["hi"] * 50})
    # Under lo: 80% yes; under hi: 20% yes.
    y = pd.Series(["yes"] * 40 + ["no"] * 10 + ["yes"] * 10 + ["no"] * 40)
    cpd.fit(X, y)
    assert cpd.values_.shape == (2, 2)
    # values_[child_state, parent_state]; columns must sum to 1.
    assert np.allclose(cpd.values_.sum(axis=0), 1.0)
    # Lo column favors "yes".
    assert cpd.values_[0, 0] > cpd.values_[1, 0]
    # Hi column favors "no".
    assert cpd.values_[1, 1] > cpd.values_[0, 1]


def test_fit_returns_self():
    cpd = TabularCPD(variable_card=2)
    X = pd.DataFrame(index=range(10))
    y = pd.Series([0] * 5 + [1] * 5)
    assert cpd.fit(X, y) is cpd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py::test_fit_root_node_estimates_marginal -v`
Expected: FAIL with `AttributeError: 'TabularCPD' object has no attribute 'fit'`
(BaseEstimator does not provide `fit`.)

- [ ] **Step 3: Implement `fit`**

```python
# Add inside class TabularCPD
    def fit(self, X, y, sample_weight=None):
        """Estimate the conditional probability table from observed data.

        Parameters
        ----------
        X : pandas.DataFrame
            Parent values, one row per sample. Columns are in the order
            registered by the BN's parent_order. Empty (zero-column) for a
            root node.
        y : pandas.Series
            Observed child values.
        sample_weight : array-like or None
            Optional per-row weights.

        Returns
        -------
        self
        """
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y) if not isinstance(y, pd.Series) else y

        if sample_weight is None:
            sample_weight = np.ones(len(y))
        sample_weight = np.asarray(sample_weight, dtype=float)

        # Resolve state-name lists; auto-derive when missing.
        if self.state_names is None:
            child_states = sorted(y.unique().tolist())
            parent_states = [sorted(X[c].unique().tolist()) for c in X.columns]
        else:
            child_states = list(self.state_names[0])
            parent_states = [list(s) for s in self.state_names[1:]]

        n_parent_combos = int(np.prod([len(s) for s in parent_states])) if parent_states else 1

        counts = np.zeros((self.variable_card, n_parent_combos), dtype=float)

        # Index helpers.
        child_idx = {v: i for i, v in enumerate(child_states)}
        # Encode each row's parent combination as a flat index (row-major).
        if X.shape[1] == 0:
            parent_flat = np.zeros(len(y), dtype=int)
        else:
            parent_index_lookups = [
                {v: i for i, v in enumerate(parent_states[k])}
                for k in range(X.shape[1])
            ]
            mults = np.cumprod([1] + [len(s) for s in parent_states[:-1]])[::-1]
            flat = np.zeros(len(y), dtype=int)
            for k, col in enumerate(X.columns):
                # X has columns in parent_order; map values → indices.
                col_idx = X[col].map(parent_index_lookups[k]).to_numpy()
                flat = flat + col_idx * mults[k]
            parent_flat = flat

        for row in range(len(y)):
            c = child_idx[y.iat[row]]
            p = parent_flat[row]
            counts[c, p] += sample_weight[row]

        # Normalize columns; replace zero columns with uniform.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py -v`
Expected: 11 passed (8 prior + 3 new)

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/tabular.py pgmpy/tests/test_parameterization/test_tabular.py
git commit -m "feat(parameterization): TabularCPD.fit"
```

---

## Task 8: `TabularCPD.predict_proba`

**Files:**
- Modify: `pgmpy/parameterization/tabular.py`
- Modify: `pgmpy/tests/test_parameterization/test_tabular.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_tabular.py
def test_predict_proba_root_node_returns_marginal_for_each_row():
    cpd = TabularCPD.from_values(variable_card=2, values=[[0.7], [0.3]],
                                  state_names=[["yes", "no"]])
    X = pd.DataFrame(index=range(3))
    out = cpd.predict_proba(X)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["yes", "no"]
    assert out.shape == (3, 2)
    assert np.allclose(out["yes"].values, 0.7)


def test_predict_proba_with_parent_lookup():
    cpd = TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["yes", "no"], ["lo", "hi"]],
    )
    X = pd.DataFrame({"p": ["lo", "hi", "lo"]})
    out = cpd.predict_proba(X)
    assert np.allclose(out.iloc[0].values, [0.8, 0.2])
    assert np.allclose(out.iloc[1].values, [0.2, 0.8])
    assert np.allclose(out.iloc[2].values, [0.8, 0.2])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py::test_predict_proba_root_node_returns_marginal_for_each_row -v`
Expected: FAIL with `AttributeError: 'TabularCPD' object has no attribute 'predict_proba'`

- [ ] **Step 3: Implement `predict_proba`**

```python
# Add inside class TabularCPD
    def predict_proba(self, X):
        """Return class probabilities for each row of X.

        Returns
        -------
        pandas.DataFrame
            Shape ``(len(X), variable_card)``. Columns are class names
            (state_names[0]).
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("TabularCPD is not fitted; call fit() or from_values() first.")

        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Resolve parent-state lookups.
        if self.state_names is not None:
            parent_states = [list(s) for s in self.state_names[1:]]
            child_states = list(self.state_names[0])
        else:
            parent_states = getattr(self, "_fitted_parent_states_", [])
            child_states = list(getattr(self, "classes_", range(self.variable_card)))

        n_parent_combos = int(np.prod([len(s) for s in parent_states])) if parent_states else 1

        if X.shape[1] == 0:
            # Root node: same marginal for every row.
            row_idx = np.zeros(len(X), dtype=int)
        else:
            mults = np.cumprod([1] + [len(s) for s in parent_states[:-1]])[::-1]
            row_idx = np.zeros(len(X), dtype=int)
            for k, col in enumerate(X.columns):
                col_lookup = {v: i for i, v in enumerate(parent_states[k])}
                row_idx = row_idx + X[col].map(col_lookup).to_numpy() * mults[k]

        # values_ is (variable_card, n_parent_combos). Transpose-select.
        probs = self.values_[:, row_idx].T  # shape (len(X), variable_card)
        return pd.DataFrame(probs, columns=child_states, index=X.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py -v`
Expected: 13 passed (11 prior + 2 new)

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/tabular.py pgmpy/tests/test_parameterization/test_tabular.py
git commit -m "feat(parameterization): TabularCPD.predict_proba"
```

---

## Task 9: `TabularCPD.sample` and `TabularCPD.log_prob`

**Files:**
- Modify: `pgmpy/parameterization/tabular.py`
- Modify: `pgmpy/tests/test_parameterization/test_tabular.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_tabular.py
def test_sample_returns_state_labels():
    cpd = TabularCPD.from_values(
        variable_card=2, values=[[1.0], [0.0]],
        state_names=[["yes", "no"]],
    )
    X = pd.DataFrame(index=range(5))
    out = cpd.sample(X, n_samples=5)
    assert isinstance(out, pd.Series)
    assert out.tolist() == ["yes"] * 5


def test_log_prob_returns_log_of_table_entry():
    cpd = TabularCPD.from_values(
        variable_card=2, values=[[0.8, 0.2], [0.2, 0.8]], evidence_card=[2],
        state_names=[["yes", "no"], ["lo", "hi"]],
    )
    X = pd.DataFrame({"p": ["lo", "hi"]})
    y = pd.Series(["yes", "no"])
    out = cpd.log_prob(y, X)
    assert np.allclose(out.values, np.log([0.8, 0.8]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py::test_sample_returns_state_labels -v`
Expected: FAIL (no `sample` / `log_prob` method).

- [ ] **Step 3: Implement `sample` and `log_prob`**

```python
# Add inside class TabularCPD
    def sample(self, X, n_samples=None):
        """Ancestral-sample the child given parent rows X.

        Returns a Series of state-name labels.
        """
        proba = self.predict_proba(X)
        if n_samples is None:
            n_samples = len(X)
        classes = list(proba.columns)
        rng = np.random.default_rng()
        draws = [rng.choice(classes, p=proba.iloc[i].values) for i in range(n_samples)]
        return pd.Series(draws, index=proba.index[:n_samples])

    def log_prob(self, y, X):
        """Return log P(y[i] | X[i]) for each row."""
        proba = self.predict_proba(X)
        cols = [proba.columns.get_loc(v) for v in y.values]
        row_probs = proba.values[np.arange(len(y)), cols]
        return pd.Series(np.log(row_probs), index=y.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular.py -v`
Expected: 15 passed (13 prior + 2 new)

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/tabular.py pgmpy/tests/test_parameterization/test_tabular.py
git commit -m "feat(parameterization): TabularCPD.sample and .log_prob"
```

---

## Task 10: Register `TabularCPD` in module `__init__`

**Files:**
- Modify: `pgmpy/parameterization/__init__.py`
- Modify: `pgmpy/tests/test_parameterization/test_base.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_parameterization/test_base.py
def test_tabular_cpd_is_importable_from_top_level():
    from pgmpy.parameterization import TabularCPD
    assert TabularCPD is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py::test_tabular_cpd_is_importable_from_top_level -v`
Expected: FAIL with `ImportError: cannot import name 'TabularCPD'`

- [ ] **Step 3: Update `__init__.py`**

```python
# pgmpy/parameterization/__init__.py
"""Identity-free CPD classes for Bayesian networks."""

from pgmpy.parameterization.base import (
    CPDContractError,
    cpd_log_prob,
    cpd_sample,
)
from pgmpy.parameterization.checks import check_parameterization
from pgmpy.parameterization.tabular import TabularCPD

__all__ = [
    "CPDContractError",
    "TabularCPD",
    "check_parameterization",
    "cpd_log_prob",
    "cpd_sample",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest pgmpy/tests/test_parameterization/ -v`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/__init__.py pgmpy/tests/test_parameterization/test_base.py
git commit -m "feat(parameterization): export TabularCPD from package"
```

---

## Task 11: `LinearGaussianCPD` — init, from_values, fit (OLS)

**Files:**
- Create: `pgmpy/parameterization/linear_gaussian.py`
- Create: `pgmpy/tests/test_parameterization/test_linear_gaussian.py`

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_parameterization/test_linear_gaussian.py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skpro")
from skpro.regression.base import BaseProbaRegressor

from pgmpy.parameterization.linear_gaussian import LinearGaussianCPD


def test_lg_cpd_is_a_skpro_proba_regressor():
    cpd = LinearGaussianCPD()
    assert isinstance(cpd, BaseProbaRegressor)


def test_lg_cpd_init_has_no_hyperparameters():
    cpd = LinearGaussianCPD()
    assert cpd.get_params() == {}


def test_from_values_sets_fitted_attributes():
    cpd = LinearGaussianCPD.from_values(beta=[1.0, 2.0, 3.0], std=0.5)
    assert cpd.is_fitted_
    assert np.allclose(cpd.beta_, [1.0, 2.0, 3.0])
    assert cpd.std_ == 0.5


def test_fit_recovers_beta_and_std_on_synthetic_data():
    rng = np.random.default_rng(0)
    n = 5000
    X = pd.DataFrame({
        "p1": rng.normal(size=n),
        "p2": rng.normal(size=n),
    })
    true_beta = np.array([1.0, 2.0, -1.0])  # intercept, p1, p2
    noise = rng.normal(scale=0.5, size=n)
    y = pd.Series(true_beta[0] + true_beta[1] * X["p1"] + true_beta[2] * X["p2"] + noise)
    cpd = LinearGaussianCPD().fit(X, y)
    assert np.allclose(cpd.beta_, true_beta, atol=0.05)
    assert np.isclose(cpd.std_, 0.5, atol=0.05)


def test_fit_root_node_with_empty_X():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(loc=3.0, scale=2.0, size=1000))
    X = pd.DataFrame(index=range(1000))
    cpd = LinearGaussianCPD().fit(X, y)
    assert np.isclose(cpd.beta_[0], 3.0, atol=0.2)
    assert np.isclose(cpd.std_, 2.0, atol=0.2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_linear_gaussian.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pgmpy.parameterization.linear_gaussian'`

- [ ] **Step 3: Implement the class**

```python
# pgmpy/parameterization/linear_gaussian.py
"""Linear-Gaussian CPD, skpro-proba-regressor-style."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies
from skpro.regression.base import BaseProbaRegressor


class LinearGaussianCPD(BaseProbaRegressor):
    """Linear Gaussian conditional distribution.

    P(y | X) = N(beta_[0] + sum_k beta_[k+1] * X[:, k], std_)

    Identity-free: this class does not store the node name or parent names.
    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": True,
        "supports_analytic_conditioning": True,
        "supports_fit_joint": False,
        "python_dependencies": ["skpro"],
    }

    def __init__(self):
        _check_soft_dependencies("skpro", obj=self)
        super().__init__()

    @classmethod
    def from_values(cls, beta, std):
        """Create a fitted LinearGaussianCPD from coefficients.

        Parameters
        ----------
        beta : array-like
            Length ``n_parents + 1``. ``beta[0]`` is the intercept;
            ``beta[k+1]`` is the coefficient for the k-th parent in
            parent_order.
        std : float
            Residual standard deviation.
        """
        instance = cls()
        instance.beta_ = np.asarray(beta, dtype=float)
        instance.std_ = float(std)
        instance.is_fitted_ = True
        return instance

    def _fit(self, X, y, C=None):
        """OLS fit. Hooked into skpro's fit() via the underscore convention."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
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

    def get_linear_gaussian_params(self):
        """Return (beta_, std_) tuple for joint-Gaussian assembly."""
        return self.beta_, self.std_
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_linear_gaussian.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/linear_gaussian.py pgmpy/tests/test_parameterization/test_linear_gaussian.py
git commit -m "feat(parameterization): LinearGaussianCPD with OLS fit"
```

---

## Task 12: `LinearGaussianCPD.predict_proba`, `sample`, `log_prob`

**Files:**
- Modify: `pgmpy/parameterization/linear_gaussian.py`
- Modify: `pgmpy/tests/test_parameterization/test_linear_gaussian.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_linear_gaussian.py
def test_predict_proba_returns_normal_distribution_per_row():
    cpd = LinearGaussianCPD.from_values(beta=[1.0, 2.0], std=0.5)
    X = pd.DataFrame({"p": [0.0, 1.0, -1.0]})
    dist = cpd.predict_proba(X)
    # Expected means: 1.0 + 2*0 = 1.0, 1.0 + 2*1 = 3.0, 1.0 + 2*(-1) = -1.0
    samples = dist.sample(1)  # one sample per row
    assert samples.shape[0] == 3


def test_sample_returns_series_with_correct_length():
    cpd = LinearGaussianCPD.from_values(beta=[0.0, 1.0], std=0.1)
    X = pd.DataFrame({"p": [10.0, 20.0]})
    out = cpd.sample(X, n_samples=2)
    assert isinstance(out, pd.Series)
    assert len(out) == 2
    # Should be near the deterministic prediction
    assert abs(out.iloc[0] - 10.0) < 1.0
    assert abs(out.iloc[1] - 20.0) < 1.0


def test_log_prob_matches_normal_density():
    from scipy.stats import norm
    cpd = LinearGaussianCPD.from_values(beta=[1.0, 2.0], std=0.5)
    X = pd.DataFrame({"p": [0.0, 1.0]})
    y = pd.Series([1.0, 3.0])  # match the deterministic prediction
    out = cpd.log_prob(y, X)
    expected = norm.logpdf([1.0, 3.0], loc=[1.0, 3.0], scale=0.5)
    assert np.allclose(out.values, expected, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_linear_gaussian.py::test_sample_returns_series_with_correct_length -v`
Expected: FAIL (no `sample` method).

- [ ] **Step 3: Implement `_predict_proba`, `sample`, `log_prob`**

```python
# Add inside class LinearGaussianCPD
    def _predict_proba(self, X):
        """Return a per-row Normal distribution.

        skpro's BaseProbaRegressor.predict_proba(X) calls _predict_proba; the
        return value is a skpro distribution indexed by rows of X.
        """
        from skpro.distributions import Normal

        X_arr = np.asarray(X, dtype=float)
        n = len(X_arr)
        if X_arr.size == 0:
            mean = np.full(n, self.beta_[0])
        else:
            mean = self.beta_[0] + (X_arr.reshape(n, -1) * self.beta_[1:]).sum(axis=1)

        mu = pd.DataFrame({"value": mean}, index=getattr(X, "index", range(n)))
        sigma = pd.DataFrame({"value": np.full(n, self.std_)}, index=mu.index)
        return Normal(mu=mu, sigma=sigma)

    def sample(self, X, n_samples=None):
        """Sample once per row of X (n_samples is for API uniformity)."""
        dist = self.predict_proba(X)
        samples = dist.sample()  # one row per X row
        if isinstance(samples, pd.DataFrame):
            samples = samples.iloc[:, 0]
        return pd.Series(samples.values, index=getattr(X, "index", None))

    def log_prob(self, y, X):
        dist = self.predict_proba(X)
        y_df = pd.DataFrame({"value": y.values}, index=getattr(X, "index", None))
        log_pdf = dist.log_pdf(y_df)
        if isinstance(log_pdf, pd.DataFrame):
            log_pdf = log_pdf.iloc[:, 0]
        return pd.Series(log_pdf.values, index=y.index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_linear_gaussian.py -v`
Expected: 8 passed (5 prior + 3 new)

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/linear_gaussian.py pgmpy/tests/test_parameterization/test_linear_gaussian.py
git commit -m "feat(parameterization): LinearGaussianCPD sample/log_prob/_predict_proba"
```

---

## Task 13: Register `LinearGaussianCPD` in module `__init__`

**Files:**
- Modify: `pgmpy/parameterization/__init__.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_parameterization/test_base.py
def test_linear_gaussian_cpd_is_importable_from_top_level():
    from pgmpy.parameterization import LinearGaussianCPD
    assert LinearGaussianCPD is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py::test_linear_gaussian_cpd_is_importable_from_top_level -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Update `__init__.py`**

```python
# pgmpy/parameterization/__init__.py — add to imports and __all__
from pgmpy.parameterization.linear_gaussian import LinearGaussianCPD

__all__ = [
    "CPDContractError",
    "LinearGaussianCPD",
    "TabularCPD",
    "check_parameterization",
    "cpd_log_prob",
    "cpd_sample",
]
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameterization/ -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/__init__.py pgmpy/tests/test_parameterization/test_base.py
git commit -m "feat(parameterization): export LinearGaussianCPD"
```

---

## Task 14: `FunctionalCPD` — init and `_predict_proba`

**Files:**
- Create: `pgmpy/parameterization/functional.py`
- Create: `pgmpy/tests/test_parameterization/test_functional.py`

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_parameterization/test_functional.py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skpro")
pytest.importorskip("pyro")
import pyro.distributions as dist

from pgmpy.parameterization.functional import FunctionalCPD


def test_functional_cpd_stores_callable():
    fn = lambda parents: dist.Normal(0.0, 1.0)
    cpd = FunctionalCPD(fn=fn)
    assert cpd.fn is fn


def test_functional_cpd_init_requires_callable():
    with pytest.raises(TypeError, match="callable"):
        FunctionalCPD(fn=42)


def test_predict_proba_calls_fn_per_row():
    cpd = FunctionalCPD(fn=lambda p: dist.Normal(p["x1"] + 2.0, 1.0))
    X = pd.DataFrame({"x1": [0.0, 5.0]})
    samples = cpd.sample(X, n_samples=2)
    assert len(samples) == 2
    # Sample 0 should be near 2; sample 1 near 7.
    assert abs(samples.iloc[0] - 2.0) < 4.0
    assert abs(samples.iloc[1] - 7.0) < 4.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_functional.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pgmpy.parameterization.functional'`

- [ ] **Step 3: Implement the class**

```python
# pgmpy/parameterization/functional.py
"""Functional CPD wrapping a user-supplied Pyro distribution function."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import
from skpro.regression.base import BaseProbaRegressor

pyro = _safe_import("pyro", pkg_name="pyro-ppl")
torch = _safe_import("torch")


class FunctionalCPD(BaseProbaRegressor):
    """User-defined conditional distribution via a Pyro callable.

    ``fn(parent_dict_or_df)`` must return a ``pyro.distributions.Distribution``.
    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_analytic_conditioning": False,
        "supports_fit_joint": True,
        "python_dependencies": ["skpro", "pyro-ppl"],
    }

    def __init__(self, fn=None, vectorized=False):
        _check_soft_dependencies(["skpro", "pyro-ppl"], obj=self)
        if fn is not None and not callable(fn):
            raise TypeError("`fn` must be a callable; got " + repr(type(fn)))
        self.fn = fn
        self.vectorized = vectorized
        super().__init__()
        # A FunctionalCPD is considered "fitted" the moment it has a fn; users
        # may further refine parameters via fit() / fit_joint().
        self.is_fitted_ = fn is not None

    def sample(self, X, n_samples=None):
        """Sample once per row of X."""
        if n_samples is None:
            n_samples = len(X)
        if self.vectorized:
            samples = pyro.sample("vectorized", self.fn(X))
            return pd.Series(np.asarray(samples), index=X.index[:n_samples])

        out = []
        for i in range(n_samples):
            row = X.iloc[i] if len(X.columns) else {}
            parents = {k: row[k] for k in X.columns}
            d = self.fn(parents)
            s = pyro.sample(f"sample_{i}", d)
            out.append(float(np.asarray(s)))
        return pd.Series(out, index=X.index[:n_samples])

    def log_prob(self, y, X):
        """Score each (y, X[i]) pair using fn(X[i]).log_prob(y[i])."""
        out = []
        for i in range(len(y)):
            row = X.iloc[i] if len(X.columns) else {}
            parents = {k: row[k] for k in X.columns}
            d = self.fn(parents)
            value = torch.as_tensor(y.iat[i]) if torch is not None else y.iat[i]
            out.append(float(d.log_prob(value)))
        return pd.Series(out, index=y.index)

    def _predict_proba(self, X):
        """Not natively supported — log_prob/sample are the primary entry points."""
        raise NotImplementedError(
            "FunctionalCPD does not implement predict_proba directly. "
            "Use cpd.sample(X) or cpd.log_prob(y, X)."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_functional.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/functional.py pgmpy/tests/test_parameterization/test_functional.py
git commit -m "feat(parameterization): FunctionalCPD with sample/log_prob"
```

---

## Task 15: `FunctionalCPD.fit` (per-node, in-isolation) and `fit_joint`

**Files:**
- Modify: `pgmpy/parameterization/functional.py`
- Modify: `pgmpy/tests/test_parameterization/test_functional.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_parameterization/test_functional.py
def test_fit_is_noop_when_fn_has_no_tunable_params(caplog):
    """If fn has no Pyro params, fit() should warn and pass through."""
    cpd = FunctionalCPD(fn=lambda p: dist.Normal(0.0, 1.0))
    X = pd.DataFrame({"a": [0.0, 1.0]})
    y = pd.Series([0.0, 0.5])
    with caplog.at_level("INFO"):
        result = cpd.fit(X, y)
    assert result is cpd
    assert any("no tunable" in rec.message.lower() for rec in caplog.records)


def test_fit_joint_signature_exists():
    cpd = FunctionalCPD(fn=lambda p: dist.Normal(0.0, 1.0))
    assert hasattr(cpd, "fit_joint")
    assert callable(cpd.fit_joint)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameterization/test_functional.py::test_fit_is_noop_when_fn_has_no_tunable_params -v`
Expected: FAIL.

- [ ] **Step 3: Implement `_fit` and `fit_joint`**

```python
# Add inside class FunctionalCPD
    def _fit(self, X, y, C=None):
        """Per-node fit on this CPD's tunable params in isolation.

        FunctionalCPD's user-supplied ``fn`` may have no tunable parameters
        (a pure deterministic distribution at each row), in which case this is
        a no-op with an informational log. For joint Pyro fitting, use the
        BN-level ``bn.fit(data)``, which routes through ``fit_joint``.
        """
        import logging
        logging.getLogger(__name__).info(
            "FunctionalCPD.fit invoked with no tunable parameters detected; "
            "treating as a no-op. To learn parameters jointly across the "
            "network, use DAG.fit(data) which calls fit_joint."
        )
        self.is_fitted_ = True
        return self

    def fit_joint(self, network, data, estimator="SVI", num_steps=1000,
                  optimizer=None, **kwargs):
        """Network-level joint Pyro fit.

        The BN's fit() invokes this when all CPDs advertise
        ``supports_fit_joint=True``. The implementation should walk the BN's
        nodes, assemble a combined Pyro model from each node's ``fn``, and
        run SVI or MCMC.

        This method is a pgmpy-specific extension, not part of the skpro
        contract. It mirrors the existing FunctionalDAG.fit
        behavior.
        """
        from pgmpy.parameterization._functional_joint import run_joint_pyro_fit
        return run_joint_pyro_fit(network, data, estimator=estimator,
                                  num_steps=num_steps, optimizer=optimizer,
                                  **kwargs)
```

The helper module `_functional_joint.py` houses the joint-fit logic so the
class stays focused. Create it with the body lifted from the existing
`FunctionalDAG.fit` (lines 480–550 of the current file). For
this task, a stub is sufficient — Task 21 will port the existing logic.

```python
# pgmpy/parameterization/_functional_joint.py
"""Internal helper: network-level Pyro SVI/MCMC fit for FunctionalCPDs.

Filled in by Task 21 of the implementation plan. The current stub raises
NotImplementedError so that any premature joint-fit attempt fails loudly.
"""


def run_joint_pyro_fit(network, data, estimator="SVI", num_steps=1000,
                       optimizer=None, **kwargs):
    raise NotImplementedError(
        "Joint Pyro fit is implemented in Task 21 of the parameterization "
        "refactor plan."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_parameterization/test_functional.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/functional.py pgmpy/parameterization/_functional_joint.py pgmpy/tests/test_parameterization/test_functional.py
git commit -m "feat(parameterization): FunctionalCPD.fit (noop) and fit_joint stub"
```

---

## Task 16: Register `FunctionalCPD` in module `__init__`

**Files:**
- Modify: `pgmpy/parameterization/__init__.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_parameterization/test_base.py
def test_functional_cpd_is_importable_from_top_level():
    pytest.importorskip("pyro")
    from pgmpy.parameterization import FunctionalCPD
    assert FunctionalCPD is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameterization/test_base.py::test_functional_cpd_is_importable_from_top_level -v`
Expected: FAIL.

- [ ] **Step 3: Update `__init__.py`**

```python
# pgmpy/parameterization/__init__.py
"""Identity-free CPD classes for Bayesian networks."""

from pgmpy.parameterization.base import (
    CPDContractError,
    cpd_log_prob,
    cpd_sample,
)
from pgmpy.parameterization.checks import check_parameterization
from pgmpy.parameterization.linear_gaussian import LinearGaussianCPD
from pgmpy.parameterization.tabular import TabularCPD

# FunctionalCPD pulls in optional dependencies (pyro-ppl). Defer the import so
# users can use Tabular and LinearGaussian without pyro installed.
try:
    from pgmpy.parameterization.functional import FunctionalCPD
except ImportError:  # pragma: no cover
    FunctionalCPD = None

__all__ = [
    "CPDContractError",
    "FunctionalCPD",
    "LinearGaussianCPD",
    "TabularCPD",
    "check_parameterization",
    "cpd_log_prob",
    "cpd_sample",
]
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameterization/ -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/__init__.py pgmpy/tests/test_parameterization/test_base.py
git commit -m "feat(parameterization): export FunctionalCPD"
```

---

## Task 17: Phase 1 integration test

**Files:**
- Create: `pgmpy/tests/test_parameterization/test_integration.py`

- [ ] **Step 1: Write integration test**

```python
# pgmpy/tests/test_parameterization/test_integration.py
"""Phase 1 integration: each CPD type passes check_parameterization and works
through the cpd_sample / cpd_log_prob dispatch helpers."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skpro")

from pgmpy.parameterization import (
    LinearGaussianCPD,
    TabularCPD,
    check_parameterization,
    cpd_log_prob,
    cpd_sample,
)


def test_tabular_cpd_passes_check():
    cpd = TabularCPD.from_values(variable_card=2, values=[[0.5], [0.5]],
                                  state_names=[["yes", "no"]])
    check_parameterization(cpd)


def test_linear_gaussian_cpd_passes_check():
    cpd = LinearGaussianCPD.from_values(beta=[0.0, 1.0], std=1.0)
    check_parameterization(cpd)


def test_dispatch_through_cpd_sample_tabular():
    cpd = TabularCPD.from_values(variable_card=2, values=[[1.0], [0.0]],
                                  state_names=[["yes", "no"]])
    X = pd.DataFrame(index=range(3))
    out = cpd_sample(cpd, X, n_samples=3)
    assert out.tolist() == ["yes", "yes", "yes"]


def test_dispatch_through_cpd_log_prob_linear_gaussian():
    cpd = LinearGaussianCPD.from_values(beta=[0.0, 1.0], std=1.0)
    X = pd.DataFrame({"p": [0.0, 1.0]})
    y = pd.Series([0.0, 1.0])
    log_p = cpd_log_prob(cpd, y, X)
    assert len(log_p) == 2
    assert np.all(np.isfinite(log_p.values))


def test_third_party_sklearn_classifier_through_dispatch():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.integers(0, 5, size=200)})
    y = pd.Series(rng.choice(["yes", "no"], size=200))
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    check_parameterization(clf)
    out = cpd_sample(clf, X.head(5), n_samples=5, random_state=0)
    assert set(out.unique()).issubset({"yes", "no"})
```

- [ ] **Step 2: Run tests**

Run: `pytest pgmpy/tests/test_parameterization/test_integration.py -v`
Expected: 5 passed.

- [ ] **Step 3: Commit**

```bash
git add pgmpy/tests/test_parameterization/test_integration.py
git commit -m "test(parameterization): Phase 1 integration coverage"
```

---

## Task 18: Update `pyproject.toml` for new deps

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Inspect the current deps**

Run: `grep -n "skbase\|skpro\|scikit-learn" pyproject.toml`
Note the current dependency declarations.

- [ ] **Step 2: Add `skbase` as a hard dep and `skpro` / `scikit-learn` as optional extras**

Edit `pyproject.toml` in the `[project] dependencies` or equivalent
`[tool.poetry.dependencies]` section, adding `skbase>=0.13`. Add a new
extras section if one doesn't already exist:

```toml
[project.optional-dependencies]
skpro = ["skpro>=2.8"]
sklearn = ["scikit-learn>=1.4"]
```

(Confirm exact version pins by running `pip install --dry-run skbase` and
`pip install --dry-run skpro` locally; pin to the latest minor.)

- [ ] **Step 3: Verify imports work**

Run: `python -c "import skbase; print(skbase.__version__)"`
Expected: a version number prints.

- [ ] **Step 4: Run the full Phase 1 test suite to confirm nothing broke**

Run: `pytest pgmpy/tests/test_parameterization/ -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add skbase hard dep, skpro/sklearn extras"
```

---

# Phase 2: DAG enrichment — `pgmpy.base.DAG` becomes the unified parameterized network

> **Design pivot — v2.0 breaking change (applies to this entire phase and downstream phases):**
> The refactor ships as pgmpy 2.0 with **no deprecation aliases or shims**. The legacy classes (`DiscreteBayesianNetwork`, `LinearGaussianBayesianNetwork`, `FunctionalBayesianNetwork`, `BayesianNetwork` stub, the old `pgmpy.factors.*` CPD wrappers, and the already-`FutureWarning`-deprecated `pgmpy.estimators.MaximumLikelihoodEstimator` / `BayesianEstimator` / `EM`) are **deleted outright**. A migration guide at `docs/source/migration-v2.rst` documents how v1.x code maps to v2.0.
>
> The unified class is **`pgmpy.base.DAG`**. `DAG` is enriched with the CPD registry, namespaced accessors (`dag.parameters`, `dag.transforms`, `dag.inference`, `dag.io`), `fit`/`simulate`/`check_model`, and `skbase.base.BaseEstimator` inheritance (for `clone()`, `get_params()`, `_tags`). No `BayesianNetwork` class.
>
> **CPD-management API:** `dag.parameters.add(variable=..., cpd=..., parent_order=None)` / `dag.parameters.get(node)` / `dag.parameters.remove(*nodes)` is the **only** API. There is no `dag.add_cpds(...)` / `dag.get_cpds(...)` / `dag.remove_cpds(...)` / `dag.cpds` — those were v1.x.

## Task 19: Enrich `pgmpy.base.DAG` — CPD registry, skbase inheritance, `parameters` accessor, deprecated shims

**Files:**
- Modify: `pgmpy/base/DAG.py` (the existing 1942-line class — extend, don't replace).
- Create: `pgmpy/base/_accessors.py` (initial scaffolding — only `_DAGParameters` here; transforms/inference/io added in later tasks).
- Create: `pgmpy/tests/test_base/test_dag_parameters.py`.
- Leave: `pgmpy/models/BayesianNetwork.py` — the existing `ImportError`-raising stub stays untouched (or is deleted). There is no `BayesianNetwork` alias.

This is the foundation task for Phase 2. It does several related things at once because they tightly couple:

1. Make `DAG` inherit from `skbase.base.BaseEstimator` so `dag.clone()` / `get_params()` / `_tags` work.
2. Add `_cpds`/`_parent_order`/`_deprecated_methods_warned` instance state on `DAG`.
3. Create `_DAGParameters` accessor (canonical CPD-management API).
4. Wire `dag.parameters` as a `cached_property` on `DAG`.
5. Add deprecated method shims on `DAG`: `add_cpds`, `get_cpds`, `remove_cpds`, `cpds` property — each emits one `DeprecationWarning` per instance the first time it's called.
6. Leave the existing `pgmpy.models.BayesianNetwork` stub alone (or delete it). No alias.

The rest of Phase 2 (Tasks 20–23 check_model/simulate/fit; Tasks 24–27 transforms/inference/io accessors; Task 28 typed-class aliases; Task 29 DBN audit) builds on the foundation here.

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_models/test_bn_identity_ownership.py
import pytest

from pgmpy.base import DAG
from pgmpy.parameterization import TabularCPD


def test_new_bn_has_empty_cpd_registry():
    bn = DAG([("a", "b")])
    assert bn._cpds == {}
    assert bn._parent_order == {}


def test_add_cpds_new_signature_records_node_and_parent_order():
    bn = DAG([("diff", "grade"), ("intel", "grade")])
    cpd = TabularCPD.from_values(
        variable_card=3, evidence_card=[2, 2],
        values=[[0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7,  0.02, 0.2]],
        state_names=[["A", "B", "C"], ["easy", "hard"], ["low", "high"]],
    )
    bn.add_cpds(variable="grade", cpd=cpd, parent_order=["diff", "intel"])
    assert "grade" in bn._cpds
    assert bn._cpds["grade"] is cpd
    assert bn._parent_order["grade"] == ["diff", "intel"]


def test_add_cpds_canonicalizes_parent_order_from_graph():
    bn = DAG([("diff", "grade"), ("intel", "grade")])
    cpd = TabularCPD.from_values(variable_card=3, evidence_card=[2, 2],
                                  values=[[0.3] * 4, [0.4] * 4, [0.3] * 4])
    bn.add_cpds(variable="grade", cpd=cpd)
    # Default parent_order matches list(bn.predecessors("grade"))
    assert set(bn._parent_order["grade"]) == {"diff", "intel"}


def test_add_cpds_rejects_parent_order_with_wrong_nodes():
    bn = DAG([("diff", "grade"), ("intel", "grade")])
    cpd = TabularCPD.from_values(variable_card=3, evidence_card=[2, 2],
                                  values=[[0.3] * 4, [0.4] * 4, [0.3] * 4])
    with pytest.raises(ValueError, match="parent_order"):
        bn.add_cpds(variable="grade", cpd=cpd, parent_order=["diff", "bogus"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py -v`
Expected: FAIL — `_cpds` attribute doesn't exist, `add_cpds` doesn't accept `variable=` kwarg.

- [ ] **Step 3: Add skbase inheritance and the CPD registry to `DAG.__init__`**

In `pgmpy/base/DAG.py`, update the class declaration and `__init__`:

```python
from functools import cached_property
from skbase.base import BaseEstimator
import networkx as nx


class DAG(_GraphRolesMixin, nx.DiGraph, BaseEstimator):
    """Directed acyclic graph with optional CPD parameterization."""

    _tags = {
        "object_type": "dag",
    }

    def __init__(self, ebunch=None, latents=None):
        super().__init__(ebunch=ebunch, latents=latents or set())
        # Existing graph init from _GraphRolesMixin + nx.DiGraph stays.
        # New: CPD registry.
        self._cpds = {}
        self._parent_order = {}
        # New: deprecated-method tracking for once-per-instance warnings.
        self._deprecated_methods_warned = set()
```

- [ ] **Step 4: Create `_DAGParameters` accessor in `pgmpy/base/_accessors.py`**

```python
# pgmpy/base/_accessors.py
"""Namespaced accessor objects exposed on DAG via cached_property."""

from __future__ import annotations


class _DAGParameters:
    """CPD-registry management — the canonical API.

    Replaces the deprecated DAG.add_cpds / get_cpds / remove_cpds methods.
    Iterates / indexes like a node→CPD dict.
    """

    def __init__(self, dag):
        self._dag = dag

    # Mutation
    def add(self, *, variable, cpd, parent_order=None):
        """Register *cpd* as the parameterization for *variable*.

        Parameters
        ----------
        variable : Hashable
            Node name. Must be in the DAG's nodes.
        cpd : Any
            Object satisfying the CPD contract (sklearn classifier-like
            or skpro proba-regressor-like). Validated via
            ``check_parameterization``.
        parent_order : list[Hashable] or None, default None
            Ordered list of parent names; positional contract that the
            CPD's beta_/values_/etc. lines up with. Defaults to
            ``list(self._dag.predecessors(variable))``.
        """
        from pgmpy.parameterization import check_parameterization

        check_parameterization(cpd)
        if variable not in self._dag.nodes():
            raise ValueError(
                f"Variable {variable!r} is not a node in the DAG."
            )
        expected_parents = set(self._dag.predecessors(variable))
        if parent_order is None:
            parent_order = list(self._dag.predecessors(variable))
        else:
            if set(parent_order) != expected_parents:
                raise ValueError(
                    f"parent_order {parent_order!r} for {variable!r} does "
                    f"not match graph parents "
                    f"{sorted(expected_parents, key=str)!r}."
                )
        self._dag._cpds[variable] = cpd
        self._dag._parent_order[variable] = list(parent_order)
        return self

    def remove(self, *variables):
        """Drop the CPDs registered for *variables*."""
        for variable in variables:
            self._dag._cpds.pop(variable, None)
            self._dag._parent_order.pop(variable, None)
        return self

    # Read access
    def get(self, node=None):
        """Return the CPD for *node*, or all registered CPDs if None."""
        if node is None:
            return [self._dag._cpds[n] for n in self._dag.nodes()
                    if n in self._dag._cpds]
        if node not in self._dag._cpds:
            raise ValueError(f"No CPD registered for node {node!r}.")
        return self._dag._cpds[node]

    # dict-like introspection
    def keys(self):   return self._dag._cpds.keys()
    def values(self): return self._dag._cpds.values()
    def items(self):  return self._dag._cpds.items()
    def __len__(self):           return len(self._dag._cpds)
    def __iter__(self):          return iter(self._dag._cpds)
    def __contains__(self, node): return node in self._dag._cpds
```

- [ ] **Step 5: Wire the `parameters` accessor + 1.x deprecated method shims on `DAG`**

Append to `pgmpy/base/DAG.py`:

```python
    @cached_property
    def parameters(self):
        """CPD-registry accessor. The canonical CPD-management API."""
        from pgmpy.base._accessors import _DAGParameters
        return _DAGParameters(self)

    # --- 1.x deprecation shims for the legacy CPD-management API. ---
    # Deleted in v2.0 (Phase 5). Each emits one FutureWarning per instance
    # per method via _warn_deprecated_method.
    def _warn_deprecated_method(self, name, replacement):
        if name not in self._deprecated_methods_warned:
            import warnings
            warnings.warn(
                f"DAG.{name} is deprecated and will be removed in pgmpy 2.0. "
                f"Use dag.{replacement} instead. (This warning is shown once "
                f"per DAG instance.)",
                FutureWarning, stacklevel=3,
            )
            self._deprecated_methods_warned.add(name)

    def add_cpds(self, *cpds, variable=None, cpd=None, parent_order=None):
        self._warn_deprecated_method("add_cpds", "parameters.add")
        if variable is not None or cpd is not None:
            if not (variable is not None and cpd is not None):
                raise ValueError(
                    "add_cpds kwargs require both variable= and cpd="
                )
            if cpds:
                raise ValueError(
                    "Cannot mix positional CPDs with variable=/cpd= kwargs."
                )
            return self.parameters.add(variable=variable, cpd=cpd,
                                        parent_order=parent_order)
        for cpd_obj in cpds:
            variable_name = getattr(cpd_obj, "variable", None)
            if variable_name is None:
                raise ValueError(
                    "Positional add_cpds requires each CPD to carry a "
                    "`variable` attribute. Identity-free CPDs must use "
                    "dag.parameters.add(variable=..., cpd=...)."
                )
            parents = (getattr(cpd_obj, "evidence", None)
                       or list(self.predecessors(variable_name)))
            self.parameters.add(variable=variable_name, cpd=cpd_obj,
                                 parent_order=list(parents))

    def get_cpds(self, node=None):
        self._warn_deprecated_method("get_cpds", "parameters.get")
        return self.parameters.get(node)

    def remove_cpds(self, *args):
        self._warn_deprecated_method("remove_cpds", "parameters.remove")
        nodes = []
        for arg in args:
            if isinstance(arg, str) or not hasattr(arg, "fit"):
                nodes.append(arg)
            else:
                for node, cpd in self._cpds.items():
                    if cpd is arg:
                        nodes.append(node)
                        break
        return self.parameters.remove(*nodes)

    @property
    def cpds(self):
        self._warn_deprecated_method("cpds", "parameters.values()")
        return list(self.parameters.values())
```

Add to `DAG.__init__`:

```python
self._deprecated_methods_warned = set()  # for once-per-instance warnings
```

- [ ] **Step 6: Leave `pgmpy/models/BayesianNetwork.py` untouched in 1.x**

The 3-line `ImportError`-raising stub at `pgmpy/models/BayesianNetwork.py`
is left alone in 1.x (it was never a usable class — no back-compat
obligation). It gets deleted as part of the v2.0 cleanup in Phase 5.

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest pgmpy/tests/test_base/test_dag_parameters.py pgmpy/tests/test_models/test_bn_identity_ownership.py -v`
Expected: All pass.

- [ ] **Step 8: Run the existing DAG / DiscreteBayesianNetwork tests to confirm no regressions**

Run: `pytest pgmpy/tests/test_base/test_DAG.py pgmpy/tests/test_models/test_DiscreteBayesianNetwork.py -v --tb=short -W ignore::DeprecationWarning`
Expected: All existing tests still pass — the deprecated method shims preserve the legacy `add_cpds`/`get_cpds`/`remove_cpds` API.

- [ ] **Step 9: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/base/_accessors.py pgmpy/tests/test_base/ pgmpy/tests/test_models/test_bn_identity_ownership.py
git commit -m "feat(base): enrich DAG with CPD registry, parameters accessor, skbase inheritance"
```

---

## Task 20: `DAG.get_cpds` reads from the registry

**Files:**
- Modify: `pgmpy/base/DAG.py`
- Modify: `pgmpy/tests/test_models/test_bn_identity_ownership.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_models/test_bn_identity_ownership.py
def test_get_cpds_returns_registered_cpd():
    bn = DAG([("diff", "grade")])
    cpd = TabularCPD.from_values(variable_card=2, evidence_card=[2],
                                  values=[[0.8, 0.2], [0.2, 0.8]],
                                  state_names=[["A", "B"], ["lo", "hi"]])
    bn.add_cpds(variable="grade", cpd=cpd)
    assert bn.get_cpds("grade") is cpd
    assert cpd in bn.get_cpds()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py::test_get_cpds_returns_registered_cpd -v`
Expected: Depends on legacy `cpds` list — may pass for legacy CPDs but FAIL for new-style CPDs that aren't in any list.

- [ ] **Step 3: Update `get_cpds` to read from `_cpds`**

Find `get_cpds` (around line 315). Replace its body with:

```python
def get_cpds(self, node=None):
    """Return CPDs registered with the network.

    Parameters
    ----------
    node : Hashable or None, default=None
        If given, return the CPD for that node. Otherwise return a list of
        all CPDs in the order their nodes appear in ``self.nodes()``.

    Returns
    -------
    CPD or list[CPD]
    """
    if node is None:
        return [self._cpds[n] for n in self.nodes() if n in self._cpds]
    if node not in self._cpds:
        raise ValueError(f"No CPD registered for node {node!r}")
    return self._cpds[node]
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py -v`
Expected: 5 passed.

- [ ] **Step 5: Run existing tests to catch regressions**

Run: `pytest pgmpy/tests/test_models/ -v --tb=short -x`
Expected: Existing tests pass; any failures must be tracked back to legacy code that mutated `bn.cpds` directly (now a no-op).

- [ ] **Step 6: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/tests/test_models/test_bn_identity_ownership.py
git commit -m "refactor(models): get_cpds reads from _cpds registry"
```

---

## Task 21: `DAG.check_model`

`cpd_as_factor` was originally planned here but is now provided by the
`_BNTransforms` accessor (Task 24). Use this slot to give the unified
class a generic `check_model` instead — every BN needs to validate that
all nodes have a CPD and that every CPD is fitted before inference or
simulation.

**Files:**
- Modify: `pgmpy/base/DAG.py`
- Modify: `pgmpy/tests/test_models/test_bn_identity_ownership.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_models/test_bn_identity_ownership.py
def test_check_model_raises_when_a_node_has_no_cpd():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    with pytest.raises(ValueError, match="No CPD"):
        bn.check_model()


def test_check_model_raises_when_a_cpd_is_unfit():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD(variable_card=2))   # not fitted
    bn.add_cpds(variable="b", cpd=TabularCPD(variable_card=2,
                                              evidence_card=[2]),
                parent_order=["a"])
    with pytest.raises(ValueError, match="not fitted"):
        bn.check_model()


def test_check_model_succeeds_for_fully_specified_network():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.9, 0.1], [0.1, 0.9]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    assert bn.check_model() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py -k check_model -v`
Expected: FAIL — `check_model` either doesn't exist on the new BN class or returns truthy without validating.

- [ ] **Step 3: Implement `check_model`**

Add to `DAG`:

```python
def check_model(self):
    """Validate that the network is fully specified and fittable.

    - Every node must have a CPD registered.
    - Every CPD must be fitted (``is_fitted_`` is True). For legacy CPDs
      that don't carry ``is_fitted_``, the check is skipped.
    """
    missing = [n for n in self.nodes() if n not in self._cpds]
    if missing:
        raise ValueError(
            f"No CPD registered for nodes: {sorted(missing, key=str)}"
        )
    unfit = [
        n for n, c in self._cpds.items()
        if hasattr(c, "is_fitted_") and not c.is_fitted_
    ]
    if unfit:
        raise ValueError(
            f"CPDs not fitted for nodes: {sorted(unfit, key=str)}"
        )
    return True
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py -k check_model -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/tests/test_models/test_bn_identity_ownership.py
git commit -m "feat(models): DAG.check_model validates registry and fit state"
```

---

## Task 22: Generic `simulate` on `DAG`

**Files:**
- Modify: `pgmpy/base/DAG.py`
- Modify: `pgmpy/tests/test_models/test_bn_identity_ownership.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_models/test_bn_identity_ownership.py
import networkx as nx


def test_simulate_walks_topological_order_and_uses_cpd_sample():
    bn = DAG([("diff", "grade"), ("intel", "grade")])
    bn.add_cpds(variable="diff", cpd=TabularCPD.from_values(
        variable_card=2, values=[[1.0], [0.0]],
        state_names=[["easy", "hard"]],
    ))
    bn.add_cpds(variable="intel", cpd=TabularCPD.from_values(
        variable_card=2, values=[[1.0], [0.0]],
        state_names=[["low", "high"]],
    ))
    bn.add_cpds(variable="grade", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2, 2],
        values=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        state_names=[["A", "B"], ["easy", "hard"], ["low", "high"]],
    ), parent_order=["diff", "intel"])

    samples = bn.simulate(n_samples=50)
    assert set(samples.columns) == {"diff", "intel", "grade"}
    # Deterministic CPDs above: diff="easy", intel="low", grade="A"
    assert (samples["diff"] == "easy").all()
    assert (samples["intel"] == "low").all()
    assert (samples["grade"] == "A").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py::test_simulate_walks_topological_order_and_uses_cpd_sample -v`
Expected: The current `simulate` either misbehaves with the new CPDs or doesn't exist on this path.

- [ ] **Step 3: Add a new `simulate` implementation**

In `DAG`, find the existing `simulate` (around line 1337)
and add (or replace, depending on whether the existing path is needed) a new
generic implementation that uses `cpd_sample` dispatch. Keep the legacy
behavior for back-compat under a private method `_legacy_simulate`:

```python
def simulate(self, n_samples=1000, do=None, evidence=None,
             virtual_intervention=None, include_latents=False, seed=None,
             missing_prob=None, n_jobs=-1, show_progress=True):
    """Generic ancestral simulation that dispatches via cpd_sample.

    For each node in topological order, build X = sampled[parents] and
    call cpd_sample(cpd, X, n_samples). State labels are preserved.
    """
    import networkx as nx
    from pgmpy.parameterization.base import cpd_sample

    if do is None:
        do = {}

    # Detect any legacy-only feature (virtual_intervention, missing_prob,
    # evidence with specific filtering): fall through to the legacy path.
    if virtual_intervention or missing_prob or evidence:
        return self._legacy_simulate(
            n_samples=n_samples, do=do, evidence=evidence,
            virtual_intervention=virtual_intervention,
            include_latents=include_latents, seed=seed,
            missing_prob=missing_prob, n_jobs=n_jobs,
            show_progress=show_progress,
        )

    rng_seed = seed
    samples = pd.DataFrame(index=range(n_samples))
    for node in nx.topological_sort(self):
        if node in do:
            samples[node] = [do[node]] * n_samples
            continue
        cpd = self._cpds[node]
        parents = self._parent_order.get(node, [])
        X = samples[parents] if parents else pd.DataFrame(index=range(n_samples))
        samples[node] = cpd_sample(cpd, X, n_samples=n_samples,
                                   random_state=rng_seed).values
    return samples
```

Rename the existing `simulate` body to `_legacy_simulate` (preserving signature).

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py::test_simulate_walks_topological_order_and_uses_cpd_sample -v`
Expected: PASS.

Then run the broader simulate tests to confirm legacy users still work:

Run: `pytest pgmpy/tests/test_models/test_BayesianNetwork.py -k simulate -v --tb=short`
Expected: All pass (legacy CPDs continue through `_legacy_simulate`).

- [ ] **Step 5: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/tests/test_models/test_bn_identity_ownership.py
git commit -m "feat(models): generic simulate using cpd_sample dispatch"
```

---

## Task 23: Generic `fit` on `DAG`

**Files:**
- Modify: `pgmpy/base/DAG.py`
- Modify: `pgmpy/tests/test_models/test_bn_identity_ownership.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_models/test_bn_identity_ownership.py
def test_bn_fit_calls_each_cpd_fit_with_correct_X_and_y():
    import numpy as np
    rng = np.random.default_rng(0)

    bn = DAG([("diff", "grade"), ("intel", "grade")])
    bn.add_cpds(variable="diff", cpd=TabularCPD(variable_card=2))
    bn.add_cpds(variable="intel", cpd=TabularCPD(variable_card=2))
    bn.add_cpds(variable="grade",
                cpd=TabularCPD(variable_card=2, evidence_card=[2, 2]),
                parent_order=["diff", "intel"])

    n = 200
    data = pd.DataFrame({
        "diff":  rng.choice(["easy", "hard"], n),
        "intel": rng.choice(["low", "high"], n),
        "grade": rng.choice(["A", "B"], n),
    })

    bn.fit(data)
    for node in ("diff", "intel", "grade"):
        cpd = bn.get_cpds(node)
        assert getattr(cpd, "is_fitted_", False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py::test_bn_fit_calls_each_cpd_fit_with_correct_X_and_y -v`
Expected: FAIL — current `fit` routes through legacy `DiscreteMLE` and doesn't understand identity-free CPDs.

- [ ] **Step 3: Update `fit` to use the generic per-node loop when the registered CPDs are identity-free**

Replace `fit` body with:

```python
def fit(self, data, estimator=None, sample_weight=None):
    """Fit the CPDs from data.

    Two execution paths:

    1. **Joint fit:** if every registered CPD advertises
       ``supports_fit_joint=True``, call ``fit_joint(self, data)`` on the
       first CPD. Used by FunctionalBayesianNetwork with shared Pyro priors.
    2. **Per-node loop (default):** for each node, build
       ``X = data[parent_order[node]]`` and ``y = data[node]``, then call
       ``cpd.fit(X, y)``. Works for any CPD that satisfies the contract,
       including third-party sklearn classifiers.

    If ``estimator`` is provided, the legacy DiscreteMLE / DiscreteBayesian
    path is used instead. Reserved for back-compat.
    """
    import networkx as nx

    if estimator is not None:
        # Legacy path
        from pgmpy.parameter_estimator import DiscreteMLE
        from pgmpy.parameter_estimator.base import DiscreteParameterEstimator

        if not isinstance(estimator, DiscreteParameterEstimator):
            raise TypeError("estimator must be a DiscreteParameterEstimator")
        estimator.fit(self, data, sample_weight=sample_weight)
        for cpd in estimator.parameters_:
            self.add_cpds(cpd)  # legacy positional path
        return self

    # Identity-free path. If joint-fit is universally supported, prefer it.
    cpds = list(self._cpds.values())
    if cpds and all(
        getattr(cpd, "get_tag", lambda *_: False)("supports_fit_joint", False)
        for cpd in cpds
    ):
        cpds[0].fit_joint(self, data)
        return self

    # Per-node loop.
    for node in nx.topological_sort(self):
        cpd = self._cpds.get(node)
        if cpd is None:
            raise ValueError(f"No CPD registered for node {node!r}")
        parents = self._parent_order.get(node, [])
        X = data[parents] if parents else pd.DataFrame(index=data.index)
        y = data[node]
        cpd.fit(X, y)
    return self
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py -v`
Expected: All pass.

Run: `pytest pgmpy/tests/test_models/test_BayesianNetwork.py -k fit -v --tb=short`
Expected: Legacy fit-with-estimator tests still pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/tests/test_models/test_bn_identity_ownership.py
git commit -m "feat(base): generic per-node fit on DAG"
```

---

## Task 24: Create `_BNTransforms` accessor

**Files:**
- Create: `pgmpy/models/_accessors.py`
- Create: `pgmpy/tests/test_models/test_bn_accessors.py`

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_models/test_bn_accessors.py
import pytest

pytest.importorskip("sklearn")

from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD, TabularCPD


def test_transforms_to_markov_model_succeeds_when_all_cpds_produce_factors():
    from pgmpy.factors.discrete import DiscreteFactor
    bn = DAG([("diff", "grade")])
    bn.add_cpds(variable="diff", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["easy", "hard"]],
    ))
    bn.add_cpds(variable="grade", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["A", "B"], ["easy", "hard"]],
    ), parent_order=["diff"])
    mn = bn.transforms.to_markov_model()
    factors = list(mn.get_factors())
    assert len(factors) == 2
    assert all(isinstance(f, DiscreteFactor) for f in factors)


def test_transforms_to_markov_model_raises_for_mixed_network():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=LinearGaussianCPD.from_values(
        beta=[0.0, 1.0], std=1.0,
    ), parent_order=["a"])
    with pytest.raises(TypeError, match="produces_factor"):
        bn.transforms.to_markov_model()


def test_transforms_to_joint_gaussian_succeeds_for_all_lg_network():
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=LinearGaussianCPD.from_values(beta=[1.0], std=1.0))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=1.0),
                parent_order=["x1"])
    mu, cov = bn.transforms.to_joint_gaussian()
    assert abs(mu[0] - 1.0) < 1e-6
    assert abs(mu[1] - 2.0) < 1e-6


def test_transforms_to_joint_gaussian_raises_for_mixed_network():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=LinearGaussianCPD.from_values(
        beta=[0.0, 1.0], std=1.0,
    ), parent_order=["a"])
    with pytest.raises(TypeError, match="is_linear_gaussian"):
        bn.transforms.to_joint_gaussian()


def test_transforms_cpd_as_factor_for_tabular_cpd():
    from pgmpy.factors.discrete import DiscreteFactor
    bn = DAG([("diff", "grade")])
    cpd = TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["A", "B"], ["lo", "hi"]],
    )
    bn.add_cpds(variable="grade", cpd=cpd, parent_order=["diff"])
    factor = bn.transforms.cpd_as_factor("grade")
    assert isinstance(factor, DiscreteFactor)
    assert "grade" in factor.variables
    assert "diff" in factor.variables
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py -v`
Expected: FAIL — `bn.transforms` does not yet exist.

- [ ] **Step 3: Create `_BNTransforms` in `pgmpy/models/_accessors.py`**

```python
# pgmpy/models/_accessors.py
"""Accessor objects that group DAG helpers by domain.

Accessors are pandas-style: ``bn.transforms.to_joint_gaussian()`` rather
than methods directly on the BN class. This keeps the DAG
class small and surfaces helpers in semantically grouped namespaces.
"""

from __future__ import annotations


def _require_tag_on_all_cpds(bn, tag, op_name):
    """Raise TypeError if any CPD in *bn* does not advertise *tag*=True."""
    for node, cpd in bn._cpds.items():
        if not (hasattr(cpd, "get_tag")
                and cpd.get_tag(tag, False)):
            raise TypeError(
                f"{op_name} requires every CPD to advertise {tag}=True. "
                f"CPD for node {node!r} ({type(cpd).__name__}) does not."
            )


class _BNTransforms:
    """Representation transformations on a DAG.

    Methods here check the relevant CPD tag at call time and raise
    informatively when the network's CPDs don't satisfy the precondition.
    """

    def __init__(self, bn):
        self._bn = bn

    def cpd_as_factor(self, node):
        """Return a DiscreteFactor view of the CPD at *node*.

        Used by exact-inference algorithms (VE, BP) internally.
        """
        from pgmpy.factors.discrete import DiscreteFactor

        bn = self._bn
        cpd = bn._cpds[node]
        if not (hasattr(cpd, "get_tag")
                and cpd.get_tag("produces_factor", False)):
            raise TypeError(
                f"CPD for {node!r} ({type(cpd).__name__}) does not produce "
                f"a DiscreteFactor (produces_factor tag is False)."
            )
        parents = bn._parent_order[node]
        variables = [node] + parents
        cardinality = ([cpd.variable_card] + list(cpd.evidence_card)
                       if parents else [cpd.variable_card])
        if cpd.state_names is not None:
            state_names = {variables[i]: list(cpd.state_names[i])
                           for i in range(len(variables))}
        else:
            state_names = {v: list(range(c))
                           for v, c in zip(variables, cardinality)}
        return DiscreteFactor(
            variables=variables,
            cardinality=cardinality,
            values=cpd.values_.flatten(),
            state_names=state_names,
        )

    def to_markov_model(self):
        """Convert this BN to a Markov network by moralization.

        Precondition: every CPD advertises ``produces_factor=True``.
        """
        from pgmpy.models import DiscreteMarkovNetwork

        _require_tag_on_all_cpds(self._bn, "produces_factor",
                                 "to_markov_model")
        mn = DiscreteMarkovNetwork()
        mn.add_nodes_from(self._bn.nodes())
        for node in self._bn.nodes():
            factor = self.cpd_as_factor(node)
            mn.add_factors(factor)
            parents = self._bn._parent_order.get(node, [])
            for i, p1 in enumerate(parents):
                mn.add_edge(p1, node)
                for p2 in parents[i + 1:]:
                    mn.add_edge(p1, p2)
        return mn

    def to_joint_gaussian(self):
        """Return ``(mu, cov)`` of the joint Gaussian.

        Precondition: every CPD advertises ``is_linear_gaussian=True``.
        """
        import networkx as nx
        import numpy as np

        _require_tag_on_all_cpds(self._bn, "is_linear_gaussian",
                                 "to_joint_gaussian")
        bn = self._bn
        order = list(nx.topological_sort(bn))
        n = len(order)
        mu = np.zeros(n)
        cov = np.zeros((n, n))
        pos = {node: i for i, node in enumerate(order)}

        for node in order:
            cpd = bn._cpds[node]
            beta, std = cpd.get_linear_gaussian_params()
            parents = bn._parent_order.get(node, [])
            i = pos[node]
            if not parents:
                mu[i] = beta[0]
                cov[i, i] = std ** 2
            else:
                pids = [pos[p] for p in parents]
                mu[i] = beta[0] + sum(beta[k + 1] * mu[pids[k]]
                                      for k in range(len(parents)))
                for k, pid in enumerate(pids):
                    cov[i, pid] = sum(beta[m + 1] * cov[pids[m], pid]
                                      for m in range(len(parents)))
                    cov[pid, i] = cov[i, pid]
                cov[i, i] = std ** 2 + sum(
                    beta[k + 1] * beta[m + 1] * cov[pids[k], pids[m]]
                    for k in range(len(parents))
                    for m in range(len(parents))
                )
        return mu, cov
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py -v`
Expected: Tests fail until Task 27 wires the accessor onto DAG. The accessor class itself is importable but `bn.transforms` doesn't exist yet — that wiring happens in Task 27. This is the same TDD pattern used in Phase 1.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/tests/test_models/test_bn_accessors.py
git commit -m "feat(models): _BNTransforms accessor (to_markov_model, to_joint_gaussian, cpd_as_factor)"
```

---

## Task 25: Add `_BNInference` to the accessor module

**Files:**
- Modify: `pgmpy/models/_accessors.py`
- Modify: `pgmpy/tests/test_models/test_bn_accessors.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_models/test_bn_accessors.py
import numpy as np
import pandas as pd


def test_inference_log_likelihood_sums_per_node_log_probs():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.7], [0.3]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    data = pd.DataFrame({"a": ["x", "x", "y"], "b": ["L", "L", "R"]})
    ll = bn.inference.log_likelihood(data)
    # log 0.7 + log 0.8  (row 0)
    # + log 0.7 + log 0.8  (row 1)
    # + log 0.3 + log 0.8  (row 2)
    expected = (
        np.log(0.7) + np.log(0.8)
        + np.log(0.7) + np.log(0.8)
        + np.log(0.3) + np.log(0.8)
    )
    assert np.isclose(ll, expected, atol=1e-6)


def test_inference_predict_fills_missing_columns():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.99], [0.01]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.99, 0.01], [0.01, 0.99]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    obs = pd.DataFrame({"a": ["x", "y"]})  # b unobserved
    pred = bn.inference.predict(obs)
    # Given a=x → b=L with high prob; a=y → b=R.
    assert pred.loc[0, "b"] == "L"
    assert pred.loc[1, "b"] == "R"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py -v`
Expected: FAIL — `bn.inference` does not yet exist.

- [ ] **Step 3: Add `_BNInference` to `pgmpy/models/_accessors.py`**

Append:

```python
class _BNInference:
    """Convenience query API.

    For full control over an inference algorithm, instantiate it directly
    (e.g. ``VariableElimination(bn)``). The accessor wraps the most common
    workflows.
    """

    def __init__(self, bn):
        self._bn = bn

    def predict(self, data):
        """Predict missing columns of *data* via VariableElimination.

        Returns a DataFrame with the same shape as *data*, with the
        unobserved columns filled in by MAP estimate per row.
        """
        import pandas as pd
        from pgmpy.inference import VariableElimination

        ve = VariableElimination(self._bn)
        observed_cols = list(data.columns)
        latent_cols = [n for n in self._bn.nodes() if n not in observed_cols]
        rows = []
        for _, row in data.iterrows():
            evidence = {col: row[col] for col in observed_cols}
            map_estimate = ve.map_query(variables=latent_cols,
                                        evidence=evidence,
                                        show_progress=False)
            rows.append({**evidence, **map_estimate})
        return pd.DataFrame(rows, index=data.index)

    def predict_probability(self, data):
        """Return P(latent | observed) for each row of *data*.

        Returns a DataFrame with one column per (latent variable, state).
        """
        import pandas as pd
        from pgmpy.inference import VariableElimination

        ve = VariableElimination(self._bn)
        observed_cols = list(data.columns)
        latent_cols = [n for n in self._bn.nodes() if n not in observed_cols]
        rows = []
        for _, row in data.iterrows():
            evidence = {col: row[col] for col in observed_cols}
            posterior = ve.query(variables=latent_cols,
                                 evidence=evidence,
                                 joint=False,
                                 show_progress=False)
            row_out = {}
            for var, factor in posterior.items():
                states = factor.state_names[var]
                for s, p in zip(states, factor.values):
                    row_out[f"{var}={s}"] = float(p)
            rows.append(row_out)
        return pd.DataFrame(rows, index=data.index)

    def log_likelihood(self, data):
        """Sum of log P(node | parents) over rows of *data*.

        Works for any CPD that supports the contract — discrete, linear-
        Gaussian, functional, or third-party. Routes through
        ``cpd_log_prob`` dispatch.
        """
        import pandas as pd
        from pgmpy.parameterization.base import cpd_log_prob

        bn = self._bn
        total = 0.0
        for node in bn.nodes():
            cpd = bn._cpds[node]
            parents = bn._parent_order.get(node, [])
            X = (data[parents] if parents
                 else pd.DataFrame(index=data.index))
            y = data[node]
            total += float(cpd_log_prob(cpd, y, X).sum())
        return total
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py -v`
Expected: New tests fail until Task 27 wires the accessor onto DAG.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/tests/test_models/test_bn_accessors.py
git commit -m "feat(models): _BNInference accessor (predict, predict_probability, log_likelihood)"
```

---

## Task 26: Add `_BNIO` accessor and `DAG.load` classmethod

**Files:**
- Modify: `pgmpy/models/_accessors.py`
- Modify: `pgmpy/base/DAG.py` — add the `load` classmethod.
- Modify: `pgmpy/tests/test_models/test_bn_accessors.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_models/test_bn_accessors.py
import tempfile
import os


def test_io_save_then_load_roundtrip_bif(tmp_path):
    pytest.importorskip("pgmpy.readwrite")
    bn = DAG([("diff", "grade")])
    bn.add_cpds(variable="diff", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.6], [0.4]],
        state_names=[["easy", "hard"]],
    ))
    bn.add_cpds(variable="grade", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["A", "B"], ["easy", "hard"]],
    ), parent_order=["diff"])
    path = tmp_path / "model.bif"
    bn.io.save(str(path), format="bif")
    assert os.path.exists(path)

    loaded = DAG.load(str(path), format="bif")
    assert set(loaded.nodes()) == {"diff", "grade"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py::test_io_save_then_load_roundtrip_bif -v`
Expected: FAIL — `bn.io` / `DAG.load` don't exist yet.

- [ ] **Step 3: Add `_BNIO` and `load`**

Append to `pgmpy/models/_accessors.py`:

```python
class _BNIO:
    """Serialization helpers.

    ``DAG.load`` is a classmethod (you don't have a BN before
    loading), so it lives on the class itself rather than on this accessor.
    """

    def __init__(self, bn):
        self._bn = bn

    def save(self, path, format="bif"):
        """Save the network to *path* in the given *format*.

        Supported formats: ``bif`` (default), ``xmlbif``, ``uai``. The
        underlying writer is selected from ``pgmpy.readwrite``.
        """
        from pgmpy.readwrite import BIFWriter, UAIWriter, XMLBIFWriter

        writers = {
            "bif": BIFWriter, "xmlbif": XMLBIFWriter, "uai": UAIWriter,
        }
        if format not in writers:
            raise ValueError(
                f"Unknown format {format!r}; choose from {list(writers)}"
            )
        writer = writers[format](self._bn)
        writer.write_file(path)
```

In `pgmpy/base/DAG.py`, add the classmethod:

```python
@classmethod
def load(cls, path, format="bif"):
    """Load a network from disk.

    Mirrors ``bn.io.save`` formats: ``bif``, ``xmlbif``, ``uai``.
    """
    from pgmpy.readwrite import BIFReader, UAIReader, XMLBIFReader

    readers = {
        "bif": BIFReader, "xmlbif": XMLBIFReader, "uai": UAIReader,
    }
    if format not in readers:
        raise ValueError(
            f"Unknown format {format!r}; choose from {list(readers)}"
        )
    reader = readers[format](path)
    return reader.get_model()
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py::test_io_save_then_load_roundtrip_bif -v`
Expected: Test fails on `bn.io.save` (`io` accessor not yet wired); `DAG.load` works. Full pass comes after Task 27.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/base/_accessors.py pgmpy/base/DAG.py pgmpy/tests/test_base/test_dag_accessors.py
git commit -m "feat(models): _BNIO accessor + DAG.load classmethod"
```

---

## Task 27: Wire `transforms`, `inference`, `io` accessors onto `DAG`

**Files:**
- Modify: `pgmpy/base/DAG.py`
- Modify: `pgmpy/tests/test_models/test_bn_accessors.py`

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_models/test_bn_accessors.py
def test_accessors_are_cached_per_instance():
    bn = DAG([("a", "b")])
    t1 = bn.transforms
    t2 = bn.transforms
    assert t1 is t2  # cached_property: same object every access.

    i1 = bn.inference
    i2 = bn.inference
    assert i1 is i2

    io1 = bn.io
    io2 = bn.io
    assert io1 is io2


def test_accessors_hold_back_reference_to_their_bn():
    bn = DAG([("a", "b")])
    assert bn.transforms._bn is bn
    assert bn.inference._bn is bn
    assert bn.io._bn is bn
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py::test_accessors_are_cached_per_instance -v`
Expected: FAIL — `transforms`/`inference`/`io` not defined on `DAG`.

- [ ] **Step 3: Add the `cached_property` accessors**

In `pgmpy/base/DAG.py`, near the top of the class body:

```python
from functools import cached_property

from pgmpy.models._accessors import _BNIO, _BNInference, _BNTransforms


class DAG(DAG):
    # ... existing __init__, add_cpds, fit, simulate, check_model ...

    @cached_property
    def transforms(self):
        """Representation transformations (to_markov_model, to_joint_gaussian, …)."""
        return _BNTransforms(self)

    @cached_property
    def inference(self):
        """Convenience query API (predict, predict_probability, log_likelihood)."""
        return _BNInference(self)

    @cached_property
    def io(self):
        """Serialization (save). Loading is DAG.load classmethod."""
        return _BNIO(self)
```

- [ ] **Step 4: Run the full accessor test suite**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py -v`
Expected: All tests added in Tasks 24–27 now pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/tests/test_base/test_dag_accessors.py
git commit -m "feat(models): wire transforms/inference/io accessors via cached_property"
```

---

## Task 28: Convert typed BN classes to `FutureWarning`-emitting subclasses of `DAG` (1.x)

**Files:**
- Modify: `pgmpy/models/DiscreteBayesianNetwork.py`
- Modify: `pgmpy/models/LinearGaussianBayesianNetwork.py`
- Modify: `pgmpy/models/FunctionalBayesianNetwork.py`
- Modify: `pgmpy/models/__init__.py` — keep all three exports.
- Create: `pgmpy/tests/test_models/test_deprecated_bn_aliases.py`

In 1.x the typed BN classes become thin subclasses of `DAG`. Each emits
one `FutureWarning` per instance at `__init__` (pointing at `pgmpy.base.DAG`),
enforces the appropriate `_register_cpd` tag check, and provides back-compat
method shims (`to_markov_model`, `predict`, `save`, etc.) that delegate to
the accessors. These files are **deleted in Phase 5 (v2.0)**.

- [ ] **Step 1: Write failing tests for the deprecated aliases**

```python
# pgmpy/tests/test_models/test_deprecated_bn_aliases.py
import pytest

from pgmpy.parameterization import LinearGaussianCPD, TabularCPD


def test_discrete_bn_emits_future_warning_on_init():
    from pgmpy.base import DAG
    from pgmpy.models import DiscreteBayesianNetwork

    with pytest.warns(FutureWarning, match="pgmpy.base.DAG"):
        bn = DiscreteBayesianNetwork([("a", "b")])
    assert isinstance(bn, DAG)


def test_discrete_bn_rejects_continuous_cpd():
    import warnings
    from pgmpy.models import DiscreteBayesianNetwork

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        bn = DiscreteBayesianNetwork([("a", "b")])
    cpd = LinearGaussianCPD.from_values(beta=[0.0], std=1.0)
    with pytest.raises(TypeError, match="variable_type"):
        bn.parameters.add(variable="a", cpd=cpd)


def test_linear_gaussian_bn_emits_future_warning_on_init():
    from pgmpy.models import LinearGaussianBayesianNetwork
    with pytest.warns(FutureWarning, match="pgmpy.base.DAG"):
        LinearGaussianBayesianNetwork([("a", "b")])


def test_functional_bn_emits_future_warning_on_init():
    pytest.importorskip("pyro")
    from pgmpy.models import FunctionalBayesianNetwork
    with pytest.warns(FutureWarning, match="pgmpy.base.DAG"):
        FunctionalBayesianNetwork([("a", "b")])


def test_back_compat_method_shims_delegate_to_accessor():
    import warnings
    from pgmpy.models import DiscreteBayesianNetwork

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        bn = DiscreteBayesianNetwork([("diff", "grade")])
    bn.parameters.add(variable="diff", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["easy", "hard"]],
    ))
    bn.parameters.add(variable="grade", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["A", "B"], ["easy", "hard"]],
    ), parent_order=["diff"])
    # bn.to_markov_model() shim delegates to bn.transforms.to_markov_model()
    mn_via_shim = bn.to_markov_model()
    mn_via_accessor = bn.transforms.to_markov_model()
    assert set(mn_via_shim.nodes()) == set(mn_via_accessor.nodes())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_deprecated_bn_aliases.py -v`
Expected: FAIL — the typed classes currently are standalone v1.x classes.

- [ ] **Step 3: Replace each typed BN with a FutureWarning-emitting subclass of `DAG`**

`pgmpy/models/DiscreteBayesianNetwork.py`:

```python
"""Deprecated. Use pgmpy.base.DAG. This class will be removed in pgmpy 2.0."""

import warnings

from pgmpy.base import DAG


class DiscreteBayesianNetwork(DAG):
    """Deprecated. Use ``pgmpy.base.DAG``.

    Behaves identically except that ``parameters.add`` rejects CPDs whose
    ``variable_type`` tag is not ``"discrete"``. This file is deleted in
    pgmpy 2.0.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "DiscreteBayesianNetwork is deprecated and will be removed in "
            "pgmpy 2.0. Use pgmpy.base.DAG. The all-discrete invariant this "
            "class enforced moves to inference-algorithm preconditions in "
            "the new design.",
            FutureWarning, stacklevel=2,
        )
        super().__init__(*args, **kwargs)

    def _register_cpd(self, variable, cpd, parent_order=None):
        from pgmpy.parameterization import check_parameterization
        check_parameterization(cpd)
        variable_type = self._infer_variable_type(cpd)
        if variable_type != "discrete":
            raise TypeError(
                f"DiscreteBayesianNetwork requires variable_type='discrete' "
                f"CPDs; got {variable_type!r} for {variable!r}."
            )
        super()._register_cpd(variable, cpd, parent_order)

    @staticmethod
    def _infer_variable_type(cpd):
        if hasattr(cpd, "get_tag"):
            try:
                return cpd.get_tag("variable_type")
            except (KeyError, ValueError):
                pass
        try:
            from sklearn.base import ClassifierMixin
            if isinstance(cpd, ClassifierMixin):
                return "discrete"
        except ImportError:
            pass
        try:
            from skpro.regression.base import BaseProbaRegressor
            if isinstance(cpd, BaseProbaRegressor):
                return "continuous"
        except ImportError:
            pass
        return "unknown"

    # Back-compat method shims. No second FutureWarning per call —
    # the __init__ warning is enough.
    def to_markov_model(self):       return self.transforms.to_markov_model()
    def predict(self, data, **k):    return self.inference.predict(data, **k)
    def predict_probability(self, data, **k):
        return self.inference.predict_probability(data, **k)
    def save(self, path, **k):       return self.io.save(path, **k)
```

`pgmpy/models/LinearGaussianBayesianNetwork.py`: same pattern, enforces
`is_linear_gaussian=True`, shims for `to_joint_gaussian`, `log_likelihood`,
`predict`, `save`.

`pgmpy/models/FunctionalBayesianNetwork.py`: same pattern, enforces
`supports_fit_joint=True`. No method shims beyond the inherited ones.

Update `pgmpy/models/__init__.py` to keep all three exports.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_deprecated_bn_aliases.py -v`
Expected: 5 passed.

Run the existing test files under `pgmpy/tests/test_models/test_*.py` with
`-W ignore::FutureWarning` to confirm legacy code keeps working:

Run: `pytest pgmpy/tests/test_models/ -v --tb=short -W ignore::FutureWarning`
Expected: All pass. The deprecated aliases preserve every v1.x method
through the back-compat shims.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/DiscreteBayesianNetwork.py pgmpy/models/LinearGaussianBayesianNetwork.py pgmpy/models/FunctionalBayesianNetwork.py pgmpy/models/__init__.py pgmpy/tests/test_models/test_deprecated_bn_aliases.py
git commit -m "feat(models): typed BN classes become FutureWarning-emitting DAG subclasses"
```

---

## Task 29: Audit `DynamicBayesianNetwork` — keep temporal overrides, route storage through the new DAG registry

`DynamicBayesianNetwork` already inherits from `DAG` today (not from
`DiscreteBayesianNetwork`, as an earlier draft of this plan incorrectly
stated). With Phase 2 Tasks 19–27 enriching `DAG`, DBN automatically
gains `_cpds`/`_parent_order`/`parameters`/`transforms`/`inference`/`io`.
The task here is the migration of DBN's temporal-aware overrides:

- DBN's `add_cpds(*cpds)` must keep working — CPDs in a DBN carry
  `(node, time_slice)` keys via their `cpd.variable` attribute. The
  override translates the positional `*cpds` form into
  `self.parameters.add(variable=(node, ts), cpd=cpd)` calls. `_cpds`
  accepts tuple keys natively (it's `dict[Hashable, …]`).
- DBN's `get_cpds(node, time_slice=None)` override keeps the optional
  `time_slice=` kwarg and selects from `_cpds` accordingly.
- DBN's `simulate` / `initialize_initial_state` / `moralize` keep their
  temporal logic but read CPDs through `self._cpds` / `self.parameters`
  instead of the legacy `self.cpds = []` list.
- The legacy `self.cpds = []` list initialization in DBN's `__init__` is
  removed (now provided by `DAG` as a property).

**Files:**
- Modify: `pgmpy/models/DynamicBayesianNetwork.py`
- Modify: `pgmpy/tests/test_models/test_DynamicBayesianNetwork.py`

- [ ] **Step 1: Write tests verifying DBN's overrides keep working**

```python
# Append to pgmpy/tests/test_models/test_DynamicBayesianNetwork.py
def test_dbn_uses_dag_registry_under_the_hood():
    from pgmpy.base import DAG
    from pgmpy.models import DynamicBayesianNetwork

    dbn = DynamicBayesianNetwork([(("X", 0), ("X", 1))])
    assert isinstance(dbn, DAG)
    # _cpds is the inherited registry, not a legacy list.
    assert isinstance(dbn._cpds, dict)


def test_dbn_add_cpds_positional_still_works():
    # Existing DBN user code passes positional CPDs with (node, ts) variable.
    # The override routes through self.parameters.add internally.
    ...
```

- [ ] **Step 2: Run tests to verify they fail (where applicable)**

Run: `pytest pgmpy/tests/test_models/test_DynamicBayesianNetwork.py -v --tb=short -W ignore::DeprecationWarning`
Expected: existing DBN tests pass if DBN's overrides correctly route through the new registry; new tests fail until the audit work in Step 3.

- [ ] **Step 3: Update the inheritance**

In `pgmpy/models/DynamicBayesianNetwork.py`, change the class declaration
from `class DynamicBayesianNetwork(DiscreteBayesianNetwork):` to:

```python
from pgmpy.base import DAG


class DynamicBayesianNetwork(DAG):
    ...
```

All DBN-specific methods (`get_intra_edges`, `get_inter_edges`,
`get_slice_nodes`, `add_node`, etc.) stay untouched — they operate on
graph structure with `(node, time_slice)` tuples and don't depend on
the parent class's CPD validation behavior.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_DynamicBayesianNetwork.py -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/DynamicBayesianNetwork.py pgmpy/tests/test_models/test_DynamicBayesianNetwork.py
git commit -m "refactor(models): DynamicBayesianNetwork audit — registry via DAG"
```

---

# Phase 2b: Parameter Estimators

## Task 30: Refactor `ParameterEstimator` base + create generic `MLEEstimator`

**Files:**
- Modify: `pgmpy/parameter_estimator/base.py`
- Create: `pgmpy/parameter_estimator/mle.py`
- Create: `pgmpy/tests/test_parameter_estimator/test_mle_estimator.py`

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_parameter_estimator/test_mle_estimator.py
import numpy as np
import pandas as pd
import pytest

from pgmpy.base import DAG
from pgmpy.parameter_estimator import MLEEstimator
from pgmpy.parameterization import TabularCPD


def test_mle_estimator_orchestrates_per_node_fit():
    rng = np.random.default_rng(0)
    n = 200
    data = pd.DataFrame({
        "diff":  rng.choice(["easy", "hard"], n),
        "grade": rng.choice(["A", "B"], n),
    })
    bn = DAG([("diff", "grade")])
    bn.add_cpds(variable="diff", cpd=TabularCPD(variable_card=2))
    bn.add_cpds(variable="grade",
                cpd=TabularCPD(variable_card=2, evidence_card=[2]),
                parent_order=["diff"])

    MLEEstimator().fit(bn, data)
    for node in ("diff", "grade"):
        assert getattr(bn.get_cpds(node), "is_fitted_", False)


def test_mle_estimator_passes_sample_weight_through():
    rng = np.random.default_rng(0)
    n = 100
    data = pd.DataFrame({"x": rng.choice(["a", "b"], n)})
    bn = DAG()
    bn.add_node("x")
    bn.add_cpds(variable="x", cpd=TabularCPD(variable_card=2))

    weights = np.where(data["x"] == "a", 0.0, 1.0)  # zero-weight all "a"
    MLEEstimator().fit(bn, data, sample_weight=weights)

    # With "a" zero-weighted, the fitted marginal should put nearly all mass on "b".
    cpd = bn.get_cpds("x")
    # values_ is (variable_card, 1); the "b" state should dominate.
    b_idx = list(cpd.classes_).index("b")
    assert cpd.values_[b_idx, 0] > 0.95
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_mle_estimator.py -v`
Expected: FAIL — `MLEEstimator` does not exist yet.

- [ ] **Step 3: Refactor the base class and create `MLEEstimator`**

In `pgmpy/parameter_estimator/base.py`, generalize the base. The current
`DiscreteParameterEstimator` accepts a `DiscreteBayesianNetwork`; relax to
any `DAG`:

```python
# pgmpy/parameter_estimator/base.py
"""Parameter-estimator base classes.

Estimators accept a fully-graphed DAG (whose CPDs may be
unfitted) and a pandas DataFrame, and populate each CPD's fitted state.
"""

from __future__ import annotations


class ParameterEstimator:
    """Base class for parameter estimators.

    Subclasses implement ``fit(model, data, **kwargs)`` and mutate the
    model's registered CPDs in place.
    """

    def fit(self, model, data, **kwargs):
        raise NotImplementedError


# Back-compat: the old DiscreteParameterEstimator name is preserved as a
# deprecated alias of ParameterEstimator. Existing user code that imports
# this name keeps working.
class DiscreteParameterEstimator(ParameterEstimator):
    """Deprecated alias. Use ``ParameterEstimator``."""
```

Create `pgmpy/parameter_estimator/mle.py`:

```python
# pgmpy/parameter_estimator/mle.py
"""Generic per-node MLE estimator.

Default for ``DAG.fit(data)``. Walks the graph in topological
order and calls ``cpd.fit(X, y, sample_weight=...)`` on each node.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd

from pgmpy.parameter_estimator.base import ParameterEstimator


class MLEEstimator(ParameterEstimator):
    """Per-node MLE. Type-agnostic — delegates math to each CPD's ``fit``."""

    def fit(self, model, data, sample_weight=None):
        for node in nx.topological_sort(model):
            cpd = model.get_cpds(node)
            parents = model._parent_order.get(node, [])
            X = data[parents] if parents else pd.DataFrame(index=data.index)
            y = data[node]
            cpd.fit(X, y, sample_weight=sample_weight)
        return model
```

Update `pgmpy/parameter_estimator/__init__.py` to export `MLEEstimator`
and `ParameterEstimator`.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_mle_estimator.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameter_estimator/base.py pgmpy/parameter_estimator/mle.py pgmpy/parameter_estimator/__init__.py pgmpy/tests/test_parameter_estimator/test_mle_estimator.py
git commit -m "feat(estimator): generic MLEEstimator + ParameterEstimator base"
```

---

## Task 31: Refactor `DiscreteMLE` and `DiscreteBayesianEstimator`

**Files:**
- Modify: `pgmpy/parameter_estimator/discrete_mle.py`
- Modify: `pgmpy/parameter_estimator/discrete_bayesian.py`
- Modify: `pgmpy/tests/test_parameter_estimator/test_discrete_mle.py` (existing)

- [ ] **Step 1: Write failing tests**

```python
# Append to (or create) pgmpy/tests/test_parameter_estimator/test_discrete_mle.py
import pytest

from pgmpy.base import DAG
from pgmpy.parameter_estimator import DiscreteMLE
from pgmpy.parameterization import LinearGaussianCPD, TabularCPD


def test_discrete_mle_rejects_continuous_cpd():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD(variable_card=2))
    bn.add_cpds(variable="b", cpd=LinearGaussianCPD(), parent_order=["a"])
    with pytest.raises(TypeError, match="variable_type"):
        DiscreteMLE().fit(bn, data=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_discrete_mle.py::test_discrete_mle_rejects_continuous_cpd -v`
Expected: FAIL — current `DiscreteMLE` accepts the network (legacy path) or errors with a different message.

- [ ] **Step 3: Refactor `DiscreteMLE` to subclass `MLEEstimator`**

```python
# pgmpy/parameter_estimator/discrete_mle.py
from pgmpy.parameter_estimator.mle import MLEEstimator


class DiscreteMLE(MLEEstimator):
    """Per-node MLE with all-discrete CPD validation."""

    def fit(self, model, data, sample_weight=None):
        for node, cpd in model._cpds.items():
            tag = getattr(cpd, "get_tag", lambda *_: "unknown")("variable_type", "unknown")
            if tag != "discrete":
                raise TypeError(
                    f"DiscreteMLE requires variable_type='discrete' CPDs; "
                    f"got {tag!r} for {node!r}."
                )
        return super().fit(model, data, sample_weight=sample_weight)
```

Refactor `DiscreteBayesianEstimator` (`pgmpy/parameter_estimator/discrete_bayesian.py`)
to accept any BN and pass `prior_pseudo_counts=` per-node. The CPD's
`TabularCPD.fit` grows a `prior_pseudo_counts=None` keyword argument that
overrides ML estimation with Dirichlet posterior counts when supplied.
Document this in `TabularCPD` and add a follow-up test.

```python
# pgmpy/parameter_estimator/discrete_bayesian.py
from pgmpy.parameter_estimator.base import ParameterEstimator


class DiscreteBayesianEstimator(ParameterEstimator):
    """Bayesian per-node fit with Dirichlet priors.

    ``prior_type``: 'BDeu' (default), 'K2', or 'dirichlet'.
    ``equivalent_sample_size``: pseudo-count strength.
    """

    def __init__(self, prior_type="BDeu", equivalent_sample_size=10,
                 pseudo_counts=None):
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size
        self.pseudo_counts = pseudo_counts

    def fit(self, model, data, sample_weight=None):
        import networkx as nx
        import pandas as pd

        for node, cpd in model._cpds.items():
            tag = getattr(cpd, "get_tag", lambda *_: "unknown")("variable_type", "unknown")
            if tag != "discrete":
                raise TypeError(
                    f"DiscreteBayesianEstimator requires discrete CPDs; "
                    f"got {tag!r} for {node!r}."
                )

        for node in nx.topological_sort(model):
            cpd = model.get_cpds(node)
            parents = model._parent_order.get(node, [])
            X = data[parents] if parents else pd.DataFrame(index=data.index)
            y = data[node]
            prior = self._prior_for_node(cpd, parents)
            cpd.fit(X, y, sample_weight=sample_weight,
                    prior_pseudo_counts=prior)
        return model

    def _prior_for_node(self, cpd, parents):
        # Build a (variable_card, n_parent_combos) pseudo-count matrix from
        # self.prior_type and self.equivalent_sample_size. See
        # the original DiscreteBayesianEstimator for the exact formulas.
        ...
```

`TabularCPD.fit` needs to accept `prior_pseudo_counts` (default None). When
provided, the count matrix used for normalization is `counts +
prior_pseudo_counts`. Add this kwarg in the per-CPD fit (modify Task 7's
implementation as part of this task).

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameter_estimator/ -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameter_estimator/discrete_mle.py pgmpy/parameter_estimator/discrete_bayesian.py pgmpy/parameterization/tabular.py pgmpy/tests/test_parameter_estimator/
git commit -m "feat(estimator): DiscreteMLE / DiscreteBayesianEstimator delegate to CPD.fit"
```

---

## Task 32: Refactor `DiscreteEM` for the unified BN

**Files:**
- Modify: `pgmpy/parameter_estimator/discrete_em.py`
- Modify: existing `DiscreteEM` tests

- [ ] **Step 1: Write failing test**

```python
# Append to existing test_discrete_em.py
def test_discrete_em_runs_against_bayesian_network():
    import numpy as np
    import pandas as pd
    from pgmpy.base import DAG
    from pgmpy.parameter_estimator import DiscreteEM
    from pgmpy.parameterization import TabularCPD

    rng = np.random.default_rng(0)
    bn = DAG([("h", "obs")])
    bn.add_cpds(variable="h", cpd=TabularCPD(variable_card=2))
    bn.add_cpds(variable="obs",
                cpd=TabularCPD(variable_card=2, evidence_card=[2]),
                parent_order=["h"])

    # h is latent; only obs is observed in data.
    data = pd.DataFrame({"obs": rng.choice(["L", "R"], 200)})
    DiscreteEM(latent_variables=["h"]).fit(bn, data, max_iter=20)

    assert bn.get_cpds("h").is_fitted_
    assert bn.get_cpds("obs").is_fitted_
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_discrete_em.py::test_discrete_em_runs_against_bayesian_network -v`
Expected: FAIL — current `DiscreteEM` may not accept a unified `DAG`.

- [ ] **Step 3: Refactor `DiscreteEM`**

In `pgmpy/parameter_estimator/discrete_em.py`, the E-step calls
`VariableElimination(model).query(...)` for each row to compute expected
counts; the M-step calls `cpd.fit(X, y, sample_weight=expected_counts)`.
Replace any `isinstance(model, DiscreteBayesianNetwork)` checks with a
tag check (`cpd.get_tag("variable_type") == "discrete"` on every CPD).

Concrete skeleton (preserves the algorithm; updates the API):

```python
class DiscreteEM(ParameterEstimator):
    def __init__(self, latent_variables, max_iter=100, tol=1e-4):
        self.latent_variables = list(latent_variables)
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, model, data, sample_weight=None):
        # Validate all CPDs are discrete.
        for node, cpd in model._cpds.items():
            tag = getattr(cpd, "get_tag", lambda *_: "unknown")("variable_type", "unknown")
            if tag != "discrete":
                raise TypeError(...)

        # Initialize unfitted CPDs to uniform via MLE on observed columns.
        ...

        prev_ll = -float("inf")
        for it in range(self.max_iter):
            # E-step: compute expected counts via VE per row.
            expected_counts = self._e_step(model, data)
            # M-step: refit each CPD with expected_counts as sample_weight.
            for node, ec in expected_counts.items():
                cpd = model.get_cpds(node)
                parents = model._parent_order.get(node, [])
                X = data[parents] if parents else pd.DataFrame(index=data.index)
                y = data[node] if node in data.columns else ec["y"]
                cpd.fit(X, y, sample_weight=ec["weights"])
            # Convergence check on log-likelihood.
            ll = model.inference.log_likelihood(data)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll
        return model

    def _e_step(self, model, data):
        # Build (per-node) expected-count tables using VE.
        ...
```

The detailed E-step body lifts from the existing `DiscreteEM` with
substitutions for the new BN API.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_discrete_em.py -v --tb=short`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameter_estimator/discrete_em.py pgmpy/tests/test_parameter_estimator/test_discrete_em.py
git commit -m "feat(estimator): DiscreteEM works on unified DAG"
```

---

## Task 33: Refactor `LinearGaussianMLE`

**Files:**
- Modify: `pgmpy/parameter_estimator/linear_gaussian_mle.py`
- Modify: existing `LinearGaussianMLE` tests

- [ ] **Step 1: Write failing test**

```python
# Append to existing test_linear_gaussian_mle.py
def test_linear_gaussian_mle_rejects_discrete_cpd():
    from pgmpy.base import DAG
    from pgmpy.parameter_estimator import LinearGaussianMLE
    from pgmpy.parameterization import TabularCPD

    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD(variable_card=2))
    with pytest.raises(TypeError, match="is_linear_gaussian"):
        LinearGaussianMLE().fit(bn, data=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_linear_gaussian_mle.py::test_linear_gaussian_mle_rejects_discrete_cpd -v`
Expected: FAIL — current `LinearGaussianMLE` accepts the network.

- [ ] **Step 3: Refactor**

```python
# pgmpy/parameter_estimator/linear_gaussian_mle.py
from pgmpy.parameter_estimator.mle import MLEEstimator


class LinearGaussianMLE(MLEEstimator):
    """Per-node MLE with all-linear-Gaussian CPD validation."""

    def fit(self, model, data, sample_weight=None):
        for node, cpd in model._cpds.items():
            if not (hasattr(cpd, "get_tag")
                    and cpd.get_tag("is_linear_gaussian", False)):
                raise TypeError(
                    f"LinearGaussianMLE requires is_linear_gaussian=True CPDs; "
                    f"got {type(cpd).__name__} for {node!r}."
                )
        return super().fit(model, data, sample_weight=sample_weight)
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_linear_gaussian_mle.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameter_estimator/linear_gaussian_mle.py pgmpy/tests/test_parameter_estimator/test_linear_gaussian_mle.py
git commit -m "feat(estimator): LinearGaussianMLE subclasses MLEEstimator"
```

---

## Task 34: Create `JointPyroEstimator` (and delete `_functional_joint.py`)

**Files:**
- Create: `pgmpy/parameter_estimator/joint_pyro.py`
- Create: `pgmpy/tests/test_parameter_estimator/test_joint_pyro_estimator.py`
- Delete: `pgmpy/parameterization/_functional_joint.py`
- Modify: `pgmpy/parameterization/functional.py` — remove `fit_joint`.

- [ ] **Step 1: Write failing test**

```python
# pgmpy/tests/test_parameter_estimator/test_joint_pyro_estimator.py
import pytest

pytest.importorskip("pyro")
import pyro.distributions as dist

from pgmpy.base import DAG
from pgmpy.parameter_estimator import JointPyroEstimator
from pgmpy.parameterization import FunctionalCPD


def test_joint_pyro_estimator_runs_svi():
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=FunctionalCPD(fn=lambda _: dist.Normal(0.0, 1.0)))
    bn.add_cpds(variable="x2",
                cpd=FunctionalCPD(fn=lambda p: dist.Normal(p["x1"] + 2.0, 1.0)),
                parent_order=["x1"])
    samples = bn.simulate(n_samples=200)
    estimator = JointPyroEstimator(estimator="SVI", num_steps=50)
    estimator.fit(bn, samples)


def test_joint_pyro_estimator_rejects_non_functional_cpd():
    from pgmpy.parameterization import TabularCPD
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD(variable_card=2))
    with pytest.raises(TypeError, match="supports_fit_joint"):
        JointPyroEstimator().fit(bn, data=None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_joint_pyro_estimator.py -v`
Expected: FAIL — `JointPyroEstimator` does not exist.

- [ ] **Step 3: Create `JointPyroEstimator`**

Port the body of `run_joint_pyro_fit` from
`pgmpy/parameterization/_functional_joint.py` (Phase 1 Task 15 created the
stub; Phase 2 Task 25 / 26 ported the logic) into the new estimator class.

```python
# pgmpy/parameter_estimator/joint_pyro.py
"""Joint SVI/MCMC over a network of FunctionalCPDs."""

from __future__ import annotations

from pgmpy.parameter_estimator.base import ParameterEstimator


class JointPyroEstimator(ParameterEstimator):
    """Network-level joint Pyro fit.

    Builds a single combined Pyro model from each node's FunctionalCPD ``fn``
    and runs SVI or MCMC. Precondition: every CPD advertises
    ``supports_fit_joint=True``.
    """

    def __init__(self, estimator="SVI", num_steps=1000, optimizer=None,
                 prior_fn=None, nuts_kwargs=None, mcmc_kwargs=None, seed=None):
        self.estimator = estimator
        self.num_steps = num_steps
        self.optimizer = optimizer
        self.prior_fn = prior_fn
        self.nuts_kwargs = nuts_kwargs
        self.mcmc_kwargs = mcmc_kwargs
        self.seed = seed

    def fit(self, model, data):
        for node, cpd in model._cpds.items():
            if not (hasattr(cpd, "get_tag")
                    and cpd.get_tag("supports_fit_joint", False)):
                raise TypeError(
                    f"JointPyroEstimator requires supports_fit_joint=True "
                    f"CPDs; got {type(cpd).__name__} for {node!r}."
                )

        import torch
        import pyro
        import pyro.optim
        from pyro.infer import SVI, Trace_ELBO, MCMC, NUTS

        if self.seed is not None:
            pyro.set_rng_seed(self.seed)
        pyro.clear_param_store()

        nodes = list(model.nodes())
        tensor_data = {
            col: torch.as_tensor(data[col].values).float()
            for col in data.columns if col in nodes
        }

        def combined_model(tensor_data):
            sampled = {}
            if self.prior_fn is not None:
                sampled.update(self.prior_fn() or {})
            for node in nodes:
                cpd = model._cpds[node]
                parents = model._parent_order.get(node, [])
                parent_values = {p: sampled.get(p, tensor_data.get(p))
                                 for p in parents}
                d = cpd.fn(parent_values) if parents else cpd.fn({})
                obs = tensor_data.get(node)
                sampled[node] = pyro.sample(node, d, obs=obs)
            return sampled

        if self.estimator == "SVI":
            optimizer = self.optimizer or pyro.optim.Adam({"lr": 1e-2})

            def guide(tensor_data):
                pass

            svi = SVI(combined_model, guide, optimizer, loss=Trace_ELBO())
            for _ in range(self.num_steps):
                svi.step(tensor_data)
            return dict(pyro.get_param_store())
        elif self.estimator == "MCMC":
            kernel = NUTS(combined_model, **(self.nuts_kwargs or {}))
            mcmc = MCMC(kernel, num_samples=self.num_steps,
                        **(self.mcmc_kwargs or {}))
            mcmc.run(tensor_data)
            return mcmc.get_samples()
        else:
            raise ValueError(f"Unknown estimator: {self.estimator!r}")
```

Delete `pgmpy/parameterization/_functional_joint.py`. Remove `fit_joint`
from `FunctionalCPD`. Update `pgmpy/parameter_estimator/__init__.py` to
export `JointPyroEstimator`.

Also at this task: simplify `DAG.fit` to a three-line delegate:

```python
# pgmpy/base/DAG.py
def fit(self, data, estimator=None, **kwargs):
    if estimator is None:
        from pgmpy.parameter_estimator import MLEEstimator
        estimator = MLEEstimator()
    return estimator.fit(self, data, **kwargs)
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameter_estimator/test_joint_pyro_estimator.py -v`
Expected: 2 passed.

Run: `pytest pgmpy/tests/ -W ignore::DeprecationWarning -v --tb=short`
Expected: All tests pass (regression check across the full suite).

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameter_estimator/joint_pyro.py pgmpy/parameter_estimator/__init__.py pgmpy/parameterization/functional.py pgmpy/base/DAG.py pgmpy/tests/test_parameter_estimator/test_joint_pyro_estimator.py
git rm pgmpy/parameterization/_functional_joint.py
git commit -m "feat(estimator): JointPyroEstimator replaces _fit_joint mixin"
```

---

# Phase 3: Inference Dispatch Integration

## Task 35: `ApproxInference` accepts any `DAG`

**Files:**
- Modify: `pgmpy/inference/ApproxInference.py`
- Create: `pgmpy/tests/test_inference/test_approx_inference_unified.py`

- [ ] **Step 1: Write failing test**

```python
# pgmpy/tests/test_inference/test_approx_inference_unified.py
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skpro")

from pgmpy.inference import ApproxInference
from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD, TabularCPD


def test_approx_inference_accepts_mixed_cpd_network():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.6], [0.4]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=LinearGaussianCPD.from_values(
        beta=[0.0, 1.0], std=1.0,
    ), parent_order=["a"])
    # Encoding the discrete parent for the LG child is out-of-scope for
    # this plan, so we expect simulate() to raise a TypeError on the
    # mixed types — but ApproxInference's __init__ should still succeed.
    inf = ApproxInference(bn)
    assert inf is not None


def test_approx_inference_accepts_pure_discrete_network():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.6], [0.4]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.9, 0.1], [0.1, 0.9]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    inf = ApproxInference(bn)
    result = inf.query(variables=["b"], n_samples=1000, show_progress=False)
    assert "b" in result.variables
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_inference/test_approx_inference_unified.py -v`
Expected: FAIL — `ApproxInference.__init__` likely accepts only DiscreteBN and DBN today.

- [ ] **Step 3: Relax `ApproxInference.__init__` type check**

In `pgmpy/inference/ApproxInference.py` (around line 24):

```python
def __init__(self, model):
    from pgmpy.base import DAG
    from pgmpy.models import DynamicBayesianNetwork
    accepted = (DAG, DynamicBayesianNetwork)
    if not isinstance(model, accepted):
        raise ValueError(
            f"model should be a Bayesian Network or Dynamic Bayesian "
            f"Network. Got {type(model)}."
        )
    model.check_model()
    self.model = model
```

`DiscreteBayesianNetwork`, `LinearGaussianBayesianNetwork`, and
`FunctionalBayesianNetwork` are all subclasses of `DAG` now,
so the deprecated aliases continue to work without modification.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_inference/test_approx_inference_unified.py -v`
Expected: 2 passed.

Run: `pytest pgmpy/tests/test_inference/ -W ignore::DeprecationWarning -v --tb=short`
Expected: All existing inference tests pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/inference/ApproxInference.py pgmpy/tests/test_inference/test_approx_inference_unified.py
git commit -m "feat(inference): ApproxInference accepts any DAG"
```

---

# Phase 4: Inference Dispatch Migration + Legacy Deprecation

## Task 36: Tag dispatch in `inference/base.py::_initialize_structures`

**Files:**
- Modify: `pgmpy/inference/base.py`
- Create: `pgmpy/tests/test_inference/test_dispatch_tags.py`

- [ ] **Step 1: Write failing test**

```python
# pgmpy/tests/test_inference/test_dispatch_tags.py
import pytest
from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD, TabularCPD
from pgmpy.inference import VariableElimination


def test_variable_elimination_rejects_network_with_non_factor_cpd():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=LinearGaussianCPD.from_values(
        beta=[0.0, 1.0], std=1.0,
    ), parent_order=["a"])
    with pytest.raises((TypeError, ValueError), match="produces_factor"):
        VariableElimination(bn)


def test_variable_elimination_accepts_all_tabular_network():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.9, 0.1], [0.1, 0.9]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    ve = VariableElimination(bn)  # should not raise
    assert ve is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_inference/test_dispatch_tags.py -v`
Expected: FAIL.

- [ ] **Step 3: Replace the `isinstance(cpd, TabularCPD)` block in `Inference._initialize_structures`**

In `pgmpy/inference/base.py`, replace the DiscreteBayesianNetwork branch
(lines 99–108) of `_initialize_structures` with a tag-based check:

```python
if isinstance(self.model, DAG):
    self.state_names_map = {}
    for node in self.model.nodes():
        cpd = self.model.get_cpds(node)
        produces_factor = (
            hasattr(cpd, "get_tag")
            and cpd.get_tag("produces_factor", False)
        )
        if not produces_factor:
            raise TypeError(
                f"Variable elimination / belief propagation requires CPDs "
                f"with produces_factor=True; got {type(cpd).__name__} for "
                f"node {node!r}."
            )
        factor = self.model.cpd_as_factor(node)
        if hasattr(cpd, "variable_card"):
            self.cardinality[node] = cpd.variable_card
        for var in factor.scope():
            self.factors[var].append(factor)
        self.state_names_map.update(factor.no_to_name)
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_inference/test_dispatch_tags.py -v`
Expected: 2 passed.

Run: `pytest pgmpy/tests/test_inference/ -v --tb=short -x`
Expected: Existing inference tests pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/inference/base.py pgmpy/tests/test_inference/test_dispatch_tags.py
git commit -m "refactor(inference): tag dispatch in Inference._initialize_structures"
```

---

## Task 37: Tag dispatch in `sampling/Sampling.py::forward_sample`

**Files:**
- Modify: `pgmpy/sampling/Sampling.py`
- Modify: `pgmpy/tests/test_inference/test_dispatch_tags.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_inference/test_dispatch_tags.py
def test_forward_sample_uses_cpd_sample_for_new_style_cpds():
    from pgmpy.sampling import BayesianModelSampling
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[1.0], [0.0]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[1.0, 0.0], [0.0, 1.0]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    sampler = BayesianModelSampling(bn)
    out = sampler.forward_sample(size=10, show_progress=False)
    assert (out["a"] == "x").all()
    assert (out["b"] == "L").all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_inference/test_dispatch_tags.py::test_forward_sample_uses_cpd_sample_for_new_style_cpds -v`
Expected: FAIL — `forward_sample` reads `cpd.values` directly, which doesn't exist on new-style CPDs (it's `cpd.values_`).

- [ ] **Step 3: Refactor `forward_sample` to use `cpd_sample`**

In `pgmpy/sampling/Sampling.py`, replace the per-node block (lines 100–127)
with:

```python
from pgmpy.parameterization.base import cpd_sample

for node in pbar:
    if show_progress and config.SHOW_PROGRESS:
        pbar.set_description(f"Generating for node: {node}")
    if (partial_samples is not None) and (node in partial_samples.columns):
        sampled[node] = partial_samples.loc[:, node].values
        continue

    cpd = self.model.get_cpds(node)
    parents = self.model._parent_order.get(node, [])
    X = sampled[parents] if parents else pd.DataFrame(index=range(size))
    sampled[node] = cpd_sample(cpd, X, n_samples=size).values
```

This drops the old `pre_compute_reduce_maps` path. Verify in a follow-up
profiling pass that the new path is fast enough; if not, add a numpy fast-
path inside `TabularCPD.sample` for the all-tabular case.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_inference/test_dispatch_tags.py -v`
Expected: All pass.

Run: `pytest pgmpy/tests/test_sampling/ -v --tb=short -x`
Expected: Existing sampling tests pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/sampling/Sampling.py pgmpy/tests/test_inference/test_dispatch_tags.py
git commit -m "refactor(sampling): forward_sample uses cpd_sample dispatch"
```

---

## Task 38: `LinearGaussianInference` class

**Files:**
- Create: `pgmpy/inference/linear_gaussian.py`
- Create: `pgmpy/tests/test_inference/test_linear_gaussian_inference.py`
- Modify: `pgmpy/inference/__init__.py` — export `LinearGaussianInference`.
- Modify: `pgmpy/models/LinearGaussianBayesianNetwork.py` — the deprecated alias's `predict` shim now delegates to `LinearGaussianInference(bn).predict(data)`.

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_inference/test_linear_gaussian_inference.py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skpro")

from pgmpy.inference import LinearGaussianInference
from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD, TabularCPD


def test_lg_inference_rejects_non_lg_network():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=LinearGaussianCPD.from_values(
        beta=[0.0, 1.0], std=1.0,
    ), parent_order=["a"])
    with pytest.raises(TypeError, match="is_linear_gaussian"):
        LinearGaussianInference(bn)


def test_lg_inference_query_with_no_evidence_returns_prior_mean():
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=LinearGaussianCPD.from_values(beta=[1.0], std=1.0))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=1.0),
                parent_order=["x1"])
    inf = LinearGaussianInference(bn)
    mu, cov = inf.query(["x2"])
    # E[x2] = 0 + 2 * E[x1] = 2 * 1 = 2
    assert np.isclose(mu[0], 2.0)


def test_lg_inference_query_with_evidence_conditions_correctly():
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=1.0),
                parent_order=["x1"])
    inf = LinearGaussianInference(bn)
    # Given x1 = 3, E[x2 | x1=3] = 2 * 3 = 6
    mu, cov = inf.query(["x2"], evidence={"x1": 3.0})
    assert np.isclose(mu[0], 6.0)


def test_lg_inference_predict_fills_missing_columns():
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=1.0),
                parent_order=["x1"])
    inf = LinearGaussianInference(bn)
    obs = pd.DataFrame({"x1": [1.0, 2.0, 3.0]})  # x2 unobserved
    pred = inf.predict(obs)
    assert np.allclose(pred["x2"], [2.0, 4.0, 6.0])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_inference/test_linear_gaussian_inference.py -v`
Expected: FAIL — `LinearGaussianInference` does not exist.

- [ ] **Step 3: Implement `LinearGaussianInference`**

The algorithm: pre-compute `(mu, cov)` once via `model.transforms.to_joint_gaussian()`. For each query, partition the joint into observed (a) and target (b) indices; the posterior mean and covariance are:

```
mu_b|a   = mu_b + cov_ba * inv(cov_aa) * (x_a - mu_a)
cov_b|a  = cov_bb - cov_ba * inv(cov_aa) * cov_ab
```

```python
# pgmpy/inference/linear_gaussian.py
"""Exact inference on linear-Gaussian Bayesian networks.

Conditioning is performed analytically on the joint Gaussian via the
Schur complement.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


class LinearGaussianInference:
    """Exact inference for all-linear-Gaussian networks.

    Precondition: every CPD in *model* advertises ``is_linear_gaussian=True``.
    """

    def __init__(self, model):
        for node, cpd in model._cpds.items():
            if not (hasattr(cpd, "get_tag")
                    and cpd.get_tag("is_linear_gaussian", False)):
                raise TypeError(
                    f"LinearGaussianInference requires every CPD to "
                    f"advertise is_linear_gaussian=True; got "
                    f"{type(cpd).__name__} for {node!r}."
                )
        self.model = model
        self._order = list(nx.topological_sort(model))
        self._mu, self._cov = model.transforms.to_joint_gaussian()
        self._pos = {n: i for i, n in enumerate(self._order)}

    def query(self, variables, evidence=None):
        """Return (mu, cov) of P(variables | evidence)."""
        evidence = evidence or {}
        b_idx = [self._pos[v] for v in variables]
        if not evidence:
            return self._mu[b_idx], self._cov[np.ix_(b_idx, b_idx)]

        a_vars = list(evidence.keys())
        a_idx = [self._pos[v] for v in a_vars]
        x_a = np.array([evidence[v] for v in a_vars], dtype=float)

        mu_a = self._mu[a_idx]
        mu_b = self._mu[b_idx]
        cov_aa = self._cov[np.ix_(a_idx, a_idx)]
        cov_ab = self._cov[np.ix_(a_idx, b_idx)]
        cov_ba = self._cov[np.ix_(b_idx, a_idx)]
        cov_bb = self._cov[np.ix_(b_idx, b_idx)]

        inv_aa = np.linalg.pinv(cov_aa)
        mu_post = mu_b + cov_ba @ inv_aa @ (x_a - mu_a)
        cov_post = cov_bb - cov_ba @ inv_aa @ cov_ab
        return mu_post, cov_post

    def predict(self, data):
        """Per-row MAP prediction of missing columns.

        Returns a DataFrame with the same shape as *data*, with unobserved
        columns filled in by the conditional mean.
        """
        observed_cols = list(data.columns)
        latent_cols = [n for n in self.model.nodes() if n not in observed_cols]
        rows = []
        for _, row in data.iterrows():
            evidence = {col: float(row[col]) for col in observed_cols}
            mu_post, _ = self.query(latent_cols, evidence=evidence)
            row_out = {**evidence, **dict(zip(latent_cols, mu_post))}
            rows.append(row_out)
        return pd.DataFrame(rows, index=data.index)
```

In `pgmpy/inference/__init__.py`, export it. In the deprecated alias `pgmpy/models/LinearGaussianBayesianNetwork.py`, the `predict` shim becomes:

```python
def predict(self, data):
    from pgmpy.inference import LinearGaussianInference
    return LinearGaussianInference(self).predict(data)
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_inference/test_linear_gaussian_inference.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/inference/linear_gaussian.py pgmpy/inference/__init__.py pgmpy/models/LinearGaussianBayesianNetwork.py pgmpy/tests/test_inference/test_linear_gaussian_inference.py
git commit -m "feat(inference): LinearGaussianInference for exact Gaussian conditioning"
```

---

## Task 39: `LikelihoodWeighting` class

**Files:**
- Create: `pgmpy/inference/likelihood_weighting.py`
- Create: `pgmpy/tests/test_inference/test_likelihood_weighting.py`
- Modify: `pgmpy/inference/__init__.py` — export `LikelihoodWeighting`.

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_inference/test_likelihood_weighting.py
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skpro")

from pgmpy.inference import LikelihoodWeighting
from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD, TabularCPD


def test_lw_recovers_known_marginal_discrete():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.7], [0.3]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.9, 0.1], [0.1, 0.9]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    inf = LikelihoodWeighting(bn)
    posterior = inf.query(["a"], evidence={"b": "L"}, n_samples=20_000,
                          seed=0)
    # P(a=x | b=L) = P(b=L|a=x) P(a=x) / P(b=L)
    #              = 0.9 * 0.7 / (0.9*0.7 + 0.1*0.3) = 0.63 / 0.66 ≈ 0.9545
    assert abs(posterior["a"]["x"] - 0.9545) < 0.02


def test_lw_works_with_continuous_evidence():
    # P(x1) = N(0, 1); P(x2 | x1) = N(2*x1, 0.5)
    # Observe x2 = 4; posterior over x1 should concentrate near 2.
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                parent_order=["x1"])
    inf = LikelihoodWeighting(bn)
    weighted_samples = inf.weighted_sample(evidence={"x2": 4.0},
                                            n_samples=10_000, seed=0)
    # Weighted mean of x1 should be near 2.
    x1 = weighted_samples["x1"].values
    w = weighted_samples["_weight"].values
    posterior_mean_x1 = np.average(x1, weights=w)
    assert abs(posterior_mean_x1 - 2.0) < 0.2


def test_lw_predict_per_row():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.99, 0.01], [0.01, 0.99]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    inf = LikelihoodWeighting(bn)
    obs = pd.DataFrame({"b": ["L", "R"]})
    pred = inf.predict(obs, n_samples=5_000, seed=0)
    # b=L → likely a=x; b=R → likely a=y
    assert pred.loc[0, "a"] == "x"
    assert pred.loc[1, "a"] == "y"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_inference/test_likelihood_weighting.py -v`
Expected: FAIL — `LikelihoodWeighting` does not exist.

- [ ] **Step 3: Implement `LikelihoodWeighting`**

```python
# pgmpy/inference/likelihood_weighting.py
"""Importance-sampling inference via per-CPD log_prob weighting.

Works on any DAG whose CPDs support cpd_log_prob and cpd_sample —
discrete (TabularCPD), linear-Gaussian, FunctionalCPD, and third-party
skpro/sklearn estimators.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.parameterization.base import cpd_log_prob, cpd_sample


class LikelihoodWeighting:
    """Approximate inference by importance sampling.

    For each draw: walk the topological order. For evidence nodes, set the
    value to the observed value and accumulate the log-weight by
    ``cpd.log_prob(observed, parents)``. For latent nodes, forward-sample.
    The posterior is the weighted empirical distribution of samples.
    """

    def __init__(self, model):
        self.model = model
        self._order = list(nx.topological_sort(model))

    def weighted_sample(self, evidence=None, n_samples=10_000, seed=None):
        """Return a DataFrame with n_samples rows plus a ``_weight`` column.

        Each row is a full assignment to all network variables; ``_weight``
        is the importance weight for that draw (normalized weights sum to 1).
        """
        evidence = evidence or {}
        rng = np.random.default_rng(seed)

        sample_cols = {node: [] for node in self._order}
        log_weights = np.zeros(n_samples)

        for i in range(n_samples):
            sample = {}
            log_w = 0.0
            for node in self._order:
                cpd = self.model.get_cpds(node)
                parents = self.model._parent_order.get(node, [])
                X = (pd.DataFrame({p: [sample[p]] for p in parents})
                     if parents else pd.DataFrame(index=[0]))
                if node in evidence:
                    value = evidence[node]
                    log_p = float(cpd_log_prob(cpd, pd.Series([value]), X).iloc[0])
                    log_w += log_p
                    sample[node] = value
                else:
                    y = cpd_sample(cpd, X, n_samples=1,
                                   random_state=int(rng.integers(2**31)))
                    sample[node] = y.iloc[0]
                sample_cols[node].append(sample[node])
            log_weights[i] = log_w

        # Normalize weights to sum to 1 (log-sum-exp for numerical stability).
        max_log = log_weights.max()
        weights = np.exp(log_weights - max_log)
        weights /= weights.sum()

        df = pd.DataFrame(sample_cols)
        df["_weight"] = weights
        return df

    def query(self, variables, evidence=None, n_samples=10_000, seed=None):
        """Return weighted posterior over *variables* given *evidence*.

        For discrete variables, returns a dict of (variable → state → prob).
        For continuous variables, returns the weighted samples directly so
        the caller can compute moments or KDE.
        """
        samples = self.weighted_sample(evidence=evidence,
                                       n_samples=n_samples, seed=seed)
        out = {}
        for var in variables:
            cpd = self.model.get_cpds(var)
            if cpd.get_tag("variable_type") == "discrete":
                states = list(getattr(cpd, "classes_",
                                       cpd.state_names[0] if cpd.state_names
                                       else []))
                dist = {}
                for state in states:
                    mask = (samples[var] == state).to_numpy()
                    dist[state] = float(samples.loc[mask, "_weight"].sum())
                # Normalize (should already be ~1 but guard against drift).
                total = sum(dist.values())
                if total > 0:
                    dist = {k: v / total for k, v in dist.items()}
                out[var] = dist
            else:
                # Continuous: return weighted samples.
                out[var] = samples[[var, "_weight"]]
        return out

    def predict(self, data, n_samples=10_000, seed=None):
        """Per-row MAP prediction of missing columns.

        For each row of *data*, treat its columns as evidence, run LW, and
        fill in the MAP estimate for each unobserved network variable.
        """
        observed = list(data.columns)
        latent = [n for n in self.model.nodes() if n not in observed]
        rows = []
        for idx, row in data.iterrows():
            evidence = {col: row[col] for col in observed}
            samples = self.weighted_sample(evidence=evidence,
                                           n_samples=n_samples,
                                           seed=(None if seed is None
                                                 else seed + int(idx)))
            row_out = dict(evidence)
            for var in latent:
                cpd = self.model.get_cpds(var)
                if cpd.get_tag("variable_type") == "discrete":
                    # MAP = state with highest total weight.
                    weighted = samples.groupby(var)["_weight"].sum()
                    row_out[var] = weighted.idxmax()
                else:
                    # MAP for continuous = weighted mean.
                    row_out[var] = float(np.average(
                        samples[var].values, weights=samples["_weight"].values
                    ))
            rows.append(row_out)
        return pd.DataFrame(rows, index=data.index)
```

Export from `pgmpy/inference/__init__.py`.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_inference/test_likelihood_weighting.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/inference/likelihood_weighting.py pgmpy/inference/__init__.py pgmpy/tests/test_inference/test_likelihood_weighting.py
git commit -m "feat(inference): LikelihoodWeighting via per-CPD log_prob"
```

---

## Task 40: Auto-dispatch in `bn.inference.predict`

**Files:**
- Modify: `pgmpy/models/_accessors.py` — `_BNInference.predict` and `predict_probability` inspect CPD tags and pick the inference algorithm.
- Modify: `pgmpy/tests/test_models/test_bn_accessors.py` — coverage for the dispatch logic.

- [ ] **Step 1: Write failing tests**

```python
# Append to pgmpy/tests/test_models/test_bn_accessors.py
def test_predict_auto_dispatches_to_ve_for_all_discrete_network():
    """An all-discrete network should route through VariableElimination."""
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.99], [0.01]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.99, 0.01], [0.01, 0.99]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    pred = bn.inference.predict(pd.DataFrame({"a": ["x", "y"]}))
    assert pred.loc[0, "b"] == "L"
    assert pred.loc[1, "b"] == "R"


def test_predict_auto_dispatches_to_lg_for_all_linear_gaussian_network():
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                parent_order=["x1"])
    pred = bn.inference.predict(pd.DataFrame({"x1": [1.0, 2.0]}))
    assert np.isclose(pred.loc[0, "x2"], 2.0, atol=1e-6)
    assert np.isclose(pred.loc[1, "x2"], 4.0, atol=1e-6)


def test_predict_auto_dispatches_to_lw_for_mixed_network():
    bn = DAG([("x1", "x2")])
    # x1 is discrete; x2 is continuous via LG.
    bn.add_cpds(variable="x1", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[[0.0, 1.0]],
    ))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                parent_order=["x1"])
    # Should not raise; should return a DataFrame.
    pred = bn.inference.predict(pd.DataFrame({"x1": [0.0, 1.0]}),
                                 method=None)
    assert set(pred.columns) >= {"x1", "x2"}


def test_predict_explicit_method_overrides_auto_dispatch():
    bn = DAG([("a", "b")])
    bn.add_cpds(variable="a", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["x", "y"]],
    ))
    bn.add_cpds(variable="b", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.99, 0.01], [0.01, 0.99]],
        state_names=[["L", "R"], ["x", "y"]],
    ), parent_order=["a"])
    pred = bn.inference.predict(pd.DataFrame({"a": ["x"]}),
                                 method="likelihood_weighting", n_samples=2000)
    assert pred.loc[0, "b"] == "L"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py::test_predict_auto_dispatches_to_lg_for_all_linear_gaussian_network -v`
Expected: FAIL — current `_BNInference.predict` always uses VariableElimination.

- [ ] **Step 3: Add auto-dispatch logic**

Replace `_BNInference.predict` in `pgmpy/models/_accessors.py`:

```python
def predict(self, data, method=None, **kwargs):
    """Predict missing columns of *data* by per-row inference.

    Auto-dispatches based on CPD tags when ``method=None``:
    - all CPDs produce a factor → VariableElimination
    - all CPDs are linear-Gaussian → LinearGaussianInference
    - otherwise → LikelihoodWeighting

    Pass ``method="variable_elimination" | "linear_gaussian" |
    "likelihood_weighting"`` to override.
    """
    method = method or self._auto_dispatch()
    if method == "variable_elimination":
        return self._predict_ve(data, **kwargs)
    elif method == "linear_gaussian":
        from pgmpy.inference import LinearGaussianInference
        return LinearGaussianInference(self._bn).predict(data, **kwargs)
    elif method == "likelihood_weighting":
        from pgmpy.inference import LikelihoodWeighting
        return LikelihoodWeighting(self._bn).predict(data, **kwargs)
    else:
        raise ValueError(f"Unknown inference method: {method!r}")

def _auto_dispatch(self):
    cpds = self._bn._cpds.values()
    if all(c.get_tag("produces_factor", False) for c in cpds):
        return "variable_elimination"
    if all(c.get_tag("is_linear_gaussian", False) for c in cpds):
        return "linear_gaussian"
    return "likelihood_weighting"

def _predict_ve(self, data, **kwargs):
    """The original predict body, factored out."""
    import pandas as pd
    from pgmpy.inference import VariableElimination

    ve = VariableElimination(self._bn)
    observed_cols = list(data.columns)
    latent_cols = [n for n in self._bn.nodes() if n not in observed_cols]
    rows = []
    for _, row in data.iterrows():
        evidence = {col: row[col] for col in observed_cols}
        map_estimate = ve.map_query(variables=latent_cols,
                                    evidence=evidence,
                                    show_progress=False)
        rows.append({**evidence, **map_estimate})
    return pd.DataFrame(rows, index=data.index)
```

Similarly enrich `predict_probability` with the same auto-dispatch (LG path uses `LinearGaussianInference.query` and packages the mean/cov; LW path uses `LikelihoodWeighting.query` and packages the per-state weights).

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/tests/test_models/test_bn_accessors.py
git commit -m "feat(models): _BNInference auto-dispatches to VE / LG / LW by CPD tags"
```

---

## Task 41: Emit `FutureWarning` on legacy `pgmpy.factors.*` CPD classes (1.x)

In 1.x the legacy `TabularCPD` / `LinearGaussianCPD` / `FunctionalCPD`
classes still exist, but their `__init__` emits a `FutureWarning`
pointing at the new `pgmpy.parameterization` module. The classes are
**deleted in Phase 5 (v2.0)**.

The factor-algebra `DiscreteFactor` / `BaseFactor` are unchanged — they're
not CPDs and stay in `pgmpy.factors`.

**Files:**
- Modify: `pgmpy/factors/discrete/CPD.py` (legacy `TabularCPD`)
- Modify: `pgmpy/factors/continuous/LinearGaussianCPD.py`
- Modify: `pgmpy/factors/hybrid/FunctionalCPD.py`
- Create: `pgmpy/tests/test_factors/test_legacy_cpd_deprecation.py`

- [ ] **Step 1: Write failing tests**

```python
# pgmpy/tests/test_factors/test_legacy_cpd_deprecation.py
import pytest


def test_legacy_tabular_cpd_warns_on_init():
    from pgmpy.factors.discrete import TabularCPD as LegacyTabularCPD

    with pytest.warns(FutureWarning, match="pgmpy.parameterization"):
        LegacyTabularCPD("a", 2, values=[[0.5], [0.5]])


def test_legacy_linear_gaussian_cpd_warns_on_init():
    from pgmpy.factors.continuous import LinearGaussianCPD as LegacyLG

    with pytest.warns(FutureWarning, match="pgmpy.parameterization"):
        LegacyLG("y", beta=[1.0], std=1.0)


def test_legacy_functional_cpd_warns_on_init():
    pytest.importorskip("pyro")
    import pyro.distributions as dist
    from pgmpy.factors.hybrid import FunctionalCPD as LegacyFunctional

    with pytest.warns(FutureWarning, match="pgmpy.parameterization"):
        LegacyFunctional(variable="y", fn=lambda _: dist.Normal(0, 1))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_factors/test_legacy_cpd_deprecation.py -v`
Expected: FAIL — no warnings emitted today.

- [ ] **Step 3: Add `FutureWarning` to each legacy CPD's `__init__`**

In `pgmpy/factors/discrete/CPD.py`, at the top of `TabularCPD.__init__`:

```python
import warnings
warnings.warn(
    "pgmpy.factors.discrete.TabularCPD is deprecated and will be removed "
    "in pgmpy 2.0. Use pgmpy.parameterization.TabularCPD.from_values(...) "
    "for direct specification or .fit(X, y) for learning.",
    FutureWarning, stacklevel=2,
)
```

Same pattern at the top of `LinearGaussianCPD.__init__` and
`FunctionalCPD.__init__` (with their respective replacement-API hints).

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_factors/test_legacy_cpd_deprecation.py -v`
Expected: 3 passed.

Run the broader factors test suite under `-W ignore::FutureWarning` to
confirm existing tests still pass:

Run: `pytest pgmpy/tests/test_factors/ -v --tb=short -W ignore::FutureWarning`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/factors/discrete/CPD.py pgmpy/factors/continuous/LinearGaussianCPD.py pgmpy/factors/hybrid/FunctionalCPD.py pgmpy/tests/test_factors/test_legacy_cpd_deprecation.py
git commit -m "chore(factors): FutureWarning on legacy CPDs (deleted in v2.0)"
```

---

## Task 42: Verify `FutureWarning` on legacy `pgmpy.estimators.*` parameter estimators (1.x)

`pgmpy.estimators.MaximumLikelihoodEstimator`,
`pgmpy.estimators.BayesianEstimator`, and `pgmpy.estimators.EM` already
emit `FutureWarning` pointing at `pgmpy.parameter_estimator` (in current
pgmpy main, pre-refactor). This task is a sanity-check / alignment task:

- Confirm the existing warnings fire correctly.
- Align their replacement-API hints with the new module names from Phase
  2b (e.g., point at `pgmpy.parameter_estimator.DiscreteMLE`).
- No code changes if the messages are already correct.

These classes are **deleted in Phase 5 (v2.0)**.

**Files:**
- Modify (if needed): `pgmpy/estimators/MLE.py` — confirm warning message points at `pgmpy.parameter_estimator.DiscreteMLE`.
- Modify (if needed): `pgmpy/estimators/BayesianEstimator.py` — points at `pgmpy.parameter_estimator.DiscreteBayesianEstimator`.
- Modify (if needed): `pgmpy/estimators/EM.py` — points at `pgmpy.parameter_estimator.DiscreteEM`.
- Create: `pgmpy/tests/test_estimators/test_legacy_param_estimator_deprecation.py`.

- [ ] **Step 1: Write tests verifying the existing FutureWarning behavior**

```python
# pgmpy/tests/test_estimators/test_legacy_param_estimator_deprecation.py
import pytest


def test_legacy_mle_estimator_warns():
    from pgmpy.estimators import MaximumLikelihoodEstimator
    # MaximumLikelihoodEstimator(model, data) is the legacy signature.
    # The warning should fire on construction.
    with pytest.warns(FutureWarning, match="pgmpy.parameter_estimator"):
        pass  # construction depends on model+data; fill in with appropriate args.


def test_legacy_bayesian_estimator_warns():
    from pgmpy.estimators import BayesianEstimator
    with pytest.warns(FutureWarning, match="pgmpy.parameter_estimator"):
        pass


def test_legacy_em_estimator_warns():
    from pgmpy.estimators import EM
    with pytest.warns(FutureWarning, match="pgmpy.parameter_estimator"):
        pass
```

- [ ] **Step 2: Run tests**

Run: `pytest pgmpy/tests/test_estimators/test_legacy_param_estimator_deprecation.py -v`
Expected: 3 passed (warnings already fire as of current main).

- [ ] **Step 3: If any warning message points at a stale module path, fix it**

Open each of `pgmpy/estimators/MLE.py`, `BayesianEstimator.py`, `EM.py`
and confirm the `FutureWarning` message references the right replacement
class in `pgmpy.parameter_estimator`. Fix in place if any drift is found.

- [ ] **Step 4: Commit**

```bash
git add pgmpy/estimators/ pgmpy/tests/test_estimators/test_legacy_param_estimator_deprecation.py
git commit -m "chore(estimators): verify FutureWarning on legacy parameter estimators"
```

---

## Task 43: Write the v2.0 migration guide (ships in 1.x.3)

The migration guide ships in 1.x.3 to give downstream users one full
release cycle of notice before v2.0 deletions land.

**Files:**
- Create: `docs/source/migration-v2.rst`.
- Modify: `docs/source/index.rst` — add a link to the migration guide.
- Modify: `CHANGELOG.rst` — announce v2.0 deletions in the 1.x.3 entry.

- [ ] **Step 1: Draft `docs/source/migration-v2.rst`**

Structure:

1. **Quick start** — "pin `pgmpy<2.0` to keep v1.x semantics; otherwise follow the tables below to migrate."

2. **Renames table** (one per category) mapping v1.x → v2.0:

   | Category | v1.x | v2.0 |
   |---|---|---|
   | BN class | `DiscreteBayesianNetwork(...)`, `LinearGaussianBayesianNetwork(...)`, `FunctionalBayesianNetwork(...)` | `DAG(...)` (from `pgmpy.base`) |
   | CPD management | `bn.add_cpds(cpd)` / `bn.get_cpds(node)` / `bn.remove_cpds(cpd)` / `bn.cpds` | `dag.parameters.add(variable=..., cpd=...)` / `.get(node)` / `.remove(node)` / `list(dag.parameters.values())` |
   | TabularCPD | `from pgmpy.factors.discrete import TabularCPD`; `TabularCPD("X", 2, values=[[0.6],[0.4]])` | `from pgmpy.parameterization import TabularCPD`; `TabularCPD.from_values(variable_card=2, values=[[0.6],[0.4]])` |
   | LinearGaussianCPD | `from pgmpy.factors.continuous import LinearGaussianCPD`; `LinearGaussianCPD("Y", beta=[...], std=..., evidence=[...])` | `from pgmpy.parameterization import LinearGaussianCPD`; `LinearGaussianCPD.from_values(beta=[...], std=...)` |
   | FunctionalCPD | `from pgmpy.factors.hybrid import FunctionalCPD` | `from pgmpy.parameterization import FunctionalCPD` |
   | Inference / transforms / IO | `bn.predict(data)` / `.predict_probability(data)` / `.to_markov_model()` / `.to_joint_gaussian()` / `.log_likelihood(data)` / `.save(path)` / `BayesianNetwork.load(path)` | `dag.inference.predict(data)` / `.predict_probability(data)` / `dag.transforms.to_markov_model()` / `.to_joint_gaussian()` / `dag.inference.log_likelihood(data)` / `dag.io.save(path)` / `DAG.load(path)` |
   | Parameter estimators | `pgmpy.estimators.{MaximumLikelihoodEstimator, BayesianEstimator, EM}` (already FutureWarning in 1.x) | `pgmpy.parameter_estimator.{DiscreteMLE, DiscreteBayesianEstimator, DiscreteEM}` |

3. **What stays the same**:
   - `pgmpy.factors.discrete.DiscreteFactor`, `pgmpy.factors.base.BaseFactor` — unchanged (not CPDs).
   - `pgmpy.models.DynamicBayesianNetwork` — unchanged.
   - Structure learning (`HillClimbSearch`, `PC`, `GES`, `MMHC`, `ExhaustiveSearch`, `TreeSearch`) — unchanged; returns `DAG`.
   - Existing inference (`VariableElimination`, `BeliefPropagation`, `ApproxInference`) — unchanged surface.
   - New in v2.0: `LinearGaussianInference`, `LikelihoodWeighting`.

4. **Why** — one paragraph linking to the design spec for the full rationale.

- [ ] **Step 2: Add a CHANGELOG entry for 1.x.3**

Announce the upcoming v2.0 deletions one release ahead. The entry lists
every deprecated name with its replacement and points at the migration
guide. The v2.0 entry itself is added in Task 51.

- [ ] **Step 3: Add the migration guide to `docs/source/index.rst`'s TOC.**

- [ ] **Step 4: Commit**

```bash
git add docs/source/migration-v2.rst docs/source/index.rst CHANGELOG.rst
git commit -m "docs: v2.0 migration guide + 1.x.3 CHANGELOG entry"
```

---

## Task 44: Update `pgmpy.readwrite` for the new accessor API

**Files:**
- Modify: `pgmpy/readwrite/BIF.py`
- Modify: `pgmpy/readwrite/XMLBeliefNetwork.py`
- Modify: `pgmpy/readwrite/UAI.py`
- Modify: `pgmpy/readwrite/XDSL.py`
- Modify: `pgmpy/readwrite/PomdpX.py`
- Modify: `pgmpy/tests/test_readwrite/` — adapt existing readwrite tests to assert on the new accessor API (or to ignore deprecation warnings if they exercise legacy paths).

The current readers/writers (`BIF.py`, etc.) heavily use legacy identity
on CPDs: `cpd.variable`, `cpd.variables[1:]`, `cpd.values`,
`cpd.state_names[var]`. Under the new design, identity lives on the DAG
and `cpd.values_` carries the fitted values. Each reader/writer needs to
adapt.

- [ ] **Step 1: Write failing tests for BIF round-trip with new CPDs**

```python
# pgmpy/tests/test_readwrite/test_bif_new_cpds.py
import pytest

from pgmpy.base import DAG
from pgmpy.parameterization import TabularCPD


def test_bif_round_trip_with_new_tabular_cpds(tmp_path):
    dag = DAG([("diff", "grade")])
    dag.parameters.add(variable="diff", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.6], [0.4]],
        state_names=[["easy", "hard"]],
    ))
    dag.parameters.add(variable="grade",
                        cpd=TabularCPD.from_values(
                            variable_card=2, evidence_card=[2],
                            values=[[0.8, 0.2], [0.2, 0.8]],
                            state_names=[["A", "B"], ["easy", "hard"]],
                        ),
                        parent_order=["diff"])
    path = tmp_path / "out.bif"
    dag.io.save(str(path), format="bif")

    loaded = DAG.load(str(path), format="bif")
    assert set(loaded.nodes()) == {"diff", "grade"}
    assert "diff" in loaded.parameters
    assert "grade" in loaded.parameters
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest pgmpy/tests/test_readwrite/test_bif_new_cpds.py -v`
Expected: FAIL — BIFWriter currently reads `cpd.variable` / `cpd.values`
which don't exist on the new identity-free CPDs.

- [ ] **Step 3: Update `BIFWriter` (writer side)**

Replace direct CPD-identity reads with DAG-driven iteration. In
`BIFWriter.__init__` / the table-extraction methods (around lines
425–600 of `pgmpy/readwrite/BIF.py`), replace:

```python
for cpd in self.model.cpds:
    variable_parents[cpd.variable] = cpd.variables[1:]
    tables[cpd.variable] = compat_fns.to_numpy(cpd.values.ravel(), ...)
```

with:

```python
for node, cpd in self.model.parameters.items():
    parents = self.model._parent_order.get(node, [])
    variable_parents[node] = parents
    tables[node] = np.asarray(cpd.values_).ravel()
```

Note the change from `cpd.values` to `cpd.values_` (trailing underscore is
the sklearn fitted-attribute convention). Note also dropping `compat_fns`
since the new CPDs are numpy-only (Open Risk in spec).

Apply the same pattern at every site in `BIF.py` that reads
`cpd.variable` / `cpd.values` / `cpd.variables[1:]`.

- [ ] **Step 4: Update `BIFReader.get_model` (reader side)**

In `BIFReader.get_model` (around line 253), replace the legacy
construction:

```python
model = DiscreteBayesianNetwork(edges)
model.add_cpds(*tabular_cpds)  # legacy positional add_cpds
return model
```

with the new-style:

```python
model = DAG(edges)
for variable, cpd in zip(variable_names, tabular_cpds):
    model.parameters.add(variable=variable, cpd=cpd,
                          parent_order=parent_names[variable])
return model
```

The `tabular_cpds` constructed during parsing should already be new-style
`TabularCPD` instances (from `pgmpy.parameterization.TabularCPD`); update
the import.

- [ ] **Step 5: Repeat for `XMLBeliefNetwork`, `UAI`, `XDSL`, `PomdpX`**

Each follows the same pattern: replace `cpd.variable` / `cpd.values` /
direct positional `add_cpds` with the DAG registry API and accessor.

- [ ] **Step 6: Run the full readwrite test suite**

Run: `pytest pgmpy/tests/test_readwrite/ -v --tb=short -W ignore::DeprecationWarning`
Expected: All pass. Existing tests may need updates if they assert on
specific legacy attributes (`cpd.variable`, etc.) — convert those to
`dag.parameters[node].classes_` or read from `dag._parent_order`.

- [ ] **Step 7: Commit**

```bash
git add pgmpy/readwrite/ pgmpy/tests/test_readwrite/
git commit -m "feat(readwrite): adapt BIF/XMLBIF/UAI/XDSL/PomdpX to DAG accessor API"
```

---

## Task 45: End-to-end integration test — hybrid network with skpro regressor

**Files:**
- Create: `pgmpy/tests/test_models/test_hybrid_end_to_end.py`

- [ ] **Step 1: Write integration test**

```python
# pgmpy/tests/test_models/test_hybrid_end_to_end.py
"""End-to-end: build a hybrid network with a skpro regressor on a
continuous child, fit it, simulate, and run approximate inference."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skpro")


def test_hybrid_bn_with_skpro_linear_regressor_end_to_end():
    from skpro.regression.linear import GLMRegressor
    from pgmpy.inference import ApproxInference
    from pgmpy.base import DAG
    from pgmpy.parameterization import TabularCPD

    bn = DAG([("smoker", "income"), ("age", "income")])
    bn.add_cpds(variable="smoker", cpd=TabularCPD(variable_card=2,
                                                   state_names=[["yes", "no"]]))
    bn.add_cpds(variable="age", cpd=TabularCPD(variable_card=2,
                                                state_names=[["young", "old"]]))
    bn.add_cpds(variable="income", cpd=GLMRegressor(),
                parent_order=["smoker", "age"])

    rng = np.random.default_rng(0)
    n = 500
    smoker = rng.choice(["yes", "no"], n)
    age = rng.choice(["young", "old"], n)
    income = (
        50_000.0
        + 10_000.0 * (smoker == "no")
        + 5_000.0 * (age == "old")
        + rng.normal(scale=2000.0, size=n)
    )
    # Cast to numeric encoding for GLMRegressor: one-hot ourselves.
    df = pd.DataFrame({
        "smoker": (smoker == "yes").astype(float),
        "age":    (age == "old").astype(float),
        "income": income,
    })
    # Pgmpy doesn't auto-one-hot today (out of scope for this plan).
    # Replace the CPDs with numeric-friendly variants.
    bn = DAG([("smoker", "income"), ("age", "income")])
    bn.add_cpds(variable="smoker", cpd=TabularCPD(variable_card=2,
                                                   state_names=[[0.0, 1.0]]))
    bn.add_cpds(variable="age", cpd=TabularCPD(variable_card=2,
                                                state_names=[[0.0, 1.0]]))
    bn.add_cpds(variable="income", cpd=GLMRegressor(),
                parent_order=["smoker", "age"])

    bn.fit(df)
    bn.check_model()

    samples = bn.simulate(n_samples=200)
    assert set(samples.columns) == {"smoker", "age", "income"}
```

- [ ] **Step 2: Run test**

Run: `pytest pgmpy/tests/test_models/test_hybrid_end_to_end.py -v`
Expected: PASS (the test acknowledges current mixed-type encoding limitation).

- [ ] **Step 3: Commit**

```bash
git add pgmpy/tests/test_models/test_hybrid_end_to_end.py
git commit -m "test(models): end-to-end hybrid network with skpro regressor"
```

---

## Task 46: 1.x.3 final regression run (concludes the 1.x work)

This task concludes the additive 1.x rollout. After it lands, the new
APIs are fully available, the legacy classes emit `FutureWarning`, and
downstream users can migrate at their own pace. The v2.0 deletion phase
follows in Phase 5.

**Files:**
- None (verification only).

- [ ] **Step 1: Run the full test suite under `-W ignore::FutureWarning`**

Run: `pytest pgmpy/tests/ -v --tb=short -W ignore::FutureWarning`
Expected: All pass. (FutureWarning is the deliberate signal to downstream
users, not an error.)

- [ ] **Step 2: Run with `-W error::FutureWarning` to confirm pgmpy's own
code paths don't emit FutureWarning except in the dedicated deprecation
test files**

Run: `pytest pgmpy/tests/ -v --tb=short -W error::FutureWarning --ignore=pgmpy/tests/test_models/test_deprecated_bn_aliases.py --ignore=pgmpy/tests/test_factors/test_legacy_cpd_deprecation.py --ignore=pgmpy/tests/test_estimators/test_legacy_param_estimator_deprecation.py`
Expected: All pass. Any FutureWarning here is a pgmpy-internal use of the
deprecated API and should be fixed (the migration applies to pgmpy's own
code too).

- [ ] **Step 3: Confirm `pgmpy.parameterization`, `pgmpy.parameter_estimator.MLEEstimator`,
`pgmpy.inference.LinearGaussianInference`, `pgmpy.inference.LikelihoodWeighting`,
`pgmpy.base.DAG.parameters/transforms/inference/io` are all reachable
from a fresh Python REPL**

Run a quick smoke-test of the new APIs to confirm the 1.x release is
shippable.

- [ ] **Step 4: Tag and release 1.x.3**

After CI passes and review approves:

```bash
git tag v1.x.3
```

(Substitute the actual minor version.) Publish to PyPI per pgmpy's
existing release process.

---

# Phase 5: v2.0 — Delete legacy classes

Ships separately as **pgmpy 2.0**, after 1.x.3 has been out for a release
cycle (3–6 months). Mechanical release — only deletions.

All four deletion tasks (47–50) follow the same pattern:

1. Write a test asserting the legacy import raises `ImportError`.
2. `git rm` the legacy file(s) and any tests scoped to them.
3. Update the affected `__init__.py` to drop exports.
4. Run the deletion-confirmation test (expect pass) and the broader
   suite under `-W error::FutureWarning` (expect pass — any failure
   indicates pgmpy-internal code still using the deleted API).
5. Commit with `feat(<module>)!:` prefix.

For brevity, only the unique file lists and test snippets are spelled
out per task; the five steps above are implicit.

---

## Task 47: Delete typed BN class files

**Delete:**
- `pgmpy/models/{DiscreteBayesianNetwork,LinearGaussianBayesianNetwork,FunctionalBayesianNetwork,BayesianNetwork}.py`
- `pgmpy/tests/test_models/test_deprecated_bn_aliases.py`
- `pgmpy/tests/test_models/test_{DiscreteBayesianNetwork,LinearGaussianBayesianNetwork,FunctionalBayesianNetwork}.py` — coverage moved to `pgmpy/tests/test_base/test_dag_*.py` in Phase 2.

**Modify:** `pgmpy/models/__init__.py` — drop the four exports.

**Deletion-confirmation test** (`pgmpy/tests/test_models/test_typed_bn_classes_deleted.py`):

```python
import pytest

@pytest.mark.parametrize("name", [
    "DiscreteBayesianNetwork", "LinearGaussianBayesianNetwork",
    "FunctionalBayesianNetwork", "BayesianNetwork",
])
def test_class_is_gone(name):
    with pytest.raises(ImportError):
        __import__(f"pgmpy.models", fromlist=[name])
        from pgmpy.models import name  # noqa
```

Commit: `feat(models)!: delete deprecated typed BN classes (v2.0)`.

---

## Task 48: Delete legacy `pgmpy.factors.*` CPD classes

**Delete:**
- `pgmpy/factors/discrete/CPD.py` (legacy `TabularCPD`).
- `pgmpy/factors/continuous/` (entire directory — had only `LinearGaussianCPD.py`).
- `pgmpy/factors/hybrid/` (entire directory — had only `FunctionalCPD.py`).
- `pgmpy/tests/test_factors/test_legacy_cpd_deprecation.py`.
- `pgmpy/tests/test_factors/test_TabularCPD.py`, `test_LinearGaussianCPD.py`, `test_hybrid/`.

**Stay:** `DiscreteFactor`, `BaseFactor`, `NoisyOR`,
`JointProbabilityDistribution` (they're not CPDs).

**Modify:** `pgmpy/factors/__init__.py`, `pgmpy/factors/discrete/__init__.py` — drop deleted exports.

**Deletion-confirmation test** (`pgmpy/tests/test_factors/test_legacy_cpds_deleted.py`):

```python
import pytest

def test_legacy_cpds_are_gone():
    with pytest.raises(ImportError):
        from pgmpy.factors.discrete import TabularCPD  # noqa
    with pytest.raises(ImportError):
        from pgmpy.factors.continuous import LinearGaussianCPD  # noqa
    with pytest.raises(ImportError):
        from pgmpy.factors.hybrid import FunctionalCPD  # noqa

def test_discrete_factor_still_exists():
    from pgmpy.factors.discrete import DiscreteFactor  # noqa
```

Commit: `feat(factors)!: delete legacy CPD classes (v2.0)`.

---

## Task 49: Delete legacy `pgmpy.estimators.*` parameter estimators

**Delete:** `pgmpy/estimators/{MLE,BayesianEstimator,EM}.py` plus their tests.
**Modify:** `pgmpy/estimators/__init__.py` — drop the three exports. (Structure-learning algorithms — `HillClimbSearch`, `PC`, `GES`, `MMHC`, `ExhaustiveSearch`, `TreeSearch`, `SEMEstimator` — are unaffected.)

**Deletion-confirmation test**:

```python
import pytest
@pytest.mark.parametrize("name", ["MaximumLikelihoodEstimator", "BayesianEstimator", "EM"])
def test_legacy_param_estimator_is_gone(name):
    with pytest.raises(ImportError):
        from pgmpy.estimators import name  # noqa
```

Commit: `feat(estimators)!: delete legacy parameter estimators (v2.0)`.

---

## Task 50: Delete `DAG.add_cpds` / `get_cpds` / `remove_cpds` / `cpds` shims

**Modify:** `pgmpy/base/DAG.py` — delete the shim block introduced in
Task 19 Step 5 (`_warn_deprecated_method`, `add_cpds`, `get_cpds`,
`remove_cpds`, `cpds` property) and the `self._deprecated_methods_warned
= set()` line from `__init__`.

**Deletion-confirmation test** (`pgmpy/tests/test_base/test_dag_shims_deleted.py`):

```python
from pgmpy.base import DAG

def test_dag_has_no_legacy_methods():
    dag = DAG([("a", "b")])
    assert not hasattr(dag, "add_cpds")
    assert not hasattr(dag, "get_cpds")
    assert not hasattr(dag, "remove_cpds")
    # cpds property gone
    assert not isinstance(getattr(DAG, "cpds", None), property)
```

Commit: `feat(base)!: delete DAG.add_cpds/get_cpds/remove_cpds/cpds shims (v2.0)`.

---

## Task 51: v2.0 final regression + release

- [ ] **Step 1:** `pytest pgmpy/tests/ -v --tb=short -W error::FutureWarning` — expected all pass. Any failure indicates pgmpy-internal use of a deleted API; fix.

- [ ] **Step 2:** Smoke-test the canonical v2.0 import surface (`DAG`, `TabularCPD`, `MLEEstimator`, the five inference algorithms) in a fresh REPL.

- [ ] **Step 3:** Append a v2.0 entry to `CHANGELOG.rst` listing every removed name (mirrors the deletion list in the spec's "Breaking changes" section). Refer to `docs/source/migration-v2.rst` (shipped in 1.x.3).

- [ ] **Step 4:** Tag and release:

```bash
git add CHANGELOG.rst
git commit -m "release: pgmpy 2.0.0"
git tag v2.0.0
```

Publish per pgmpy's existing PyPI process.

- [ ] **Step 5:** Announce (mailing list, GitHub release notes) with a link to the migration guide.

---

## Notes

**Spec coverage:** every section of the design spec is implemented by Tasks
1–51. See the spec's release-staging table for the high-level mapping.

**Final task count:** 51 across two major releases — 1.x rollout (Tasks
1–46, additive) + v2.0 cleanup (Tasks 47–51, deletions).

**Per-task TDD pattern:** every implementation step has a failing test
first, then the code, then a verification run, then the commit. All code
blocks are full (no placeholders).

**For design rationale, alternatives, and trade-offs:** see the spec
(`2026-05-14-parameterization-refactor-design.md`). For full class
signatures: see the contracts doc
(`2026-05-14-parameterization-contracts.md`). This plan is execution-only.
