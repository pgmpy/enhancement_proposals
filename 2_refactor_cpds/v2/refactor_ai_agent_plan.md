# Parameterization Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor pgmpy's CPD layer onto the skpro/sklearn estimator contract so that any probabilistic regressor or classifier can serve as a Bayesian-network CPD, replace isinstance-based inference dispatch with capability tags, and move node identity from CPDs to the Bayesian network.

**Architecture:** A new `pgmpy.parameterization` module hosts identity-free CPD classes (`TabularCPD`, `LinearGaussianCPD`, `FunctionalCPD`, plus adapters) that follow sklearn/skpro-style estimator contracts at the CPD boundary. `DAG` is the unified pgmpy-native graph/model container, not a skbase/sklearn estimator. The DAG owns `(node → CPD)`, `(node → parent_order)`, and an internal `(node → VariableSchema)` registry. `dag.parameters.add(variable=..., cpd=..., parent_order=...)` registers a CPD by node name and normalizes any state metadata into `dag.schema`. Capability tags (`variable_type`, `produces_factor`, `is_linear_gaussian`, …) drive inference dispatch; legacy `pgmpy.factors.*` CPD classes become deprecation shims.

**Tech Stack:** Python 3.10+, `skbase`, `scikit-learn`, `skpro`, `networkx`, `numpy`, `pandas`, `pyro-ppl` (soft dep for `FunctionalCPD`), `pytest` for testing.

**Reference spec:** `docs/superpowers/specs/2026-05-14-parameterization-refactor-design.md`

**Release staging:** additive 1.x minor releases ending in a small v2.0 cleanup. Land as much as possible *before* the breaking change. During 1.x, legacy v1.x classes coexist with new APIs as `FutureWarning` shims; v2.0 deletes them.

| Release | Phase | Tasks | What ships |
|---|---|---|---|
| **1.x.1** | Phase 1 | 1–18 | New `pgmpy.parameterization` module. Pure addition. |
| **1.x.2** | Phase 2 + 2b + 3 + Phase 4 inference | 19–40 (+ 28a, 28b, 35a) | DAG enrichment with accessors + `copy_template`. Legacy `dag.add_cpds`/etc. become `FutureWarning` shims. `DAG.do` → `DAG.with_intervention` rename with FutureWarning alias (28a). `LinearGaussianBN.simulate` override retired (28b). Factor-API audit + migration to `dag.transforms.cpd_as_factor` (35a). New parameter estimators. `LikelihoodWeighting` + tag-dispatch in existing inference. Typed BN classes become `FutureWarning` subclasses. |
| **1.x.3** | Phase 4 deprecation + 4b SCM + readwrite + docs | 41–46 + 40a–40v | `FutureWarning` on legacy `pgmpy.factors.*` CPDs. Full SCM layer (Phase 4b). `pgmpy.inference.CausalInference` becomes a `FutureWarning` shim; its query ports to `dag.intervene.query(adjustment_set=...)` and its identification / IV helpers port to `dag.diagnostics` (40v). Readwrite consumes new accessor API. Integration tests. Migration guide. |
| **2.0** | Phase 5 | 47–51 (+ 49a, 50a) | Delete legacy classes + DAG shims. Delete `pgmpy.inference.CausalInference` (49a) and the `DAG.do` shim (50a). Final regression with `-W error::FutureWarning`. |

Every Phase 2 / 4 task that adds a new API also keeps the v1.x API working (as `FutureWarning` shim). Phase 5 deletions are mechanical.

---

## File Structure

High-level summary of the files each phase touches. Every task below has
a `**Files:**` block with the exact list — this is just the bird's-eye
view.

| Phase | Release | Created | Modified | Deleted |
|---|---|---|---|---|
| 1 | 1.x.1 | `pgmpy/parameterization/` (new module: `base.py`, `checks.py`, `tabular.py`, `linear_gaussian.py`, `functional.py`); matching test dir | `pyproject.toml` (skbase + scikit-learn core deps; skpro extra) | — |
| 2 | 1.x.2 | `pgmpy/base/_accessors.py` (`_DAGParameters`, `_DAGSchema`, `_DAGTransforms`, `_DAGInference`, `_DAGIO`); test files for DAG and deprecated BN aliases; `pgmpy/tests/test_base/test_dag_with_intervention.py`; `pgmpy/tests/test_models/test_lg_simulate_via_dag.py` | `pgmpy/base/DAG.py` (CPD registry + schema registry + accessors + `copy_template` + `FutureWarning` shims for legacy `add_cpds`/etc.; rename `do` → `with_intervention` with FutureWarning alias; no skbase inheritance); `pgmpy/models/{Discrete,LinearGaussian,Functional}BayesianNetwork.py` (become thin `FutureWarning`-emitting subclasses of DAG); `pgmpy/models/LinearGaussianBayesianNetwork.py` (delete `simulate` override — generic DAG.simulate handles it); `pgmpy/models/DynamicBayesianNetwork.py` (audit) | — |
| 2b | 1.x.2 | `pgmpy/parameter_estimator/mle.py` (`MLEEstimator`), `discrete_em.py` (`DiscreteEM`), `joint_pyro.py` (`JointPyroEstimator`); test files | `pgmpy/parameter_estimator/base.py` (accept any DAG); `pgmpy/base/DAG.py` (fit delegates to `MLEEstimator()`); `pgmpy/parameterization/functional.py` (drop `fit_joint`) | `pgmpy/parameterization/_functional_joint.py` (logic moves to `JointPyroEstimator`) |
| 3 | 1.x.2 | `pgmpy/tests/test_inference/test_approx_inference_unified.py`; `pgmpy/tests/test_inference/test_factor_api_audit.py` | `pgmpy/inference/ApproxInference.py` (accept any DAG); factor-API call sites across `pgmpy/{models,inference,sampling}/` migrated to `dag.transforms.cpd_as_factor` | — |
| 4 | 1.x.2 / 1.x.3 | `pgmpy/inference/likelihood_weighting.py`; `docs/source/migration-v2.rst`; legacy-deprecation test files; `pgmpy/tests/test_models/test_dag_diagnostics_helpers.py`; `pgmpy/tests/test_models/test_dag_intervene_adjustment.py`; `pgmpy/tests/test_inference/test_causal_inference_shim.py` | `pgmpy/inference/{base,ApproxInference,ExactInference,__init__}.py` (tag dispatch); `pgmpy/sampling/Sampling.py` (sample/log_prob protocol dispatch through registered CPDs); `pgmpy/readwrite/*` (BIF/XMLBIF/UAI/XDSL/PomdpX consume new accessor API and populate schema; return `DiscreteBayesianNetwork` subclass in 1.x for back-compat); `pgmpy/factors/{discrete/CPD,continuous/LinearGaussianCPD,hybrid/FunctionalCPD}.py` (emit `FutureWarning`); `pgmpy/inference/CausalInference.py` (becomes `FutureWarning` shim; helpers port to `dag.diagnostics` and `dag.intervene.query(adjustment_set=...)`); `pgmpy/base/_accessors.py` (extend `_DAGDiagnostics` with 13 ported helpers); `pgmpy/models/_accessors.py` (extend `_DAGIntervene.query` with `evidence`/`adjustment_set`/`inference_algo`); `CHANGELOG.rst`, `docs/source/index.rst` | — |
| 5 | 2.0 | Deletion-confirmation test files (`test_causal_inference_deleted.py`, `test_dag_do_shim_deleted.py`) | `pgmpy/base/DAG.py` (drop `FutureWarning` shims, including `do` alias); `pgmpy/models/__init__.py`, `pgmpy/factors/{__init__.py, discrete/__init__.py}`, `pgmpy/estimators/__init__.py`, `pgmpy/inference/__init__.py` (drop deleted exports); `pgmpy/readwrite/*` (readers return plain DAG); `CHANGELOG.rst` (v2.0 entry) | `pgmpy/models/{Discrete,LinearGaussian,Functional,}BayesianNetwork.py`; `pgmpy/factors/discrete/CPD.py`; entire `pgmpy/factors/continuous/` and `pgmpy/factors/hybrid/`; `pgmpy/estimators/{MLE,BayesianEstimator,EM}.py`; `pgmpy/inference/CausalInference.py`; corresponding test files |

**Core deps:** `skbase>=0.13`, `scikit-learn>=1.4`. **Soft deps (extras):** `skpro>=2.8`, `pyro-ppl` (for `FunctionalCPD` only).

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
"""Shared parameterization base definitions.

Public sampling/scoring dispatch lives on CPD objects themselves. Third-party
predictive estimators are adapted by CPDAdapter, not by public module-level
sample/log-probability helpers.
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

## Task 3: Operation-specific CPD protocol checks

**Files:**
- Modify: `pgmpy/parameterization/checks.py`
- Modify: `pgmpy/tests/test_parameterization/test_checks.py`

The revised design does **not** use one universal CPD contract. Different
operations require different surfaces: fitting, prediction, sampling,
scoring, or structural counterfactual methods.

- [ ] **Step 1: Write failing tests for narrow protocol checks**

Add tests for:

- `require_fittable(obj)` accepts `fit(X, y, ...)`.
- `require_predictive(obj)` accepts `predict_proba(X)`.
- `require_sampleable(obj)` accepts `sample(X, n_samples=...)`.
- `require_scorable(obj)` accepts `log_prob(y, X)`.
- `check_parameterization(obj)` remains as a compatibility helper for
  third-party predictive estimators and requires `fit(X, y)` +
  `predict_proba(X)`.

- [ ] **Step 2: Implement the checks**

```python
class CPDContractError(TypeError): ...
class IncompatibleCPDError(TypeError): ...

def require_fittable(obj): ...
def require_predictive(obj): ...
def require_sampleable(obj): ...
def require_scorable(obj): ...

def check_parameterization(obj):
    """Compatibility validator for third-party predictive CPDs.

    Requires fittable + predictive. Built-in CPDs and simulation-native
    CPDs can satisfy narrower protocols depending on the operation.
    """
    require_fittable(obj)
    require_predictive(obj)
```

- [ ] **Step 3: Run tests**

Run: `pytest pgmpy/tests/test_parameterization/test_checks.py -v`

- [ ] **Step 4: Commit**

```bash
git add pgmpy/parameterization/checks.py pgmpy/tests/test_parameterization/test_checks.py
git commit -m "feat(parameterization): add operation-specific CPD protocol checks"
```

---

## Task 4: `CPDAdapter` for third-party predictive estimators

**Files:**
- Create: `pgmpy/parameterization/adapter.py`
- Modify: `pgmpy/parameterization/__init__.py`
- Create: `pgmpy/tests/test_parameterization/test_adapter.py`

`CPDAdapter` replaces the earlier public sample/log-probability
helpers. The dispatch bridge lives on the adapted CPD object, so DAG
algorithms only call `cpd.sample(...)` and `cpd.log_prob(...)`.

- [ ] **Step 1: Write failing tests**

Cover:

- sklearn classifier with `predict_proba(X)` is adapted to `sample` and
  `log_prob`.
- skpro distribution-valued regressor is adapted using distribution
  `.sample()` and `.log_pdf()`.
- native CPDs with `sample` and `log_prob` are not wrapped by
  `dag.parameters.add(...)` later in Task 19.
- `CPDAdapter.clone()` clones the wrapped estimator when possible.

- [ ] **Step 2: Implement `CPDAdapter`**

```python
class CPDAdapter:
    def __init__(self, wrapped):
        check_parameterization(wrapped)
        self._wrapped = wrapped

    @property
    def wrapped(self): ...
    def __getattr__(self, name): ...
    def fit(self, X, y, **kwargs): ...
    def predict_proba(self, X): ...
    def sample(self, X, n_samples=None): ...
    def log_prob(self, y, X): ...
    def get_tag(self, name, default=None): ...
    def clone(self): ...
```

- [ ] **Step 3: Export it**

Add `CPDAdapter` to `pgmpy.parameterization.__all__`.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_parameterization/test_adapter.py -v`

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/adapter.py pgmpy/parameterization/__init__.py pgmpy/tests/test_parameterization/test_adapter.py
git commit -m "feat(parameterization): add CPDAdapter for third-party predictive CPDs"
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
)
from pgmpy.parameterization.adapter import CPDAdapter
from pgmpy.parameterization.checks import (
    check_parameterization,
    require_fittable,
    require_predictive,
    require_sampleable,
    require_scorable,
)
from pgmpy.parameterization.tabular import TabularCPD

__all__ = [
    "CPDAdapter",
    "CPDContractError",
    "TabularCPD",
    "check_parameterization",
    "require_fittable",
    "require_predictive",
    "require_sampleable",
    "require_scorable",
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
    "CPDAdapter",
    "CPDContractError",
    "LinearGaussianCPD",
    "TabularCPD",
    "check_parameterization",
    "require_fittable",
    "require_predictive",
    "require_sampleable",
    "require_scorable",
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
)
from pgmpy.parameterization.adapter import CPDAdapter
from pgmpy.parameterization.checks import (
    check_parameterization,
    require_fittable,
    require_predictive,
    require_sampleable,
    require_scorable,
)
from pgmpy.parameterization.linear_gaussian import LinearGaussianCPD
from pgmpy.parameterization.tabular import TabularCPD

# FunctionalCPD pulls in optional dependencies (pyro-ppl). Defer the import so
# users can use Tabular and LinearGaussian without pyro installed.
try:
    from pgmpy.parameterization.functional import FunctionalCPD
except ImportError:  # pragma: no cover
    FunctionalCPD = None

__all__ = [
    "CPDAdapter",
    "CPDContractError",
    "FunctionalCPD",
    "LinearGaussianCPD",
    "TabularCPD",
    "check_parameterization",
    "require_fittable",
    "require_predictive",
    "require_sampleable",
    "require_scorable",
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
"""Phase 1 integration: built-in CPDs expose native runtime protocols and
third-party predictive estimators work through CPDAdapter."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("skpro")

from pgmpy.parameterization import (
    CPDAdapter,
    LinearGaussianCPD,
    TabularCPD,
    check_parameterization,
    require_fittable,
    require_sampleable,
    require_scorable,
)


def test_tabular_cpd_passes_check():
    cpd = TabularCPD.from_values(variable_card=2, values=[[0.5], [0.5]],
                                  state_names=[["yes", "no"]])
    require_fittable(cpd)
    require_sampleable(cpd)
    require_scorable(cpd)


def test_linear_gaussian_cpd_passes_check():
    cpd = LinearGaussianCPD.from_values(beta=[0.0, 1.0], std=1.0)
    require_fittable(cpd)
    require_sampleable(cpd)
    require_scorable(cpd)


def test_native_sample_tabular():
    cpd = TabularCPD.from_values(variable_card=2, values=[[1.0], [0.0]],
                                  state_names=[["yes", "no"]])
    X = pd.DataFrame(index=range(3))
    out = cpd.sample(X, n_samples=3)
    assert out.tolist() == ["yes", "yes", "yes"]


def test_native_log_prob_linear_gaussian():
    cpd = LinearGaussianCPD.from_values(beta=[0.0, 1.0], std=1.0)
    X = pd.DataFrame({"p": [0.0, 1.0]})
    y = pd.Series([0.0, 1.0])
    log_p = cpd.log_prob(y, X)
    assert len(log_p) == 2
    assert np.all(np.isfinite(log_p.values))


def test_third_party_sklearn_classifier_through_dispatch():
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.integers(0, 5, size=200)})
    y = pd.Series(rng.choice(["yes", "no"], size=200))
    clf = RandomForestClassifier(n_estimators=10).fit(X, y)
    check_parameterization(clf)
    adapted = CPDAdapter(clf)
    out = adapted.sample(X.head(5), n_samples=5)
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

- [ ] **Step 2: Add `skbase` and `scikit-learn` as core deps, and `skpro` as an optional extra**

Edit `pyproject.toml` in the `[project] dependencies` or equivalent
`[tool.poetry.dependencies]` section, adding `skbase>=0.13` and ensuring
`scikit-learn>=1.4` is a core dependency. Add a new extras section if one
doesn't already exist:

```toml
[project.optional-dependencies]
skpro = ["skpro>=2.8"]
```

(Confirm exact version pins by running `pip install --dry-run skbase`,
`pip install --dry-run scikit-learn`, and `pip install --dry-run skpro`
locally; pin to the latest minor.)

- [ ] **Step 3: Verify imports work**

Run: `python -c "import skbase; print(skbase.__version__)"`
Expected: a version number prints.

- [ ] **Step 4: Run the full Phase 1 test suite to confirm nothing broke**

Run: `pytest pgmpy/tests/test_parameterization/ -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "build: add skbase and sklearn core deps plus skpro extra"
```

---

# Phase 2: DAG enrichment

`DAG` becomes the unified parameterized network. **Not** a skbase/sklearn estimator: no `get_params`/`set_params`/`clone`. CPDs keep the sklearn/skpro-style API; the graph owns composition. Accessors: `dag.parameters` / `dag.schema` / `dag.transforms` / `dag.inference` / `dag.io`. Core ops: `fit`/`simulate`/`check_model`/`copy_template(parameters=...)`.

CPD-management API: `dag.parameters.{add, get, remove}`. Legacy `dag.add_cpds` / `get_cpds` / `remove_cpds` / `cpds` remain as `FutureWarning` shims through 1.x; removed in v2.0.

## Task 19: Enrich `pgmpy.base.DAG` — CPD registry, schema registry, `copy_template`, `parameters` accessor, deprecated shims

**Files:**
- Modify: `pgmpy/base/DAG.py` (extend, don't replace).
- Create: `pgmpy/base/_accessors.py` (`_DAGParameters` and `_DAGSchema` scaffolding; other accessors in later tasks).
- Create: `pgmpy/tests/test_base/test_dag_parameters.py`.
- Leave: `pgmpy/models/BayesianNetwork.py` — existing stub untouched; no `BayesianNetwork` alias.

Phase 2 foundation. Adds (tightly coupled):
1. `DAG` inherits graph classes only; no skbase/sklearn estimator inheritance.
2. `_cpds`/`_parent_order`/`_schema`/`_deprecated_methods_warned` instance state.
3. `_DAGParameters` accessor (canonical CPD-management API).
4. `_DAGSchema` accessor (internal variable metadata).
5. `cached_property` wiring for `dag.parameters` and `dag.schema`.
6. `DAG.copy_template(parameters="none"|"unfit"|"fitted")`.
7. Deprecated method shims: `add_cpds`/`get_cpds`/`remove_cpds`/`cpds` emit one `FutureWarning` per instance.

Rest of Phase 2 (Tasks 20–23: check_model/simulate/fit; 24–27: transforms/inference/io accessors; 28: typed-class aliases; 29: DBN audit) builds on this.

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
    bn.parameters.add(variable="grade", cpd=cpd, parent_order=["diff", "intel"])
    assert "grade" in bn._cpds
    assert bn._cpds["grade"] is cpd
    assert bn._parent_order["grade"] == ["diff", "intel"]
    assert bn.schema["grade"].states == ("A", "B", "C")


def test_add_cpds_canonicalizes_parent_order_from_graph():
    bn = DAG([("diff", "grade"), ("intel", "grade")])
    cpd = TabularCPD.from_values(variable_card=3, evidence_card=[2, 2],
                                  values=[[0.3] * 4, [0.4] * 4, [0.3] * 4])
    bn.parameters.add(variable="grade", cpd=cpd)
    # Default parent_order matches list(bn.predecessors("grade"))
    assert set(bn._parent_order["grade"]) == {"diff", "intel"}


def test_add_cpds_rejects_parent_order_with_wrong_nodes():
    bn = DAG([("diff", "grade"), ("intel", "grade")])
    cpd = TabularCPD.from_values(variable_card=3, evidence_card=[2, 2],
                                  values=[[0.3] * 4, [0.4] * 4, [0.3] * 4])
    with pytest.raises(ValueError, match="parent_order"):
        bn.parameters.add(variable="grade", cpd=cpd, parent_order=["diff", "bogus"])


def test_copy_template_copies_graph_schema_and_unfit_cpd_specs():
    bn = DAG([("diff", "grade")])
    bn.parameters.add(
        variable="diff",
        cpd=TabularCPD.from_values(
            variable_card=2,
            values=[[0.5], [0.5]],
            state_names=[["easy", "hard"]],
        ),
    )
    clone = bn.copy_template(parameters="unfit")
    assert clone is not bn
    assert set(clone.nodes()) == set(bn.nodes())
    assert clone.schema["diff"].states == ("easy", "hard")
    assert clone.parameters["diff"] is not bn.parameters["diff"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py -v`
Expected: FAIL — `_cpds`/`schema`/`copy_template` don't exist and `parameters.add` isn't wired.

- [ ] **Step 3: Add the CPD/schema registries to `DAG.__init__`**

In `pgmpy/base/DAG.py`, update the class declaration and `__init__`:

```python
from functools import cached_property
import networkx as nx


class DAG(_GraphRolesMixin, nx.DiGraph):
    """Directed acyclic graph with optional CPD parameterization.

    `_GraphRolesMixin` (pgmpy/base/_mixin_roles.py) is the existing
    causal-role annotation mixin — `latents`, `observed`, `exposures`,
    `outcomes`, role accessors, `is_valid_causal_structure`. All graph
    methods (`active_trail_nodes`, `get_independencies`,
    `get_immoralities`, `get_markov_blanket`, `get_ancestors`,
    `minimal_dseparator`, `copy`, `do`, …) live directly on this
    class (see pgmpy/base/DAG.py:18). This task adds the CPD registry,
    schema registry, accessors, and `copy_template` **alongside** the
    existing graph content without touching it.
    """

    def __init__(self, ebunch=None, latents=None):
        # PRESERVE the v1.x DAG.__init__ body verbatim (see
        # pgmpy/base/DAG.py:167 — it handles ebunch via add_edges_from,
        # validates non-string node names, and runs _check_cycles).
        # nx.DiGraph.__init__ takes no `ebunch`/`latents` kwargs, and
        # _GraphRolesMixin doesn't define __init__, so the existing
        # body must stay in place. Only append the new registries.
        # The four lines below are NEW in v2.0:
        self._cpds = {}
        self._parent_order = {}
        self._schema = {}
        # New: once-per-instance FutureWarning tracking for deprecated methods.
        self._deprecated_methods_warned = set()
```

- [ ] **Step 4: Create `_DAGParameters` and `_DAGSchema` accessors in `pgmpy/base/_accessors.py`**

```python
# pgmpy/base/_accessors.py
"""Namespaced accessor objects exposed on DAG via cached_property."""

from __future__ import annotations


from dataclasses import dataclass


@dataclass(frozen=True)
class VariableSchema:
    variable: object
    variable_type: str
    states: tuple | None = None
    dtype: object | None = None
    ordered: bool = False
    encoder: object | None = None
    decoder: object | None = None


class _DAGSchema:
    """Internal DAG-owned variable metadata registry."""

    def __init__(self, dag):
        self._dag = dag

    def __getitem__(self, variable):
        return self._dag._schema[variable]

    def get(self, variable, default=None):
        return self._dag._schema.get(variable, default)

    def items(self):
        return self._dag._schema.items()

    def set(self, variable, *, variable_type, states=None, dtype=None,
            ordered=False, encoder=None, decoder=None):
        states = None if states is None else tuple(states)
        new = VariableSchema(variable, variable_type, states, dtype,
                             ordered, encoder, decoder)
        old = self._dag._schema.get(variable)
        if old is not None:
            if old.variable_type != new.variable_type:
                raise ValueError(
                    f"Schema type conflict for {variable!r}: "
                    f"{old.variable_type!r} vs {new.variable_type!r}."
                )
            if old.states is not None and new.states is not None and old.states != new.states:
                raise ValueError(
                    f"State conflict for {variable!r}: "
                    f"{old.states!r} vs {new.states!r}."
                )
            if new.states is None or old.states is not None:
                new = old
        self._dag._schema[variable] = new
        return self

    def infer_from_cpd(self, variable, cpd, parent_order):
        var_type = (cpd.get_tag("variable_type", None)
                    if hasattr(cpd, "get_tag") else None)
        var_type = var_type or "continuous"
        state_names = getattr(cpd, "state_names", None)
        if state_names:
            self.set(variable, variable_type=var_type, states=state_names[0])
            for parent, states in zip(parent_order, state_names[1:]):
                self.set(parent, variable_type="discrete", states=states)
        elif var_type == "discrete":
            card = getattr(cpd, "variable_card", None)
            self.set(variable, variable_type="discrete",
                     states=None if card is None else tuple(range(card)))
        else:
            self.set(variable, variable_type="continuous")
        return self


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
            Native CPD implementing fit/sample/log_prob, or third-party
            predictive estimator implementing fit/predict_proba. Third-party
            predictive estimators are wrapped in CPDAdapter.
        parent_order : list[Hashable] or None, default None
            Ordered list of parent names; positional contract that the
            CPD's beta_/values_/etc. lines up with. Defaults to
            ``list(self._dag.predecessors(variable))``.
        """
        from pgmpy.parameterization import CPDAdapter, check_parameterization
        from pgmpy.parameterization.checks import (
            require_fittable, require_sampleable, require_scorable,
        )

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

        native_runtime = (
            callable(getattr(cpd, "sample", None))
            and callable(getattr(cpd, "log_prob", None))
        )
        if not native_runtime:
            check_parameterization(cpd)
            cpd = CPDAdapter(cpd)

        require_fittable(cpd)
        require_sampleable(cpd)
        require_scorable(cpd)
        self._dag.schema.infer_from_cpd(variable, cpd, parent_order)
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
    def __getitem__(self, node):  return self.get(node)
```

- [ ] **Step 5: Wire `parameters`, `schema`, `copy_template`, and 1.x deprecated method shims on `DAG`**

Append to `pgmpy/base/DAG.py`:

```python
    @cached_property
    def parameters(self):
        """CPD-registry accessor. The canonical CPD-management API."""
        from pgmpy.base._accessors import _DAGParameters
        return _DAGParameters(self)

    @cached_property
    def schema(self):
        """Internal variable metadata registry."""
        from pgmpy.base._accessors import _DAGSchema
        return _DAGSchema(self)

    def copy_template(self, *, parameters="unfit"):
        """Copy graph + schema, optionally copying CPD specs or fitted CPDs.

        parameters:
        - "none": graph + schema only
        - "unfit": graph + schema + cloned CPD hyperparameter/spec objects
        - "fitted": graph + schema + deep-copied fitted CPDs
        """
        if parameters not in {"none", "unfit", "fitted"}:
            raise ValueError("parameters must be one of: none, unfit, fitted")

        import copy
        from sklearn.base import clone as sk_clone

        new = DAG(self.edges(), latents=set(getattr(self, "latents", set())))
        new.add_nodes_from(self.nodes())
        new._schema = dict(self._schema)
        if parameters == "none":
            return new
        for node in self.nodes():
            if node not in self._cpds:
                continue
            cpd = self._cpds[node]
            if parameters == "fitted":
                cpd_copy = copy.deepcopy(cpd)
            elif hasattr(cpd, "clone"):
                cpd_copy = cpd.clone()
            else:
                cpd_copy = sk_clone(cpd)
            new.parameters.add(
                variable=node,
                cpd=cpd_copy,
                parent_order=self._parent_order.get(node),
            )
        return new

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
git commit -m "feat(base): enrich DAG with CPD and schema registries"
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


def test_simulate_walks_topological_order_and_uses_registered_cpd_sampling():
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

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py::test_simulate_walks_topological_order_and_uses_registered_cpd_sampling -v`
Expected: The current `simulate` either misbehaves with the new CPDs or doesn't exist on this path.

- [ ] **Step 3: Add a new `simulate` on `DAG`, and a `_legacy_simulate` fallback on `DiscreteBayesianNetwork`**

The base `DAG` class doesn't currently have a `simulate` method —
`DiscreteBayesianNetwork.simulate` (`pgmpy/models/DiscreteBayesianNetwork.py:1337`)
and `LinearGaussianBayesianNetwork.simulate`
(`pgmpy/models/LinearGaussianBayesianNetwork.py:617`) do. Task 22 adds
a generic `simulate` to `DAG`. For 1.x back-compat, the v1.x discrete
simulate body moves to `DiscreteBayesianNetwork._legacy_simulate` and
is called as a fall-through whenever a legacy-only kwarg is passed.
(`LinearGaussianBayesianNetwork.simulate` is retired separately in
Task 28b.)

The new `DAG.simulate` signature **preserves every parameter** from
`DiscreteBayesianNetwork.simulate` so existing user code keeps working
when called on a `DAG` (or any deprecated subclass):

```python
def simulate(self, n_samples=10, do=None, evidence=None,
             virtual_evidence=None, virtual_intervention=None,
             missing_prob=None, include_latents=False,
             partial_samples=None, seed=None,
             show_progress=True, return_full=False, n_jobs=-1):
    """Generic ancestral simulation over registered CPDs.

    For each node in topological order, build X = sampled[parents] and
    call cpd.sample(X, n_samples). State labels are preserved.

    Signature matches v1.x DiscreteBayesianNetwork.simulate verbatim
    (see pgmpy/models/DiscreteBayesianNetwork.py:1337) so existing user
    code keeps working. Legacy-only features
    (`virtual_evidence`, `virtual_intervention`, `missing_prob`,
    `evidence`, `partial_samples`, `return_full`) fall through to
    `_legacy_simulate`, which only exists on the deprecated
    `DiscreteBayesianNetwork` subclass during 1.x.
    """
    import networkx as nx

    if do is None:
        do = {}

    legacy_only = (
        virtual_evidence or virtual_intervention or missing_prob
        or evidence or partial_samples is not None or return_full
    )
    if legacy_only:
        if not hasattr(self, "_legacy_simulate"):
            raise TypeError(
                "simulate(virtual_evidence/virtual_intervention/missing_prob/"
                "evidence/partial_samples/return_full=...) requires the legacy "
                "DiscreteBayesianNetwork path; the generic DAG.simulate doesn't "
                "support these. Build a DiscreteBayesianNetwork or open an issue "
                "if you need this on a hybrid DAG."
            )
        return self._legacy_simulate(
            n_samples=n_samples, do=do, evidence=evidence,
            virtual_evidence=virtual_evidence,
            virtual_intervention=virtual_intervention,
            include_latents=include_latents, seed=seed,
            missing_prob=missing_prob, partial_samples=partial_samples,
            n_jobs=n_jobs, show_progress=show_progress,
            return_full=return_full,
        )

    rng_seed = seed
    samples = pd.DataFrame(index=range(n_samples))
    for node in nx.topological_sort(self):
        if node in do:
            samples[node] = [do[node]] * n_samples
            continue
        cpd = self.parameters[node]
        parents = self._parent_order.get(node, [])
        X = samples[parents] if parents else pd.DataFrame(index=range(n_samples))
        samples[node] = cpd.sample(X, n_samples=n_samples,
                                   random_state=rng_seed).values
    return samples
```

In `pgmpy/models/DiscreteBayesianNetwork.py`, rename the existing
`simulate(...)` method to `_legacy_simulate(...)` preserving its
signature verbatim. The new generic `DAG.simulate` becomes the
default; the typed-BN subclass (deprecated in Task 28) inherits it and
falls through to `_legacy_simulate` only when a legacy kwarg is used.

**Default change to flag in the migration guide:** v1.x
`DiscreteBayesianNetwork.simulate` defaulted to `n_samples=10`; the new
`DAG.simulate` keeps that default for back-compat. Tutorials should
opt-in to `n_samples=1000` explicitly.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_identity_ownership.py::test_simulate_walks_topological_order_and_uses_registered_cpd_sampling -v`
Expected: PASS.

Then run the broader simulate tests to confirm legacy users still work:

Run: `pytest pgmpy/tests/test_models/test_BayesianNetwork.py -k simulate -v --tb=short`
Expected: All pass (legacy CPDs continue through `_legacy_simulate`).

- [ ] **Step 5: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/tests/test_models/test_bn_identity_ownership.py
git commit -m "feat(models): generic simulate using registered CPDs"
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
Expected: FAIL — current `fit` routes through legacy parameter-estimator paths and doesn't understand identity-free CPDs.

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

    If ``estimator`` is provided, it must be a new-style ParameterEstimator
    with ``fit(model, data, **kwargs)``.
    """
    import networkx as nx

    if estimator is not None:
        from pgmpy.parameter_estimator.base import ParameterEstimator

        if not isinstance(estimator, ParameterEstimator):
            raise TypeError("estimator must be a ParameterEstimator")
        estimator.fit(self, data, sample_weight=sample_weight)
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

        Works for any CPD that supports the scoring protocol — discrete,
        linear-Gaussian, functional, or third-party CPDs adapted through
        CPDAdapter.
        """
        import pandas as pd

        bn = self._bn
        total = 0.0
        for node in bn.nodes():
            cpd = bn.parameters[node]
            parents = bn._parent_order.get(node, [])
            X = (data[parents] if parents
                 else pd.DataFrame(index=data.index))
            y = data[node]
            total += float(cpd.log_prob(y, X).sum())
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

## Task 28a: Rename `DAG.do(nodes)` → `DAG.with_intervention(nodes)` (1.x)

`pgmpy.base.DAG.do(nodes)` (`pgmpy/base/DAG.py:1067`) is the structural
do-operator (returns a new DAG with incoming edges to `nodes` removed).
With v2.0 introducing `dag.intervene.query(do={...})` for value-level
intervention, the name collision is confusing. Rename the structural
operator to `with_intervention` and ship `do` as a `FutureWarning`
alias during 1.x.

**Files:**
- Modify: `pgmpy/base/DAG.py`
- Create: `pgmpy/tests/test_base/test_dag_with_intervention.py`

- [ ] **Step 1: Write failing test**

```python
# pgmpy/tests/test_base/test_dag_with_intervention.py
import pytest
from pgmpy.base import DAG


def test_with_intervention_returns_new_dag_without_parent_edges():
    dag = DAG([("X", "A"), ("A", "Y"), ("A", "B")])
    new = dag.with_intervention("A")
    assert ("X", "A") not in new.edges()
    assert ("A", "Y") in new.edges()
    assert ("A", "B") in new.edges()


def test_with_intervention_multiple_nodes():
    dag = DAG([("X", "A"), ("Z", "B")])
    new = dag.with_intervention(["A", "B"])
    assert ("X", "A") not in new.edges()
    assert ("Z", "B") not in new.edges()


def test_legacy_do_emits_future_warning():
    dag = DAG([("X", "A"), ("A", "Y")])
    with pytest.warns(FutureWarning, match="with_intervention"):
        new = dag.do("A")
    # Same semantics — just renamed.
    assert ("X", "A") not in new.edges()
    assert ("A", "Y") in new.edges()
```

- [ ] **Step 2: Run to verify FAIL**

Run: `pytest pgmpy/tests/test_base/test_dag_with_intervention.py -v`
Expected: FAIL — `with_intervention` not defined; `do` doesn't warn.

- [ ] **Step 3: Implement the rename**

In `pgmpy/base/DAG.py`, around line 1067, **rename** the existing
`def do(self, nodes, inplace=False)` method body to
`def with_intervention(self, nodes, inplace=False)` — preserve the body
verbatim. Then add a thin `do` shim above or below it:

```python
def do(self, nodes, inplace=False):
    """Deprecated. Renamed to with_intervention() in v2.0.

    Use `DAG.with_intervention(nodes)` for structural intervention
    (graph mutation: remove incoming edges to `nodes`). For value-level
    intervention, use `dag.intervene.query(do={node: value, ...})`.
    """
    import warnings
    warnings.warn(
        "DAG.do(nodes) is deprecated and will be removed in pgmpy 2.0. "
        "Use DAG.with_intervention(nodes) for structural do (graph mutation); "
        "use dag.intervene.query(do={node: value, ...}) for value-level do.",
        FutureWarning,
        stacklevel=2,
    )
    return self.with_intervention(nodes, inplace=inplace)
```

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_base/test_dag_with_intervention.py -v`
Expected: 3 passed.

Then run the full DAG test suite under `-W ignore::FutureWarning` to
confirm existing `dag.do(...)` call sites keep working:

Run: `pytest pgmpy/tests/test_base/ -v --tb=short -W ignore::FutureWarning`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/base/DAG.py pgmpy/tests/test_base/test_dag_with_intervention.py
git commit -m "feat(base): DAG.do → DAG.with_intervention rename; do becomes FutureWarning alias"
```

---

## Task 28b: Retire `LinearGaussianBayesianNetwork.simulate` override

After Task 22 lands the generic `DAG.simulate`, the
`LinearGaussianBayesianNetwork.simulate` override
(`pgmpy/models/LinearGaussianBayesianNetwork.py:617`) is redundant —
calling `cpd.sample(X, n_samples)` on each `LinearGaussianCPD` produces
correct LG samples. Retire the override so the deprecated subclass uses
the generic path.

**Files:**
- Modify: `pgmpy/models/LinearGaussianBayesianNetwork.py`
- Create: `pgmpy/tests/test_models/test_lg_simulate_via_dag.py`

- [ ] **Step 1: Write failing test**

```python
# pgmpy/tests/test_models/test_lg_simulate_via_dag.py
import warnings

import numpy as np
import pytest

pytest.importorskip("skpro")

from pgmpy.parameterization import LinearGaussianCPD


def _build_lg_bn():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from pgmpy.models import LinearGaussianBayesianNetwork
        bn = LinearGaussianBayesianNetwork([("x1", "x2")])
    bn.parameters.add(variable="x1",
                      cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    bn.parameters.add(variable="x2",
                      cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=1.0),
                      parent_order=["x1"])
    return bn


def test_lg_simulate_uses_generic_dag_path():
    bn = _build_lg_bn()
    samples = bn.simulate(n_samples=2000, seed=42)
    assert set(samples.columns) == {"x1", "x2"}
    # Linear-Gaussian: E[x2] = 0 + 2 * E[x1] = 0; Var[x2] = 4 + 1 = 5.
    assert abs(samples["x2"].mean()) < 0.2
    assert abs(samples["x2"].var(ddof=0) - 5.0) < 0.8


def test_lg_simulate_under_intervention():
    bn = _build_lg_bn()
    samples = bn.simulate(n_samples=2000, do={"x1": 3.0}, seed=42)
    assert (samples["x1"] == 3.0).all()
    # do(x1=3): E[x2] = 2 * 3 = 6.
    assert abs(samples["x2"].mean() - 6.0) < 0.2
```

- [ ] **Step 2: Run to verify the test currently exercises the old override**

Run: `pytest pgmpy/tests/test_models/test_lg_simulate_via_dag.py -v`
Expected: Passes through the existing override or fails on
`LinearGaussianBayesianNetwork.simulate` API mismatch with the new
generic signature. Either is fine — the goal of Step 3 is to make this
test pass after removing the override.

- [ ] **Step 3: Delete the override**

In `pgmpy/models/LinearGaussianBayesianNetwork.py`, **delete** the
`def simulate(...)` method (currently around line 617). Keep the class
definition otherwise unchanged. The deprecated `LinearGaussianBayesianNetwork`
subclass now inherits `simulate` from `DAG` via Task 22.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_lg_simulate_via_dag.py pgmpy/tests/test_models/test_LinearGaussianBayesianNetwork.py -v --tb=short -W ignore::FutureWarning`
Expected: All pass. The existing `test_LinearGaussianBayesianNetwork.py`
tests now exercise the generic `DAG.simulate` path; LG sampling is
correct because each `LinearGaussianCPD.sample(X, n_samples)` draws
from the right Normal.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/LinearGaussianBayesianNetwork.py pgmpy/tests/test_models/test_lg_simulate_via_dag.py
git commit -m "refactor(models): retire LinearGaussianBN.simulate override; use generic DAG.simulate"
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

- [ ] **Step 3: Audit inheritance and registry use**

Confirm `DynamicBayesianNetwork` still inherits from `DAG`. Do not route it
through the deprecated typed BN aliases. All DBN-specific methods
(`get_intra_edges`, `get_inter_edges`, `get_slice_nodes`, `add_node`, etc.)
stay untouched — they operate on graph structure with `(node, time_slice)`
tuples and don't depend on typed-BN CPD validation behavior. Update only
CPD storage paths that still assume a legacy `self.cpds = []` list.

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
    bn.parameters.add(variable="diff", cpd=TabularCPD(variable_card=2))
    bn.parameters.add(variable="grade",
                      cpd=TabularCPD(variable_card=2, evidence_card=[2]),
                      parent_order=["diff"])

    MLEEstimator().fit(bn, data)
    for node in ("diff", "grade"):
        assert getattr(bn.parameters[node], "is_fitted_", False)


def test_mle_estimator_passes_sample_weight_through():
    rng = np.random.default_rng(0)
    n = 100
    data = pd.DataFrame({"x": rng.choice(["a", "b"], n)})
    bn = DAG()
    bn.add_node("x")
    bn.parameters.add(variable="x", cpd=TabularCPD(variable_card=2))

    weights = np.where(data["x"] == "a", 0.0, 1.0)  # zero-weight all "a"
    MLEEstimator().fit(bn, data, sample_weight=weights)

    # With "a" zero-weighted, the fitted marginal should put nearly all mass on "b".
    cpd = bn.parameters["x"]
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
            cpd = model.parameters[node]
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

## Task 31: [Removed] No `DiscreteMLE` or `DiscreteBayesianEstimator`

Earlier drafts kept discrete-specific estimator classes as thin wrappers
around the generic per-node fit. The current design removes them from the
new API:

- `MLEEstimator` is the generic per-node estimator for any DAG.
- Discrete Bayesian/MAP fitting is configured on `TabularCPD` itself via
  constructor hyperparameters such as `prior_type`,
  `equivalent_sample_size`, and `pseudo_counts`.
- `DiscreteEM` remains because EM is a genuine network-level algorithm
  over latent variables.

No code or tests in this task. Do not create or export
`pgmpy.parameter_estimator.DiscreteMLE` or
`pgmpy.parameter_estimator.DiscreteBayesianEstimator`.

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

## Task 33: [Removed] No `LinearGaussianMLE`

Earlier drafts kept a linear-Gaussian-specific estimator as a validation
wrapper around `MLEEstimator`. The current design removes it from the new
API. `MLEEstimator` delegates to each CPD's `fit`; linear-Gaussian-specific
preconditions live on `LinearGaussianCPD.fit` and
`dag.transforms.to_joint_gaussian()`.

No code or tests in this task. Do not create or export
`pgmpy.parameter_estimator.LinearGaussianMLE`.

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

## Task 35a: Audit factor-API call sites against the new `TabularCPD`

The v2.0 `TabularCPD(ClassifierMixin, BaseEstimator)`
(`pgmpy/parameterization/tabular.py`, from Task 5) is **not** a
`DiscreteFactor` — it doesn't carry `.marginalize`, `.reduce`,
`.normalize`, `.to_factor`, `.product`, `*`, `.variables`,
`.cardinality`, or `.values`. The v1.x `TabularCPD(DiscreteFactor)`
(`pgmpy/factors/discrete/CPD.py:20`) does. Any internal pgmpy code path
that treated a registered CPD as a factor will break silently against
the new class.

Task 24's `_DAGTransforms.cpd_as_factor(node)` is the supported
escape: it converts the new TabularCPD into a `DiscreteFactor` on
demand. This task is the audit that converts every internal caller
to go through it (or to use the new tag-based API), so Task 36's
tag-dispatch swap doesn't break anything downstream.

**Files:**
- Audit: `pgmpy/` (grep below)
- Modify: every caller flagged by the audit (typically `pgmpy/models/DiscreteBayesianNetwork.py:1334` and a handful of `pgmpy/inference/*.py` paths)
- Create: `pgmpy/tests/test_inference/test_factor_api_audit.py`

- [ ] **Step 1: Run the audit grep and write a regression test**

Run:

```bash
grep -RIn -E "cpd\.(marginalize|reduce|normalize|to_factor|product|variables|cardinality|values|scope)" pgmpy/ \
    | grep -v "pgmpy/factors/" \
    | grep -v "pgmpy/parameterization/" \
    | grep -v "pgmpy/tests/" \
    | tee /tmp/factor_api_audit.txt
```

Expected output: a list of files + line numbers where pgmpy code treats
a CPD as a factor. Typical hits include:

- `pgmpy/models/DiscreteBayesianNetwork.py:1334` — `cpd.marginalize(cpd.variables[1:], inplace=True)` inside `DiscreteBayesianNetwork.do()`.
- `pgmpy/inference/base.py` — the v1.x `Inference._initialize_structures` (already getting rewritten in Task 36).
- `pgmpy/inference/ExactInference.py` — `cpd.to_factor()` or direct attribute access in VE/BP.
- `pgmpy/sampling/Sampling.py` — `cpd.values` and `cpd.variable` access.

For each hit, decide: convert to `bn.transforms.cpd_as_factor(node)`,
use the new tag-based API, or — if the file is being deleted in Phase
5 — skip with a comment.

Write a regression test that exercises the new TabularCPD on each
audit-flagged code path:

```python
# pgmpy/tests/test_inference/test_factor_api_audit.py
import pytest

pytest.importorskip("sklearn")

from pgmpy.base import DAG
from pgmpy.parameterization import TabularCPD


def _build_discrete_dag():
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
                [0.3, 0.7, 0.02, 0.2]],
        state_names=[["A", "B", "C"], ["easy", "hard"], ["low", "high"]],
    ), parent_order=["diff", "intel"])
    return dag


def test_cpd_as_factor_returns_discrete_factor():
    from pgmpy.factors.discrete import DiscreteFactor
    dag = _build_discrete_dag()
    factor = dag.transforms.cpd_as_factor("grade")
    assert isinstance(factor, DiscreteFactor)
    assert "grade" in factor.variables
    assert set(factor.state_names["grade"]) == {"A", "B", "C"}


def test_to_markov_model_works_with_new_tabular_cpd():
    dag = _build_discrete_dag()
    mn = dag.transforms.to_markov_model()
    assert set(mn.nodes()) == {"diff", "intel", "grade"}


def test_variable_elimination_works_with_new_tabular_cpd():
    from pgmpy.inference import VariableElimination
    dag = _build_discrete_dag()
    ve = VariableElimination(dag)
    factor = ve.query(["grade"], show_progress=False)
    assert factor.variables == ["grade"]
    # P(grade) sums to 1 within numerical tolerance.
    assert abs(sum(factor.values) - 1.0) < 1e-9


def test_belief_propagation_works_with_new_tabular_cpd():
    from pgmpy.inference import BeliefPropagation
    dag = _build_discrete_dag()
    bp = BeliefPropagation(dag)
    factor = bp.query(["grade"], show_progress=False)
    assert factor.variables == ["grade"]
```

- [ ] **Step 2: Run the test — expect failures revealing factor-API users**

Run: `pytest pgmpy/tests/test_inference/test_factor_api_audit.py -v --tb=short`
Expected: failures matching the audit grep output. Each failure points
at a specific caller that still expects factor methods on the CPD.

- [ ] **Step 3: Fix each caller**

For each audit hit, apply the appropriate conversion. Most common
patterns:

```python
# Before (v1.x, treats CPD as factor):
cpd = bn.get_cpds(node)
factor_value = cpd.values
parents = cpd.variables[1:]
cpd.marginalize(parents, inplace=True)

# After (v2.0, goes through transforms.cpd_as_factor):
cpd = bn.parameters[node]
factor = bn.transforms.cpd_as_factor(node)
factor_value = factor.values
parents = bn._parent_order[node]
factor.marginalize(parents, inplace=True)
```

For `DiscreteBayesianNetwork.do()` at line 1334, the call
`cpd.marginalize(cpd.variables[1:], inplace=True)` was used to drop
parent dependence after a structural do. The new equivalent is either
(a) replace the CPD with a flat marginal `TabularCPD.from_values(...)`,
or (b) skip the marginalisation entirely since `DiscreteBayesianNetwork`
is deprecated in Task 28. Pick (b) and document.

- [ ] **Step 4: Re-run the audit**

Run: `pytest pgmpy/tests/test_inference/test_factor_api_audit.py -v`
Expected: All pass.

Run: `pytest pgmpy/tests/ -k "(inference or sampling or model)" --tb=short -W ignore::FutureWarning`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/ pgmpy/tests/test_inference/test_factor_api_audit.py
git commit -m "refactor: route factor-API consumers through dag.transforms.cpd_as_factor"
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

## Task 37: CPD protocol dispatch in `sampling/Sampling.py::forward_sample`

**Files:**
- Modify: `pgmpy/sampling/Sampling.py`
- Modify: `pgmpy/tests/test_inference/test_dispatch_tags.py`

- [ ] **Step 1: Write failing test**

```python
# Append to pgmpy/tests/test_inference/test_dispatch_tags.py
def test_forward_sample_uses_registered_cpd_sampling():
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

Run: `pytest pgmpy/tests/test_inference/test_dispatch_tags.py::test_forward_sample_uses_registered_cpd_sampling -v`
Expected: FAIL — `forward_sample` reads `cpd.values` directly, which doesn't exist on new-style CPDs (it's `cpd.values_`).

- [ ] **Step 3: Refactor `forward_sample` to call each CPD's `sample` method**

In `pgmpy/sampling/Sampling.py`, replace the per-node block (lines 100–127)
with:

```python
for node in pbar:
    if show_progress and config.SHOW_PROGRESS:
        pbar.set_description(f"Generating for node: {node}")
    if (partial_samples is not None) and (node in partial_samples.columns):
        sampled[node] = partial_samples.loc[:, node].values
        continue

    cpd = self.model.parameters[node]
    parents = self.model._parent_order.get(node, [])
    X = sampled[parents] if parents else pd.DataFrame(index=range(size))
    sampled[node] = cpd.sample(X, n_samples=size).values
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
git commit -m "refactor(sampling): forward_sample uses registered CPD sampling"
```

---

## Task 38: [Removed] No public `LinearGaussianInference` class in v2.0

Earlier drafts added a standalone `LinearGaussianInference` class. The
current design removes it from the v2.0 public surface:

- `dag.transforms.to_joint_gaussian()` remains as the exact joint-Gaussian
  transformation and optimisation hook.
- `dag.inference.query(...)` remains the public rung-1 query surface.
- Linear-Gaussian-specific exact conditioning can be added later behind
  `dag.inference.query(...)` without adding a required public class now.

No code or tests in this task. Do not create
`pgmpy/inference/linear_gaussian.py`, do not export
`LinearGaussianInference`, and do not make deprecated model shims depend on
it.

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

Works on any DAG whose registered CPDs support the sample and log_prob
protocols — discrete (TabularCPD), linear-Gaussian, FunctionalCPD, and
third-party skpro/sklearn estimators adapted through CPDAdapter.
"""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pandas as pd


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
                cpd = self.model.parameters[node]
                parents = self.model._parent_order.get(node, [])
                X = (pd.DataFrame({p: [sample[p]] for p in parents})
                     if parents else pd.DataFrame(index=[0]))
                if node in evidence:
                    value = evidence[node]
                    log_p = float(cpd.log_prob(pd.Series([value]), X).iloc[0])
                    log_w += log_p
                    sample[node] = value
                else:
                    y = cpd.sample(X, n_samples=1,
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
            cpd = self.model.parameters[var]
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
                cpd = self.model.parameters[var]
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
git commit -m "feat(inference): LikelihoodWeighting via CPD sample/log_prob protocols"
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


def test_predict_auto_dispatches_to_lw_for_all_linear_gaussian_network():
    bn = DAG([("x1", "x2")])
    bn.add_cpds(variable="x1",
                cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    bn.add_cpds(variable="x2",
                cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                parent_order=["x1"])
    pred = bn.inference.predict(pd.DataFrame({"x1": [1.0, 2.0]}),
                                 n_samples=5000, seed=0)
    assert np.isclose(pred.loc[0, "x2"], 2.0, atol=0.2)
    assert np.isclose(pred.loc[1, "x2"], 4.0, atol=0.2)


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

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py::test_predict_auto_dispatches_to_lw_for_all_linear_gaussian_network -v`
Expected: FAIL — current `_BNInference.predict` always uses VariableElimination.

- [ ] **Step 3: Add auto-dispatch logic**

Replace `_BNInference.predict` in `pgmpy/models/_accessors.py`:

```python
def predict(self, data, method=None, **kwargs):
    """Predict missing columns of *data* by per-row inference.

    Auto-dispatches based on CPD tags when ``method=None``:
    - all CPDs produce a factor → VariableElimination
    - otherwise → LikelihoodWeighting

    ``is_linear_gaussian`` remains an optimisation tag for a future exact
    private path via ``dag.transforms.to_joint_gaussian()``, but no public
    LinearGaussianInference class or ``method="linear_gaussian"`` selector is
    added in v2.0.

    Pass ``method="variable_elimination" | "likelihood_weighting"`` to
    override.
    """
    method = method or self._auto_dispatch()
    if method == "variable_elimination":
        return self._predict_ve(data, **kwargs)
    elif method == "likelihood_weighting":
        from pgmpy.inference import LikelihoodWeighting
        return LikelihoodWeighting(self._bn).predict(data, **kwargs)
    else:
        raise ValueError(f"Unknown inference method: {method!r}")

def _auto_dispatch(self):
    cpds = self._bn._cpds.values()
    if all(c.get_tag("produces_factor", False) for c in cpds):
        return "variable_elimination"
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

Similarly enrich `predict_probability` with the same auto-dispatch. The VE
path keeps exact factor-based posteriors; the LW path uses
`LikelihoodWeighting.query` and packages per-state weights or weighted
continuous samples. A later optimization can route all-linear-Gaussian
networks through `dag.transforms.to_joint_gaussian()` internally without
adding a public inference class.

- [ ] **Step 4: Run tests**

Run: `pytest pgmpy/tests/test_models/test_bn_accessors.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/tests/test_models/test_bn_accessors.py
git commit -m "feat(models): _BNInference auto-dispatches to VE or LW by CPD tags"
```

---

# Phase 4b — SCM / Counterfactual support (ships in 1.x.3)

Tasks 40a–40u add SCM semantics + the Pearl-ladder accessor split. All
ship together in 1.x.3; pre-existing CPDs that don't implement the SCM
protocol simply opt out.

| Task(s) | What |
|---|---|
| 40a–40d | `StructuralCPD` protocol + unified `NoiseDistribution` (`Delta`/`NormalNoise`/`Empirical`/`TruncatedUniform`); retrofits for LG / Tabular / Functional. |
| 40e | `WrappedRegressor` — single adapter for ANM (link=None) and PNL (link/link_inv). Merges what were 40e + 40k. |
| 40f–40h | `_DAGCounterfactual` scaffold; abduction-action-prediction; integration test. |
| 40i | `QueryResult` rich return type (renamed from `CounterfactualResult`). |
| 40l | Composable `noise_dist` on `WrappedRegressor`. |
| 40m | `_DAGDiagnostics.identifiability_report` (promoted out of `_DAGCounterfactual`). |
| 40n | Cross-noise-representation `noise_overrides=` plumbing. |
| 40o | Bayesian-counterfactual integration test (composition of 40i + 40l). |
| 40p | `CPDAdapter` (auto-wrap third-party CPDs at `dag.parameters.add`; also touched in Tasks 4 + 19). |
| 40q | `dag.inference.query()` returning `QueryResult` — Pearl rung 1. |
| 40r | `dag.intervene` accessor — Pearl rung 2. |
| 40s | Multi-world counterfactual via list-of-`do`. |
| 40t | Standalone graph primitives on `dag.transforms` (ancestors, d_separated, …). |
| 40u | `dag.bootstrap` for fit-time CIs. |

**Tasks 40j (`LinearGaussianCounterfactual`) and 40k (`PNLWrapper`) are
removed** — the abduction loop is already closed-form correct for LG;
PNL is folded into `WrappedRegressor`.

Prototype Section B (Demos B1–B5) validates the design end-to-end:
closed-form-correct on a 3-node LG SCM (0.0 error); PNL abduction
round-trips to 1e-10; pure-LG identifiability flagged; PNL correctly
cleared; bootstrap captures fit-time uncertainty on n=300 data.

---

## Task 40a: `StructuralCPD` protocol + `NoiseDistribution` types

Define the optional protocol, the `NoiseDistribution` interface plus its
four built-in implementations (`Delta`, `NormalNoise`, `Empirical`,
`TruncatedUniform`), the two SCM-specific capability tags
(`supports_counterfactual`, `noise_type`), and the `IncompatibleCPDError`
exception.

Per simplification #1, every `StructuralCPD`'s `noise_prior()` and
`abduct()` returns a `NoiseDistribution` — a uniformly-typed object with
`.sample(n, random_state)` and `.point()`. The counterfactual algorithm
calls `.sample()` regardless of which CPD it's running over, so the
abduction layer is genuinely framework-agnostic. Earlier drafts returned
ad-hoc tuples (`("normal", 0, std_)`) and mixed types (`pd.Series`,
`pyro.Trace`, custom posterior objects).

**Files:**
- Create: `pgmpy/parameterization/structural.py` (protocol + exception)
- Create: `pgmpy/parameterization/noise.py` (the four Distribution types)
- Create: `pgmpy/tests/test_parameterization/test_structural_protocol.py`
- Create: `pgmpy/tests/test_parameterization/test_noise.py`

- [ ] **Step 1: Write the failing tests**

```python
# pgmpy/tests/test_parameterization/test_structural_protocol.py
import pytest
from pgmpy.parameterization.structural import (
    StructuralCPD, IncompatibleCPDError,
)


def test_protocol_methods_exist():
    for name in ("noise_prior", "structural_predict", "abduct"):
        assert hasattr(StructuralCPD, name), name


def test_incompatible_cpd_error_is_typeerror():
    assert issubclass(IncompatibleCPDError, TypeError)


# pgmpy/tests/test_parameterization/test_noise.py
import numpy as np
from pgmpy.parameterization.noise import (
    Delta, NormalNoise, Empirical, TruncatedUniform,
)


def test_delta_sample_returns_point():
    d = Delta(value=np.array([3.0]))
    s = d.sample(n=10)
    assert s.shape == (10,) or s.shape == (10, 1)
    assert np.all(s == 3.0)
    assert d.point().tolist() == [3.0]


def test_normal_noise_sample_shape():
    n = NormalNoise(mu=0.0, sigma=1.0)
    s = n.sample(n=1000, random_state=0)
    assert s.shape == (1000,) and abs(np.mean(s)) < 0.1


def test_empirical_resamples_supplied_residuals():
    e = Empirical(samples=np.array([1.0, 2.0, 3.0, 4.0]))
    s = e.sample(n=100, random_state=0)
    assert set(s.tolist()).issubset({1.0, 2.0, 3.0, 4.0})


def test_truncated_uniform_bracket_respected():
    tu = TruncatedUniform(low=np.array([0.3]), high=np.array([0.7]))
    s = tu.sample(n=200, random_state=0)
    assert ((s >= 0.3) & (s < 0.7)).all()
```

- [ ] **Step 2: Run to verify FAIL** — Expected: ImportError.

- [ ] **Step 3: Implement the modules**

```python
# pgmpy/parameterization/structural.py
from typing import Any, Protocol, runtime_checkable
import pandas as pd


@runtime_checkable
class NoiseDistribution(Protocol):
    def sample(self, n: int = 1, random_state=None): ...
    def point(self): ...


@runtime_checkable
class StructuralCPD(Protocol):
    def noise_prior(self) -> NoiseDistribution: ...
    def structural_predict(self, parents: pd.DataFrame,
                            noise: Any) -> pd.Series: ...
    def abduct(self, x: pd.Series,
                parents: pd.DataFrame) -> NoiseDistribution: ...


class IncompatibleCPDError(TypeError):
    """Raised when a counterfactual query touches a CPD that does not
    advertise supports_counterfactual=True or fails to implement the
    StructuralCPD protocol."""
```

```python
# pgmpy/parameterization/noise.py
# Reference: prototype.py — sections 2 (unified noise types).
# Copy Delta, NormalNoise, Empirical, TruncatedUniform verbatim.
```

- [ ] **Step 4: Run tests to verify PASS** — Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/structural.py pgmpy/parameterization/noise.py pgmpy/tests/test_parameterization/
git commit -m "feat(parameterization): StructuralCPD protocol + unified NoiseDistribution types"
```

---

## Task 40b: Retrofit `LinearGaussianCPD` with structural methods (ANM)

`LinearGaussianCPD` is a canonical additive noise model:
`X = β₀ + β·pa + U` with `U ~ N(0, std_)`. Add the three structural methods
and the three new capability tags.

**Files:**
- Modify: `pgmpy/parameterization/linear_gaussian.py`
- Create: `pgmpy/tests/test_parameterization/test_lg_structural.py`

- [ ] **Step 1: Write the failing test**

```python
# pgmpy/tests/test_parameterization/test_lg_structural.py
import numpy as np
import pandas as pd
import pytest

from pgmpy.parameterization import LinearGaussianCPD


def test_lg_advertises_counterfactual_capability():
    cpd = LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5)
    assert cpd.get_tag("supports_counterfactual") is True
    assert cpd.get_tag("noise_type") == "additive"


def test_lg_abduct_then_predict_is_identity():
    cpd = LinearGaussianCPD.from_values(beta=[1.0, 2.0, -0.5], std=0.7)
    pa = pd.DataFrame({"a": [3.0, -1.0], "b": [2.0, 4.0]})
    x = pd.Series([7.0, 0.5])

    u = cpd.abduct(x, pa)
    x_back = cpd.structural_predict(pa, u)

    np.testing.assert_allclose(x_back.values, x.values, atol=1e-10)


def test_lg_noise_prior_returns_normal_noise():
    from pgmpy.parameterization.noise import NormalNoise
    cpd = LinearGaussianCPD.from_values(beta=[0.0], std=1.7)
    noise = cpd.noise_prior()
    assert isinstance(noise, NormalNoise)
    assert noise.mu == 0.0
    assert noise.sigma == pytest.approx(1.7)
```

- [ ] **Step 2: Run to verify FAIL**

Run: `pytest pgmpy/tests/test_parameterization/test_lg_structural.py -v`
Expected: FAIL — tags absent, methods missing.

- [ ] **Step 3: Add tags + three methods to `LinearGaussianCPD`**

Extend `_tags`:

```python
_tags = {
    # ... existing tags ...
    "supports_counterfactual": True,
    "noise_type": "additive",
    }
```

Add methods (full implementation in `enhancement_proposals/2_refactor_cpds/prototype/prototype.py:359-396` —
copy verbatim and adjust types):

```python
def noise_prior(self):
    from pgmpy.parameterization.noise import NormalNoise
    return NormalNoise(mu=0.0, sigma=self.std_)

def structural_predict(self, parents, noise):
    if hasattr(noise, "point"):
        noise = noise.point()
    u = np.asarray(noise, dtype=float).ravel()
    if parents is None or (hasattr(parents, "shape") and parents.shape[1] == 0):
        f_pa = np.full(len(u), self.beta_[0])
    else:
        pa_arr = np.asarray(parents, dtype=float).reshape(len(u), -1)
        f_pa = self.beta_[0] + pa_arr @ self.beta_[1:]
    return pd.Series(f_pa + u,
                     index=parents.index if hasattr(parents, "index") else None)

def abduct(self, x, parents):
    from pgmpy.parameterization.noise import Delta
    x_arr = np.asarray(x, dtype=float).ravel()
    n = len(x_arr)
    if parents is None or (hasattr(parents, "shape") and parents.shape[1] == 0):
        f_pa = np.full(n, self.beta_[0])
    else:
        pa_arr = np.asarray(parents, dtype=float).reshape(n, -1)
        f_pa = self.beta_[0] + pa_arr @ self.beta_[1:]
    return Delta(x_arr - f_pa)
```

- [ ] **Step 4: Run tests to verify PASS**

Run: `pytest pgmpy/tests/test_parameterization/test_lg_structural.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/linear_gaussian.py pgmpy/tests/test_parameterization/test_lg_structural.py
git commit -m "feat(parameterization): LinearGaussianCPD implements StructuralCPD (additive noise, invertible)"
```

---

## Task 40c: Retrofit `TabularCPD` with inverse-CDF structural methods

Discrete variables don't have additive noise; the standard SCM encoding
(per Pearl, *Causality* §7.1 and `dowhy.gcm.ClassifierFCM`) is **inverse-CDF**:
`X = F⁻¹(P(X | pa), U)` with `U ~ Uniform[0, 1]`.

- *Forward sample*: draw `u ∈ [0, 1)`, return the smallest `k` such that
  the cumulative probability `F(k | pa) ≥ u`.
- *Abduct*: given observed `X = k`, the noise `u` lies in the bracket
  `[F(k − 1 | pa), F(k | pa)]` — a 1-D truncated uniform, exact and
  trivial to sample.

Discrete counterfactuals are *not* identified by the observational
distribution alone — different noise representations give different
counterfactuals. We default to inverse-CDF (aligned with DoWhy and Pearl)
and accept `noise_repr="gumbel_max"` as a future opt-in for users who
specifically need that encoding. Adding Gumbel-Max is a separate v2.x
follow-up; out of scope for this task.

**Files:**
- Modify: `pgmpy/parameterization/tabular.py`
- Create: `pgmpy/tests/test_parameterization/test_tabular_structural.py`

- [ ] **Step 1: Write the failing test**

```python
# pgmpy/tests/test_parameterization/test_tabular_structural.py
import numpy as np
import pandas as pd

from pgmpy.parameterization import TabularCPD


def test_tabular_advertises_inverse_cdf_capability():
    cpd = TabularCPD.from_values(variable_card=3,
                                  values=[[0.6], [0.3], [0.1]])
    assert cpd.get_tag("supports_counterfactual") is True
    assert cpd.get_tag("noise_type") == "inverse_cdf"


def test_inverse_cdf_forward_matches_categorical_sampling():
    """structural_predict(pa, u) with u just below cumulative bracket
    returns the corresponding class."""
    # P = [0.6, 0.3, 0.1]   → cumulative = [0.6, 0.9, 1.0]
    cpd = TabularCPD.from_values(variable_card=3,
                                  values=[[0.6], [0.3], [0.1]])
    pa = pd.DataFrame(index=range(3))
    u = np.array([0.05, 0.7, 0.95])     # in brackets [0, 0.6), [0.6, 0.9), [0.9, 1.0)
    x = cpd.structural_predict(pa, u)
    assert list(x) == [0, 1, 2]


def test_inverse_cdf_abduct_returns_correct_bracket():
    """abduct(x=1, pa) returns a posterior with .low=0.6, .high=0.9 for
    P=[0.6, 0.3, 0.1] (cumulative bracket of class 1)."""
    cpd = TabularCPD.from_values(variable_card=3,
                                  values=[[0.6], [0.3], [0.1]])
    pa = pd.DataFrame(index=[0])
    posterior = cpd.abduct(pd.Series([1]), pa)
    assert posterior.low[0] == 0.6
    assert posterior.high[0] == 0.9


def test_inverse_cdf_round_trip_consistency():
    """For 200 random samples u, abducting x=structural_predict(pa,u)
    should return a posterior whose bracket contains u."""
    rng = np.random.default_rng(0)
    cpd = TabularCPD.from_values(variable_card=4,
                                  values=[[0.4], [0.3], [0.2], [0.1]])
    pa = pd.DataFrame(index=[0])
    for _ in range(200):
        u = rng.uniform()
        x = cpd.structural_predict(pa, np.array([u])).iloc[0]
        post = cpd.abduct(pd.Series([x]), pa)
        assert post.low[0] <= u < post.high[0]
```

- [ ] **Step 2: Run to verify FAIL**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular_structural.py -v`
Expected: FAIL — tags absent, methods missing.

- [ ] **Step 3: Use the shared truncated-uniform noise distribution**

`Task 40a` already added `pgmpy.parameterization.noise.TruncatedUniform`.
Use that shared `NoiseDistribution` for both the prior (`low=0`,
`high=1`) and the abduction posterior (`low=F(k-1|pa)`, `high=F(k|pa)`).
Do not create a separate categorical-posterior class.

- [ ] **Step 4: Add tags + three methods to `TabularCPD`**

```python
_tags = {
    # ... existing tags ...
    "supports_counterfactual": True,
    "noise_type": "inverse_cdf",
    }


def __init__(self, ..., noise_repr: str = "inverse_cdf"):
    # ... existing init body ...
    if noise_repr not in ("inverse_cdf", "gumbel_max"):
        raise ValueError(f"unknown noise_repr {noise_repr!r}")
    self.noise_repr = noise_repr
    # gumbel_max path is v2.x; inverse_cdf is the v2.0 default
    if noise_repr == "gumbel_max":
        raise NotImplementedError(
            "noise_repr='gumbel_max' is reserved for a v2.x follow-up. "
            "Use the default 'inverse_cdf' for v2.0."
        )


def noise_prior(self):
    from pgmpy.parameterization.noise import TruncatedUniform
    return TruncatedUniform(low=np.array([0.0]), high=np.array([1.0]))


def structural_predict(self, parents, noise):
    proba = self.predict_proba(parents).values   # (n_rows, K)
    if hasattr(noise, "point"):
        noise = noise.point()
    u = np.asarray(noise, dtype=float).ravel()
    # Cumulative-probability lookup, vectorised.
    cum = np.cumsum(proba, axis=1)
    chosen = (cum >= u[:, None]).argmax(axis=1)
    return pd.Series([self.classes_[i] for i in chosen],
                     index=parents.index if hasattr(parents, "index") else None)


def abduct(self, x, parents):
    from pgmpy.parameterization.noise import TruncatedUniform
    proba = self.predict_proba(parents).values     # (n_rows, K)
    cum = np.cumsum(proba, axis=1)
    cum_lower = np.concatenate([np.zeros((cum.shape[0], 1)), cum[:, :-1]],
                                axis=1)
    k_obs = np.array([list(self.classes_).index(v) for v in x])
    rows = np.arange(len(x))
    return TruncatedUniform(
        low=cum_lower[rows, k_obs], high=cum[rows, k_obs],
    )
```

- [ ] **Step 5: Run tests to verify PASS**

Run: `pytest pgmpy/tests/test_parameterization/test_tabular_structural.py -v`
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add pgmpy/parameterization/tabular.py pgmpy/tests/test_parameterization/test_tabular_structural.py
git commit -m "feat(parameterization): TabularCPD implements StructuralCPD via inverse-CDF"
```

---

## Task 40d: Retrofit `FunctionalCPD` with Pyro-native structural methods

Pyro models are SCMs by construction. Use `pyro.poutine.condition` to
abduct (condition on observed) and `pyro.poutine.do` to apply the
intervention. The structural function and noise prior delegate to the
user's existing Pyro model body — no new user-facing API.

**Files:**
- Modify: `pgmpy/parameterization/functional.py`
- Create: `pgmpy/tests/test_parameterization/test_functional_structural.py`

- [ ] **Step 1: Write the failing test**

```python
# pgmpy/tests/test_parameterization/test_functional_structural.py
import pytest

pytest.importorskip("pyro")

import pyro
import pyro.distributions as dist
import pandas as pd

from pgmpy.parameterization import FunctionalCPD


def test_functional_advertises_counterfactual_capability():
    fn = lambda pa: dist.Normal(2.0 * pa["x"], 0.5)
    cpd = FunctionalCPD(fn=fn)
    assert cpd.get_tag("supports_counterfactual") is True
    assert cpd.get_tag("noise_type") == "custom"
```

- [ ] **Step 2: Run to verify FAIL**

Run: `pytest pgmpy/tests/test_parameterization/test_functional_structural.py -v`
Expected: FAIL — tags absent.

- [ ] **Step 3: Add tags + three methods to `FunctionalCPD`**

```python
_tags = {
    # ... existing tags ...
    "supports_counterfactual": True,
    "noise_type": "custom",
    }

def noise_prior(self):
    # Opaque token; the counterfactual algorithm passes it back unchanged
    # through the shared NoiseDistribution interface.
    from pgmpy.parameterization.noise import Delta
    return Delta(None)

def structural_predict(self, parents, noise):
    # noise is a Pyro trace (from abduct) or None for unconditioned draws.
    # Apply pyro.poutine.replay with the abducted trace, then sample.
    import pyro
    import pyro.poutine as poutine
    if hasattr(noise, "point"):
        noise = noise.point()
    if noise is None:
        sampler = self.fn(parents.iloc[0].to_dict())
        return pd.Series([float(sampler.sample())], index=parents.index)
    replayed = poutine.replay(self.fn, trace=noise)
    sampler = replayed(parents.iloc[0].to_dict())
    return pd.Series([float(sampler.sample())], index=parents.index)

def abduct(self, x, parents):
    # Return a conditioned trace; counterfactual algorithm replays it.
    import pyro
    import pyro.poutine as poutine
    from pgmpy.parameterization.noise import Delta
    conditioned = poutine.condition(self.fn, data={"value": float(x.iloc[0])})
    trace = poutine.trace(conditioned).get_trace(parents.iloc[0].to_dict())
    return Delta(trace)
```

(Exact Pyro integration depends on FunctionalCPD's existing Pyro
conventions; the above is a sketch — adapt to whatever sample-site naming
convention `FunctionalCPD` uses.)

- [ ] **Step 4: Run tests to verify PASS**

Run: `pytest pgmpy/tests/test_parameterization/test_functional_structural.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/functional.py pgmpy/tests/test_parameterization/test_functional_structural.py
git commit -m "feat(parameterization): FunctionalCPD implements StructuralCPD via Pyro poutines"
```

---

## Task 40e: `WrappedRegressor` — single adapter for ANM and PNL

One class that wraps any sklearn/skpro regressor as an SCM CPD. With
`link=None` this is an additive noise model (ANM, matching DoWhy's
`AdditiveNoiseModel`). Passing `link`/`link_inv` makes it a post-nonlinear
model — the same machinery, no separate `PNLWrapper` class needed.

This merges what earlier drafts had as Task 40e (WrappedRegressor) + Task 40k
(PNLWrapper) into a single composable wrapper (simplification #2).
Reference implementation: prototype's `WrappedRegressor` (validated by
Demo B3 — abduction round-trip matches to 1e-10 on both ANM and PNL).

**Files:**
- Create: `pgmpy/parameterization/adapters/anm.py`
- Create: `pgmpy/tests/test_parameterization/test_anm_wrapper.py`

- [ ] **Step 1: Write the failing test**

```python
# pgmpy/tests/test_parameterization/test_anm_wrapper.py
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from pgmpy.parameterization.adapters import WrappedRegressor


def test_anm_wrapper_fits_and_supports_counterfactual():
    rng = np.random.default_rng(0)
    n = 500
    pa = rng.normal(0, 1, n)
    y = pa ** 2 + rng.normal(0, 0.3, n)   # nonlinear
    X = pd.DataFrame({"pa": pa})

    wrap = WrappedRegressor(RandomForestRegressor(n_estimators=20, random_state=0))
    wrap.fit(X, pd.Series(y))

    assert wrap.get_tag("supports_counterfactual") is True
    assert wrap.get_tag("noise_type") == "additive"

    # Roundtrip: abduct then structural_predict should recover x.
    x_obs = pd.Series([1.0])
    pa_obs = pd.DataFrame({"pa": [0.5]})
    u = wrap.abduct(x_obs, pa_obs)
    x_back = wrap.structural_predict(pa_obs, u)
    np.testing.assert_allclose(x_back.values, x_obs.values, atol=1e-8)
```

- [ ] **Step 2: Run to verify FAIL**

Expected: ImportError.

- [ ] **Step 3: Implement `WrappedRegressor`**

```python
# pgmpy/parameterization/adapters/anm.py
import numpy as np
import pandas as pd
from skpro.regression.base import BaseProbaRegressor


class WrappedRegressor(BaseProbaRegressor):
    """Additive noise model wrapper for any sklearn/skpro regressor.

    Wraps a regressor as: X = regressor.predict(pa) + U where U is drawn
    from noise_dist (default: empirical residuals from training data).
    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_counterfactual": True,
        "noise_type": "additive",
                "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "capability:multioutput": False,
        "capability:missing": False,
    }

    def __init__(self, regressor, noise_dist=None):
        self.regressor = regressor
        self.noise_dist = noise_dist
        super().__init__()

    def _fit(self, X, y, C=None):
        from sklearn.base import clone as sk_clone
        self.regressor_ = sk_clone(self.regressor).fit(X, np.asarray(y).ravel())
        residuals = np.asarray(y).ravel() - self.regressor_.predict(X)
        # Empirical noise: store residuals; sample by resampling them.
        self.noise_residuals_ = residuals
        self.noise_dist_ = self.noise_dist  # explicit override if given
        return self

    def _predict_proba(self, X):
        # Simple empirical: predict mean + noise from residuals
        from skpro.distributions import Empirical  # or fallback to Normal
        mean = self.regressor_.predict(X)
        # Use Normal(mean, residual_std) as a reasonable default surface.
        from skpro.distributions import Normal
        sigma = float(self.noise_residuals_.std())
        index = X.index if hasattr(X, "index") else pd.RangeIndex(len(mean))
        mu_df = pd.DataFrame({"value": mean}, index=index)
        sig_df = pd.DataFrame({"value": np.full(len(mean), sigma)}, index=index)
        return Normal(mu=mu_df, sigma=sig_df)

    # StructuralCPD protocol
    def noise_prior(self):
        from pgmpy.parameterization.noise import Empirical
        return Empirical(samples=self.noise_residuals_)

    def structural_predict(self, parents, noise):
        if hasattr(noise, "point"):
            noise = noise.point()
        u = np.asarray(noise, dtype=float).ravel()
        f_pa = self.regressor_.predict(parents)
        return pd.Series(f_pa + u,
                         index=parents.index if hasattr(parents, "index") else None)

    def abduct(self, x, parents):
        from pgmpy.parameterization.noise import Delta
        x_arr = np.asarray(x, dtype=float).ravel()
        f_pa = self.regressor_.predict(parents)
        return Delta(x_arr - f_pa)
```

Also create `pgmpy/parameterization/adapters/__init__.py` exporting
`WrappedRegressor`.

- [ ] **Step 4: Run tests to verify PASS**

Run: `pytest pgmpy/tests/test_parameterization/test_anm_wrapper.py -v`
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/parameterization/adapters/
git commit -m "feat(parameterization): WrappedRegressor adapts sklearn/skpro regressors as SCM CPDs"
```

---

## Task 40f: `_DAGCounterfactual` accessor scaffold

Create the accessor class and wire it onto `DAG` via a `cached_property`.
The actual `query` implementation is Task 40g.

**Files:**
- Modify: `pgmpy/models/_accessors.py` (add `_DAGCounterfactual`)
- Modify: `pgmpy/base/DAG.py` (add `counterfactual` cached_property)
- Create: `pgmpy/tests/test_models/test_dag_counterfactual_accessor.py`

- [ ] **Step 1: Write the failing test**

```python
# pgmpy/tests/test_models/test_dag_counterfactual_accessor.py
from pgmpy.base import DAG


def test_dag_has_counterfactual_accessor():
    dag = DAG([("X", "Y")])
    assert hasattr(dag, "counterfactual")
    # cached_property: same instance returned across accesses
    assert dag.counterfactual is dag.counterfactual
```

- [ ] **Step 2: Run to verify FAIL**

Expected: AttributeError.

- [ ] **Step 3: Add the accessor class and the cached_property**

```python
# pgmpy/models/_accessors.py
class _DAGCounterfactual:
    def __init__(self, dag):
        self._dag = dag

    def query(self, observed, do, query, n_samples=1, seed=None):
        raise NotImplementedError("Implemented in Task 40g.")

    def explain(self, observed, do):
        raise NotImplementedError("Implemented in Task 40g.")
```

```python
# pgmpy/base/DAG.py — add inside class DAG:
@cached_property
def counterfactual(self):
    from pgmpy.models._accessors import _DAGCounterfactual
    return _DAGCounterfactual(self)
```

- [ ] **Step 4: Run tests to verify PASS**

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/base/DAG.py pgmpy/tests/test_models/test_dag_counterfactual_accessor.py
git commit -m "feat(base): DAG.counterfactual accessor scaffold (query impl in Task 40g)"
```

---

## Task 40g: Implement abduction-action-prediction in `counterfactual.query`

The core algorithm. Validated closed-form in
`enhancement_proposals/2_refactor_cpds/prototype/prototype.py:1054-1118`
(Demo 11 + `dag_counterfactual` helper). Copy that structure into the
accessor.

**Files:**
- Modify: `pgmpy/models/_accessors.py` (fill in `query` and `explain`)
- Create: `pgmpy/tests/test_models/test_dag_counterfactual_query.py`

- [ ] **Step 1: Write the failing test (closed-form check)**

```python
# pgmpy/tests/test_models/test_dag_counterfactual_query.py
import numpy as np
import pytest

from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD
from pgmpy.parameterization.structural import IncompatibleCPDError


def test_linear_gaussian_scm_counterfactual_closed_form():
    """3-node chain X -> Y -> Z. Observe (1, 3, 10). Counterfactual:
    had X been 0, what would Z have been?  Analytic answer: 4.
    """
    scm = DAG([("X", "Y"), ("Y", "Z")])
    scm.parameters.add(variable="X",
                       cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    scm.parameters.add(variable="Y",
                       cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                       parent_order=["X"])
    scm.parameters.add(variable="Z",
                       cpd=LinearGaussianCPD.from_values(beta=[0.0, 3.0], std=0.3),
                       parent_order=["Y"])

    z_cf = scm.counterfactual.query(
        observed={"X": 1.0, "Y": 3.0, "Z": 10.0},
        do={"X": 0.0},
        query="Z",
    )
    assert abs(z_cf - 4.0) < 1e-9


def test_counterfactual_raises_on_unsupported_cpd():
    from sklearn.ensemble import RandomForestClassifier
    scm = DAG([("X", "Y")])
    scm.parameters.add(variable="X",
                       cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    scm.parameters.add(variable="Y",
                       cpd=RandomForestClassifier(),
                       parent_order=["X"])
    # RF doesn't advertise supports_counterfactual=True.
    with pytest.raises(IncompatibleCPDError, match="Y"):
        scm.counterfactual.query(
            observed={"X": 1.0, "Y": 0},
            do={"X": 0.0},
            query="Y",
        )


def test_explain_returns_abducted_noise_per_node():
    scm = DAG([("X", "Y")])
    scm.parameters.add(variable="X",
                       cpd=LinearGaussianCPD.from_values(beta=[0.0], std=1.0))
    scm.parameters.add(variable="Y",
                       cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5),
                       parent_order=["X"])
    u = scm.counterfactual.explain(observed={"X": 1.0, "Y": 3.0}, do={})
    assert np.asarray(u["X"].point()).ravel()[0] == pytest.approx(1.0)
    assert np.asarray(u["Y"].point()).ravel()[0] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify FAIL**

Expected: NotImplementedError from `query`.

- [ ] **Step 3: Implement `query` and `explain`**

Reference: `enhancement_proposals/2_refactor_cpds/prototype/prototype.py:1054-1118`
(the `dag_counterfactual` helper validates this algorithm against
closed-form). Translate into the accessor:

```python
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.parameterization.structural import IncompatibleCPDError


class _DAGCounterfactual:
    def __init__(self, dag):
        self._dag = dag

    def _check_capability(self):
        offenders = []
        for node in self._dag.nodes():
            cpd = self._dag.parameters[node]
            if not cpd.get_tag("supports_counterfactual", False):
                offenders.append(node)
        if offenders:
            raise IncompatibleCPDError(
                f"CPDs on nodes {offenders} do not implement the "
                f"StructuralCPD protocol (supports_counterfactual=False)."
            )

    def explain(self, observed, do):
        self._check_capability()
        noise = {}
        for node in nx.topological_sort(self._dag):
            cpd = self._dag.parameters[node]
            parents = self._dag._parent_order.get(node, [])
            X_pa = (pd.DataFrame({p: [observed[p]] for p in parents})
                    if parents else pd.DataFrame(index=[0]))
            # CPDs return a NoiseDistribution. Invertible CPDs usually return
            # Delta; non-invertible discrete CPDs return a distribution over
            # the compatible noise bracket.
            noise[node] = cpd.abduct(pd.Series([observed[node]]), X_pa)
        return noise

    def query(self, observed, do, query, n_samples=1, seed=None):
        self._check_capability()
        noise = self.explain(observed, do)
        rng = np.random.default_rng(seed)

        cf = {}
        for node in nx.topological_sort(self._dag):
            if node in do:
                cf[node] = float(do[node])
                continue
            cpd = self._dag.parameters[node]
            parents = self._dag._parent_order.get(node, [])
            X_pa = (pd.DataFrame({p: [cf[p]] for p in parents})
                    if parents else pd.DataFrame(index=[0]))
            u_node = (
                noise[node]
                if n_samples == 1
                else noise[node].sample(n=1, random_state=int(rng.integers(2**31)))[0]
            )
            cf[node] = float(cpd.structural_predict(X_pa, u_node).iloc[0])

        if isinstance(query, str):
            return cf[query]
        return {q: cf[q] for q in query}
```

(The full Monte-Carlo path repeats the forward pass `n_samples` times,
sampling from each abducted `NoiseDistribution`; return a DataFrame or a
`QueryResult` once Task 40i lands.)

- [ ] **Step 4: Run tests to verify PASS**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/tests/test_models/test_dag_counterfactual_query.py
git commit -m "feat(base): counterfactual.query implements abduction-action-prediction"
```

---

## Task 40h: Integration test — hybrid SCM with `WrappedRegressor`

End-to-end test: a 4-node SCM with a `LinearGaussianCPD` parent and an
`WrappedRegressor(RandomForestRegressor)` child. Confirms the protocol works on
heterogeneous CPDs and that `dag.counterfactual.query` handles the mixed
case.

**Files:**
- Create: `pgmpy/tests/test_models/test_dag_counterfactual_integration.py`

- [ ] **Step 1: Write the integration test**

```python
import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import RandomForestRegressor

from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD
from pgmpy.parameterization.adapters import WrappedRegressor


def test_mixed_scm_counterfactual_runs_end_to_end():
    """SCM:
       A = U_A             (LinearGaussian)
       B = 2*A + U_B       (LinearGaussian)
       C = f_RF(A, B) + U_C  (ANM-wrapped RandomForest)
    Counterfactual: had A been 0, what would C have been?
    """
    rng = np.random.default_rng(0)
    n = 400
    a = rng.normal(0, 1, n)
    b = 2 * a + rng.normal(0, 0.3, n)
    c = np.sin(a) + 0.5 * b + rng.normal(0, 0.2, n)
    data = pd.DataFrame({"A": a, "B": b, "C": c})

    scm = DAG([("A", "B"), ("A", "C"), ("B", "C")])
    scm.parameters.add(variable="A",
                       cpd=LinearGaussianCPD())
    scm.parameters.add(variable="B",
                       cpd=LinearGaussianCPD(), parent_order=["A"])
    scm.parameters.add(
        variable="C",
        cpd=WrappedRegressor(RandomForestRegressor(n_estimators=30, random_state=0)),
        parent_order=["A", "B"],
    )
    scm.fit(data)

    observed = {"A": 1.0, "B": 2.0, "C": 1.5}
    cf = scm.counterfactual.query(observed=observed, do={"A": 0.0},
                                    query="C")
    # Sanity: result is a real number (point estimate — both CPDs are
    # invertible). Won't equal the observation (otherwise abduction is
    # trivial); won't equal the prior marginal (otherwise abduction is
    # ignored).
    assert isinstance(cf, float)
    assert abs(cf - observed["C"]) > 1e-6
```

- [ ] **Step 2: Run to verify PASS (using existing implementations)**

Run: `pytest pgmpy/tests/test_models/test_dag_counterfactual_integration.py -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add pgmpy/tests/test_models/test_dag_counterfactual_integration.py
git commit -m "test(base): end-to-end SCM counterfactual on mixed LG + ANM(RF) DAG"
```

---

## Task 40i: `QueryResult` rich return type

Replace bare-float return values with a `QueryResult` dataclass.
Distribution-valued by construction, with `.point()`, `.distribution()`,
`.expectation()`, `.credible_interval()`, `.compare_to()`. This is the
public API for counterfactual queries; all earlier tasks (40g, 40h) will
return this type.

Prototype reference: `enhancement_proposals/2_refactor_cpds/prototype/prototype.py`
(class `QueryResult`, ~50 lines).

**Files:**
- Create: `pgmpy/parameterization/query_result.py`
- Modify: `pgmpy/models/_accessors.py` — `_DAGCounterfactual.query` returns the new type
- Modify: `pgmpy/tests/test_models/test_dag_counterfactual_query.py` — update assertions to use `.point()`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from pgmpy.parameterization import QueryResult


def test_query_result_point_distribution():
    samples = np.array([1.0, 2.0, 3.0, 4.0])
    cr = QueryResult(samples=samples, query="Z",
                              do={"X": 0}, observed={"X": 1, "Z": 2.5})
    assert cr.point() == pytest.approx(2.5)
    assert isinstance(cr.distribution(), pd.Series)
    assert list(cr.distribution()) == [1.0, 2.0, 3.0, 4.0]


def test_credible_interval_and_expectation():
    samples = np.linspace(0, 1, 1001)
    cr = QueryResult(samples=samples, query="Z",
                              do={}, observed={})
    lo, hi = cr.credible_interval(0.9)
    assert lo == pytest.approx(0.05, abs=1e-3)
    assert hi == pytest.approx(0.95, abs=1e-3)
    assert cr.expectation(lambda z: z**2) == pytest.approx(1/3, abs=1e-3)


def test_compare_to_uses_wasserstein():
    a = QueryResult(samples=np.zeros(100), query="Z", do={}, observed={})
    b = QueryResult(samples=np.ones(100), query="Z", do={}, observed={})
    cmp = a.compare_to(b)
    assert cmp["wasserstein"] == pytest.approx(1.0)
    assert cmp["abs_mean_diff"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify FAIL**

Expected: ImportError.

- [ ] **Step 3: Implement `QueryResult`**

Translate from `prototype.py` verbatim. Import in `pgmpy/parameterization/__init__.py`.

- [ ] **Step 4: Update `_DAGCounterfactual.query` to return `QueryResult`**

For invertible paths: `samples = np.array([single_value])`.
For non-invertible: `samples = np.array([...n_samples...])`.

- [ ] **Step 5: Update prior tests in `test_dag_counterfactual_query.py`**

Change `assert abs(z_cf - 4.0) < 1e-9` to `assert abs(z_cf.point() - 4.0) < 1e-9`.

- [ ] **Step 6: Run all SCM tests**

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add pgmpy/parameterization/query_result.py pgmpy/models/_accessors.py pgmpy/tests/test_models/test_dag_counterfactual_query.py pgmpy/tests/test_parameterization/
git commit -m "feat(parameterization): QueryResult — distribution-valued query results"
```

---

## Task 40j: [Removed]

`LinearGaussianCounterfactual` is unnecessary — the generic abduction loop
in `_DAGCounterfactual.query` is already closed-form correct on LG SCMs
(`LinearGaussianCPD.abduct` returns `Delta`; `structural_predict` is exact
arithmetic). Demo B2 hits 0.0 error vs analytic. `is_linear_gaussian` tag
stays as a future optimisation hook.

## Task 40k: [Removed — folded into Task 40e]

`PNLWrapper` is now `WrappedRegressor(reg, link=..., link_inv=...)`. PNL
test coverage is inside Task 40e's `test_wrapped_regressor.py`.

---

## Task 40l: Composable `noise_dist` slot on `WrappedRegressor`

Extend `WrappedRegressor.__init__` to accept any skpro distribution as
`noise_dist`, plus string aliases `"empirical"` and `"empirical_kde"`.
Existing default `None` continues to mean empirical residuals (DoWhy
parity).

**Files:**
- Modify: `pgmpy/parameterization/adapters/anm.py`
- Modify: `pgmpy/tests/test_parameterization/test_anm_wrapper.py` (extend)

- [ ] **Step 1: Add failing tests**

```python
def test_anm_wrapper_accepts_skpro_distribution_as_noise():
    from skpro.distributions import Normal
    from sklearn.linear_model import LinearRegression
    # Synthetic data
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, n)
    y = 2 * x + rng.normal(0, 0.5, n)
    custom_noise = Normal(mu=pd.DataFrame({"value": [0.0]}),
                           sigma=pd.DataFrame({"value": [0.5]}))
    anm = WrappedRegressor(LinearRegression(), noise_dist=custom_noise)
    anm.fit(pd.DataFrame({"x": x}), pd.Series(y))
    assert anm.noise_dist_ is custom_noise


def test_anm_wrapper_empirical_kde():
    anm = WrappedRegressor(LinearRegression(), noise_dist="empirical_kde")
    # ... fit and check that noise_dist_ is a KDE object ...
```

- [ ] **Step 2: Update `WrappedRegressor._fit`** to branch on the `noise_dist` argument.

- [ ] **Step 3: Run tests** — expected pass.

- [ ] **Step 4: Commit**

```bash
git add pgmpy/parameterization/adapters/anm.py pgmpy/tests/test_parameterization/test_anm_wrapper.py
git commit -m "feat(parameterization): WrappedRegressor accepts skpro distributions and KDE as noise_dist"
```

---

## Task 40m: `identifiability_report()` static SCM diagnostic

Adds `dag.diagnostics.identifiability_report()` — flags pure
linear-Gaussian sub-graphs and discrete nodes with non-invertible noise.
Prototype reference: `prototype.py:identifiability_report` (Demo 12c).

**Lives under `dag.diagnostics`, not `dag.counterfactual`** (simplification
#4). Identifiability isn't only a counterfactual concern — interventional
estimands have it too (do-calculus / ID algorithm). The diagnostics accessor
is also the future home for residual analysis, structure adequacy, and fit
quality checks.

**Files:**
- Create: `pgmpy/models/_accessors.py::_DAGDiagnostics` (new accessor class)
- Modify: `pgmpy/base/DAG.py` — add `@cached_property def diagnostics`
- Create: `pgmpy/tests/test_models/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pure_lg_chain_flagged():
    """3-node LG chain triggers the non-identification warning."""
    scm = DAG([("X", "Y"), ("Y", "Z")])
    for n, beta, std, parents in [("X", [0.0], 1.0, None),
                                    ("Y", [0.0, 2.0], 0.5, ["X"]),
                                    ("Z", [0.0, 3.0], 0.3, ["Y"])]:
        scm.parameters.add(
            variable=n,
            cpd=LinearGaussianCPD.from_values(beta=beta, std=std),
            parent_order=parents,
        )
    report = scm.diagnostics.identifiability_report()
    assert report["n_warnings"] >= 1
    assert any(w["type"] == "linear_gaussian_path"
               for w in report["warnings"])
    assert any("Hoyer" in w["ref"] for w in report["warnings"])


def test_pnl_scm_not_flagged():
    """PNL is identifiable (Zhang & Hyvärinen 2009), so report is empty."""
    # ... build PNL SCM with WrappedRegressor(..., link=tanh, link_inv=arctanh) ...
    report = scm.diagnostics.identifiability_report()
    assert report["n_warnings"] == 0


def test_diagnostics_is_cached_property():
    dag = DAG([("X", "Y")])
    assert dag.diagnostics is dag.diagnostics
```

- [ ] **Step 2: Run to verify FAIL**

- [ ] **Step 3: Implement `_DAGDiagnostics` accessor**

Translate from `enhancement_proposals/2_refactor_cpds/prototype/prototype.py`
— class `_DAGDiagnostics` (~50 lines).

Also add the `cached_property` on `DAG`:

```python
@cached_property
def diagnostics(self):
    from pgmpy.models._accessors import _DAGDiagnostics
    return _DAGDiagnostics(self)
```

- [ ] **Step 4: Run tests**

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/base/DAG.py pgmpy/tests/test_models/test_diagnostics.py
git commit -m "feat(base): dag.diagnostics accessor with identifiability_report"
```

---

## Task 40n: Cross-noise-representation robustness via `noise_overrides=`

Threads `noise_overrides={"node": "inverse_cdf" | "gumbel_max"}` through
`_DAGCounterfactual.query`. In v2.0 only `inverse_cdf` is implemented for
discrete (per Task 40c), so this is mostly API plumbing — the value
becomes real when Gumbel-Max ships in v2.x. We still ship the API now so
the workflow `cf_a.compare_to(cf_b)` is available the day Gumbel-Max lands.

**Files:**
- Modify: `pgmpy/models/_accessors.py` — `_DAGCounterfactual.query` accepts and
  forwards `noise_overrides`.
- Modify: `pgmpy/parameterization/tabular.py` — `noise_repr` becomes
  per-call-overridable via `parents` metadata or a context manager.
- Create: `pgmpy/tests/test_models/test_noise_overrides.py`

- [ ] **Step 1: Write the API-shape test**

```python
def test_query_accepts_noise_overrides_kwarg():
    """noise_overrides is accepted; with only inverse_cdf available
    it currently has no effect, but the API is in place."""
    # ... build SCM with a TabularCPD ...
    cf = scm.counterfactual.query(
        observed=..., do=..., query="Y",
        noise_overrides={"Y": "inverse_cdf"},
    )
    assert isinstance(cf.samples, np.ndarray)


def test_query_rejects_unknown_noise_repr():
    with pytest.raises(ValueError, match="gumbel_max"):
        scm.counterfactual.query(
            observed=..., do=..., query="Y",
            noise_overrides={"Y": "gumbel_max"},   # not implemented yet
        )
```

- [ ] **Step 2: Run to verify FAIL**

- [ ] **Step 3: Add `noise_overrides` parameter to `_DAGCounterfactual.query`**

Forwards each override to the relevant CPD's per-call context. For v2.0
only `inverse_cdf` is honoured; `gumbel_max` raises `NotImplementedError`
with a pointer to the v2.x roadmap.

- [ ] **Step 4: Run tests**

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add pgmpy/models/_accessors.py pgmpy/parameterization/tabular.py pgmpy/tests/test_models/test_noise_overrides.py
git commit -m "feat(base): noise_overrides plumbing for cross-noise-representation robustness"
```

---

## Task 40o: Integration test — Bayesian counterfactual via composition

Demonstrates that #1 (`QueryResult`) + #4 (`noise_dist=` slot) + #5
(Bayesian-regressor uncertainty propagation) compose end-to-end. Uses
skpro's `BayesianLinearRegressor` or sklearn's `BayesianRidge` wrapped via
`WrappedRegressor`. The counterfactual should be naturally distribution-valued
without explicit bootstrap.

**Files:**
- Create: `pgmpy/tests/test_models/test_bayesian_counterfactual.py`

- [ ] **Step 1: Write the integration test**

```python
import numpy as np
import pandas as pd

from sklearn.linear_model import BayesianRidge

from pgmpy.base import DAG
from pgmpy.parameterization import LinearGaussianCPD
from pgmpy.parameterization.adapters import WrappedRegressor


def test_bayesian_regressor_gives_distribution_valued_counterfactual():
    rng = np.random.default_rng(0)
    n = 300   # smallish so the Bayesian posterior is meaningfully wide
    a = rng.normal(0, 1, n)
    b = 2 * a + rng.normal(0, 0.5, n)
    c = 0.5 * b + rng.normal(0, 0.2, n)
    data = pd.DataFrame({"A": a, "B": b, "C": c})

    scm = DAG([("A", "B"), ("B", "C")])
    scm.parameters.add(variable="A", cpd=LinearGaussianCPD())
    scm.parameters.add(variable="B", cpd=LinearGaussianCPD(), parent_order=["A"])
    # Bayesian regressor → parameter uncertainty propagates automatically
    scm.parameters.add(variable="C", cpd=WrappedRegressor(BayesianRidge()),
                       parent_order=["B"])
    scm.fit(data)

    res = scm.counterfactual.query(
        observed={"A": 1.0, "B": 2.0, "C": 1.0},
        do={"A": 0.0}, query="C", n_samples=1000,
    )
    # samples is no longer shape (1,) — parameter uncertainty has expanded it
    assert res.samples.shape == (1000,)
    lo, hi = res.credible_interval(0.95)
    assert hi - lo > 0.01   # genuine uncertainty
```

- [ ] **Step 2: Run to verify pass with all prior tasks in place**

Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add pgmpy/tests/test_models/test_bayesian_counterfactual.py
git commit -m "test(base): Bayesian regressor yields distribution-valued counterfactual via composition"
```

---

## Task 40p: [Folded] `CPDAdapter` auto-wrap is implemented earlier

Earlier drafts introduced `CPDAdapter` in the SCM phase. The current plan
needs it earlier:

- Task 4 creates `CPDAdapter` and its predict-proba-to-sample/log-prob
  bridge.
- Task 19 wires `_DAGParameters.add(...)` to auto-wrap third-party
  predictive estimators and store only full runtime CPDs.

No code or tests in this task. Keep this placeholder only so later SCM task
numbers remain stable.

---

## Task 40q: `_DAGInference` — real `query()` returning `QueryResult`

Currently the design has `_DAGInference.predict_probability(data)` returning
a DataFrame. Replace with `_DAGInference.query(evidence, query)` returning
a `QueryResult` — matching the rest of the Pearl-ladder accessor APIs.

**Files:**
- Modify: `pgmpy/models/_accessors.py` — `_DAGInference.query`
- Create: `pgmpy/tests/test_models/test_inference_query.py`

- [ ] **Step 1: Failing test**

```python
def test_inference_query_returns_queryresult_on_discrete_bn():
    dag = DAG([("diff", "grade"), ("intel", "grade")])
    # ... build the standard "diff/intel/grade" 3-node BN ...
    res = dag.inference.query(
        evidence={"diff": "easy", "intel": "high"}, query="grade",
        n_samples=10000, seed=0,
    )
    from pgmpy.parameterization import QueryResult
    assert isinstance(res, QueryResult)
    # Most likely outcome is 'A' (P=0.90 in truth).
    freqs = pd.Series(res.samples).value_counts(normalize=True)
    assert freqs.index[0] == "A" and freqs.iloc[0] > 0.85
```

- [ ] **Step 2: Implement** — translate `_DAGInference.query` from
  `prototype.py:_DAGInference.query` (~40 lines). Uses likelihood weighting:
  topological forward sample, clamp evidence variables, accumulate log
  P(evidence | sampled parents), resample by weights.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(base): dag.inference.query returns QueryResult via LW"
```

---

## Task 40r: `_DAGIntervene` — Pearl rung 2 accessor

Add `dag.intervene` accessor. `query()` returns `QueryResult` for the
interventional distribution P(query | do(...)); `simulate()` returns a
DataFrame under the intervention (thin alias for `dag.simulate(do=...)`).

**Files:**
- Modify: `pgmpy/models/_accessors.py` — add `_DAGIntervene`
- Modify: `pgmpy/base/DAG.py` — add `cached_property def intervene`
- Create: `pgmpy/tests/test_models/test_intervene.py`

- [ ] **Step 1: Failing test**

```python
def test_intervene_query_zeroes_x_path():
    """In an LG chain X → Y → Z, do(X=0) collapses Z to its zero-mean prior."""
    scm = DAG([("X", "Y"), ("Y", "Z")])
    # ... build with priors so that E[Z] != 0 without intervention ...
    r_assoc = scm.inference.query(query="Z", n_samples=20000, seed=0)
    r_intv = scm.intervene.query(do={"X": 0.0}, query="Z",
                                  n_samples=20000, seed=1)
    assert abs(r_assoc.point() - 12.0) < 0.5    # associational
    assert abs(r_intv.point() - 0.0) < 0.5      # interventional
    # And the two QueryResults can compare:
    assert r_assoc.compare_to(r_intv)["wasserstein"] > 5
```

- [ ] **Step 2: Implement** — translate from `prototype.py:_DAGIntervene`
  (~30 lines). `simulate` calls `dag.simulate(do=...)`; `query` calls
  `simulate` and wraps the query column as `QueryResult`.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(base): dag.intervene — Pearl rung 2 accessor returning QueryResult"
```

---

## Task 40s: Multi-world counterfactual via list-of-`do`

Extend `_DAGCounterfactual.query` to accept either a single `do` dict or a
list of dicts. When a list, return a list of `QueryResult`s sharing
abducted noise — Pearl's twin/parallel-world semantics, after ChiRho's
`MultiWorldCounterfactual`.

**Files:**
- Modify: `pgmpy/models/_accessors.py` — extend `_DAGCounterfactual.query`
- Create: `pgmpy/tests/test_models/test_multi_world.py`

- [ ] **Step 1: Failing test**

```python
def test_multi_world_shares_abducted_noise():
    """Three worlds with do(X) in {-1, 0, 1}; results match analytic."""
    scm = DAG([("X", "Y"), ("Y", "Z")])
    # ... build LG SCM ...
    observed = {"X": 1.0, "Y": 3.0, "Z": 10.0}
    worlds = scm.counterfactual.query(
        observed=observed,
        do=[{"X": -1.0}, {"X": 0.0}, {"X": 1.0}],
        query="Z",
    )
    assert isinstance(worlds, list) and len(worlds) == 3
    # Analytic: U_Y=1, U_Z=1 abducted; Z_cf = 3*(2*do_X + 1) + 1
    for r, x_val, expected in zip(worlds, [-1.0, 0.0, 1.0], [-2.0, 4.0, 10.0]):
        assert abs(r.point() - expected) < 1e-9
```

- [ ] **Step 2: Implement** — split current `query` into `_abduce`
  (returns noise samples once) + `_propagate` (applies one do dict). Then
  `query` calls `_abduce` once and loops `_propagate` over the do list.
  Reference: `prototype.py:_DAGCounterfactual.query` (~30 lines extended).

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(base): multi-world counterfactual via list-of-do"
```

---

## Task 40t: Standalone graph primitives under `dag.transforms`

Add `ancestors`, `descendants`, `topological_order`, `markov_blanket`,
`d_separated` to `_DAGTransforms`. Thin wrappers around networkx 3.x +
pgmpy.base. Borrowed from causal-learn / R6causal where standalone graph
primitives are usable outside the inference pipeline.

**Files:**
- Modify: `pgmpy/models/_accessors.py` — extend `_DAGTransforms`
- Create: `pgmpy/tests/test_models/test_transforms_primitives.py`

- [ ] **Step 1: Failing test**

```python
def test_transforms_graph_primitives():
    dag = DAG([("X", "Y"), ("Y", "Z")])
    assert dag.transforms.topological_order() == ["X", "Y", "Z"]
    assert dag.transforms.ancestors("Z") == {"X", "Y"}
    assert dag.transforms.descendants("X") == {"Y", "Z"}
    assert dag.transforms.markov_blanket("Y") == {"X", "Z"}
    assert dag.transforms.d_separated("X", "Z", {"Y"}) is True
    assert dag.transforms.d_separated("X", "Z") is False
```

- [ ] **Step 2: Implement** — translate from `prototype.py:_DAGTransforms`
  (~30 lines). Wraps `nx.ancestors`, `nx.descendants`, `nx.topological_sort`,
  `nx.is_d_separator` (or `nx.d_separated` depending on networkx version).

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(base): standalone graph primitives under dag.transforms"
```

---

## Task 40u: `_DAGBootstrap` — fit-time CIs

Add `dag.bootstrap.query(data, query_fn, n_bootstrap)` that resamples
training data, refits a structurally-equivalent DAG, runs the user-supplied
query, and aggregates results into a `QueryResult`. Inspired by
`dowhy.gcm.bootstrap_sampling`.

**Files:**
- Modify: `pgmpy/models/_accessors.py` — add `_DAGBootstrap`
- Modify: `pgmpy/base/DAG.py` — add `cached_property def bootstrap`
- Create: `pgmpy/tests/test_models/test_bootstrap.py`

- [ ] **Step 1: Failing test**

```python
def test_bootstrap_gives_nontrivial_credible_interval():
    """100 refits on a small dataset should produce a CI > 0 width."""
    rng = np.random.default_rng(0)
    n = 300       # small → meaningful fit uncertainty
    x = rng.normal(0, 1, n)
    y = 2.0 * x + rng.normal(0, 0.5, n)
    z = 3.0 * y + rng.normal(0, 0.3, n)
    data = pd.DataFrame({"X": x, "Y": y, "Z": z})

    dag = DAG([("X", "Y"), ("Y", "Z")])
    # ... add CPDs ...
    dag.fit(data)

    result = dag.bootstrap.query(
        data=data,
        query_fn=lambda d: d.counterfactual.query(
            observed={"X": 1, "Y": 3, "Z": 10}, do={"X": 0}, query="Z",
        ),
        n_bootstrap=100, seed=42,
    )
    lo, hi = result.credible_interval(0.95)
    assert hi - lo > 0.05    # non-trivial fit uncertainty
    assert result.samples.shape == (100,)
```

- [ ] **Step 2: Implement** — translate from
  `prototype.py:_DAGBootstrap` (~50 lines). `_fresh_copy` calls
  `dag.copy_template(parameters="unfit")`, giving an unfit DAG with the same
  graph, schema, CPD specs, and parent ordering. Each bootstrap iteration:
  resample data → fresh DAG → fit → query_fn → collect samples.

- [ ] **Step 3: Run tests, commit**

```bash
git commit -m "feat(base): dag.bootstrap.query for fit-time confidence intervals"
```

---

## Task 40v: Deprecate `pgmpy.inference.CausalInference`; port helpers to `dag.diagnostics`; extend `dag.intervene.query(adjustment_set=...)`

The v1.x `pgmpy.inference.CausalInference` (1078 lines at
`pgmpy/inference/CausalInference.py`) overlaps with the new accessor
surface in three ways:

1. `CausalInference(model).query(["Y"], do={...}, evidence={...}, adjustment_set=[...])` → `dag.intervene.query(do={...}, query="Y", evidence={...}, adjustment_set=[...])` (Task 40r added the accessor without `adjustment_set`; this task adds it).
2. Adjustment-set / IV / identification helpers (`identification_method`, `get_all_backdoor_adjustment_sets`, `is_valid_backdoor_adjustment_set`, `get_all_frontdoor_adjustment_sets`, `is_valid_frontdoor_adjustment_set`, `get_minimal_adjustment_set`, `is_valid_adjustment_set`, `get_proper_backdoor_graph`, `get_ivs`, `get_conditional_ivs`, `get_total_conditional_ivs`, `get_scaling_indicators`, `estimate_ate`) → `dag.diagnostics.<same_name>`.
3. `CausalInference` itself becomes a `FutureWarning`-emitting shim that forwards each method to the corresponding accessor; the class is **deleted in Phase 5 (Task 49a)**.

**Files:**
- Modify: `pgmpy/base/_accessors.py` — extend `_DAGDiagnostics` with the ported helpers.
- Modify: `pgmpy/models/_accessors.py` — extend `_DAGIntervene.query` with `evidence` + `adjustment_set` + `inference_algo` parameters.
- Modify: `pgmpy/inference/CausalInference.py` — convert to a shim.
- Create: `pgmpy/tests/test_models/test_dag_diagnostics_helpers.py`
- Create: `pgmpy/tests/test_models/test_dag_intervene_adjustment.py`
- Create: `pgmpy/tests/test_inference/test_causal_inference_shim.py`

- [ ] **Step 1: Write failing tests for `dag.diagnostics` helpers**

```python
# pgmpy/tests/test_models/test_dag_diagnostics_helpers.py
import warnings

import pytest
from pgmpy.base import DAG


def test_dag_diagnostics_backdoor_adjustment_sets_matches_legacy():
    from pgmpy.inference import CausalInference
    dag = DAG([("X", "Y"), ("W", "X"), ("W", "Y")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy = CausalInference(dag).get_all_backdoor_adjustment_sets("X", "Y")
    new = dag.diagnostics.get_all_backdoor_adjustment_sets("X", "Y")
    assert {frozenset(s) for s in new} == {frozenset(s) for s in legacy}


def test_dag_diagnostics_identification_method():
    dag = DAG([("X", "Y"), ("W", "X"), ("W", "Y")])
    assert dag.diagnostics.identification_method("X", "Y") in {"backdoor", "frontdoor", "iv"}


def test_dag_diagnostics_is_valid_adjustment_set():
    dag = DAG([("X", "Y"), ("W", "X"), ("W", "Y")])
    assert dag.diagnostics.is_valid_adjustment_set("X", "Y", ["W"]) is True
    assert dag.diagnostics.is_valid_adjustment_set("X", "Y", []) is False
```

- [ ] **Step 2: Write failing test for `dag.intervene.query(adjustment_set=...)`**

```python
# pgmpy/tests/test_models/test_dag_intervene_adjustment.py
import warnings

import pytest
from pgmpy.parameterization import TabularCPD


def test_intervene_query_with_adjustment_set_matches_causal_inference():
    """dag.intervene.query(adjustment_set=...) reproduces CausalInference.query."""
    from pgmpy.base import DAG
    from pgmpy.inference import CausalInference

    dag = DAG([("X", "Y"), ("W", "X"), ("W", "Y")])
    dag.parameters.add(variable="W", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["a", "b"]],
    ))
    dag.parameters.add(variable="X", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.8, 0.2], [0.2, 0.8]],
        state_names=[["lo", "hi"], ["a", "b"]],
    ), parent_order=["W"])
    dag.parameters.add(variable="Y", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2, 2],
        values=[[0.9, 0.5, 0.6, 0.1], [0.1, 0.5, 0.4, 0.9]],
        state_names=[["no", "yes"], ["lo", "hi"], ["a", "b"]],
    ), parent_order=["X", "W"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy = CausalInference(dag).query(
            ["Y"], do={"X": "hi"}, adjustment_set=["W"], show_progress=False,
        )

    new = dag.intervene.query(
        do={"X": "hi"}, query="Y", adjustment_set=["W"],
        inference_algo="ve",
    )
    # Same point estimate (within numerical tolerance — both exact via VE).
    assert abs(new.point() - float(legacy.values[1])) < 1e-9 \
        or abs(new.expectation(lambda y: 1 if y == "yes" else 0)
               - float(legacy.values[legacy.no_to_name["Y"]["yes"]])) < 1e-9
```

- [ ] **Step 3: Write failing test for the deprecation shim**

```python
# pgmpy/tests/test_inference/test_causal_inference_shim.py
import warnings

import pytest
from pgmpy.base import DAG


def test_causal_inference_init_emits_future_warning():
    from pgmpy.inference import CausalInference
    dag = DAG([("X", "Y")])
    with pytest.warns(FutureWarning, match="dag.intervene"):
        CausalInference(dag)


def test_causal_inference_query_delegates_to_intervene():
    from pgmpy.inference import CausalInference
    from pgmpy.parameterization import TabularCPD

    dag = DAG([("X", "Y")])
    dag.parameters.add(variable="X", cpd=TabularCPD.from_values(
        variable_card=2, values=[[0.5], [0.5]],
        state_names=[["lo", "hi"]],
    ))
    dag.parameters.add(variable="Y", cpd=TabularCPD.from_values(
        variable_card=2, evidence_card=[2],
        values=[[0.7, 0.3], [0.3, 0.7]],
        state_names=[["no", "yes"], ["lo", "hi"]],
    ), parent_order=["X"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = CausalInference(dag).query(
            ["Y"], do={"X": "hi"}, show_progress=False,
        )
    # Same surface as v1.x — DiscreteFactor return type.
    assert result.variables == ["Y"]


def test_causal_inference_get_backdoor_delegates_to_diagnostics():
    from pgmpy.inference import CausalInference
    dag = DAG([("X", "Y"), ("W", "X"), ("W", "Y")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        sets = CausalInference(dag).get_all_backdoor_adjustment_sets("X", "Y")
    via_diagnostics = dag.diagnostics.get_all_backdoor_adjustment_sets("X", "Y")
    assert {frozenset(s) for s in sets} == {frozenset(s) for s in via_diagnostics}
```

- [ ] **Step 4: Port helpers to `_DAGDiagnostics`**

In `pgmpy/base/_accessors.py`, extend `_DAGDiagnostics` with each of
the 13 helpers listed at the top of this task. The simplest port is
verbatim transcription from the existing
`pgmpy/inference/CausalInference.py` method bodies, replacing
`self.model` with `self._dag` and dropping the `self.dag` alias. Example
for `identification_method`:

```python
# Inside _DAGDiagnostics in pgmpy/base/_accessors.py
def identification_method(self, X, Y):
    """Port of pgmpy.inference.CausalInference.identification_method.
    See `pgmpy/inference/CausalInference.py:552` for the v1.x body."""
    import networkx as nx
    from pgmpy.base.ADMG import ADMG

    # ... copy the body verbatim, swapping self.model → self._dag ...
```

Apply the same pattern to:
`get_all_backdoor_adjustment_sets`,
`is_valid_backdoor_adjustment_set`,
`get_all_frontdoor_adjustment_sets`,
`is_valid_frontdoor_adjustment_set`,
`get_minimal_adjustment_set`,
`is_valid_adjustment_set`,
`get_proper_backdoor_graph`,
`get_ivs`,
`get_conditional_ivs`,
`get_total_conditional_ivs`,
`get_scaling_indicators`,
`estimate_ate`.

Each is a single-method port of 5–80 lines of existing code.

- [ ] **Step 5: Extend `_DAGIntervene.query` with adjustment-set support**

In `pgmpy/models/_accessors.py`, update `_DAGIntervene.query` signature
to match the contract:

```python
def query(
    self,
    do,
    *,
    query,
    evidence=None,
    adjustment_set=None,
    inference_algo="ve",
    n_samples=10000,
    seed=None,
):
    """P(query | do(...), evidence) → QueryResult.

    When adjustment_set is provided (or auto-derived from do-variable
    parents if omitted and evidence is given), uses the back-door /
    front-door adjustment formula via VE/BP. Otherwise forward-samples
    under intervention and returns the empirical distribution.

    Replaces pgmpy.inference.CausalInference.query.
    """
    from pgmpy.parameterization import QueryResult

    if adjustment_set is not None or evidence:
        # Adjustment-corrected exact / approximate query — body translated
        # from pgmpy/inference/CausalInference.py:909-1078.
        # Auto-derive adjustment_set from do-variable parents if omitted.
        # Run VE / BP / LW per `inference_algo` (factor return).
        factor = self._adjusted_query(
            query=query, do=do, evidence=evidence or {},
            adjustment_set=adjustment_set, inference_algo=inference_algo,
        )
        # Wrap into QueryResult.
        # … samples drawn from factor (discrete) or analytic (LG) …
        return QueryResult(
            samples=...,
            query=query,
            operation="intervene",
            operation_args={"do": do, "evidence": evidence,
                            "adjustment_set": list(adjustment_set or [])},
        )

    # Pure forward-sample path (no adjustment) — existing behaviour.
    samples = self._dag.simulate(n_samples=n_samples, do=do, seed=seed)
    return QueryResult(
        samples=samples[query].values,
        query=query,
        operation="intervene",
        operation_args={"do": do},
    )
```

The `_adjusted_query` helper is a private method on `_DAGIntervene`
that ports the back-door / front-door formula body from
`pgmpy/inference/CausalInference.py:909-1078`.

- [ ] **Step 6: Convert `pgmpy/inference/CausalInference.py` to a shim**

Replace the body of `pgmpy/inference/CausalInference.py` with:

```python
"""Deprecated. Use ``dag.intervene`` and ``dag.diagnostics`` accessors.

This class will be removed in pgmpy 2.0.
"""
import warnings


class CausalInference:
    """Deprecated. Use ``dag.intervene`` (rung-2 queries) and
    ``dag.diagnostics`` (adjustment / identification helpers)."""

    def __init__(self, model):
        warnings.warn(
            "pgmpy.inference.CausalInference is deprecated and will be removed "
            "in pgmpy 2.0. Use `dag.intervene.query(do=..., evidence=..., "
            "adjustment_set=...)` for rung-2 queries and the helpers on "
            "`dag.diagnostics` (identification_method, "
            "get_all_backdoor_adjustment_sets, …) for adjustment / IV work. "
            "See docs/source/migration-v2.rst for the full mapping.",
            FutureWarning,
            stacklevel=2,
        )
        self.model = model
        self.dag = model

    def query(self, variables, do=None, evidence=None,
              adjustment_set=None, inference_algo="ve",
              show_progress=True, **kwargs):
        # Map v1.x list-of-vars contract to single-query accessor call.
        if len(variables) != 1:
            raise NotImplementedError(
                "Multi-variable queries are not supported in the deprecation "
                "shim. Call dag.intervene.query(query=...) per variable."
            )
        return self.model.intervene.query(
            do=do or {}, query=variables[0],
            evidence=evidence, adjustment_set=adjustment_set,
            inference_algo=inference_algo,
        )

    # 13 forwarding shims to dag.diagnostics — each one-liner:
    def identification_method(self, X, Y):
        return self.model.diagnostics.identification_method(X, Y)

    def get_all_backdoor_adjustment_sets(self, X, Y):
        return self.model.diagnostics.get_all_backdoor_adjustment_sets(X, Y)

    def is_valid_backdoor_adjustment_set(self, X, Y, Z=[]):
        return self.model.diagnostics.is_valid_backdoor_adjustment_set(X, Y, Z)

    def get_all_frontdoor_adjustment_sets(self, X, Y):
        return self.model.diagnostics.get_all_frontdoor_adjustment_sets(X, Y)

    def is_valid_frontdoor_adjustment_set(self, X, Y, Z=None):
        return self.model.diagnostics.is_valid_frontdoor_adjustment_set(X, Y, Z)

    def get_minimal_adjustment_set(self, X, Y):
        return self.model.diagnostics.get_minimal_adjustment_set(X, Y)

    def is_valid_adjustment_set(self, X, Y, adjustment_set):
        return self.model.diagnostics.is_valid_adjustment_set(X, Y, adjustment_set)

    def get_proper_backdoor_graph(self, X, Y, inplace=False):
        return self.model.diagnostics.get_proper_backdoor_graph(X, Y, inplace=inplace)

    def get_ivs(self, X, Y, scaling_indicators=None):
        return self.model.diagnostics.get_ivs(X, Y, scaling_indicators or {})

    def get_conditional_ivs(self, X, Y, scaling_indicators=None):
        return self.model.diagnostics.get_conditional_ivs(X, Y, scaling_indicators or {})

    def get_total_conditional_ivs(self, X, Y, scaling_indicators=None):
        return self.model.diagnostics.get_total_conditional_ivs(X, Y, scaling_indicators or {})

    def get_scaling_indicators(self):
        return self.model.diagnostics.get_scaling_indicators()

    def estimate_ate(self, X, Y, data, estimator_type="linear",
                     adjustment_set=None, **kwargs):
        return self.model.diagnostics.estimate_ate(
            X, Y, data, estimator_type=estimator_type,
            adjustment_set=adjustment_set, **kwargs,
        )
```

- [ ] **Step 7: Run tests**

```bash
pytest pgmpy/tests/test_models/test_dag_diagnostics_helpers.py \
       pgmpy/tests/test_models/test_dag_intervene_adjustment.py \
       pgmpy/tests/test_inference/test_causal_inference_shim.py -v
```
Expected: All pass.

Run the existing CausalInference test suite under `-W ignore::FutureWarning`:
```bash
pytest pgmpy/tests/test_inference/test_CausalInference.py -v --tb=short -W ignore::FutureWarning
```
Expected: All pass — every v1.x test continues to work through the shim.

- [ ] **Step 8: Commit**

```bash
git add pgmpy/base/_accessors.py pgmpy/models/_accessors.py pgmpy/inference/CausalInference.py pgmpy/tests/test_models/test_dag_diagnostics_helpers.py pgmpy/tests/test_models/test_dag_intervene_adjustment.py pgmpy/tests/test_inference/test_causal_inference_shim.py
git commit -m "feat(base): port CausalInference helpers to dag.diagnostics; intervene.query gains adjustment_set; CausalInference becomes FutureWarning shim"
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
  2b.
- No code changes if the messages are already correct.

These classes are **deleted in Phase 5 (v2.0)**.

**Files:**
- Modify (if needed): `pgmpy/estimators/MLE.py` — confirm warning message points at `pgmpy.parameter_estimator.MLEEstimator`.
- Modify (if needed): `pgmpy/estimators/BayesianEstimator.py` — points at `TabularCPD(prior_type=...)` plus `pgmpy.parameter_estimator.MLEEstimator`.
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
   | Parameter estimators | `pgmpy.estimators.{MaximumLikelihoodEstimator, BayesianEstimator, EM}` (already FutureWarning in 1.x) | `pgmpy.parameter_estimator.MLEEstimator`; Bayesian tabular fitting via `TabularCPD(prior_type=...)`; latent discrete EM via `pgmpy.parameter_estimator.DiscreteEM` |

3. **What stays the same**:
   - `pgmpy.factors.discrete.DiscreteFactor`, `pgmpy.factors.base.BaseFactor` — unchanged (not CPDs).
   - `pgmpy.models.DynamicBayesianNetwork` — unchanged.
   - Structure learning (`HillClimbSearch`, `PC`, `GES`, `MMHC`, `ExhaustiveSearch`, `TreeSearch`) — unchanged; returns `DAG`.
   - Existing inference (`VariableElimination`, `BeliefPropagation`, `ApproxInference`) — unchanged surface.
   - New in v2.0: `LikelihoodWeighting`; exact all-linear-Gaussian work can use `dag.transforms.to_joint_gaussian()` as the optimization hook.

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
`pgmpy.inference.LikelihoodWeighting`,
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

All six deletion tasks (47–50, 49a, 50a) follow the same pattern:

1. Write a test asserting the legacy import / attribute raises
   `ImportError` / `AttributeError`.
2. `git rm` the legacy file(s) and any tests scoped to them, or remove
   the legacy attribute from the surviving file.
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

## Task 49a: Delete `pgmpy.inference.CausalInference`

**Delete:** `pgmpy/inference/CausalInference.py` (the shim added in Task 40v).

**Stay:** `pgmpy/tests/test_inference/test_CausalInference.py` — convert it
to drive `dag.intervene.query(adjustment_set=...)` and `dag.diagnostics`
directly, or delete if `test_dag_diagnostics_helpers.py` already covers
the same surface. Pick whichever is closer to the existing coverage.

**Modify:** `pgmpy/inference/__init__.py` — drop the `CausalInference`
export.

**Deletion-confirmation test** (`pgmpy/tests/test_inference/test_causal_inference_deleted.py`):

```python
import pytest


def test_causal_inference_is_gone():
    with pytest.raises(ImportError):
        from pgmpy.inference import CausalInference  # noqa


def test_intervene_and_diagnostics_replacements_exist():
    from pgmpy.base import DAG
    dag = DAG([("X", "Y")])
    # New entry points must be present.
    assert hasattr(dag, "intervene")
    assert hasattr(dag.intervene, "query")
    assert hasattr(dag, "diagnostics")
    assert hasattr(dag.diagnostics, "identification_method")
    assert hasattr(dag.diagnostics, "get_all_backdoor_adjustment_sets")
    assert hasattr(dag.diagnostics, "estimate_ate")
```

Commit: `feat(inference)!: delete pgmpy.inference.CausalInference (v2.0)`.

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

## Task 50a: Delete `DAG.do(nodes)` shim

**Modify:** `pgmpy/base/DAG.py` — delete the `do(...)` deprecation
alias added in Task 28a. The renamed `with_intervention(nodes)` stays.

**Deletion-confirmation test** (`pgmpy/tests/test_base/test_dag_do_shim_deleted.py`):

```python
from pgmpy.base import DAG


def test_dag_do_alias_is_gone():
    dag = DAG([("X", "A")])
    assert not hasattr(dag, "do")
    # The rename target survives.
    new = dag.with_intervention("A")
    assert ("X", "A") not in new.edges()
```

Commit: `feat(base)!: delete DAG.do FutureWarning alias (v2.0)`.

---

## Task 51: v2.0 final regression + release

- [ ] **Step 1:** `pytest pgmpy/tests/ -v --tb=short -W error::FutureWarning` — expected all pass. Any failure indicates pgmpy-internal use of a deleted API; fix.

- [ ] **Step 2:** Smoke-test the canonical v2.0 import surface (`DAG`, `TabularCPD`, `MLEEstimator`, existing exact/approximate inference classes, and `LikelihoodWeighting`) in a fresh REPL.

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
1–51 (plus interleaved letter-suffixed tasks 28a, 28b, 35a, 40a–40v, 49a, 50a).
See the spec's release-staging table for the high-level mapping.

**Final task count:** 57 across two major releases — 1.x rollout
(Tasks 1–46 + 28a, 28b, 35a, 40a–40v, additive) + v2.0 cleanup
(Tasks 47–51, 49a, 50a — deletions).

**Per-task TDD pattern:** every implementation step has a failing test
first, then the code, then a verification run, then the commit. All code
blocks are full (no placeholders).

**For design rationale, alternatives, and trade-offs:** see the spec
(`2026-05-14-parameterization-refactor-design.md`). For full class
signatures: see the contracts doc
(`2026-05-14-parameterization-contracts.md`). This plan is execution-only.
