# Prototype findings

A standalone runnable prototype validating the v2.0 parameterization
refactor design. Run with:

```sh
PYTHONPATH=/path/to/skpro python prototype.py
```

## Demos and results

| # | Demo | Result |
|---|---|---|
| 1 | Discrete BN from values → simulate | ✅ PASS — empirical marginals match the specified CPT |
| 2 | Discrete BN fit from 2000 rows via `MLEEstimator` | ✅ PASS — recovers values to ±0.01 |
| 3 | Hybrid BN with skpro `GLMRegressor` as a CPD | ✅ PASS — fits, samples, no wrapping needed |
| 4 | `cpd_log_prob` dispatch on Tabular + LG | ✅ PASS — matches scipy.stats.norm.logpdf |
| 5 | skbase `clone()` / `get_params()` on DAG | ✅ PASS — after a small init-args fix (see below) |
| 6 | `parameters` accessor dict-like behavior | ✅ PASS — cached_property identity, `__contains__`, etc. |
| 7 | sklearn `RandomForestClassifier` as a discrete CPD | ✅ PASS — with caveat (see "Confirmed limitation") |
| 8 | LikelihoodWeighting on hybrid skpro network | ✅ PASS — LW posterior matches analytic conjugate-Gaussian to 2% on mean, 10% on variance with 20k samples |

## What the prototype validates

### The core design works

- **Identity-free CPDs** with `_tags` work cleanly. `TabularCPD(BaseEstimator, ClassifierMixin)` and `LinearGaussianCPD(BaseProbaRegressor)` both follow their upstream contracts.
- **Multi-inheritance** `DAG(nx.DiGraph, skbase.BaseObject)` works. The MRO resolves cleanly: `[DAG, DiGraph, Graph, BaseObject, _FlagManager, ...]`. Both `nx.DiGraph.__init__` and `skbase.BaseObject.__init__` need explicit calls (no auto via `super()`) since they're cooperative-multiple-inheritance unfriendly.
- **`cached_property` accessors** work — `dag.parameters is dag.parameters` is True across multiple accesses.
- **`MLEEstimator`** orchestrates per-node `cpd.fit(X, y)` correctly across all three CPD types (Tabular, LinearGaussian, third-party skpro/sklearn).
- **`cpd_sample` / `cpd_log_prob` dispatch helpers** correctly route through:
  - native `sample()` / `log_prob()` when defined (our own CPDs),
  - `predict_proba(X).sample()` / `.log_pdf()` for skpro distributions,
  - `predict_proba(X)` matrix indexing for sklearn classifiers.

### Third-party estimators work without wrapping

The big bet: **skpro's `GLMRegressor` and sklearn's `RandomForestClassifier` drop straight in as CPDs.**

```python
from skpro.regression.linear import GLMRegressor
dag.parameters.add(variable="x2", cpd=GLMRegressor(), parent_order=["x1"])
dag.fit(data)        # works
dag.simulate(...)    # works
```

Both estimators:
- pass `check_parameterization` (have `fit(X, y)` and `predict_proba(X)`),
- are fitted correctly by the `MLEEstimator` per-node loop,
- get sampled and scored correctly via the dispatch helpers.

### Inference end-to-end

Demo 8 implements likelihood weighting on the hybrid network and recovers the analytic Bayesian posterior:

- Network: `x1 ~ N(0, 1)`, `x2 | x1 ~ N(2*x1, 0.5)` (GLMRegressor).
- Evidence: `x2 = 4`.
- Analytic posterior: `P(x1 | x2=4) = N(32/17 ≈ 1.882, 1/17 ≈ 0.059)`.
- LW estimate (20k samples): `mean ≈ 1.851, var ≈ 0.065`.

This validates the entire prediction pipeline (forward sampling + per-evidence log_prob weighting) through a third-party estimator.

## Gaps the prototype found

### 1. `_DAGParameters.__getitem__` was missing from the spec

The original spec listed `keys`, `values`, `items`, `__contains__`, `__iter__`, `__len__`, plus `add` / `get` / `remove` — but **not `__getitem__`**. Demo 2 hit this when trying `dag.parameters["diff"]`, which is the natural way to look up a CPD by node name. Added in the prototype as:

```python
def __getitem__(self, node):
    if node not in self._dag._cpds:
        raise KeyError(f"No CPD registered for node {node!r}")
    return self._dag._cpds[node]
```

**Action for the plan**: add `__getitem__` to the `_DAGParameters` contract in the contracts doc and the spec.

### 2. skbase `get_params` / `clone` requires init args be stored as instance attributes (unchanged)

skbase's `get_params()` introspects `__init__`'s signature, then reads `self.<param_name>` for each. `clone()` then uses those values to construct a new instance and verifies the new instance round-trips.

This means **`DAG.__init__` must store `ebunch` and `latents` as `self.ebunch` and `self.latents` exactly as passed in** — not transformed, not copied. If you do `self.ebunch = list(ebunch)`, clone fails because the new instance's `ebunch` is a different list object than the original's.

```python
def __init__(self, ebunch=None, latents=None):
    self.ebunch = ebunch         # store unchanged
    self.latents = latents       # store unchanged
    nx.DiGraph.__init__(self, incoming_graph_data=ebunch)
    SkbaseBaseObject.__init__(self)
    # ... runtime state under _-prefixed attributes (not init args)
```

**Action for the plan**: Task 19 Step 3 should call this out explicitly. The runtime state (`_cpds`, `_parent_order`, `_latents`) goes under `_`-prefixed attributes; the init args stay un-prefixed and unchanged.

### 3. Confirmed limitation: third-party estimators require numeric parents

Demo 7 initially used string-valued parents (`"young"` / `"old"`); RandomForestClassifier raised `ValueError: could not convert string to float: 'old'`. Fixed by numerically encoding parents upfront.

This is the **"Mixed-type parent preprocessing" non-goal** already documented in the spec. The prototype confirms it's a real and visible limitation:

- pgmpy historically handles string state-names natively.
- Third-party estimators (skpro, sklearn) require numeric `X`.
- When a categorical parent feeds a third-party child, the user must encode upfront (or pgmpy needs a default one-hot wrapper).

**Action for the plan**: confirm this is acceptable for v2.0 ship and that the migration guide explicitly tells users to one-hot or label-encode categorical parents before feeding them to a sklearn/skpro CPD. The "default one-hot encoder" is a follow-up enhancement, not a blocker.

### 4. skpro's `BaseProbaRegressor` exposes `is_fitted` not `is_fitted_`

In the spec / plan I described CPDs having an `is_fitted_` attribute (sklearn trailing-underscore convention). The skpro `BaseProbaRegressor` actually exposes `is_fitted` (no underscore — it's a property). Our own `LinearGaussianCPD.from_values` set `self.is_fitted_ = True` but skpro's machinery doesn't see it; the skpro-managed flag is `_is_fitted` (underscore prefix, set by skpro after `_fit`).

For `LinearGaussianCPD.from_values` to make the CPD usable, we set `instance._is_fitted = True` (the skpro internal flag). `cpd.is_fitted` then returns True.

**Action for the plan**: the contracts doc and Task 11 should specify `LinearGaussianCPD.from_values` sets `_is_fitted = True` (skpro convention) rather than `is_fitted_ = True` (sklearn convention). The two estimator ecosystems differ on this detail. Document the convention per CPD class.

## Things the prototype did NOT validate

The prototype is deliberately scoped to parameterization + fitting + prediction. It does **not** exercise:

- `FunctionalCPD` (requires pyro; would need a separate prototype).
- `JointPyroEstimator` (same).
- `VariableElimination` / `BeliefPropagation` tag dispatch (these are pgmpy-internal — the prototype's standalone DAG isn't enough to test them).
- `_DAGTransforms.to_markov_model` / `to_joint_gaussian` (would need DiscreteFactor and full inference machinery).
- `_DAGIO` save/load (would need pgmpy's readwrite module).
- Read/write integration (BIF, XMLBIF, UAI).
- DBN handling (temporal-slice key tuples).
- The `cpd.get_tag()` interface — the prototype reads `cpd._tags[name]` directly via `_get_tag`. Real CPDs inheriting from skbase get `get_tag` for free; our own implementations need to add a `get_tag` method (or inherit it from skbase too).

These are "out of prototype scope" but the existing plan covers them.

## Verdict

The core design is sound. The prototype shows:

1. **skpro integration works.** A real `GLMRegressor` fits, samples, and supports likelihood weighting through the dispatch helpers without any pgmpy-specific adaptation.
2. **Sklearn integration works** with the documented numeric-parent caveat.
3. **The multi-inheritance pattern** (DAG ← nx.DiGraph + skbase.BaseObject) is workable — needs init args stored as instance attrs.
4. **Three small gaps** to fold into the plan: `__getitem__` on the accessor, the init-args storage convention, and the `is_fitted` vs `is_fitted_` naming difference between sklearn and skpro.

No design-level surprises. The plan is implementable as written, modulo these small additions.
