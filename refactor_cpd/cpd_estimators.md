## Refactoring pgmpy's CPDs into Estimators with sklearn/skpro Interoperability

Contributors: @ankurankan, @daehyun

### Introduction

A pgmpy Bayesian network is parameterized by one of three CPD families, and
they do not share a contract:

- `TabularCPD(DiscreteFactor)` — a discrete CPT tightly coupled to factor
  semantics through inheritance.
- `LinearGaussianCPD(BaseFactor)` — a separate continuous CPD with its own
  surface.
- `FunctionalCPD` — a Pyro-based class with a different operational surface
  again.

Two problems follow from this.

First, each class bakes in **graph identity**. A CPD is constructed with the
node it parameterizes and that node's parents
(`TabularCPD(variable="grade", evidence=["diff", "intel"], ...)`,
`LinearGaussianCPD(variable="Y", beta=..., evidence=["X1", "X2"])`). The local
conditional model and the place it sits in a particular graph are fused, so a
fitted CPD cannot be reused, inspected, or tested as a standalone "model for one
variable."

Second, because the three families have no common, minimal surface, a "local
conditional model" is not an extensible abstraction. A user who wants to
parameterize a node with an ordinary scikit-learn classifier or an skpro
probabilistic regressor has no path that does not involve subclassing pgmpy
internals — even though those estimators already expose exactly the operations a
CPD needs (`fit`, `predict_proba`).

This proposal refactors CPDs into **identity-free, sklearn/skpro-style
estimators** and defines a small **duck-typed contract** so that any estimator
carrying the required methods can act as a conditional distribution for a node,
wrapped automatically by an adapter when needed.

**Scope.** This document covers the local CPD boundary only. Two neighboring
concerns are intentionally out of scope and are referenced only where the
boundary matters:

- *Who owns node identity, parent order, variable schema, and parent encoding?*
  That is the parameterized-`DAG` layer (a separate proposal).
- *Structural and counterfactual semantics* (`X = f(Pa(X), U)`, abduction,
  noise models). That is a further, optional layer on top of the contract
  defined here.

### Proposed Solution

Introduce a `pgmpy.parameterization` module in which CPDs are sklearn/skpro-style
estimators and the CPD contract is a set of small protocols rather than a base
class.

1. **Identity-free estimators.** A CPD stores local model shape and fitted
   parameters only. It follows the standard estimator lifecycle: `__init__`
   takes hyperparameters, `fit(X, y)` learns from data, and fitted state lives in
   trailing-underscore attributes (`values_`, `beta_`, `std_`, `classes_`). A
   `from_values(...)` constructor covers direct parameter specification where that
   is natural (tabular CPTs, linear-Gaussian coefficients). The CPD no longer
   stores `variable`, `evidence`, or `parents`.

2. **Built-ins re-based on standard estimator classes.** Rather than inventing a
   pgmpy base hierarchy, the built-in CPDs inherit the relevant upstream base
   directly:

   - `TabularCPD(ClassifierMixin, BaseEstimator)` — a discrete categorical CPT,
     no longer a `DiscreteFactor`.
   - `LinearGaussianCPD(BaseProbaRegressor)` — an skpro probabilistic regressor.
   - `FunctionalCPD(BaseProbaRegressor)` — the Pyro-backed continuous CPD.

3. **A duck-typed, operation-specific contract.** Instead of one fat base class,
   the contract is split into protocols, each describing a single capability:
   `FittableCPD` (`fit`), `PredictiveCPD` (`predict_proba`), `SampleableCPD`
   (`sample`), and `ScorableCPD` (`log_prob`). Every operation checks only the
   protocol surface it actually consumes. This is what lets ordinary estimators
   qualify: a `LogisticRegression` already satisfies `FittableCPD` and
   `PredictiveCPD`.

4. **A single adapter for third-party estimators.** `CPDAdapter` wraps any object
   that exposes `fit(X, y)` + `predict_proba(X)` and derives the rest of the CPD
   surface (`sample`, `log_prob`, tag access) from it. Registration follows one
   rule:

   1. if the object already satisfies the CPD operations natively, use it as-is;
   2. else if it exposes `predict_proba`, wrap it in `CPDAdapter`;
   3. else reject it with a clear `CPDContractError`.

The result is one reusable local-model interface. Native pgmpy CPDs, scikit-learn
classifiers, and skpro probabilistic regressors all participate in fitting,
sampling, and scoring through the same minimal contract, and inheritance is never
required of third-party code.

### Alternative Solutions

| Alternative | Why not choose it |
|---|---|
| One large `BaseCPD` base class carrying every method (including structural/SCM operations) | Forces ordinary conditional models to implement (or stub) machinery they do not need, and makes the contract far heavier than an sklearn-style estimator. |
| A pgmpy `BaseConditionalCPD` ABC that every CPD must subclass | Re-introduces mandatory inheritance. A bare `LogisticRegression` or skpro regressor would have to be subclassed or re-wrapped to qualify, defeating the main extensibility goal. Operation-specific protocols give the same guarantees by duck typing, which is lighter and more Pythonic. |
| Keep node identity (`variable`, `evidence`) on the CPD | Continues to fuse the local model to one graph, so CPDs remain non-reusable and external estimators (which know nothing of pgmpy identity) cannot be dropped in. |
| Keep `TabularCPD(DiscreteFactor)` coupled by inheritance | Makes "is a factor" the universal CPD contract, which continuous and third-party CPDs cannot satisfy. Factor materialization should be an optional capability, not the base class. |
| Require users to hand-wrap and encode every external estimator | Makes the extensibility story brittle and verbose; the point is to accept ordinary `fit`/`predict_proba` objects directly. |

### Details of proposed solution

#### Lifecycle and construction

CPDs follow sklearn/skpro construction conventions:

- `__init__` stores hyperparameters only (e.g. `prior_type`,
  `equivalent_sample_size` for `TabularCPD`).
- `fit(X, y)` returns `self` and populates fitted attributes.
- `from_values(...)` builds a fitted CPD directly from known parameters.

One cross-framework wrinkle is the fitted-state flag: scikit-learn marks fitted
estimators with a trailing-underscore attribute (`is_fitted_`), while skpro uses
an `_is_fitted` flag exposed through an `is_fitted` property. Each CPD follows the
convention of its base class, and `from_values` sets the correct one
(`TabularCPD.from_values` → `is_fitted_ = True`; `LinearGaussianCPD.from_values`
→ `_is_fitted = True`).

```python
class TabularCPD(ClassifierMixin, BaseEstimator):
    """Discrete categorical CPT. Identity-free."""

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "supports_fit_joint": False,
        "python_dependencies": ["scikit-learn"],
    }

    def __init__(self, variable_card, evidence_card=None, state_names=None,
                 prior_type=None, equivalent_sample_size=10, pseudo_counts=None):
        ...

    # Fitted attributes (sklearn convention)
    values_: np.ndarray         # (variable_card, prod(evidence_card))
    classes_: np.ndarray        # child state labels
    is_fitted_: bool

    @classmethod
    def from_values(cls, variable_card, values, evidence_card=None,
                    state_names=None) -> "TabularCPD": ...

    def fit(self, X, y, sample_weight=None) -> "TabularCPD": ...
    def predict_proba(self, X) -> pd.DataFrame: ...   # (len(X), variable_card)
    def sample(self, X, n_samples=1) -> pd.Series: ...
    def log_prob(self, y, X) -> pd.Series: ...


class LinearGaussianCPD(BaseProbaRegressor):
    """P(y | X) = N(beta_[0] + X @ beta_[1:], std_). Identity-free."""

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": True,
        "supports_fit_joint": False,
        "python_dependencies": ["skpro"],
    }

    def __init__(self): ...

    beta_: np.ndarray           # length n_parents + 1; beta_[0] = intercept
    std_: float

    @classmethod
    def from_values(cls, beta, std) -> "LinearGaussianCPD": ...

    def fit(self, X, y, sample_weight=None) -> "LinearGaussianCPD": ...
    def predict_proba(self, X) -> "skpro.distributions.Normal": ...
    def sample(self, X, n_samples=1) -> pd.Series: ...
    def log_prob(self, y, X) -> pd.Series: ...
```

`FunctionalCPD(BaseProbaRegressor)` keeps its Pyro-backed surface
(`sample`/`log_prob` are its primary entry points; it has no `predict_proba`) and
advertises `supports_fit_joint=True`.

The `_tags` / `get_tag(...)` machinery comes from the skbase/skpro base classes:
`BaseEstimator` above is `skbase.base.BaseEstimator` (not
`sklearn.base.BaseEstimator`), while `ClassifierMixin` is scikit-learn's. Pairing
them gives `TabularCPD` both the sklearn classifier API and the skbase tag system,
which is also the same tag system third-party skpro estimators already carry.

#### The CPD contract as protocols

The contract is a set of `typing.Protocol`s, each describing one capability, plus
`require_*` guards and a convenience validator. Operations import only the guard
they need.

```python
from typing import Any, Protocol


class FittableCPD(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs): ...

class PredictiveCPD(Protocol):
    def predict_proba(self, X: pd.DataFrame): ...

class SampleableCPD(Protocol):
    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...

class ScorableCPD(Protocol):
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...


class CPDContractError(TypeError): ...
class IncompatibleCPDError(TypeError): ...

def require_fittable(obj: Any) -> None: ...
def require_sampleable(obj: Any) -> None: ...
def require_scorable(obj: Any) -> None: ...
def require_predictive(obj: Any) -> None: ...

def check_parameterization(obj: Any) -> None:
    """Validate a third-party predictive CPD: FittableCPD + PredictiveCPD.
    Built-in CPDs satisfy narrower protocols depending on the operation."""
```

Checking only the consumed surface is what keeps the door open to heterogeneous
backends: a sampler needs `SampleableCPD`, a likelihood-weighting routine needs
`ScorableCPD`, and a fit loop needs `FittableCPD` — none of them demand that a CPD
be a particular class.

#### The adapter

`CPDAdapter` makes any `fit`/`predict_proba` object satisfy the full CPD surface.
It forwards unknown attributes to the wrapped estimator and synthesizes the
missing operations.

```python
class CPDAdapter:
    """Make any predict_proba-style object satisfy the full CPD contract."""

    def __init__(self, wrapped: Any) -> None: ...
    @property
    def wrapped(self) -> Any: ...
    def __getattr__(self, name): ...        # forward to wrapped

    def fit(self, X, y, **kwargs): ...
    def predict_proba(self, X): ...
    def sample(self, X, n_samples=None) -> pd.Series: ...
    def log_prob(self, y, X) -> pd.Series: ...
    def get_tag(self, name, default=None): ...
    def clone(self) -> "CPDAdapter": ...
```

`sample` and `log_prob` dispatch in three tiers:

1. if the wrapped object defines `sample` / `log_prob` natively, call those;
2. else if `predict_proba(X)` returns a distribution (skpro convention), use
   `dist.sample()` / `dist.log_pdf(y)`;
3. else treat `predict_proba(X)` as a class-probability matrix (sklearn
   classifier convention) and index into it.

Built-in CPDs already implement the full surface, so they pass through
**unwrapped**; only third-party predictive estimators are adapted, and the
wrapping is automatic at registration time (see boundary notes).

#### Capability tags

Dispatch should rely on capability tags rather than `isinstance` checks. The CPD
contract uses a small tag set (the structural/counterfactual tags belong to the
separate SCM layer and are omitted here):

| Tag | Type | Purpose |
|---|---|---|
| `variable_type` | `"discrete" \| "continuous"` | Output-type-aware dispatch (e.g. prediction). |
| `produces_factor` | `bool` | Whether the CPD can be materialized into a `DiscreteFactor` for exact inference. |
| `is_linear_gaussian` | `bool` | Marks linear-Gaussian CPDs for LG-specific optimizations. |
| `supports_fit_joint` | `bool` | Whether the CPD participates in joint (e.g. Pyro) fitting. |
| `python_dependencies` | `list[str]` | Soft-dependency gating. |

| Tag | `TabularCPD` | `LinearGaussianCPD` | `FunctionalCPD` |
|---|:-:|:-:|:-:|
| `variable_type` | discrete | continuous | continuous |
| `produces_factor` | ✓ | ✗ | ✗ |
| `is_linear_gaussian` | ✗ | ✓ | ✗ |
| `supports_fit_joint` | ✗ | ✗ | ✓ |

Third-party CPDs read tags via `cpd.get_tag(name, default)`. Sensible defaults
come from the upstream base class — `sklearn.ClassifierMixin` →
`variable_type="discrete"`, `skpro.BaseProbaRegressor` →
`variable_type="continuous"` — and the remaining tags default to `False`, which
simply routes adapted estimators to the sampling/scoring code paths rather than
the factor path.

#### Exact-inference boundary

Because `TabularCPD` no longer inherits `DiscreteFactor`, exact inference consumes
factors produced *from* a CPD rather than treating the CPD as a factor. A separate
transformation (owned by the graph layer) materializes a factor when the
`produces_factor` capability is advertised and the parent schema is finite. This
keeps the local CPD contract general — continuous and third-party CPDs need not
pretend to be factors — without weakening exact inference for the discrete cases
that still support it.

#### Boundary notes (out of scope, referenced for clarity)

- **Identity, parent order, schema, and encoding live on the DAG.** Because CPDs
  are identity-free, the graph layer is responsible for recording which node a CPD
  parameterizes, the ordered parent list used to build `X`, the variable schema
  (state names, types), and any categorical encoding. This is a separate proposal.
- **Parent encoding is the one practical caveat.** A bare scikit-learn estimator
  raises `could not convert string to float` on categorical parents, because
  pgmpy passes parent columns through as-is. The supported pattern is to encode
  inside the estimator — `sklearn.Pipeline(OneHotEncoder, ...)` or a
  `ColumnTransformer`. Graph-owned default encoders are a natural future extension
  point but are not part of this contract.
- **Structural / counterfactual semantics are a further layer.** Explicit
  exogenous noise (`X = f(Pa(X), U)`), abduction, and noise distributions sit on
  top of the contract defined here and are deliberately excluded.

A runnable prototype of this design fit three heterogeneous CPDs — an
`skpro.GLMRegressor`, a `sklearn.RandomForestClassifier`, and a
`sklearn.Pipeline(StandardScaler, RandomForestClassifier)` — through a single
per-node fit loop, with the `Pipeline` auto-wrapped by `CPDAdapter`, and forward
sampling matched the data-generating truth to within ~1%. The "parents must be
numeric" caveat above was the one limitation surfaced.

### User journeys with the solution

#### Journey 1: a built-in discrete CPD, no identity

```python
from pgmpy.parameterization import TabularCPD

cpd = TabularCPD.from_values(
    variable_card=2,
    values=[[0.6], [0.4]],
    state_names=[["easy", "hard"]],
)

cpd.sample(X)          # draw child states for parent rows X
cpd.predict_proba(X)   # class probabilities over classes_
```

The CPD is a standalone, fitted estimator. It carries no `variable` or `evidence`;
nothing ties it to a particular graph.

#### Journey 2: a built-in linear-Gaussian CPD

```python
from pgmpy.parameterization import LinearGaussianCPD

cpd = LinearGaussianCPD().fit(X, y)   # X: parent columns, y: continuous child
dist = cpd.predict_proba(X)           # an skpro Normal distribution
cpd.sample(X)

# or specify parameters directly
cpd = LinearGaussianCPD.from_values(beta=[0.2, -2.0, 3.0], std=1.0)
```

#### Journey 3: a scikit-learn classifier as a CPD

```python
from sklearn.linear_model import LogisticRegression
from pgmpy.parameterization import CPDAdapter

cpd = CPDAdapter(LogisticRegression())
cpd.fit(X, y)

cpd.sample(X)          # derived from predict_proba
cpd.log_prob(y, X)     # derived from predict_proba
```

`LogisticRegression` is accepted by duck typing: it satisfies `FittableCPD` and
`PredictiveCPD`. `CPDAdapter` supplies `sample`, `log_prob`, and tag access from
its `predict_proba` matrix — no subclassing of pgmpy required.

#### Journey 4: an skpro probabilistic regressor as a CPD

```python
from skpro.regression.ensemble import NGBoostRegressor
from pgmpy.parameterization import CPDAdapter

cpd = CPDAdapter(NGBoostRegressor())
cpd.fit(X, y)

dist = cpd.predict_proba(X)   # a skpro distribution
cpd.sample(X)                 # dist.sample()
cpd.log_prob(y, X)            # dist.log_pdf(y)
```

Here `predict_proba` returns a distribution object, so the adapter routes
sampling and scoring through the distribution rather than a probability matrix —
the same contract, a different backend.

#### Journey 5: a scikit-learn `Pipeline` as a CPD (handling categorical parents)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from pgmpy.parameterization import CPDAdapter

cpd = CPDAdapter(Pipeline([
    ("encode", OneHotEncoder(handle_unknown="ignore")),
    ("clf", RandomForestClassifier()),
]))
cpd.fit(X, y)   # the Pipeline encodes categorical parents internally
```

Because pgmpy passes parent columns through unmodified, encoding categorical
parents is done inside the estimator. A `Pipeline` (or `ColumnTransformer`) is the
supported pattern until graph-owned default encoders are added.

#### Journey 6 (boundary): dropping a raw estimator onto a graph node

```python
from skpro.regression.ensemble import NGBoostRegressor

# Identity (variable, parent_order), schema, and encoding are the DAG's
# responsibility (separate proposal). Shown only to make the payoff concrete:
# a raw estimator is auto-wrapped via CPDAdapter at registration.
dag.parameters.add(variable="Y", cpd=NGBoostRegressor(), parent_order=["X"])
dag.fit(data)
dag.simulate(n_samples=1000)
```

This last journey reaches into the DAG layer, which is out of scope for this
document. It is included only to show the end goal of the CPD refactor: with an
identity-free, duck-typed CPD contract, parameterizing a node with an arbitrary
sklearn or skpro estimator becomes a one-line registration.
