## Design 1: Conditional CPD Boundary and Adapter Layer

Contributors: @ankurankan, @daehyun

### Introduction

pgmpy currently has three incompatible CPD families:

- `TabularCPD`, which is tightly coupled to factor semantics.
- `LinearGaussianCPD`, which is a separate continuous CPD class.
- `FunctionalCPD`, which is Pyro-based and has a different operational surface.

This makes it difficult to treat "local conditional model for one node" as a reusable abstraction. It also blocks a core extensibility goal for the refactor: a user should be able to register an ordinary sklearn classifier or skpro probabilistic estimator as a CPD without having to subclass pgmpy internals.

There is a second pressure on the design. In causal workflows, users do not usually think in terms of arbitrary CPDs. They think in terms of structural mechanisms such as ANM, PNL, and linear additive-noise models. Those models carry stronger semantics than an ordinary conditional distribution, especially for counterfactuals.

The design therefore needs to support two related but distinct scenarios:

1. ordinary probabilistic CPDs representing `P(X | Pa(X))`
2. structural mechanisms representing `X = f(Pa(X), U)`

This mirrors the direction taken by packages such as DoWhy GCM, bnlearn, and pyAgrum: ordinary conditional models are the base layer, and structural or counterfactual semantics are explicit additional information rather than the default meaning of every CPD.

This document defines the local CPD boundary only. It does not decide the graph-level API, schema ownership, or causal query surface. Those belong in `02_parameterized_dag.md` and `03_scm_counterfactuals.md`.

### Proposed Solution

Introduce a new `pgmpy.parameterization` module with one public CPD boundary and one explicit structural refinement:

- `BaseConditionalCPD` is the base abstraction for ordinary conditional models.
- `BaseStructuralCPD` refines it for models with explicit exogenous noise semantics.
- `BaseAbductableStructuralCPD` refines it again for models that support noise recovery needed for counterfactual abduction.
- pgmpy-native models subclass these base classes.
- third-party sklearn and skpro estimators are accepted through protocol checks and registration-time adapters rather than inheritance.

The key rule is that a CPD does not own graph identity. It stores local model shape and fitted parameters only. Node ownership, parent order, encoders, and state metadata move to the graph layer.

This gives pgmpy one reusable local-model interface while still allowing first-class structural classes such as ANM and PNL to exist as explicit, user-facing models.

### Alternative Solutions

| Alternative | Why not choose it |
|---|---|
| One large `BaseCPD` containing probabilistic and structural methods | Forces ordinary CPDs to pretend they are SCMs and makes the base contract much heavier than sklearn-style estimators need. |
| Require every custom CPD to inherit pgmpy base classes | Prevents direct reuse of sklearn and skpro objects, which is one of the main goals of the refactor. |
| Keep probabilistic CPDs and structural mechanisms as completely separate hierarchies | Loses the shared local-model surface for fitting, sampling, scoring, and registration, and makes mixed graphs harder to support. |
| Keep node identity on the CPD | Continues to tie local models to one graph and blocks direct estimator reuse. |

### Details of proposed solution

#### Public abstract layers

The local contract should be split into a narrow probabilistic base and an explicit structural refinement.

```python
class BaseConditionalCPD(ABC):
    def fit(self, X: pd.DataFrame, y: pd.Series): ...
    def conditional_distribution(self, X: pd.DataFrame): ...
    def sample(self, X: pd.DataFrame, random_state=None) -> pd.Series: ...
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...
    def get_tags(self) -> dict[str, Any]: ...


class BaseStructuralCPD(BaseConditionalCPD):
    def noise_distribution(self): ...
    def structural_predict(self, X: pd.DataFrame, noise) -> pd.Series: ...


class BaseAbductableStructuralCPD(BaseStructuralCPD):
    def abduct(self, y: pd.Series, X: pd.DataFrame): ...
```

`BaseConditionalCPD` is enough for associational inference, estimation, and simulation. `BaseStructuralCPD` is enough for mechanism-aware simulation under intervention. `BaseAbductableStructuralCPD` is the stronger contract required for unit-level counterfactual workflows.

`conditional_distribution(X)` is the unifying local prediction method. For discrete models, it may represent class probabilities. For continuous models, it may represent a parametric or object-valued conditional distribution. This avoids forcing one `predict_proba` shape onto both categorical and continuous cases.

#### Third-party protocols and adapters

Third-party estimators should not be required to inherit pgmpy classes. Instead, pgmpy should recognize a small set of protocols and wrap them when needed.

```python
class SupportsPredictProba(Protocol):
    def predict_proba(self, X): ...


class SupportsSample(Protocol):
    def sample(self, X, random_state=None): ...


class SupportsLogProb(Protocol):
    def log_prob(self, y, X): ...
```

The main adapters are:

- `ClassifierCPDAdapter` for sklearn-like classifiers with `predict_proba`
- `DistributionCPDAdapter` for skpro-like estimators whose `predict_proba` returns distribution-valued output
- `CPDAdapter` as the common wrapper base

Registration should follow this rule:

1. If the object is already a `BaseConditionalCPD`, store it directly.
2. Else, if it exposes `predict_proba`, adapt it into a `BaseConditionalCPD`.
3. Else, reject it as not satisfying the local CPD contract.

This keeps inheritance optional and gives users the direct experience they expect:

- pass a `LogisticRegression` object for a discrete node
- pass an skpro probabilistic regressor for a continuous node
- pass an `ANM` or `PNL` object for a structural node

#### Built-in pgmpy classes

The built-in CPD set should be organized by semantics rather than by old inheritance accidents:

- `TabularCPD(BaseConditionalCPD)`
- `LinearGaussianCPD(BaseConditionalCPD)`
- `FunctionalCPD(BaseConditionalCPD)` unless it explicitly exposes structural noise semantics
- `ANM(BaseAbductableStructuralCPD)`
- `PNL(BaseAbductableStructuralCPD)`

This gives users a simple rule: use a conditional CPD when you only need `P(X | Pa(X))`; use ANM, PNL, or another structural class when you need explicit noise semantics and counterfactual support.

#### Lifecycle and construction

CPDs should follow sklearn-style construction:

- `__init__` stores hyperparameters only
- fitted parameters live in fitted attributes such as `values_`, `coef_`, `classes_`, `noise_model_`, and `_is_fitted`

Two construction paths should be supported:

- `fit(X, y)` for learning from data
- `from_values(...)` for direct parameter specification where that is natural, especially for tabular CPDs

Node identity is not part of this lifecycle. The CPD does not store `variable`, `evidence`, or `parents` as its primary identity.

#### Capability tags

Dispatch should use capability tags rather than hard-coded class checks. The most important tags are:

- `child_type`: `"categorical"` or `"continuous"`
- `can_materialize_factor`: whether the CPD can be converted to a `DiscreteFactor`
- `supports_log_prob`
- `supports_sampling`
- `supports_counterfactual`
- `supports_abduction`
- `noise_type`

This is especially important for adapted third-party estimators. A classifier-backed CPD might support exact factor materialization if the graph has finite parent schemas, while an ANM supports structural rollout and abduction but not exact discrete factorization.

#### Exact inference boundary

The new `TabularCPD` should not be required to inherit `DiscreteFactor`. Exact inference can continue to consume factors, but factor materialization should happen through a separate transformation owned by the graph layer.

For adapted classifiers, exact inference is only possible when:

- the child is categorical
- the parent set is finite and fully schema-defined
- the CPD advertises `can_materialize_factor=True`

That keeps the local CPD boundary general without weakening exact inference for the discrete cases that can still support it.

### User journeys with the solution

#### User journey 1: built-in discrete CPD

```python
from pgmpy.parameterization import TabularCPD

cpd = TabularCPD.from_values(
    variable_card=2,
    values=[[0.6], [0.4]],
    state_names=[["easy", "hard"]],
)
```

The user gets a native pgmpy conditional CPD with no graph identity baked into it.

#### User journey 2: sklearn classifier as a CPD

```python
from sklearn.linear_model import LogisticRegression
from pgmpy.parameterization import adapt_cpd

cpd = adapt_cpd(
    LogisticRegression(),
    variable_type="categorical",
    state_names=["no", "yes"],
)
```

The estimator is accepted because it exposes `predict_proba`. pgmpy wraps it in a `ClassifierCPDAdapter` that provides `sample`, `log_prob`, and capability tags.

#### User journey 3: skpro probabilistic regressor as a CPD

```python
from skpro.regression.ensemble import NGBoostRegressor
from pgmpy.parameterization import adapt_cpd

cpd = adapt_cpd(
    NGBoostRegressor(),
    variable_type="continuous",
)
```

The estimator is adapted as a continuous `BaseConditionalCPD` and can participate in fitting, likelihood evaluation, and approximate inference.

#### User journey 4: explicit structural mechanism

```python
from sklearn.ensemble import RandomForestRegressor
from pgmpy.parameterization import ANM, NormalNoise

cpd = ANM(
    regressor=RandomForestRegressor(),
    noise=NormalNoise(),
)
```

The user opts into structural semantics explicitly. The object is still a local-node model, but it now carries the additional contract needed by the causal layer.
