# pgmpy v2.0 Class Contracts

Companion to the design spec (`2026-05-14-parameterization-refactor-design.md`)
and plan (`2026-05-14-parameterization-refactor.md`). Lists **only** the
public contracts — class declarations, capability tags, fitted attributes,
and method signatures. No implementation, no test code.

---

## `pgmpy.parameterization`

Identity-free CPDs. Inherit from skbase/skpro/sklearn base classes; no
`variable` / `evidence` attributes (identity lives on the DAG).

### `TabularCPD`

```python
from skbase.base import BaseEstimator
from sklearn.base import ClassifierMixin


class TabularCPD(BaseEstimator, ClassifierMixin):
    """Discrete categorical CPT.

    Bayesian fitting is configured via __init__ hyperparameters
    (prior_type, equivalent_sample_size). EM fitting is orchestrated
    at the network level by DiscreteEM, which passes sample_weight.
    """

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "is_linear_gaussian": False,
        "supports_analytic_conditioning": True,
        "supports_fit_joint": False,
        "python_dependencies": ["scikit-learn"],
    }

    # __init__ (hyperparameters only)
    def __init__(
        self,
        variable_card: int,
        evidence_card: list[int] | None = None,
        state_names: list[list] | None = None,
        # Bayesian fitting policy (None = MLE):
        prior_type: Literal["BDeu", "K2", "dirichlet"] | None = None,
        equivalent_sample_size: float = 10,
        pseudo_counts: ArrayLike | None = None,  # explicit override
    ) -> None: ...

    # Fitted attributes (sklearn convention — trailing underscore)
    values_: np.ndarray         # shape (variable_card, prod(evidence_card))
    classes_: np.ndarray        # child state labels
    is_fitted_: bool

    # Construction from explicit parameters (mark fitted without running fit)
    @classmethod
    def from_values(
        cls,
        variable_card: int,
        values: ArrayLike,
        evidence_card: list[int] | None = None,
        state_names: list[list] | None = None,
    ) -> "TabularCPD": ...

    # Lifecycle
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: ArrayLike | None = None,
    ) -> "TabularCPD":
        """ML (or MAP if prior_type is set) estimate of the CPT."""

    # sklearn classifier contract
    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...
        # shape (len(X), variable_card), columns = classes_

    # pgmpy-specific
    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...
```

### `LinearGaussianCPD`

```python
from skpro.regression.base import BaseProbaRegressor


class LinearGaussianCPD(BaseProbaRegressor):
    """P(y | X) = N(beta_[0] + X @ beta_[1:], std_)."""

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": True,
        "supports_analytic_conditioning": True,
        "supports_fit_joint": False,
        "python_dependencies": ["skpro"],
    }

    def __init__(self) -> None: ...

    # Fitted attributes (skpro convention — _is_fitted is the underlying flag)
    beta_: np.ndarray           # length n_parents + 1; beta_[0] = intercept
    std_: float                 # residual std

    @classmethod
    def from_values(cls, beta: ArrayLike, std: float) -> "LinearGaussianCPD": ...

    def fit(self, X, y, sample_weight: ArrayLike | None = None) -> "LinearGaussianCPD": ...
        # OLS

    def predict_proba(self, X: pd.DataFrame) -> "skpro.distributions.Normal": ...

    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...

    # Consumed by dag.transforms.to_joint_gaussian / LinearGaussianInference
    def get_linear_gaussian_params(self) -> tuple[np.ndarray, float]: ...
```

### `FunctionalCPD`

```python
class FunctionalCPD(BaseProbaRegressor):
    """User-defined distribution via a Pyro callable."""

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": False,
        "supports_analytic_conditioning": False,
        "supports_fit_joint": True,
        "python_dependencies": ["skpro", "pyro-ppl"],
    }

    # fn: dict[parent_name, value] → pyro.distributions.Distribution
    def __init__(self, fn: Callable | None = None, vectorized: bool = False) -> None: ...

    fn: Callable
    is_fitted_: bool

    def fit(self, X, y, sample_weight=None) -> "FunctionalCPD": ...
        # Per-node fit on tunable params (often a no-op; joint fit is
        # JointPyroEstimator's job).

    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...
    # No predict_proba — sample/log_prob are the primary entry points.
```

### Module helpers

```python
# pgmpy.parameterization.checks
class CPDContractError(TypeError): ...

def check_parameterization(obj: Any) -> None:
    """Validate that obj has fit(X, y) and predict_proba(X)."""

# pgmpy.parameterization.base
def cpd_sample(cpd, X: pd.DataFrame, n_samples=None, random_state=None) -> pd.Series:
    """Dispatch: cpd.sample → cpd.predict_proba(X).sample → class-prob draw."""

def cpd_log_prob(cpd, y: pd.Series, X: pd.DataFrame) -> pd.Series:
    """Dispatch: cpd.log_prob → predict_proba.log_pdf / log_pmf → indexed log probs."""
```

---

## `pgmpy.base.DAG`

The unified parameterized network. Replaces the v1.x typed BN classes.

```python
from functools import cached_property
from skbase.base import BaseEstimator
import networkx as nx


class DAG(_GraphRolesMixin, nx.DiGraph, BaseEstimator):
    """DAG with optional CPD parameterization.

    Inherits graph machinery from nx.DiGraph and estimator infrastructure
    (clone, get_params, _tags) from skbase.
    """

    _tags = {"object_type": "dag"}

    # Init args MUST be stored unchanged as self.<name> for skbase clone()
    # to round-trip correctly. Runtime state goes under _-prefixed names.
    def __init__(
        self,
        ebunch: Iterable[tuple] | None = None,
        latents: set[Hashable] | None = None,
    ) -> None: ...

    # Internal state
    _cpds: dict[Hashable, Any]                    # node → CPD instance
    _parent_order: dict[Hashable, list[Hashable]] # node → ordered parents

    # Accessors (same instance returned across calls)
    @cached_property
    def parameters(self) -> "_DAGParameters": ...
    @cached_property
    def transforms(self) -> "_DAGTransforms": ...
    @cached_property
    def inference(self) -> "_DAGInference": ...
    @cached_property
    def io(self) -> "_DAGIO": ...

    # Core operations
    def check_model(self) -> bool: ...
    def fit(self, data: pd.DataFrame,
            estimator: "ParameterEstimator | None" = None,
            **kwargs) -> "DAG":
        """Default estimator: MLEEstimator (per-node loop)."""
    def simulate(self, n_samples: int = 1000,
                 do: dict[Hashable, Any] | None = None,
                 seed: int | None = None) -> pd.DataFrame: ...

    @classmethod
    def load(cls, path: str, format: str = "bif") -> "DAG": ...

    # Inherited unchanged: clone(), get_params(), set_params() from skbase;
    # nodes(), edges(), predecessors(), successors(), subgraph(), ... from
    # nx.DiGraph; active_trail_nodes(), get_independencies(),
    # get_immoralities(), get_markov_blanket(), ... from existing DAG body.
```

---

## Accessor classes (`pgmpy.base._accessors`)

Internal classes accessed via `dag.parameters` / `transforms` / `inference` /
`io`. Users don't instantiate directly.

### `_DAGParameters`

```python
class _DAGParameters:
    """CPD-registry management. Canonical CPD-management API."""

    def __init__(self, dag: DAG) -> None: ...

    # Mutation
    def add(
        self, *,
        variable: Hashable,
        cpd: Any,                                # passes check_parameterization
        parent_order: list[Hashable] | None = None,
    ) -> "_DAGParameters": ...
    def remove(self, *variables: Hashable) -> "_DAGParameters": ...

    # Read
    def get(self, node: Hashable | None = None) -> Any | list[Any]: ...
    def __getitem__(self, node: Hashable) -> Any: ...
    def __contains__(self, node: Hashable) -> bool: ...

    # Dict-like introspection
    def keys(self) -> KeysView[Hashable]: ...
    def values(self) -> ValuesView[Any]: ...
    def items(self) -> ItemsView[Hashable, Any]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Hashable]: ...
```

### `_DAGTransforms`

```python
class _DAGTransforms:
    """Type-conditional transformations. Each method checks the relevant
    CPD tag at call time and raises TypeError on precondition failure."""

    def __init__(self, dag: DAG) -> None: ...

    # Requires every CPD to have produces_factor=True
    def to_markov_model(self) -> "DiscreteMarkovNetwork": ...
    def cpd_as_factor(self, node: Hashable) -> "DiscreteFactor": ...

    # Requires every CPD to have is_linear_gaussian=True
    def to_joint_gaussian(self) -> tuple[np.ndarray, np.ndarray]: ...  # (mu, cov)
```

### `_DAGInference`

```python
class _DAGInference:
    """Convenience query API. Auto-dispatches on CPD tags when method=None:
      - all produces_factor=True   → VariableElimination
      - all is_linear_gaussian=True → LinearGaussianInference
      - otherwise                   → LikelihoodWeighting
    """

    def __init__(self, dag: DAG) -> None: ...

    def predict(self, data: pd.DataFrame, method: str | None = None,
                **kwargs) -> pd.DataFrame: ...
    def predict_probability(self, data: pd.DataFrame, method: str | None = None,
                            **kwargs) -> pd.DataFrame: ...
    def log_likelihood(self, data: pd.DataFrame) -> float: ...
```

### `_DAGIO`

```python
class _DAGIO:
    """Serialization. Load is a DAG classmethod (no DAG exists before loading)."""

    def __init__(self, dag: DAG) -> None: ...
    def save(self, path: str,
             format: Literal["bif", "xmlbif", "uai"] = "bif") -> None: ...
```

---

## `pgmpy.parameter_estimator`

Network-level fitting algorithms. Per-CPD fitting policy (e.g., Bayesian
priors for `TabularCPD`) is configured via the CPD's `__init__`
hyperparameters, not a separate estimator.

### `ParameterEstimator` (base)

```python
class ParameterEstimator:
    """fit(model, data, **kwargs) mutates the model's CPDs in place."""

    _tags = {"supports_weighted_data": False, "supports_latent_variables": False}

    def fit(self, model: DAG, data: pd.DataFrame, **kwargs) -> DAG: ...
```

### `MLEEstimator`

```python
class MLEEstimator(ParameterEstimator):
    """Default. Walks topological order, calls cpd.fit(X, y) per node.

    If a CPD has prior_type configured (e.g., TabularCPD(prior_type="BDeu")),
    that CPD applies the prior internally — no special handling here.
    """

    _tags = {"supports_weighted_data": True}

    def fit(self, model, data, sample_weight=None) -> DAG: ...
```

### `DiscreteEM`

```python
class DiscreteEM(ParameterEstimator):
    """EM for discrete BNs with latent variables.

    E-step: VariableElimination per row → expected counts.
    M-step: cpd.fit(X, y, sample_weight=expected_counts).
    """

    _tags = {"supports_latent_variables": True}

    def __init__(self, latent_variables: Iterable[Hashable],
                 max_iter: int = 100, tol: float = 1e-4) -> None: ...
    def fit(self, model, data, sample_weight=None) -> DAG: ...
```

### `JointPyroEstimator`

```python
class JointPyroEstimator(ParameterEstimator):
    """Joint SVI/MCMC for FunctionalCPD networks.

    Precondition: every CPD has supports_fit_joint=True.
    """

    def __init__(
        self,
        estimator: Literal["SVI", "MCMC"] = "SVI",
        num_steps: int = 1000,
        optimizer: "pyro.optim.PyroOptim | None" = None,
        prior_fn: Callable | None = None,
        nuts_kwargs: dict | None = None,
        mcmc_kwargs: dict | None = None,
        seed: int | None = None,
    ) -> None: ...

    def fit(self, model, data) -> dict:
        """Returns the fitted param store (SVI) or posterior samples (MCMC)."""
```

> **Note on omitted classes:** There is no separate `DiscreteMLE`,
> `LinearGaussianMLE`, or `DiscreteBayesianEstimator` class.
>
> - `DiscreteMLE` / `LinearGaussianMLE` would be `MLEEstimator` subclasses
>   that just add a type-validation precondition. Skipping them; users get
>   the precondition automatically when `MLEEstimator` calls `cpd.fit(X, y)`
>   on a type-mismatched CPD (the CPD itself errors out).
> - `DiscreteBayesianEstimator` is unnecessary because Bayesian fitting is
>   configured via `TabularCPD(prior_type="BDeu", ...)` hyperparameters.
>   `dag.fit(data)` with the default `MLEEstimator` does Bayesian fitting
>   automatically on CPDs that declare a prior.

---

## `pgmpy.inference`

All inference algorithms accept any `DAG`. Each checks CPD tags at
construction and raises `TypeError` if requirements aren't met.

### Existing classes (v1.x signatures preserved)

`VariableElimination`, `BeliefPropagation`, and `ApproxInference` keep
their v1.x public surface unchanged — same `__init__(model)`,
`query(...)`, `map_query(...)`, etc. The only change is that `model`
relaxes to any `DAG` and tag-based preconditions replace `isinstance`
checks:

- `VariableElimination(dag)`, `BeliefPropagation(dag)` — require every CPD
  to advertise `produces_factor=True`.
- `ApproxInference(dag)` — rejection sampling; works for discrete evidence
  on any DAG.

For full method signatures see the v1.x docstrings; they are unchanged
in v2.0.

### `LinearGaussianInference` (new)

```python
class LinearGaussianInference:
    """Exact inference on linear-Gaussian networks via joint Gaussian + Schur complement.

    Precondition: every CPD has is_linear_gaussian=True.
    """

    def __init__(self, model: DAG) -> None: ...

    def query(self, variables: list[Hashable],
              evidence: dict[Hashable, float] | None = None,
              ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mu, cov) of P(variables | evidence)."""

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Per-row MAP prediction of missing columns."""
```

### `LikelihoodWeighting` (new)

```python
class LikelihoodWeighting:
    """Importance sampling via per-CPD log_prob weighting.

    Works on any DAG whose CPDs support cpd_sample / cpd_log_prob —
    including third-party skpro / sklearn estimators.

    For continuous evidence ApproxInference's rejection sampling
    degenerates; use LikelihoodWeighting instead.
    """

    def __init__(self, model: DAG) -> None: ...

    def weighted_sample(self, evidence=None, n_samples: int = 10_000,
                        seed: int | None = None) -> pd.DataFrame:
        """Returns DataFrame with n_samples rows + a '_weight' column."""

    def query(self, variables, evidence=None, n_samples: int = 10_000,
              seed: int | None = None) -> dict[Hashable, Any]:
        """Discrete vars → {state: prob}; continuous → weighted samples."""

    def predict(self, data: pd.DataFrame, n_samples: int = 10_000,
                seed: int | None = None) -> pd.DataFrame: ...
```

---

## Capability tags reference

| Tag | Type | Purpose |
|---|---|---|
| `variable_type` | `"discrete" \| "continuous"` | `_DAGInference.predict` auto-dispatch. |
| `produces_factor` | `bool` | Required by `VariableElimination`, `BeliefPropagation`, `dag.transforms.to_markov_model`. |
| `is_linear_gaussian` | `bool` | Required by `LinearGaussianInference`, `dag.transforms.to_joint_gaussian`. |
| `supports_analytic_conditioning` | `bool` | Informational; some CPDs support analytic Bayes inversion. |
| `supports_fit_joint` | `bool` | Required by `JointPyroEstimator`. |
| `python_dependencies` | `list[str]` | Soft-dep gating via `_check_soft_dependencies`. |

| Tag | `TabularCPD` | `LinearGaussianCPD` | `FunctionalCPD` |
|---|:-:|:-:|:-:|
| `variable_type` | `"discrete"` | `"continuous"` | `"continuous"` |
| `produces_factor` | ✓ | ✗ | ✗ |
| `is_linear_gaussian` | ✗ | ✓ | ✗ |
| `supports_analytic_conditioning` | ✓ | ✓ | ✗ |
| `supports_fit_joint` | ✗ | ✗ | ✓ |

Third-party CPDs read tags via `cpd.get_tag(name, default)` (provided by
skbase). Tags default to safe values based on the upstream base class:

- `sklearn.base.ClassifierMixin` → `variable_type="discrete"`
- `skpro.regression.base.BaseProbaRegressor` → `variable_type="continuous"`

Other tags default to `False` for third-party CPDs, routing them to
`LikelihoodWeighting` for inference.
