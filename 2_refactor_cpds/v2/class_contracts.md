# pgmpy v2.0 Class Contracts

Companion to `refactor_design.md` (overview) and the split design docs:
`01_cpd_boundary.md`, `02_parameterized_dag.md`,
`03_scm_counterfactuals.md`. Lists **only** the public contracts —
class declarations, tags, fitted attributes, method signatures.

At the time of the split, this file remains a **combined reference across
all three sub-proposals**. It is intentionally broader than any single
design doc and should not be read as forcing Designs 1, 2, and 3 to be
accepted together.

**Core principles** (see design doc for rationale):

- `DAG` is a pgmpy-native graph/model container, not a sklearn/skbase
  estimator. No `get_params`/`set_params`/`clone`; refit/bootstrap use
  `dag.copy_template(parameters="none"|"unfit"|"fitted")`.
- CPDs are sklearn/skpro-style estimators: `__init__` hyperparameters,
  `fit(X, y)`, fitted attributes, per-CPD tags, CPD-level cloning.
- CPD contract is split into operation-specific protocols (`FittableCPD`,
  `SampleableCPD`, `ScorableCPD`, `PredictiveCPD`) plus optional
  `StructuralCPD` for counterfactual reasoning. Operations check only
  the protocol surface they consume.
- `dag.schema` is an internal DAG-owned registry, populated automatically
  from CPDs + data + pandas categoricals; manual `dag.schema.set(...)`
  is an escape hatch. Precedence policy in design doc.
- `dag.parameters.add(...)` validates graph membership, parent-order
  equality, schema/CPD type compatibility, and tabular cardinality/state
  compatibility.
- Discrete counterfactual semantics are explicit (`noise_repr`). Inverse-CDF
  is the v2.0 default; Gumbel-Max is opt-in (deferred to v2.x).

---

## `pgmpy.parameterization`

Identity-free CPDs. Inherit from skbase/skpro/sklearn base classes; no
`variable` / `evidence` attributes (identity lives on the DAG).

### `StructuralCPD` protocol

Optional protocol for CPDs that expose explicit exogenous noise, enabling
Pearl's abduction-action-prediction. A CPD opts in by implementing the
three methods and advertising `supports_counterfactual=True`.

```python
from typing import Protocol


class NoiseDistribution(Protocol):
    """Minimal sampling distribution interface (concrete types:
    Delta, NormalNoise, Empirical, TruncatedUniform)."""
    def sample(self, n: int = 1, random_state=None) -> np.ndarray: ...
    def point(self): ...    # mean / mode / single value


class StructuralCPD(Protocol):
    """CPDs that expose explicit exogenous noise: p(X | pa) = ∫δ(X − f(pa, U)) p(U) dU."""

    def noise_prior(self) -> NoiseDistribution:
        """Prior p(U) over the exogenous noise."""

    def structural_predict(self, parents: pd.DataFrame, noise: ArrayLike) -> pd.Series:
        """Deterministic forward pass: X = f(parents, noise)."""

    def abduct(self, x: pd.Series, parents: pd.DataFrame) -> NoiseDistribution:
        """Recover noise U that explains (X=x, PA=parents).

        Invertible CPDs (LG, ANM, PNL) → Delta (point). Non-invertible
        CPDs (Tabular inverse-CDF, Functional) → real distribution.
        The counterfactual algorithm calls .sample(n) uniformly.
        """
```

CPDs that don't implement the protocol simply opt out;
`dag.counterfactual.query(...)` raises `IncompatibleCPDError` listing
offending nodes.

### `QueryResult`

Rich return type for `dag.{inference,intervene,counterfactual,bootstrap}.query`.
Distribution-valued by construction so non-invertible noise,
likelihood-weighted inference, Bayesian-regressor parameter uncertainty,
and bootstrap-over-fit uncertainty all share one result vocabulary.

```python
from dataclasses import dataclass, field


@dataclass
class QueryResult:
    samples: np.ndarray          # (n_samples,) for scalar; (n_samples, k) for list
    query: str | list[str]
    operation: str               # "predict" | "intervene" | "counterfact"
    operation_args: dict         # e.g. {"do": ..., "observed": ...}
    meta: dict = field(default_factory=dict)

    def point(self) -> float: ...
    def distribution(self) -> pd.Series: ...
    def expectation(self, fn) -> float: ...
    def credible_interval(self, level: float = 0.95) -> tuple[float, float]: ...
    def compare_to(self, other) -> dict:
        """Pairwise comparison (Wasserstein-1 + |Δmean|). Used for
        cross-noise-representation robustness and A/B sensitivity."""
```

Convenience tabular/batch APIs (`dag.inference.predict`,
`dag.intervene.simulate`) may still return pandas objects for ergonomics.

### `TabularCPD`

```python
from sklearn.base import ClassifierMixin
from skbase.base import BaseEstimator


class TabularCPD(ClassifierMixin, BaseEstimator):
    """Discrete categorical CPT.

    Bayesian fitting is configured via __init__ hyperparameters
    (prior_type, equivalent_sample_size). EM fitting is orchestrated
    at the network level by DiscreteEM.
    """

    _tags = {
        "variable_type": "discrete",
        "produces_factor": True,
        "supports_fit_joint": False,
        # SCM/counterfactual via inverse-CDF: X = F^{-1}(P(X|pa), U),
        # U ~ Uniform[0, 1]. Abduction returns TruncatedUniform on the
        # cumulative-probability bracket. Pluggable via noise_repr.
        "supports_counterfactual": True,
        "noise_type": "inverse_cdf",
        "python_dependencies": ["scikit-learn"],
    }

    def __init__(
        self,
        variable_card: int,
        evidence_card: list[int] | None = None,
        state_names: list[list] | None = None,
        prior_type: Literal["BDeu", "K2", "dirichlet"] | None = None,
        equivalent_sample_size: float = 10,
        pseudo_counts: ArrayLike | None = None,
        noise_repr: Literal["inverse_cdf", "gumbel_max"] = "inverse_cdf",
    ) -> None: ...

    # Fitted attributes (sklearn convention — trailing underscore)
    values_: np.ndarray         # shape (variable_card, prod(evidence_card))
    classes_: np.ndarray        # child state labels
    is_fitted_: bool

    @classmethod
    def from_values(cls, variable_card: int, values: ArrayLike,
                    evidence_card: list[int] | None = None,
                    state_names: list[list] | None = None) -> "TabularCPD": ...

    def fit(self, X: pd.DataFrame, y: pd.Series,
            sample_weight: ArrayLike | None = None) -> "TabularCPD":
        """ML (or MAP if prior_type is set) estimate of the CPT."""

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame: ...
        # shape (len(X), variable_card), columns = classes_
    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...

    # StructuralCPD protocol (inverse-CDF; Gumbel-Max is v2.x).
    def noise_prior(self) -> NoiseDistribution: ...
        # TruncatedUniform(low=[0.], high=[1.])
    def structural_predict(self, parents, noise) -> pd.Series:
        """X = first k where F(k|pa) >= noise."""
    def abduct(self, x, parents) -> NoiseDistribution:
        """TruncatedUniform(low=F(x-1|pa), high=F(x|pa)) per row."""
```

### `LinearGaussianCPD`

```python
from skpro.regression.base import BaseProbaRegressor


class LinearGaussianCPD(BaseProbaRegressor):
    """P(y | X) = N(beta_[0] + X @ beta_[1:], std_).

    Canonical additive noise model: X = β·[1, pa] + U, U ~ N(0, std_).
    StructuralCPD with point-invertible abduction.
    """

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "is_linear_gaussian": True,
        "supports_fit_joint": False,
        "supports_counterfactual": True,
        "noise_type": "additive",
        "python_dependencies": ["skpro"],
    }

    def __init__(self) -> None: ...

    # Fitted attributes (skpro convention — _is_fitted property)
    beta_: np.ndarray           # length n_parents + 1; beta_[0] = intercept
    std_: float                 # residual std

    @classmethod
    def from_values(cls, beta: ArrayLike, std: float) -> "LinearGaussianCPD": ...

    def fit(self, X, y, sample_weight: ArrayLike | None = None) -> "LinearGaussianCPD": ...
    def predict_proba(self, X: pd.DataFrame) -> "skpro.distributions.Normal": ...
    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...

    # Consumed by dag.transforms.to_joint_gaussian
    def get_linear_gaussian_params(self) -> tuple[np.ndarray, float]: ...

    # StructuralCPD protocol
    def noise_prior(self) -> NoiseDistribution: ...           # NormalNoise(0.0, std_)
    def structural_predict(self, parents, noise) -> pd.Series:
        """X = beta_[0] + parents @ beta_[1:] + noise."""
    def abduct(self, x, parents) -> NoiseDistribution:
        """Delta(value = x − (beta_[0] + parents @ beta_[1:]))."""
```

### `FunctionalCPD`

```python
class FunctionalCPD(BaseProbaRegressor):
    """User-defined distribution via a Pyro callable. SCM-native: noise is
    a pyro.sample site, structural function is the model body."""

    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "supports_fit_joint": True,
        "supports_counterfactual": True,
        "noise_type": "custom",
        "python_dependencies": ["skpro", "pyro-ppl"],
    }

    # fn: dict[parent_name, value] → pyro.distributions.Distribution
    def __init__(self, fn: Callable | None = None, vectorized: bool = False) -> None: ...

    fn: Callable
    is_fitted_: bool

    def fit(self, X, y, sample_weight=None) -> "FunctionalCPD":
        """Per-node fit on tunable params (often no-op; joint fit is
        JointPyroEstimator's job)."""
    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...
    # No predict_proba — sample/log_prob are the primary entry points.

    # StructuralCPD protocol (Pyro poutines wrapped as NoiseDistribution).
    def noise_prior(self) -> NoiseDistribution: ...
    def structural_predict(self, parents, noise) -> pd.Series: ...
    def abduct(self, x, parents) -> NoiseDistribution:
        """pyro.poutine.condition + replay wrapped as a NoiseDistribution."""
```

### `WrappedRegressor`

Single adapter for ANM + PNL via composition (merges what was
`ANMWrapper` + `PNLWrapper`). With `link=None`, an additive noise model;
with `link`/`link_inv`, post-nonlinear. Improves on
`dowhy.gcm.{AdditiveNoiseModel, PostNonlinearModel}` by collapsing the
hierarchy, making `noise_dist` a first-class slot, and propagating
parameter uncertainty automatically when the wrapped regressor is
Bayesian (`BayesianRidge`, NGBoost, MAPIE, etc.).

```python
class WrappedRegressor(BaseProbaRegressor):
    _tags = {
        "variable_type": "continuous",
        "produces_factor": False,
        "supports_counterfactual": True,
        "noise_type": "additive",   # per-instance: "post_nonlinear" if link given
    }

    def __init__(
        self,
        regressor: Any,                              # any .fit / .predict object
        *,
        link: Callable | None = None,                # e.g. np.tanh; None → ANM
        link_inv: Callable | None = None,            # required if link given
        noise_dist: NoiseDistribution | None = None, # None → Empirical(residuals)
    ) -> None: ...

    # Fitted attributes
    regressor_: Any
    noise_dist_: NoiseDistribution
    noise_residuals_: np.ndarray          # residuals on link_inv scale

    def fit(self, X, y) -> "WrappedRegressor": ...

    # StructuralCPD protocol
    def noise_prior(self) -> NoiseDistribution: ...
    def structural_predict(self, parents, noise) -> pd.Series:
        """link(reg.predict(pa) + u) if link else reg.predict(pa) + u."""
    def abduct(self, x, parents) -> NoiseDistribution:
        """Point: Delta(link_inv(x) - reg.predict(pa))  (identity link_inv if None).
        Bayesian regressor: Empirical(samples)."""
```

**Use-site:**

```python
WrappedRegressor(GradientBoostingRegressor())                            # ANM
WrappedRegressor(LinearRegression(), link=np.tanh, link_inv=np.arctanh)  # PNL
WrappedRegressor(GLM(), noise_dist=Empirical(historical_residuals))      # heavy-tailed ANM
WrappedRegressor(BayesianRidge())                                        # distribution-valued counterfactual
```

### Module helpers

```python
# pgmpy.parameterization.checks
class CPDContractError(TypeError): ...
class IncompatibleCPDError(TypeError): ...

class FittableCPD(Protocol):
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs): ...

class SampleableCPD(Protocol):
    def sample(self, X: pd.DataFrame, n_samples: int = 1) -> pd.Series: ...

class ScorableCPD(Protocol):
    def log_prob(self, y: pd.Series, X: pd.DataFrame) -> pd.Series: ...

class PredictiveCPD(Protocol):
    def predict_proba(self, X: pd.DataFrame): ...

def require_fittable(obj: Any) -> None: ...
def require_sampleable(obj: Any) -> None: ...
def require_scorable(obj: Any) -> None: ...
def require_predictive(obj: Any) -> None: ...

def check_parameterization(obj: Any) -> None:
    """Compatibility validator for third-party predictive CPDs:
    FittableCPD + PredictiveCPD. Built-in CPDs satisfy narrower
    protocols depending on the operation."""
```

### `CPDAdapter`

Adapts any object with `fit(X, y)` + `predict_proba(X)` to the full pgmpy
CPD interface. `_DAGParameters.add()` auto-wraps third-party objects at
registration; built-in CPDs (`TabularCPD`, `LinearGaussianCPD`,
`WrappedRegressor`) pass through unwrapped.

```python
class CPDAdapter:
    """Make any predict_proba-style object satisfy the full CPD contract."""

    def __init__(self, wrapped: Any) -> None: ...
    @property
    def wrapped(self) -> Any: ...
    def __getattr__(self, name): ...           # forward to wrapped

    def fit(self, X, y, **kwargs): ...
    def predict_proba(self, X): ...
    def sample(self, X, n_samples=None) -> pd.Series: ...
    def log_prob(self, y, X) -> pd.Series: ...
    def get_tag(self, name, default=None): ...
    def clone(self) -> "CPDAdapter": ...
```

Replaces the earlier module-level `_dispatch_sample`/`_dispatch_log_prob`
helpers.

### Noise distributions

`pgmpy.parameterization.noise` — concrete `NoiseDistribution` types:

```python
@dataclass
class Delta:
    """Degenerate point distribution. Returned by abduct() for invertible CPDs."""
    value: np.ndarray
    def sample(self, n=1, random_state=None): ...
    def point(self): ...

@dataclass
class NormalNoise:
    mu: float = 0.0
    sigma: float = 1.0
    def sample(self, n=1, random_state=None): ...
    def point(self): ...

@dataclass
class Empirical:
    """Resamples training residuals or any supplied array."""
    samples: np.ndarray
    def sample(self, n=1, random_state=None): ...
    def point(self): ...

@dataclass
class TruncatedUniform:
    """1-D truncated uniform per row. Vectorised."""
    low: np.ndarray
    high: np.ndarray
    def sample(self, n=1, random_state=None): ...
    def point(self): ...
```

Third-party CPDs may return any object satisfying the `NoiseDistribution`
protocol (`.sample(n, random_state)` + `.point()`).

---

## `pgmpy.base.DAG`

```python
from functools import cached_property
import networkx as nx


class DAG(_GraphRolesMixin, nx.DiGraph):
    """DAG with optional CPD parameterization.

    `_GraphRolesMixin` is the existing causal-role annotation mixin in
    `pgmpy/base/_mixin_roles.py` (`latents`, `observed`, `exposures`,
    `outcomes`, role accessors, `is_valid_causal_structure`). All graph
    methods (`active_trail_nodes`, `get_independencies`,
    `get_immoralities`, `get_markov_blanket`, `get_ancestors`, `copy`,
    `with_intervention` *(formerly `do`)*, …) live directly on this
    class and survive the v2.0 refactor unchanged. Deliberately not a
    skbase/sklearn estimator.
    """

    def __init__(
        self,
        ebunch: Iterable[tuple] | None = None,
        latents: set[Hashable] | None = None,
    ) -> None: ...

    # Internal state
    _cpds: dict[Hashable, Any]                    # node → CPD instance
    _parent_order: dict[Hashable, list[Hashable]] # node → ordered parents
    _schema: dict[Hashable, "VariableSchema"]     # node → inferred metadata

    # Accessors (cached_property — same instance across calls).
    # Pearl-rung-mapped:
    @cached_property
    def parameters(self) -> "_DAGParameters": ...
    @cached_property
    def schema(self) -> "_DAGSchema": ...
    @cached_property
    def inference(self) -> "_DAGInference": ...           # rung 1
    @cached_property
    def intervene(self) -> "_DAGIntervene": ...           # rung 2
    @cached_property
    def counterfactual(self) -> "_DAGCounterfactual": ... # rung 3
    @cached_property
    def transforms(self) -> "_DAGTransforms": ...
    @cached_property
    def diagnostics(self) -> "_DAGDiagnostics": ...
    @cached_property
    def bootstrap(self) -> "_DAGBootstrap": ...
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
    def copy_template(
        self,
        *,
        parameters: Literal["none", "unfit", "fitted"] = "unfit",
    ) -> "DAG":
        """Copy graph + schema, optionally cloning CPD specs or deep-copying fitted CPDs.
        Used by bootstrap and refit workflows in place of estimator clone."""

    @classmethod
    def load(cls, path: str, format: str = "bif") -> "DAG": ...

    # Already on this class (unchanged from v1.x — see pgmpy/base/DAG.py):
    #   active_trail_nodes(), get_independencies(), get_immoralities(),
    #   get_markov_blanket(), get_ancestors(), minimal_dseparator(),
    #   moralize(), copy(), with_intervention() (formerly do()).
    # From _GraphRolesMixin: latents/observed/exposures/outcomes roles.
    # From nx.DiGraph: nodes()/edges()/predecessors()/successors()/subgraph().
```

---

## Accessor classes (`pgmpy.base._accessors`)

Internal classes accessed via `dag.<accessor>`. Users don't instantiate
directly.

### `_DAGSchema`

```python
@dataclass(frozen=True)
class VariableSchema:
    variable: Hashable
    variable_type: Literal["discrete", "continuous"]
    states: tuple | None = None
    dtype: Any | None = None
    ordered: bool = False
    encoder: Any | None = None
    decoder: Any | None = None


class _DAGSchema:
    """Internal schema registry.

    Schema is collected from multiple sources with explicit precedence
    (highest authority first):
      1. dag.schema.set(...) user override.
      2. CPD-authored state_names (via infer_from_cpd).
      3. Pandas categorical metadata (via infer_from_data).
      4. Observed data uniques, sorted(key=str) (via infer_from_data).
      5. CPD cardinality fallback (tuple(range(variable_card))).

    Higher-authority state orderings are preserved; lower sources only
    validate that observed values are a subset of declared states. See
    design doc "Schema precedence policy".
    """

    def __init__(self, dag: DAG) -> None: ...
    def __getitem__(self, variable: Hashable) -> VariableSchema: ...
    def get(self, variable: Hashable, default=None) -> VariableSchema | None: ...
    def items(self) -> ItemsView[Hashable, VariableSchema]: ...

    # Advanced override (escape hatch).
    def set(self, variable: Hashable, *, variable_type, states=None,
            dtype=None, ordered=False, encoder=None, decoder=None) -> "_DAGSchema": ...

    # Internal inference hooks.
    def infer_from_cpd(self, variable, cpd, parent_order) -> "_DAGSchema": ...
    def infer_from_data(self, data: pd.DataFrame) -> "_DAGSchema": ...
```

### `_DAGParameters`

```python
class _DAGParameters:
    """CPD-registry management."""

    def __init__(self, dag: DAG) -> None: ...

    def add(self, *, variable: Hashable, cpd: Any,
            parent_order: list[Hashable] | None = None) -> "_DAGParameters":
        """Auto-wraps cpd in CPDAdapter if it doesn't natively implement
        sample(X, n_samples) + log_prob(y, X). Validates graph membership,
        parent-order equality, schema/CPD type compatibility, and tabular
        cardinality/state compatibility. Normalizes implied state names
        into dag.schema."""
    def remove(self, *variables: Hashable) -> "_DAGParameters": ...

    def get(self, node: Hashable | None = None) -> Any | list[Any]: ...
    def __getitem__(self, node: Hashable) -> Any: ...
    def __contains__(self, node: Hashable) -> bool: ...

    # Dict-like
    def keys(self) -> KeysView[Hashable]: ...
    def values(self) -> ValuesView[Any]: ...
    def items(self) -> ItemsView[Hashable, Any]: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Hashable]: ...
```

### `_DAGTransforms`

```python
class _DAGTransforms:
    """Graph primitives + type-conditional transformations."""

    def __init__(self, dag: DAG) -> None: ...

    # Universal graph primitives.
    def ancestors(self, node: Hashable) -> set: ...
    def descendants(self, node: Hashable) -> set: ...
    def topological_order(self) -> list: ...
    def markov_blanket(self, node: Hashable) -> set: ...
    def d_separated(self, x: Hashable, y: Hashable,
                     given: set | None = None) -> bool: ...

    # Requires produces_factor=True.
    def to_markov_model(self) -> "DiscreteMarkovNetwork": ...
    def cpd_as_factor(self, node: Hashable) -> "DiscreteFactor": ...

    # Requires is_linear_gaussian=True.
    def to_joint_gaussian(self) -> tuple[np.ndarray, np.ndarray]: ...
```

### `_DAGInference` — Pearl rung 1 (associational)

```python
class _DAGInference:
    """P(query | evidence). Auto-dispatches:
      - all produces_factor=True → VariableElimination
      - otherwise                → LikelihoodWeighting
    Both return QueryResult.
    """

    def __init__(self, dag: DAG) -> None: ...

    def query(
        self,
        evidence: dict[Hashable, Any] | None = None,
        *, query: Hashable, n_samples: int = 10000, seed: int | None = None,
    ) -> QueryResult: ...

    # Convenience batch APIs returning DataFrame.
    def predict(self, data: pd.DataFrame,
                query: list[Hashable] | None = None, **kwargs) -> pd.DataFrame: ...
    def log_likelihood(self, data: pd.DataFrame) -> float: ...
```

### `_DAGIntervene` — Pearl rung 2 (interventional)

Replaces the implicit `dag.simulate(do=...)` overloading.

```python
class _DAGIntervene:
    """P(query | do(...)). Forward-samples under intervention (severs
    edges into do-variables); no reweighting."""

    def __init__(self, dag: DAG) -> None: ...

    def simulate(self, do: dict[Hashable, Any], n_samples: int = 1000,
                 seed: int | None = None) -> pd.DataFrame:
        """Thin alias for dag.simulate(do=...)."""
    def query(
        self,
        do: dict[Hashable, Any],
        *,
        query: Hashable,
        evidence: dict[Hashable, Any] | None = None,
        adjustment_set: Iterable[Hashable] | None = None,
        inference_algo: Literal["ve", "bp", "lw"] | "Inference" = "ve",
        n_samples: int = 10000,
        seed: int | None = None,
    ) -> QueryResult:
        """P(query | do(...), evidence) → QueryResult.

        Subsumes the legacy rung-2 surface:
          * Forward sampling under intervention when adjustment_set is None.
          * Back-door / front-door adjustment when adjustment_set is provided.
            Auto-derived from do-variable parents if omitted and evidence is
            given (matches `pgmpy.inference.CausalInference.query` semantics).
            Computed via VE/BP for `produces_factor=True` networks; falls back
            to LikelihoodWeighting otherwise.

        Replaces `pgmpy.inference.CausalInference.query(do=..., adjustment_set=...)`.
        """
```

### `_DAGCounterfactual` — Pearl rung 3 (counterfactual)

```python
class _DAGCounterfactual:
    """Pearl's abduction-action-prediction over StructuralCPDs."""

    def __init__(self, dag: DAG) -> None: ...

    def query(
        self,
        observed: dict[Hashable, Any],
        do: dict[Hashable, Any] | list[dict[Hashable, Any]],
        query: Hashable | list[Hashable],
        n_samples: int | None = None,
        noise_overrides: dict[Hashable, str] | None = None,
        seed: int | None = None,
    ) -> "QueryResult | list[QueryResult]":
        """P(query | do, observed). Algorithm:
          1. Capability check: every node has supports_counterfactual=True.
          2. Abduction: cpd.abduct(observed[node], observed[parents]) per node.
             Invertible CPDs → Delta; non-invertible → noise posterior (n_samples).
          3. Action: override structural function for each node in `do`.
          4. Prediction: structural_predict with abducted noise.

        If `do` is a list, all interventions share the same abducted noise
        (Pearl twin/parallel-world semantics, ChiRho-style); returns
        list[QueryResult] of the same length.

        `n_samples` default: 1 if all invertible + no Bayesian regressors;
        1000 otherwise.

        `noise_overrides` (v2.0): plumbing only. Raises NotImplementedError
        for non-default noise_repr; activates when Gumbel-Max ships in v2.x.
        Use case: cross-noise-representation robustness via
        result_a.compare_to(result_b).
        """

    def explain(self, observed: dict, do: dict) -> dict:
        """Abducted noise per node, for diagnostics."""
```

### `_DAGDiagnostics`

```python
class _DAGDiagnostics:
    """Static SCM and inference diagnostics. Future home for residual
    analysis, structure adequacy, fit-quality."""

    def __init__(self, dag: DAG) -> None: ...

    def identifiability_report(
        self,
        query_type: Literal["counterfactual", "interventional", "associational"]
            = "counterfactual",
    ) -> dict:
        """Flag known non-identification patterns.

        - counterfactual: pure linear-Gaussian sub-graphs (Hoyer 2009);
          discrete nodes with non-invertible noise (Oberst & Sontag 2019).
        - interventional / associational: v2.x.

        Returns: {"warnings": [{"type", "nodes", "ref", "message"}, ...],
                  "n_warnings": int}
        """

    # Adjustment-set / identification helpers — port of v1.x
    # pgmpy.inference.CausalInference (deleted in 2.0). Same signatures
    # and return types; just accessed through dag.diagnostics. See plan
    # Task 36a for the per-method porting work.
    def identification_method(self, X, Y) -> str: ...
    def get_all_backdoor_adjustment_sets(self, X, Y): ...
    def is_valid_backdoor_adjustment_set(self, X, Y, Z=[]) -> bool: ...
    def get_all_frontdoor_adjustment_sets(self, X, Y): ...
    def is_valid_frontdoor_adjustment_set(self, X, Y, Z=None) -> bool: ...
    def get_minimal_adjustment_set(self, X, Y): ...
    def is_valid_adjustment_set(self, X, Y, adjustment_set) -> bool: ...
    def get_proper_backdoor_graph(self, X, Y, inplace=False): ...
    def get_ivs(self, X, Y, scaling_indicators=None): ...
    def get_conditional_ivs(self, X, Y, scaling_indicators=None): ...
    def get_total_conditional_ivs(self, X, Y, scaling_indicators=None): ...
    def get_scaling_indicators(self): ...
    def estimate_ate(self, X, Y, data, estimator_type="linear",
                      adjustment_set=None, **kwargs) -> float: ...

    # Future v2.x:
    # def residuals(self, node=None) -> pd.DataFrame: ...
    # def structure_report(self) -> dict: ...
```

### `_DAGBootstrap`

Bootstrap-over-fit confidence intervals. Resamples data, refits a fresh
DAG via `copy_template(parameters="unfit")`, runs the user-supplied
query, aggregates into a `QueryResult`. Inspired by
[`dowhy.gcm.bootstrap_sampling`](https://www.pywhy.org/dowhy/v0.9.1/user_guide/gcm_based_inference/estimating_confidence_intervals.html).

**Limitation**: depends on `copy_template`, which clones each CPD via
(in order) `cpd.clone()` → `sklearn.base.clone(cpd)` → `type(cpd)()`.
v2.0 supports `TabularCPD` / `LinearGaussianCPD` / `WrappedRegressor` /
sklearn / skpro estimators with a working clone. `FunctionalCPD`
bootstrap is deferred to v2.x (needs `fn`-aware clone or
build-from-config pathway).

```python
class _DAGBootstrap:
    def __init__(self, dag: DAG) -> None: ...

    def query(
        self,
        data: pd.DataFrame,
        query_fn: Callable[[DAG], QueryResult],
        n_bootstrap: int = 200,
        seed: int | None = None,
    ) -> QueryResult:
        """Each iter: resample data → copy_template(parameters="unfit") →
        fit → query_fn(fitted_dag). Aggregate per-iter samples.

        Example:
            dag.bootstrap.query(
                data=data,
                query_fn=lambda d: d.counterfactual.query(
                    observed=..., do={...}, query="Z"),
                n_bootstrap=200,
            )
        """
```

### `_DAGIO`

```python
class _DAGIO:
    """Serialization. Load is a DAG classmethod (no DAG exists pre-load)."""

    def __init__(self, dag: DAG) -> None: ...
    def save(self, path: str,
             format: Literal["bif", "xmlbif", "uai"] = "bif") -> None: ...
```

---

## `pgmpy.parameter_estimator`

Network-level fitting algorithms. Per-CPD policy (e.g. Bayesian priors)
is configured on the CPD; the estimator orchestrates across the graph.

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
    """Walks topological order, calls cpd.fit(X, y). If a CPD has
    prior_type configured (TabularCPD(prior_type="BDeu")), that CPD
    applies the prior internally — no special handling here.
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
    Precondition: every CPD has supports_fit_joint=True."""

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
        """Returns fitted param store (SVI) or posterior samples (MCMC)."""
```

> **Note on omitted classes:** No separate `DiscreteMLE`,
> `LinearGaussianMLE`, or `DiscreteBayesianEstimator`. The first two
> would be `MLEEstimator` subclasses adding type-validation
> preconditions — users get those automatically when `MLEEstimator`
> calls `cpd.fit(X, y)` on a type-mismatched CPD. The third is
> unnecessary because Bayesian fitting is configured via
> `TabularCPD(prior_type="BDeu", ...)`.

---

## `pgmpy.inference`

All inference algorithms accept any `DAG`. Each checks CPD tags at
construction and raises `TypeError` on requirement failure.

### Existing classes (v1.x signatures preserved)

`VariableElimination`, `BeliefPropagation`, `ApproxInference` keep their
v1.x public surface unchanged (same `__init__(model)`, `query(...)`,
`map_query(...)`) and return their v1.x types (`DiscreteFactor`,
sample dicts, etc.) — **not** `QueryResult`. The new accessor surface
(`_DAGInference.query`, `_DAGIntervene.query`) wraps these underlying
returns into `QueryResult`.

The only behavioural change is at construction: `model` relaxes from
`DiscreteBayesianNetwork | DynamicBayesianNetwork` to any `DAG`, and
tag-based preconditions replace `isinstance` checks:

- `VariableElimination` / `BeliefPropagation` — every CPD must advertise `produces_factor=True`.
- `ApproxInference` — current v1.x impl rejects non-discrete networks at `pgmpy/inference/ApproxInference.py:24`; **Phase 3 Task 35** relaxes this so it accepts any `DAG` with discrete evidence. After relaxation: rejection sampling on forward samples; degenerates on continuous evidence (use `LikelihoodWeighting` for that).

### Removed: `pgmpy.inference.CausalInference`

The v1.x `CausalInference` class (1078 lines covering backdoor / frontdoor
/ IV / adjustment / ATE estimation) is replaced by:

- `dag.intervene.query(do=..., evidence=..., adjustment_set=...)` for the
  rung-2 query surface (subsumes `CausalInference.query`).
- `dag.diagnostics` for the identification / adjustment-set discovery
  helpers (`identification_method`, `get_all_backdoor_adjustment_sets`,
  `is_valid_backdoor_adjustment_set`, `get_all_frontdoor_adjustment_sets`,
  `is_valid_frontdoor_adjustment_set`, `get_minimal_adjustment_set`,
  `get_ivs`, `get_conditional_ivs`, `get_total_conditional_ivs`,
  `get_scaling_indicators`, `get_proper_backdoor_graph`,
  `is_valid_adjustment_set`, `estimate_ate`).

1.x ships `CausalInference` as a `FutureWarning`-emitting shim that
forwards each method to the accessor; 2.0 deletes the class. Migration
guide enumerates the per-method mapping.

### Removed: Linear-Gaussian "closed-form" classes

Earlier drafts spec'd `LinearGaussianInference` and
`LinearGaussianCounterfactual`. Removed: the generic abduction-loop in
`_DAGCounterfactual` is already closed-form correct on LG SCMs
(`LinearGaussianCPD.abduct` returns `Delta`, `structural_predict` is
exact arithmetic). Demo B2 hits 0.0 error vs analytic. For LG inference
(rung 1), `_DAGInference.query` via LW gives correct conditionals in
the limit; an exact-on-LG optimization path can be added in v2.x behind
the same API. The `is_linear_gaussian` tag stays as an optimisation
hook.

### `LikelihoodWeighting` (new)

```python
class LikelihoodWeighting:
    """Importance sampling via per-CPD log_prob weighting. Works on any
    DAG whose registered CPDs support sample / log_prob — including
    third-party skpro / sklearn estimators after CPDAdapter wrapping.

    For continuous evidence ApproxInference's rejection sampling
    degenerates; use LikelihoodWeighting instead.
    """

    def __init__(self, model: DAG) -> None: ...

    def weighted_sample(self, evidence=None, n_samples: int = 10_000,
                        seed: int | None = None) -> pd.DataFrame:
        """DataFrame with n_samples rows + a '_weight' column."""

    def query(self, variables, evidence=None, n_samples: int = 10_000,
              seed: int | None = None) -> dict[Hashable, Any]:
        """Discrete vars → {state: prob}; continuous → weighted samples."""

    def predict(self, data: pd.DataFrame, n_samples: int = 10_000,
                seed: int | None = None) -> pd.DataFrame: ...
```

---

## Capability tags reference

Reduced 11 → 7 tags (dropped `noise_invertible`, `parameter_uncertainty`,
`is_mixture`, `supports_analytic_conditioning`).

| Tag | Type | Purpose |
|---|---|---|
| `variable_type` | `"discrete" \| "continuous"` | `_DAGInference.predict` auto-dispatch. |
| `produces_factor` | `bool` | Required by `VariableElimination`, `BeliefPropagation`, `dag.transforms.to_markov_model`. |
| `is_linear_gaussian` | `bool` | Required by `dag.transforms.to_joint_gaussian`; reserved for future LG-specific optimisations. |
| `supports_fit_joint` | `bool` | Required by `JointPyroEstimator`. |
| `supports_counterfactual` | `bool` | Required by `_DAGCounterfactual.query` and `_DAGDiagnostics.identifiability_report`. Implies the CPD implements `noise_prior`, `structural_predict`, `abduct`. |
| `noise_type` | `"additive" \| "post_nonlinear" \| "inverse_cdf" \| "gumbel_max" \| "custom"` | Counterfactual algorithm selection; consumed by `identifiability_report`. |
| `python_dependencies` | `list[str]` | Soft-dep gating via `_check_soft_dependencies`. |

| Tag | `TabularCPD` | `LinearGaussianCPD` | `FunctionalCPD` | `WrappedRegressor` |
|---|:-:|:-:|:-:|:-:|
| `variable_type` | discrete | continuous | continuous | continuous |
| `produces_factor` | ✓ | ✗ | ✗ | ✗ |
| `is_linear_gaussian` | ✗ | ✓ | ✗ | ✗ |
| `supports_fit_joint` | ✗ | ✗ | ✓ | ✗ |
| `supports_counterfactual` | ✓ | ✓ | ✓ | ✓ |
| `noise_type` | `inverse_cdf`¹ | `additive` | `custom` | `additive` or `post_nonlinear`² |

¹ `TabularCPD(noise_repr="gumbel_max")` flips this to `gumbel_max` (v2.x); different counterfactual answers, so the choice is user-facing.
² Per-instance: `additive` if `link=None` (ANM), `post_nonlinear` if `link`/`link_inv` given.

Third-party CPDs read tags via `cpd.get_tag(name, default)` when
available. Defaults from upstream base: `sklearn.ClassifierMixin` →
`variable_type="discrete"`; `skpro.BaseProbaRegressor` →
`variable_type="continuous"`. Other tags default `False`, routing
third-party CPDs to `LikelihoodWeighting`.
