## Framework-Agnostic CPD Parameterization for Bayesian Networks

Contributors: @ankurankan

> **Companion documents:**
> - `2026-05-14-parameterization-contracts.md` — full class signatures and tag tables.
> - `2026-05-14-parameterization-refactor.md` — phased implementation plan.

### Introduction

A pgmpy Bayesian network is parameterized by one CPD per node. Today three CPD classes exist: `TabularCPD`
(multinomial), `LinearGaussianCPD` (Gaussian with linear mean), and `FunctionalCPD` (arbitrary distribution via a Pyro
callable).

The conditional relationship between a node and its parents is not restricted to these three shapes — any probabilistic
regressor (continuous child) or classifier (discrete child) is a valid CPD. A rich ecosystem already exists (skpro for
regression, scikit-learn for classification), but there is no path to use any of them as a pgmpy CPD without writing a
bespoke wrapper.

The current design also has three structural issues that block broader extensibility:

1. **No shared contract.** The three CPDs share no methods for sampling, scoring, or fitting. Each BN class has its own
   `simulate()` that hard-codes the CPD type. Inference branches on `isinstance(cpd, TabularCPD)` in dozens of places.
2. **Inconsistent interfaces.** No CPD implements `log_prob`, which blocks likelihood weighting, importance sampling,
   and score-based structure learning for non-tabular CPDs.
3. **CPDs carry node identity.** Each CPD stores `variable` and `evidence`, precluding the use of any third-party
   estimator as a CPD without creating a wrapper class for it.

### Goals

- Allow any skpro probabilistic regressor or sklearn probabilistic classifier to serve as a CPD with zero wrapping.
- Collapse the three parallel `simulate()` / `fit()` pipelines for each BN into one generic implementation.
- Replace `isinstance` dispatch in inference with capability tags.
- Ship as a v2.0 breaking change, preceded by an additive 1.x rollout so downstream users have one release-cycle of
  FutureWarning notice.

### Proposed Solution

Adopt the existing skpro / sklearn estimator contract as pgmpy's CPD contract. No new `BaseParameterization` class.
Concrete moves:

1. **New module `pgmpy.parameterization`.** Houses the redesigned CPDs:
   - `TabularCPD` inherits `skbase.BaseEstimator + sklearn.ClassifierMixin`.
   - `LinearGaussianCPD` and `FunctionalCPD` inherit `skpro.regression.base.BaseProbaRegressor`.

2. **Identity-free CPDs.** CPDs no longer store `variable` / `evidence`. The Bayesian network owns the `(node -> CPD)`
   and `(node -> parent_order)` mappings. Third-party estimators slot in without any pgmpy-specific attributes.

3. **DAG-centric enrichment.** The existing `pgmpy.base.DAG` is enriched in place to serve as the unified parameterized
   network. We create a new alias `BayesianNetwork` class that is exactly same as the `DAG` class. This is to make it
   approachable for people who are used to `BayesianNetworks` and not `DAG`. The three typed BN classes
   (`DiscreteBayesianNetwork`, `LinearGaussianBayesianNetwork`, `FunctionalBayesianNetwork`) collapse into `DAG`.

4. **Namespaced accessors.** Type-specific helpers live on `cached_property` accessors (pandas-style):
   - `dag.parameters`: CPD-registry management (add / get / remove / dict-like).
   - `dag.transforms`: `to_markov_model`, `to_joint_gaussian`, `cpd_as_factor`.
   - `dag.inference`: `predict`, `predict_probability`, `log_likelihood` (auto-dispatches on CPD tags).
   - `dag.io`: `save`. (Load is a `DAG.load` classmethod.)

   The DAG class itself stays small (~7 methods); the rest of the surface is reached through accessors.

5. **Capability tags drive inference dispatch.** Each CPD class declares a `_tags` dict (`variable_type`,
   `produces_factor`, `is_linear_gaussian`, `supports_fit_joint`, …). Inference algorithms query the tags via
   `cpd.get_tag(name)` rather than `isinstance` checks. Networks with mixed CPD types automatically route to the
   appropriate inference algorithm.

6. **Fitting policy on the CPD; fitting algorithm on the estimator.** Per-CPD fitting hyperparameters (e.g., Bayesian
   prior for `TabularCPD`) live in the CPD's `__init__`, matching skpro's pattern (`GLMRegressor` vs
   `BayesianLinearRegressor` analogue). Network-level algorithms that genuinely span nodes (EM, joint Pyro fit for
   FunctionalCPD networks) live in separate estimator classes:
   - `MLEEstimator` — orchestrator. Walks the graph, calls `cpd.fit(X, y)`
     per node. Default for `dag.fit(data)`.
   - `DiscreteEM` — E/M loop with VariableElimination E-step.
   - `JointPyroEstimator` — joint SVI/MCMC for FunctionalCPD networks.

   Per-CPD Bayesian fitting is configured via the CPD's hyperparameters:
   `TabularCPD(variable_card=3, prior_type="BDeu", equivalent_sample_size=10)`. No separate `DiscreteBayesianEstimator` needed.

7. **All I/O uses pandas.** `X: pd.DataFrame` keyed by parent variable names;
   `y: pd.Series` indexed by sample. Outputs from `sample` and `log_prob` are `pd.Series`. Discrete CPDs use state-name
   *labels* at the I/O boundary; integer encoding stays internal.

8. **Numpy-only for new CPDs.** `TabularCPD` and `LinearGaussianCPD` use numpy arithmetic directly. The torch backend
   (used by `DiscreteFactor` for VE/BP) is unchanged; `FunctionalCPD` continues to require torch via Pyro.

### Alternative Solutions (rejected)

| Alternative | Reason rejected |
|---|---|
| Bespoke `BaseParameterization` class with explicit wrappers | Doubles the maintenance surface; every third-party estimator needs a wrapper subclass or registration. Adopting skpro/sklearn contracts directly gives us `clone()`, `get_params()`, `_tags`, and `check_estimator` for free. |
| Additive only — new `SkproCPD` alongside the old classes | Leaves three parallel `simulate()` pipelines and dozens of isinstance checks in inference. The underlying structural cleanup never happens. |
| sklearn-only (no skpro dependency) | Recreates the abstraction that skpro already provides (`predict_proba` returning a broadcastable distribution) and prevents existing skpro estimators from slotting in directly. |
| Hard-depend on both sklearn and skpro | Adds skpro as a hard dep for users who only build discrete networks. Soft-dep gated by `_check_soft_dependencies` is cheaper. |
| Keep node identity on the CPD; attach to third-party estimators at `add_cpds` time | Monkey-patching attributes onto third-party instances is fragile, breaks `clone()` and `get_params()`, and introduces sync risk between the CPD's idea of its identity and the BN's mapping. |
| New `BayesianNetwork` class as the unified type | Adds another class to learn. The existing `DAG` is the natural home — every parameterized network is a DAG; an unparameterized DAG just has an empty `parameters` accessor. |
| Capability mixins on the unified class | The user-facing class size doesn't shrink (mixin methods are still on `dag`). Namespaced accessors via `cached_property` give the same separation with better discoverability. |
| Deprecation aliases shipped as one v2.0 release | The deprecation infrastructure is the most complex part of the plan and exists entirely for back-compat. A 1.x-first rollout with `FutureWarning` shims, ending in a small v2.0 deletion release, is cheaper to land incrementally. |

### Design details

This section covers the choices that aren't obvious from the contracts document. For full signatures and tag tables, see
the contracts doc.

#### Identity ownership

CPDs are pure parametric shapes — they don't know which node they parameterize or what its parents are named. The DAG
holds the mapping:

```python
class DAG:
    _cpds: dict[Hashable, Any]                       # node → CPD instance
    _parent_order: dict[Hashable, list[Hashable]]    # node → ordered parents
```

`parent_order` is recorded explicitly at `add(...)` time because
`nx.predecessors` ordering isn't a guaranteed stable contract across
NetworkX versions, and the order matters for positional coefficients like
`LinearGaussianCPD.beta_`.

This is what makes third-party estimators drop in without wrapping. A
skpro `BayesianLinearRegressor()` has no `variable` slot — that's fine;
the DAG names the node, the regressor just fits the conditional
distribution.

#### Capability tags

Each CPD class declares a `_tags` dict consumed by skbase's tag machinery
(inherited from `skbase.BaseEstimator`):

| Tag | Type | Used by |
|---|---|---|
| `variable_type` | `"discrete" \| "continuous"` | `dag.inference.predict` auto-dispatch |
| `produces_factor` | `bool` | `VariableElimination`, `BeliefPropagation` |
| `is_linear_gaussian` | `bool` | `LinearGaussianInference`, `dag.transforms.to_joint_gaussian` |
| `supports_fit_joint` | `bool` | `JointPyroEstimator` |
| `python_dependencies` | `list[str]` | Soft-dep gating via `_check_soft_dependencies` |

Third-party CPDs read tags via `obj.get_tag(name, default)` (inherited from
skbase). Defaults are inferred from the upstream base class —
`sklearn.ClassifierMixin` → `variable_type="discrete"`,
`skpro.BaseProbaRegressor` → `variable_type="continuous"`. These defaults
route third-party CPDs to `LikelihoodWeighting` for inference (no exact
algorithm's tag requirements are met).

#### CPD lifecycle: hyperparameters in `__init__`, fitted state via `from_values` or `fit`

CPDs follow the sklearn lifecycle. `__init__` takes hyperparameters only:

```python
cpd = TabularCPD(variable_card=3, evidence_card=[2, 2],
                 state_names=[["A","B","C"], ["lo","hi"], ["y","n"]],
                 prior_type="BDeu", equivalent_sample_size=10)
```

Fitted state populates trailing-underscore (sklearn) or underscore-prefixed
(skpro) attributes:

- `TabularCPD`: `values_`, `classes_`, `is_fitted_` (sklearn convention).
- `LinearGaussianCPD`: `beta_`, `std_`, `_is_fitted` (skpro convention —
  skpro's `BaseProbaRegressor` exposes `is_fitted` as a property tied to
  `_is_fitted`).

Two construction paths:

- `TabularCPD(variable_card=3).fit(X, y)` — learn from data.
- `TabularCPD.from_values(variable_card=3, values=[[…]])` — direct
  parameter specification (a classmethod that sets the fitted attributes
  and marks the estimator fitted without going through `fit`).

The two conventions on `is_fitted` are unavoidable: sklearn and skpro
disagree on the spelling. `TabularCPD.from_values` sets `is_fitted_ = True`
(sklearn style); `LinearGaussianCPD.from_values` sets `_is_fitted = True`
(skpro style). Documented per class in the contracts doc.

#### Accessor pattern

Each accessor is a small class with a back-reference to the DAG, exposed
via `cached_property`:

```python
class DAG:
    @cached_property
    def parameters(self): return _DAGParameters(self)
    @cached_property
    def transforms(self): return _DAGTransforms(self)
    @cached_property
    def inference(self): return _DAGInference(self)
    @cached_property
    def io(self): return _DAGIO(self)
```

`cached_property` makes `dag.parameters is dag.parameters` true across
calls (same instance returned).

Each accessor method that requires CPD-type homogeneity (e.g.,
`transforms.to_joint_gaussian`) checks the relevant tag on every CPD at
call time and raises `TypeError` with a clear, node-named message if the
precondition fails.

`_DAGParameters` is dict-like: supports `add` / `get` / `remove` /
`keys` / `values` / `items` / `__len__` / `__iter__` / `__contains__` /
`__getitem__`.

#### Fitting: policy on the CPD, algorithm on the estimator

The CPD's `fit` is the per-node math primitive. Two orthogonal optional
hooks accommodate the common fitting variations without any `if/else` over
method names:

```python
def fit(self, X, y, sample_weight=None):
    counts = self._weighted_counts(X, y, sample_weight)
    if self.prior_type is not None:   # set via __init__
        counts = counts + self._pseudo_counts()
    self.values_ = counts / counts.sum(axis=0, keepdims=True)
    return self
```

- **MLE**: instantiate with no prior (`TabularCPD(variable_card=3)`).
- **Bayesian (MAP)**: instantiate with prior (`TabularCPD(variable_card=3,
  prior_type="BDeu")`).
- **EM**: handled by `DiscreteEM` estimator at the network level — its
  M-step calls `cpd.fit(X, y, sample_weight=expected_counts)`.

Network-level algorithms that genuinely cross node boundaries live in
estimator classes (`MLEEstimator`, `DiscreteEM`, `JointPyroEstimator`).
These accept any DAG, walk the graph, and orchestrate CPD-level fitting.
`dag.fit(data, estimator=None)` defaults to `MLEEstimator()`.

This pattern mirrors sklearn: `RandomForestClassifier.fit(X, y,
sample_weight=None)` is one method that handles unweighted and weighted
cases via a single optional argument, not via branching on method names.

#### Inference

Inference algorithms accept any DAG. Each checks the CPDs' tags at
construction and raises `TypeError` if requirements aren't met.

**Exact**:
- `VariableElimination` / `BeliefPropagation` — require every CPD to
  advertise `produces_factor=True`. Internally consume `DiscreteFactor`s
  built via `dag.transforms.cpd_as_factor(node)`.
- `LinearGaussianInference` (new) — requires every CPD to advertise
  `is_linear_gaussian=True`. Pre-computes the joint Gaussian via
  `dag.transforms.to_joint_gaussian()` and conditions on evidence via the
  Schur complement.

**Approximate**:
- `ApproxInference` (existing) — forward sampling + rejection sampling.
  Works for discrete evidence in any network.
- `LikelihoodWeighting` (new) — importance sampling via per-CPD
  `log_prob`. Required for skpro CPDs because rejection sampling fails on
  continuous evidence (match probability is zero).

**Auto-dispatch** from `dag.inference.predict(data)`:

```
all CPDs have produces_factor=True   → VariableElimination
all CPDs have is_linear_gaussian=True → LinearGaussianInference
otherwise                              → LikelihoodWeighting
```

Users can override via `method=` kwarg or by instantiating the inference
class directly for full control.

#### DBN handling

`DynamicBayesianNetwork` inherits from `DAG` (true today, unchanged).
With DAG enriched, DBN automatically gains `_cpds` / `_parent_order` /
the four accessors. DBN keeps its temporal-aware overrides for
`add_cpds(*cpds)` / `get_cpds(node, time_slice=...)` / `simulate`, since
its CPD keys are `(node, time_slice)` tuples. The `_cpds` dict accepts
tuple keys natively.

### Release staging

The refactor ships as additive 1.x minor releases ending in a v2.0
deletion cleanup. Downstream users get a full release-cycle of
`FutureWarning` notice before anything breaks.

| Release | What ships |
|---|---|
| **1.x.1** | New `pgmpy.parameterization` module. Pure addition; no v1.x code touched. |
| **1.x.2** | DAG enrichment with accessors. Legacy `dag.add_cpds`/`get_cpds`/`remove_cpds`/`cpds` kept as `FutureWarning`-emitting shims. Typed BN classes (`DiscreteBayesianNetwork` etc.) become `FutureWarning`-emitting subclasses of `DAG`. New parameter estimators + inference algorithms. |
| **1.x.3** | `FutureWarning` on legacy `pgmpy.factors.*` CPDs. Readwrite update. Migration guide. |
| **2.0** | Delete legacy classes and `DAG.add_cpds`/etc. shims. Mechanical release. |

### Breaking changes (pgmpy 2.0)

These v1.x names exist in 1.x and are deleted in 2.0. See
`docs/source/migration-v2.rst` for the migration map (shipped in 1.x.3).

**Classes deleted:**
- `pgmpy.models.{DiscreteBayesianNetwork, LinearGaussianBayesianNetwork, FunctionalBayesianNetwork, BayesianNetwork}`.
- `pgmpy.factors.discrete.TabularCPD` (the legacy class — `DiscreteFactor` stays).
- `pgmpy.factors.continuous.LinearGaussianCPD` and the `pgmpy.factors.continuous` package.
- `pgmpy.factors.hybrid.FunctionalCPD` and the `pgmpy.factors.hybrid` package.
- `pgmpy.estimators.{MaximumLikelihoodEstimator, BayesianEstimator, EM}`.

**Methods deleted:**
- `DAG.add_cpds` / `get_cpds` / `remove_cpds` / `cpds` (replaced by `dag.parameters` accessor).

**API renames** (full table in the migration guide):
- `bn.add_cpds(cpd)` → `dag.parameters.add(variable=..., cpd=...)`
- `TabularCPD("X", 2, values=...)` → `TabularCPD.from_values(variable_card=2, values=...)`
- `bn.predict(data)` → `dag.inference.predict(data)`
- `bn.to_markov_model()` → `dag.transforms.to_markov_model()`
- `bn.save(path)` → `dag.io.save(path)` / `DAG.load(path)`

### Out-of-scope (non-goals)

- **Discrete children via skpro.** skpro is regression-only. Discrete
  children use `TabularCPD` or any sklearn-classifier-style estimator.
- **Posterior over parents under continuous-black-box CPDs.** Black-box
  regressors cannot Bayes-invert analytically. Likelihood weighting
  degenerates for deep continuous evidence. Full MCMC is the real answer
  and is deferred.
- **Mixed-type parent preprocessing.** A continuous-child CPD whose
  parents include discrete variables needs one-hot or label encoding.
  The user encodes upfront; a default `sklearn.compose` integration is a
  follow-up. The migration guide flags this explicitly.
- **General MCMC inference** (Gibbs, NUTS, HMC). Deferred.
- **Torch backend for `TabularCPD` and `LinearGaussianCPD`.** Numpy-only
  for the new CPDs. `FunctionalCPD` continues to require torch via Pyro.
- **`bn.fit_update(data)` online update.** Folded into the v1.x
  `DiscreteBayesianEstimator` shim if needed; the new API will provide
  a first-class equivalent as a follow-up.
- **`NoisyOR` port.** Specialized `TabularCPD`-like CPD. Stays in
  `pgmpy.factors.discrete` for v1.x; ported to the new module as a
  follow-up.

### Open risks

- **skbase surface sufficiency.** Relies on skbase providing `_tags`,
  `get_tag`, `get_params`, `set_params`, `clone`, `is_fitted`. Confirmed
  working in the prototype (`docs/superpowers/prototype/prototype.py`).
- **skbase + nx.DiGraph multi-inheritance.** Requires init args
  (`ebunch`, `latents`) to be stored as instance attributes *unchanged*
  before calling parent inits; runtime state goes under `_`-prefixed
  names. Confirmed working in the prototype.
- **`is_fitted` naming difference.** sklearn uses `is_fitted_`; skpro
  uses `_is_fitted`. `TabularCPD.from_values` and
  `LinearGaussianCPD.from_values` set the appropriate flag per
  convention. Documented per class.
- **Migration cost.** isinstance-on-CPD-type is pervasive in pgmpy's own
  code. The 1.x rollout window keeps both paths working; v2.0 cleanup is
  mechanical but touches many files.
- **Performance regression from pandas I/O.** Per-row pandas construction
  in `cpd_sample` / `cpd_log_prob` could be measurably slower than
  current numpy-internal sampling. Mitigation: benchmark on
  `bnlearn/alarm`-scale networks before each release; add numpy fast
  paths inside `TabularCPD.sample` / `LinearGaussianCPD.sample` if needed,
  keeping the pandas API at the boundary.

### User journeys

**1. Discrete BN, new idiomatic API.**

```python
from pgmpy.base import DAG
from pgmpy.parameterization import TabularCPD

student = DAG([("diff", "grade"), ("intel", "grade")])
student.parameters.add(variable="diff",
                        cpd=TabularCPD.from_values(variable_card=2,
                                                    values=[[0.6], [0.4]]))
student.parameters.add(variable="intel",
                        cpd=TabularCPD.from_values(variable_card=2,
                                                    values=[[0.7], [0.3]]))
student.parameters.add(
    variable="grade",
    cpd=TabularCPD.from_values(
        variable_card=3, evidence_card=[2, 2],
        values=[[0.3, 0.05, 0.9, 0.5],
                [0.4, 0.25, 0.08, 0.3],
                [0.3, 0.7,  0.02, 0.2]]),
    parent_order=["diff", "intel"],
)
student.simulate(n_samples=1000)
```

**2. Hybrid BN with a third-party skpro regressor as a CPD.**

```python
from pgmpy.base import DAG
from pgmpy.inference import ApproxInference
from pgmpy.parameterization import TabularCPD
from skpro.regression.ensemble import NGBoostRegressor

bn = DAG([("smoker", "income"), ("age", "income")])
bn.parameters.add(variable="smoker", cpd=TabularCPD(variable_card=2))
bn.parameters.add(variable="age",    cpd=TabularCPD(variable_card=5))
bn.parameters.add(variable="income", cpd=NGBoostRegressor())

bn.fit(data)                          # MLEEstimator orchestrates per-node fit
samples = bn.simulate(n_samples=10_000)
ApproxInference(bn).query(
    variables=["income"], evidence={"smoker": "yes", "age": "30-40"},
)
```

`VariableElimination(bn)` would raise `IncompatibleCPDError` here because
`NGBoostRegressor` does not advertise `produces_factor=True`.
`LikelihoodWeighting(bn)` is the right choice for continuous evidence on
this network.

**3. Bayesian fitting via CPD hyperparameter (no separate estimator).**

```python
from pgmpy.base import DAG
from pgmpy.parameterization import TabularCPD

dag = DAG([("diff", "grade"), ("intel", "grade")])
dag.parameters.add(variable="diff",
                    cpd=TabularCPD(variable_card=2,
                                    prior_type="BDeu",
                                    equivalent_sample_size=10))
dag.parameters.add(variable="intel",
                    cpd=TabularCPD(variable_card=2,
                                    prior_type="BDeu"))
dag.parameters.add(variable="grade",
                    cpd=TabularCPD(variable_card=3, evidence_card=[2, 2],
                                    prior_type="BDeu"),
                    parent_order=["diff", "intel"])
dag.fit(data)   # MLEEstimator → per-node cpd.fit() → applies BDeu prior internally
```

Same pattern as skpro: `GLMRegressor()` is frequentist;
`BayesianLinearRegressor(prior=...)` is Bayesian. For pgmpy, the Bayesian
mode is a hyperparameter on the same `TabularCPD` class.

**4. Linear-Gaussian network with exact inference.**

```python
from pgmpy.base import DAG
from pgmpy.inference import LinearGaussianInference
from pgmpy.parameterization import LinearGaussianCPD

dag = DAG([("x1", "x2"), ("x2", "x3")])
dag.parameters.add(variable="x1",
                    cpd=LinearGaussianCPD.from_values(beta=[1.0], std=2.0))
dag.parameters.add(variable="x2",
                    cpd=LinearGaussianCPD.from_values(beta=[-5, 0.5], std=2.0),
                    parent_order=["x1"])
dag.parameters.add(variable="x3",
                    cpd=LinearGaussianCPD.from_values(beta=[4, -1], std=1.7),
                    parent_order=["x2"])

mu, cov = dag.transforms.to_joint_gaussian()
LinearGaussianInference(dag).query(["x3"], evidence={"x1": 2.0})
```
