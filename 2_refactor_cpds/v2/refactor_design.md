## Framework-Agnostic CPD Parameterization for Bayesian Networks

Contributors: @ankurankan, @daehyun

> **Companion documents:**
> - `class_contracts.md` — full class signatures and tag tables.
> - `refactor_ai_agent_plan.md` — phased implementation plan.

### Motivation

A pgmpy Bayesian network is parameterized by one CPD per node. Today three CPD classes exist: `TabularCPD` (multinomial), `LinearGaussianCPD` (Gaussian with linear mean), and `FunctionalCPD` (arbitrary distribution via a Pyro callable). That design works for classical discrete BNs but does not scale to two user groups this refactor must serve:

1. **Classical Bayesian-network users** want structure → parameters → query under evidence or intervention → summary statistics. They should not need to author schemas or structural-noise objects.
2. **Causal-inference users** need structure learning, parameter learning, optional SCM learning for counterfactuals, plus associational, interventional, counterfactual, simulation, identification, and diagnostic operations.

Both groups need the same abstraction: a graph whose local mechanisms are composable conditional models. Any probabilistic regressor or classifier should be usable as a CPD; skpro and sklearn already provide them, but pgmpy has no clean path without bespoke wrappers.

The current design has six structural issues blocking extensibility:

1. **No shared operation contract.** The three CPDs don't expose a consistent surface for fitting, prediction, sampling, scoring, or structural-causal operations.
2. **No general scoring/sampling boundary.** Missing `sample`/`log_prob` support blocks likelihood weighting, importance sampling, and score-based workflows.
3. **CPDs carry graph identity.** Storing `variable`/`evidence` makes them hard to reuse and incompatible with ordinary sklearn/skpro models.
4. **State metadata has no stable owner.** Discrete states, dtypes, encoders are scattered across CPDs and file formats.
5. **Counterfactuals need explicit noise.** Classical CPDs give `P(X | Pa(X))`; counterfactuals also need a structural function and exogenous-noise representation.
6. **The graph is being asked to act like an estimator.** A DAG is a mutable graph/model container, not a sklearn estimator with constructor-parameter clone semantics.

### Goals

- Keep the classical BN path ergonomic — no required schema/noise authoring.
- Make the CPD boundary sklearn/skpro-style: `__init__` hyperparameters, `fit(X, y)`, `predict_proba(X)`, fitted attributes, CPD-level cloning, capability tags. Native and third-party CPDs share one local-model abstraction.
- Keep `DAG` a pgmpy-native graph/model container, not an estimator. The DAG owns node-to-CPD registration, parent ordering, schema, accessors, I/O, and explicit `copy_template(parameters=...)`.
- Let users specify CPDs directly (`from_values`) while DAG normalizes state names, cardinalities, dtypes, encoders into `dag.schema` during registration, fitting, and read/write.
- Replace class-based inference dispatch with capability tags and small operation-specific protocols (`FittableCPD`, `PredictiveCPD`, `SampleableCPD`, `ScorableCPD`, optional `StructuralCPD`).
- Provide a clean causal layering: `dag.inference` (associational), `dag.intervene` (interventional), `dag.counterfactual` (abduction-action-prediction), `dag.diagnostics` (identification + model diagnostics).
- Represent exogenous noise explicitly for SCM-capable CPDs. Built-in structural CPDs expose `noise_prior`/`abduct`/`structural_predict` through a uniform `NoiseDistribution` interface.
- Make discrete counterfactual semantics explicit (`noise_repr`); different encodings can imply different counterfactual answers.
- Collapse duplicated typed-BN code paths into shared DAG accessors. Preserve compatibility during 1.x via `FutureWarning` shims.
- Ship through additive 1.x releases before v2.0 cleanup.

### Non-goals

- Do not make `DAG` a sklearn/skbase estimator.
- Do not require users to manually populate schema for normal workflows.
- Do not require every CPD to support counterfactuals (opt-in via `StructuralCPD`).
- Do not expose a public `LinearGaussianInference` class in v2.0 (`is_linear_gaussian` tag + `dag.transforms.to_joint_gaussian()` remain optimisation hooks).
- Do not solve every discrete SCM encoding in v2.0 (inverse-CDF only; Gumbel-Max deferred to v2.x, and must be explicit because it changes counterfactual answers).

### Proposed Solution

Adopt the existing skpro/sklearn estimator contract at the CPD boundary, not at the graph-container boundary. Concrete moves:

1. **New module `pgmpy.parameterization`.** Houses redesigned CPDs:
   - `TabularCPD` inherits `sklearn.ClassifierMixin + skbase.BaseEstimator`.
   - `LinearGaussianCPD`, `FunctionalCPD`, `WrappedRegressor` inherit `skpro.regression.base.BaseProbaRegressor`.

2. **Identity-free CPDs.** CPDs no longer store `variable`/`evidence`. The DAG owns `(node → CPD)` and `(node → parent_order)` mappings. Third-party estimators slot in without pgmpy-specific attributes.

3. **DAG-centric enrichment without estimator inheritance.** `pgmpy.base.DAG` is enriched in place. Inherits from `nx.DiGraph` + existing pgmpy graph-roles mixin only — *not* skbase/sklearn. The three typed BN classes (`DiscreteBayesianNetwork`, etc.) become `FutureWarning`-emitting subclasses during 1.x and are removed in v2.0. No new `BayesianNetwork` alias; users migrate to `DAG`.

4. **Namespaced accessors mapped to Pearl's three rungs + tooling.** `cached_property` accessors (pandas-style):
   - `dag.parameters`: CPD-registry (add/get/remove/dict-like). Auto-wraps third-party CPDs in `CPDAdapter` on `add()`.
   - `dag.schema`: internal variable metadata (states, dtypes, types, encoders). Populated automatically; user override is an escape hatch.
   - `dag.inference` *(rung 1)*: `query(evidence, query) → QueryResult`. Auto-dispatches VE for fully discrete, LW otherwise.
   - `dag.intervene` *(rung 2)*: `query(do, query, evidence=None, adjustment_set=None) → QueryResult`; `simulate(do, n_samples) → DataFrame`. **Unified rung-2 surface** — subsumes the five v1.x entry points: `DiscreteBayesianNetwork.simulate(do=...)` and `LinearGaussianBayesianNetwork.simulate(do=...)` (typed BN classes deprecated); `DiscreteBayesianNetwork.do(nodes)` (typed BN class deprecated); `pgmpy.base.DAG.do(nodes)` (renamed `DAG.with_intervention(nodes)` — distinct from value-level `dag.intervene.query(do=...)`); and `pgmpy.inference.CausalInference.query(do=..., adjustment_set=...)` (whole class becomes a `FutureWarning` shim in 1.x and is deleted in 2.0). The `adjustment_set` parameter on `dag.intervene.query` reproduces the back-door / front-door adjustment formulas from `CausalInference`.
   - `dag.counterfactual` *(rung 3)*: `query(observed, do, query) → QueryResult` (list if `do` is a list — multi-world sharing abducted noise); `explain(observed, do)` for per-node noise.
   - `dag.transforms`: graph primitives (`ancestors`, `descendants`, `topological_order`, `markov_blanket`, `d_separated`) + type-conditional transformations (`to_markov_model`, `to_joint_gaussian`, `cpd_as_factor`).
   - `dag.diagnostics`: `identifiability_report(query_type=...)`. Future home for residual analysis, structure-adequacy.
   - `dag.bootstrap`: `query(data, query_fn, n_bootstrap=...) → QueryResult` for fit-time CIs over any other accessor.
   - `dag.io`: `save`. (Load is `DAG.load` classmethod.)

   All `query()` methods return `QueryResult` uniformly. The DAG class itself stays small (~7 methods); template-copy workflows use explicit pgmpy semantics:

   ```python
   dag.copy_template(parameters="none")    # graph + inferred schema
   dag.copy_template(parameters="unfit")   # graph + schema + cloned CPD specs
   dag.copy_template(parameters="fitted")  # graph + schema + deep-copied CPDs
   ```

5. **Capability tags drive inference dispatch.** Each CPD class declares `_tags` (`variable_type`, `produces_factor`, `is_linear_gaussian`, `supports_fit_joint`, `supports_counterfactual`, `noise_type`, `python_dependencies`). Inference algorithms query via `cpd.get_tag(name)`, not `isinstance`.

6. **Fitting policy on the CPD; fitting algorithm on the estimator.** Per-CPD fitting hyperparameters (e.g. Bayesian prior for `TabularCPD`) live in the CPD's `__init__`. Network-level algorithms that cross node boundaries live in separate estimator classes:
   - `MLEEstimator` — orchestrator. Walks the graph, calls `cpd.fit(X, y)` per node. Default for `dag.fit(data)`.
   - `DiscreteEM` — E/M loop with VE E-step.
   - `JointPyroEstimator` — joint SVI/MCMC for FunctionalCPD networks.

   No separate `DiscreteBayesianEstimator`; Bayesian fitting is `TabularCPD(prior_type="BDeu", ...)`.

7. **All I/O uses pandas.** `X: pd.DataFrame` keyed by parent variable names; `y: pd.Series` indexed by sample. `sample`/`log_prob` return `pd.Series`. Discrete CPDs use state-name *labels* at the I/O boundary; integer encoding stays internal. `dag.schema` is the canonical place for state names and encoding metadata.

8. **Numpy-only for new CPDs.** `TabularCPD` and `LinearGaussianCPD` use numpy directly. Torch backend (used by `DiscreteFactor` for VE/BP) is unchanged; `FunctionalCPD` continues to require torch via Pyro.

9. **StructuralCPD protocol for counterfactual reasoning.** Optional protocol — three methods (`noise_prior`, `structural_predict`, `abduct`) plus two tags (`supports_counterfactual`, `noise_type`). **Noise is uniformly typed**: every `noise_prior()` and `abduct()` returns a `NoiseDistribution` (`.sample(n)` + `.point()`). Built-in types: `Delta`, `NormalNoise`, `Empirical`, `TruncatedUniform`. The counterfactual algorithm calls `.sample()` identically across CPD types.

   Built-in CPDs implement the protocol:
   - `LinearGaussianCPD` → `noise_prior() = NormalNoise(0, std_)`, `abduct() = Delta(x − β·pa)` (canonical ANM).
   - `TabularCPD` → inverse-CDF: `noise_prior() = TruncatedUniform([0],[1])`, `abduct() = TruncatedUniform([F(x−1|pa)],[F(x|pa)])`. Convention from Pearl §7.1 / `dowhy.gcm.ClassifierFCM`. Gumbel-Max is a future opt-in via `noise_repr="gumbel_max"`.
   - `FunctionalCPD` → SCM-native via Pyro `do`/`condition` poutines, wrapped in a `NoiseDistribution`.
   - `WrappedRegressor(regressor, *, link=None, link_inv=None, noise_dist=None)` — one adapter for ANM (link=None) and PNL (link given). Pluggable `noise_dist`; parameter uncertainty propagates automatically when the wrapped regressor is Bayesian.

   **Design improvements over DoWhy / ChiRho / pyAgrum / pomegranate** (all in v2.0 scope):

   1. **Pearl's three rungs cleanly mapped to three accessors**, each returning `QueryResult`. ChiRho composes via effect handlers (requires Pyro); pyAgrum splits across `BayesNet` + `CausalModel`; DoWhy buries the rungs in separate functions. One mental model: pick your rung, call `.query()`.
   2. **Protocol over rigid hierarchy.** One `StructuralCPD` protocol + tags vs DoWhy's three separate mechanism classes (ANM, PNL, ClassifierFCM). New mechanism shapes via wrappers, not subclassing.
   3. **`QueryResult` rich return type.** DoWhy returns floats; we return an object with `point/distribution/expectation/credible_interval/compare_to`, uniform across rungs.
   4. **Unified `NoiseDistribution`.** Every CPD's noise methods return the same shape. Adding mixture/copula/GP CPDs requires no algorithm changes.
   5. **`CPDAdapter` for sklearn / sklearn.Pipeline / skpro.** Any `fit(X, y) + predict_proba(X)` object becomes a CPD on `dag.parameters.add()`. DoWhy supports sklearn only inside ANM; pomegranate uses its own distributions; we accept the entire ecosystem natively. `sklearn.Pipeline`/`ColumnTransformer` is the recommended path for mixed-type parent preprocessing.
   6. **Multi-world counterfactuals via list-of-`do` (ChiRho-style).** Useful for path-specific effects and contrastive explanations.
   7. **`dag.bootstrap` — fit-time CIs for any query.** Wraps any other accessor in a refit loop. Works across all three rungs.
   8. **Composable noise distributions.** `WrappedRegressor(GLM(), noise_dist=Empirical(...))` or any `NoiseDistribution` — DoWhy hardcodes empirical residuals.
   9. **`identifiability_report()` under `dag.diagnostics`.** Flags non-identification (LG sub-paths per Hoyer 2009; non-invertible discrete noise per Oberst-Sontag 2019). DoWhy computes non-identified counterfactuals silently.
   10. **Standalone graph primitives on `dag.transforms`** (causal-learn / R6causal-style) — usable outside the inference pipeline.

   **Prototype Section B (Demos B1–B5) validates the design.** B2 hits the closed-form analytic counterfactual exactly (0.0 error) on a 3-node LG SCM with multi-world recovery across three intervention values. B3 round-trips PNL abduction to floating-point precision. B4's identifiability check flags pure-LG and clears PNL (Zhang-Hyvärinen 2009). B5 bootstrap-refits 100× and produces a tight credible interval on n=300 data.

### Alternative Solutions (rejected)

| Alternative | Reason rejected |
|---|---|
| Bespoke `BaseParameterization` class with explicit wrappers | Doubles maintenance surface; every third-party estimator needs a wrapper. Adopting skpro/sklearn at the CPD boundary gives `clone()`/`get_params()`/tags/checks for free. |
| Additive only — new `SkproCPD` alongside the old classes | Leaves three parallel `simulate()` pipelines and dozens of isinstance checks. Structural cleanup never happens. |
| sklearn-only (no skpro dependency) | Recreates skpro's `predict_proba`-returning-Distribution abstraction; prevents existing skpro estimators from slotting in. |
| Hard-depend on skpro | Adds skpro as hard dep for users only building discrete networks. Keep skpro soft-dep; sklearn core because `TabularCPD` follows the sklearn classifier contract. |
| Keep node identity on the CPD; attach to third-party estimators at `add_cpds` time | Monkey-patching attributes onto third-party instances is fragile, breaks clone/get_params. |
| New `BayesianNetwork` class as the unified type | Adds another class to learn. `DAG` is the natural home. |
| Capability mixins on the unified class | The user-facing class doesn't shrink. Namespaced accessors give the same separation with better discoverability. |
| Deprecation aliases shipped as one v2.0 release | Deprecation infrastructure is the most complex part. 1.x-first rollout with `FutureWarning` shims, ending in a small v2.0 deletion release, is cheaper to land incrementally. |
| Inherit `DAG` from skbase/sklearn estimator base | Constructor-parameter clone semantics don't match a graph whose state comes from mutations, CPD registration, schema inference, fitting, and load. Use `copy_template(parameters=...)` instead. |

### Design details

For full signatures and tag tables, see the contracts doc.

#### Identity ownership

CPDs are pure parametric shapes. The DAG holds:

```python
class DAG:
    _cpds: dict[Hashable, Any]                       # node → CPD instance
    _parent_order: dict[Hashable, list[Hashable]]    # node → ordered parents
    _schema: dict[Hashable, VariableSchema]          # node → inferred metadata
```

`parent_order` is recorded explicitly at `add(...)` time because `nx.predecessors` ordering isn't a guaranteed stable contract across NetworkX versions, and the order matters for positional coefficients like `LinearGaussianCPD.beta_`.

This is what makes third-party estimators drop in without wrapping. A skpro `BayesianLinearRegressor()` has no `variable` slot — that's fine; the DAG names the node, the regressor fits the conditional distribution.

Variable domains and encodings are DAG-owned. `TabularCPD.from_values(..., state_names=...)` is the simple user-facing API; `dag.parameters.add(...)` records those states in `dag.schema` under the actual variable names.

#### Schema precedence policy

`dag.schema` collects variable metadata from multiple sources. Conflicts are resolved by explicit precedence (highest authority first):

1. **Explicit user override** via `dag.schema.set(variable, ...)`.
2. **CPD-authored states** from `TabularCPD.from_values(..., state_names=...)` or `TabularCPD(..., state_names=...)`. Registered by `dag.parameters.add(...)` via `_DAGSchema.infer_from_cpd`. **State ordering preserved as declared.**
3. **Pandas categorical metadata** (`series.cat.categories`, `series.cat.ordered`).
4. **Observed data uniques.** Order `sorted(unique, key=str)`. Fallback only — no reordering of declared states.
5. **CPD cardinality fallback** (`variable_card` → `tuple(range(card))`).

**Invariant**: once a higher-authority source sets `states`, lower sources don't reorder them. Lower sources validate consistency: observed values outside the declared state set raise a node-named `ValueError`. Conflicting `variable_type` across sources is always a hard conflict.

#### Capability tags

| Tag | Type | Used by |
|---|---|---|
| `variable_type` | `"discrete" \| "continuous"` | `dag.inference.predict` auto-dispatch |
| `produces_factor` | `bool` | `VariableElimination`, `BeliefPropagation` |
| `is_linear_gaussian` | `bool` | `dag.transforms.to_joint_gaussian`; reserved for future LG-specific inference optimisations |
| `supports_fit_joint` | `bool` | `JointPyroEstimator` |
| `supports_counterfactual` | `bool` | `dag.counterfactual` + `dag.diagnostics` |
| `noise_type` | `"additive" \| "post_nonlinear" \| "inverse_cdf" \| "gumbel_max" \| "custom"` | Counterfactual algorithm selection; `identifiability_report` |
| `python_dependencies` | `list[str]` | Soft-dep gating |

Dropped from earlier drafts: `noise_invertible` (derived from `noise_type`), `parameter_uncertainty` (auto-detected at fit time), `supports_analytic_conditioning` (never queried), `is_mixture` (informal).

Third-party CPDs read tags via `obj.get_tag(name, default)` when available. Defaults from upstream base: `sklearn.ClassifierMixin` → `variable_type="discrete"`; `skpro.BaseProbaRegressor` → `variable_type="continuous"`. Other tags default `False`, routing third-party CPDs to `LikelihoodWeighting`.

#### CPD lifecycle

`__init__` takes hyperparameters only:

```python
cpd = TabularCPD(variable_card=3, evidence_card=[2, 2],
                 state_names=[["A","B","C"], ["lo","hi"], ["y","n"]],
                 prior_type="BDeu", equivalent_sample_size=10)
```

Fitted state populates either trailing-underscore (sklearn) or leading-underscore (skpro) attributes:

- `TabularCPD`: `values_`, `classes_`, `is_fitted_` (sklearn convention).
- `LinearGaussianCPD`: `beta_`, `std_`, `_is_fitted` (skpro convention — `BaseProbaRegressor` exposes `is_fitted` as a property tied to `_is_fitted`).

Two construction paths:

- `TabularCPD(variable_card=3).fit(X, y)` — learn from data.
- `TabularCPD.from_values(variable_card=3, values=[[…]])` — direct parameter specification, marks fitted without going through `fit`.

The two `is_fitted` conventions are unavoidable (sklearn and skpro disagree). `TabularCPD.from_values` sets `is_fitted_ = True`; `LinearGaussianCPD.from_values` sets `_is_fitted = True`. Documented per class.

#### Accessor pattern

Each accessor is a small class with back-reference to the DAG, exposed via `cached_property`. Full v2.0 surface:

```python
class DAG(_GraphRolesMixin, nx.DiGraph):
    @cached_property
    def parameters(self): return _DAGParameters(self)
    @cached_property
    def schema(self): return _DAGSchema(self)
    @cached_property
    def inference(self): return _DAGInference(self)            # rung 1
    @cached_property
    def intervene(self): return _DAGIntervene(self)            # rung 2
    @cached_property
    def counterfactual(self): return _DAGCounterfactual(self)  # rung 3
    @cached_property
    def transforms(self): return _DAGTransforms(self)
    @cached_property
    def diagnostics(self): return _DAGDiagnostics(self)
    @cached_property
    def bootstrap(self): return _DAGBootstrap(self)
    @cached_property
    def io(self): return _DAGIO(self)
```

`cached_property` makes `dag.parameters is dag.parameters` true across calls.

Each accessor method that requires CPD-type homogeneity (e.g. `transforms.to_joint_gaussian`) checks the relevant tag at call time and raises `TypeError` with a node-named message on precondition failure.

`_DAGParameters` is dict-like: `add`/`get`/`remove`/`keys`/`values`/`items`/`__len__`/`__iter__`/`__contains__`/`__getitem__`.

#### Fitting: policy on the CPD, algorithm on the estimator

The CPD's `fit` is the per-node math primitive. Two orthogonal optional hooks:

```python
def fit(self, X, y, sample_weight=None):
    counts = self._weighted_counts(X, y, sample_weight)
    if self.prior_type is not None:
        counts = counts + self._pseudo_counts()
    self.values_ = counts / counts.sum(axis=0, keepdims=True)
    return self
```

- **MLE**: `TabularCPD(variable_card=3)`.
- **Bayesian (MAP)**: `TabularCPD(variable_card=3, prior_type="BDeu")`.
- **EM**: `DiscreteEM` estimator at network level; its M-step calls `cpd.fit(X, y, sample_weight=expected_counts)`.

`dag.fit(data, estimator=None)` defaults to `MLEEstimator()`. Network-level algorithms that cross node boundaries live in estimator classes.

#### Inference

Each algorithm accepts any DAG; checks tags at construction; raises `TypeError` on requirement failure.

**Exact:**
- `VariableElimination` / `BeliefPropagation` — require every CPD `produces_factor=True`. Consume `DiscreteFactor`s via `dag.transforms.cpd_as_factor(node)`.

**Approximate:**
- `ApproxInference` (existing) — forward sampling + rejection.
- `LikelihoodWeighting` (new) — importance sampling via per-CPD `log_prob`. Required for skpro CPDs (rejection fails on continuous evidence).

**Auto-dispatch** from `dag.inference.query(...)`:

```
all CPDs have produces_factor=True   → VariableElimination
otherwise                              → LikelihoodWeighting
```

`is_linear_gaussian` + `dag.transforms.to_joint_gaussian()` stay as optimisation hooks. A closed-form LG conditioning impl can be added behind the same `dag.inference.query(...)` surface in v2.x without changing the public class layout.

#### Counterfactual reasoning (SCM layer)

`dag.counterfactual` implements Pearl's three-step procedure on any DAG whose CPDs implement `StructuralCPD` and advertise `supports_counterfactual=True`.

**Algorithm** — `dag.counterfactual.query(observed, do, query, n_samples=None, noise_overrides=None) → QueryResult`:

1. **Capability check.** Every node on the abduction path must advertise `supports_counterfactual=True`; otherwise raise `IncompatibleCPDError`.
2. **Abduction.** Topological iterate; call `cpd.abduct(x=observed[node], parents=observed[parents])`. Invertible CPDs (LG, ANM-wrapped, PNL-wrapped) return a `Delta`. Non-invertible (Tabular inverse-CDF, Functional) return a noise posterior; we draw `n_samples` from it.
3. **Action.** Override structural function for each `do` variable. If `do` is a list, repeat 3–4 per intervention (shared abducted noise → twin/parallel worlds).
4. **Prediction.** Topological iterate; call `cpd.structural_predict(parents=cf_parents, noise=abducted_u)`.

**Return type — `QueryResult`:**

```python
@dataclass
class QueryResult:
    samples: np.ndarray            # (n_samples,); invertible cases use n_samples=1
    query: str | list[str]
    operation: str                 # "counterfact" | "intervene" | "predict"
    operation_args: dict           # {"observed": ..., "do": ...} for counterfact

    def point(self) -> float: ...
    def distribution(self) -> pd.Series: ...
    def expectation(self, fn) -> float: ...
    def credible_interval(self, level=0.95): ...
    def compare_to(self, other) -> dict: ...   # Wasserstein, |Δmean|
```

For invertible-path queries, `samples` is shape `(1,)`. When the wrapped regressor is Bayesian, `samples` automatically becomes `(n_samples,)` — parameter uncertainty propagates without bootstrap.

**Per CPD class:**

| CPD | `structural_predict` | `abduct` (→ `NoiseDistribution`) | result shape |
|---|---|---|---|
| `LinearGaussianCPD` | `β₀ + β·pa + u` | `Delta(x − (β₀ + β·pa))` | `(1,)` — point-invertible |
| `WrappedRegressor(reg)` *ANM* | `reg.predict(pa) + u` | `Delta(x − reg.predict(pa))` | `(1,)` for point-predict reg; `(n,)` for Bayesian |
| `WrappedRegressor(reg, link, link_inv)` *PNL* | `link(reg.predict(pa) + u)` | `Delta(link_inv(x) − reg.predict(pa))` | `(1,)` for point-predict reg |
| `TabularCPD` (inverse-CDF) | `F⁻¹(P(X\|pa), u)` | `TruncatedUniform([F(x−1\|pa)],[F(x\|pa)])` | `(n_samples,)` |
| `FunctionalCPD` (Pyro) | model body | `PyroNoise(...)` (poutine wrap) | `(n_samples,)` |

**Out of scope for v2.0 (deferred to v2.x):** Gumbel-Max encoding; full identification algorithms (do-calculus/ID); mediation / path-specific / nested counterfactuals.

This SCM layer makes pgmpy a contender against [DoWhy / `dowhy.gcm`](https://www.pywhy.org/dowhy/main/user_guide/modeling_gcm/index.html) for users who want SCM modelling **inside** a Bayesian-network framework.

#### DBN handling

`DynamicBayesianNetwork` inherits from `DAG` (unchanged). With DAG enriched, DBN automatically gains `_cpds` / `_parent_order` / the DAG accessors. DBN keeps temporal-aware overrides for `add_cpds(*cpds)` / `get_cpds(node, time_slice=...)` / `simulate`, since its CPD keys are `(node, time_slice)` tuples. The `_cpds` dict accepts tuple keys natively.

### Release staging

Additive 1.x minor releases ending in a v2.0 deletion cleanup. Downstream users get one release-cycle of `FutureWarning` notice.

| Release | What ships |
|---|---|
| **1.x.1** | New `pgmpy.parameterization` module. Pure addition. |
| **1.x.2** | DAG enrichment with accessors. Legacy `dag.add_cpds`/`get_cpds`/`remove_cpds`/`cpds` kept as `FutureWarning` shims. Typed BN classes become `FutureWarning` subclasses of `DAG`. `DAG.do(nodes)` renamed to `DAG.with_intervention(nodes)` with a `FutureWarning` alias. `LinearGaussianBayesianNetwork.simulate` override retired; LG sampling goes through the generic `DAG.simulate`. Factor-API audit converts internal call sites to `dag.transforms.cpd_as_factor` so the new `TabularCPD(ClassifierMixin, BaseEstimator)` doesn't break VE / BP / `to_markov_model`. New parameter estimators + inference algorithms. |
| **1.x.3** | `FutureWarning` on legacy `pgmpy.factors.*` CPDs. **Full SCM layer**: `StructuralCPD` protocol with unified `NoiseDistribution` (`Delta`/`NormalNoise`/`Empirical`/`TruncatedUniform`); built-in CPDs gain the protocol; single `WrappedRegressor` adapter (ANM + PNL); `CPDAdapter` auto-wrap; Pearl-ladder accessors `dag.inference`/`dag.intervene`/`dag.counterfactual` returning `QueryResult`; multi-world counterfactual (list-of-`do`); `dag.diagnostics.identifiability_report`; `dag.bootstrap.query` for fit-time CIs; standalone graph primitives under `dag.transforms`. `pgmpy.inference.CausalInference` becomes a `FutureWarning` shim; its `query(do=..., adjustment_set=...)` ports to `dag.intervene.query(adjustment_set=...)` and its 13 identification / adjustment / IV helpers port to `dag.diagnostics`. Readwrite update. Migration guide. |
| **2.0** | Delete legacy classes and `DAG.add_cpds`/etc. shims. Delete `pgmpy.inference.CausalInference` and the `DAG.do` alias. Mechanical release. |

### Breaking changes (pgmpy 2.0)

These v1.x names exist in 1.x and are deleted in 2.0. See `docs/source/migration-v2.rst`.

**Classes deleted:**
- `pgmpy.models.{DiscreteBayesianNetwork, LinearGaussianBayesianNetwork, FunctionalBayesianNetwork, BayesianNetwork}`.
- `pgmpy.factors.discrete.TabularCPD` (legacy — `DiscreteFactor` stays).
- `pgmpy.factors.continuous.LinearGaussianCPD` and the `pgmpy.factors.continuous` package.
- `pgmpy.factors.hybrid.FunctionalCPD` and the `pgmpy.factors.hybrid` package.
- `pgmpy.estimators.{MaximumLikelihoodEstimator, BayesianEstimator, EM}`.
- `pgmpy.inference.CausalInference` — `query(do=..., adjustment_set=...)` ports to `dag.intervene.query(do=..., adjustment_set=...)`; the identification / adjustment-set / IV helpers (`identification_method`, `get_all_backdoor_adjustment_sets`, `is_valid_backdoor_adjustment_set`, `get_all_frontdoor_adjustment_sets`, `is_valid_frontdoor_adjustment_set`, `get_minimal_adjustment_set`, `is_valid_adjustment_set`, `get_proper_backdoor_graph`, `get_ivs`, `get_conditional_ivs`, `get_total_conditional_ivs`, `get_scaling_indicators`, `estimate_ate`) port to `dag.diagnostics`.

**Methods deleted:**
- `DAG.add_cpds` / `get_cpds` / `remove_cpds` / `cpds` (replaced by `dag.parameters`).
- `DAG.do(nodes)` — renamed `DAG.with_intervention(nodes)` to disambiguate from the value-level `dag.intervene.query(do=...)`. 1.x ships both names; `DAG.do` emits `FutureWarning`. The semantics are unchanged (graph mutation: remove incoming edges to the listed nodes).

**API renames** (full table in migration guide):
- `bn.add_cpds(cpd)` → `dag.parameters.add(variable=..., cpd=...)`
- `TabularCPD("X", 2, values=...)` → `TabularCPD.from_values(variable_card=2, values=...)`
- `bn.predict(data)` → `dag.inference.predict(data)`
- `bn.to_markov_model()` → `dag.transforms.to_markov_model()`
- `bn.save(path)` → `dag.io.save(path)` / `DAG.load(path)`
- `dag.do(nodes)` → `dag.with_intervention(nodes)` (structural — removes incoming edges to `nodes`; distinct from value-level `dag.intervene.query(do=...)`).
- `CausalInference(model).query(["Y"], do={...}, evidence={...}, adjustment_set=[...])` → `dag.intervene.query(do={...}, query="Y", evidence={...}, adjustment_set=[...])`.
- `CausalInference(model).identification_method(X, Y)` → `dag.diagnostics.identification_method(X, Y)`; same renaming pattern for all other adjustment / IV helpers listed above.

### Out-of-scope (non-goals)

- **Discrete children via skpro.** skpro is regression-only. Discrete children use `TabularCPD` or any sklearn-classifier-style estimator.
- **Posterior over parents under continuous-black-box CPDs.** Black-box regressors can't Bayes-invert analytically; LW degenerates for deep continuous evidence. Full MCMC is deferred.
- **Mixed-type parent preprocessing.** Continuous-child CPDs with discrete parents need one-hot / label encoding. User encodes upfront or uses `sklearn.Pipeline` / `ColumnTransformer`; default `sklearn.compose` integration is a follow-up.
- **General MCMC inference** (Gibbs, NUTS, HMC). Deferred.
- **Torch backend** for `TabularCPD`/`LinearGaussianCPD`. Numpy-only.
- **`bn.fit_update(data)` online update.** Follow-up; not folded into a `DiscreteBayesianEstimator` replacement.
- **`NoisyOR` port.** Stays in `pgmpy.factors.discrete` for 1.x; ported as a follow-up.
- **Gumbel-Max discrete encoding.** v2.0 ships inverse-CDF only; Gumbel-Max is a future explicit alternative because it can imply different counterfactuals.
- **Full identification-aware counterfactual inference.** v2.0 computes Pearl's abduction-action-prediction for a specified SCM and ships static diagnostics; full ID / do-calculus identification analysis is deferred.
- **Mediation / path-specific / nested counterfactuals.** v2.0 ships the basic surface; refinements are follow-ups built on the same protocol.

### Open risks

- **CPD ecosystem surface sufficiency.** Relies on sklearn/skpro CPDs providing enough of `fit`/`predict_proba`/`sample`/`log_prob`/`get_tag`/`clone` for participating operations. The DAG does not rely on skbase clone.
- **DAG copy semantics.** Bootstrap/refit rely on `copy_template(parameters=...)`, not `clone()`. Copy must preserve graph structure, latents, inferred schema, parent order, and CPD specs or fitted state per mode.
- **Schema inference conflicts.** Schema comes from CPDs, data, pandas categoricals, read/write formats. Conflicts fail loudly with variable-named errors (no silent reordering).
- **`is_fitted` naming difference.** sklearn `is_fitted_` vs skpro `_is_fitted`. Documented per class.
- **Migration cost.** isinstance-on-CPD-type is pervasive in current code. 1.x rollout keeps both paths; v2.0 cleanup is mechanical but touches many files.
- **Performance regression from pandas I/O.** Per-row pandas in `sample`/`log_prob` may be measurably slower than numpy-internal. Mitigation: benchmark on `bnlearn/alarm`-scale before each release; add numpy fast paths inside built-in CPDs if needed.

### User journeys

**1. Discrete BN, new idiomatic API.**

```python
from pgmpy.base import DAG
from pgmpy.parameterization import TabularCPD

student = DAG([("diff", "grade"), ("intel", "grade")])
student.parameters.add(variable="diff",  cpd=TabularCPD.from_values(variable_card=2, values=[[0.6], [0.4]]))
student.parameters.add(variable="intel", cpd=TabularCPD.from_values(variable_card=2, values=[[0.7], [0.3]]))
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

**2. Hybrid BN with a third-party skpro regressor.**

```python
from skpro.regression.ensemble import NGBoostRegressor

bn = DAG([("smoker", "income"), ("age", "income")])
bn.parameters.add(variable="smoker", cpd=TabularCPD(variable_card=2))
bn.parameters.add(variable="age",    cpd=TabularCPD(variable_card=5))
bn.parameters.add(variable="income", cpd=NGBoostRegressor())   # auto-wrapped by CPDAdapter
bn.fit(data)
bn.inference.query(evidence={"smoker": "yes", "age": "30-40"}, query="income")
```

`VariableElimination(bn)` would raise `TypeError` here because `NGBoostRegressor` does not advertise `produces_factor=True`. `LikelihoodWeighting(bn)` is the right choice for continuous evidence.

**3. Bayesian fitting via CPD hyperparameter** (no separate estimator).

```python
dag.parameters.add(variable="diff",  cpd=TabularCPD(variable_card=2, prior_type="BDeu", equivalent_sample_size=10))
dag.parameters.add(variable="intel", cpd=TabularCPD(variable_card=2, prior_type="BDeu"))
dag.parameters.add(variable="grade", cpd=TabularCPD(variable_card=3, evidence_card=[2, 2], prior_type="BDeu"),
                    parent_order=["diff", "intel"])
dag.fit(data)   # MLEEstimator → per-node cpd.fit() → applies BDeu prior internally
```

Same pattern as skpro: `GLMRegressor()` frequentist, `BayesianLinearRegressor(prior=...)` Bayesian.

**4. Linear-Gaussian network with generic inference.**

```python
dag = DAG([("x1", "x2"), ("x2", "x3")])
dag.parameters.add(variable="x1", cpd=LinearGaussianCPD.from_values(beta=[1.0], std=2.0))
dag.parameters.add(variable="x2", cpd=LinearGaussianCPD.from_values(beta=[-5, 0.5], std=2.0), parent_order=["x1"])
dag.parameters.add(variable="x3", cpd=LinearGaussianCPD.from_values(beta=[4, -1], std=1.7), parent_order=["x2"])

mu, cov = dag.transforms.to_joint_gaussian()
dag.inference.query(query="x3", evidence={"x1": 2.0})
```

**5. Counterfactual on a structural causal model.**

User observes `(X=1, Y=3, Z=10)` and asks: had X been 0, what would Z have been? Needs Pearl's abduction-action-prediction.

```python
# SCM: X = U_X; Y = 2*X + U_Y; Z = 3*Y + U_Z; all noise ~ Normal.
scm = DAG([("X", "Y"), ("Y", "Z")])
scm.parameters.add(variable="X", cpd=LinearGaussianCPD.from_values(beta=[0.0],      std=1.0))
scm.parameters.add(variable="Y", cpd=LinearGaussianCPD.from_values(beta=[0.0, 2.0], std=0.5), parent_order=["X"])
scm.parameters.add(variable="Z", cpd=LinearGaussianCPD.from_values(beta=[0.0, 3.0], std=0.3), parent_order=["Y"])

result = scm.counterfactual.query(observed={"X": 1.0, "Y": 3.0, "Z": 10.0}, do={"X": 0.0}, query="Z")
result.point()                  # → 4.0 (closed form via point-invertible LG path)
result.credible_interval(0.95)  # → (4.0, 4.0) — point in pure-LG case

# Multi-world: same abducted noise, three interventions.
worlds = scm.counterfactual.query(observed={"X": 1, "Y": 3, "Z": 10}, do=[{"X": -1}, {"X": 0}, {"X": 1}], query="Z")
[w.point() for w in worlds]   # → [-2.0, 4.0, 10.0]

# Identifiability check warns: pure linear-Gaussian → non-identified.
scm.diagnostics.identifiability_report()
# {"warnings": [{"type": "linear_gaussian_path", ...}], "n_warnings": 1}
```

Contrast with the *interventional* query `E[Z | do(X=0)] = 0` (prior-marginal, ignores observation). The counterfactual uses the abducted noise that *would have produced* the observed state, and therefore differs.

When the wrapped regressor is Bayesian (or the path contains non-invertible CPDs), `result.samples` is a real distribution; `.credible_interval`/`.compare_to` quantify the uncertainty. Swapping `LinearGaussianCPD` for `WrappedRegressor(BayesianRidge())` propagates parameter uncertainty into the counterfactual without an extra bootstrap call.
