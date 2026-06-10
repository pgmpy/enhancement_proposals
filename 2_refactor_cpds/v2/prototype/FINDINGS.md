# Prototype findings

A standalone runnable prototype validating the v2.0 parameterization
refactor design. Run with:

```sh
PYTHONPATH=/path/to/skpro python prototype.py
```

## Demos and results

8 focused demos organized into two user journeys (~1800 lines total).

### Section A — Classic Bayesian Network user

| # | Demo | Result |
|---|---|---|
| A1 | Discrete BN — full lifecycle (build → fit → simulate → template copy) | ✅ PASS — `P(grade=A \| diff=easy, intel=high) = 0.894` vs truth 0.90; marginals within ±0.01; `dag.parameters is dag.parameters` (cached); `dag.copy_template(parameters="unfit")` produces fresh equivalent DAG with inferred schema |
| A2 | Hybrid BN — `skpro.GLMRegressor` + `sklearn.RandomForestClassifier` + `sklearn.Pipeline` as CPDs | ✅ PASS — three heterogeneous CPDs fit through one `MLEEstimator`; forward sampling matches truth to 1%; `sklearn.Pipeline(StandardScaler, RandomForestClassifier)` auto-wrapped by `CPDAdapter`. (No automatic string→numeric encoding for parents; user encodes upfront or uses sklearn Pipeline / ColumnTransformer.) |
| A3 | `dag.inference.query()` returning `QueryResult` | ✅ PASS — `P(grade \| diff=easy, intel=high)` recovers `P(A)=0.893` (truth 0.90); LW with evidence-clamping; `.compare_to()` confirms different conditioning gives different result. Rung 1 mapped to `dag.inference`. |

### Section B — Causal inference user

| # | Demo | Result |
|---|---|---|
| B1 | SCM — `E[Z]` (rung 1) vs `E[Z\|do(X=0)]` (rung 2) | ✅ PASS — associational `E[Z] ≈ 12.0` via `dag.inference.query`; interventional `E[Z\|do(X=0)] ≈ 0.0` via `dag.intervene.query`; wasserstein ≈12. Pearl rungs 1 and 2 served through uniform accessors. |
| B2 | Counterfactual + multi-world | ✅ PASS — `QueryResult.point() = 4.0` exact analytic match on 3-node LG SCM; `.meta["abducted_noise"] = {X:1, Y:1, Z:1}`; multi-world (do as list) returns three QueryResults `(do(X=-1)→-2.0; do(X=0)→4.0; do(X=1)→10.0)` sharing abducted noise |
| B3 | `WrappedRegressor` — single class for ANM and PNL | ✅ PASS — ANM (link=None) and PNL (link=tanh, link_inv=arctanh) both recover LR coef ≈ 2.0; PNL abduction round-trip matches to 1e-10. Protocol-over-hierarchy validated. |
| B4 | `dag.diagnostics.identifiability_report()` + standalone graph primitives | ✅ PASS — pure-LG chain flagged with Hoyer-2009 reference; PNL correctly NOT flagged (Zhang-Hyvärinen 2009); `dag.transforms.{topological_order, ancestors, descendants, markov_blanket, d_separated}` all functional |
| B5 | `dag.bootstrap.query` — fit-time CIs | ✅ PASS — 100 refits on n=300 data → credible interval `(3.772, 4.066)` around point `3.928`; non-trivial parameter uncertainty captured |

## What the prototype validates

- **Identity-free CPDs** with `_tags` work cleanly. `TabularCPD(ClassifierMixin, BaseEstimator)` and `LinearGaussianCPD(BaseProbaRegressor)` both follow their upstream contracts.
- **DAG is not an estimator.** `DAG` inherits from `nx.DiGraph` only; CPDs keep the sklearn/skpro-style API. Refit workflows use `dag.copy_template(...)`.
- **DAG-owned schema** is internal and auto-populated. `TabularCPD(..., state_names=...)` stays the simple user path; `dag.parameters.add(...)` normalizes state metadata into `dag.schema` under the precedence policy (CPD-authored > pandas categorical > observed > cardinality fallback).
- **`cached_property` accessors** return the same instance across calls.
- **`MLEEstimator`** orchestrates per-node `cpd.fit(X, y)` across Tabular / LinearGaussian / third-party skpro / sklearn / sklearn.Pipeline.
- **Operation-specific CPD protocols** (`FittableCPD`, `SampleableCPD`, `ScorableCPD`, `PredictiveCPD`) replace a single universal validator. Runtime operations check only the surface they consume.
- **`CPDAdapter` dispatch** routes through native `sample`/`log_prob` when defined, else `predict_proba(X).sample()` / `.log_pdf()` (skpro distributions), else `predict_proba(X)` matrix indexing (sklearn classifiers).
- **Third-party estimators drop straight in** as CPDs: `dag.parameters.add(variable="y", cpd=GLMRegressor(), parent_order=["x"])` → `dag.fit(data)` → `dag.simulate(...)` works.

## Gaps the prototype found, with actions for the plan

| # | Gap | Action |
|---|---|---|
| 1 | `_DAGParameters.__getitem__` was missing from the original spec | Added to contracts; `dag.parameters["node"]` works |
| 2 | Original design proposed `DAG(nx.DiGraph, skbase.BaseObject)` with `clone()` semantics; mutation breaks this | DAG no longer inherits from skbase; `dag.copy_template(parameters="none"|"unfit"|"fitted")` is the pgmpy-native copy contract |
| 3 | Third-party estimators require numeric parents (sklearn `ValueError: could not convert string to float`) | Confirmed limitation; spec calls out "mixed-type parent preprocessing" as user's responsibility. Recommended pattern: `sklearn.Pipeline(OneHotEncoder, ...)` or `ColumnTransformer`. `dag.schema` is the future extension point for default encoders |
| 4 | Mixture & NN CPDs (`MixtureCPD` with EM, sklearn `MLPClassifier`) — earlier iterations had standalone demos | Compressed into Demos A2 (RF) and B3 (WrappedRegressor); the framework-agnostic claim holds. Joint end-to-end NN training is deliberately out of scope (Pyro/NumPyro territory) |
| 5 | skpro `BaseProbaRegressor` tag inheritance fragile — subclasses must re-declare `X_inner_mtype`, `y_inner_mtype`, `capability:multioutput`, `capability:missing` | Added to per-CPD class authoring checklist. Not pgmpy-specific (MDNRegressor does the same) |
| 6 | `is_fitted` (skpro, property tied to `_is_fitted`) vs `is_fitted_` (sklearn). `from_values` must set the right convention per CPD class | Documented per CPD class. `TabularCPD.from_values` → `is_fitted_ = True`; `LinearGaussianCPD.from_values` → `_is_fitted = True` |
| 7 | Schema state-ordering conflict between CPD-declared and data-derived orders | Resolved via explicit precedence policy (CPD-authored states are authoritative; data validates subset membership only). See `02_parameterized_dag.md` "Schema precedence policy" |

## What the prototype did NOT validate

Out of prototype scope but covered by the plan:

- `FunctionalCPD` (pyro), `JointPyroEstimator`.
- `VariableElimination` / `BeliefPropagation` tag dispatch (pgmpy-internal).
- `_DAGTransforms.to_markov_model` / `to_joint_gaussian` (need DiscreteFactor + inference machinery).
- `_DAGIO` save/load + readwrite (BIF, XMLBIF, UAI).
- DBN handling (temporal-slice key tuples).
- Centralised `cpd.get_tag` defaulting policy across third-party shapes.

## API simplifications validated

| # | Simplification | Validated in |
|---|---|---|
| 1 | Unified `NoiseDistribution` types (`Delta`, `NormalNoise`, `Empirical`, `TruncatedUniform`) | Counterfactual algorithm calls `.sample(n)` uniformly across LG/Tabular/Functional |
| 2 | Single `WrappedRegressor` for ANM and PNL (replaces ANMWrapper + PNLWrapper) | Demo B3 — one class, two link modes, identical abduction code |
| 3 | Reduced tag set (11 → 7; dropped `noise_invertible`, `parameter_uncertainty`, `is_mixture`, `supports_analytic_conditioning`) | Every CPD class uses the reduced set |
| 4 | `dag.diagnostics` accessor (was `dag.counterfactual.identifiability_report`) | Demo B4 |
| 5 | Adapter-local dispatch (`CPDAdapter` owns predict-proba → sample/log_prob bridge) | Built-ins pass through unwrapped; third-party predictive estimators adapted at registration |

## Competitive borrows validated

| Change | Borrowed from | Validated in |
|---|---|---|
| Pearl rungs → 3 accessors returning `QueryResult` | ChiRho effect handlers; pyAgrum CausalModel | Demos A3, B1, B2 |
| `dag.intervene` accessor | ChiRho `do()` primitive | Demo B1 |
| `dag.inference.query()` returning `QueryResult` | sktime composable wrappers | Demo A3 |
| `CPDAdapter` auto-wrap | cleaner than DoWhy's `PredictionModel` | Demo A2 |
| `sklearn.Pipeline` as CPD | own design | Demo A2 |
| Multi-world counterfactual via list-of-`do` | ChiRho `MultiWorldCounterfactual` | Demo B2 |
| Standalone graph primitives on `dag.transforms` | causal-learn / R6causal | Demo B4 |
| `dag.bootstrap` for fit-time CIs | `dowhy.gcm.bootstrap_sampling` | Demo B5 |

Dropped: `LinearGaussianInference` and `LinearGaussianCounterfactual` —
the generic abduction loop is already closed-form correct for LG SCMs
(Demo B2 hits 0.0 error). `is_linear_gaussian` tag stays as a future
optimisation hook.

## Verdict

The core design is sound. All 8 demos pass.

1. **Two user journeys** map cleanly to Pearl's three rungs. Section A uses `dag.inference`; Section B walks rungs 2 + 3 plus diagnostics and bootstrap.
2. **All three rungs return `QueryResult`** uniformly — same `.point()` / `.distribution()` / `.credible_interval()` / `.compare_to()` vocabulary.
3. **skpro + sklearn + sklearn.Pipeline integration works** at the same contract level (Demo A2).
4. **Counterfactual reasoning is closed-form correct** on LG SCMs (Demo B2: 0.0 error; multi-world recovers all three analytic answers; B3: PNL abduction to 1e-10).
5. **Fit-time uncertainty** captured by `dag.bootstrap` (Demo B5).
6. Competitive borrows + API simplifications applied as designed.

No design-level surprises. The plan is implementable as written.
