## Pairwise Causal Discovery for pgmpy

Contributor: [@hanara2112](https://github.com/hanara2112) (Aryaman Bahl, IIIT Hyderabad)
Mentors: ankurankan et al.

### Introduction

pgmpy currently provides constraint-based (PC) and score-based (HillClimbSearch, GES) algorithms for multivariate causal discovery. In the bivariate setting, causal direction is not identifiable from observational data alone: X -> Y and Y -> X produce the same observed correlation. Recovering direction requires additional assumptions on the data-generating process, such as functional asymmetry, noise structure, temporal ordering, or interventions.

Pairwise causal discovery methods introduce such assumptions to make direction identifiable. Examples include additive-noise models (ANM), IGCI, and LiNGAM, used both as standalone estimators and as orientation primitives in larger pipelines. This proposal adds ANM, IGCI, and Bivariate LiNGAM to pgmpy; refactors the existing `llm_pairwise_orient` into an estimator (`LLMPairwise`) following the same contract; extends `ExpertInLoop` to accept arbitrary pairwise estimators; and adds benchmark evaluation utilities (`PairwiseAccuracy`, Tübingen weight loading) following pgmpy's existing causal-discovery API.

### Proposed Solution

All four estimators — `ANM`, `IGCI`, `BivariateLiNGAM`, and `LLMPairwise` — subclass `_BaseCausalDiscovery` directly, with no intermediate pairwise base class (the algorithms share little internally, so an extra layer would mostly be ceremony). Once fit, each one stores its result on the same `causal_graph_` and `adjacency_matrix_` attributes pgmpy's other structure learners already use, plus a small `predicted_direction_` reader over `causal_graph_` for the natural bivariate question of which way the arrow points. The two-column check lives inside each `_fit`, since the shared `_check_fit_data` also runs for multivariate algorithms and can't enforce a bivariate shape itself. The three statistical estimators override `__sklearn_tags__` to reject categorical input; `LLMPairwise` doesn't, since it reads variable names rather than values. All four accept `show_progress` and `random_state`, matching `PC` and `ExpertInLoop`.

The four estimators differ in what they assume about the data and how they pick a direction:

- **ANM** (Hoyer et al., 2009): assumes Y = f(X) + E with noise E independent of X. The reverse direction generally cannot be written this way. Fits a nonparametric regression in each direction, then picks the direction whose residuals look most independent of the input. Accepts any continuous CI test for the independence check (HSIC by default; KCI, GCM, Pearsonr also work); discrete-only tests like chi-square are rejected at fit time.
- **IGCI** (Janzing et al., 2012): assumes Y is a near-deterministic function of X, with P(X) and f chosen independently of each other. Compares a slope-based or entropy-based statistic in both directions; the one where this "independence of cause and mechanism" holds wins. Fast (O(n log n), no kernels or bandwidths), but degrades quickly when noise is present.
- **Bivariate LiNGAM** (Hyvärinen and Smith, 2013): assumes Y = β·X + E with non-Gaussian noise E independent of X; non-Gaussianity is what makes the direction identifiable. Runs one OLS fit, then scores both directions with the Hyvärinen–Smith likelihood ratio under a configurable nonlinearity g (default `np.tanh`). O(n), no hyperparameters to tune.
- **LLMPairwise** (refactor of `llm_pairwise_orient`): the only non-statistical method here. It ignores the numeric data and asks an LLM which direction is more likely, given the variable names and optional text descriptions. Useful when domain knowledge is in the names but the data are too noisy or scarce for a statistical method.

Putting all four behind the same `_BaseCausalDiscovery` contract lets `ExpertInLoop` accept any of them via a single `pairwise_estimator=` keyword, replacing the current `orientation_fn` + `functools.partial` workflow.

CI tests will follow pgmpy's existing dual-form convention (`ANM(ci_test="hsic")` or `ANM(ci_test=HSIC(...))`); regressors, kernel approximators, and LiNGAM nonlinearities are passed as instances/callables directly, matching sklearn convention. This enables sklearn-style pipelines and hyperparameter search while keeping the common case ergonomic.

**Reused pgmpy infrastructure:**

- `_BaseCausalDiscovery` and `_check_fit_data` for the standard `fit` contract and input validation;
- `pgmpy.base.DAG` for fitted causal graphs (no custom graph abstraction is introduced);
- `pgmpy.ci_tests` for residual-independence testing in ANM and optional LiNGAM scoring; HSIC (PR #3254) is the default;
- `pgmpy.datasets.load_dataset("tubingen/<pair_id>")` for benchmark evaluation;
- `pgmpy.metrics._BaseSupervisedMetric` as the basis for `PairwiseAccuracy`.

`PairwiseAccuracy` is a thin subclass of `_BaseSupervisedMetric`: `evaluate(true, est)` returns `1.0` if the predicted cause→effect ordering matches the ground-truth edge, and `0.0` otherwise. The base class's `evaluate` only handles one graph pair at a time, so weighted accuracy and AUC across the Tübingen benchmark go through a separate `evaluate_many(true_graphs, est_graphs, weights=None, scores=None)` that returns all three numbers at once. Whether `evaluate_many` belongs on `PairwiseAccuracy` or moves up to `_BaseSupervisedMetric` so other metrics can reuse it is something I'd like to settle with mentors in Phase 1.

### Alternative Solutions

*Function-based orientation wrappers.*
One possible design is to expose helper functions such as `anm_pairwise_orient(x, y, data)` and plug them into `ExpertInLoop` similarly to the current `llm_pairwise_orient` workflow. This approach was rejected in favor of estimator composability: adding one wrapper per method does not scale well and duplicates integration logic. Instead, all pairwise methods in this proposal follow the same estimator contract (`fit`, `causal_graph_`, etc.), allowing `ExpertInLoop` to consume arbitrary initialized estimator instances directly.

*Introduce a shared `_BasePairwiseDiscovery` base class.*
An earlier version of this proposal added a thin pairwise base class with helper methods for direction scoring shared across ANM, IGCI, and LiNGAM. This was rejected during review on the grounds that (a) the algorithms differ substantially internally, so shared scaffolding would be thin and not meaningfully reduce duplication, and (b) such an abstraction is easier to extract later from working code than to design upfront. Each estimator therefore subclasses `_BaseCausalDiscovery` directly and implements its own `_fit`.

*Implement only ANM.*
ANM-pHSIC is one of the strongest individual methods on the Tübingen benchmark, with a weighted accuracy of 63 ± 10% (Mooij et al., 2016, Sec. 6). However, its assumptions fail precisely in regimes where IGCI (low-noise deterministic settings) and LiNGAM (linear non-Gaussian settings) perform best.

#### File layout

**New files**

- `pgmpy/causal_discovery/{ANM,IGCI,BivariateLiNGAM,LLMPairwise}.pypgmpy/metrics/pairwise_accuracy.pypgmpy/tests/test_causal_discovery/test_{ANM,IGCI,BivariateLiNGAM,LLMPairwise,ExpertInLoop_pairwise}.py`
- `pgmpy/tests/test_metrics/test_pairwise_accuracy.py`
- `examples/Pairwise_Causal_Discovery.ipynb`

**Modified files**

- `pgmpy/causal_discovery/__init__.py` — re-export the four new estimators
- `pgmpy/causal_discovery/ExpertInLoop.py` — add `pairwise_estimator=` keyword
- `pgmpy/datasets/_base.py` — parse Tübingen `pairmeta.txt` per-pair weights and expose them via `load_dataset("tubingen/<id>")`, so callers use weights without metric coupling

### Algorithm details

**Defaults and key knobs**

- **ANM**: `GaussianProcessRegressor` for nonlinear regression by default; `Nystroem + Ridge` available for larger data. Optional train/test split for more conservative residual estimation.
- **IGCI**: slope-based and entropy-based scoring both supported; configurable reference measure (`"uniform"` or `"gaussian"`); ties handled defensively to avoid finite-difference instability.
- **Bivariate LiNGAM**: `nonlinearity=` accepts any callable (default `np.tanh`; `np.exp`-based variant also supported per the original paper). Optional CI-test-based scoring via `score_method=HSIC()`. Raises a Gaussianity warning when Shapiro–Wilk on residuals fails, since identifiability degrades under Gaussian noise.

**Diagnostics**

Beyond the standard `causal_graph_`, `adjacency_matrix_`, and `predicted_direction_`, each estimator exposes:

- `direction_score_: float` — `S(Y→X) − S(X→Y)`, where `S` is the method-specific score (smaller = preferred). Positive means the forward direction wins.
- `confidence_: float` in `[0, 1]` — normalized score margin (`|direction_score_| / (|S_forward| + |S_reverse|)`) for IGCI and LiNGAM; HSIC p-value gap for ANM. Values near 0 indicate weak identifiability or violated assumptions.

Method-specific introspection attributes (the actual fitted models are exposed so users can inspect them):

- **ANM**: `regressor_forward_/backward_`, `residuals_forward_/backward_`, `hsic_p_forward_/backward_`.
- **IGCI**: `forward_C_`, `backward_C_`, `reference_measure_`.
- **BivariateLiNGAM**: `beta_forward_/backward_`, `lr_statistic_`, `gaussianity_p_forward_/backward_`, `nonlinearity_`.
- **LLMPairwise**: `prompt_`, `response_` (for auditability).

Benchmark evaluation follows Mooij et al. (2016): weighted accuracy on the Tübingen pairs using benchmark-provided per-pair weights (which exclude known-confounded pairs).

### Testing Plan

**Direction recovery** (each estimator against the assumption it makes):

- ANM: nonlinear additive-noise synthetic data, plus the Tübingen benchmark.
- IGCI: near-deterministic low-noise synthetic data, plus Tübingen.
- Bivariate LiNGAM: linear non-Gaussian synthetic data, plus Tübingen.

Accuracy is reported across ≥5 seeds and at two sample sizes (n=200, n=2000) to cover both small-sample stability and asymptotic behaviour.

**Edge cases:** constant inputs, NaN/inf, insufficient samples, tied observations (IGCI), Gaussian-residual warning (LiNGAM), reproducibility under fixed `random_state`.

**Integration:** `ExpertInLoop` driven with each of the four estimators passed as `pairwise_estimator=`, including the `orientation_cache_` reuse path so `LLMPairwise` is not re-queried for cached edges.

**Benchmark:** weighted accuracy and AUC on the Tübingen Cause–Effect Pairs via `PairwiseAccuracy.evaluate_many`, compared against the numbers reported in Mooij et al. (2016).

**Metric:** `PairwiseAccuracy.evaluate` and `evaluate_many` validated against small hand-computed examples, covering the unweighted, weighted, and AUC paths.

### User Journeys

**1. Pairwise causal discovery on a single variable pair**

```python
from pgmpy.causal_discovery import ANM
from pgmpy.datasets import load_dataset

pair = load_dataset("tubingen/1").data

est = ANM().fit(pair)

est.predicted_direction_
est.confidence_
est.causal_graph_
```

The same interface applies to `IGCI`, `BivariateLiNGAM`, and `LLMPairwise`.

---

**2. Swapping components (CI test, regressor, nonlinearity)**

```python
from pgmpy.causal_discovery import ANM
from pgmpy.ci_tests import KCI
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

est = ANM(
    ci_test=KCI(),
    regressor=make_pipeline(Nystroem(n_components=200), Ridge()),
).fit(pair)
```

Estimators, CI tests, and regressors compose via standard sklearn idioms, so users can swap in faster kernel approximations or alternative independence tests without touching pgmpy internals.

---

**3. Using pairwise estimators inside `ExpertInLoop`**

```python
from pgmpy.causal_discovery import ExpertInLoop, ANM

dag = ExpertInLoop(
    pairwise_estimator=ANM()
).fit(data)
```

Since all pairwise methods follow the same estimator contract, statistical and LLM-based orientation methods can be used interchangeably.

---

**4. Benchmark evaluation on the Tübingen dataset**

```python
from pgmpy.causal_discovery import ANM
from pgmpy.datasets import load_dataset
from pgmpy.metrics import PairwiseAccuracy

metric = PairwiseAccuracy()

truths, preds = [], []

for pair_id in range(1, 109):
    ds = load_dataset(f"tubingen/{pair_id}")

    if ds.data.shape[1] != 2:
        continue

    est = ANM().fit(ds.data)

    truths.append(ds.ground_truth)
    preds.append(est.causal_graph_)

metric.evaluate_many(truths, preds)
```

### Timeline

About 350 hours over 14 weeks (12 GSoC weeks plus a 2-week buffer).

- **Weeks 1-2 (50 h):** Finalize shared pairwise API, utilities, and test plan.
- **Weeks 3-5 (100 h):** Implement ANM with GPR + fast Nystroem/Ridge backend; benchmark on Tübingen.
- **Weeks 6-7 (50 h):** Implement IGCI with slope and entropy scoring.
- **Weeks 8-10 (100 h):** Implement Bivariate LiNGAM with Hyvärinen–Smith scoring and optional HSIC scoring.
- **Weeks 11-12 (30 h):** Unified integration with `LLMPairwise` and `ExpertInLoop`.
- **Weeks 13-14 (20 h):** Final benchmarks, Tübingen parser, evaluation utilities, and tutorial notebook.
- **Total:** 350 hours.

### References

- Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *Advances in Neural Information Processing Systems*, 21. [paper](https://proceedings.neurips.cc/paper/2008/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html)
- Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J., Daniušis, P., Steudel, B., & Schölkopf, B. (2012). Information-geometric approach to inferring causal directions. *Artificial Intelligence*, 182–183, 1–31. [doi:10.1016/j.artint.2012.01.002](https://doi.org/10.1016/j.artint.2012.01.002)
- Hyvärinen, A., & Smith, S. M. (2013). Pairwise likelihood ratios for estimation of non-Gaussian structural equation models. *Journal of Machine Learning Research*, 14, 111–152. [JMLR](https://www.jmlr.org/papers/v14/hyvarinen13a.html)
- Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., & Schölkopf, B. (2016). Distinguishing cause from effect using observational data: methods and benchmarks. *Journal of Machine Learning Research*, 17(32), 1–102. [arXiv:1412.3773](https://arxiv.org/abs/1412.3773)
