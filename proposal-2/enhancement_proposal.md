## Pairwise Causal Discovery for pgmpy

Contributor: [@hanara2112](https://github.com/hanara2112) (Aryaman Bahl, IIIT Hyderabad)
Mentors: ankurankan et al.

### Introduction

pgmpy's structure-learning algorithms (PC, GES, MMHC) work by reading off conditional independencies from the data. In the bivariate case there are no conditioning sets to test, so the joint P(X, Y) factors both ways and the direction X -> Y vs Y -> X is unidentifiable from the joint alone.

The usual way around this is to restrict the structural model and pick the direction that fits the restriction better. ANM, IGCI and Bivariate LiNGAM are three of the most common such restrictions, with twenty-plus years of literature and benchmarks behind them. They show up in three places in practice:

- as standalone direction estimators for two-variable problems,
- as an edge-orientation step in PC / FCI for edges that no v-structure rule covers,
- as the pairwise primitive that DirectLiNGAM and similar multivariate algorithms call internally.

pgmpy has none of them today. It does have a natural place to slot them in. `ExpertInLoop` already accepts a pluggable `orientation_fn(x, y, ...) -> (source, target)`, defaulting to `llm_pairwise_orient`. The estimators in this proposal can serve as a statistical, no-LLM-required alternative for that hook, alongside their use as standalone estimators.

The HSIC marginal independence test that ANM depends on is in review as PR #3254; it is expected to land before GSoC coding starts.

### Proposed Solution

Three estimators in `pgmpy/causal_discovery/`, a small shared base class, and a benchmarking metric in `pgmpy/metrics/`.

All three algorithms reduce to comparing a directional score S(X -> Y) against S(Y -> X) and reporting the direction with the smaller score. The base class `_BasePairwiseDiscovery` handles that comparison and the fitted-attribute contract; subclasses implement one method:

```
_score_direction(cause: ndarray, effect: ndarray) -> float
```

The estimators inherit from `_BaseCausalDiscovery`, so they accept a `pd.DataFrame` via `fit` and populate `causal_graph_` (an `nx.DiGraph`) and `adjacency_matrix_` the same way `PC` and `GES` do.

#### Algorithms

**ANM** (Hoyer et al., 2009).
Model: Y = f(X) + E with E independent of X. The reverse direction generically does not admit an additive-noise representation. Algorithm: fit a nonparametric regression in each direction, take residuals, measure dependence of residuals on the input, pick the direction whose residuals are more independent. The dependence measure is HSIC. The merged `pgmpy.ci_tests.HSIC` is reused directly.

**IGCI** (Janzing et al., Artificial Intelligence, 2012).
Model: a near-deterministic Y = f(X) where P(X) and f are independent. Algorithm: compare a slope-based or entropy-based statistic in both directions; the direction satisfying the independence-of-cause-and-mechanism condition wins. No kernels, no bandwidth, O(n log n). Degrades quickly under noise.

**Bivariate LiNGAM** (Hyvarinen and Smith, JMLR 2013).
Model: Y = beta * X + E with E independent of X and non-Gaussian. Algorithm: one OLS fit, then the Hyvarinen-Smith likelihood-ratio statistic with a fixed nonlinearity g (default tanh). O(n), no hyperparameters of substance. This complements DirectLiNGAM (which is multivariate) rather than replicating it.

#### Supporting pieces

- **HSIC**: in review, reused by ANM as the residual-independence score.
- **Orientation adapters**: one thin function per estimator (`anm_pairwise_orient`, `igci_pairwise_orient`, `lingam_pairwise_orient`) in `pgmpy/utils/utils.py`, re-exported from `pgmpy/utils/__init__.py`. `ExpertInLoop` calls `orientation_fn(u, v)` with only the two variable names ([ExpertInLoop.py:370](pgmpy/causal_discovery/ExpertInLoop.py#L370)), so the data is bound at call site via `functools.partial`, exactly the way `llm_pairwise_orient` is used today. Each adapter has signature `(x, y, data, **kwargs) -> (source, target)` and `partial(anm_pairwise_orient, data=df)` is what gets passed in.
- **`PairwiseAccuracy`** metric in `pgmpy/metrics/`. Per-pair evaluation reuses the existing `SHD` / `CorrelationScore` shape -- `metric(true_causal_graph=DAG, est_causal_graph=DAG) -> {0, 1}`. Aggregation across a benchmark (weighted accuracy and weighted AUC) is exposed via a sibling `PairwiseAccuracy.evaluate_many(true_graphs, est_graphs, weights=None, scores=None)` helper, since `_BaseSupervisedMetric.evaluate` is intentionally single-graph and type-checks one DAG per call ([metrics/_base.py:23](pgmpy/metrics/_base.py#L23)). Whether the aggregation lives on `PairwiseAccuracy` or as a standalone `pairwise_benchmark` function is an API call I will confirm with mentors in Phase 1.

### Alternative Solutions

*Skip the base class.* The three algorithms share little internal logic, so technically nothing forces a shared base. I would still keep one, because the surface they expose is the same: two columns in, `predicted_direction_` / `direction_score_` / `confidence_` out. Without a base class three copies of that boilerplate drift apart over time. The class is around 60 lines.

*Implement only ANM.* ANM-pHSIC is one of the strongest individual methods on the Tubingen benchmark, with a weighted accuracy of 63 ± 10% (Mooij et al. 2016, Sec. 6). Its assumption fails on exactly the data that IGCI handles best (low-noise deterministic) and that LiNGAM handles best (linear non-Gaussian). Users who want pairwise orientation through `ExpertInLoop` benefit from being able to pick the method that fits their data regime, which is the main reason to ship all three.

*Wrap an existing library, e.g. `causal-learn`.* Possible, but it adds a heavy dependency for three short algorithms and mismatches pgmpy's API conventions (sklearn-style `fit`, `causal_graph_`, `adjacency_matrix_`). Native implementations come out to roughly 250-400 lines each including tests, which is small enough to maintain in tree.

### Details of Proposed Solution

#### File layout

```
pgmpy/
  causal_discovery/
    _base.py                       # APPEND _BasePairwiseDiscovery + helpers
    __init__.py                    # re-export ANM, IGCI, BivariateLiNGAM
    ANM.py                         # NEW
    IGCI.py                        # NEW
    BivariateLiNGAM.py             # NEW
  ci_tests/hsic.py                 # in review (PR #3254)
  utils/utils.py                   # ADD three *_pairwise_orient adapters
  utils/__init__.py                # re-export the three adapters
  datasets/_base.py                # ADD pairmeta.txt parser for Tubingen weights
  metrics/pairwise_accuracy.py     # NEW
  tests/
    test_causal_discovery/
      test_pairwise_base.py        # NEW
      test_ANM.py                  # NEW
      test_IGCI.py                 # NEW
      test_BivariateLiNGAM.py      # NEW
    test_metrics/
      test_pairwise_accuracy.py    # NEW
examples/Pairwise_Causal_Discovery.ipynb   # NEW
```

Three new estimator files plus one metric file. The base class and helpers are appended to the existing [`_base.py`](pgmpy/causal_discovery/_base.py) next to `_BaseCausalDiscovery`, `_ConstraintMixin`, and `_ScoreMixin` -- the same pattern pgmpy already uses to keep causal-discovery scaffolding consolidated. No new soft dependencies: NumPy, SciPy, and scikit-learn cover everything.

#### Reused vs new

Reused from pgmpy (only imports, no new code):

- `_BaseCausalDiscovery` and its `_check_fit_data` for input validation;
- `_BaseCausalDiscovery.score(X | true_graph, metric=...)` for downstream scoring;
- `pgmpy.ci_tests.HSIC` for ANM's residual-independence score;
- `pgmpy.base.DAG` for `causal_graph_` (same type as `PC` / `GES` produce);
- `pgmpy.datasets.load_dataset("tubingen/<pair_id>")` (pair IDs 1-108 are already supported; returns a `Dataset` with `.data` and `.ground_truth`);
- `pgmpy.metrics._BaseSupervisedMetric` as the parent of `PairwiseAccuracy`.

New helpers (appended to `pgmpy/causal_discovery/_base.py` alongside `_BasePairwiseDiscovery`, each unit-tested before the algorithms that use them):

- `_compare_directional_scores(s_fwd, s_bwd) -> (direction_idx, signed_diff)`. The single place where the score comparison lives.
- `_min_max_score_confidence(s_fwd, s_bwd) -> float in [0, 1]`. Normalises |s_fwd - s_bwd| relative to their sum.
- `_ksg_mi(x, y, k=7)`. Kraskov-Stogbauer-Grassberger kNN mutual information; used by Bivariate LiNGAM when `score_method="ksg"`.
- `_kl_entropy(x, k=5)`. Kozachenko-Leonenko kNN entropy; used by IGCI when `score_method="entropy"`.

All four are private (leading underscore) and live in the same module as the base class. No separate `_pairwise_utils.py` file is created.

No custom DAG class is introduced. Output graphs are built with `pgmpy.base.DAG([(u, v)])` and converted to an adjacency matrix with `nx.to_pandas_adjacency(..., weight=1, dtype="int")`, identical to how `HillClimbSearch`, `GES` and `PC` already do it.

#### Base class sketch

```python
# appended to pgmpy/causal_discovery/_base.py, next to _BaseCausalDiscovery
class _BasePairwiseDiscovery(_BaseCausalDiscovery):
    """Shared scaffolding for ANM, IGCI and Bivariate LiNGAM.

    Subclasses implement ``_score_direction(cause, effect) -> float``.
    Lower score means more consistent with cause -> effect.
    """

    def _fit(self, X: pd.DataFrame):
        # _check_fit_data (in the parent) already validates non-empty / finite /
        # hashable input. The check below tightens that to exactly 2 columns,
        # which is the only extra precondition pairwise estimators have.
        if X.shape[1] != 2:
            raise ValueError(
                f"Pairwise estimators require exactly 2 variables, "
                f"got {X.shape[1]}."
            )
        self.variables_ = list(X.columns)
        a, b = X.columns
        s_ab = self._score_direction(X[a].to_numpy(), X[b].to_numpy())
        s_ba = self._score_direction(X[b].to_numpy(), X[a].to_numpy())

        direction_idx, signed_diff = _compare_directional_scores(s_ab, s_ba)
        self.direction_score_ = signed_diff
        self.predicted_direction_ = (a, b) if direction_idx == 0 else (b, a)
        self.confidence_ = _min_max_score_confidence(s_ab, s_ba)

        u, v = self.predicted_direction_
        self.causal_graph_ = DAG([(u, v)])
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )
        return self

    def _score_direction(self, cause, effect) -> float:
        raise NotImplementedError
```

`causal_graph_` is a `pgmpy.base.DAG`, the same type `PC`, `GES`, and `HillClimbSearch` populate ([HillClimbSearch.py:249](pgmpy/causal_discovery/HillClimbSearch.py#L249), [GES.py:488](pgmpy/causal_discovery/GES.py#L488), [PC.py:264](pgmpy/causal_discovery/PC.py#L264)). Downstream metrics (`SHD`, `CorrelationScore`) and inference code therefore accept the output without conversion.

`_score_direction` is the only method subclasses are required to implement. Validation, attribute writing and graph construction stay in the base class.

#### Estimator surface

```python
ANM(regressor="gp", score_method="hsic",
    data_splitting=False, random_state=None)
IGCI(score_method="slope", reference_measure="uniform",
     tie_eps=1e-8)
BivariateLiNGAM(score_method="lr", nonlinearity="tanh",
                gaussianity_threshold=0.05)
```

All three populate, in addition to the shared attributes set by the base class:

- ANM: `hsic_forward_`, `hsic_backward_` (scalar p-values).
- IGCI: `forward_C_`, `backward_C_` (scalar directional statistics).
- BivariateLiNGAM: `beta_`, `lr_statistic_`, `gaussianity_pvalue_`.

Intermediate arrays (residuals, kernel matrices) are not stored as fitted attributes, matching the pattern in `PC` / `GES` / `HillClimbSearch` which expose only the graph and a handful of scalars.

##### ANM defaults

- `regressor="gp"`: sklearn `GaussianProcessRegressor` with an RBF plus white-noise kernel. When n > 2000 (configurable via `gp_max_samples`), the backend switches to `Nystroem(n_components=300)` + `Ridge` to keep cost at O(n * m^2) instead of O(n^3). Number of Nystroem components is exposed as `nystroem_components`.
- `score_method="hsic"`: reuses `pgmpy.ci_tests.HSIC` (PR #3254, in review). That test is a `_BaseCITest` constructed with a DataFrame and called with column names ([hsic.py:81-83, 165](pgmpy/ci_tests/hsic.py#L81-L83)), not a numpy-array helper. ANM bridges this by wrapping the input column and the residual vector into a two-column DataFrame and calling `HSIC(data=df).run_test("x", "residual", [])` once per direction. The alternative -- lifting `_center_kernel` and `_hsic_gamma_pvalue` into a numpy-level helper that both `HSIC` and `ANM` call -- is cleaner but touches in-review code, so it will be discussed with mentors in Phase 1 once PR #3254 has landed. `"regression_error"` is the O(n) baseline from Blobaum et al. 2018.
- `data_splitting=False`: maximises power on small samples. Setting it to `True` uses an honest train/test split for the residuals.

##### IGCI defaults

- `score_method="slope"`: finite differences of log|f'(x)| on sorted data. `"entropy"` uses `kl_entropy` with k=5. Mooij et al. (2016) find IGCI performance varies considerably across implementations and reference measures, so both variants are exposed and the user can pick.
- `reference_measure="uniform"` (min-max to [0, 1]) or `"gaussian"` (z-score).
- `tie_eps=1e-8`: tied x-values (relative to range(X)) are collapsed before slope computation; otherwise log(0) blows up. Pairs where both dx and dy fall below tolerance are dropped.

##### Bivariate LiNGAM defaults

- `score_method="lr"`: Hyvarinen-Smith statistic, one OLS, O(n). `"ksg"` uses `_ksg_mi` on residuals; `"hsic"` uses `pgmpy.ci_tests.HSIC` (PR #3254, in review) on residuals via the same DataFrame-wrap pattern as ANM.
- `nonlinearity="tanh"` (default) or `"exp"` (u * exp(-u^2 / 2)). Only used by `"lr"`.
- `gaussianity_threshold=0.05`: Shapiro-Wilk test on the residuals. A `UserWarning` is raised when the residuals look Gaussian, since the method loses identifiability there.

#### Diagnostics

The base class exposes `confidence_` in [0, 1] (`|s_fwd - s_bwd| / (s_fwd + s_bwd)`). Low confidence means the directional scores are nearly tied -- the typical fingerprint of a violated assumption (e.g. a hidden common cause, or a method-data regime mismatch). Callers can threshold it themselves; the proposal does not introduce a separate boolean attribute or warning, since none of the existing causal-discovery estimators in pgmpy do that today.

The Tubingen benchmark separately annotates known-confounded pairs with weight 0; benchmark evaluation excludes those by weight following Mooij et al. 2016. This is a benchmark-level concern, not an estimator-level one.

### Testing Plan

Tests are written alongside the API, before the algorithm code. Each estimator's test file covers four categories. All tests use seeded synthetic data and are deterministic. Target wall time per file is under five seconds.

**Direction recovery on the algorithm's own assumed model.**

- ANM: Y = f(X) + E with f nonlinear (tanh, polynomial); X and E non-Gaussian.
- IGCI: deterministic Y = f(X), X ~ Uniform[0, 1], f monotone nonlinear.
- LiNGAM: Y = 0.7 * X + E with X, E iid Laplace.

For each, at n in {200, 1000} and over 100 seeds, assert recovery rate >= 0.9.

**Symmetry.** Swap column names; assert `predicted_direction_` swaps accordingly and `direction_score_` flips sign. Catches index-ordering bugs.

**Hyperparameter sanity.** Fit twice with different non-default options for each method-specific knob; assert fitted attributes differ. Catches silent no-op parameters.

**Edge cases** (each parameterised):

- constant column -> `ValueError` with a clear message;
- n = 2 -> `ValueError`, not a meaningless direction;
- three or more columns -> base-class `ValueError` (covered in `test_pairwise_base.py`);
- tied x for IGCI -> tie handling collapses without log(0);
- Gaussian residuals for LiNGAM -> `UserWarning` raised, `gaussianity_pvalue_` reflects it;
- NaN / inf -> blocked by `_check_fit_data`; one regression test confirms the error message surfaces;
- `random_state` set -> two fits produce bit-identical attributes (regression check for the GP and Nystroem backends).

**Cross-package consistency.**
One regression test per algorithm using the original paper's reference setup: the simulated example from Hoyer et al. 2009 Section 5.1 for ANM, the AbsBivariate synthetic from Janzing et al. 2012 for IGCI, the bivariate non-Gaussian example from Hyvarinen and Smith 2013 for LiNGAM.

One direction-agreement test against `causal-learn` on three fixed Tubingen pairs (1, 4, 47). Skipped when `causal-learn` is not installed.

**API conformance.**

- `sklearn.utils.estimator_checks.parametrize_with_checks` for each estimator, with the same `expected_failed_checks` dict that `HillClimbSearch` and `GES` already use ([test_HillClimbSearch.py:9-31](pgmpy/tests/test_causal_discovery/test_HillClimbSearch.py#L9-L31)). Two checks (`check_fit_score_takes_y`, `check_n_features_in_after_fitting`) are known to fail for causal-discovery estimators and are skipped via that dict; the new estimators inherit the same exemptions.
- `causal_graph_` is a `pgmpy.base.DAG` with one edge; `adjacency_matrix_` is a 2x2 `pd.DataFrame` with column order matching `feature_names_in_`.
- The `*_pairwise_orient` adapters return `(source, target)` and are usable as a drop-in `orientation_fn` for `ExpertInLoop` after `partial(adapter, data=df)`. One regression test exercises this end-to-end against `ExpertInLoop`.

**Benchmark integration.**
One test marked `@pytest.mark.slow` runs each estimator on the CauseEffectPairs benchmark (100 pairs from 37 source datasets, Mooij et al. 2016). The assertion for ANM is weighted accuracy within ±10% of 63%, which is the confidence interval Mooij et al. themselves report (63 ± 10%, AUC 0.74 ± 0.05). IGCI numbers vary by implementation; the test asserts only that the slope and entropy variants score above 50%. Mooij et al. do not benchmark a bivariate LiNGAM, so the LiNGAM test asserts the recovery rate on the synthetic non-Gaussian setup from Hyvarinen and Smith (2013), not a Tubingen number. Excluded from default CI per existing pgmpy convention.

**Metric tests.**
`test_pairwise_accuracy.py`: small cases against hand-computed values (three-pair toy dataset with known weights), undecided predictions counted as incorrect, and AUC against a known ranking.

### Code Quality

All public methods carry numpy/SciPy-style docstrings and type hints, validated by `blackdoc` and the `ruff D` / `DOC` rule sets per [`pyproject.toml`](pyproject.toml). Every PR passes `pre-commit run --all-files` (black, isort, flake8, ruff, blackdoc) before review. This mirrors the existing pgmpy contribution workflow rather than introducing new lint rules.

### User Journeys

**1. ANM on a single pair.**

```python
from pgmpy.causal_discovery import ANM
from pgmpy.datasets import load_dataset

pair = load_dataset("tubingen/1").data       # columns ["x", "y"]
est = ANM().fit(pair)

est.predicted_direction_           # ("x", "y") if X is the cause
est.direction_score_               # signed; > 0 favours x -> y
est.confidence_                    # in [0, 1]
est.hsic_forward_, est.hsic_backward_   # scalar HSIC p-values per direction
est.causal_graph_                  # pgmpy.base.DAG with edge x -> y
```

Switching the regressor for large pairs:

```python
ANM(regressor="nystroem", nystroem_components=500).fit(big_pair)
```

**2. IGCI, slope vs entropy.**

```python
from pgmpy.causal_discovery import IGCI

pair = load_dataset("tubingen/47").data

slope = IGCI(score_method="slope").fit(pair)
ent   = IGCI(score_method="entropy", kl_neighbors=5).fit(pair)

slope.predicted_direction_, slope.forward_C_
ent.predicted_direction_,   ent.forward_C_   # may disagree on noisy pairs
```

**3. Bivariate LiNGAM with the Gaussianity diagnostic.**

```python
import warnings
from pgmpy.causal_discovery import BivariateLiNGAM, ANM

est = BivariateLiNGAM(score_method="lr").fit(pair)
est.predicted_direction_, est.beta_, est.lr_statistic_

if est.gaussianity_pvalue_ > 0.05:
    warnings.warn("LiNGAM unreliable on this pair; falling back to ANM.")
    est = ANM().fit(pair)
```

**4. Statistical orientation inside `ExpertInLoop`.**

```python
from functools import partial

from pgmpy.causal_discovery import ExpertInLoop
from pgmpy.utils import anm_pairwise_orient

orientation_fn = partial(anm_pairwise_orient, data=data)
dag = ExpertInLoop(orientation_fn=orientation_fn).fit(data)
```

`anm_pairwise_orient(x, y, data)` slices `data[[x, y]]`, fits an `ANM`, and returns `est.predicted_direction_`. `ExpertInLoop` only passes `(u, v)` to its `orientation_fn` ([ExpertInLoop.py:370](pgmpy/causal_discovery/ExpertInLoop.py#L370)), so the data is bound with `functools.partial` ahead of time. This is identical to the existing pattern for `llm_pairwise_orient` ([ExpertInLoop.py:138-143](pgmpy/causal_discovery/ExpertInLoop.py#L138-L143)). Same shape for `igci_pairwise_orient` and `lingam_pairwise_orient`.

**5. Benchmark across the full Tubingen set.**

```python
from pgmpy.causal_discovery import ANM, IGCI, BivariateLiNGAM
from pgmpy.datasets import load_dataset
from pgmpy.metrics import PairwiseAccuracy

metric = PairwiseAccuracy()
for cls in (ANM, IGCI, BivariateLiNGAM):
    truths, preds, scores = [], [], []
    for pair_id in range(1, 109):
        ds = load_dataset(f"tubingen/{pair_id}")
        if ds.data.shape[1] != 2:        # 8 of 108 pairs are multivariate
            continue
        est = cls().fit(ds.data)
        truths.append(ds.ground_truth)
        preds.append(est.causal_graph_)
        scores.append(est.direction_score_)
    result = metric.evaluate_many(truths, preds, scores=scores)
    print(cls.__name__, result.accuracy, result.auc)
```

`evaluate_many` calls the per-pair `metric.evaluate(true_causal_graph, est_causal_graph)` (which inherits the single-DAG contract from `_BaseSupervisedMetric`) under the hood, then aggregates. Tubingen per-pair weights are not currently loaded by `pgmpy.datasets` ([datasets/_base.py:216-231](pgmpy/datasets/_base.py#L216-L231) only loads `pair{N:04}.txt` and `pair{N:04}_graph.txt`), so adding a `pairmeta.txt` parser is a small extra deliverable inside Phase 5; `weights` defaults to uniform until that lands.

### Timeline

Around 350 hours over 12 + 2 weeks. 

- **Weeks 1-2, ~50 h.** `_BasePairwiseDiscovery` and the four private helpers appended to `_base.py`, plus the test plan reviewed with mentors before any algorithm PR.
- **Weeks 3-5, ~100 h.** ANM with GP and Nystroem backends, full test suite, mid-term Tubingen benchmark (ANM only). *Mid-term checkpoint.*
- **Weeks 6-7, ~50 h.** IGCI with slope and entropy scoring, tie-handling tests.
- **Weeks 8-10, ~100 h.** Bivariate LiNGAM with LR, KSG, and HSIC scoring, plus the Gaussianity diagnostic.
- **Week 11, ~25 h.** Three `*_pairwise_orient` adapters re-exported from `pgmpy/utils/__init__.py`, an `ExpertInLoop` integration test using `functools.partial`, the `pairmeta.txt` parser for Tubingen per-pair weights in `pgmpy/datasets/_base.py`, and the tutorial notebook.
- **Weeks 12-14, ~25 h.** `PairwiseAccuracy` (including its `evaluate_many` aggregator) polished against the now-loaded weights; exploratory work on PC / FCI undirected-edge orientation using the new estimators. The latter likely carries into post-GSoC.
