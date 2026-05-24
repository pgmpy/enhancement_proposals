## Pairwise Causal Discovery for pgmpy

Contributor: [@hanara2112](https://github.com/hanara2112) (Aryaman Bahl, IIIT Hyderabad)
Mentors: ankurankan et al.

### Introduction

pgmpy currently provides constraint-based (PC) and score-based (HillClimbSearch, GES) algorithms for multivariate causal discovery. In the bivariate setting causal direction is not identifiable from observational data alone, since $X \rightarrow Y$ and $Y \rightarrow X$ produce the same observed correlation. Recovering direction requires additional assumptions about the data-generating process.

This proposal adds two statistical pairwise methods — **ANM** and **IGCI** — and refactors the existing `llm_pairwise_orient` helper into an estimator (**LLMPairwise**) that follows the same contract. `ExpertInLoop` is extended to accept any of these via a single `pairwise_estimator=` keyword. The bivariate linear non-Gaussian case is already covered by the in-flight LiNGAM PR and is therefore out of scope here. PNL and bQCD are listed as future work rather than primary goals.

### Proposed Solution

`ANM`, `IGCI`, and `LLMPairwise` each subclass `_BaseCausalDiscovery` directly. There is no intermediate pairwise base class — the three algorithms share little internally, and an extra layer can be extracted later from working code if it ever becomes useful.

After `fit`, each estimator populates the standard `causal_graph_` and `adjacency_matrix_` attributes that pgmpy's existing structure learners already use, plus one bivariate-specific convenience attribute:

- `predicted_direction_`: a tuple `("X", "Y")` giving the predicted (cause, effect). This is a small property over `causal_graph_` that lets users avoid `next(iter(estimator.causal_graph_.edges()))` for the single-edge case.

A method-specific `direction_score_: float` is also exposed, equal to $S(Y \rightarrow X) - S(X \rightarrow Y)$ where $S$ is the score the method minimizes. Positive means the forward direction wins. No universal `confidence_` attribute is introduced, since the score scales are not comparable across methods.

The two-column input check lives inside each `_fit`, since the shared `_check_fit_data` is reused with multivariate algorithms and cannot enforce a bivariate shape itself. `ANM` and `IGCI` mark categorical input as unsupported via `__sklearn_tags__`; `LLMPairwise` reads variable names and therefore does not. All three accept the standard `show_progress` and `seed` keywords used elsewhere in pgmpy.

The three estimators:

- **ANM** (Hoyer et al., 2009): assumes $Y = f(X) + E$ with $E \perp X$. Fits a nonparametric regression in each direction and picks the direction whose residuals look more independent of the input. Accepts any continuous CI test (HSIC by default; KCI, GCM, Pearsonr also work).
- **IGCI** (Janzing et al., 2012): assumes $Y$ is a near-deterministic function of $X$ and that $P(X)$ and $f$ are chosen independently. Picks the direction with the smaller slope-based or entropy-based statistic. $O(n \log n)$, no kernels or bandwidths.
- **LLMPairwise** (refactor of `llm_pairwise_orient`): ignores the numeric data and asks an LLM which direction is more likely given the variable names and optional descriptions. Useful when the data are too noisy or scarce for a statistical method but the names carry domain meaning.

Behind the same `_BaseCausalDiscovery` contract, `ExpertInLoop` accepts any of them via `pairwise_estimator=`, replacing the current `orientation_fn` + `functools.partial` approach.

CI tests follow pgmpy's existing dual-form convention (`ANM(ci_test="hsic")` or `ANM(ci_test=HSIC(...))`); regressors are passed as sklearn instances directly. This keeps the common case ergonomic and the configurable case standard.

**Reused pgmpy infrastructure:**

- `_BaseCausalDiscovery` and `_check_fit_data` for the standard `fit` contract and input validation
- `pgmpy.base.DAG` for fitted causal graphs (no custom graph abstraction)
- `pgmpy.ci_tests` for residual-independence testing in ANM; HSIC (PR #3254) is the default
- `ExpertInLoop` as the integration point for combining a pairwise estimator with the multivariate skeleton

**Evaluation.** Benchmark evaluation on the Tübingen Cause-Effect Pairs is done in `examples/Pairwise_Causal_Discovery.ipynb`, not in core library code. Accuracy is a direct comparison of `predicted_direction_` against the ground-truth edge; AdjacencyConfusionMatrix is reused where it applies. No new metric class is introduced.

### Future work

The following are deliberately out of scope but are natural follow-ups once the core lands:

- **PNL** (Zhang & Hyvärinen, 2009): post-nonlinear model $Y = f_2(f_1(X) + E)$. Extends ANM's assumptions and would share most of ANM's regression and CI-test plumbing.
- **bQCD** (Tagasovska et al., 2020): quantile-based, hyperparameter-free, complementary to functional-model methods.

### Alternative Solutions

*Function-based orientation wrappers.* Helpers like `anm_pairwise_orient(x, y, data)` plugged into `ExpertInLoop` similarly to today's `llm_pairwise_orient` were considered. Rejected because adding one wrapper per method duplicates integration logic; estimators following a single contract are easier to compose and test.

*Shared `_BasePairwiseDiscovery` base class.* An earlier draft included one. Dropped because the three algorithms differ substantially internally, so the shared scaffolding would be thin. An abstraction is easier to extract later than to design upfront.

*Implement only ANM.* ANM-pHSIC is one of the stronger individual methods on Tübingen ($63 \pm 10\%$ weighted accuracy in Mooij et al., 2016), but its assumptions fail in regimes where IGCI does well (low-noise, near-deterministic). Including both gives complementary coverage at a modest implementation cost.

#### File layout

**New files**

- `pgmpy/causal_discovery/ANM.py`
- `pgmpy/causal_discovery/IGCI.py`
- `pgmpy/causal_discovery/LLMPairwise.py`
- `pgmpy/tests/test_causal_discovery/test_{ANM,IGCI,LLMPairwise,ExpertInLoop_pairwise}.py`
- `examples/Pairwise_Causal_Discovery.ipynb`

**Modified files**

- `pgmpy/causal_discovery/__init__.py` — re-export the three new estimators
- `pgmpy/causal_discovery/ExpertInLoop.py` — add `pairwise_estimator=` keyword

### Algorithm details

**Defaults**

- **ANM**: `GaussianProcessRegressor` for nonlinear regression by default; `Nystroem + Ridge` available as a faster alternative for larger samples. HSIC as the default residual-independence test, with any pgmpy CI test accepted as an override.
- **IGCI**: slope-based scoring by default; entropy-based scoring available. Configurable reference measure (`"uniform"` or `"gaussian"`). Tied observations are handled defensively to avoid finite-difference instability.
- **LLMPairwise**: same prompt structure as `llm_pairwise_orient`, with `prompt_` and `response_` retained for auditability.

### Testing Plan

**Direction recovery** on synthetic data, the natural setting for each method's assumptions:

- ANM: nonlinear additive-noise data.
- IGCI: near-deterministic low-noise data.

Reported across at least 5 seeds and at two sample sizes ($n = 200$ and $n = 2000$).

**Edge cases:** constant inputs, NaN/inf, insufficient samples, tied observations (IGCI), reproducibility under fixed `seed`.

**Integration:** `ExpertInLoop` driven with each of the three estimators as `pairwise_estimator=`, including the `orientation_cache_` reuse path so `LLMPairwise` is not re-queried for cached edges.

**Tübingen benchmark** lives in `examples/Pairwise_Causal_Discovery.ipynb`, not in the test suite — it's too slow and noisy for CI. Results are compared informally against the numbers in Mooij et al. (2016).

### User Journeys

**1. Pairwise causal discovery on a single variable pair**

```python
from pgmpy.causal_discovery import ANM
from pgmpy.datasets import load_dataset

pair = load_dataset("tubingen/1").data

est = ANM().fit(pair)

est.predicted_direction_   # e.g. ("X", "Y")
est.direction_score_
est.causal_graph_
```

The same interface applies to `IGCI` and `LLMPairwise`.

---

**2. Swapping the CI test or regressor**

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

CI tests and regressors compose via standard sklearn idioms, so swapping them does not require any pgmpy-internal changes.

---

**3. Using a pairwise estimator inside `ExpertInLoop`**

```python
from pgmpy.causal_discovery import ExpertInLoop, ANM

dag = ExpertInLoop(
    pairwise_estimator=ANM()
).fit(data)
```

Since all three pairwise estimators follow the same contract, statistical and LLM-based orientation can be used interchangeably.

### References

- Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *Advances in Neural Information Processing Systems*, 21. [paper](https://proceedings.neurips.cc/paper/2008/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html)
- Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J., Daniušis, P., Steudel, B., & Schölkopf, B. (2012). Information-geometric approach to inferring causal directions. *Artificial Intelligence*, 182–183, 1–31. [doi:10.1016/j.artint.2012.01.002](https://doi.org/10.1016/j.artint.2012.01.002)
- Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., & Schölkopf, B. (2016). Distinguishing cause from effect using observational data: methods and benchmarks. *Journal of Machine Learning Research*, 17(32), 1–102. [arXiv:1412.3773](https://arxiv.org/abs/1412.3773)
- Zhang, K., & Hyvärinen, A. (2009). On the identifiability of the post-nonlinear causal model. *Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI)*. [paper](https://arxiv.org/abs/1205.2599)
- Tagasovska, N., Chavez-Demoulin, V., & Vatter, T. (2020). Distinguishing cause from effect using quantiles: bivariate quantile causal discovery. *International Conference on Machine Learning (ICML)*. [arXiv:1801.10579](https://arxiv.org/abs/1801.10579)
