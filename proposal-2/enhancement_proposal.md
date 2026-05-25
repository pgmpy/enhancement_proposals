## Pairwise Causal Discovery for pgmpy

Contributor: [@hanara2112](https://github.com/hanara2112) (Aryaman Bahl, IIIT Hyderabad)
Mentors: ankurankan et al.

### Introduction

pgmpy currently provides constraint-based (PC) and score-based (HillClimbSearch, GES) algorithms for multivariate causal discovery. In the bivariate setting causal direction is not identifiable from observational data alone, since $X \rightarrow Y$ and $Y \rightarrow X$ produce the same observed correlation. Recovering direction requires additional assumptions about the data-generating process.

This proposal adds four statistical pairwise methods (**ANM**, **IGCI**, **PNL**, **bQCD**) and refactors the existing `llm_pairwise_orient` helper into an estimator (**LLMPairwise**) that follows the same contract. `ExpertInLoop` is extended to accept any of these via a single `pairwise_estimator=` keyword. The bivariate linear non-Gaussian case is already covered by the in-flight LiNGAM PR, so it is not duplicated here.

### Proposed Solution

`ANM`, `IGCI`, `PNL`, `bQCD`, and `LLMPairwise` each subclass `_BaseCausalDiscovery` directly. No intermediate pairwise base class is introduced (see *Alternatives*).

After `fit`, each estimator populates the standard `causal_graph_` and `adjacency_matrix_` attributes used by pgmpy's existing structure learners; the predicted direction is read off `causal_graph_`, just like with any other fitted DAG. A method-specific `direction_score_: float` is also exposed, equal to $S(Y \rightarrow X) - S(X \rightarrow Y)$ where $S$ is the score the method minimizes. Positive means the forward direction wins. No universal `confidence_` attribute is introduced, since the score scales are not comparable across methods.

The two-column input check lives inside each `_fit`, since the shared `_check_fit_data` is also used by multivariate algorithms and cannot enforce a bivariate shape. The four statistical estimators mark categorical input as unsupported via `__sklearn_tags__`. `LLMPairwise` reads variable names rather than values, so it does not. All five accept the standard `show_progress` and `seed` keywords used elsewhere in pgmpy.

The five estimators:

- **ANM** (Hoyer et al., 2009): assumes $Y = f(X) + E$ with $E \perp X$. Fits a nonparametric regression in each direction and picks the direction whose residuals look more independent of the input. Accepts any continuous CI test (HSIC by default; KCI, GCM, Pearsonr also work).
- **IGCI** (Janzing et al., 2012): assumes $Y$ is a near-deterministic function of $X$ and that $P(X)$ and $f$ are chosen independently. Picks the direction with the smaller slope-based or entropy-based statistic. $O(n \log n)$, no kernels or bandwidths.
- **PNL** (Zhang & Hyvärinen, 2009): the post-nonlinear model $Y = f_2(f_1(X) + E)$ with $E \perp X$. Generalizes ANM and so covers regimes ANM does not, such as sensor saturation or monotone post-processing. Fits the two nonlinearities and an unmixing step in each direction, then uses the same residual-independence check as ANM.
- **bQCD** (Tagasovska et al., 2020): bivariate quantile causal discovery. Scores each direction by the complexity of the conditional quantile function and picks the simpler one. Copula-based with no kernel bandwidths, so there are no hyperparameters to tune.
- **LLMPairwise** (refactor of `llm_pairwise_orient`): ignores the numeric data and asks an LLM which direction is more likely given the variable names and optional descriptions. Useful when the data are too noisy or scarce for a statistical method but the names carry domain meaning.

Because all five share the standard estimator contract, `ExpertInLoop` accepts any of them via `pairwise_estimator=`, replacing the current `orientation_fn` + `functools.partial` approach.

CI tests follow pgmpy's existing dual-form convention (`ANM(ci_test="hsic")` or `ANM(ci_test=HSIC(...))`); regressors are passed as sklearn instances directly. This keeps the common case ergonomic and the configurable case standard.

**Reused pgmpy infrastructure:**

- `_BaseCausalDiscovery` and `_check_fit_data` for the standard `fit` contract and input validation
- `pgmpy.base.DAG` for fitted causal graphs (no custom graph abstraction)
- `pgmpy.ci_tests` for residual-independence testing in ANM and PNL; HSIC (currently in flight as a separate PR) is the default
- `ExpertInLoop` as the integration point for combining a pairwise estimator with the multivariate skeleton

**Evaluation.** Benchmark evaluation on the Tübingen Cause-Effect Pairs is done in `examples/Pairwise_Causal_Discovery.ipynb`, not in core library code. Accuracy is computed in the notebook by comparing the single edge in `causal_graph_` against the ground-truth edge; AdjacencyConfusionMatrix is reused where it applies. No new metric class is introduced.

### Alternative Solutions

*Function-based orientation wrappers.* Helpers like `anm_pairwise_orient(x, y, data)` plugged into `ExpertInLoop` similarly to today's `llm_pairwise_orient` were considered. Rejected because adding one wrapper per method duplicates integration logic; estimators following a single contract are easier to compose and test.

*Shared `_BasePairwiseDiscovery` base class.* An earlier draft included one, with helper methods for direction scoring shared across the statistical methods. Dropped during review on two grounds: the algorithms differ enough internally that shared scaffolding would be thin, and such an abstraction is easier to extract from working code than to design upfront. Each estimator therefore subclasses `_BaseCausalDiscovery` directly.

*Implement only ANM.* ANM-pHSIC is one of the stronger individual methods on Tübingen ($63 \pm 10\%$ weighted accuracy in Mooij et al., 2016), but its assumptions fail in exactly the regimes where IGCI (low-noise, near-deterministic), PNL (post-nonlinear distortion), and bQCD (no functional model at all) perform well. Including all four gives complementary coverage at a modest implementation cost.

#### File layout

**New files**

- `pgmpy/causal_discovery/ANM.py`
- `pgmpy/causal_discovery/IGCI.py`
- `pgmpy/causal_discovery/PNL.py`
- `pgmpy/causal_discovery/bQCD.py`
- `pgmpy/causal_discovery/LLMPairwise.py`
- `pgmpy/tests/test_causal_discovery/test_{ANM,IGCI,PNL,bQCD,LLMPairwise,ExpertInLoop_pairwise}.py`
- `examples/Pairwise_Causal_Discovery.ipynb`

**Modified files**

- `pgmpy/causal_discovery/__init__.py` — re-export the five new estimators
- `pgmpy/causal_discovery/ExpertInLoop.py` — add `pairwise_estimator=` keyword

### Algorithm details

**Defaults**

- **ANM**: `GaussianProcessRegressor` for nonlinear regression by default; `Nystroem + Ridge` available as a faster alternative for larger samples. HSIC as the default residual-independence test, with any pgmpy CI test accepted as an override.
- **IGCI**: slope-based scoring by default; entropy-based scoring available. Configurable reference measure (`"uniform"` or `"gaussian"`). Tied observations are handled defensively to avoid finite-difference instability.
- **PNL**: default fitting follows Zhang & Hyvärinen (2009); concrete regressor choices for $f_1$ and the inverse of $f_2$ will be finalized against the reference implementation in Phase 1. Same CI-test override path as ANM.
- **bQCD**: default scoring follows the reference implementation in Tagasovska et al. (2020). No hyperparameters on the default path.
- **LLMPairwise**: same prompt structure as `llm_pairwise_orient`, with `prompt_` and `response_` retained for auditability.

### Testing Plan

**Direction recovery** on synthetic data, the natural setting for each method's assumptions:

- ANM: nonlinear additive-noise data.
- IGCI: near-deterministic low-noise data.
- PNL: post-nonlinear distorted additive-noise data.
- bQCD: data with mismatched conditional-quantile complexity in each direction.

Reported across at least 5 seeds and at two sample sizes ($n = 200$ and $n = 2000$).

**Edge cases:** constant inputs, NaN/inf, insufficient samples, tied observations (IGCI), reproducibility under fixed `seed`.

**Integration:** `ExpertInLoop` driven with each of the five estimators as `pairwise_estimator=`, including the `orientation_cache_` reuse path so `LLMPairwise` is not re-queried for cached edges.

**Tübingen benchmark** lives in `examples/Pairwise_Causal_Discovery.ipynb` rather than in the test suite, since it is too slow and too noisy for CI. Results are compared informally against the numbers in Mooij et al. (2016).

### User Journeys

**1. Pairwise causal discovery on a single variable pair**

```python
from pgmpy.causal_discovery import ANM
from pgmpy.datasets import load_dataset

pair = load_dataset("tubingen/1").data

est = ANM().fit(pair)

est.causal_graph_          # pgmpy DAG with the single oriented edge
est.direction_score_
```

The same interface applies to `IGCI`, `PNL`, `bQCD`, and `LLMPairwise`.

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
import pandas as pd
from pgmpy.causal_discovery import ExpertInLoop, ANM

data = pd.read_csv("my_dataset.csv")  # any multivariate DataFrame

dag = ExpertInLoop(
    pairwise_estimator=ANM()
).fit(data)
```

Since all five pairwise estimators follow the same contract, statistical and LLM-based orientation can be used interchangeably.

### References

- Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *Advances in Neural Information Processing Systems*, 21. [paper](https://proceedings.neurips.cc/paper/2008/hash/f7664060cc52bc6f3d620bcedc94a4b6-Abstract.html)
- Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J., Daniušis, P., Steudel, B., & Schölkopf, B. (2012). Information-geometric approach to inferring causal directions. *Artificial Intelligence*, 182–183, 1–31. [doi:10.1016/j.artint.2012.01.002](https://doi.org/10.1016/j.artint.2012.01.002)
- Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., & Schölkopf, B. (2016). Distinguishing cause from effect using observational data: methods and benchmarks. *Journal of Machine Learning Research*, 17(32), 1–102. [arXiv:1412.3773](https://arxiv.org/abs/1412.3773)
- Zhang, K., & Hyvärinen, A. (2009). On the identifiability of the post-nonlinear causal model. *Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI)*. [paper](https://arxiv.org/abs/1205.2599)
- Tagasovska, N., Chavez-Demoulin, V., & Vatter, T. (2020). Distinguishing cause from effect using quantiles: bivariate quantile causal discovery. *International Conference on Machine Learning (ICML)*. [arXiv:1801.10579](https://arxiv.org/abs/1801.10579)
