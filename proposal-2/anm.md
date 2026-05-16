## ANM: Implementation Details

Supplement to the main enhancement proposal. Goes deeper than the proposal on identifiability, hyperparameter justification, regression-backend trade-offs, and the test plan. Read alongside `enhancement_proposal.md`; this file does not restate what is already there.

### Identifiability (why this works at all)

The bivariate Additive Noise Model is:

```
Y = f(X) + E,     E independent of X.
```

The non-trivial fact is that under mild conditions, *no* backward model `X = g(Y) + E'` with `E' independent of Y` reproduces the same joint `P(X, Y)`. That is what makes ANM identifiable.

The most useful statement is **Hoyer et al. (2009), Theorem 1**: if `f` is non-linear and `X` and `E` admit smooth strictly positive densities, then the backward ANM holds for at most a measure-zero set of distributions of `(X, E)`. Peters et al. (2014, *JMLR*) extended this from "for almost all noise distributions" to a much stronger result via the differential-equation characterisation: the joint admits an ANM in *both* directions only if a specific third-order ODE on `log p(X)` and `log p(E)` is satisfied -- which is a fragile condition that generically fails.

What this means in practice:

| Setting | Identifiable? |
|---|---|
| Non-linear `f`, non-Gaussian noise | Yes (Hoyer 2009) |
| Non-linear `f`, Gaussian noise | Yes generically (Peters 2014) |
| Linear `f`, non-Gaussian noise | Yes -- but this is LiNGAM territory |
| Linear `f`, Gaussian noise | **No** -- the ODE condition is trivially satisfied |
| `f` flat over part of the domain | Degenerate; identifiability degrades |

The estimator does not gate on these conditions at runtime. Instead it exposes `confidence_` (the score asymmetry), which collapses toward zero exactly when identifiability is weak. The prototype demonstrates this: on a linear-Gaussian pair, the prototype's `confidence_` drops to roughly half of what it shows on a true non-linear ANM pair (see `api_walkthrough.ipynb`, cell 3).

### Algorithm (finer-grain pseudocode than the main proposal)

```
input  : DataFrame with two columns [a, b]
output : (predicted_direction, scores)

x <- X[a].to_numpy()
y <- X[b].to_numpy()

# Direction a -> b
gp_fwd <- GaussianProcessRegressor(kernel=RBF + WhiteKernel, normalize_y=True)
gp_fwd.fit(x.reshape(-1, 1), y)
r_fwd <- y - gp_fwd.predict(x.reshape(-1, 1))
s_fwd <- HSIC(x, r_fwd)     # via DataFrame-wrap into pgmpy.ci_tests.HSIC

# Direction b -> a
gp_bwd <- GaussianProcessRegressor(...)
gp_bwd.fit(y.reshape(-1, 1), x)
r_bwd <- x - gp_bwd.predict(y.reshape(-1, 1))
s_bwd <- HSIC(y, r_bwd)

if s_fwd < s_bwd:
    return (a -> b), (s_fwd, s_bwd)
else:
    return (b -> a), (s_fwd, s_bwd)
```

The two GP fits are independent and can be run in parallel; pgmpy's existing `joblib.Parallel` infrastructure (used by `_ConstraintMixin._build_skeleton`) is available, but for n_jobs=2 it is rarely worth the overhead. Left out of v1 and noted as a follow-up.

### Regression backend

Default is sklearn's `GaussianProcessRegressor` with `RBF + WhiteKernel`. Three reasons:

1. **Identifiability respects the regression class.** Peters et al. (2014, Theorem 28) shows ANM identifiability is preserved as long as the regressor's hypothesis class is rich enough to fit the true `f`. GP regression with an RBF kernel is universal in this sense; OLS is not (it would force ANM to behave like linear LiNGAM, which is a different identifiability regime).
2. **HSIC pairs naturally with kernel regression.** Both share the median-bandwidth heuristic and the same kernel philosophy, so the residual-vs-input independence test does not see kernel mismatch artefacts.
3. **sklearn-compatible.** The proposal claims "any sklearn-compatible regressor" is accepted via the `regressor=` constructor argument. GP is the default, but users can pass anything implementing `fit` / `predict`.

#### Nyström switch for large n

Exact GP regression is O(n^3). On Tubingen pairs (n typically 500-5000) this is fine. On larger datasets it is not. The proposal switches to `sklearn.kernel_approximation.Nystroem(n_components=300) + Ridge` when `n > gp_max_samples` (default 2000), giving O(n * m^2) cost.

The Nyström approximation preserves the **kernel-richness** of GP regression -- which is what identifiability needs -- while losing the **predictive-variance** structure -- which ANM does not consume. So the trade-off is genuinely free for direction recovery; the user only loses the ability to read calibrated uncertainty out of the regression fit, which the estimator does not expose anyway.

#### `data_splitting` flag

Off by default. With `data_splitting=True`, the data are partitioned 50/50: one half fits the GP, the other half computes residuals. This avoids the standard regression-bias inflation (the GP fits the residual structure as well as the function, which biases HSIC slightly toward "more independent than reality"). The cost is roughly halved effective sample size, which on small pairs is the dominant concern. Mooij et al. (2016) report results both with and without splitting; their preferred protocol uses splitting only for n >= 500.

### Score functions

#### `score_method="hsic"` (default)

HSIC V-statistic with RBF kernels, median bandwidth, gamma null calibration. Implemented by the in-review PR #3254. Lower means residuals are more independent of the input. The integration uses the DataFrame-wrap pattern:

```python
df = pd.DataFrame({"input": cause, "residual": residuals})
test = HSIC(data=df).run_test("input", "residual", [])
# In production: use test.statistic_ for ranking, test.p_value_ for thresholding.
```

#### `score_method="regression_error"` (baseline)

The Blöbaum et al. (2018, *AISTATS*) alternative. Compares mean squared regression errors in both directions instead of HSIC. O(n) instead of O(n^2) for HSIC. Theoretically less defensible (the asymmetry is empirically robust but does not have ANM's identifiability backing), but ~50x faster on large pairs. Useful as a sanity-check pass before running HSIC.

The proposal exposes this as a fall-back option, not the default.

### Known failure modes

| Mode | What happens | Mitigation |
|---|---|---|
| Linear Gaussian | Both directions look equally good; `confidence_` collapses | User reads `confidence_`; no automatic action |
| Strong noise (low SNR) | GP overfits the noise; both directions look independent | `data_splitting=True` helps; large n helps more |
| Small n (< 100) | HSIC null distribution is unreliable; estimator may flip directions across seeds | Test plan asserts recovery rate only at n >= 200 |
| Hidden common cause | Both directions show residual dependence; HSIC gets high in both | Low `confidence_`; user-visible. **Not** detectable by the method itself in general (Mooij et al. 2016, Sec. 5 -- the bivariate confounded case is non-identifiable without further restrictions) |
| Discrete or heavily-tied data | RBF bandwidth collapses; HSIC unstable | Out of scope; the estimator is documented as continuous-only |
| `f` non-monotone with severe extrapolation | GP extrapolates linearly outside the training support; residuals get artefacts | This is a Gaussian-process limitation, not an ANM limitation; users can swap to `regressor="nystroem"` which handles this slightly differently |

### Test cases tied to source-paper figures

Beyond the unit tests in the proposal's "Testing Plan", three regression tests reproduce specific results from the source papers:

1. **Hoyer et al. (2009), Figure 2(a)** -- the simulated example with `f(x) = b * exp(-x^2 / s) + x` and additive uniform noise. Assert recovery on n = 200 over 20 seeds.
2. **Peters et al. (2014), Section 5.1** -- `Y = X^3 + E` with `E ~ Uniform[-1, 1]`. Assert recovery at n = 100, and assert non-recovery (or low confidence) on the reverse where `E` is set to Gaussian (Theorem 1's exclusion case).
3. **Mooij et al. (2016), Table 1, "AN-pHSIC" row** -- the slow-benchmark test runs the full Tubingen set and asserts weighted accuracy within ten percentage points of the published number (~63%).

The first two are deterministic with seeded RNG and run in CI. The third is gated by `@pytest.mark.slow`.

### Open questions to settle with mentors in Phase 1

1. **Lifting `_center_kernel` / `_hsic_gamma_pvalue` out of the HSIC test into a shared helper.** The DataFrame-wrap pattern works and is the proposed v1 path, but it costs one DataFrame allocation per `_score_direction` call. If mentors are happy to touch PR #3254 after merge, a numpy-level helper is cleaner. I would like a decision on this in week 1.
2. **`data_splitting` default.** Mooij et al. set it on for n >= 500. The proposal defaults it off (maximises power on small samples). A configurable threshold (`split_threshold=500`) is the natural compromise; worth confirming.
3. **Should `ANM` log a warning when n < some threshold where the gamma p-value is unreliable?** The HSIC test itself handles this internally (returns p=1.0 for n < 6), but ANM users may want to see the warning surfaced. I lean toward "no, the user reads `confidence_`", but happy to be overruled.

### References

- Hoyer, P. O., Janzing, D., Mooij, J. M., Peters, J., & Schölkopf, B. (2009). Nonlinear causal discovery with additive noise models. *NeurIPS*.
- Peters, J., Mooij, J. M., Janzing, D., & Schölkopf, B. (2014). Causal discovery with continuous additive noise models. *JMLR*, 15.
- Mooij, J. M., Peters, J., Janzing, D., Zscheischler, J., & Schölkopf, B. (2016). Distinguishing cause from effect using observational data. *JMLR*, 17(32).
- Blöbaum, P., Janzing, D., Washio, T., Shimizu, S., & Schölkopf, B. (2018). Analysis of cause-effect inference by comparing regression errors. *AISTATS*.
- Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Schölkopf, B., & Smola, A. J. (2008). A kernel statistical test of independence. *NeurIPS*.
