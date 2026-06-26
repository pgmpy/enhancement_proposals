# Backend-Agnostic Shared Layers for Causal Discovery (Scoring & CI Tests)

## Contributors

- @ankurankan

## Introduction

pgmpy is growing from classic causal-discovery methods (`PC`, `GES`, `HillClimbSearch`) to also include
continuous-optimization / deep-learning methods (`NOTEARS`, `DAGMA`, `CASTLE`). This raises two requirements the current
code cannot meet cleanly:

1. **Uniform data input** — accept pandas, Polars, cuDF, PyArrow, and raw numpy/torch without per-method special casing.
2. **Shared functionality on CPU *and* GPU** — structure scoring and CI tests must run on numpy, torch, and cupy from a
   single implementation, including keeping cuDF data on the GPU end to end.

Today every score and CI test accepts a **dataframe + variable names** and is pandas/numpy-bound, mixing two unrelated
concerns — *dataframe ingestion* and *numeric computation* — in each estimator. There are really two orthogonal
compatibility axes:

- **Dataframe axis** (pandas/Polars/cuDF/…), solved by **narwhals**.
- **Array axis** (numpy/torch/cupy), solved by the **Python Array API** via `array-api-compat`.

Tangling them inside every estimator multiplies the test surface (every score × every dataframe × every array backend)
and blocks the new optimization methods — which already hold a tensor and no dataframe — from reusing the shared
computations. This proposal funnels all input through one converted-once representation (`ArrayData`), so the backend
follows the input data and the heavy computations are written once — the array-core + thin-conversion-at-`fit` pattern
used by scikit-learn (`validate_data`), XGBoost (`DMatrix`), and statsmodels (array core + formula layer).

## Proposed Solution

One representation, converted once at the `fit` boundary, consumed everywhere.

**`ArrayData` + `ArrayData.from_data` — the single ingestion point.** `ArrayData` is a small immutable bundle: one
on-device matrix (integer codes for discrete data, floats for continuous), per-column cardinalities, and a name↔index
map. `ArrayData.from_data(data)` is an **idempotent factory** — a dataframe (pandas/Polars/cuDF/PyArrow) is ingested via
narwhals, encoded natively (`replace_strict`, on-device for cuDF) and materialized on the resolved backend; a raw
array/tensor is wrapped with default names; an existing `ArrayData` passes through untouched. This is the *only* place
narwhals lives and the only place the cuDF→cupy / DLPack device handoff happens. (It replaces what an earlier draft
called a separate `DataAdapter` class.)

**Convert once at `fit`; everything downstream takes `ArrayData`.** A causal-discovery method's `fit` calls
`ArrayData.from_data(data)` once and threads the result down. Scores and CI tests are **metric classes that operate on
an `ArrayData` directly** (`xp` ops on `ad.matrix[:, idx]`) — there is no separate "kernel" layer beneath them, because
`ad.matrix` is already an Array-API array. Since `from_data` is idempotent, the same classes still work standalone on a
raw frame (`BIC(frame)` converts internally), and multi-phase methods (MMHC) share one `ArrayData` across their CI-test
and scoring phases.

**Uniform for the optimization methods too.** `NOTEARS`/`DAGMA` `fit` the same way — `ArrayData.from_data(data)` — then
optimize on `ad.matrix` (a constant on-device tensor). Their loss reuses a small set of bare-array **differentiable
leaves** (`residuals → log-likelihood`) shared with the Gaussian score. These leaves are the *only* bare-array helpers;
they take raw residuals (from the free weight matrix), not `ArrayData`, because the optimizer has no dataframe and no OLS
fit.

**Backend resolved inside `from_data`** by a single precedence chain (no per-algorithm hard coding):
`explicit arg > algorithm requirement (NOTEARS requires torch) > algorithm preference (HillClimb prefers numpy) > input
data's native device > global default`. Algorithms declare `requires`/`prefers` via skbase tags; `config.set_backend` is
demoted to the lowest-precedence default.

## Alternative Solutions

| Option | Why not |
|---|---|
| Keep dataframes in every estimator (status quo) | Tangles both axes; test surface = scores × frames × backends; optimization methods cannot reuse computations. |
| Convert all input to pandas at the door | Loses GPU/native; measured **4–8× slower** ingestion than native Polars/cuDF on ≥100k rows. |
| Global backend switch only (current `config`, Keras-style) | The Array-API consortium lists a global backend switch as a non-goal for *consuming* libraries; ignores the input's device. |
| Fixed backend per algorithm, no shared layer (gCastle-style) | Each method handles I/O its own way → fragmented API, the opposite of the goal. |
| A separate array-kernel layer beneath the metric classes | Redundant once `ArrayData` exists — its `matrix` is already an Array-API array, so each metric class operates on it directly. Only the autodiff *leaves* stay bare-array. |

Where the leaders landed: scikit-learn / SciPy follow **array-in = array-out** (backend follows the input) with a thin
conversion at `fit` — exactly what `ArrayData.from_data` is.

## Details of proposed solution

```python
@dataclass(frozen=True)
class ArrayData:               # bespoke lightweight container; no cross-package standard exists
    matrix: Array              # (n, d) Array-API array; int codes (discrete) or float (continuous)
    cardinalities: Array       # (d,) ints; unused for continuous
    columns: list[str]         # name -> index via columns.index(name)
    dtypes: dict[str, str]     # "N"/"C"/"O" per column

    @property
    def namespace(self):       # backend + device/dtype derived from the array, not stored
        return array_api_compat.array_namespace(self.matrix)

    @classmethod
    def from_data(cls, data, *, state_names=None, backend=None) -> "ArrayData":
        """Idempotent factory: an ArrayData passes through; a dataframe is narwhals-ingested,
        encoded natively (replace_strict, on-device for cuDF) and materialized on the resolved
        backend; a raw array/tensor is wrapped with default column names."""
```

`ArrayData` is intentionally a **bespoke, lightweight container** — there is no cross-package standard for an
ingested-data bundle (scikit-learn uses *none*, just an array + `feature_names_in_`; XGBoost `DMatrix` / statsmodels
`ModelData` are library-specific; xarray `DataArray` is the heavyweight labeled-array option). A frozen dataclass is the
pragmatic middle. The *contents* are standardized: `matrix` is an Array-API array carrying its own `.device`/`.dtype`,
so the backend is **derived** (`array_namespace(matrix)`) rather than stored.

Metric classes operate on an `ArrayData` directly — `from_data` makes the input uniform, and there is no separate kernel:

```python
class BIC:                                            # score
    def __init__(self, data, **kw):
        self._ad = ArrayData.from_data(data)          # passthrough if already an ArrayData
    def local_score(self, variable, parents):
        ad = self._ad
        i = ad.columns.index(variable)
        js = [ad.columns.index(p) for p in parents]
        ...                                           # xp ops on ad.matrix[:, [i, *js]] -> score

class ChiSquare:                                      # CI test — identical shape
    def __init__(self, data, **kw):
        self._ad = ArrayData.from_data(data)
    def __call__(self, X, Y, Z=(), significance_level=0.05):
        ad = self._ad
        x, y = ad.columns.index(X), ad.columns.index(Y)
        z = [ad.columns.index(c) for c in Z]
        ...                                           # xp contingency on ad.matrix -> p_value
        return p_value >= significance_level
```

The only bare-array helpers are the **differentiable leaves** the optimization methods share with the Gaussian score:

```python
# residuals -> Gaussian log-likelihood; tensor in/out, no reduction or caching.
# NOTEARS/DAGMA feed residuals from the free weight matrix; the score feeds OLS residuals.
def gaussian_loglik(xp, resid, n): ...
```

**Convert once, at the `fit` boundary** — classic and autodiff methods share the same front door:

```python
def fit(self, data, ...):
    ad = ArrayData.from_data(data, backend=self._resolve_backend(data))
    score = BIC(ad)                       # passthrough — no re-conversion
    ...                                   # NOTEARS instead optimizes on ad.matrix
```

**Representation is fixed when the `ArrayData` is built.** The single `matrix` is *either* integer codes (discrete) *or*
floats (continuous), decided in `from_data` from dtype inference plus the method's declared data type. Pure-discrete →
codes + cardinalities; pure-continuous → floats. **Mixed / conditional-Gaussian** (needs both in one bundle) does not fit
a single matrix and stays on the deferred pandas path. Integer-as-discrete follows pgmpy's existing rule (mark
categorical or pass `state_names`).

**Encoding lives in `from_data`, by necessity.** Categorical labels cannot live in a numeric array, so the
value→integer-code step happens at ingestion; metric classes are label-free and address columns by index. `from_data`
owns the label↔code↔index mapping.

**Scope.** Count-based (chi²/G²) and covariance-based (partial-correlation / Fisher-Z) CI tests refactor cleanly to `xp`
metric classes. The residual-based tests (`gcm`, `generalized_cov`) use sklearn estimators + `pd.get_dummies` and stay
numpy/host (declare `prefers: numpy`) — not made backend-agnostic here.

**What `ArrayData.from_data` absorbs.** The existing helpers move inside it: `to_narwhals` / `infer_dtypes` /
`to_backend_array` (`utils/dataframe.py`), `collect_state_names` / `build_state_names` / `encode_columns`
(`utils/tabular.py`), and the legacy `preprocess_data` / `get_dataset_type` (`utils/utils.py`) — the last two superseded,
with `get_dataset_type` becoming a derived `dataset_type` property. More importantly it consolidates the ingestion
currently *duplicated* in each base class: the `preprocess_data` + `build_state_names` `__init__` pattern
(`estimators/base.py`, `parameter_estimator/base.py`, the score bases), the scattered `isinstance(pd.DataFrame)` /
numpy→frame coercion with default `x{i}` names (`causal_discovery/_base.py`, `prediction/_base.py`, several models), and
the inline name↔index map.

**What it does not.** The per-call `xp` math helpers (`get_namespace`, `xp_gammaln`, `xp_bincount`, `xp_lstsq`) and the
per-query contingency builder `get_state_counts_array` (it runs on every `local_score` from the already-encoded matrix —
computation, not ingestion); `StateNameMixin` (factor/CPD state metadata, downstream); unrelated utilities
(`discretize`, timeseries, sampling validation).

**Migration (backwards compatible).** The moved standalone functions stay as thin shims delegating to `from_data`;
`_check_fit_data` is *split* — data coercion → `from_data`, sklearn protocol (`validate_data` / `n_features_in_`) stays
on the estimator; the deprecated `estimators/CITests.py` is left untouched.

**Out of scope (deferred).** Migrating `factors`/`inference`/`sampling` off the global-switch (`config`/`compat_fns`)
model; native on-GPU conditional-Gaussian scoring; teaching `causal_discovery._check_fit_data` to pass native frames
through (needed for `GES.fit(cuDF)` to stay on-device end to end, rather than only a directly-constructed score).

## User journeys with the solution

1. **pandas user (unchanged).** `BIC(df).local_score("A", ("B",))` works exactly as today; the numpy path is
   byte-identical.
2. **Polars user.** Same API, no manual `.to_pandas()`; encoding runs natively in Polars (faster at scale).
3. **cuDF + GPU.** `BIC(cudf_frame)` converts once via `from_data` and scores **on the GPU** — no host round-trip.
   (Wiring this through `GES.fit(cuDF)` end to end needs the `_check_fit_data` change listed under out-of-scope.)
4. **NOTEARS/DAGMA author.** `fit(data)` → `ArrayData.from_data(data)` → optimize on `ad.matrix`; the loss reuses
   `gaussian_loglik` with residuals from the free weight matrix — the same data front door as the classic methods, data
   staying on device through the gradient loop.
5. **New-metric author.** Write one class that operates on `ArrayData` via `xp`; it inherits all array backends
   (numpy/torch/cupy) and all input types (via `from_data`) for free.
