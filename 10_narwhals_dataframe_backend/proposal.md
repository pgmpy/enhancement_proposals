## Replace Hardcoded `pandas` Assumptions with `narwhals` for DataFrame-Agnostic Data Ingestion

Contributors: **[`@direkkakkar319`](https://github.com/direkkakkar319-ops)**

---

### Introduction

This proposal addresses the narwhals part of **[`Issue 3395`](https://github.com/pgmpy/pgmpy/issues/3395)**.
It complements the maintainer's architectural design in **[`PR #14`](https://github.com/pgmpy/enhancement_proposals/pull/14)** by **[`@ankurankan`](https://github.com/ankurankan)**.
pgmpy currently hardcodes `pandas.DataFrame` as the only accepted tabular input type across its entire public API-parameter estimators, conditional independence tests, structure scores, causal discovery algorithms, and model-level methods like `fit()`, `predict()`, and `simulate()`.

This creates several practical problems like:
1. **Ecosystem lock-in.** Users working with **[`Polars`](https://github.com/pola-rs/polars)** , **[`PyArrow`](https://github.com/apache/arrow)** , **[`cuDF`](https://github.com/rapidsai/cudf)** or **[`Modin`](https://github.com/modin-project/modin)** must manually convert their data to pandas before calling any pgmpy function, and convert back afterward.

2. **Unnecessary full pandas conversions.** Converting third-party DataFrames (e.g., Polars, PyArrow, cuDF) to pandas duplicates data in memory and forces users to pay the conversion overhead upfront. Supporting Narwhals lets pgmpy skip the full pandas conversion entirely, allowing each backend to perform column encoding natively.

3. **Maintenance burden.** The codebase uses pandas-specific idioms (`pd.api.types.is_integer_dtype`, `pd.factorize`, `groupby(...).size().unstack()`, `pd.MultiIndex`, `pd.Categorical`, etc.) in many source files. Every new feature must be written against `Pandas` internals, and any upstream  `Pandas` deprecation requires a sweep across the entire codebase.

4. **Growing ecosystem fragmentation.** `Polars`, adoption is accelerating rapidly. Libraries that accept only `Pandas` are increasingly seen as legacy. Supporting multiple DataFrame backends positions `Pgmpy` for long-term relevance.

---

### Proposed Solution

Introduce `narwhals` as a lightweight compatibility layer between pgmpy's internal logic and the user's DataFrame library.
Narwhals provides a Polars-inspired API that wraps native DataFrames *without copying data*.
It supports pandas, Polars, PyArrow, cuDF, Modin, and other backends, and has zero required dependencies. It only uses libraries the user already has installed.

**Core pattern:** At each public API entry point (estimator `.fit()`, CI test `__init__`, model `.fit()` / `.predict()` / `.simulate()`), wrap the incoming DataFrame using `nw.from_native(df, eager_only=True)`, perform all internal operations using the narwhals API, and convert back to the user's native format via `.to_native()` before returning.

**What does NOT change:**
- The internal factor/inference layer continues to operate on numpy arrays (or torch tensors via `array-api-compat` in the future). The narwhals boundary stops at the point where data is converted into `numpy` arrays for numerical computation.
- All existing pandas-based user code continues to work identically. pandas is a first-class narwhals backend.

**Dependency & Compatibility Policy:**
- **Pinning:** Add `narwhals>=2.11` as a core dependency in `pyproject.toml`. Version `>=2.11` is required because `replace_strict(..., default=...)`, which the contingency table encoding relies on, was introduced in Narwhals 2.11.
- **Stable Import:** In all pgmpy codebase files, import from the stable API namespace: `import narwhals.stable.v2 as nw` rather than top-level `narwhals`. The `stable.v2` namespace provides a [backwards-compatibility guarantee](https://narwhals-dev.github.io/narwhals/backcompat/), ensuring that future Narwhals releases won't change behavior under pgmpy's feet.

---

### Alternative Solutions

#### 1. Convert Everything to `pandas` at the Boundary

Accept any DataFrame type, immediately convert to pandas via `nw.from_native(DataFrame).to_pandas()`, then proceed with existing pandas code unchanged.

**[`docs-"to_pandas()"`](https://narwhals-dev.github.io/narwhals/api-reference/dataframe/#narwhals.dataframe.DataFrame.to_pandas)**

- **Pros:** Minimal code changes. Existing internal logic stays untouched.This can serve as an initial **Phase 0 step**: adding `nw.from_native(data, eager_only=True).to_pandas()` at every public entry point gives users multi-backend input immediately across all pgmpy APIs, allowing subsequent modules to drop the conversion and migrate to native Narwhals incrementally.
- **Cons:** Defeats the long-term purpose as an end state. Users still pay the full conversion overhead, memory duplication, and device-to-host transfer for cuDF. No multi-backend execution, merely an ingestion shim.

---

#### 2. Use `narwhals` for Transparent, Zero-Copy Wrapping **(Proposed Solution)**

Wrap DataFrames at entry points, operate through narwhals' common API internally, return native types to the user.

- **Pros:** Zero-copy. True multi-backend support. Single code path to maintain. Narwhals is well-maintained and adopted by scikit-lego, Hamilton, and other ecosystem libraries [ecosystem libraries](https://narwhals-dev.github.io/narwhals/ecosystem/).Lightweight, zero-dependency
- **Cons:** Requires migrating internal pandas idioms to narwhals equivalents. While categorical dtypes are supported via `nw.Categorical` and `nw.Enum`, operations relying on `pd.MultiIndex` have no direct equivalent and require restructuring. Narwhals provides pandas index interoperability helpers (`maybe_get_index`, `maybe_align_index`) to ease transitional paths. One-time migration effort.

**Conclusion:** Option 2 is the clear winner. It provides genuine multi-backend support with minimal ongoing maintenance.

---

### Details of Proposed Solution

#### Overview

The migration is organized by module. Each module below can be migrated as a separate backward-compatible PR(as advised). The modules are listed in dependency order - earlier modules have no dependencies on later ones, so they can be merged first.

The narwhals boundary is simple: `nw.from_native(data, eager_only=True)` is called once at each entry gate. There are two entry gates - one for estimators (`_initialize_fit` in `base.py`) and one for CI tests (`__init__` in `power_divergence.py`). Eager-only enforcement ensures that if an uncollected LazyFrame (e.g. `polars.LazyFrame`) is passed, Narwhals immediately raises a clear error rather than failing unexpectedly downstream. Everything downstream receives an eager narwhals DataFrame and works with the narwhals API. At the numpy boundary (TabularCPD, `np.bincount`), data gets converted to numpy arrays and narwhals is no longer involved.

```
Runtime data flow:

  User passes pandas/polars/pyarrow DataFrame
                    |
                    v
  Entry Gate: nw.from_native(data, eager_only=True)
  (base.py _initialize_fit  OR  power_divergence.py __init__)
                    |
                    v
  Internal functions use narwhals API
  (preprocess_data, collect_state_names, get_state_counts, etc.)
                    |
                    v
  Numpy Boundary: .to_numpy() / np.array()
  (TabularCPD, bincount)
```

---

#### Module: `pgmpy/utils/tabular.py`

**Depends on:** Nothing (leaf module)

**Functions to migrate:** [`collect_state_names()`](pgmpy/utils/tabular.py#L7), [`build_state_names()`](pgmpy/utils/tabular.py#L12), [`get_state_counts()`](pgmpy/utils/tabular.py#L32), [`encode_columns()`](pgmpy/utils/tabular.py#L72)

`collect_state_names` and `build_state_names` are called by every discrete estimator.

```python
# Before
def collect_state_names(data: pd.DataFrame, variable: str) -> list:
    return sorted(list(data.loc[:, variable].dropna().unique()))

# After
def _observed(s: nw.Series) -> nw.Series:
    s = s.drop_nulls()
    if s.dtype.is_float():
        # NOTE: Polars/PyArrow keep NaN after drop_nulls, pandas doesn't
        s = s.filter(~s.is_nan())
    return s.unique()


def collect_state_names(data: nw.DataFrame, variable: str) -> list:
    # Use get_column() to safely support integer column names (in Narwhals, data[0] selects row 0)
    return _observed(data.get_column(variable)).sort().to_list()
```

> **Difference in Pandas and Polars or PyArraws Behaviour:**
> 1. **Null vs. NaN parity:** In pandas, `.dropna()` removes both `None` and float `NaN`. In Polars and PyArrow, `.drop_nulls()` only drops `None`, leaving float `NaN` in place. Filtering `~s.is_nan()` on float columns guarantees identical state collection across all backends.
> 2. **Integer column names:** pgmpy variables can be integers or strings. In Narwhals, `data[0]` selects row 0, not column 0. Using `data.get_column(variable)` guarantees safe column selection for all hashable variable names.

**reference-docs**

**[`docs-"get_column()"`](https://narwhals-dev.github.io/narwhals/api-reference/dataframe/#narwhals.dataframe.DataFrame.get_column)**

**[`docs-"is_nan()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.is_nan)**

**[`docs-"drop_nulls()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.drop_nulls)**

**[`docs-"to_list()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.to_list)**

**[`docs-"unique()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.unique)**

The hardest function to migrate: uses `groupby(...).size().unstack()`, `pd.MultiIndex.from_product`, and `.reindex()`. The proposed approach is to route through the existing `get_state_counts_array()` (which already operates on integer codes + numpy), avoiding the pandas-heavy path entirely:

```python
# Before
def get_state_counts(data, state_names, variable, parents=(), sample_weight=None, reindex=True):
    parents = list(parents)
    if sample_weight is None:
        if not parents:
            state_count_data = data.loc[:, variable].value_counts()
            return state_count_data.reindex(state_names[variable]).fillna(0).to_frame()
        state_count_data = data.groupby([variable] + parents, observed=True).size().unstack(parents)
    else:
        ...

# After
def get_state_counts(data, state_names, variable, parents=(), sample_weight=None, reindex=True):
    codes, cardinalities = encode_columns(data, state_names)
    return get_state_counts_array(codes, cardinalities, variable, parents, sample_weight)
```

To support this path without pandas, `encode_columns()` must also be migrated. The current code uses `Index.get_indexer` (not `pd.Categorical`). The narwhals-idiomatic replacement is [`Series.replace_strict(mapping, default=-1)`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.replace_strict), which maps values to integer codes natively within the narwhals layer and avoids a Python-level loop:

```python
# Before
def encode_columns(data: pd.DataFrame, state_names: dict) -> tuple[dict, dict]:
    codes = {}
    cardinalities = {}
    for col in data.columns:
        cats = state_names[col]
        idx = pd.Index(cats)
        codes[col] = np.asarray(idx.get_indexer(data[col]), dtype=np.int64)
        cardinalities[col] = len(cats)
    return codes, cardinalities

# After
def encode_columns(data: nw.DataFrame, state_names: dict) -> tuple[dict, dict]:
    codes = {}
    cardinalities = {}
    for col in data.columns:
        s = data.get_column(col)
        cats = state_names[col]
        val_code = {v: i for i, v in enumerate(cats)}  # equivalent to Index.get_indexer
        if (old := _observed(s)):
            new = [val_code.get(v, -1) for v in old]
            codes[col] = s.replace_strict(old, new, default=-1, return_dtype=nw.Int64).to_numpy()
        else:
            codes[col] = np.full(len(s), -1, dtype=np.int64)
        cardinalities[col] = len(cats)
    return codes, cardinalities
```

**[`docs-"replace_strict()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.replace_strict)**

> **Performance note:** `replace_strict` operates within the narwhals layer, so the encoding happens natively in the backend engine (pandas, Polars, PyArrow, etc.) rather than in a Python-level loop. The final `.to_numpy()` call does copy data to a numpy array - for GPU-resident backends like cuDF, this triggers a device-to-host transfer, which is unavoidable since downstream computation (`np.bincount`) requires numpy. True GPU-native computation will be addressed in the array-api phase.

**Backward compatibility:** The return type of `get_state_counts` changes from `pd.DataFrame` to `np.ndarray`. This is a **public API break**. All **internal** callers already do `np.array(state_counts)` immediately after calling this function, so the change is transparent therefore the internal behavior is unchanged.
If external users depend on the DataFrame return type, we will:
1. Keep `get_state_counts` returning a `pd.DataFrame` and add a new `get_state_counts_array` caller internally.
2. Or add a `DeprecationWarning` for one release cycle before changing the return type.

The preferred option during implementation will be decided with the maintainer.

**Output difference**
```python
# for this input
df = pd.DataFrame({"B": [0, 0, 0, 1], "A": [0, 0, 1, 1]})
get_state_counts(
    data=df,
    state_names={"A": [0, 1], "B": [0, 1]},
    variable="A",  # The variable we want to count
    parents=["B"],  # Its parent(s)
)

# Before -- this was converted by the estimator in numpy format after the input `np.array(state_counts)`
B     0    1
A           
0   2.0  0.0
1   1.0  1.0

# After
array([[2.0, 0.0],[1.0, 1.0]])

```
---

#### Module: `pgmpy/utils/utils.py`

**Depends on:** Nothing (leaf module)

**Functions to migrate:** [`preprocess_data()`](pgmpy/utils/utils.py#L293)

Every discrete estimator calls this first. The current code uses `pd.api.types.*` for dtype inference and mutates columns in-place with `df[col] = df[col].astype("category")`. Polars and PyArrow DataFrames are immutable, so narwhals does not allow item assignment. Instead, we use `df.with_columns()` to reconstruct the dataframe with the casted columns:

```python
# Before (pandas-only, mutates in-place)
import pandas as pd

def preprocess_data(df):
    df = df.copy()
    dtypes = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            df[col] = df[col].astype("int")
            dtypes[col] = "N"
        elif pd.api.types.is_numeric_dtype(df[col]):
            dtypes[col] = "N"
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            dtypes[col] = "C"
            df[col] = df[col].astype("category")
        # ...
    return (df, dtypes)


# After (narwhals-compatible, uses with_columns)
# nw.from_native() is NOT called here. _initialize_fit() already wraps the
# DataFrame before calling preprocess_data().
import narwhals.stable.v2 as nw

def preprocess_data(df):
    dtypes = {}
    casts = []
    for col in df.columns:
        col_dtype = df[col].dtype
        if col_dtype.is_numeric():
            dtypes[col] = "N"
        elif col_dtype == nw.String:
            dtypes[col] = "C"
            casts.append(nw.col(col).cast(nw.Categorical))
        elif col_dtype == nw.Categorical:
            if nw.is_ordered_categorical(df[col]):
                dtypes[col] = "O"
            else:
                dtypes[col] = "C"
        elif col_dtype == nw.Enum:
            dtypes[col] = "O"
        else:
            raise ValueError(
                f"Couldn't infer datatype of column: {col} from data. "
                "Try specifying the appropriate datatype to the column."
            )

    if casts:
        df = df.with_columns(casts)

    return (df, dtypes)
```

**Backward compatibility:** Fully backward compatible. `with_columns` returns a new DataFrame with the casted columns, preserving the original behavior of returning a transformed DataFrame. When the input is a narwhals-wrapped pandas DataFrame, `nw.col(col).cast(nw.Categorical)` maps to `pd.Categorical` under the hood. Ordered categoricals continue to be mapped to `"O"` preserving compatibility with existing tests.

---

#### Module: `pgmpy/parameter_estimator/base.py`

**Depends on:** `tabular.py`, `utils.py`

**Functions to migrate:** [`_initialize_fit()`](pgmpy/parameter_estimator/base.py#L172)

This is the primary entry gate for all parameter estimators. Adding `nw.from_native()` every estimator's `fit()` method gets narwhals support automatically:

```python
# Before
def _initialize_fit(self, model, data, sample_weight=None):
    self._data, self._dtypes = preprocess_data(data)
    self._model = model
    self._sample_weight = sample_weight
    self.state_names_ = self._build_fitted_state_names(model, self._data)

# After
def _initialize_fit(self, model, data, sample_weight=None):
    data = nw.from_native(data, eager_only=True)
    self._data, self._dtypes = preprocess_data(data)
    self._model = model
    self._sample_weight = sample_weight
    self.state_names_ = self._build_fitted_state_names(model, self._data)
```

**Backward compatibility:** Fully backward compatible. `nw.from_native()` on a pandas DataFrame returns a narwhals-wrapped pandas DataFrame. All downstream code works the same.

**narwhals docs:**
[`from_native()`](https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_native)

---

#### Module: `pgmpy/parameter_estimator/discrete_mle.py`

**Depends on:** `base.py`, `tabular.py`

**Functions to migrate:** [`_estimate_cpd()`](pgmpy/parameter_estimator/discrete_mle.py#L65), [`fit()`](pgmpy/parameter_estimator/discrete_mle.py#L91)

This is the pilot target for the migration. The full flow is: `fit()` -> `_initialize_fit()` -> `preprocess_data()` -> `build_state_names()` -> per-node: `_estimate_cpd()` -> `get_state_counts()` -> `TabularCPD(state_counts)`.

Since `get_state_counts()` now returns a 2D numpy array (see `tabular.py` module above), `_estimate_cpd()` needs to use numpy indexing instead of pandas `.iloc` and `.values`:

```python
# Before
@staticmethod
def _estimate_cpd(model, data, state_names: dict, node, sample_weight=None) -> TabularCPD:
    parents = sorted(model.get_parents(node))
    state_counts = get_state_counts(
        data=data,
        state_names=state_names,
        variable=node,
        parents=parents,
        sample_weight=sample_weight,
    )
    state_counts.iloc[:, (state_counts.values == 0).all(axis=0)] = 1.0

    parents_cardinalities = [len(state_names[parent]) for parent in parents]
    node_cardinality = len(state_names[node])

    cpd = TabularCPD(
        node,
        node_cardinality,
        np.array(state_counts),
        evidence=parents,
        evidence_card=parents_cardinalities,
        state_names={var: state_names[var] for var in chain([node], parents)},
    )
    cpd.normalize()
    return cpd

# After
@staticmethod
def _estimate_cpd(model, data, state_names: dict, node, sample_weight=None) -> TabularCPD:
    parents = sorted(model.get_parents(node))
    state_counts = get_state_counts(
        data=data,
        state_names=state_names,
        variable=node,
        parents=parents,
        sample_weight=sample_weight,
    )
    state_counts[:, (state_counts == 0).all(axis=0)] = 1.0

    parents_cardinalities = [len(state_names[parent]) for parent in parents]
    node_cardinality = len(state_names[node])

    cpd = TabularCPD(
        node,
        node_cardinality,
        state_counts,
        evidence=parents,
        evidence_card=parents_cardinalities,
        state_names={var: state_names[var] for var in chain([node], parents)},
    )
    cpd.normalize()
    return cpd
```

**Backward compatibility:** Fully backward compatible. The `fit()` method still accepts pandas DataFrames. The internal change from `.iloc` to numpy indexing is transparent to users.

---

#### Module: `pgmpy/ci_tests/power_divergence.py`

**Depends on:** Nothing (independent entry gate)

**Functions to migrate:** [`__init__()`](pgmpy/ci_tests/power_divergence.py#L130)

CI tests have a different pandas usage pattern than estimators. They use column selection and `pd.factorize()` for encoding. This module has its own entry gate (separate from the estimator `_initialize_fit()`) since CI tests are not estimators.

Narwhals does not expose `factorize` directly, but the same result can be achieved:

```python
# Before
for col in data.columns:
    codes, uniques = pd.factorize(data[col], sort=False, use_na_sentinel=True)
    self._codes[col] = np.ascontiguousarray(codes, dtype=np.int64)
    .........

# After
df = nw.from_native(data, eager_only=True)
for col in df.columns:
    col_series = df[col]
    unique_vals = col_series.drop_nulls().unique().sort().to_list()
    val_to_code = {v: i for i, v in enumerate(unique_vals)}
    native_values = col_series.to_list()
    codes = np.array([val_to_code.get(v, -1) for v in native_values], dtype=np.int64)
    self._codes[col] = codes
    .........
```

> **Code ordering difference:** `pd.factorize(sort=False)` assigns codes in *first-seen* order, while `unique().sort()` assigns codes in *sorted* order. These produce different integer code assignments for same data. The narwhals path uses sorted order, which is like how `encode_columns` / `build_state_names` works elsewhere in pgmpy (state names are always sorted). The CI test result (p-value, statistic) is order-independent - only the contingency table cell counts matter, not which integer maps to which label. However, this difference must be documented and tested to avoid surprises.

**Backward compatibility:** Fully backward compatible for external users -the p-value and test decision are identical. pandas DataFrames are wrapped transparently.

**reference-docs**
**[`docs-"from_native()"`](https://narwhals-dev.github.io/narwhals/api-reference/narwhals/#narwhals.from_native)**

**[`docs-"unique()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.unique)**

**[`docs-"sort()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.sort)**

**[`docs-"drop_nulls()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.drop_nulls)**

**[`docs-"to_list()"`](https://narwhals-dev.github.io/narwhals/api-reference/series/#narwhals.series.Series.to_list)**

> **NOTE**:
>  For performance-critical paths, we may keep `pd.factorize()` as an optimized specialization when the input is pandas, and use the narwhals path for other backends. This is a detail to decide during implementation.

---

#### Module: `pgmpy/models/DiscreteBayesianNetwork.py`

**Depends on:** All estimator modules

**Functions to audit:** [`fit()`](pgmpy/models/DiscreteBayesianNetwork.py#L602), [`predict()`](pgmpy/models/DiscreteBayesianNetwork.py#L730), [`simulate()`](pgmpy/models/DiscreteBayesianNetwork.py#L1393)

1. **`fit()`**: Passes data directly to an estimator; the `_initialize_fit()` entry gate in `base.py` already handles the `nw.from_native(data, eager_only=True)` wrapping. Fully backward compatible.

2. **`predict()` (Index-less backend adaptation)**:
   - Current implementation relies on the pandas index (`t.index.tolist()`, `complete_data.index = row`, `sort_index()`) to preserve row alignment for missing variables.
   - Because Polars and PyArrow are index-less, `predict()` will be refactored to track rows via **positional integer indices** rather than relying on a pandas `Index`.
   - Results are converted back to the input's native type via `.to_native()`. When the input is pandas, the original index is restored; when the input is an index-less backend (Polars, PyArrow), a standard native DataFrame is returned with rows matching the input order.

3. **`simulate()` (Output type decision)**:
   - Unlike ingestion methods, `simulate(n_samples=...)` generates synthetic data from scratch and does not receive an input DataFrame to infer a native type from.
   - For backward compatibility, `simulate()` will continue to return a `pandas.DataFrame` by default.
   - An optional `output_type: str = "pandas"` argument (supporting `"polars"`, `"pyarrow"`) will be introduced to allow generating native frames directly without manual user conversion.

**Backward compatibility:** Fully backward compatible. Existing pandas workflows remain unchanged.

---

#### Rollout Summary

Each module above is a separate PR. The table below shows the merge order:

| PR | Module | Backward Compatible |
|----|--------|---------------------|
| 1 | `pgmpy/utils/tabular.py` | Yes |
| 2 | `pgmpy/utils/utils.py` | Yes |
| 3 | `pgmpy/parameter_estimator/base.py` | Yes (depends on PR 1, 2) |
| 4 | `pgmpy/parameter_estimator/discrete_mle.py` | Yes (depends on PR 3) |
| 5 | `pgmpy/ci_tests/power_divergence.py` | Yes (independent) |
| 6 | `pgmpy/models/DiscreteBayesianNetwork.py` | Yes (depends on PR 3, 4) |

After the pilot modules above, the same pattern extends to the rest of the codebase:

| Scope | Modules |
|-------|---------|
| Remaining parameter estimators | `discrete_bayesian.py`, `discrete_em.py`, `linear_gaussian_mle.py` |
| Remaining CI tests | `fisher_z.py`, `pearsonr.py`, `pillai_trace.py` |
| Structure scores | `BDeu`, `BDs`, `BIC`, `AIC`, `K2`, `LogLikelihood` |
| Causal discovery | `PC`, `GES`, `HillClimbSearch`, `ChowLiu` |

---

#### **Testing**

Each migrated module will include parameterized tests that run the same assertions across multiple backends.

We will adopt a backend parametrization pattern modeled after [Fairlearn's `test/conftest.py`](https://github.com/fairlearn/fairlearn/blob/main/test/conftest.py):
- **Centralized Fixtures:** Pytest fixtures will parametrize incoming tabular data across `pandas`, `polars`, and `pyarrow`.
- **Dynamic Skipping:** Optional backends (`polars`, `pyarrow`, `cudf`) will be guarded with `pytest.importorskip()` so tests run when the library is present and skip cleanly when not installed without failing CI.
- **Backend Parity:** Assertions will verify exact parity across backends for learned parameters, state names ordering, contingency tables, and CI test statistics/p-values.

---

### User Journeys with the Solution

#### 1. Existing `pandas` User (No Change Required)

```python
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE

data = pd.DataFrame({"A": [0, 1, 0, 1], "B": [1, 0, 1, 0]})
model = DiscreteBayesianNetwork([("A", "B")])
estimator = DiscreteMLE()
estimator.fit(model, data)
print(estimator.parameters_)
```

#### 2. `Polars` User (New Capability)

```python
import polars as pl
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE

data = pl.DataFrame({"A": [0, 1, 0, 1], "B": [1, 0, 1, 0]})
model = DiscreteBayesianNetwork([("A", "B")])
estimator = DiscreteMLE()
estimator.fit(model, data)
print(estimator.parameters_)
```

#### 3. CI Test with `PyArrow` Table

```python
import pyarrow as pa
from pgmpy.ci_tests import ChiSquare
import numpy as np

np.random.seed(42)
table = pa.table({"X": np.random.randint(0, 2, 1000),
                   "Y": np.random.randint(0, 2, 1000),
                   "Z": np.random.randint(0, 3, 1000)})

test = ChiSquare(data=table)
result = test(X="X", Y="Y", Z=["Z"], significance_level=0.05)
print(f"Independent: {result}, p-value: {test.p_value_:.4f}")
```

#### 4. `cuDF` User on G.P.U

```python
import cudf
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteMLE

data = cudf.DataFrame({"A": [0, 1, 0, 1], "B": [1, 0, 1, 0]})
model = DiscreteBayesianNetwork([("A", "B")])
estimator = DiscreteMLE()
estimator.fit(model, data)
```

> **Note:** The narwhals boundary does not guarantee zero device-to-host transfer for GPU-resident DataFrames. `encode_columns` calls `.to_list()` on each column, which copies column data to a Python list and pulls cuDF data to host memory. This is the same cost as calling `.to_pandas()` on individual columns. The benefit is that the full DataFrame is never materialized as a pandas DataFrame; only the columns needed for encoding are touched. A future CUDA-native `np.bincount` path (e.g., via CuPy) could eliminate this transfer entirely.

#### 5. Causal Discovery with `Polars`

```python
import polars as pl
from pgmpy.causal_discovery import PC

data = pl.read_csv("my_dataset.csv")
model = PC(ci_test="chi_square").fit(data)
print(model.edges())
```
