## Simulation Mixin for the Datasets Module

Contributors: @Gitanaskhan26

### Introduction

pgmpy's `datasets` module currently supports two kinds of data sources: static files fetched from HuggingFace Hub
(`_BaseDataset.load_dataframe`) and data generated from covariance matrices hosted on the Hub (`_CovarianceMixin`).
Both paths ultimately return a `pd.DataFrame` through the same `load_dataset()` entry point.

Several datasets relevant to causal discovery benchmarking are *semi-simulated*: the ground-truth graph is known, and
data is generated from it via a parametric process. The News dataset (LDA-based, from Johansson et al.) and IHDP
(Hill 2011) fall into this category. Storing a single pre-computed snapshot of these datasets is limiting because:

- Users cannot vary sample size, noise levels, or other hyperparameters.
- Reproducibility requires explicit seed control, which static files cannot provide.
- The benchmarking framework benefits from being able to sweep hyperparameters across simulations.

This proposal introduces a `_SimulationMixin` class that slots into the existing mixin architecture alongside
`_CovarianceMixin` and `_TubingenBenchmarkMixin`, and extends `load_dataset()` to forward simulation hyperparameters.

Ref: [pgmpy#3336](https://github.com/pgmpy/pgmpy/issues/3336),
[example_datasets#12 (comment)](https://github.com/pgmpy/example_datasets/pull/12#issuecomment-4077141897)

---

### Proposed Solution

Extend the datasets module with three changes, all in `pgmpy/datasets/_base.py`:

1. A `_SimulationMixin` class that overrides `load_dataframe()` and `load_ground_truth()`.
2. A mandatory `_simulate(**kwargs) -> tuple[pd.DataFrame, DAG]` classmethod on simulator dataset classes.
3. A `**kwargs` parameter on `load_dataset()` that gets forwarded to `load_dataframe()` for simulated datasets.

Everything else—tag lookup, `list_datasets()`, the `Dataset` dataclass—stays the same.

---

### Alternative Solutions

**A. Separate `simulate_dataset()` function**

The issue title mentions a `simulate_dataset` API. I considered a standalone function parallel to `load_dataset()`, but
it fragments the entry point. Users would need to know whether a dataset is static or simulated before choosing which
function to call. The existing precedent with `_TubingenBenchmarkMixin` shows that pgmpy already handles dataset-specific
arguments through the same `load_dataset()` path (the `tubingen/<pair_id>` special case in `_base.py:251-277`). Following
that pattern keeps the surface area small.

If we later want a `simulate_dataset()` convenience wrapper, it can be a thin wrapper around `load_dataset()` that
validates `is_simulated=True` on the target class. But I don't think it should be the primary API.

**B. Config objects instead of `**kwargs`**

An alternative is typed config objects per simulator (e.g., `LinearGaussianConfig(n_samples=5000, seed=42)`). This gives
better IDE autocomplete and type safety. The downside is that it adds one class per simulator dataset and makes the
`load_dataset()` signature depend on which config you're passing. I think `**kwargs` is the right call for now—it
matches how `_TubingenBenchmarkMixin.load_dataframe(pair_id)` already takes positional arguments—and we can add config
objects later if the kwargs get complex enough to warrant them.

**C. Separate `_SimulatedBaseDataset` class instead of a mixin**

We could introduce a new base class that inherits from `_BaseDataset` and adds simulation behavior. The problem is that
this doesn't compose well. `_CovarianceMixin` is a mixin specifically because some covariance-based datasets could
hypothetically also be simulated. The mixin pattern is already established and works.

---

### Details of proposed solution

#### 1. The `_SimulationMixin` class

The mixin follows the same pattern as `_CovarianceMixin` (see `_base.py:172-213`): it overrides `load_dataframe()` to
change where the data comes from. The key difference is that simulation also produces a ground-truth DAG, so the mixin
overrides `load_ground_truth()` as well.

```python
class _SimulationMixin:
    """
    Mixin for datasets where data is generated on-the-fly via a simulation process.
    Overrides ``load_dataframe`` and ``load_ground_truth`` to call the subclass's
    ``_simulate`` method.

    Subclasses must implement ``_simulate(**kwargs) -> tuple[pd.DataFrame, DAG]``.
    The first element is the simulated data, and the second is the ground-truth
    causal graph used to generate it.

    When using this mixin, it should be the first parent class so that its
    ``load_dataframe`` and ``load_ground_truth`` take precedence in the MRO
    (same convention as ``_CovarianceMixin``).
    """

    # Class-level cache so that load_dataframe() and load_ground_truth()
    # return consistent results from a single _simulate() call.
    _cached_data: pd.DataFrame | None = None
    _cached_ground_truth: DAG | None = None

    @classmethod
    def _simulate(cls, **kwargs) -> tuple[pd.DataFrame, DAG]:
        """
        Generate simulated data and the ground-truth DAG.

        Must be implemented by each simulator dataset class.

        Returns
        -------
        tuple[pd.DataFrame, DAG]
            (data, ground_truth_dag)
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _simulate()."
        )

    @classmethod
    def load_dataframe(cls, **kwargs) -> pd.DataFrame:
        """
        Simulate data by calling _simulate() with the given hyperparameters.
        Caches both the data and ground-truth DAG on the class so that
        a subsequent load_ground_truth() call returns the matching graph.
        """
        data, ground_truth = cls._simulate(**kwargs)
        cls._cached_data = data
        cls._cached_ground_truth = ground_truth
        return data

    @classmethod
    def load_ground_truth(cls) -> DAG:
        """Return the ground-truth DAG from the most recent simulation."""
        if cls._cached_ground_truth is None:
            # If someone calls load_ground_truth before load_dataframe,
            # run a simulation with default parameters.
            cls.load_dataframe()
        return cls._cached_ground_truth
```

**Why this design:**

- **`_simulate` returns a tuple.** A simulated dataset's ground truth is produced *during* generation (the graph is
  constructed, then data is sampled from it). Splitting this into two independent methods would either duplicate work or
  require hidden shared state. The tuple return makes the coupling explicit.

- **Class-level cache.** `load_dataset()` calls `load_dataframe()` and `load_ground_truth()` independently
  (`_base.py:287-293`). The cache ensures they refer to the same simulation run. This mirrors how
  `_CovarianceMixin.load_dataframe()` reads its covariance matrix from Hub once and generates data from it—except here
  the "source" is the `_simulate()` call itself.

- **Classmethods, not instance methods.** Every method in the current dataset architecture is a classmethod
  (`_base.py:132-169`). Switching to instance methods for simulation datasets would break the uniformity that
  `load_dataset()` relies on (it calls `target_cls.load_dataframe()`, not `target_cls().load_dataframe()`). Per-call
  hyperparameters are passed as kwargs instead of stored as instance state. This is slightly less ergonomic than instance
  methods, but it's consistent.

- **Mixin sets `is_simulated` automatically.** Each concrete simulator class should set `"is_simulated": True` in its
  `_tags` dict. I considered having the mixin override `_tags` itself, but `skbase` tag resolution across multiple
  inheritance is already well-defined—child class tags override parent tags—so it's cleaner to be explicit in each
  simulator class. This also lets `list_datasets(is_simulated=True)` work without changes.

#### 2. Changes to `load_dataset()`

The only change needed in `load_dataset()` is accepting and forwarding `**kwargs`:

```diff
-def load_dataset(name: str) -> Dataset:
+def load_dataset(name: str, **kwargs) -> Dataset:
```

And in the return block for non-Tubingen datasets (`_base.py:287-293`):

```diff
     return Dataset(
         name=name,
-        data=target_cls.load_dataframe(),
+        data=target_cls.load_dataframe(**kwargs),
         expert_knowledge=target_cls.load_expert_knowledge(),
         ground_truth=target_cls.load_ground_truth(),
         tags=target_cls.get_class_tags(),
     )
```

For non-simulated datasets, `load_dataframe()` takes no arguments (`_base.py:133`), so passing non-empty `**kwargs`
will raise a `TypeError` from Python's call mechanics. I think that's the right behavior—it's a clear error message
with no extra code. If we want a friendlier error, we could add an explicit check:

```python
if kwargs and not issubclass(target_cls, _SimulationMixin):
    raise ValueError(
        f"Dataset '{name}' is not a simulation dataset and does not accept extra arguments. "
        f"Got: {list(kwargs.keys())}"
    )
```

I'd lean toward adding this check. It costs one line and turns `TypeError: load_dataframe() got an unexpected keyword
argument 'n_samples'` into something that tells the user *why* their call failed.

#### 3. Reference simulator: `LinearGaussianSCM`

For the initial validation of the architecture, I'll implement a linear Gaussian SCM simulator. This is a natural
choice because pgmpy already has `LinearGaussianBayesianNetwork.get_random()` and `.simulate()`, so the simulator
is thin glue between existing functionality. See the prototype in
[`prototype_linear_gaussian_scm.py`](prototype_linear_gaussian_scm.py).

#### 4. File-level summary of changes

| File | Change |
|------|--------|
| `pgmpy/datasets/_base.py` | Add `_SimulationMixin` class (~30 lines). Add `**kwargs` to `load_dataset()` signature and forward to `load_dataframe()`. Add explicit error for kwargs on non-simulated datasets. |
| `pgmpy/datasets/__init__.py` | Export `_SimulationMixin`. |
| `pgmpy/datasets/linear_gaussian_scm.py` | **[NEW]** Reference simulator implementation (~40 lines). |
| `pgmpy/tests/test_datasets/` | **[NEW]** Tests for mixin routing, kwargs forwarding, seed reproducibility, error on kwargs for static datasets. |

#### 5. Open questions

**Should we also update the Tubingen special case?** The `tubingen/<pair_id>` branch in `load_dataset()`
(`_base.py:251-277`) hardcodes `pair_id` handling rather than using `**kwargs`. It would be more consistent to route
Tubingen through `**kwargs` too (e.g., `load_dataset("tubingen", pair_id=42)`), but that changes existing user-facing
behavior. I'd defer this to a separate PR to keep scope tight.

**Tag values for variable-size datasets.** Static datasets have fixed `n_variables` and `n_samples` in their tags.
Simulated datasets don't—these depend on the kwargs. I set them to `None` in the class tags, and `load_dataset()`
could update the returned `Dataset.tags` dict with the actual values after simulation (the way it already does for
Tubingen at `_base.py:266-267`). Is that the right approach, or should tags remain static?

**Cache invalidation.** The class-level cache means that calling `load_dataset("linear_gaussian_scm", seed=1)` followed
by `load_dataset("linear_gaussian_scm", seed=2)` will overwrite the first cache. This is fine for normal usage but
could surprise someone holding a reference to the class. An alternative is to skip caching entirely and have
`load_dataset()` call `_simulate()` once and pass both results into the `Dataset` constructor directly. That would
require a small refactor of the `load_dataset()` return block for simulated datasets. I can go either way here—would
appreciate your preference.

---

### User journeys with the solution

**Basic simulation with defaults**

```python
>>> from pgmpy.datasets import load_dataset
>>> ds = load_dataset("linear_gaussian_scm")
>>> ds.data.shape
(1000, 5)
>>> ds.ground_truth.edges()
OutEdgeView([('X_2', 'X_0'), ('X_3', 'X_0'), ...])
```

**Custom hyperparameters**

```python
>>> ds = load_dataset("linear_gaussian_scm", n_samples=5000, n_nodes=10, edge_prob=0.2, seed=42)
>>> ds.data.shape
(5000, 10)
```

**Reproducibility via seed**

```python
>>> ds1 = load_dataset("linear_gaussian_scm", n_samples=500, seed=42)
>>> ds2 = load_dataset("linear_gaussian_scm", n_samples=500, seed=42)
>>> ds1.data.equals(ds2.data)
True
>>> ds1.ground_truth.edges() == ds2.ground_truth.edges()
True
```

**Filtering simulated datasets**

```python
>>> from pgmpy.datasets import list_datasets
>>> list_datasets(is_simulated=True)
['linear_gaussian_scm']
```

**Error on kwargs for static datasets**

```python
>>> load_dataset("sachs_continuous", n_samples=100)
ValueError: Dataset 'sachs_continuous' is not a simulation dataset and does not accept
extra arguments. Got: ['n_samples']
```

**Writing a new simulator** (future contributor)

```python
from pgmpy.datasets._base import _BaseDataset, _SimulationMixin

class MyCustomSimulator(_SimulationMixin, _BaseDataset):
    _tags = {
        "name": "my_custom",
        "is_simulated": True,
        "has_ground_truth": True,
        # ... remaining tags ...
    }

    @classmethod
    def _simulate(cls, n_samples=500, my_param=0.5, seed=None, **kwargs):
        # build graph, sample data, return (df, dag)
        ...
        return data, dag
```

Then `load_dataset("my_custom", n_samples=1000, my_param=0.8)` works with no changes to the framework.

---

### Rollout plan

**Phase 1 (this PR):** `_SimulationMixin` + `load_dataset(**kwargs)` + `LinearGaussianSCM` + tests.

**Phase 2:** Add IHDP simulator (Hill 2011 DGP). This is a good second test of the architecture because it requires
loading covariates from a static file and then simulating treatment/outcome on top of them—a hybrid of static and
simulated.

**Phase 3:** Add News simulator (LDA-based, from Johansson et al.). This one has heavier dependencies (LDA fitting)
and may need optional-dependency guards.
