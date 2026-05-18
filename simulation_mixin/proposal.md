## Simulation Mixin for the Datasets Module

Contributors: @Gitanaskhan26

### Introduction

pgmpy's `datasets` module currently supports two kinds of data sources: 
1. `_BaseDataset`: static CSV data files fetched from HuggingFace Hub.
2. `_CovarianceMixin`: Data simulated from covariance matrices hosted as CSV files on HuggingFace Hub.

Both paths ultimately return a `Dataset` object through the same `load_dataset()` entry point.

In causal inference literature, another common type of dataset that are commonly used for benchmarking are semi-simulated methods. Common examples include IHDP dataset (Hill 2011), LDA-based NEWS dataset (Johansson et al.). While pgmpy provides `*BayesianNetwork.simulate` method for simulating datasets from given ground-truth models, it is not possible to generate data for these semi-simulated methods as they can not be represented using a pgmpy model. Additionally there are Bayesian Network parameters that pgmpy currently does not support for example, post-nonlinear additive noise models. 

To allow users to simulate data from such models, we propose to add a `_SimulationMixin` class. Datasets inheriting this class can define a generalized simulation mechanism while maintaining the same unified interface of `load_dataset`/`list_datasets` and returning a `Dataset` object for the user.

### Goals

- Provide standard semi-simulated datasets common in causal inference literature.
- Provide flexibility to the user to simulate data with different hyperparameters/sample sizes.
- Maintain the same unified dataset module interface.

### References
Ref: [pgmpy#3336](https://github.com/pgmpy/pgmpy/issues/3336),
[example_datasets#12 (comment)](https://github.com/pgmpy/example_datasets/pull/12#issuecomment-4077141897)

---

### Proposed Solution

Extend the datasets module with three changes, all in `pgmpy/datasets/_base.py`:

1. A `_SimulationMixin` class that defines two methods: `load_dataframe` and `load_ground_truth` overriding the `_BaseDataset`'s methods. Inheriting dataset classes implement these methods directly with their simulation logic.
2. Add `n_samples` and `seed` parameters on `load_dataset` method. In case of non `_SimulationMixin` datasets, we would return `n_samples` sampled from the full dataset.
3. Add a `**sim_kwargs` argument to `load_dataset` for simulator-specific hyperparameters. Static datasets reject unknown kwargs naturally via Python's `TypeError`.


Everything else—tag lookup, `list_datasets()`, the `Dataset` dataclass—stays the same, except that `Dataset.ground_truth` widens from `DAG | None` to `CausalGraph | None` to support different causal graph types (`DAG`, `PDAG`, `ADMG`, `MAG`; additional types like `PAG` can be added if pgmpy gains them).

---

### Alternative Solutions

**A. Separate `simulate_dataset()` function**

The issue title mentions a `simulate_dataset` API. I considered a standalone function parallel to `load_dataset()`, but it fragments the entry point. Users would need to know whether a dataset is static or simulated before choosing which function to call. The existing precedent with `_TubingenBenchmarkMixin` shows that pgmpy already handles dataset-specific arguments through the same `load_dataset()` path (the `tubingen/<pair_id>` special case in `_base.py:251-277`). Following that pattern keeps the surface area small.

If we later want a `simulate_dataset()` convenience wrapper, it can be a thin wrapper around `load_dataset()` that validates `is_simulated=True` on the target class. But I don't think it should be the primary API.

**B. Config objects instead of `sim_kwargs`**

An alternative is typed config objects per simulator (e.g., `LinearGaussianConfig(edge_prob=0.3)`). This gives better IDE autocomplete and type safety. The downside is that it adds one class per simulator dataset and makes the `load_dataset()` signature depend on which config you're passing. I think `sim_kwargs` is the right call for now and we can add config objects later if the kwargs get complex enough to warrant them.

**C. Separate `_SimulatedBaseDataset` class instead of a mixin**

We could introduce a new base class that inherits from `_BaseDataset` and adds simulation behavior. The problem is that this doesn't compose well. `_CovarianceMixin` is a mixin specifically because some covariance-based datasets could hypothetically also be simulated. The mixin pattern is already established and works.

---

### Details of proposed solution

#### 1. The `_SimulationMixin` class

The mixin follows the same structural pattern as `_CovarianceMixin` (`_base.py:172-213`): it defines the `load_dataframe` and `load_ground_truth` contract for simulated datasets while keeping dataset-specific logic inside the dataset classes.

The mixin itself only declares the contract. How each concrete simulator structures its internals—whether it uses shared helpers, separate methods, or inline logic—is up to the class. This matches how `_CovarianceMixin` and `_TubingenBenchmarkMixin` work: they put logic directly in their overridden methods without enforcing a particular internal decomposition.

The ground-truth graph is not restricted to `DAG`. Depending on the simulator, it could be a `PDAG` (equivalence class), `ADMG` (latent confounders), or `MAG`. pgmpy's graph classes don't share a single base class, so the return type uses a `Union` (additional types like `PAG` can be added if pgmpy gains them):

```python
from __future__ import annotations

from typing import Union

import pandas as pd

from pgmpy.base import DAG, ADMG, MAG, PDAG

# MAG is included for completeness; PAG does not exist in pgmpy yet.
# Current simulators will return DAG, PDAG, or ADMG.
CausalGraph = Union[DAG, PDAG, ADMG, MAG]


class _SimulationMixin:
    """
    Mixin for simulated datasets. Concrete classes must override
    ``load_dataframe()`` and ``load_ground_truth()`` with their own
    simulation logic.

    When using this mixin, it should be the first parent class so that its
    methods take precedence in the MRO (same convention as
    ``_CovarianceMixin``).
    """

    @classmethod
    def load_dataframe(cls, n_samples=None, seed=None, **sim_kwargs) -> pd.DataFrame:
        """Generate and return simulated data. Must be implemented by each simulator."""
        raise NotImplementedError(
            f"{cls.__name__} must implement load_dataframe()."
        )

    @classmethod
    def load_ground_truth(cls, **sim_kwargs) -> CausalGraph:
        """Construct and return the ground-truth graph. Must be implemented by each simulator."""
        raise NotImplementedError(
            f"{cls.__name__} must implement load_ground_truth()."
        )
```

**Why this design:**

- **`load_dataframe` and `load_ground_truth` are overridden on the mixin.** This is the same contract as `_CovarianceMixin` and `_TubingenBenchmarkMixin`. `load_dataframe()` dispatch is fully polymorphic. For `load_ground_truth()`, `load_dataset` uses a small `issubclass` check to forward `seed` and simulator kwargs only to `_SimulationMixin` datasets, since static ground truths take no parameters.

- **No enforced internal structure.** The mixin doesn't prescribe how simulators decompose their logic. Some simulators might share a `_build_model` helper between `load_dataframe` and `load_ground_truth`. Others (like IHDP) have completely independent logic in each method. This is left to the simulator author.

- **Classmethods, not instance methods.** Every method in the current dataset architecture is a classmethod (`_base.py:132-169`). Switching to instance methods would break the uniformity that `load_dataset` relies on.

- **Graph type is not restricted to DAG.** Different simulators produce different graph types. The `CausalGraph` type alias captures this without forcing all simulators into a single graph class.

- **Concrete simulator classes set `is_simulated` tag.** Each simulator class should set `"is_simulated": True` in its `_tags` dict. This lets `list_datasets(is_simulated=True)` work without changes.

#### 2. Changes to `load_dataset`

The key point: dataset-specific logic still lives in the dataset classes. `load_dataset` only needs a small branch for ground-truth loading because static ground truths take no simulation parameters, while simulated ground truths may depend on `seed` and simulator kwargs.

The only change is adding `n_samples`, `seed`, and `**sim_kwargs` to the signature and forwarding them:

```python
def load_dataset(
    name: str,
    n_samples: int | None = None,
    seed: int | None = None,
    **sim_kwargs,
) -> Dataset:
    target_cls = _resolve_dataset_class(name)  # existing lookup logic

    df = target_cls.load_dataframe(n_samples=n_samples, seed=seed, **sim_kwargs)

    if issubclass(target_cls, _SimulationMixin):
        gt = target_cls.load_ground_truth(seed=seed, **sim_kwargs)
    else:
        gt = target_cls.load_ground_truth()

    return Dataset(
        name=name,
        data=df,
        expert_knowledge=target_cls.load_expert_knowledge(),
        ground_truth=gt,
        tags=target_cls.get_class_tags(),
    )
```

`**sim_kwargs` are forwarded through the common method contract. Static datasets (`_BaseDataset`) don't accept `**sim_kwargs` in their `load_dataframe()` — passing simulator arguments to them raises a `TypeError` naturally from Python's call mechanics. No custom validation needed.

To support `n_samples` and `seed`, the existing `_BaseDataset.load_dataframe()` adds subsampling after its current CSV-loading logic:

```python
# In _BaseDataset — add n_samples/seed after existing logic (lines 166-168)
@classmethod
def load_dataframe(cls, n_samples=None, seed=None) -> pd.DataFrame:
    # ... existing CSV/HuggingFace loading logic stays unchanged ...
    if n_samples is not None:
        n_samples = min(n_samples, len(df))
        df = df.sample(n=n_samples, random_state=seed)
    return df
```

Static datasets keep the existing `load_ground_truth()` signature because their ground-truth graph is fixed and does not depend on `n_samples`, `seed`, or simulator kwargs. For `_SimulationMixin` datasets, `load_ground_truth()` can accept `seed` and simulator kwargs when the generated graph depends on them.

`_CovarianceMixin.load_dataframe()` also needs to accept `n_samples` and `seed` for compatibility, since covariance datasets are already marked `is_simulated=True` and `load_dataset()` will forward these parameters.

Note: the `Dataset` dataclass's `ground_truth` field needs to widen from `DAG | None` to `CausalGraph | None` to accommodate the different graph types that simulators can return.

#### 3. Parameter discoverability via `sim_params` tag

One issue with routing everything through `load_dataset()` is that the user has no way to discover what simulator arguments are available without reading the source code. To address this, each simulator class declares its expected parameters in a structured `sim_params` tag:

```python
_tags = {
    ...,
    "sim_params": {
        "n_nodes": {"default": 5, "desc": "Number of variables in the DAG"},
        "edge_prob": {"default": 0.3, "desc": "Probability of edge between any two nodes"},
        "noise_scale": {"default": 1.0, "desc": "Standard deviation of additive Gaussian noise"},
    },
}
```

This enables **pre-call introspection**: users can filter simulated datasets with `list_datasets(is_simulated=True)` and inspect parameters with `SomeSimulator.get_class_tag("sim_params")` to see what the simulator accepts before calling `load_dataset()`.

An alternative is to use `inspect.signature()` to extract parameter names and defaults from `load_dataframe()` and `load_ground_truth()` at runtime, which would be zero-maintenance. The downside is that it doesn't give descriptions. We could use both: `inspect` for auto-discovery, `sim_params` for descriptions when the simulator author provides them. Open to feedback on which approach is preferred.

#### 4. Reference simulator: `LinearGaussianSCM`

For the initial validation of the mixin architecture, I'll implement a linear Gaussian SCM simulator. This is a proof-of-concept rather than a core use case — pgmpy can already simulate from linear Gaussian models natively via `LinearGaussianBayesianNetwork.get_random()` + `.simulate()`. The value of wrapping it in a `_SimulationMixin` class is to validate the mixin mechanics (tag routing, `sim_kwargs` forwarding, `Dataset` construction) before building the more complex simulators that actually need this architecture. See the prototype in [`prototype_linear_gaussian_scm.py`](prototype_linear_gaussian_scm.py).

#### 5. File-level summary of changes

| File | Change |
|------|--------|
| `pgmpy/datasets/_base.py` | Add `_SimulationMixin` class (~15 lines). Update `load_dataset()` signature with `n_samples`, `seed`, `**sim_kwargs`. Add `n_samples`/`seed` subsampling to `_BaseDataset.load_dataframe()` after existing CSV logic. Update `_CovarianceMixin.load_dataframe()` to accept `n_samples`/`seed` for compatibility. Widen `Dataset.ground_truth` type from `DAG \| None` to `CausalGraph \| None`. |
| `pgmpy/datasets/__init__.py` | Export `_SimulationMixin`. |
| `pgmpy/datasets/linear_gaussian_scm.py` | **[NEW]** Reference simulator implementation (~40 lines), including `sim_params` tag. |
| `pgmpy/tests/test_datasets/` | **[NEW]** Tests for mixin routing, sim_kwargs forwarding, seed reproducibility, subsampling for static datasets (including `n_samples > len(df)`). |

#### 6. Open questions

**Should we also update the Tubingen special case?** The `tubingen/<pair_id>` branch in `load_dataset()` (`_base.py:251-277`) hardcodes `pair_id` handling rather than using `**sim_kwargs`. It would be more consistent to route Tubingen through the new parameters too (e.g., `load_dataset("tubingen", pair_id=42)`), but that changes existing user-facing behavior. I'd defer this to a separate PR to keep scope tight.

**Tag values for variable-size datasets.** Static datasets have fixed `n_variables` and `n_samples` in their class tags. Simulated datasets don't—these depend on the kwargs. Each simulator class is responsible for setting these in its own `_tags` or updating them within its `load_dataframe()`/`load_ground_truth()` methods. `load_dataset()` does not modify tags.

**`sim_params` tag vs. `inspect.signature()`.** The `sim_params` tag requires manual maintenance — if someone changes `load_dataframe()` or `load_ground_truth()`, they need to update the tag too. `inspect.signature()` would auto-extract parameter names and defaults at zero maintenance cost, but can't provide descriptions. Feedback welcome on which approach (or a hybrid) is preferred.

---

### User journeys with the solution

#### 1. Linear Gaussian SCM (proof-of-concept, DAG ground truth)

This wraps pgmpy's existing `LGBN.get_random()` + `.simulate()` in the mixin pattern. It's a proof-of-concept — pgmpy can already do this natively — but it validates the mixin mechanics end-to-end.

```python
>>> from pgmpy.datasets import load_dataset
>>> ds = load_dataset("linear_gaussian_scm", n_samples=2000, seed=42, n_nodes=8, edge_prob=0.3)
>>> ds.data.shape
(2000, 8)
>>> type(ds.ground_truth)
<class 'pgmpy.base.DAG.DAG'>
>>> ds.ground_truth.edges()
OutEdgeView([('X_2', 'X_0'), ('X_3', 'X_0'), ...])
```

The implementation for this class:

```python
class LinearGaussianSCM(_SimulationMixin, _BaseDataset):
    _tags = {
        "name": "linear_gaussian_scm", "is_simulated": True, "has_ground_truth": True,
        "sim_params": {
            "n_nodes": {"default": 5, "desc": "Number of variables in the DAG"},
            "edge_prob": {"default": 0.3, "desc": "Probability of edge between any two nodes"},
            "noise_scale": {"default": 1.0, "desc": "Std dev of additive Gaussian noise"},
        },
        ...
    }

    @classmethod
    def _build_model(cls, seed=None, n_nodes=5, edge_prob=0.3, noise_scale=1.0, **kwargs):
        """Shared helper: builds the LGBN model (graph + CPDs)."""
        return LGBN.get_random(n_nodes=n_nodes, edge_prob=edge_prob,
                               scale=noise_scale, seed=seed)

    @classmethod
    def load_ground_truth(cls, seed=None, n_nodes=5, edge_prob=0.3, **kwargs):
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob, **kwargs)
        return DAG(model.edges())

    @classmethod
    def load_dataframe(cls, n_samples=1000, seed=None, n_nodes=5,
                       edge_prob=0.3, noise_scale=1.0, **kwargs):
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob,
                                  noise_scale=noise_scale, **kwargs)
        return model.simulate(n_samples=n_samples, seed=seed)
```

`_build_model()` is a shared helper on the concrete class (not the mixin). Both methods call it with the same seed to get the same model.

#### 2. Additive Noise Model (non-linear functions, DAG ground truth)

ANM generates data where each variable is a non-linear function of its parents plus independent additive noise: `X_j = f_j(pa(X_j)) + N_j`. Causal direction is identifiable under this model when the functions and noise distributions satisfy certain conditions (Hoyer et al., 2008).

```python
>>> ds = load_dataset("anm", n_samples=3000, seed=7, n_nodes=6, function_type="gp", noise_dist="exponential")
>>> ds.data.shape
(3000, 6)
>>> type(ds.ground_truth)
<class 'pgmpy.base.DAG.DAG'>
```

The implementation for this class takes different kwargs than Linear Gaussian:

```python
class AdditiveNoiseModel(_SimulationMixin, _BaseDataset):
    _tags = {"name": "anm", "is_simulated": True, "has_ground_truth": True, ...}

    @classmethod
    def load_ground_truth(cls, seed=None, n_nodes=5, edge_prob=0.3, **kwargs):
        return DAG.get_random(n_nodes=n_nodes, edge_prob=edge_prob, seed=seed)

    @classmethod
    def load_dataframe(cls, n_samples=1000, seed=None, n_nodes=5, edge_prob=0.3,
                       function_type="gp", noise_dist="gaussian", **kwargs):
        dag = cls.load_ground_truth(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob)
        rng = np.random.default_rng(seed)
        # forward-sample through the DAG with non-linear functions
        # and additive noise drawn from noise_dist
        ...
        return data
```

Here `load_dataframe()` calls `load_ground_truth()` internally to get the DAG, then samples from it.
The internal decomposition is up to the simulator class.

#### 3. IHDP (semi-simulated: static covariates + synthetic treatment/outcome)

IHDP (Hill, 2011) is a hybrid dataset. Covariates come from a real RCT (the Infant Health and Development Program),
but treatment assignment and potential outcomes are simulated on top of them.

The ground-truth DAG is defined deterministically by the DGP specification:

- `X_i → T` for all 25 covariates (treatment assignment depends on covariates)
- `X_i → Y` for all 25 covariates (outcome depends on covariates)
- `T → Y` (treatment affects outcome)

No latent confounders in the standard specification — the simulation is fully observed, so the ground truth is a `DAG`, not an `ADMG`. The simulator constructs this DAG structure once (it's fixed), then simulates `T` via logistic assignment and `Y` using Response Surface A or B from the original paper.

This tests that `_SimulationMixin` can coexist with `_get_raw_data()` — the class uses the Hub to fetch the static covariate file, then simulates the rest.

```python
>>> ds = load_dataset("ihdp", n_samples=None, seed=123, treatment_noise=0.1, outcome_fn="response_surface_A")
>>> ds.data.shape
(747, 27)  # n_samples=None uses all 747 subjects from the original RCT
>>> type(ds.ground_truth)
<class 'pgmpy.base.DAG.DAG'>
>>> "treatment" in ds.data.columns and "y_obs" in ds.data.columns
True
>>> ds.ground_truth.number_of_edges()
51  # 25 covariates → T, 25 covariates → Y, T → Y
```

The implementation for this class:

```python
class IHDP(_SimulationMixin, _BaseDataset):
    _tags = {
        "name": "ihdp", "is_simulated": True, "has_ground_truth": True,
        "sim_params": {
            "treatment_noise": {"default": 0.0, "desc": "Noise added to treatment assignment"},
            "outcome_fn": {"default": "response_surface_A", "desc": "Response surface from Hill 2011"},
        },
        ...
    }

    covariate_url = "data/ihdp_covariates.csv"

    @classmethod
    def load_ground_truth(cls, **kwargs):
        # Fixed DAG from Hill 2011
        covariate_names = [f"x{i}" for i in range(1, 26)]
        edges = [(x, "treatment") for x in covariate_names]
        edges += [(x, "y_obs") for x in covariate_names]
        edges.append(("treatment", "y_obs"))
        return DAG(edges)

    @classmethod
    def load_dataframe(cls, n_samples=None, seed=None,
                       treatment_noise=0.0, outcome_fn="response_surface_A", **kwargs):
        raw = cls._get_raw_data(cls.covariate_url)
        covariates = pd.read_csv(io.BytesIO(raw))
        if n_samples is not None:
            covariates = covariates.sample(n=n_samples, random_state=seed)
        rng = np.random.default_rng(seed)
        # simulate treatment assignment and potential outcomes
        ...
        return data
```

Key difference from the other simulators: `load_ground_truth()` returns a fixed DAG (no kwargs needed), and `load_dataframe()` loads real covariates from the Hub before simulating treatment/outcome on top of them. Each method is fully independent—no shared helpers needed.

#### General user journeys

**Reproducibility via seed**

```python
>>> ds1 = load_dataset("linear_gaussian_scm", n_samples=500, seed=42)
>>> ds2 = load_dataset("linear_gaussian_scm", n_samples=500, seed=42)
>>> ds1.data.equals(ds2.data)
True
>>> ds1.ground_truth.edges() == ds2.ground_truth.edges()
True
```

**Subsampling a static dataset**

```python
>>> ds = load_dataset("sachs_continuous", n_samples=100, seed=42)
>>> ds.data.shape
(100, 11)
```

**Filtering simulated datasets**

```python
>>> from pgmpy.datasets import list_datasets
>>> "linear_gaussian_scm" in list_datasets(is_simulated=True)
True
```

**Passing simulator arguments to a static dataset**

```python
>>> load_dataset("sachs_continuous", edge_prob=0.3)
TypeError: load_dataframe() got an unexpected keyword argument 'edge_prob'
```

Static datasets don't accept `**sim_kwargs` in their `load_dataframe()` signature, so Python raises a `TypeError`
naturally—no custom validation needed.

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
    def load_ground_truth(cls, **kwargs):
        # construct and return the causal graph
        ...
        return dag

    @classmethod
    def load_dataframe(cls, n_samples=500, seed=None, my_param=0.5, **kwargs):
        # generate data
        ...
        return data
```

Then `load_dataset("my_custom", n_samples=1000, seed=7, my_param=0.8)` works with no changes to the framework.

---

### Rollout plan

**Phase 1 (this PR):** `_SimulationMixin` + `load_dataset(n_samples, seed, **sim_kwargs)` + `LinearGaussianSCM` + tests.

**Phase 2:** Add ANM simulator (non-linear functions + additive noise).

**Phase 3:** Add IHDP simulator (Hill 2011 DGP). This is a good test of the architecture because it requires
loading covariates from a static file and then simulating treatment/outcome on top of them—a hybrid of static and
simulated.
