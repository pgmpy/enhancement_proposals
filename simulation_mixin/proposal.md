## Simulation Mixin for the Datasets Module

Contributors: @Gitanaskhan26

### Introduction

pgmpy's `datasets` module currently supports two kinds of data sources: static files fetched from HuggingFace Hub
(`_BaseDataset.load_dataframe`) and data generated from covariance matrices hosted on the Hub (`_CovarianceMixin`).
Both paths ultimately return a `pd.DataFrame` through the same `load_dataset()` entry point.

pgmpy already provides `model.simulate()` for forward-sampling from fitted probabilistic models
(e.g., `LinearGaussianBayesianNetwork.simulate()`, `DiscreteBayesianNetwork.simulate()`). This mixin targets a
different use case: benchmarking datasets defined by external data-generating processes whose parameterization doesn't
map cleanly to an existing pgmpy model class. Examples include the IHDP response surfaces (Hill 2011), the News
LDA-based generation (Johansson et al.), and post-nonlinear additive noise models. Users often need to sweep
hyperparameters, vary sample sizes, and reproduce ground-truth graphs across these literature-defined DGPs without
manually constructing pgmpy models each time. Storing a single pre-computed snapshot is limiting because:

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

1. A `_SimulationMixin` class with a mandatory `_simulate()` classmethod.
2. Explicit `n_samples` and `seed` parameters on `load_dataset()`, universal across static and simulated datasets.
3. A separate `**sim_kwargs` on `load_dataset()` for simulator-specific hyperparameters, forwarded only to
   `_SimulationMixin` datasets.

Everything else—tag lookup, `list_datasets()`, the `Dataset` dataclass—stays the same, except that `Dataset.ground_truth`
widens from `DAG | None` to `DAG | PDAG | ADMG | MAG | PAG | None` to support different causal graph types.

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

**B. Config objects instead of `**sim_kwargs`**

An alternative is typed config objects per simulator (e.g., `LinearGaussianConfig(edge_prob=0.3)`). This gives better
IDE autocomplete and type safety. The downside is that it adds one class per simulator dataset and makes the
`load_dataset()` signature depend on which config you're passing. I think `**sim_kwargs` is the right call for now—and
we can add config objects later if the kwargs get complex enough to warrant them.

**C. Separate `_SimulatedBaseDataset` class instead of a mixin**

We could introduce a new base class that inherits from `_BaseDataset` and adds simulation behavior. The problem is that
this doesn't compose well. `_CovarianceMixin` is a mixin specifically because some covariance-based datasets could
hypothetically also be simulated. The mixin pattern is already established and works.

---

### Details of proposed solution

#### 1. The `_SimulationMixin` class

The mixin follows the same structural pattern as `_CovarianceMixin` (`_base.py:172-213`), but it is fully stateless—no
class-level caching. `_simulate()` is a pure function: same inputs, same outputs (given a seed). `load_dataset()` acts
as the single orchestrator that calls `_simulate()` once and wires both outputs into the `Dataset` constructor.

The ground-truth graph returned by `_simulate()` is not restricted to `DAG`. Depending on the simulator, it could be
a `PDAG` (equivalence class), `ADMG` (latent confounders), `MAG`, or `PAG`. pgmpy's graph classes don't share a single
base class (`DAG`/`PDAG` extend `nx.DiGraph`, `ADMG` extends `MultiDiGraph`, `MAG` extends `AncestralBase` which
extends `nx.Graph`), so the return type uses a `Union`:

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
    Mixin for datasets where data is generated on-the-fly via a simulation
    process. Subclasses must implement ``_simulate()``.

    This mixin is stateless. It does not cache simulation results.
    ``load_dataset()`` is the intended entry point: it calls ``_simulate()``
    once and passes both the data and ground-truth graph directly to the
    ``Dataset`` constructor.

    If users call ``_simulate()`` directly, they are responsible for passing
    the same ``seed`` to get reproducible results.

    When using this mixin, it should be the first parent class so that its
    methods take precedence in the MRO (same convention as
    ``_CovarianceMixin``).
    """

    @classmethod
    def _simulate(
        cls,
        n_samples: int | None = None,
        seed: int | None = None,
        **sim_kwargs,
    ) -> tuple[pd.DataFrame, CausalGraph]:
        """
        Generate simulated data and the ground-truth causal graph.

        Must be implemented by each simulator dataset class.

        Parameters
        ----------
        n_samples : int or None
            Number of samples to generate.
        seed : int or None
            Seed for reproducibility.
        **sim_kwargs
            Simulator-specific hyperparameters.

        Returns
        -------
        tuple[pd.DataFrame, CausalGraph]
            (data, ground_truth_graph)
        """
        raise NotImplementedError(
            f"{cls.__name__} must implement _simulate()."
        )
```

**Why this design:**

- **`_simulate` returns a tuple.** A simulated dataset's ground truth is produced *during* generation (the graph is
  constructed, then data is sampled from it). Splitting this into two independent methods would either duplicate work or
  require hidden shared state. The tuple return makes the coupling explicit.

- **Stateless, no caching.** Caching at the class level would be incorrect—if the user calls `_simulate()` with
  different `seed` or `n_samples` values, a cached result would be stale. Instead, `load_dataset()` calls `_simulate()`
  once per invocation and passes both results into the `Dataset` constructor directly. This avoids the consistency
  problem entirely.

- **Classmethods, not instance methods.** Every method in the current dataset architecture is a classmethod
  (`_base.py:132-169`). Switching to instance methods for simulation datasets would break the uniformity that
  `load_dataset()` relies on (it calls `target_cls.load_dataframe()`, not `target_cls().load_dataframe()`). Per-call
  hyperparameters are passed as function arguments instead of stored as instance state. This is slightly less ergonomic
  than instance methods, but it's consistent.

- **Graph type is not restricted to DAG.** Different simulators produce different graph types. An ANM on a known DAG
  returns a `DAG`. A simulator that generates data from an equivalence class might return a `PDAG`. A simulator with
  latent confounders returns an `ADMG`. The `CausalGraph` type alias captures this without forcing all simulators into
  a single graph class.

- **Mixin sets `is_simulated` tag.** Each concrete simulator class should set `"is_simulated": True` in its
  `_tags` dict. I considered having the mixin override `_tags` itself, but `skbase` tag resolution across multiple
  inheritance is already well-defined—child class tags override parent tags—so it's cleaner to be explicit in each
  simulator class. This also lets `list_datasets(is_simulated=True)` work without changes.

#### 2. Changes to `load_dataset()`

Two changes: (a) add explicit `n_samples` and `seed` parameters, and (b) branch on simulation vs. static datasets.

Making `n_samples` and `seed` explicit top-level parameters (rather than burying them in `**kwargs`) keeps the API
uniform across all datasets. For static datasets, `n_samples` subsamples the loaded data with the given `seed`. For
simulated datasets, both are forwarded to `_simulate()` along with any simulator-specific `**sim_kwargs`.

```python
def load_dataset(
    name: str,
    n_samples: int | None = None,
    seed: int | None = None,
    **sim_kwargs,
) -> Dataset:
    target_cls = _resolve_dataset_class(name)  # existing lookup logic

    if issubclass(target_cls, _SimulationMixin):
        # Simulated dataset: call _simulate() directly, wire both outputs
        # into the Dataset constructor. No intermediate caching.
        data, ground_truth = target_cls._simulate(
            n_samples=n_samples, seed=seed, **sim_kwargs
        )

        # Dynamic tags are derived from the returned data, not mutated
        # on the class. Class-level _tags stay static; we compute the
        # actual n_samples/n_variables from the DataFrame shape and
        # record the sim_kwargs that were used for reproducibility.
        tags = target_cls.get_class_tags()
        tags["n_samples"] = data.shape[0]
        tags["n_variables"] = data.shape[1]
        tags["sim_params_used"] = {
            "n_samples": data.shape[0],
            "seed": seed,
            **sim_kwargs,
        }

        return Dataset(
            name=name,
            data=data,
            expert_knowledge=None,
            ground_truth=ground_truth,
            tags=tags,
        )
    else:
        # Static dataset: sim_kwargs should be empty
        if sim_kwargs:
            raise ValueError(
                f"Dataset '{name}' is not a simulation dataset and does not "
                f"accept simulator arguments. Got: {list(sim_kwargs.keys())}"
            )

        df = target_cls.load_dataframe()

        # Subsample if n_samples is specified
        if n_samples is not None:
            n_samples = min(n_samples, len(df))
            df = df.sample(n=n_samples, random_state=seed)

        return Dataset(
            name=name,
            data=df,
            expert_knowledge=target_cls.load_expert_knowledge(),
            ground_truth=target_cls.load_ground_truth(),
            tags=target_cls.get_class_tags(),
        )
```

Note: the `Dataset` dataclass's `ground_truth` field needs to widen from `DAG | None` to `CausalGraph | None` to
accommodate the different graph types that simulators can return.

#### 3. Parameter discoverability via `sim_params` tag

One issue with routing everything through `load_dataset()` is that the user has no way to discover what simulator
arguments are available without reading the `_simulate()` source code. To address this, each simulator class declares
its expected parameters in a structured `sim_params` tag:

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

This enables two things:

- **Pre-call introspection.** `list_datasets(is_simulated=True)` can surface available parameters. A user can also
  check `SomeSimulator.get_class_tag("sim_params")` to see what the simulator accepts before calling `load_dataset()`.

- **Post-call reproducibility.** After simulation, `load_dataset()` attaches the actual values used (including resolved
  defaults) to `Dataset.tags["sim_params_used"]`. The user can always inspect `ds.tags["sim_params_used"]` to see
  exactly how the data was generated.

An alternative is to use `inspect.signature()` to extract parameter names and defaults from `_simulate()` at runtime,
which would be zero-maintenance. The downside is that it doesn't give descriptions. We could use both: `inspect` for
auto-discovery, `sim_params` for descriptions when the simulator author provides them. Open to feedback on which
approach is preferred.

#### 4. Reference simulator: `LinearGaussianSCM`

For the initial validation of the mixin architecture, I'll implement a linear Gaussian SCM simulator. This is a
proof-of-concept rather than a core use case — pgmpy can already simulate from linear Gaussian models natively via
`LinearGaussianBayesianNetwork.get_random()` + `.simulate()`. The value of wrapping it in a `_SimulationMixin` class
is to validate the mixin mechanics (tag routing, `sim_kwargs` forwarding, `Dataset` construction) before building
the more complex simulators that actually need this architecture. See the prototype in
[`prototype_linear_gaussian_scm.py`](prototype_linear_gaussian_scm.py).

#### 5. File-level summary of changes

| File | Change |
|------|--------|
| `pgmpy/datasets/_base.py` | Add `_SimulationMixin` class (~25 lines, stateless). Update `load_dataset()` signature with `n_samples`, `seed`, `**sim_kwargs`. Add simulation vs. static branching. Widen `Dataset.ground_truth` type from `DAG \| None` to `CausalGraph \| None` (the `Dataset` dataclass lives in this file, line 18-34). |
| `pgmpy/datasets/__init__.py` | Export `_SimulationMixin`. |
| `pgmpy/datasets/linear_gaussian_scm.py` | **[NEW]** Reference simulator implementation (~40 lines), including `sim_params` tag. |
| `pgmpy/tests/test_datasets/` | **[NEW]** Tests for mixin routing, sim_kwargs forwarding, seed reproducibility, `sim_params_used` tag on returned Dataset, subsampling for static datasets (including `n_samples > len(df)`), error on sim_kwargs for static datasets. |

#### 6. Open questions

**Should we also update the Tubingen special case?** The `tubingen/<pair_id>` branch in `load_dataset()`
(`_base.py:251-277`) hardcodes `pair_id` handling rather than using `**sim_kwargs`. It would be more consistent to route
Tubingen through the new parameters too (e.g., `load_dataset("tubingen", pair_id=42)`), but that changes existing
user-facing behavior. I'd defer this to a separate PR to keep scope tight.

**Tag values for variable-size datasets.** Static datasets have fixed `n_variables` and `n_samples` in their class tags.
Simulated datasets don't—these depend on the kwargs. The class tags keep `None` for these fields; `load_dataset()`
computes the actual values from the returned DataFrame shape and writes them to the *returned* `Dataset.tags` dict
(not the class). This avoids mutating class-level state while still giving the user accurate metadata on the
`Dataset` object. Same approach as Tubingen (`_base.py:266-267`).

**`sim_params` tag vs. `inspect.signature()`.** The `sim_params` tag requires manual maintenance — if someone changes
`_simulate()`, they need to update the tag too. `inspect.signature()` would auto-extract parameter names and defaults
at zero maintenance cost, but can't provide descriptions. Feedback welcome on which approach (or a hybrid) is
preferred.

---

### User journeys with the solution

#### 1. Linear Gaussian SCM (proof-of-concept, DAG ground truth)

This wraps pgmpy's existing `LGBN.get_random()` + `.simulate()` in the mixin pattern. It's a proof-of-concept —
pgmpy can already do this natively — but it validates the mixin mechanics end-to-end.

```python
>>> from pgmpy.datasets import load_dataset
>>> ds = load_dataset("linear_gaussian_scm", n_samples=2000, seed=42, n_nodes=8, edge_prob=0.3)
>>> ds.data.shape
(2000, 8)
>>> type(ds.ground_truth)
<class 'pgmpy.base.DAG.DAG'>
>>> ds.tags["sim_params_used"]
{'n_samples': 2000, 'seed': 42, 'n_nodes': 8, 'edge_prob': 0.3}
```

The `_simulate()` for this class:

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
    def _simulate(cls, n_samples=1000, seed=None, n_nodes=5, edge_prob=0.3, noise_scale=1.0, **kwargs):
        model = LGBN.get_random(n_nodes=n_nodes, edge_prob=edge_prob, scale=noise_scale, seed=seed)
        data = model.simulate(n_samples=n_samples, seed=seed)
        return data, DAG(model.edges())
```

Simulator-specific kwargs here: `n_nodes`, `edge_prob`, `noise_scale`. The `n_samples` and `seed` come from
`load_dataset()`'s explicit parameters.

#### 2. Additive Noise Model (non-linear functions, DAG ground truth)

ANM generates data where each variable is a non-linear function of its parents plus independent additive noise:
`X_j = f_j(pa(X_j)) + N_j`. Causal direction is identifiable under this model when the functions and noise
distributions satisfy certain conditions (Hoyer et al., 2008).

```python
>>> ds = load_dataset("anm", n_samples=3000, seed=7, n_nodes=6, function_type="gp", noise_dist="exponential")
>>> ds.data.shape
(3000, 6)
>>> type(ds.ground_truth)
<class 'pgmpy.base.DAG.DAG'>
```

The `_simulate()` for this class takes different kwargs than Linear Gaussian:

```python
class AdditiveNoiseModel(_SimulationMixin, _BaseDataset):
    _tags = {"name": "anm", "is_simulated": True, "has_ground_truth": True, ...}

    @classmethod
    def _simulate(cls, n_samples=1000, seed=None, n_nodes=5, function_type="gp",
                  noise_dist="gaussian", edge_prob=0.3, **kwargs):
        rng = np.random.default_rng(seed)
        dag = DAG.get_random(n_nodes=n_nodes, edge_prob=edge_prob, seed=seed)
        # generate data by forward-sampling through the DAG with non-linear functions
        # and additive noise drawn from noise_dist
        ...
        return data, dag
```

This tests that the API handles simulator-specific kwargs (`function_type`, `noise_dist`) cleanly, while `n_samples`
and `seed` stay at the `load_dataset()` level.

#### 3. IHDP (semi-simulated: static covariates + synthetic treatment/outcome)

IHDP (Hill, 2011) is a hybrid dataset. Covariates come from a real RCT (the Infant Health and Development Program),
but treatment assignment and potential outcomes are simulated on top of them.

The ground-truth DAG is defined deterministically by the DGP specification:

- `X_i → T` for all 25 covariates (treatment assignment depends on covariates)
- `X_i → Y` for all 25 covariates (outcome depends on covariates)
- `T → Y` (treatment affects outcome)

No latent confounders in the standard specification — the simulation is fully observed, so the ground truth is a
`DAG`, not an `ADMG`. The simulator constructs this DAG structure once (it's fixed), then simulates `T` via logistic
assignment and `Y` using Response Surface A or B from the original paper.

This tests that `_SimulationMixin` can coexist with `_get_raw_data()` — the class uses the Hub to fetch the static
covariate file, then simulates the rest.

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

The `_simulate()` for this class:

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
    def _simulate(cls, n_samples=None, seed=None, treatment_noise=0.0,
                  outcome_fn="response_surface_A", **kwargs):
        # Load real covariates from Hub (uses _BaseDataset._get_raw_data)
        raw = cls._get_raw_data(cls.covariate_url)
        covariates = pd.read_csv(io.BytesIO(raw))

        if n_samples is not None:
            covariates = covariates.sample(n=n_samples, random_state=seed)

        # Build fixed ground-truth DAG
        covariate_names = list(covariates.columns)
        edges = [(x, "treatment") for x in covariate_names]
        edges += [(x, "y_obs") for x in covariate_names]
        edges.append(("treatment", "y_obs"))
        dag = DAG(edges)

        # Simulate treatment assignment and potential outcomes
        rng = np.random.default_rng(seed)
        ...
        return data, dag
```

Key difference from the Linear Gaussian case: `n_samples=None` has meaningful semantics (use all original subjects),
the ground-truth DAG is fixed rather than random, and the simulator mixes static data loading with on-the-fly
simulation.

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
>>> list_datasets(is_simulated=True)
['anm', 'ihdp', 'linear_gaussian_scm']
```

**Error on sim_kwargs for static datasets**

```python
>>> load_dataset("sachs_continuous", edge_prob=0.3)
ValueError: Dataset 'sachs_continuous' is not a simulation dataset and does not accept
simulator arguments. Got: ['edge_prob']
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
    def _simulate(cls, n_samples=500, seed=None, my_param=0.5, **kwargs):
        # build graph, sample data, return (df, graph)
        ...
        return data, dag
```

Then `load_dataset("my_custom", n_samples=1000, seed=7, my_param=0.8)` works with no changes to the framework.

---

### Rollout plan

**Phase 1 (this PR):** `_SimulationMixin` + `load_dataset(n_samples, seed, **sim_kwargs)` + `LinearGaussianSCM` + tests.

**Phase 2:** Add ANM simulator (non-linear functions + additive noise).

**Phase 3:** Add IHDP simulator (Hill 2011 DGP). This is a good test of the architecture because it requires
loading covariates from a static file and then simulating treatment/outcome on top of them—a hybrid of static and
simulated.
