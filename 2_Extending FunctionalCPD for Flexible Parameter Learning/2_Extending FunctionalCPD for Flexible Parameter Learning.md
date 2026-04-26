## Extending FunctionalCPD for Flexible Parameter Learning

Contributors: @daehyun99

### Introduction

- In my view, pgmpy currently appears to place more emphasis on structure learning—especially `Causal Discovery`, which is closer to academic interests—than on practical `Parameter Learning`.
<br>

> Real-world Bayesian networks commonly have high-cardinality nodes. <br>
> Issue #1776 shows another user hit 64 TiB allocation on an 82-node network. <br>
> This is silently killing pgmpy adoption in production systems. [[1](https://github.com/pgmpy/pgmpy/issues/3203)]

- The issue cited above suggests that, for certain domains, pgmpy may see lower adoption in practice.
- This is because the current `DiscreteBayesianNetwork`(`DiscreteBN`) supports only `TabularCPD`, while `FunctionalBayesianNetwork`(`FunctionalBN`) supports only `FunctionalCPD`.
- Given the practical need to handle diverse forms of data, adopting pgmpy in real-world settings can introduce significant constraints.
<br>

- In addition, pgmpy is considering support for multiple types of CPDs within a single Bayesian network. [[2](https://github.com/pgmpy/pgmpy/issues/2343)], [[3](https://github.com/pgmpy/pgmpy/issues/2344)]
<br>

- Therefore, this proposal suggests adding support so that `FunctionalBN` can handle multiple types of CPDs, such as `TabularCPD`, `LinearGaussianCPD`, and `skpro` models.

![1](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/1.jpg)

- The current inheritance structure of pgmpy is shown in the image above.
- A previous consideration was to refactor `FunctionalBN` and `FunctionalCPD` to broaden their scope of support.
- However, `_CoreGraph` has already been merged, and a refactoring of the `DAG` class is planned. [[11](https://github.com/pgmpy/pgmpy/issues/2376)]
- Therefore, I believe that the refactoring of the `DAG` class (to inherit from `_CoreGraph`) should come first. [[4](https://github.com/pgmpy/pgmpy/issues/2385)]

### Proposed Solution

#### Create separate `Refactor_DAG` and `FunctionalBN` classes first, and then refactor them part to part later.

![3](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/3.jpg)

1. Create `Refactor_DAG` class.
    - It will be inherting from `_CoreGraph` [[10](https://github.com/pgmpy/pgmpy/blob/5239dfe1f6ab4327e165209b3b9f36c9fa0b6b15/pgmpy/base/_base.py#L10)]
    - It will be implementing minimal range of method.
2. Create `FunctionalBN` class.
3. Create `FunctionalCPD` class.
4. Create `*Adapter` class. (`_BaseCPDAdapter`, `TabularCPDAdapter`, `LinearGaussianCPDAdapter`, `SkproAdapter`)
5. Create `FunctionalEstimator` class.
6. Create `FunctionalInference` class.
7. (later) Refactoring `DAG` class part to part. (we can consider introducing `_GraphConverterMixin` in same time. [[7](https://github.com/pgmpy/pgmpy/issues/2933)])

### Alternative Solutions

#### A solution that directly refactors the `DAG`
![2](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/2.jpg)

- The `DAG` class carries a significant amount of responsibility, and many models inherit from it. [[12](https://github.com/pgmpy/pgmpy/blob/5239dfe1f6ab4327e165209b3b9f36c9fa0b6b15/pgmpy/base/DAG.py#L17)]
- Therefore, I believe that directly refactoring the `DAG` class, as shown above, would make it difficult to predict what kinds of issues might arise.
- Also, It could block other issues that are currently in progress. [[5](https://github.com/pgmpy/pgmpy/issues/2835)], [[6](https://github.com/pgmpy/pgmpy/issues/3296)]
- There is a possibility that existing classes such as `BIFReader`, `BIFWriter` may not function properly.
<br>

- While refactoring the PDAG, we found that the GES algorithm also needs to be refactored. [[9](https://github.com/daehyun99/pgmpy/pull/70)]
- We need to reduce the scope of the project without blocking or interfering with other contributors’ work.
<br>

#### How to fit, run inference, and sample in the model [[8](https://github.com/pgmpy/pgmpy/pull/3260#issuecomment-4161481486)]
```python

model.fit()

model.sample()

model.query()

```

### Details of proposed solution
✨: Optional(Nice to have)

#### `_CoreGraph`
| Method | Input | Return |
| - | - | - |
| `get_node()` | `node`,<br>`data`,<br>`include_models` | `node`'s info |
| `get_nodes()` | `data`,<br>`include_models` | All `node`'s info |

```python
# DAG
dag.get_node("B")
{
    node: "B",
    parents: {"A", "C"},
    children: {"F"},
    roles: {"latents"},
}

FBN.get_node("B", data=True, include_models=True)
{
    node: "B",
    parents: {"A", "C"},
    children: {"F"},
    roles: {"latents"},
    distribution: {
        ... # distribution instance or setting info
    },
    estimator: {
        ... # estimoatr instance
    },
    d_info: {
        ... # data info
    }
}

```

#### `Refactor_DAG`
| Method | Input | Return |
| - | - | - |
| `__init__()` | `ebunch`,<br> `latents`,<br> `exposures`,<br> `outcomes`,<br> `roles` | - |
| `add_edge()` | `u`,<br> `v`,<br> `edge_type` | - |
| `add_edges_from()` | `ebunchs` | - |
| `remove_edge()` | `u`,<br> `v`,<br> `edge_type` | - |
| `remove_edges_from()` | `ebunchs` | - |
| `is_valid_dag()` | - | `bool` |

- The `is_valid_dag` verifies that it meets the conditions of the `DAG`, such as cycle confirmation. [[14]](https://github.com/pgmpy/pgmpy/pull/2579)

#### `FunctionalBayesianNetwork`
| Method | Input | Return |
| - | - | - |
| `__init__()` | `ebunch`,<br> `latents`,<br> `exposures`,<br> `outcomes`,<br> `roles` | - |
| `add_cpd()` | `node`,<br> `distribution`,<br> `estimator` | - |
| `add_cpds(*cpds)` | `FunctionalCPD` instance | - |
| `remove_cpd()` | `node`| - |
| `remove_cpds(*cpds)` | `FunctionalCPD` instance | - |
| `get_cpd()` | `node` | `FunctionalCPD` |
| `get_cpds()` | - | `list[FunctionalCPD]` |
| `get_node()` | `node`,<br>`data`,<br>`include_models` | `node`'s info |
| `get_nodes()` | `data`,<br>`include_models` | All `node`'s info |
| `check_model()` | - | `bool` |
| `get_cardinality()` | `node` | `cardinality: int` |
| ✨`get_random()` | `n_nodes`,<br> `edge_prob`,<br> `n_states`,<br> `latents`,<br> `seed` | `FunctionalBayesianNetwork` |
| ✨`get_random_cpds()` | `n_states`,<br> `latents`,<br> `seed` | `FunctionalCPD` |
| ✨`save()` | `file_path`, `filetype` | - |
| ✨`load()` | `file_path`, `filetype` | - |

- `get_random` `get_random_cpds` [[6]((https://github.com/pgmpy/pgmpy/issues/3296))]
- I am not yet aware of a format that supports multiple types of CPDs.
- I am thinking of supporting only the Python pickle format for `save` and `load`. (Security issues related to the Python pickle format [[13](https://docs.python.org/3/library/pickle.html)])

#### `FunctionalCPD`
| Method | Input | Return |
| - | - | - |
| `__init__()` | `node(=variable)`,<br>`distribution`,<br>`estimator` | `FunctionalCPD` instance |
| `fit()` | `data: pandas.DataFrame` | `FunctionalCPD` |
| `predict_proba()` | `data: pandas.DataFrame` | `Distribution` instance |
| `sample()` | `n_samples`,<br>`random_state` | `samples: pandas.DataFrame` |
| ✨`plot()` | - | `matplotlib images` |
| `state_dict()` | - | `dict[str, Any]` |

- `plot() `:Users can visualize the form of the CPD and understand the data at a glance. [[15]](https://github.com/sktime/skpro/blob/main/examples/03_skpro_distributions.ipynb)

#### `_BaseCPDAdapter`, `TabularCPDAdapter`, `LinearGaussianCPDAdapter`, `SkproAdapter`
| Method | Input | Return |
| - | - | - |
| `fit()` | `data: pandas.DataFrame` | `self` |
| `predict_proba()` | `data: pandas.DataFrame` | `Distribution` instance |
| `sample()` | `n_samples`,<br>`random_state` | `samples`: `pandas.DataFrame` or `skpro.Distribution` |
| ✨`plot()` | - | `matplotlib images` |
| `__repr__()` | - | - |

#### `DiscreteBayesianNetwork`
| Method | Input | Return |
| - | - | - |
| ✨`to_FunctionalBN()` | - | `FunctionalBayesianNetwork` |

#### `FunctionalEstimator`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `model`,<br>`data` | - |

- Orchestration Parametor Learning with `FunctionalCPD`

#### `FunctionalInference`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `get_distribution()` | - | - |
| `query()` | `variables` | - |

- Orchestration Inference with `FunctionalCPD`

### User journeys with the solution

#### UseCase 1: Load the existing `DiscreteBN` and use it as a `FunctionalBN`.
```python
from pgmpy.example_models import load_model
from pgmpy.parameter_estimator import MaximumLikelihoodEstimator, FunctionalEstimator
from skpro.regression.bayesian import BayesianLinearRegressor

alarm_model = load_model("bnlearn/alarm") # DiscreteBN
alarm_samples = alarm_model.simulate(int(1e3))

# Parameter Learning in Discrete Data
DiscreteBN = DiscreteBayesianNetwork(ebunch=alarm_model.edges())

est1 = MaximumLikelihoodEstimator()
est1.fit(DiscreteBN, alarm_samples)

alarm_model.nodes() # or alarm_model.get_nodes()
# NodeView(('HISTORY', 'CVP', 'PCWP', 'HYPOVOLEMIA', 'LVEDVOLUME', 'LVFAILURE',  ...))

# User Customizing setting
FunctionalBN = DiscreteBN.to_FunctionBN()
for node in alarm_model.nodes():
    FunctionalBN.add_cpd(node=node, distribution="tabular", estimator=MaximumLikelihoodEstimator())
FunctionalBN.add_cpd(node="HYPOVOLEMIA", distribution="Normal", estimator=BayesianLinearRegressor())

# Parameter Learning
est2 = FunctionalEstimator()
FunctionalEstimator.fit(FunctionalBN, alarm_samples)

# Inference
infer = FunctionalInference(FunctionalBN)
infer.query()

```

#### UseCase 2: SL -> PL

![8](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/8.excalidraw.png)

```python
from pgmpy.example_models import load_model

from pgmpy.causal_discovery import PC
from pgmpy.models.functional import FunctionalBayesianNetwork as FunctionalBN
from pgmpy.parameter_estimator import MaximumLikelihoodEstimator
from pgmpy.factors.functional import FunctionalCPD

from skpro.regression.bayesian import BayesianLinearRegressor

alarm_model = load_model("bnlearn/alarm") # DiscreteBN
alarm_samples = alarm_model.simulate(int(1e3))

# Causal Discovery, Structural
est1 = PC(ci_test='chi_square', variant="stable", max_cond_vars=4, return_type='dag')
est1.fit(alarm_samples)

# Analysis and Improvement of Structural Learning Results
est.causal_graph_.to_graphviz()

est.causal_graph_.get_nodes()


# Parameter Learning
FunctionalBN = FunctionalBN(est.causal_graph_)

CVP = FunctionalCPD(node="CVP", distribution="tabular", estimator=MaximumLikelihoodEstimator())
HYPOVOLEMIA = FunctionalCPD(node="HYPOVOLEMIA", distribution="normal", estimator=BayesianLinearRegressor())
LVFAILURE = FunctionalCPD(node="LVFAILURE", distribution="linear", estimator=MaximumLikelihoodEstimator())

FunctionalBN.add_cpds(CVP, HYPOVOLEMIA, LVFAILURE)

# Parameter Learning
est2 = FunctionalEstimator()
est2.fit(FunctionalBN, alarm_samples)

# Inference
infer = FunctionalInference(FunctionalBN)
infer.query()

```

### Reference
- [[1]Issue #3203](https://github.com/pgmpy/pgmpy/issues/3203)
- [[2]Issue #2343](https://github.com/pgmpy/pgmpy/issues/2343)
- [[3]Issue #2344](https://github.com/pgmpy/pgmpy/issues/2344)
- [[4]Issue #2385](https://github.com/pgmpy/pgmpy/issues/2385)
- [[5]Issue #2835](https://github.com/pgmpy/pgmpy/issues/2835)
- [[6]Issue #3296](https://github.com/pgmpy/pgmpy/issues/3296)
- [[7]Issue #2933](https://github.com/pgmpy/pgmpy/issues/2933)
- [[8]Pull Request #3260](https://github.com/pgmpy/pgmpy/pull/3260#issuecomment-4161481486)
- [[9]Forked Repo - Pull Request #70](https://github.com/daehyun99/pgmpy/pull/70)
- [[10]Issue #](https://github.com/pgmpy/pgmpy/blob/5239dfe1f6ab4327e165209b3b9f36c9fa0b6b15/pgmpy/base/_base.py#L10)
- [[11]Issue #2376](https://github.com/pgmpy/pgmpy/issues/2376)
- [[12]](https://github.com/pgmpy/pgmpy/blob/5239dfe1f6ab4327e165209b3b9f36c9fa0b6b15/pgmpy/base/DAG.py#L17)
- [[13]Python 3.14.4 Documentation](https://docs.python.org/3/library/pickle.html)
- [[14]Issue #2579](https://github.com/pgmpy/pgmpy/pull/2579)
- [[15]skpro's examples file](https://github.com/sktime/skpro/blob/main/examples/03_skpro_distributions.ipynb)
