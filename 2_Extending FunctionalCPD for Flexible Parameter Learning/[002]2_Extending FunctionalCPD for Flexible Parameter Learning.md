## Extending FunctionalCPD for Flexible Parameter Learning

Contributors: @daehyun99

### summary

![9-3](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/9-3.excalidraw.png)

Starting with the conclusion:

* Implement `_StateNamesMixin` and store the `state_names` information of the data in `self._state_names`.
* Refactor `_GraphRolesMixin` and improve lookup performance by storing node `role` information in `self._role_nodes` and `self._node_roles`.
* Implement `_FactorMixin` and improve internal compatibility between Factor and CPD by storing `factor` and `cpd` information in `self._factors`.
* Have `BaseFactor` inherit from `sklearn.BaseEstimator` to improve compatibility with skpro models.
* Develop a view method (`get_node()`) that allows users to inspect the data stored in `self._state_names`, `self._factors`, and `self._role_nodes` in an integrated way.

### Introduction

- In my view, pgmpy currently appears to place more emphasis on structure learning—especially `Causal Discovery`, which is closer to academic interests—than on practical `Parameter Learning`.
<br>

> Real-world Bayesian networks commonly have high-cardinality nodes. <br>
> Issue #1776 shows another user hit 64 TiB allocation on an 82-node network. <br>
> This is silently killing pgmpy adoption in production systems. [[#3203](https://github.com/pgmpy/pgmpy/issues/3203)]

- The issue cited above suggests that, for certain domains, pgmpy may see lower adoption in practice.
- This is because the current `DiscreteBayesianNetwork`(`DiscreteBN`) supports only `TabularCPD`, while `FunctionalBayesianNetwork`(`FunctionalBN`) supports only `FunctionalCPD`.
- Given the practical need to handle diverse forms of data, adopting pgmpy in real-world settings can introduce significant constraints.
<br>

- In addition, pgmpy is considering support for multiple types of CPDs within a single Bayesian network. [[#2343](https://github.com/pgmpy/pgmpy/issues/2343)], [[#2344](https://github.com/pgmpy/pgmpy/issues/2344)]
<br>

- The goal of this proposal is to expand pgmpy’s internal and external compatibility so that it can be adopted as a strong choice in both academic and practical settings, while improving maintainability and development productivity.

### Proposed Solution

#### 0. Summary of Previous Discussion and Agenda

* In the previous discussion we discussed moving toward developing a unified Bayesian network, instead of extending compatibility with skpro through `FunctionBayesianNetwork`, `FunctionCPD`, and the `Adapter` pattern. [[#3366]](https://github.com/pgmpy/pgmpy/pull/3366), [[#3260]](https://github.com/pgmpy/pgmpy/pull/3260)
* We also discussed the need for `get_edge()` and `get_node()` to provide information to users, and considered storing each node’s information as NetworkX-style attributes.

```python
@dataclass
class NodeObject:
    node: Hashable
    roles: Any = None
    local_model: Any = None  # CPD or skpro model
    estimator: Any = None  # "mle", "em", "auto"; "auto" is for skpro models
    data: Any = None  # data_info

G.add_node(
    "A",
    node_info=NodeObject(
        node="A",
        roles="latent",
        local_model=TabularCPD(...)
    )
)
```

* In this proposal, the agenda items that I consider necessary are as follows.

##### Proposed Solution

1. Compatibility
   1-1. Consider how to store factors, CPDs, and skpro models. (Implement `_FactorMixin`)
   1-2. CPDs and Factors require `state_name` information. (Refactor `_StateNamesMixin`)
   1-3. Externally, consider compatibility with skpro models.
2. Consider the parameter learning way. (Implement `HybridEstimator`)
3. Consider refactoring how node roles are stored. (Refactor `_GraphRolesMixin`)
4. Consider a `get_node()` method that provides information to users.

##### Alternative Solutions && Additional Solutions

5. Consider having `UndirectedGraph` inherit from `CoreGraph`.
6. Provide information on available estimators for user.
7. Consider inference way. (Implement the `HEPIS-BN` algorithm)

* The architecture that takes the above agenda items into account is as follows. The blue shapes indicate the refactoring scope proposed in this project.

![9](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/9-2.excalidraw.png)

#### 1. Compatibility
```
DAG <-> PDAG, ADMG
MAG <-> PAG
DAG <-> UndirectGraph
BayesianNetwork <-> ClusterGraph, JunctionTree, FactorGraph
factor <-> CPD
```

* The first thing I considered was compatibility. Internally, the following three types of compatibility should be considered:

1. ~~Compatibility between base graph models~~
   (This was not considered because it is outside the scope of the current project.)
2. Compatibility between `undirected-based models` and `BayesianNetworks`.
3. Compatibility between `Factor`s and `CPD`s, since `undirected-based models` use Factors while `BayesianNetworks` use CPDs.

* Externally, the following two types of compatibility should be considered:

1. Compatibility between Factors, CPDs, and skpro models.
2. ~~Compatibility with daft, lavvan, and I/O formats (BIF, NET)~~
   (This will be addressed in issue #2933. [[#2933](https://github.com/pgmpy/pgmpy/issues/2933)])

#### 1-1. Consider how to store Factors, CPDs, and skpro models. (Implement `_FactorMixin`)

* Initially, I considered having each node store distribution information, similar to how a node has its role as an attribute. This seemed appropriate for Bayesian networks.
* However, models such as `ClusterGraph`, `JunctionTree`, and `FactorGraph` use Factors instead of CPDs.
* A Factor is a concept that includes CPDs and represents relationships among multiple nodes. Therefore, since a single Factor can have multiple variables, I thought it would not be conceptually appropriate for a single node to “have” a Factor.
* Therefore, I propose creating a `_FactorMixin` class.

```python
class _FactorMixin:
    """ Wrapper class """
    self._factors = dict() # key: variables, value: FactorObject

    def _get_factors():
        ...

    def _get_factor(node: Hashable):
        return self.factor[node]

    def _add_factors(factors: Object):
        ...

class DAG(_DirectBase):
    ...

class ClustorGraph(UndirectGraph, _FactorMixin):
    def get_factors():
        return self._get_factors()
    def get_factor(variables, factor):
        return self._get_factor(node)
    def add_factors(factors: list[str, object]):
        self._add_factors(factors)

class BayesianNetwork(DAG, _FactorMixin):
    def get_cpds():
        return self._get_factors()
    def get_cpd(variable, cpd):
        return self._get_factor(node)
    def add_cpds(cpds: list[str, object]):
        self._add_factors(cpds)

```

* When a method for adding CPDs or Factors is executed in each model, the Factor, CPD, or skpro model class is stored in `self._factors`. This is the same as the previous approach of storing them in `self.cpds` or `self.factors` in each model.
* The reason `self._factors` is a `dictionary` is to preserve order and provide fast lookup.
* With this approach, users can continue to use existing APIs such as `BN.get_cpds()` and `JunctionTree.add_factors()`, while also avoiding unnecessary feature development.
* In addition, by unifying the storage method, internal compatibility between Factors and CPDs is improved.

##### UseCase and API
```python
# variables is dict's key
# factor or cpd are dict's value

# add cpd or factor
JunctionTree.add_factor(variables=("diff", "intel", "grade"), factor = DiscreteFactor())
BayesianNetwork.add_cpd(variable = "grade", cpd = TabularCPD())
BayesianNetwork.add_cpd(variable = "diff", cpd = GAM(distribution="gamma"))

# add several cpds
BayesianNetwork.add_cpds(
    ["grade", TabularCPD()],
    ["diff", GAM(distribution="gamma")]
)

```

#### 1-2. CPDs and Factors require `state_name` information. (Refactor `_StateNamesMixin`)

* In 1-1, a CPD is added as in `BayesianNetwork.add_cpd(variable="grade", cpd=TabularCPD())`.
* However, CPDs and Factors require `state_name` information for their `variables`. Taking this into account, adding `state_name` to the code above would look as follows.

```py
BayesianNetwork.add_cpd(
    variable = "diff",
    cpd = GAM(distribution="gamma"),
    data = {
        "info": "discrete",
        "state_names":{
            "diff": ["easy", "hard"],
        }
    }
)
```

* The API above increases the amount of information that users need to provide, which is `not` user-friendly.
* For large models, the amount of information users need to input becomes even greater.
* Therefore, I propose using `pgmpy/utils/tabular.py` and `pgmpy/utils/state_name.py`, and having `pgmpy/model` inherit from the refactored `_StateNamesMixin`. [[pgmpy/utils/state_name.py]](https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/utils/state_name.py), [[pgmpy/utils/tabular.py]](https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/utils/tabular.py)

```python
class _StateNamesMixin:
    """ Wrapper class """
    self._state_names = dict(variables: states) # variable : variable's states

    def get_state_names():
        ...

    def build_state_names(data):
        ...

    def set_state_names(key, value):
        ...
```

* Like `_FactorMixin`, `state_names` are stored in dictionary form in `self._state_names`.
* When the user executes the `build_state_names(data)` method, `self._state_names` is completed based on the information in the data.
* If missing data exists, the user can check the `state_name` for each variable using `get_state_names()` and manually modify it.
* The completed `state_names` information is then used for model learning and inference.

#### 1-3. Externally, compatibility with skpro models should be considered.

* In the case of skpro models, executing `predict_proba()` returns a distribution object, and the distribution object provides methods such as `sample()` and `plot()`. [[examples/03_skpro_distributions.ipynb]](https://github.com/sktime/skpro/blob/main/examples/03_skpro_distributions.ipynb)
* Therefore, `BaseFactor` should be inherit from `sklearn.BaseEstimator`.
* In addition, we can implement distribution object(`Categorical`) in the `pgmpy/distribution` path. [[skpro/distributions]](https://github.com/sktime/skpro/tree/main/skpro/distributions)

![10](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/10.excalidraw.png)


```python
# pgmpy/factor/
class BaseFactor(BaseEstimator):
    def fit():
        self._fit()

    def predict():
        self._predict()

    def predict_proba():
        self._predict_proba()   

class DiscreteFactor(BaseFactor):
    def _fit():
        ...

    def _predict():
        ...

    def _predict_proba():
        ...

class ContinousCPD(ContinousFactor):
    ...

class TabularCPD(DiscreteFactor):
    def _predict_proba():
        return Categorical()
    ...

# pgmpy/distribution/
class Categorical():
    def plot():
        ...
    def sample():
        ...

```

#### 2. Consider the parameter learning way. (Implement `HybridEstimator`)
* Previously, I considered storing `estimator` information together as node attributes.
* However, pgmpy follows a factory-pattern-oriented approach, and `estimator` information is held by dedicated parameter learning estimators such as `DiscreteMLE`.
* Therefore, my proposal is to implement a `HybridEstimator` class and specify the learning method for each `variable` through a `config`.
* `HybridEstimator` is responsible only for orchestrating learning based on the `config` information.

```python
est_config = {
    "grade": "DiscreteMLE",
    "diff": "auto", # skpro model's fitting
    "intel": ...
}

HybridEstimator()

HybridEstimator.fit(model, data, est_config)

```

#### 3. Consider refactoring how node roles are stored. (Refactor `_GraphRolesMixin`)

Algorithms such as `DoubleMLRegressor` use role lookup functionality such as `get_role()`. [[pgmpy/prediction/DoubleMLRegressor.py]](https://github.com/pgmpy/pgmpy/blob/adeb0bd766557cb82e87868acfd7e066f2aed517/pgmpy/prediction/DoubleMLRegressor.py#L233)
However, the current node roles are stored as attributes, and `get_role(role)` iterates over all nodes and returns only the nodes with the corresponding role. [[pgmpy/base/_mixin_roles.py]](https://github.com/pgmpy/pgmpy/blob/adeb0bd766557cb82e87868acfd7e066f2aed517/pgmpy/base/_mixin_roles.py#L8)
This is costly.
Finally, considering the storage approaches of `_FactorMixin` and `_StateNamesMixin` together, I believe there is a need to unify the way data is stored.

```python
class _GraphRolesMixin:
    """Mixin class for handling roles in a causal graph."""
    self._role_nodes = dict(role: nodes)
    self._node_roles = dict(node: roles)
```
Therefore, I propose storing node roles as shown above.
With this storage approach, the average time complexity of lookup is `O(1)`, which should improve performance.
It also makes it possible to look up roles by node and nodes by role.

#### 4. Consider a `get_node()` method that provides information to users.
* In the `_StateNamesMixin`, `_FactorMixin`, and `_GraphRolesMixin` described above, data is stored in `self._state_names`, `self._factors`, and `self._role_nodes`.
* A view method is needed so that users can inspect each piece of information in an integrated way.

```python
# DAG
dag.get_node("B")
{
    node: "B",
    roles: {"latents"},
}

BN.get_node("B", data=True, include_models=True)
{
    node: "B",
    parents: {"A", "C"},
    roles: {"latents"},
    distribution: TabularCPD(),
    data: {
        "info": "discrete",
        "state_names": {
            "B": ...
        }
    }
}

JunctionTree.get_node("B", data=True, include_models=True)
{
    node: "B",
    factor: {
        {"A", "B", "C"}: DiscreteFactor(),
        {"B", "D"}: DiscreteFactor()
    },
    data: {
        "info": "discrete",
        "state_names": {
            "A": ...,
            "B": ...,
            "C": ...,
            "D": ...,
        }
    }
}

```


### Alternative Solutions && Additional Solutions

#### 5. UndirectGraph가 CoreGraph 상속 (https://github.com/pgmpy/pgmpy/pull/3385)

![9](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/9.excalidraw.png)

* `UndirectGraph` inherits from `nx.Graph` and `_CoreGraph` inherits from `nx.MultiGraph`.
* The previous decision not to have `UndirectGraph` inherit from `CoreGraph` was due to the fact that `UndirectGraph` and `DirectGraph` behave very differently.
* I believe the only difference between `nx.Graph` and `nx.MultiGraph` is that `nx.MultiGraph` allows multiple edges between two nodes.
* If `UndirectGraph` inherits from `_CoreGraph`, then we can use unified methods such as `get_edges()`, `add_edge()`, `copy()`, and so on.
* In addition, since the edge representation becomes consistent, compatibility between `BayesianNetwork` <-> `ClusterGraph`, `JunctionTree`, `FactorGraph` would be improved.
- I think refactoring will be low-effort if the `weight` parameter in `UndirectGraph.add_edge()` isn't needed, as it appears to be unused.

#### 6. Provide information on available estimators for user.

![11](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/11.excalidraw.png)

* The available estimators vary depending on the form of the `data`, such as whether missing data exists.
* However, `pgmpy` does not yet provide users with information about which estimators are available depending on the form of the data.
* This is because the data type is specified in the model itself, as in `DiscreteBayesianNetwork`.
* If several forms of `BayesianNetwork` are unified into a single model, it may be useful to provide information about available estimators based on the form of the data in the `causal inference pipeline`.

```python
model = BayesianNetwork()
model = model.from_data(data) # data: hybrid data(not including missing-data)
{
    "model_candidates": {
        "hybrid",
        "functional"
    # "linear_gaussian",
    },
    "observed variables": ["A", "B"],
    "variable_types": {
        "A": "discrete",
        "B": "continuous",
    },
    "cpd_candidates": {
        "A": ["TabularCPD", "skpro.regression.gam.GAMRegressor", "FunctionalCPD"],
        "B": ["LinearGaussianCPD", "skpro.regression.gam.GAMRegressor", "FunctionalCPD"],
    },
    "algorithm_candidates": {
        "structure_learning": ["PC", "GES"],
        "parameter_learning": ["HybridEstimator"],
        "approx_inference": ["HEPIS-BN"],
    }
}

model.set_model("hybrid") # If all data types are discrete, all CPDs are automatically set to `TabularCPD` with `model.set_model("discrete")`.

# Perform structure learning, parameter learning(setting CPD type), and inference.
```

#### 7. Consider inference way. (Implement the `HEPIS-BN` algorithm)

> I think implementing the inference algorithm as well would make the project scope too broad and therefore difficult.
> However, I have also been considering a highly general inference algorithm that could be used after parameterizing machine learning models as CPDs.

https://proceedings.mlr.press/v2/yuan07a.html

- I am still reading the paper and need more time to fully understand it, but I think the paper’s `HEPIS-BN` inference algorithm could be a potentially strong candidate.
- The algorithm is said to have high computational cost, meaning long inference time, but provides highly accurate approximate inference.
- It can perform inference for hybrid Bayesian networks with various types of data and CPDs.
- Therefore, I believe it could offer significant advantages from both a compatibility perspective and a practical application perspective, especially for hybrid data and offline environments such as biotechnology and process monitoring.

### Details of proposed solution
✨: Optional(Nice to have)

#### `_CoreGraph`
| Method | Input | Return |
| - | - | - |
| `get_node()` | `node`,<br>`data`,<br>`include_models` | `node`'s info |
| `get_nodes()` | `data`,<br>`include_models` | All `node`'s info |

#### `_FactorMixin`
| Method | Input | Return |
| `_get_factor()` | `variable` | `factor` or `CPD`'s info |
| `_add_factor()` | `variable`<br> `factor` | - |
| `_remove_factor()` | `variable` | - |
| `_get_factors()` | - | All `factor` or `CPD`'s info |
| `_add_factors()` | `list[variable, factor]` | - |
| `_remove_factors()` | `list[variable, factor]` | - |

#### `_StateNamesMixin`
| Method | Input | Return |
| `build_state_names()` | `data` | - |
| `get_state_name()` | `key` | `key`<br>`value` |
| `add_state_name()` | `key`<br>`value` | - |
| `remove_state_name()` | `key`<br>`value` | - |
| `get_state_names()` | - | All `state`'s info |
| `add_state_names()` | `list[key, value]` | - |
| `remove_state_names()` | `list[key, value]` | - |

#### `BayesianNetwork`
| Method | Input | Return |
| - | - | - |
| `__init__()` | `ebunch`,<br> `latents`,<br> `exposures`,<br> `outcomes`,<br> `roles` | - |
| `add_cpd()` | `variable` | - |
| `add_cpds(*cpds)` | `list[variable, cpd]` | - |
| `remove_cpd()` | `variable`| - |
| `remove_cpds(*cpds)` | `list[variable, cpd]` | - |
| `get_node()` | `node`,<br>`data`,<br>`include_models` | `node`'s info |
| `get_nodes()` | `data`,<br>`include_models` | All `node`'s info |
| `check_model()` | - | `bool` |
| `get_cardinality()` | `node` | `cardinality: int` |
| ✨`get_random()` | `n_nodes`,<br> `edge_prob`,<br> `n_states`,<br> `latents`,<br> `seed` | `BayesianNetwork` |
| ✨`get_random_cpds()` | `n_states`,<br> `latents`,<br> `seed` | `cpd` |
| ✨`save()` | `file_path`, `filetype` | - |
| ✨`load()` | `file_path`, `filetype` | - |

- `get_random` `get_random_cpds` [[#3296]((https://github.com/pgmpy/pgmpy/issues/3296))]
- I am thinking of supporting only the Python pickle format for `save` and `load`. (Security issues related to the Python pickle format [[13](https://docs.python.org/3/library/pickle.html)])

#### `HybridEstimator`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `model`,<br>`data`<br>`config` | - |


### User journeys with the solution

#### UseCase 1: SL -> PL

```python
from pgmpy.example_models import load_model

from pgmpy.causal_discovery import PC
from pgmpy.models import BayesianNetwork as BN
from pgmpy.parameter_estimator import HybridEstimator

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
model = model(est.causal_graph_)

model.add_cpd(variable ="CVP", cpd=TabularCPD())
model.add_cpd(variable ="HYPOVOLEMIA", cpd=BayesianLinearRegressor())
model.add_cpd(variable ="LVFAILURE", cpd=TabularCPD())

# Parameter Learning
est_config = {
    "CVP": "mle",
    "HYPOVOLEMIA": "auto",
    "LVFAILURE": "mle"
}
est2 = HybridEstimator()
est2.fit(model, alarm_samples, est_config)

# Inference
infer = Inference(model)
infer.query()

```

