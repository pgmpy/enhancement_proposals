## Extending `CPD` for Flexible Parameter Learning

Contributors: @daehyun99

##### Related Paper Link
- [[1]Bayesian networks with a logistic regression model for the conditional probabilities](https://www.sciencedirect.com/science/article/pii/S0888613X08000121?via%3Dihub)
- [[2]API design for machine learning software: experiences from the scikit-learn project](https://arxiv.org/abs/1309.0238)

### summary

Starting with the conclusion and plan:

1. Implement `_ParameterMixin` and improve internal compatibility between Factor and CPD by storing `factor` and `cpd` information in `self._parameters`.
2. Implement `_BaseParameter` inherit from `skbase.BaseEstimator` to improve compatibility with skpro models.
3. Implement `paramter/TabularCPD`, `paramter/LineargaussianCPD`, `paramter/FunctionalCPD`
4. Add warning message in `factors/TabularCPD`, `factors/LineargaussianCPD`, `factors/FunctionalCPD`
5. Implement `HybridEstimator` for fitting with skpro model.

### Introduction

> Real-world Bayesian networks commonly have high-cardinality nodes. <br>
> Issue #1776 shows another user hit 64 TiB allocation on an 82-node network. <br>
> This is silently killing pgmpy adoption in production systems. [[#3203](https://github.com/pgmpy/pgmpy/issues/3203)]

- The issue cited above suggests that, for certain domains, pgmpy may see lower adoption in practice.
- This is because the current `DiscreteBayesianNetwork`(`DiscreteBN`) supports only `TabularCPD`, `NoisyORCPD`, while `FunctionalBayesianNetwork`(`FunctionalBN`) supports only `FunctionalCPD` with `pyro`.
- Given the practical need to handle diverse forms of data, adopting pgmpy in real-world settings can introduce significant constraints.
<br>

### Proposed Solution

#### 0. Summary of Previous Discussion and Agenda

- Ref: [Version 2](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/[002]2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning.md)

##### Proposed Solution

1. Consider how to store Factors, CPDs, and skpro models. (Implement `_ParameterMixin`) <br>
2. Externally, consider compatibility with skpro models.<br>
3. Internally, consider compatibility with pyro.<br>
4. Consider the parameter learning way. (Implement `HybridEstimator`)

##### Not include things in this project.
- Supproting `sklearn`'s model: This proposal is focusing on skpro.
- Refactoring Factor class: This proposal is foufocusing on cpd.
- Considering the detailed way of inference, intervene, counterfactual : We can consider later with detailed cpd contract.

##### Alternative Solutions && Additional Solutions

- `dag.paramters.add(variable, cpd)` (`accessor` pattern, `component` pattern)

#### 1. Consider how to store Factors, CPDs, and skpro models. (Implement `_ParameterMixin`)

![12](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/12.excalidraw.png)

```
factor <-> CPD <-> skpro model
```

* Initially, I considered having each node store distribution information, similar to how a node has its role as an attribute. This seemed appropriate for Bayesian networks.
* However, models such as `ClusterGraph`, `JunctionTree`, and `FactorGraph` use Factors instead of CPDs.
* A Factor is a concept that includes CPDs and represents relationships among multiple nodes. Therefore, since a single Factor can have multiple variables, I thought it would not be conceptually appropriate for a single node to “have” a Factor.
* Therefore, I propose creating a `_ParameterMixin` class.

```python
class _ParameterMixin:
    """ Store class of CPD, Factor """
    self._parameters = dict(frozenset: FactorObject) # key: frozenset(variables), value: FactorObject

    def _get_parameters():
        ...

    def _get_parameter(node: Hashable):
        ...

    def _add_parameter(node, parameter):
        ...

class _FactorMixin(_ParameterMixin):
    def get_factors():
        """Docs"""
        return self._get_parameters()

    def get_factor(node):
        """Docs"""
        return self._get_parameter(node)

    def add_factor(node, factor):
        """Docs"""
        self._add_parameters(node, factor)

class _CPDMixin(DAG, _ParameterMixin):
    def get_cpds():
        """Docs"""
        return self._get_parameters()

    def get_cpd(node):
        """Docs"""
        return self._get_parameter(node)

    def add_cpd(node, cpd):
        """Docs"""
        self._add_parameters(node, cpd)

class ClustorGraph(UndirectGraph, _FactorMixin):
    ...

class BayesianNetwork(DAG, _CPDMixin):
    ...

```

* When a method for adding CPDs or Factors is executed in each model, the Factor, CPD, or skpro model class is stored in `self._parameters`. This is the same as the previous approach of storing them in `self.cpds` or `self.factors` in each model.
* The reason `self._parameters` is a `dictionary` is to preserve order and provide fast lookup.
* With this approach, users can continue to use existing APIs such as `BN.get_cpds()` and `JunctionTree.add_factors()`, while also avoiding unnecessary feature development.
* In addition, by unifying the storage method, internal compatibility between Factors and CPDs is improved.

#### 2. Externally, compatibility with skpro models should be considered.

- `_BaseParameter` should be inherit from `skbase.BaseEstimator`.
- Discrete CPD will be inherit from `sklearn.ClassiferMixin`
- Continuous CPD will be inherit from `sklearn.RegressorMixin`

![10](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/10.excalidraw.png)

#### 3. Internally, consider compatibility with pyro.

> We might need to temporarily remove it and come up with a way to allow users to specify pyro models. [Ref from HackMD Note]

I think this approach would be preferable.<br>
If we use `FunctionalCPD` in a Hybrid BayesianNetwork, there should be no major difficulty in parameter_learning, even with a `pyro`-based implementation.<br>
However, considering compatibility and maintainability with the `inference`, `intervene`, and `counterfactual` algorithms we plan to develop later, a `pyro`-style syntax could reduce maintainability.<br>

I think `FunctionalCPD` should not be based on `pyro`. Instead of should be based on `skpro.distribution`.<br>
Instead, to support the future use of `pyro`’s `SVI` and `MCMC`, I suggest Implementing `SVIEstimator` and `MCMCEstimator`, and adding a `to_pyro()` method to each CPD class(`TabularCPD`, `LinearGaussianCPD`, `FunctionalCPD`).<br>

Also, I believe that we can expand `FunctionalCPD` to `ANM`, `PNL` when based on `skpro.distribution` easily.<br>

#### 4. Consider the parameter learning way. (Implement `HybridEstimator`)
* Previously, I considered storing `estimator` information together as node attributes.
* However, `estimator` information is held by dedicated parameter learning estimators such as `DiscreteMLE`.
* Therefore, my proposal is to implement a `HybridEstimator` class and specify the learning method for each `variable` through a `config`.
* `HybridEstimator` is responsible only for orchestrating learning based on the `config` information.

```python
est_config = {
    "grade": "DiscreteMLE",
    "diff": "auto", # skpro model's fitting
    "intel": ...
}

est = HybridEstimator()

est.fit(model, data, est_config)

```

### Details of proposed solution
✨: Optional(Nice to have)

#### `_ParameterMixin`
| Method | Input | Return |
| - | - | - |
| `_get_parameter()` | `node` | `factor` or `CPD`'s info |
| `_get_parameters()` | - | All `parameter` or `CPD`'s info |
| `_add_parameter()` | `node`,<br> `parameter` | - |
| `_add_parameters()` | `list[node, parameter]` | - |
| `_remove_parameter()` | `node` | - |
| `_remove_parameters()` | `list[node]` | - |

- `Attributes`:
    - `_parameter: dict(frozenset: FactorObject)`

#### `_BaseParameter(skbase.BaseEstimator)`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame`, <br>`y: pd.DataFrame` | - |
| `predict()` | `X: pd.DataFrame` | `y: np.ndarray` |
| `predict_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `predict_log_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `sample()` | `X: pd.DataFrame`,<br>`n_samples: int`| `y: list[np.ndarray]` |
| `from_values()` | `is_fitted: bool` | - |
| `get_tag()` | `name: str`,<br>`default: Any` | tag's info |

- `Attributes`:
    - `_tags: dict`
    - `_is_fitted: bool`

#### `TabularCPD(_BaseParameter, ClassifierMixin)`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame`, <br>`y: pd.DataFrame` | - |
| `predict()` | `X: pd.DataFrame` | `y: np.ndarray` |
| `predict_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `predict_log_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `sample()` | `X: pd.DataFrame`,<br>`n_samples: int`| `y: list[np.ndarray]` |
| `from_values()` | `cards: list[int]`,<br>`values: np.ndarray`,<br>`state_names: dict` | - |
| `get_tag()` | `name: str`,<br>`default: Any` | tag's info |

- `Attributes`:
    - `_tags: dict`
    - `_is_fitted: bool`

#### `LinearGaussianCPD(_BaseParameter, RegressorMixin)`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame`, <br>`y: pd.DataFrame` | - |
| `predict()` | `X: pd.DataFrame` | `y: np.ndarray` |
| `predict_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `predict_log_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `sample()` | `X: pd.DataFrame`,<br>`n_samples: int`| `y: list[np.ndarray]` |
| `from_values()` | `beta: list[float]`,<br>`std: float`,<br>`state_names: list[str]` | - |
| `get_tag()` | `name: str`,<br>`default: Any` | tag's info |

#### `FunctionalCPD(_BaseParameter, RegressorMixin)`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame`, <br>`y: pd.DataFrame` | - |
| `predict()` | `X: pd.DataFrame` | `y: np.ndarray` |
| `predict_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `predict_log_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `sample()` | `X: pd.DataFrame`,<br>`n_samples: int`| `y: list[np.ndarray]` |
| `from_values()` | `fn` | - |
| `get_tag()` | `name: str`,<br>`default: Any` | tag's info |

#### `HybridEstimator`
| Method | Input | Return |
| - | - | - |
| `__init__()` | `config` | - |
| `fit()` | `model`,<br>`X: pd.DataFrame`,<br>`y: pd.DataFrame`,<br>`config: dict` | - |

- `Attributes`:
    - `_tags: dict`
    - `_is_fitted: bool`

### User journeys with the solution

#### UseCase 1: Hybrid Parameter Learning

```python
from pgmpy.models import BayesianNetwork as BN
from pgmpy.parameter_estimator import HybridEstimator

from skpro.regression.bayesian import BayesianLinearRegressor

# Parameter Learning
model = model(est.causal_graph_)

model.add_cpd(node ="CVP", cpd=TabularCPD())
model.add_cpd(node ="HYPOVOLEMIA", cpd=CPDAdapter(BayesianLinearRegressor()))
model.add_cpd(node ="LVFAILURE", cpd=TabularCPD())

# Parameter Learning
est_config = {
    "CVP": "DiscreteMLE",
    "HYPOVOLEMIA": "auto",
    "LVFAILURE": "DiscreteMLE"
}
est2 = HybridEstimator()
est2.fit(model, X, y, est_config)

# Inference
infer = Inference(model)
infer.query()

```

### Potentional Next Steps
- Implement `LikelihoodWeighting` for Inference with Hybrid data.
- Merge several BayesianNetwork(`FunctionalBN`, `DiscreteBN`, ...) and Implement single `BayesianNetwork`
- Implement SCM(Structure Causal Model) or GCM(Graphical Causal Model) and refactor BN(Bayesian Network) to can not calculate intervene.
- Implement `QueryResult` dataclass
