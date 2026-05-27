## Extending `CPD` for Flexible Parameter Learning

Contributors: @daehyun99

##### Related Paper Link
- [[1]Bayesian networks with a logistic regression model for the conditional probabilities](https://www.sciencedirect.com/science/article/pii/S0888613X08000121?via%3Dihub)
- [[2]API design for machine learning software: experiences from the scikit-learn project](https://arxiv.org/abs/1309.0238)

### summary

![12](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/12.excalidraw.png)

Starting with the conclusion:
* Implement `_FactorMixin` and improve internal compatibility between Factor and CPD by storing `factor` and `cpd` information in `self._factors`.
* Have `BaseFactor` inherit from `skbase.BaseEstimator` to improve compatibility with skpro models.
* Refactoring `TabularCPD`, `LineargaussianCPD`, `FunctionalCPD`
* Implement `HybridEstimator` for fitting with sklearn, skpro model

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

1. Consider how to store factors, CPDs, and skpro models. (Implement `_FactorMixin`) <br>
2. Externally, consider compatibility with skpro models.<br>
3. Consider the parameter learning way. (Implement `HybridEstimator`)

##### Alternative Solutions && Additional Solutions

- `dag.paramters.add(variable, cpd)` (`accessor` pattern, `component` pattern)

#### 1. Consider how to store Factors, CPDs, and skpro models. (Implement `_FactorMixin`)
```
factor <-> CPD <-> skpro, sklearn model
```

* Initially, I considered having each node store distribution information, similar to how a node has its role as an attribute. This seemed appropriate for Bayesian networks.
* However, models such as `ClusterGraph`, `JunctionTree`, and `FactorGraph` use Factors instead of CPDs.
* A Factor is a concept that includes CPDs and represents relationships among multiple nodes. Therefore, since a single Factor can have multiple variables, I thought it would not be conceptually appropriate for a single node to “have” a Factor.
* Therefore, I propose creating a `_FactorMixin` class.

```python
class _FactorMixin:
    """ Wrapper class """
    self._factors = dict(frozenset: FactorObject) # key: frozenset(variables), value: FactorObject

    def _get_factors():
        ...

    def _get_factor(node: Hashable):
        return self.factor[node]

    def _add_factors(factors: Object):
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

#### 2. Externally, compatibility with skpro models should be considered.

* `BaseFactor` should be inherit from `sklearn.BaseEstimator`.

![10](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/99_Images/10.excalidraw.png)


```python
# pgmpy/factor/
class _BaseFactor(skbase.BaseEstimator):
    def __init__():
        self.factor_ = None
        
    def fit():
        self._fit()

    def predict_proba():
        self._predict_proba()   

class TabularCPD, LinearGaussianCPD, FunctionalCPD(_BaseFactor): # TabularCPD can inherit from DiscreteFactor later.
    _tags = {...}
    def __init__(self):
        ...

    def from_values(values, evidence, ..)
        ...
        
    def fit(self, X)
        ...
        
    def predict_proba(self, X):
        ...
    
    def predict_log_proba(self, X)
        ...
    
    def sample(self, X, n_samples=None):
        ...

    def get_tag(self, name, default=None):   
        ...

```

#### 3. Consider the parameter learning way. (Implement `HybridEstimator`)
* Previously, I considered storing `estimator` information together as node attributes.
* However, pgmpy follows a strategy-pattern-oriented approach, and `estimator` information is held by dedicated parameter learning estimators such as `DiscreteMLE`.
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

#### `_FactorMixin`
| Method | Input | Return |
| - | - | - |
| `_get_factor()` | `variable` | `factor` or `CPD`'s info |
| `_add_factor()` | `variable`<br> `factor` | - |
| `_remove_factor()` | `variable` | - |
| `_get_factors()` | - | All `factor` or `CPD`'s info |
| `_add_factors()` | `list[variable, factor]` | - |
| `_remove_factors()` | `list[variable, factor]` | - |

#### `_BaseFactor`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame` | `y: pd.DataFrame` |
| `predict_proba()` | `X: pd.DataFrame` | `pgmpy/Distribution` |

#### `TabularCPD`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame` | `y: pd.DataFrame` |
| `predict_proba()` | `X: pd.DataFrame` | `pgmpy/Distribution` |

#### `LinearGaussianCPD`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame` | `y: pd.DataFrame` |
| `predict_proba()` | `X: pd.DataFrame` | `pgmpy/Distribution` |

#### `FunctionalCPD`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame` | `y: pd.DataFrame` |
| `predict_proba()` | `X: pd.DataFrame` | `pgmpy/Distribution` |

#### `HybridEstimator`
| Method | Input | Return |
| - | - | - |
| `__init__()` | `config` | - |
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

# Parameter Learning
model = model(est.causal_graph_)

model.add_cpd(variable ="CVP", cpd=TabularCPD())
model.add_cpd(variable ="HYPOVOLEMIA", cpd=BayesianLinearRegressor())
model.add_cpd(variable ="LVFAILURE", cpd=TabularCPD())

# Parameter Learning
est_config = {
    "CVP": "DiscreteMLE",
    "HYPOVOLEMIA": "auto",
    "LVFAILURE": "DiscreteMLE"
}
est2 = HybridEstimator()
est2.fit(model, alarm_samples, est_config)

# Inference
infer = Inference(model)
infer.query()

```
