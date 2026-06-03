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
| `from_values()` | `values: np.ndarray()`,<br>`state_names: dict`,<br>`is_fitted: bool`,<br>`cards: list[variable, evidence1, evidence2, ...]` | - |
| `get_tag()` | `name: str`,<br>`default: Any` | tag's info |
| ✨`to_pyro()` | - | `torch.tensor` |

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
| `from_values()` | `values: np.ndarray(std, betae1, betae2, ...)`,<br>`state_names: list[str]`,<br>`is_fitted: bool` | - |
| `get_tag()` | `name: str`,<br>`default: Any` | tag's info |
| ✨`to_pyro()` | - | `torch.tensor` |

- `Attributes`:
    - `_tags: dict`
    - `_is_fitted: bool`

#### `FunctionalCPD(_BaseParameter, RegressorMixin)`
| Method | Input | Return |
| - | - | - |
| `__init__()` | - | - |
| `fit()` | `X: pd.DataFrame`, <br>`y: pd.DataFrame` | - |
| `predict()` | `X: pd.DataFrame` | `y: np.ndarray` |
| `predict_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `predict_log_proba()` | `X: pd.DataFrame` | `y: list[np.ndarray]` |
| `sample()` | `X: pd.DataFrame`,<br>`n_samples: int`| `y: list[np.ndarray]` |
| `from_values()` | `values: np.ndarray`,<br>`state_names: dict`,<br>`is_fitted: bool`,<br>`fn`,<br>`dist: skpro.dist`,<br>`noise: skpro.dist` | - |
| `get_tag()` | `name: str`,<br>`default: Any` | tag's info |
| ✨`to_pyro()` | - | `torch.tensor` |

- `Attributes`:
    - `_tags: dict`
    - `_is_fitted: bool`


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

#### UseCase 2: Prototype of fitted FunctionalCPD

```py
# cpd_A -> cpd_C
# cpd_B -> cpd_C

data.columns
# Index(['A', 'B', 'C'], dtype='object')
# A: discrete
# B: continous
# C: continous

cpd_A = TabularCPD()
cpd_A.from_values(
    values=[[0.2], [0.8]],
    state_names={
        "A": ["False", "True"],
    },
    is_fitted=True,
)

cpd_B = LinearGaussianCPD()
cpd_B.from_values(
    values=np.array([0.1, 0.2]),
    state_names=["std", "B"],
    is_fitted=True,
)

# CLG(Conditional Linear Gaussian)
from functools import partial

def custom_fn_for_C(X, values, dist, noise):
    if X["A"] == False:
        param_values = values[0]
    elif X["A"] == True:
        param_values = values[1]

    cpd = LinearGaussianCPD()
    cpd.from_values(
        values=param_values,
        state_names=["std", "B"],
        is_fitted=True,
    )
    mu = cpd.predict(X["B"])
    return dist(mu) + noise

cpd_C = FunctionalCPD()
cpd_C.from_values(    
    values = np.array([
        # raw_std, beta0, beta_X
        [0.0,     1.0,   2.0],   # A = False
        [0.5,     3.0,  -1.0],   # A = True
    ]),
    state_names={
        "A": ["False", "True"],
        "B": None,
        "C": None,
    },
    is_fitted=True,
    fn = partial(
        custom_fn_for_C,
        dist=skpro.distribution.Normal,
        noise=skpro.distribution.Normal(0, 1),
    )
)

bn = BayesianNetwork([("A", "C"), ("B", "C")])
bn.add_cpd("A", cpd_A)
bn.add_cpd("B", cpd_B)
bn.add_cpd("C", cpd_C)

infer = Inference()
infer.query()
```

#### UseCase 3: Prototype of `FunctionalCPD`'s fitting logic.
```py
# Define custom function
def custom_fn(X, model1, model2, values, dist, noise):
    # custom_GAM = Beta + 0.5 * skpro.MDN() + 0.6 * skpro.NGBoostRegressor()
    Beta = 0.2
    mdn = model1
    ngb = model2

    mu1 = mdn.predict(X["A"])
    mu2 = ngb.predict(X["B"])
    mu = Beta + 0.5 * mu1 + 0.6 * mu2
    return dist(mu) + noise

cpd = FunctionalCPD()

cpd.from_values(    
    is_fitted=False,
    fn = partial(
        custom_fn,
        model1=skpro.MDN(),
        model2=skpro.NGBoostRegressor(),
        dist=skpro.distribution.Normal,
        noise=skpro.distribution.Normal(0, 1),
    )
)

model.add_cpd(node ="LVFAILURE", cpd=cpd)

# Define custom function's fitting logic
def custom_fn_fitting(X):
    model1 = skpro.MDN()
    model2 = skpro.NGBoostRegressor()
    
    pred1 = np.zeros(len(y))
    pred2 = np.zeros(len(y))
    
    for _ in range(10):
        target1 = (y - beta - weight2 * pred2) / weight1
        model1.fit(X["A"], target1)
        pred1 = model1.predict(X["A"])
        
        target2 = (y - beta - weight1 * pred1) / weight2
        model2.fit(X["B"], target2)
        pred2 = model2.predict(X["B"])
        
    fitted_fn = partial(
        custom_fn,
        model1=model1,
        model2=model2,
        dist=skpro.distribution.Normal,
        noise=skpro.distribution.Normal(0, 1)
    )
    
    fitted_cpd = FunctionalCPD(
        variable="LVFAILURE",
        parents=list(X.columns)
    )
    fitted_cpd.from_values(
        is_fitted=True,
        fn=fitted_fn
    )
    
    return fitted_cpd


est_config = {
    "CVP": "DiscreteMLE",
    "HYPOVOLEMIA": "auto",
    "LVFAILURE": custom_fn_fitting,
}
est2 = HybridEstimator()
est2.fit(model, X, y, est_config)

```

### Requirement and Contract
- `skpro`'s Regression model can not be root node's distribution.
    - `skpro` is only support supervised learning. 
    - Supervised learning's data format is labeling data(X -> y).
    - So, Can not express `P(A)`

### Potentional Next Steps
- Implement `LikelihoodWeighting` for Inference with Hybrid data.
- Merge several BayesianNetwork(`FunctionalBN`, `DiscreteBN`, ...) and Implement single `BayesianNetwork`
- Implement SCM(Structure Causal Model) or GCM(Graphical Causal Model) and refactor BN(Bayesian Network) to can not calculate intervene.
- Implement `QueryResult` dataclass
- Implement `_StateNamesMixin` and store the `state_names` information of the data in `self._state_names`. [Ref V2's Chapter 1-2](/2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning/[002]2_Extending%20FunctionalCPD%20for%20Flexible%20Parameter%20Learning.md)
