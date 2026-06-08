## [Refactoring_CPD]

- Mentor: @ankurankan
- Mentee: @daehyun99

### Introduction

> Real-world Bayesian networks commonly have high-cardinality nodes. <br>
> Issue #1776 shows another user hit 64 TiB allocation on an 82-node network. <br>
> This is silently killing pgmpy adoption in production systems. [[#3203](https://github.com/pgmpy/pgmpy/issues/3203)]

- The issue cited above suggests that, for certain domains, pgmpy may see lower adoption in practice.
- This is because the current `DiscreteBayesianNetwork`(`DiscreteBN`) supports only `TabularCPD`, `NoisyORCPD`, while `FunctionalBayesianNetwork`(`FunctionalBN`) supports only `FunctionalCPD` with `pyro`.
- Given the practical need to handle diverse forms of data, adopting pgmpy in real-world settings can introduce significant constraints.
<br>

### Proposed Solution

#### 00. Basic Contract

1. `fit()` return `self`
2. `predict()` is not implemented.
3. `predict_proba()` is return distribution object of `skpro.distribution`'s format.(exclude `FunctionalCPD`)
4. Using skpro style(`dist.log_pdf(y)`, `dist.log_pmf(y)`).
5. Using sklearn style's tag system(`__pgmpy_tags__()`).
6. But, distribution object is following skpro style's tag system(`_tags = {}`)

#### [01. Implement `BaseParameter`, `Tags`, `ParameterTags`](/Refactoring_CPD/01_Implement_BaseParameter_ParameterTags.md)

#### [02. Implement `CategoricalDistribution`](/Refactoring_CPD/02_Implement_CategoricalDistribution.md)

#### [03. Implement `TabularCPD`, `LinearGaussianCPD`, `FunctionalCPD`](/Refactoring_CPD/03_Implement_CPDs.md)

#### [04. Implement `SkproAdapter`, `SklearnAdapter`](/Refactoring_CPD/04_Implement_CPDAdapter.md)

#### [05. Potential ideas](/Refactoring_CPD/05_Potential_ideas.md)

### User journeys with the solution

#### Journey 1: a built-in discrete CPD, no identity

```python
from pgmpy.parameterization import TabularCPD

cpd = TabularCPD.from_values(
    variable_card=2,
    values=[[0.6], [0.4]],
    state_names=[["easy", "hard"]],
)

cpd.sample(X)          # draw child states for parent rows X
cpd.predict_proba(X)   # class probabilities over classes_
```

The CPD is a standalone, fitted estimator. It carries no node identity, and
nothing ties it to a particular graph.

#### Journey 2: a built-in linear-Gaussian CPD

```python
from pgmpy.parameterization import LinearGaussianCPD

cpd = LinearGaussianCPD().fit(X, y)   # X: parent columns, y: continuous child
dist = cpd.predict_proba(X)           # an skpro Normal distribution
cpd.sample(X)

# or specify parameters directly; parent_order labels the beta_[1:] positions
cpd = LinearGaussianCPD.from_values(
    beta=[0.2, -2.0, 3.0], std=1.0, parent_order=["X1", "X2"]
)
```

#### Journey 3: a scikit-learn classifier as a CPD

```python
from sklearn.linear_model import LogisticRegression
from pgmpy.parameterization.adapter import SklearnAdapter

cpd = SklearnAdapter(LogisticRegression())
cpd.fit(X, y)

samples = cpd.sample(X)          # derived from predict_proba
dist = cpd.predict_proba(X)
dist.log_pdf(y)
```

#### Journey 4: Hybrid BN
```py
from pgmpy.graph import DAG
from skpro.regression.gam import GAMRegressor
from pgmpy.parameterization.adapter import SkproAdapter

dag = DAG([("A", "C"), ("B", "C")])

cpd1 = TabularCPD().from_values()
cpd2 = LinearGaussianCPD().from_values()
cpd3 = SkproAdapter(GAMRegressor)

dag.add_cpd("A", cpd1)
dag.add_cpd("B", cpd2)
dag.add_cpd("C", cpd3)

```

#### Journey 5: Regression
```py
from sklearn.linear_model import LogisticRegression
from pgmpy.parameterization.adapter import SklearnAdapter

cpd1 = SklearnAdapter(LogisticRegression())

cpd1.fit(X, y)
dist = cpd1.predict_proba(X) # dist == skpro.distribution.Normal
res = dist.log_pdf(y) # res == 2D np.ndarray

```

#### Journey 6: Classifier
```py
from sklearn.ensemble import RandomForestClassifier
from pgmpy.parameterization.adapter import SklearnAdapter

cpd1 = SklearnAdapter(RandomForestClassifier())

cpd1.fit(X, y)
dist = cpd1.predict_proba(X) # dist == pgmpy.distribution.CategoricalDistribution
res = dist.log_pmf(y) # res == 2D np.ndarray

```
