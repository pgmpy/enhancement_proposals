## Credal Network Support

Contributors: Zach-schwark

### Introduction

<!--[Describe the problem statement and the motivation]-->

PGMPY and other probabilistic graphical model libraries do not support credal networks. There are very few libraries that support Credal Networks, namely pyagrum and the LCN library by IBM. However these libaries are very limited in their functionality, and do not give you the granular control that PGMpy does.

Additionally, there is very little focus on combining causality with credal sets and credal networks. This could be an interesting avenue to explore where there is more uncertainty in the problem you are researching.


### Proposed Solution

<!--[Describe the solution proposed]-->

Credal Networks are a generalization of Bayesian Networks, where the parameters at each node are credal sets ( ie imprecise probabilities). This is useful where there is inherent uncertainty in the problem you are dealing with. The goal is to expose the uncatainty through imprecise probabilities instead of only quantifying the uncertainty through a single point probability. This will also contribute towards explainable AI and Causality, as users can explain why there is uncertainty and what factors are causing the uncertainty.


The proposed solution would be to create customizable user friendly credal network support for PGMPY that supports granular control.

The solution will be mainly implemented through a parameterization class, parameter estimattion class, and an inference class.

For parameterization, I will need to add a new Conditional Probability Distribution type for Credal Probability Distributions. Initially we will only focus on discrete credal CDPs.

For parameter estimation, I plan on using the Imprecise Dirichlet Model (IDM) [Walley, 1996](https://doi.org/10.1111/j.2517-6161.1996.tb02065.x). However, I plan on making this implementation fine grainied enough to train on one data point at a time, instead of only interacting with the learning procedure though a batch learning procedure.

Inference will be implemented using the Credal Variable Elimination Algorithm [Mauá et al.](https://www.sciencedirect.com/science/article/pii/S0888613X12000904). This class will be based off of the current VariableElimination class.

Unlike point probabilty bayesian Networks, you cannot use Maximum a Posteriori (MAP) estimation for credal networks. This is because each node is parameterised by a set of probability distributions rather than a single distribution, so a decision must account for every distribution in that set. Therfore, I will implement a Decision maaking class which specifies different wassys in which decisions are made under imprecise probabilities, see the paper ["Decision making under uncertainty using imprecise probabilities" by Matthias C.M. Troffaes](https://www.sciencedirect.com/science/article/pii/S0888613X06000387).


### Alternative Solutions

<!--[Discuss potential alternative solution and compare them if any]-->
The alternative solutions would be to go with another python libary but as discussed above they are not granular enough or user friendly.

### Details of proposed solution

<!--[Adds details for the solution; prototype implementation or API design]-->
A detailed explaination can be found in the "credal_network_extension_plan.md" file in this folder.

Outline of Implementation:

- New `credal_tabular.py` under the `cpds/` directory. This class will inherit the `BaseParameter` class.
- Implementation of both `fit()` and `predict_proba()` methods that are compatible with credal sets, aswell as other methods needed from the BaseParameter class.
- For inference, a class called `DiscreteCredalParameterEstimator` will be created that will inherit `BaseParameterEstimator`.
- The IDM algorithm will be used as the parameter estimation algorithm. This will be done by creating a `discrete_credal_idm.py` estimator class, which will expose two methods:
  - 1) **`update(observation, weight=1.0)`**: a single-observation update. Given one data point, it increments the counts of the parameters that the observation touches by `weight`, and recomputes the affected credal sets. The default weight of 1 gives the standard IDM count update from Walley and the papers that follow it.
  - 2) **`fit(data)`**: the batch method. It follows the standard IDM process by iterating through the dataset and applying `update` to each row, so that at the end you are left with credal sets based purely on counts. This is the general-purpose entry point for anyone wanting to learn a credal network from data.
- A new `CredalVariableElimination` classed based on the current `VariableElimination` class.
- For decision making (ie obtaining a single prediction), a new class for these decision making methods wil; be created under the inferece directory.
- The classes inside the factors files will also need to be extended to support Credal Conditional Probability Distributions.


### User journeys with the solution

#### Real-Life Use Cases
<!--[Add examples of different use cases]-->
1) Causal inference under partial identifiability:

A structural causal model can be mapped exactly into a credal network, so interventions and counterfactuals are computed by standard credal network updating. Unidentifiable queries return bounds rather than failing.

2) Credal classification:

Using a credal network for classification tasks where data is sparse and incomplete.

3) Expert-elictited risk assessment.

Experts can specify credal sets for situations where data is scarce. This would help in assessing the risk of an event happening in circumstances where there is little and incomplete data.

There are various other use cases, however I have only listed a few.

#### User Journeys

Parameter Estimation
```python
from pgmpy.models import DiscreteCredalNetwork
from pgmpy.estimators import DiscreteCredalParameterEstimator
import pandas as pd

model = DiscreteCredalNetwork([("A","B"), ("A","C")])
data = pd.DataFrame()

# Batch learning
model.fit(data, estimator=DiscreteCredalParameterEstimator)
# or
estimator = DiscreteCredalParameterEstimator()
estimator.fit(model,data,weight=1.0)

# Single update step
single_observation = data.iloc[0]
estimator.update(single_observation,weight=1.0)
```

Inference
```python
from pgmpy.models import DiscreteCredalNetwork
from pgmpy.inference import CredalVariableElimination
from pgmpy.inference.decision_criteria import EAdmissibility

model = DiscreteCredalNetwork([("A","B"), ("A","C")])
infer = CredalVariableElimination(model)

posterior = infer.query(
    variables = ["A"],
    evidence={"B":"True", "C":"False"},
)

# Returning a single decision output as per a certain decision criterion

decision_result = infer.decision_query(
    variables = ["A"],
    evidence={"B":"True", "C":"False"},
    decision_criteria = EAdmissibility
)

# or the different decision criterion can be implemented into the CredalVariableElimination class

decision_result = infer.decision_query(
    variables = ["A"],
    evidence={"B":"True", "C":"False"},
    decision_criteria = "e-admissibility"
)

```
