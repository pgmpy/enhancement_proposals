# Enhancement Proposal: Causal Influence Diagrams

This is the enhancement proposal for building the Causal Influence Diagram Class for pgmpy.

## Objective

Add a `CausalInfluenceDiagram` class to pgmpy for representing causal decision problems. A causal influence diagram (CID) combines chance variables, decision variables, and utility variables in a directed acyclic graph. It makes the assumptions about causal dependencies and decision information explicit, and provides a basis for evaluating policies and selecting decisions.

The proposed class would inherit from `pgmpy.base.DAG` and use the parameterization API in `pgmpy.parameterization` for conditional distributions. 


Each node of a causal influence diagram has exactly one of the following roles:

- **Chance**: a random variable with a conditional distribution given its graph parents.
- **Decision**: a choice controlled by a decision maker. Its parents represent information available when the decision is made.
- **Utility**: a real-valued outcome, represented by a utility function of its parents. A utility node is not a random variable and has no probability distribution of its own.

The graph remains acyclic because it inherits the existing `DAG` cycle validation. The class should track node roles using node attributes and expose convenient role properties such as `chance_nodes`, `decision_nodes`, and `utility_nodes`. Nodes can be declared at construction or added through the inherited `add_node` method with a role attribute. Existing DAG operations should remain usable.

- Default nodes should have the `chance` role.

Arcs into a chance node describe its conditional distribution. Arcs into a decision node describe information available at decision time
Arcs into a utility node identify the variables used by its utility function. 

- Evaluate using backward induction algorithm. 

## Parameterization 

| Node role | Parameterization | Example |
|---|---|---|
| **Chance** | Probability distribution / CPD | $P(X \mid Pa(X))$ |
| **Decision** | Policy / decision rule | $\pi(A \mid Pa(A))$ |
| **Utility** | Utility function | $U(X) = f(Pa(X))$ |

Store parameterizations by utilizing pgmpy's existing parameterization functionalities

- Chance nodes should accept all the parameterization models from the `/parameterization` module.
- Decision policies accept a conditional parameterization for the decision variable given its information parents. For discrete decisions, `TabularCPD` can represent stochastic policies. A deterministic policy can be represented by a small adapter implementing the same parameterization interface.
- Utility nodes accept a callable that maps a DataFrame of parent assignments to one real utility per row. A table-backed utility can be added as a helper. Utility is kept distinct from probability parameterizations because current `pgmpy.parameterization` classes represent conditional distributions, not real-valued utility functions.

For every assigned conditional parameterization, validate that its target matches the node and its `evidence_` variables match the graph parents (order-insensitively). The implementation should distinguish “not assigned” from an assigned but unfitted parameterization. Utility functions should be checked for finite numeric output when evaluated. Whether utilities are additive across utility nodes should be explicit; the proposal recommends summing their outputs.

## Proposed architecture

```python
from pgmpy.base import DAG


class CausalInfluenceDiagram(DAG):
    """A DAG with chance, decision, and utility variables."""
```

## Proposed methods

### `set_probabilities()`

```python
def set_probabilities(parameterization): ...
```
- Assign the conditional distribution for a chance node. Accept a parameterization object from the `/parameterization` module whose `variable_` names `node` and whose `evidence_` agrees with the graph parents. If fitting from data is later supported directly, it should be a separate method so assignment does not ambiguously imply fitting.

### `set_policy()`

```python
def set_policy(policy): ...
```
Assign the policy for a decision node. A policy takes values of the parent nodes as input and gives out values of the decision node. It can be represented using a `DeterministicAdapter` or a MAP query over a parameterization object.

### `set_utility()`

```python
def set_utility(utility): ...
```
Assign a utility function to a utility node. The callable receives parent values as a pandas DataFrame and returns a one-dimensional numeric result. A convenience form may accept a mapping from parent-state tuples to utility values for finite discrete domains.

The inherited `add_node` method should accept a role attribute, defaulting to `chance`, and role accessors such as `chance_nodes`, `decision_nodes`, and `utility_nodes` should be provided. Assignment methods should report clear errors when a distribution, policy, or utility is assigned to the wrong kind of node.


### `check_model()`

- Similar to the Bayesian newtork classes for vaidating. The check_model would validate the model and the parameters.
- Check that parameterization targets and evidence match their assigned node and graph parents.
- Check that utility nodes are leaves and therefore have no outgoing edges. A utility node may have incoming edges from the variables used by its utility function.


## Testing Plan
- Test whether a defined CID can be sampled, since parameterization objects own the sampling logic.

## User Journey 

```python
from pgmpy.base import CausalInfluenceDiagram

cid = CausalInfluenceDiagram(
    ebunch=[("Treatment", "Health"), ("Treatment", "PatientUtility"),
            ("Health", "PatientUtility")],
    chance_nodes=["Health"],
    decision_nodes=["Treatment"],
    utility_nodes=["PatientUtility"],
)


# Nodes can also be added with the inherited add_node method. Nodes without a
# role attribute default to chance.
cid.add_node("Age")
cid.add_node("Risk", role="chance")

cid.set_probabilities(
    "Health",
    TabularCPD.from_values(
        variable="Health",
        values=[[0.8, 0.3], [0.2, 0.7]],
        evidence=["Treatment"],
        state_names={"Health": ["well", "ill"], "Treatment": ["control", "drug"]},
    ),
)
cid.set_policy(
    "Treatment",
    TabularCPD.from_values(
        variable="Treatment",
        values=[[0.5], [0.5]],
        state_names={"Treatment": ["control", "drug"]},
    ),
)

cid.set_utility(
    "PatientUtility",
    DeterministicAdapter.from_values(
        lambda values: pd.Series(
            [10 if health == "well" else 0 for health in values["Health"]],
            index=values.index,
        ) - 2 * (values["Treatment"] == "drug")
    ),
)
```