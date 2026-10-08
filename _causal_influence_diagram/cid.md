# Enhancement Proposal: Causal Influence Diagrams

This is the enhancement proposal for building the Causal Influence Diagram Class for pgmpy.

## Objective

Add a `CausalInfluenceDiagram` class to pgmpy for representing causal decision problems. A causal influence diagram (CID) combines chance variables, decision variables, and utility variables in a directed acyclic graph. It makes the assumptions about causal dependencies and decision information explicit, and provides a basis for evaluating policies and selecting decisions.

The proposed class would inherit from `pgmpy.base.DAG` and use the parameterization API in `pgmpy.parameterization` for conditional distributions. 


Each node of a causal influence diagram has exactly one of the following roles:

- **Chance**: a random variable with a conditional distribution given its graph parents.
- **Decision**: a choice controlled by a decision maker. Its parents represent information available when the decision is made.
- **Utility**: a real-valued outcome, represented by a utility function of its parents. A utility node is not a random variable and has no probability distribution of its own.

The graph remains acyclic and uses the existing `DAG` cycle validation. The class should track node roles using node attributes and expose convenient role properties such as `chance_nodes`, `decision_nodes`, and `utility_nodes`. Nodes can be declared at construction or through explicit add/set methods. Existing DAG operations should remain usable.

- TO : DO should adding an undeclared node may default to `chance` for compatibility with ordinary graph construction? Or should the API require explicit roles from the outset. This default should be settled during implementation.

Arcs into a chance node describe its conditional distribution. Arcs into a decision node describe information available at decision time
Arcs into a utility node identify the variables used by its utility function. 

- evaluate using backward induction algorithm 

## Parameterization 

| Node role | Parameterization | Example |
|---|---|---|
| **Chance** | Probability distribution / CPD | $P(X \mid Pa(X))$ |
| **Decision** | Policy / decision rule | $\pi(A \mid Pa(A))$ |
| **Utility** | Utility function | $U(X,A)$ |
| **Unspecified** | None yet | `None` |

Store parameterizations by utilizing pgmpy's existing parameterization functionalities

- Chance nodes accept `TabularCPD` 
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

```python
def set_probabilities(self, node, parameterization): ...
def set_policy(self, decision, policy): ...
def set_utility(self, node, utility): ...
```

### `set_probabilities()`

Assign the conditional distribution for a chance node. Accept a `BaseParameter` whose `variable_` names `node` and whose `evidence_` agrees with the graph parents. If fitting from data is later supported directly, it should be a separate method so assignment does not ambiguously imply fitting.

### `set_policy()`

Assign the policy for a decision node. The policy's target is the decision, and its conditioning variables must be available information parents of that decision. An optional future API may expose policy optimization separately from policy assignment.

### `set_utility()`

Assign a utility function to a utility node. The callable receives parent values as a pandas DataFrame and returns a one-dimensional numeric result. A convenience form may accept a mapping from parent-state tuples to utility values for finite discrete domains.

Useful supporting methods for a first release include `add_chance_node`, `add_decision_node`, `add_utility_node`, and `get_*` accessors. They should follow existing pgmpy conventions for graph mutation and report clear errors for assigning a distribution, policy, or utility to the wrong kind of node.

## Testing Plan

- test whether the graph is not acyclic
- Test role assignment, mutation, invalid role overlaps, and copying/subgraph behavior.
- test: if utility is not empty then the node should not have children
- test: intervention support do(x)
- test: to : do


## User Journey 

```python
from pgmpy.base import CID

cid = CID(
    ebunch=[("U","X"), ("X","Y")],
    roles = {"chance": "X", "utilities"="U", "decisions"="Y"},
    )


# add method for adding nodes

cid.add_chance_node("Health")
cid.add_decision_node("Treatment")
cid.add_utility_node("PatientUtility")

cid.add_edge("Treatment", "Health")
cid.add_edge("Treatment", "PatientUtility")
cid.add_edge("Health", "PatientUtility")

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

## llm draft 
cid.set_utility(
    "PatientUtility",
    lambda values: pd.Series(
        [10 if health == "well" else 0 for health in values["Health"]]
        - 2 * (values["Treatment"] == "drug"),
        index=values.index,
    ),
)
```