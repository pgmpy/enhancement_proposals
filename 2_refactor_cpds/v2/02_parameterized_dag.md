## Design 2: Parameterized DAG, Registration, and Inference Dispatch

Contributors: @ankurankan, @daehyun

### Introduction

Once local CPDs become reusable conditional models, the graph layer becomes the place where identity, schema, and preparation logic must live. This is the main architectural change needed to make arbitrary sklearn and skpro estimators practical as CPDs.

Today the codebase spreads parameterized behavior across `DiscreteBayesianNetwork`, `LinearGaussianBayesianNetwork`, `FunctionalBayesianNetwork`, and `DynamicBayesianNetwork`. Each class owns its own CPD registry and custom `fit`, `simulate`, and `check_model` behavior. Inference, sampling, and read/write code then reach into CPD internals directly.

That structure works when the local model types are hard-coded and graph-specific. It does not scale to the more open boundary proposed in Design 1, where a node may be parameterized by a native pgmpy CPD, an adapted sklearn classifier, an skpro probabilistic regressor, or a structural mechanism such as ANM.

Other packages split this concern in a similar way. BN-centric libraries such as CausalNex and pomegranate let the graph own ordinary conditional models, while packages such as DoWhy GCM and pyAgrum layer structural semantics on top rather than collapsing everything into one base graph contract.

This document therefore focuses on one question: how should a parameterized `DAG` own, adapt, and consume those local models?

### Proposed Solution

Make `DAG` the owner of parameterized node registration, parent order, and variable schema, while keeping the local model contract from Design 1 independent of the graph.

The graph should expose one primary registration API:

- `dag.set_local_model(node, model, *, parents=None, schema=None, role=None)`

and two convenience aliases:

- `dag.add_cpd(...)` for ordinary conditional models
- `dag.add_mechanism(...)` for structural models

Registration should accept either native pgmpy CPDs or raw third-party estimators. If the object is not already a pgmpy CPD but satisfies the required protocol, the graph adapts it automatically.

This design keeps `DAG` a pgmpy graph container rather than turning it into an sklearn estimator. Exact discrete inference remains available through explicit factor materialization, while approximate and simulation-based methods dispatch on capabilities exposed by the registered local models.

### Alternative Solutions

| Alternative | Why not choose it |
|---|---|
| Leave CPD registries on typed BN subclasses | Preserves duplicated logic and forces every new CPD shape to be threaded through several model classes. |
| Require users to wrap and encode every external estimator manually | Makes the extensibility story brittle and defeats the main goal of accepting ordinary sklearn and skpro objects directly. |
| Make `DAG` itself a sklearn or skbase estimator | Conflicts with graph mutation, subgraphing, ancestral slicing, and existing copy semantics. |
| Keep exact inference coupled directly to `TabularCPD(DiscreteFactor)` | Prevents exact inference from being a consumer of factorized discrete conditionals and instead makes factor inheritance the universal CPD contract. |

### Details of proposed solution

#### Graph-owned parameter state

The graph owns the parameterization metadata:

```python
class DAG:
    _cpds: dict[Hashable, BaseConditionalCPD]
    _parent_order: dict[Hashable, list[Hashable]]
    _schema: dict[Hashable, VariableSchema]
```

- `_cpds` maps node to the registered local model
- `_parent_order` preserves the ordered parent list expected by positional models
- `_schema` stores state names, variable type, encoders, and other node metadata

This is what makes external estimator adaptation realistic. A bare sklearn classifier does not know pgmpy state names, parent order, or categorical encodings. The graph has to own that context.

#### Registration API

The recommended public surface is:

```python
class DAG:
    def set_local_model(self, node, model, *, parents=None, schema=None, role=None): ...
    def add_cpd(self, node, cpd, *, parents=None, schema=None): ...
    def add_mechanism(self, node, mechanism, *, parents=None, schema=None): ...
```

Registration should behave as follows:

1. validate that the declared parents match the graph structure
2. resolve or update node schema
3. store parent order for the node
4. if `model` is a pgmpy CPD, store it directly
5. else, if `model` exposes `predict_proba`, adapt it using the Design 1 adapter boundary
6. else, reject the object as not satisfying the local-model contract

`role` is advisory rather than identity-defining. For example, `role="structural"` may require that the resolved object is a `BaseStructuralCPD`.

#### Data preparation and encoding

To support "any sklearn classifier with `predict_proba`" in practice, the graph must prepare node-local design matrices. That includes:

- ordering parent columns according to `_parent_order`
- applying schema-aware categorical encodings
- preserving state name mappings for the child variable
- decoding model outputs back into user-facing labels where appropriate

Without this layer, the promise of accepting arbitrary classifiers is mostly illusory because many estimators only accept numeric features.

#### Capability-driven inference dispatch

Inference should dispatch on capabilities rather than on CPD class names.

- exact discrete inference requires `can_materialize_factor=True`
- linear-Gaussian optimizations can use `child_type="continuous"` plus a linear-Gaussian capability tag
- approximate inference can use `sample` and `log_prob` without assuming factor semantics

The critical boundary is explicit factor materialization:

```python
dag.transforms.cpd_as_factor(node)
```

This transformation can enumerate parent configurations from graph-owned schema and ask the registered CPD for its conditional distribution over the child. That keeps exact inference intact without requiring every discrete CPD to inherit `DiscreteFactor`.

#### Fit and simulate

A unified graph surface should be possible even if the internal engines remain specialized:

```python
dag.fit(data, estimator=None)
dag.simulate(...)
```

`fit` delegates node-local estimation to the registered CPDs after schema-aware preparation of `X` and `y`. `simulate` can share a common surface while still allowing different internal paths for discrete ancestral sampling, linear-Gaussian conditioning, or structural-mechanism rollout.

#### Compatibility and migration

The existing typed BN classes can become compatibility shims during 1.x:

- `DiscreteBayesianNetwork`
- `LinearGaussianBayesianNetwork`
- `FunctionalBayesianNetwork`

Their CPD-management methods should gradually forward into the graph-owned registry rather than continue to own separate storage.

`DynamicBayesianNetwork` remains a separate migration track because of time-slice node identities and time-aware CPD APIs.

Read/write code should migrate away from inspecting `cpd.variable` and `cpd.variables[1:]` directly. Serialization should iterate over graph-owned node registration and schema instead.

#### Copy semantics

Parameterized graph copying needs an explicit contract. The recommended surface is:

- `copy()` preserves structure, schema, and registered local models
- `copy_template(parameters="none"|"unfit"|"fitted")` gives explicit control when users want a structural copy without all fitted state

This is more predictable than making users guess whether parameterization survives ordinary graph operations.

### User journeys with the solution

#### User journey 1: register a raw sklearn classifier on a node

```python
from sklearn.linear_model import LogisticRegression

dag.set_local_model(
    "Y",
    LogisticRegression(),
    parents=["A", "B"],
    schema={"states": ["no", "yes"]},
)
```

The graph records parent order and schema, adapts the estimator into a conditional CPD, and takes responsibility for encoding parent columns and decoding child states.

#### User journey 2: mix native and adapted probabilistic CPDs

```python
dag.add_cpd("A", TabularCPD.from_values(...))
dag.set_local_model("B", LogisticRegression(), parents=["A"])
dag.set_local_model("C", NGBoostRegressor(), parents=["A", "B"])
```

The graph can host a mixed parameterization without forcing all nodes into one old typed-BN class.

#### User journey 3: exact inference on an adapted discrete model

```python
factor = dag.transforms.cpd_as_factor("B")
```

If the schema is finite and the adapted classifier advertises `can_materialize_factor=True`, exact discrete inference can still proceed through factor algorithms.

#### User journey 4: structural model registration without changing the graph API

```python
dag.add_mechanism(
    "X",
    ANM(regressor=RandomForestRegressor(), noise=NormalNoise()),
    parents=["Z1", "Z2"],
)
```

The graph registration surface stays the same. What changes is the capability set of the local model, which later enables counterfactual workflows from Design 3.
