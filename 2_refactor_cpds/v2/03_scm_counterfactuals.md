## Design 3: Structural Models, Interventions, and Counterfactuals

Contributors: @ankurankan, @daehyun

### Introduction

Design 1 establishes a reusable CPD boundary for ordinary conditional models. Design 2 makes the graph own identity, schema, and registration. Neither of those decisions by itself answers a separate question that many causal users care about most: how should pgmpy expose structural mechanisms such as ANM and PNL, and how should those mechanisms participate in intervention and counterfactual workflows?

This matters because an ordinary CPD and a structural mechanism do not carry the same information.

- An ordinary CPD specifies `P(X | Pa(X))`.
- A structural mechanism specifies `X = f(Pa(X), U)` together with a distribution over `U`.

The first is enough for associational inference and many intervention workflows. The second is what users usually have in mind when they ask for ANM, PNL, abducted noise, or unit-level counterfactuals.

Other packages make this split explicit. DoWhy GCM separates probabilistic, structural, and invertible structural models. bnlearn and pyAgrum do not treat fitted BN CPDs as automatically SCM-ready. ChiRho is structural-first and requires explicit exogenous-noise semantics for counterfactuals. pgmpy should follow the same direction: counterfactual support should be explicit, not silently inferred from an ordinary CPD.

### Proposed Solution

Add an optional structural layer on top of the conditional CPD boundary from Design 1.

- Keep ordinary probabilistic CPDs as the default local model type.
- Introduce first-class structural CPD classes such as `ANM` and `PNL`.
- Require explicit structural semantics for counterfactual queries.
- Separate intervention and counterfactual support by capability.

In this design, `dag.intervene(...)` may work with ordinary conditional CPDs, while `dag.counterfactual(...)` requires structural local models and abduction support on the relevant ancestors.

### Alternative Solutions

| Alternative | Why not choose it |
|---|---|
| Treat every CPD as SCM-ready by default | Semantically too strong. A BN CPD does not uniquely determine a structural-noise representation, especially in the discrete case. |
| Expose only a generic `FunctionalCPD` for causal models | Too backend-specific and does not match the vocabulary most users expect when they ask for ANM, PNL, or additive-noise models. |
| Build a separate SCM API that does not reuse the CPD boundary from Design 1 | Duplicates local-model contracts and makes it harder to mix probabilistic and structural nodes in one graph. |
| Limit the causal layer to Pyro-only machinery | Excludes common structural models that can be expressed directly with sklearn regressors and simple noise models. |

### Details of proposed solution

#### Structural abstract layers

The causal layer should reuse the local-model hierarchy from Design 1:

```python
class BaseStructuralCPD(BaseConditionalCPD):
    def noise_distribution(self): ...
    def structural_predict(self, X: pd.DataFrame, noise) -> pd.Series: ...


class BaseAbductableStructuralCPD(BaseStructuralCPD):
    def abduct(self, y: pd.Series, X: pd.DataFrame): ...
```

`BaseStructuralCPD` is enough for mechanism-aware simulation. `BaseAbductableStructuralCPD` is the stronger interface needed when the query requires recovering unit-level noise from observed evidence.

#### First-class structural model families

The first structural classes exposed to users should be the ones they already know:

- `ANM`
- `PNL`
- linear additive-noise variants derived from `LinearGaussianCPD`

Recommended constructor shapes:

```python
ANM(regressor, noise=NormalNoise())
PNL(regressor, transform, inverse_transform, noise=NormalNoise())
```

Expected semantics:

- `ANM`: `X = f(Pa(X)) + U`
- `PNL`: `X = g(f(Pa(X)) + U)` with explicit inverse transform when abduction is needed

This keeps the API aligned with the language users already use in causal modeling papers and software.

#### Noise distribution contract

Structural models should share a small noise interface:

```python
class NoiseDistribution(ABC):
    def sample(self, n, random_state=None): ...
    def log_prob(self, value): ...
    def point(self): ...
```

Representative implementations include:

- `Delta`
- `NormalNoise`
- `EmpiricalNoise`
- `TruncatedNormalNoise`

This makes structural rollout and abduction generic across model families.

#### Relation between probabilistic and structural views

A structural model should still satisfy the ordinary conditional CPD contract. In other words, ANM and PNL should be usable anywhere a `BaseConditionalCPD` is accepted.

That means a structural model must be able to produce an ordinary conditional distribution view, for example by combining its deterministic mechanism with its noise distribution. This is what allows structural models to participate in observational and interventional workflows without a separate graph abstraction.

The reverse direction is not automatic. An ordinary CPD should not silently be treated as a structural model unless the user opts into a specific structuralization convention.

#### Intervention versus counterfactual support

The graph-level causal API should distinguish between interventions and counterfactuals:

- `dag.intervene(...)` can operate on ordinary conditional CPDs through truncated factorization or mechanism replacement
- `dag.counterfactual(...)` requires structural semantics on the relevant part of the graph

Counterfactual execution should follow the standard three-step pattern:

1. abduction: infer noise values or posterior noise information from evidence
2. action: modify the intervened mechanisms
3. prediction: roll the structural system forward under the abducted noise state

If some required node only has an ordinary conditional CPD, the API should fail explicitly rather than guess hidden structural semantics.

#### Discrete structural semantics

Discrete tabular CPDs are the main place where semantic overreach is dangerous. A table for `P(X | Pa(X))` does not uniquely determine a counterfactual model.

The design should therefore avoid declaring all `TabularCPD` objects counterfactual-capable by default. If pgmpy later wants discrete SCM support, it should be opt-in and user-visible, for example:

```python
CategoricalStructuralCPD.from_conditional(
    cpd,
    noise_repr="inverse_cdf",
)
```

This keeps the added assumptions explicit.

#### Query surface

The causal accessor surface can remain small:

- `dag.intervene(...)`
- `dag.counterfactual(...)`
- optional `dag.bootstrap(...)` for uncertainty summaries

`pgmpy.identification` should remain the home of adjustment and frontdoor logic unless there is a strong reason to pull it back under a DAG accessor. The causal layer can call into identification helpers without collapsing those modules together.

### User journeys with the solution

#### User journey 1: fit an ANM and ask a counterfactual question

```python
dag.add_mechanism(
    "Y",
    ANM(regressor=RandomForestRegressor(), noise=NormalNoise()),
    parents=["X", "Z"],
)

dag.counterfactual(
    target={"Y": None},
    intervention={"X": 1},
    evidence={"X": 0, "Y": 2.4, "Z": 3},
)
```

The user works with a named structural family rather than having to assemble a generic functional CPD by hand.

#### User journey 2: use a PNL model with explicit invertibility

```python
mechanism = PNL(
    regressor=RandomForestRegressor(),
    transform=np.exp,
    inverse_transform=np.log,
    noise=NormalNoise(),
)

dag.add_mechanism("Y", mechanism, parents=["X"])
```

Because the inverse transform is explicit, the mechanism can advertise `supports_abduction=True` and participate in abduction-heavy counterfactuals.

#### User journey 3: mix ordinary CPDs and structural mechanisms in one graph

```python
dag.set_local_model("A", LogisticRegression(), parents=["Z"])
dag.add_mechanism("Y", ANM(regressor=RandomForestRegressor(), noise=NormalNoise()), parents=["A", "X"])
```

Observational and interventional queries can still use both nodes. Counterfactual queries are only allowed if the relevant ancestors expose the structural and abduction capabilities they need.

#### User journey 4: explicit failure for non-structural counterfactuals

```python
dag.counterfactual(
    target={"B": None},
    intervention={"A": 1},
    evidence={"A": 0, "B": "yes"},
)
```

If `B` only has an adapted classifier CPD and no structural semantics, pgmpy should raise a clear error explaining that counterfactual queries require structural local models rather than silently choosing a noise representation.
