## Expand Causal Identification Module

**Contributors**: [kajal-jotwani](https://github.com/kajal-jotwani)

### Introduction
 
pgmpy already supports several graphical causal identification strategies, such as
the backdoor criterion, frontdoor criterion, and instrumental variables. These existing methods are **role-based**: they inspect the causal graph and assign roles to nodes (e.g., identifying a valid adjustment set or instrument). All three methods follow a unified `_BaseIdentification` interface they accept a `causal_graph` with `exposures` and `outcomes` roles assigned, and return a **modified graph** with additional role assignments (e.g., `adjustment`, `frontdoor`, `iv`).
 
While this role-based design is clean and composable, it has a fundamental limitation: **it is incomplete**. Role-based methods can only identify causal effects for which a specific graphical criterion is satisfied. There exist valid causal models where none of these criteria apply, yet the interventional distribution `P(y | do(x))`
is still theoretically identifiable from purely observational data.
 
The canonical example is the **bow-arc** graph from Shpitser & Pearl (2006):
 
```
X → Z → Y
X ↔ Z       (bidirected arc — latent confounder U affects both X and Z)
```
 
In this ADMG:
- **No valid backdoor adjustment set exists** - the latent confounder between X and Z
  is unobserved, so it cannot be conditioned on.
- **Frontdoor criterion does not apply** - there is a bidirected arc from X to Z,
  meaning Z itself is confounded with X.
- **No standard instrumental variable** - there is no variable that affects X without
also being affected by the confounder.
Yet `P(Y | do(X))` is identifiable, and the ID algorithm derives the correct formula automatically:
 
```
P(Y | do(X)) = Σ_Z [ P(Z | X) · Σ_{X'} [ P(Y | Z, X') · P(X') ] ]
```
This project proposes extending pgmpy's identification capabilities with
**formula-returning** identification methods. Unlike role-based approaches, these
algorithms output a closed-form symbolic expression over observed distributions for
the target causal effect, or raise an exception when identification is not possible.
The ID algorithm (Shpitser & Pearl, 2006) is **complete** for semi-Markovian
models - it strictly subsumes what backdoor, frontdoor, and IV criteria can achieve individually.

### Proposed Solution

The project has two main components.

**Component 1: `ProbabilityExpression`** - a new base class and symbolic expression
hierarchy, placed in `pgmpy/identification/`, that represents the closed-form
formulas output by identification algorithms. It supports pretty-printing (LaTeX),
simplification, and numeric evaluation against observed data.
 
**Component 2: Four identification algorithms** - `ID`, `IDC`, `IDStar`, and `SigmaID` - each inheriting from `ProbabilityExpression`. They accept a causal graph as input and return a `ProbabilityExpression` tree representing the identified formula, or raise an exception when identification fails.
 
These algorithms do **not** inherit from `_BaseIdentification`. That base class is designed for role-based methods whose `identify()` returns a modified graph with role assignments. The ID-family algorithms return symbolic formulas a fundamentally different contract. A new, parallel hierarchy rooted at `ProbabilityExpression` is the correct design.
 
```
# Existing — unchanged
_BaseIdentification
    ├── Adjustment
    └── Frontdoor
 
# New — formula-returning
ProbabilityExpression               (pgmpy/identification/probability_expressions.py)
    ├── Prob, Marginal, Product, Quotient   ← expression tree nodes
    ├── ID        ← P(y | do(x))
    ├── IDC       ← P(y | do(x), z)
    ├── IDStar    ← P(y_x)  counterfactual queries
    └── SigmaID   ← P(y | do(x)) from multiple data sources
```
 
The four algorithms share a unified call pattern:
 
```python
result = ID().identify(admg)       # returns ProbabilityExpression
print(result.to_latex())           # renders LaTeX formula
result.evaluate(data=df)           # numeric estimate from observed data
```
 
| Algorithm | Query | Required roles | Failure |
|---|---|---|---|
| `ID` | `P(y \| do(x))` | `exposures`, `outcomes` | `HedgeException` |
| `IDC` | `P(y \| do(x), z)` | `exposures`, `outcomes`, `conditioning` | `HedgeException` |
| `IDStar` | `P(y_x)` counterfactual | `exposures`, `outcomes` + query | `HedgingException` |
| `SigmaID` | `P(y \| do(x))` multi-source | `exposures`, `outcomes` + sigma graphs | `SigmaHedgeException` |
 

### Alternative Solutions



### Details of proposed solution

#### `ProbabilityExpression` - Base Class and Expression Hierarchy
 
The ID-family algorithms output mathematical formulas, not graph annotations. We
need a class hierarchy to represent these formulas as Python objects. Each formula
is a tree built from four node types:
 
- **`Prob(variables, do, cond)`** - an atomic probability term, e.g. `P(Y | do(X))`
  or `P(Z | X)`. At the symbolic level, `do` and `cond` are sets of variable names,
  not value assignments (those come only at `evaluate()` time).
- **`Marginal(expr, summed_vars)`** - represents `Σ_{S} expr`, summing out a set of
  variables.
- **`Product(factors)`** - represents `expr_1 · expr_2 · ...`
- **`Quotient(numerator, denominator)`** - represents `numerator / denominator`,
  used by IDC.

`ProbabilityExpression` is the abstract base for all of these, and also for the
identification algorithm classes. It defines:
 
- `identify(causal_graph)` - validates the graph and calls `_identify()`. Only
  algorithm subclasses implement `_identify()`.
- `to_latex()` and `__repr__()` - abstract; every expression node must implement
  these.
- `simplify()` - optional simplification (default: identity).
- `evaluate(data)` - numeric evaluation against a `pd.DataFrame`.
The hierarchy is deliberately modelled on how SymPy organises expression trees:
every node - leaf or composite - shares the same base class.
 
```python
# pgmpy/identification/probability_expressions.py
 
class ProbabilityExpression(ABC):
    supported_graph_types: tuple = ()
 
    def identify(self, causal_graph) -> "ProbabilityExpression":
        self._validate_query(causal_graph)
        return self._identify(causal_graph)
 
    def _identify(self, causal_graph):
        raise NotImplementedError   # overridden only by algorithm subclasses
 
    @abstractmethod
    def to_latex(self) -> str: ...
 
    @abstractmethod
    def __repr__(self) -> str: ...
 
    def simplify(self) -> "ProbabilityExpression":
        return self  # default: no simplification
 
    def evaluate(self, data: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError
 
 
class Prob(ProbabilityExpression):
    """Atomic: P(variables | do(do_vars), cond_vars)"""
    def __init__(self, variables: frozenset,
                 do: frozenset = frozenset(),
                 cond: frozenset = frozenset()):
        self.variables = variables
        self.do = do
        self.cond = cond
 
    def to_latex(self) -> str: ...
    def __repr__(self) -> str: ...
 
 
class Marginal(ProbabilityExpression):
    """Σ_{summed_vars} expr"""
    def __init__(self, expr: ProbabilityExpression, summed_vars: frozenset): ...
    def to_latex(self) -> str: ...
    def __repr__(self) -> str: ...
 
 
class Product(ProbabilityExpression):
    """expr_1 · expr_2 · ..."""
    def __init__(self, factors: list): ...
    def to_latex(self) -> str: ...
    def __repr__(self) -> str: ...
 
 
class Quotient(ProbabilityExpression):
    """numerator / denominator"""
    def __init__(self, numerator: ProbabilityExpression,
                 denominator: ProbabilityExpression): ...
    def to_latex(self) -> str: ...
    def __repr__(self) -> str: ...
 
 
class HedgeException(Exception):
    """Raised when a causal effect is not identifiable."""
    def __init__(self, G, hedge):
        self.hedge = hedge
```
 
The frontdoor formula, as a `ProbabilityExpression` tree, illustrates how the
pieces compose:
 
```python
# P(Y | do(X)) = Σ_M [ P(M|X) · Σ_{X'} [ P(Y|M,X') · P(X') ] ]
 
expr = Marginal(
    Product([
        Prob(frozenset({"M"}), cond=frozenset({"X"})),
        Marginal(
            Product([
                Prob(frozenset({"Y"}), cond=frozenset({"M", "X"})),
                Prob(frozenset({"X"})),
            ]),
            summed_vars=frozenset({"X"}),
        ),
    ]),
    summed_vars=frozenset({"M"}),
)
 
print(expr.to_latex())
# \sum_{M} \left[ P(M \mid X) \cdot \sum_{X} \left[ P(Y \mid M, X) P(X) \right] \right]
```
 
#### ID and IDC - Complete Identification of P(y | do(x)) and P(y | do(x), z)

These two algorithms will be implemented as the part of the Issue [#2529](https://github.com/pgmpy/pgmpy/issues/2529). Before the Implementation of the ID algorithms we need to add method to `get_c_components()` Issue [#3079](https://github.com/pgmpy/pgmpy/issues/3079) to the `ADMG` class.

The `ID` algorithm (Shpitser & Pearl, 2006) is complete for semi-Markovian causal models. It can identify any identifiable interventional distribution, and when identification is not possible, it raises a `HedgeException` that includes the subgraph responsible.
 
The algorithm works recursively on the graph's **C-components** (groups of nodes connected through latent confounders, that is, connected components of the bidirected skeleton). It tries nine rules in order: restricting to ancestors of the outcome, pushing redundant variables into the intervention set, decomposing across independent C-components, and expressing identification in terms of observed conditionals. If none of these apply, a hedge is found and the effect is not identifiable.
 
```python
# pgmpy/identification/id_algorithm.py
 
class ID(ProbabilityExpression):
    """
    Identifies P(y | do(x)) in semi-Markovian causal models.
 
    Examples
    --------
    >>> from pgmpy.base import ADMG
    >>> from pgmpy.identification import ID, HedgeException
    >>> admg = ADMG(
    ...     directed_ebunch=[("X", "Z"), ("Z", "Y")],
    ...     bidirected_ebunch=[("X", "Z")],
    ...     roles={"exposures": "X", "outcomes": "Y"},
    ... )
    >>> result = ID().identify(admg)
    >>> print(result.to_latex())
    \sum_{Z} \left[ P(Z \mid X) \cdot \sum_{X} \left[ P(Y \mid Z, X) P(X) \right] \right]
    """
    supported_graph_types = (ADMG,)
 
    def _identify(self, causal_graph) -> ProbabilityExpression:
        y = frozenset(causal_graph.get_role("outcomes"))
        x = frozenset(causal_graph.get_role("exposures"))
        P = Prob(frozenset(causal_graph.nodes()))
        return _id_recursive(y, x, P, causal_graph)
```
 
The core recursion lives in a standalone module-level function `_id_recursive(y, x, P, G)`. It is not a method on `ID` because the recursion calls itself on internal subgraphs that have no role assignments, so the class-level validation would break it.
 
The table below lists every `ADMG` method the algorithm needs. Some already exist in the current codebase; the rest will be added as part of this project:
 
| Method | Status | Notes |
|---|---|---|
| `G.get_ancestors(node_set)` | Already exists | Traverses directed edges only, which is exactly what the algorithm needs |
| `G.get_ancestral_graph(node_set)` | Already exists | Returns the induced ADMG subgraph over ancestor nodes |
| `G.is_mseparated(u, v, observed)` | Already exists | Used by IDC for the d-separation check |
| `G.get_c_components()` | To be added ([#3079](https://github.com/pgmpy/pgmpy/issues/3079)) | `get_district(node)` already exists per node; this extends it to partition the full graph |
| `G.remove_incoming_edges(node_set)` | To be added | Returns a new ADMG with all directed edges into `node_set` removed, used to compute `G_{\bar{X}}` |
| `G.topological_sort(node_set)` | To be added | Topological order restricted to a given node set, built on top of the existing directed-edge structure |
| `G.predecessors_before(vi, ordered)` | To be added | Returns nodes appearing before `vi` in a given topological ordering |
 
`IDC` handles the conditional case `P(y | do(x), z)`. It first checks if the intervention is actually needed using a d-separation test on `G_{\bar{X}}`. If it is, it breaks the query into a ratio of two `ID` calls:
 
```
P(y | do(x), z) = P(y, z | do(x)) / P(z | do(x))
```
 
```python
class IDC(ProbabilityExpression):
    """
    Identifies P(y | do(x), z). Requires a 'conditioning' role on the graph.
 
    Examples
    --------
    >>> admg = ADMG(
    ...     directed_ebunch=[("X", "Z"), ("Z", "Y"), ("X", "Y")],
    ...     bidirected_ebunch=[("X", "Y")],
    ...     roles={"exposures": "X", "outcomes": "Y", "conditioning": "Z"},
    ... )
    >>> result = IDC().identify(admg)
    >>> print(result.to_latex())
    """
    supported_graph_types = (ADMG,)
 
    def _identify(self, causal_graph) -> ProbabilityExpression:
        y = frozenset(causal_graph.get_role("outcomes"))
        x = frozenset(causal_graph.get_role("exposures"))
        z = frozenset(causal_graph.get_role("conditioning"))
 
        G_bar_x = causal_graph.remove_incoming_edges(x)
        if G_bar_x.is_mseparated(y, x, conditional_set=z):
            return Prob(y, cond=x | z)
 
        P = Prob(frozenset(causal_graph.nodes()))
        return Quotient(
            _id_recursive(y | z, x, P, causal_graph),
            _id_recursive(z,     x, P, causal_graph),
        )
```
 
#### `IDStar` - Counterfactual Identification of `P(y_x)`
 
ID* (Shpitser & Pearl, 2008) handles counterfactual queries, basically questions like "what would Y have been if X had been set to x?", written as `P(Y_x = y)` in potential outcomes notation.
 
The algorithm builds a **parallel worlds graph** (twin network): a combined ADMG that duplicates nodes for each intervention context in the query, and connects the worlds through their shared latent variables (bidirected edges). Identification then runs on this expanded graph.
 
```python
class IDStar(ProbabilityExpression):
    """
    Identifies counterfactual queries P(y_x) via the ID* algorithm.
 
    Parameters
    ----------
    counterfactual_query : list[tuple]
        Each tuple is (variable, intervention_context), e.g.
        [("Y", {"X": 1})] represents P(Y_{X=1}).
 
    Examples
    --------
    >>> result = IDStar(
    ...     counterfactual_query=[("Y", {"X": 1})]
    ... ).identify(admg)
    >>> print(result.to_latex())
    """
    supported_graph_types = (ADMG,)
 
    def __init__(self, counterfactual_query: list):
        self.counterfactual_query = counterfactual_query
 
    def _identify(self, causal_graph) -> ProbabilityExpression:
        twin_graph = self._construct_twin_network(causal_graph)
        return _id_star_recursive(causal_graph, twin_graph,
                                  self.counterfactual_query)
```
 
#### `SigmaID` - Identification from Multiple Data Sources
 
sigma-ID (Bareinboim & Pearl, 2012) handles the case where you have more than one data source, for example an observational dataset alongside a partial randomised experiment. It generalises the ID algorithm to make use of this extra information, using sigma-calculus annotated graphs to track which variables were randomised in each source.
 
```python
class SigmaID(ProbabilityExpression):
    """
    Identifies P(y | do(x)) from multiple observational and/or experimental
    distributions using the sigma-ID algorithm.
 
    Parameters
    ----------
    sigma_graphs : list[ADMG]
        One ADMG per available data source, annotated to indicate which
        variables were randomised in that source.
 
    Examples
    --------
    >>> result = SigmaID(
    ...     sigma_graphs=[obs_admg, rct_admg]
    ... ).identify(obs_admg)
    >>> print(result.to_latex())
    """
    supported_graph_types = (ADMG,)
 
    def __init__(self, sigma_graphs: list):
        self.sigma_graphs = sigma_graphs
 
    def _identify(self, causal_graph) -> ProbabilityExpression:
        ...
```

### Module Structure
 
```
pgmpy/
├── identification/
│   ├── __init__.py                  <- exports all public classes
│   ├── base.py                      <- _BaseIdentification (unchanged)
│   ├── adjustment.py                <- Adjustment (unchanged)
│   ├── frontdoor.py                 <- Frontdoor (unchanged)
│   ├── probability_expressions.py   <- NEW: ProbabilityExpression,
│   │                                   Prob, Marginal, Product, Quotient,
│   │                                   HedgeException
│   ├── id_algorithm.py              <- NEW: ID, IDC, _id_recursive()
│   ├── id_star.py                   <- NEW: IDStar, _id_star_recursive()
│   └── sigma_id.py                  <- NEW: SigmaID
```
 

## User Journeys with the Solution

#### Journey 1: Researcher - Automatic identification in a confounded model

A researcher has a model where a latent variable confounds both the treatment X
and the mediator Z, so no backdoor adjustment exists. Instead of manually
deriving the identification formula, they let the ID algorithm do it.

```python
from pgmpy.base import ADMG
from pgmpy.identification import ID

admg = ADMG(
    directed_ebunch=[("X", "Z"), ("Z", "Y")],
    bidirected_ebunch=[("X", "Z")],
    roles={"exposures": "X", "outcomes": "Y"},
)

result = ID().identify(admg)
print(result.to_latex())
# \sum_{Z} \left[ P(Z \mid X) \cdot \sum_{X} \left[ P(Y \mid Z, X) P(X) \right] \right]
```

#### Journey 2: Data Scientist - Checking identifiability before estimation

Before spending time on estimation, a data scientist wants to know upfront
whether the causal effect can be identified from observational data at all.
If not, the algorithm tells them exactly which subgraph makes it impossible.

```python
from pgmpy.identification import ID, HedgeException

try:
    result = ID().identify(admg)
    print(result.to_latex())
except HedgeException as e:
    print(f"Not identifiable. Hedge: {list(e.hedge.nodes())}")
```

#### Journey 3: Epidemiologist - Conditional interventional distribution

An epidemiologist wants to estimate the effect of a treatment X on outcome Y,
but needs to condition on an observed baseline variable Z, for example a
pre-treatment health score. IDC handles this directly without needing to
manually apply Bayes' rule on top of the ID result.

```python
from pgmpy.base import ADMG
from pgmpy.identification import IDC

admg = ADMG(
    directed_ebunch=[("X", "Z"), ("Z", "Y"), ("X", "Y")],
    bidirected_ebunch=[("X", "Y")],
    roles={"exposures": "X", "outcomes": "Y", "conditioning": "Z"},
)
result = IDC().identify(admg)
print(result.to_latex())
```

### References
 
1. Shpitser, I. and Pearl, J. (2006). *Identification of Joint Interventional
   Distributions in Recursive Semi-Markovian Causal Models.* AAAI-06.
2. Shpitser, I. and Pearl, J. (2006). *Identification of Conditional Interventional
   Distributions.* UAI-06.
3. Shpitser, I. and Pearl, J. (2008). *Complete Identification Methods for the
   Causal Hierarchy.* JMLR, 9, 1941–1979.
4. Bareinboim, E. and Pearl, J. (2012). *Causal Inference by Surrogate Experiments:
   z-Identifiability.* UAI-12.