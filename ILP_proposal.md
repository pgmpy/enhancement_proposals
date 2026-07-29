# Causal Discovery Using Integer Linear Programming

**Contributors**: [Jatin Bhardwaj](https://github.com/jatinbhardwaj-093)

## Introduction

Learning Directed Acyclic Graphs (DAGs) from continuous observational data is a core task in causal discovery. While heuristic and continuous optimization approaches provide scalable approximations for causal graph learning, exact mathematical programming algorithms offer unique guarantees by finding globally optimal DAG structures accompanied by formal optimality gap certificates.

This proposal aims to introduce an exact Mixed-Integer Linear Programming (MILP) algorithm, `ILPSearch`, into `pgmpy.causal_discovery`. By formulating continuous Linear Structural Equation Models (SEMs) into exact integer programming constraints over directed candidate edges, `ILPSearch` will eliminate the risk of local minima while incorporating prior structural knowledge seamlessly.
By relying on SciPy's built-in MILP solver (`scipy.optimize.milp`), `ILPSearch` provides exact causal structure learning without requiring external solver installations or licensing setup.

---

## Proposed Solution

### Algorithm Steps

The proposed `ILPSearch` algorithm will follow a 5-step pipeline during execution in `_fit(X)`:

1. **Step 1: Empirical Covariance Computation**:
   Compute the sample empirical covariance matrix from input data matrix $\mathcal{X} \in \mathbb{R}^{n \times m}$:
   $$S = \frac{1}{n} \mathcal{X}^T \mathcal{X}$$

2. **Step 2: Superstructure & `ExpertKnowledge` Resolution**:
   If `expert_knowledge` is provided, fit it to resolve candidate search space edges $\mathcal{M} = (V, E)$, required edges, forbidden edges, and temporal orderings. If `expert_knowledge` is None, automatically estimate the moral graph superstructure via Graphical Lasso (sparse inverse covariance estimation).

3. **Step 3: Big-$M$ Parameter Estimation**:
   For each variable $X_k$, run an unconstrained least-squares regression against its candidate parents in $\mathcal{M}$ to find unconstrained coefficients $\hat{\beta}_{jk}^*$. Compute the Big-$M$ bound:
   $$M = 2 \max_{(j,k) \in \vec{E}} |\hat{\beta}_{jk}^*|$$

4. **Step 4: MILP Optimization Model Setup**:
   Construct the Mixed-Integer Linear Program (MILP) matrix and bound vectors for `scipy.optimize.milp`:
   - Define decision variables ($\beta_{jk}, z_{jk}, g_{jk}, \psi_k$).
   - Formulate linear objective and Big-$M$ weight bounds.
   - Add tournament orientation constraints and Layered Network acyclicity constraints.
   - Enforce fixed variable bounds from `required_edges_` ($z_{jk}=1$) and `forbidden_edges_` ($z_{jk}=0$).

5. **Step 5: SciPy Solver Call & Graph Extraction**:
   Invoke `scipy.optimize.milp` to global optimality (or within `opt_gap`). Extract active directed edges ($z_{jk}=1$ and $\beta_{jk} \neq 0$), construct the final `pgmpy.base.DAG` object, and store it in `self.causal_graph_` alongside `self.adjacency_matrix_`.

---

### Mathematical & Logical Formulation

#### 1. Optimization Objective ($\text{PNL}\mathcal{M}$)

For a continuous linear Structural Equation Model (SEM) $X_k = \sum_{j \in \text{pa}(k)} \beta_{jk} X_j + \delta_k$ with sample covariance matrix $S = \frac{1}{n} \mathcal{X}^T \mathcal{X}$, the loss function is:

$$l_n(B) = \frac{1}{2} \text{tr}\left\{ (I - B)(I - B)^T S \right\} = \frac{1}{2n} \sum_{k=1}^m \sum_{d=1}^n \left( x_{dk} - \sum_{(j,k) \in \vec{E}} \beta_{jk} x_{dj} \right)^2$$

The overall optimization problem minimizes the loss plus a sparsity penalty over the candidate directed edge set $\vec{E}$:

$$\min_{B} \quad l_n(B) + \lambda \phi(B) \quad \text{s.t. } G(B) \text{ is an induced DAG from } \overrightarrow{\mathcal{M}}$$

#### 2. Layered Network (LN) Model Specifications

##### $\ell_0$-Regularization Model ($\ell_0\text{-LN}$):
$$\begin{aligned}
\min_{\beta, z, g, \psi} \quad & \frac{1}{2} \text{tr}\left\{ (I - B)(I - B)^T S \right\} + \lambda \sum_{(j,k) \in \vec{E}} g_{jk} \\
\text{s.t.} \quad & -M g_{jk} \le \beta_{jk} \le M g_{jk}, \quad \forall (j,k) \in \vec{E} & \text{(Big-}M\text{ Weight Bound)} \\
& g_{jk} \le z_{jk}, \quad \forall (j,k) \in \vec{E} & \text{(Active Edge Link)} \\
& z_{jk} + z_{kj} = 1, \quad \forall (j,k) \in \vec{E}, \; j < k & \text{(Tournament Choice)} \\
& z_{jk} - (m-1) z_{kj} \le \psi_k - \psi_j, \quad \forall (j,k) \in \vec{E} & \text{(Layer Acyclicity)} \\
& 1 \le \psi_k \le m, \quad \forall k \in V & \text{(Layer Bounds)} \\
& z_{jk}, g_{jk} \in \{0, 1\}, \quad \beta_{jk} \in \mathbb{R}, \quad \psi_k \in \mathbb{R}
\end{aligned}$$

##### $\ell_1$-Regularization Model ($\ell_1\text{-LN}$):
$$\begin{aligned}
\min_{\beta, z, \psi} \quad & \frac{1}{2} \text{tr}\left\{ (I - B)(I - B)^T S \right\} + \lambda \sum_{(j,k) \in \vec{E}} |\beta_{jk}| \\
\text{s.t.} \quad & -M z_{jk} \le \beta_{jk} \le M z_{jk}, \quad \forall (j,k) \in \vec{E} & \text{(Big-}M\text{ Weight Bound)} \\
& z_{jk} + z_{kj} = 1, \quad \forall (j,k) \in \vec{E}, \; j < k & \text{(Tournament Choice)} \\
& z_{jk} - (m-1) z_{kj} \le \psi_k - \psi_j, \quad \forall (j,k) \in \vec{E} & \text{(Layer Acyclicity)} \\
& 1 \le \psi_k \le m, \quad \forall k \in V & \text{(Layer Bounds)} \\
& z_{jk} \in \{0, 1\}, \quad \beta_{jk} \in \mathbb{R}, \quad \psi_k \in \mathbb{R}
\end{aligned}$$

#### 3. Logical Mapping for `ExpertKnowledge` Constraints

- **Search Space ($\mathcal{M}$)**: Includes only variables $z_{jk}, \beta_{jk}$ for pairs $(j,k) \in \text{search\_space\_}$.
- **Forbidden Edges**: Adds hard constraints $z_{jk} = 0, \beta_{jk} = 0$.
- **Required Edges**: Adds hard constraints $z_{jk} = 1$.
- **Temporal Tier Ordering**: For node $j$ in tier $T(j)$ and node $k$ in tier $T(k)$ where $T(j) < T(k)$, adds $\psi_j + 1 \le \psi_k$ and $z_{kj} = 0$.

---

## Alternative Solutions

1. **Cutting Plane (CP) Approach**:
   - Uses binary edge variables and dynamically adds cycle-elimination constraints whenever cycles are detected during solver execution (via DFS callbacks).
   - *Comparison*: Requires custom solver callbacks which are not supported by standard `scipy.optimize.milp`.

2. **Linear Ordering (LO) Approach**:
   - Enforces a linear ordering of nodes using binary pairwise ordering variables and 3-cycle elimination constraints.
   - *Comparison*: Generates $O(m^3)$ constraints for $m$ nodes, which slows down branch-and-bound solves.

3. **Topological Ordering (TO) Approach**:
   - Uses a binary permutation matrix to assign topological positions to nodes.
   - *Comparison*: Suffers from high symmetry (multiple topological orderings represent the exact same DAG), causing redundant search tree branches.

---

## Details of proposed solution

#### Implementation Workflow

1. **Process Superstructure & `ExpertKnowledge`**:
   - If `expert_knowledge` is provided, call `expert_knowledge.fit(X)` to extract `search_space_`, `required_edges_`, `forbidden_edges_`, and temporal orderings.
   - If `expert_knowledge` is None, estimate candidate superstructure edges via `GraphicalLassoCV` or default to complete graph.

2. **Estimate Big-M Bounds**:
   - Run OLS regressions for each node against candidate parents to estimate raw coefficients.
   - Compute Big-M bound `M = max(2.0 * max_beta, 10.0)` for decision variable bounds.

3. **Construct `scipy.optimize.milp` Inputs**:
   - Concatenate decision vector `x = [z, beta, g, psi]` (binary edge indicators `z`, continuous weights `beta`, $L_0$ penalty flags `g`, layer variables `psi`).
   - Define objective vector `c` with penalty `l_penalty` and `integrality` array.
   - Set `Bounds(lb, ub)` for variables and lock `required_edges_` (`lb=1`) and `forbidden_edges_` (`ub=0`).
   - Build sparse `LinearConstraint(A, clb, cub)` enforcing Big-M bounds, tournament choice, layer acyclicity, and temporal tier ordering.

4. **Execute Solver & Extract DAG**:
   - Solve the MILP model using SciPy's `milp` function.
   - Extract active directed edges into a `DAG`.

### Class API Design

```python
class ILPSearch(BaseCausalDiscovery):
    """
    Exact score-based causal discovery for continuous data using Integer Programming.

    Implements the Layered Network (LN) formulation to find globally optimal DAGs
    with L0 or L1 sparsity penalties using scipy.optimize.milp.

    Parameters
    ----------
    penalty : {'l0', 'l1'}, default='l0'
        Regularization penalty type.

    l_penalty : float, default=0.1
        Sparsity penalty coefficient (lambda).

    expert_knowledge : ExpertKnowledge instance or None, default=None
        Prior expert knowledge specifying search space (superstructure graph),
        forbidden edges, required edges, or temporal ordering. If None, a moral
        graph is automatically estimated from data.

    opt_gap : float, default=0.001
        Target relative MIP optimality gap tolerance.
    """

    def __init__(
        self,
        penalty="l0",
        l_penalty=0.1,
        expert_knowledge=None,
        opt_gap=0.001,
    ):
        self.penalty = penalty
        self.l_penalty = l_penalty
        self.expert_knowledge = expert_knowledge
        self.opt_gap = opt_gap

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the ILPSearch algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : ILPSearch
            Fitted instance with learned graph stored in `self.causal_graph_`.
        """
        # 1. Resolve ExpertKnowledge search space, required/forbidden edges
        # 2. Map Layered Network problem to scipy.optimize.milp
        # 3. Solve using SciPy HiGHS backend
        # 4. Store fitted DAG in self.causal_graph_
        return self
```

### Zero-Extra-Dependency Integration

Since `scipy` is already a core dependency of `pgmpy`, `ILPSearch` relies directly on `scipy.optimize.milp` to execute exact integer programming out of the box.

---

## Usage

### Case 1: Learning an Optimal Causal Graph from Continuous Data

```python
import pandas as pd
from pgmpy.causal_discovery import ILPSearch

# Load continuous dataset
df = pd.read_csv("continuous_data.csv")

# Initialize ILP causal discovery algorithm with L0 sparsity penalty
model = ILPSearch(penalty="l0", l_penalty=0.1)

# Fit globally optimal DAG
model.fit(df)

# Retrieve learned DAG
dag = model.causal_graph_
print("Learned DAG Edges:", dag.edges())
```

### Case 2: Using `ExpertKnowledge` (Superstructure & Structural Constraints)

```python
import pandas as pd
from pgmpy.causal_discovery import ExpertKnowledge, ILPSearch

# Define expert knowledge (superstructure search space & edge constraints)
ek = ExpertKnowledge(
    search_space=[("A", "B"), ("B", "C"), ("A", "C")],
    required_edges=[("A", "B")],
    forbidden_edges=[("C", "A")]
)

model = ILPSearch(
    penalty="l1",
    l_penalty=0.05,
    expert_knowledge=ek
)

model.fit(df)
dag = model.causal_graph_
```

---

## References

[1] [Integer Linear Programming for the Bayesian network structure learning problem.](https://linkinghub.elsevier.com/retrieve/pii/S0004370215000417)

[2] [Integer Programming for Learning Directed Acyclic Graphs from Continuous Data](https://pubsonline.informs.org/doi/epdf/10.1287/ijoo.2019.0040)

[3] [Branch and Bound](https://en.wikipedia.org/w/index.php?title=Branch_and_bound&oldid=1352876192)

[4] [Branch and Cut](https://en.wikipedia.org/w/index.php?title=Branch_and_cut&oldid=1284905949)

[5] [Cutting Plane Method](https://en.wikipedia.org/wiki/Cutting-plane_method)
