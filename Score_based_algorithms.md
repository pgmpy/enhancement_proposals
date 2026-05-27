## [Title]

Contributors: @anusa-saha (Anusa Saha, NIT Agartala)

### Introduction

#### Extension of Score Based Causal Discovery Algorithms

The current repository mainly supports the GES (Greedy Equivalence Search) algorithm for causal discovery. Although GES is an important score-based method, it becomes computationally expensive for large and dense datasets because it directly searches in DAG/CPDAG space. Modern causal discovery research has introduced more scalable and efficient approaches to overcome these limitations.

This proposal aims to extend the repository by implementing a sequence of advanced algorithms:

- FGES (Fast Greedy Equivalence Search):
    FGES improves the scalability of traditional GES through optimized graph search.
- Non parameteric GES: 
    A non-parametric extension allows the framework to model complex nonlinear dependencies and non-Gaussian relationships
- SP (Sparsest Permutation):
    SP introduces permutation-based search instead of direct DAG-space optimization.
- GRaSP (Greedy Relaxations of the Sparsest Permutation):
    SP introduces permutation-based search instead of direct DAG-space optimization.
- BOSS (Best Order Score Search):
    BOSS further improves scalability using efficient caching and optimized search strategies.

Together, these implementations will provide a complete progression of modern score-based causal discovery algorithms within a unified and research-oriented framework.

### Proposed Solution

1. The proposed solution aims to significantly extend the current repository, which presently supports only the Greedy Equivalence Search (GES) algorithm [https://proceedings.mlr.press/v244/nazaret24a.html], into a comprehensive framework for modern score-based causal discovery. The extension will focus on implementing a sequence of scalable and research-oriented algorithms that represent the methodological evolution of causal structure learning.

2. The first stage of the implementation will introduce FGES (Fast Greedy Equivalence Search) [https://link.springer.com/article/10.1007/s41060-016-0032-z], an fast and scalable enhancement of the traditional GES algorithm . This proposal introduces a lightweight FGES-style optimization layer into the existing GES implementation while preserving the current legality logic, CPDAG operations, and public API structure in the `pgmpy/causal_discovery/GES.py` . The goal is to improve scalability and runtime performance without rewriting the core mathematical implementation.

3. After implementing FGES algorithm, the next natural extension is to develop a non-parametric version of the algorithm. While FGES primarily improves the efficiency and scalability of score-based causal discovery under linear Gaussian assumptions, a non-parametric extension allows the framework to model complex nonlinear dependencies and non-Gaussian relationships that are common in real-world data. This progression is logically well-founded because it separates the challenges of search correctness, scalability, and statistical expressivity into successive stages. By first establishing a robust and efficient search infrastructure through FGES, the same architecture can later support more flexible scoring functions such as kernel-based methods, Gaussian processes, or neural conditional likelihood models without redesigning the overall search procedure. 

4. After establishing the Non parametric GES framework, the work will be extended with the Sparsest Permutation (SP) algorithm to introduce a fundamentally different and more scalable approach to causal discovery. While FGES significantly improves the efficiency of classical GES through optimized graph-space search, it still performs structure learning directly in DAG/CPDAG space. As the number of variables increases, this approach becomes computationally expensive due to repeated CPDAG updates, legality checks for acyclicity, and large candidate graph search spaces. 
    These limitations motivate the need for a more structured and efficient optimization strategy. The SP algorithm addresses these challenges by reformulating causal discovery as a permutation optimization problem instead of direct graph optimization. Rather than searching over all possible graph structures, SP searches over variable orderings and constructs DAGs consistent with those orderings. This naturally enforces acyclicity and reduces the complexity associated with direct DAG-space operations. 
Note: This extension is particularly important because SP forms the theoretical foundation for later algorithms such as GRaSP and BOSS, both of which build upon permutation-search principles introduced by SP. Therefore, implementing SP serves as the next sensible step toward developing a complete and modern scalable causal discovery framework within the repository.

4. The next stage of the extension will implement GRaSP (Greedy Relaxations of the Sparsest Permutation), which improves the practicality and efficiency of permutation-based causal discovery introduced by the SP algorithm. Although SP provides an elegant reformulation of causal discovery through permutation optimization, evaluating large numbers of permutations remains computationally expensive for high-dimensional datasets. This creates scalability challenges when searching for optimal variable orderings in complex causal systems.
    GRaSP addresses these limitations by introducing a greedy local search strategy that efficiently explores permutation space instead of exhaustively evaluating possible orderings. Rather than performing expensive global permutation optimization, GRaSP incrementally improves candidate permutations using local relaxation operations and score-guided updates. This makes permutation-based causal discovery significantly more practical for larger graphs and dense causal structures.
Another major contribution of GRaSP is the introduction of grow-shrink based parent optimization, which improves parent selection efficiency during DAG construction. Instead of evaluating all possible parent combinations, GRaSP dynamically expands and prunes parent sets to optimize sparsity and local score improvement.

5. Finally, the last stage will be the BOSS (Best Order Score Search), which represents a scalable and optimized refinement of permutation-based causal discovery methods such as GRaSP. While GRaSP significantly improves the practicality of permutation search through greedy local relaxations and grow-shrink based parent optimization, it still suffers from repeated score computations and redundant parent-set evaluations during large-scale searches. These repeated computations become increasingly expensive for high-dimensional and dense causal graphs, limiting scalability and runtime performance.
    BOSS addresses these limitations by introducing more efficient order-search strategies along with reusable computation structures known as Grow-Shrink Trees (GSTs). Instead of repeatedly recomputing parent evaluations for similar permutations, GSTs cache and reuse grow-shrink computation results across multiple search iterations. This substantially reduces redundant score calculations and improves overall search efficiency.
Another important improvement introduced by BOSS is its simplified and efficient order-based hill climbing strategy. Rather than performing expensive global permutation relaxations, BOSS greedily updates variable orderings using score-guided local moves, making large-scale permutation optimization significantly faster and more scalable.

### Details of proposed solution

#### Proposal 1: FGES Optimizations for GES:
This proposal introduces lightweight FGES-style scalability optimizations into the current GES implementation while preserving the existing legality conditions, CPDAG semantics, and graph operation logic already implemented in the estimator.

The goal is to improve runtime scalability on larger graphs without rewriting the mathematical core of the algorithm.
**The file where these changes are being proposed** : `pgmpy/causal_discovery/GES.py`

**Proposed Changes in API:**
Import additions:
```python
import os
from concurrent.futures import ThreadPoolExecutor
```
```python
def __init__(
    self,
    scoring_method: str | BaseStructureScore | None = None,
    return_type: str = "pdag",
    min_improvement: float = 1e-6,

    # FGES-specific options
    variant: str = "ges",
    n_jobs: int = 1,
    max_parents: int | None = None,
):
```
**Function of each hyperparameter:**

| Parameter     | Purpose                                                                          |
| ------------- | -------------------------------------------------------------------------------- |
| `variant`     | Enables optimized FGES-style execution while preserving existing GES behavior    |
| `n_jobs`      | Parallelizes candidate edge scoring using multiple workers                       |
| `max_parents` | Restricts parent-set size to improve locality and reduce combinatorial explosion |

**Example Usage:**
```python
#Standard GES search
ges = GES(
    scoring_method="bic-g"
)

#FGES optimized search
fges = GES(
    variant="fges",
    scoring_method="bic-g",
    n_jobs=16,
    max_parents=4,
)
```

**Architectural Design:**

The implementation intentionally keeps all graph legality and CPDAG logic unchanged.

The following existing methods remain fully shared between both variants:

    - `insert()`
    - `delete()`
    - `turn()`
    - clique legality checks
    - semidirected path checks
    - CPDAG calibration and conversion

The FGES variant modifies only:

    - candidate scheduling
    - candidate scoring execution
    - parent-set restrictions
    - locality-aware updates

This minimizes code duplication while preserving correctness.

**Proposed Modifications:**
NOTE: One small correction, the `_ScoreMixin` is inherited by the `class GES(_ScoreMixin, _BaseCausalDiscovery):` but no function calls have been done. It should be removed for removing ambiguity
```python
# Class signature — remove _ScoreMixin as it is unused
class GES(_BaseCausalDiscovery):
```

**`_fit()` Changes**
1. Initialization (after `score_fn` is assigned)

**Current:**
```python
score = get_scoring_method(self.scoring_method, X)
score_fn = score.local_score
```
**Add immediately after:**
```python
n_workers = os.cpu_count() if self.n_jobs == -1 else self.n_jobs
use_parallel = self.variant == "fges" and n_workers != 1
```
2. Three nested scoring functions (add after `ordered_tuple`, before Step 2)

These three functions contain the existing per-`(u, v)` loop bodies, with only the `max_parents` guard added to `_score_insert`. They close over `current_model`, `score_fn`, `ordered_tuple`, and `self` all of which are read-only during parallel scoring.

```python
def _score_insert(u, v):
    T0 = current_model.undirected_neighbors(v) - current_model.all_neighbors(u)
    subsets = [[*T, False] for T in powerset(list(T0))]
    valid_insert_ops = []

    while subsets:
        entry = subsets.pop(0)
        T, passed_cond_2 = set(entry[:-1]), entry[-1]

        na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
        na_vuT = na_vu.union(T)

        cond_1 = current_model.is_clique(na_vuT)
        if not cond_1:
            subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
            continue

        if passed_cond_2:
            cond_2 = True
        else:
            cond_2 = not current_model.has_semidirected_path(
                v, u, blocked_nodes=na_vuT
            )
            if cond_2:
                for s in subsets:
                    if T.issubset(set(s[:-1])):
                        s[-1] = True

        if cond_1 and cond_2:
            parents_v = current_model.directed_parents(v)
            new_parents = ordered_tuple(
                na_vuT | parents_v | {u}, current_model
            )
            old_parents = ordered_tuple(
                na_vuT | parents_v, current_model
            )

            # max_parents guard — only addition vs original loop body
            if (
                self.variant == "fges"
                and self.max_parents is not None
                and len(new_parents) > self.max_parents
            ):
                continue

            score_delta = score_fn(v, new_parents) - score_fn(v, old_parents)
            new_model = self.insert(u, v, T, current_model)
            if new_model.has_acyclic_extension():
                valid_insert_ops.append((score_delta, u, v, T))

    if not valid_insert_ops:
        return 0.0, None
    best_op = max(valid_insert_ops, key=lambda x: x[0])
    return best_op[0], best_op


def _score_delete(u, v):
    if not current_model.has_edge(u, v):
        raise ValueError(f"No edge exists between nodes {(u, v)} to delete.")

    na_vu = current_model.undirected_neighbors(v) & current_model.all_neighbors(u)
    subsets = [[*H, False] for H in powerset(list(na_vu))]
    valid_delete_ops = []

    while subsets:
        entry = subsets.pop(0)
        H, cond_1 = set(entry[:-1]), entry[-1]

        if not cond_1 and current_model.is_clique(na_vu - H):
            cond_1 = True
            for s in subsets:
                if H.issubset(set(s[:-1])):
                    s[-1] = True

        if cond_1:
            aux = (na_vu - H) | current_model.directed_parents(v) | {u}
            old_parents = ordered_tuple(aux, current_model)
            new_parents = ordered_tuple(aux - {u}, current_model)
            score_delta = score_fn(v, new_parents) - score_fn(v, old_parents)
            valid_delete_ops.append((score_delta, u, v, H))

    if not valid_delete_ops:
        return 0.0, None
    best_op = max(valid_delete_ops, key=lambda x: x[0])
    return best_op[0], best_op


def _score_turn(u, v):
    valid_turn_ops = []

    if current_model.has_edge(u, v) and current_model.has_edge(v, u):
        non_adjacents = (
            current_model.undirected_neighbors(v)
            - current_model.all_neighbors(u)
            - {u}
        )

        if len(non_adjacents) > 0:
            C0 = current_model.undirected_neighbors(v) - {u}
            subsets = [
                [*set(C), False]
                for C in powerset(list(C0))
                if len(set(C) & non_adjacents) > 0
            ]

            while subsets:
                entry = subsets.pop(0)
                C = set(entry[:-1])

                cond_1 = current_model.is_clique(C)
                if not cond_1:
                    subsets = [s for s in subsets if not C.issubset(set(s[:-1]))]
                    continue

                subgraph = nx.DiGraph(
                    current_model.subgraph(current_model.chain_component(v))
                )
                na_vu = (
                    current_model.undirected_neighbors(v)
                    & current_model.all_neighbors(u)
                )

                if not self._separates({u, v}, C, na_vu - C, subgraph):
                    continue

                parents_v = current_model.directed_parents(v)
                parents_u = current_model.directed_parents(u)

                new_score = score_fn(
                    v, ordered_tuple(parents_v | C | {u}, current_model)
                ) + score_fn(
                    u, ordered_tuple(parents_u | (C & na_vu), current_model)
                )
                old_score = score_fn(
                    v, ordered_tuple(parents_v | C, current_model)
                ) + score_fn(
                    u, ordered_tuple(parents_u | (C & na_vu) | {v}, current_model)
                )
                score_delta = new_score - old_score

                new_model = self.turn(u, v, C, current_model)
                if new_model.has_acyclic_extension():
                    valid_turn_ops.append((score_delta, u, v, C))

    else:
        T0 = current_model.undirected_neighbors(v) - current_model.all_neighbors(u)
        subsets = [[*T, False] for T in powerset(list(T0))]

        while subsets:
            entry = subsets.pop(0)
            T, passed_cond_2 = set(entry[:-1]), entry[-1]

            na_vu = (
                current_model.undirected_neighbors(v)
                & current_model.all_neighbors(u)
            )
            C = na_vu.union(T)

            cond_1 = current_model.is_clique(C)
            if not cond_1:
                subsets = [s for s in subsets if not T.issubset(set(s[:-1]))]
                continue

            if passed_cond_2:
                cond_2 = True
            else:
                cond_2 = not current_model.has_semidirected_path(
                    v,
                    u,
                    blocked_nodes=C | current_model.undirected_neighbors(u),
                    ignore_direct_edge=True,
                )
                if cond_2:
                    for s in subsets:
                        if T.issubset(set(s[:-1])):
                            s[-1] = True

            if cond_1 and cond_2:
                parents_v = current_model.directed_parents(v)
                parents_u = current_model.directed_parents(u)

                new_score = score_fn(
                    v, ordered_tuple(C | parents_v | {u}, current_model)
                ) + score_fn(
                    u, ordered_tuple(parents_u - {v}, current_model)
                )
                old_score = score_fn(
                    v, ordered_tuple(C | parents_v, current_model)
                ) + score_fn(
                    u, ordered_tuple(parents_u, current_model)
                )
                score_delta = new_score - old_score

                new_model = self.turn(u, v, T, current_model)
                if new_model.has_acyclic_extension():
                    valid_turn_ops.append((score_delta, u, v, T))

    if not valid_turn_ops:
        return 0.0, None
    best_op = max(valid_turn_ops, key=lambda x: x[0])
    return best_op[0], best_op
```

3. Shared locality-refresh helper (inline, not a function)

This block is repeated after each graph-modifying step in all three phases. The
changed-node pair comes from `op_to_add`, `op_to_delete`, or `op_to_turn`
respectively — substitute accordingly.

```python
# Replace OP_NODE_A / OP_NODE_B with the actual op tuple indices for each phase:
#   forward:  op_to_add[1],    op_to_add[2]
#   backward: op_to_delete[1], op_to_delete[2]
#   turning:  op_to_turn[1],   op_to_turn[2]

if self.variant == "fges":
    changed_nodes = {OP_NODE_A, OP_NODE_B}
    local_neighbors = set()
    for node in changed_nodes:
        local_neighbors.update(current_model.all_neighbors(node))
    active_nodes = changed_nodes | local_neighbors
```

4. Forward phase replacement (Step 2)

Replace the entire Step 2 `while True` block with:

```python
# Step 2: Forward phase
active_nodes = set(self.variables_)   # full scan on first iteration

while True:
    candidate_nodes = (
        active_nodes if self.variant == "fges" else set(current_model.nodes())
    )

    potential_edges = []
    for u, v in combinations(sorted(candidate_nodes), 2):
        if not current_model.has_edge(u, v) and not current_model.has_edge(v, u):
            potential_edges.append((u, v))
            potential_edges.append((v, u))

    score_deltas = np.zeros(len(potential_edges))
    insertion_ops = []

    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                executor.map(lambda uv: _score_insert(uv[0], uv[1]), potential_edges)
            )
        for sd, op in results:
            score_deltas[len(insertion_ops)] = sd
            insertion_ops.append(op)
    else:
        for index, (u, v) in enumerate(potential_edges):
            sd, op = _score_insert(u, v)
            score_deltas[index] = sd
            insertion_ops.append(op)

    if (len(potential_edges) == 0) or (np.all(score_deltas < self.min_improvement)):
        break

    op_to_add = insertion_ops[np.argmax(score_deltas)]
    if op_to_add is None:
        break

    current_model = self.insert(op_to_add[1], op_to_add[2], op_to_add[3], current_model)
    current_model = current_model.to_cpdag()

    # Locality refresh — next iteration only rescans affected nodes
    if self.variant == "fges":
        changed_nodes = {op_to_add[1], op_to_add[2]}
        local_neighbors = set()
        for node in changed_nodes:
            local_neighbors.update(current_model.all_neighbors(node))
        active_nodes = changed_nodes | local_neighbors
```

5. Backward phase replacement (Step 3)

Replace the entire Step 3 `while True` block with:

```python
# Step 3: Backward phase
active_nodes = set(self.variables_)   # reset: full scan on first iteration

while True:
    all_removals = self._legal_edge_deletions(current_model)

    potential_removals = (
        [
            (u, v) for u, v in all_removals
            if u in active_nodes or v in active_nodes
        ]
        if self.variant == "fges"
        else all_removals
    )

    score_deltas = np.zeros(len(potential_removals))
    deletion_ops = []

    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                executor.map(lambda uv: _score_delete(uv[0], uv[1]), potential_removals)
            )
        for sd, op in results:
            score_deltas[len(deletion_ops)] = sd
            deletion_ops.append(op)
    else:
        for index, (u, v) in enumerate(potential_removals):
            sd, op = _score_delete(u, v)
            score_deltas[index] = sd
            deletion_ops.append(op)

    if (len(potential_removals) == 0) or (np.all(score_deltas < self.min_improvement)):
        break

    op_to_delete = deletion_ops[np.argmax(score_deltas)]
    if op_to_delete is None:
        break

    current_model = self.delete(
        op_to_delete[1], op_to_delete[2], op_to_delete[3], current_model
    )
    current_model = current_model.to_cpdag()

    # Locality refresh
    if self.variant == "fges":
        changed_nodes = {op_to_delete[1], op_to_delete[2]}
        local_neighbors = set()
        for node in changed_nodes:
            local_neighbors.update(current_model.all_neighbors(node))
        active_nodes = changed_nodes | local_neighbors
```

6. Turning phase replacement (Step 4)

Replace the entire Step 4 `while True` block with:

```python
# Step 4: Turning phase
active_nodes = set(self.variables_)   # reset: full scan on first iteration

while True:
    potential_turns = []
    for u, v in sorted(current_model.edges()):
        if self.variant == "fges" and u not in active_nodes and v not in active_nodes:
            continue
        potential_turns.append((v, u))

    score_deltas = np.zeros(len(potential_turns))
    turn_ops = []

    if use_parallel:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(
                executor.map(lambda uv: _score_turn(uv[0], uv[1]), potential_turns)
            )
        for sd, op in results:
            score_deltas[len(turn_ops)] = sd
            turn_ops.append(op)
    else:
        for index, (u, v) in enumerate(potential_turns):
            sd, op = _score_turn(u, v)
            score_deltas[index] = sd
            turn_ops.append(op)

    if (len(potential_turns) == 0) or (np.all(score_deltas < self.min_improvement)):
        break

    op_to_turn = turn_ops[np.argmax(score_deltas)]
    if op_to_turn is None:
        break

    current_model = self.turn(
        op_to_turn[1], op_to_turn[2], op_to_turn[3], current_model
    )
    current_model = current_model.to_cpdag()

    # Locality refresh
    if self.variant == "fges":
        changed_nodes = {op_to_turn[1], op_to_turn[2]}
        local_neighbors = set()
        for node in changed_nodes:
            local_neighbors.update(current_model.all_neighbors(node))
        active_nodes = changed_nodes | local_neighbors
```

**Why This Design?**: This proposal intentionally avoids: creating a separate FGES.py, duplicating legality logic, rewriting CPDAG operations, introducing additional abstractions. The current implementation already contains: correct local score decomposition, correct legality conditions, correct CPDAG graph operations. The proposed changes therefore focus exclusively on improving scalability and reducing unnecessary computation.





