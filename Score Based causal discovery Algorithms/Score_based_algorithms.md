## [Title]

Contributors: @anusa-saha (Anusa Saha, NIT Agartala)

### Introduction

**Extension of Score Based Causal Discovery Algorithms**

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

**Proposal 1: FGES Optimizations for GES:**

This proposal introduces lightweight FGES-style scalability optimizations into the current GES implementation while preserving the existing legality conditions, CPDAG semantics, and graph operation logic already implemented in the estimator.
The primary goal is to improve runtime scalability on larger graphs without rewriting the mathematical core of the algorithm or introducing a separate estimator implementation.
The proposed modifications target only the scheduling and execution strategy of candidate operations, while fully reusing the current score decomposition, legality conditions, and graph transformation operators already present in the implementation.
**The file where these changes are being proposed** : `pgmpy/causal_discovery/GES.py`

Motivation: The current GES implementation performs exhaustive rescanning of candidate edge operations after every graph modification. While this behavior is mathematically correct, it becomes increasingly expensive for larger graphs due to:
- repeated global candidate enumeration
- repeated local score evaluation
- combinatorial parent-set enumeration
- repeated graph legality validation

FGES-style locality optimization improves scalability by restricting rescanning to regions of the graph affected by recent structural modifications, while preserving the correctness of the underlying graph operations. The objective of this proposal is therefore not to redesign GES, but to introduce scalable execution strategies around the existing implementation.

Design Philosophy and Choices: A major design decision in this proposal is to avoid introducing a separate FGES.py implementation.Instead, the proposal extends the current estimator using a lightweight variant switch:
```python
variant="ges" | "fges"
```

This preserves a single shared implementation for: legality checks, CPDAG semantics, graph operators and score computation.

The following methods remain fully unchanged and shared between both variants: `insert()`, `delete()`, `turn()`, clique legality conditions, semidirected path checks, CPDAG calibration and conversion

This approach minimizes:
- code duplication
- maintenance overhead
- semantic divergence between implementations

The FGES variant modifies only:
- candidate scheduling
- locality-aware rescanning
- parallel score evaluation
- parent-set bounding
**Proposed Changes in API:**
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
**Core Optimizations:**
1. Locality-Aware Candidate Rescanning: The current GES implementation globally rescans candidate operations after every structural modification. The proposed FGES variant instead tracks recently modified nodes, and their immediate neighborhoods.
Only operations involving these affected regions are reconsidered during subsequent iterations. This significantly reduces unnecessary candidate evaluations while preserving graph connectivity and local score consistency.
Representative update logic:
```python
if self.variant == "fges":
    changed_nodes = {op_to_add[1], op_to_add[2]}

    local_neighbors = set()
    for node in changed_nodes:
        local_neighbors.update(current_model.all_neighbors(node))

    active_nodes = changed_nodes | local_neighbors
```
This locality-refresh mechanism is shared across: forward, backward and turning phases.
2. Parallel Candidate Scoring: Candidate edge evaluations are independent local score computations and therefore naturally parallelizable. The proposal introduces optional parallel score evaluation using joblib threading:
```python
results = Parallel(n_jobs=n_workers,backend="threading")(
    delayed(_score_insert)(u, v)
    for u, v in potential_edges
)
```
Thread-based execution was intentionally chosen over process-based multiprocessing because the implementation heavily uses shared graph objects, nested scoring closures are reused directly and process-based execution introduced substantial serialization overhead. The threading backend therefore preserves the existing implementation structure while enabling lightweight parallel score evaluation.

3. Parent-Set Bounding: One of the dominant runtime bottlenecks in score-based causal discovery is the exponential growth of parent-set enumeration. The proposal introduces an optional max_parents constraint to bound local search complexity:
```python
if (
    self.variant == "fges"
    and self.max_parents is not None
    and len(new_parents) > self.max_parents
):
    continue
```
This optimization is especially important for: dense graphs, large variable counts and high-dimensional continuous datasets. The parameter is optional and preserves the original exhaustive GES behavior when unset.

**Scalability Rationale**: The proposed modifications target the dominant computational bottlenecks of large-scale GES execution.
Classical GES repeatedly rescans a quadratic candidate space: `O(n^2)`while insert and turn operators additionally require exponential subset enumeration: `O(2^{|T_0|})`
The proposed FGES-style modifications improve scalability through:
- locality-aware rescanning
- reduced repeated candidate evaluation
- bounded parent-set exploration
- parallel local-score computation

These optimizations become increasingly important for sparse large-scale graphs, dense local neighborhoods, and high-dimensional datasets. 

Benchmark Summary

Initial experiments on the Sachs, Child, Alarm, Asia and Cancer example models, the benchmarks indicate that the proposed modifications preserve graph scores and edge counts while significantly improving runtime performance.

Experiments on the Alarm benchmark demonstrate that the proposed FGES-style locality optimizations substantially reduce runtime relative to baseline GES while preserving comparable graph quality. In particular, the optimized implementation achieved approximately 3–4× lower runtime by restricting candidate rescanning to locally affected graph regions and removing repeated temporary graph materialization during candidate evaluation.

The results further indicate that graph-query operations and subset enumeration now dominate runtime complexity, suggesting that the primary graph-copy and DAG-validation bottlenecks of the original implementation were successfully mitigated.

Check the benchmark code and results in `fges_parallelization.ipynb`

The proposal also removes inheritance from _ScoreMixin, which is currently unused in the estimator implementation:
```python
class GES(_BaseCausalDiscovery):
```
This removes ambiguity and simplifies the class hierarchy without affecting functionality.

**Summary**
This proposal introduces lightweight FGES-style scalability optimizations into the existing GES implementation while intentionally preserving:
- existing legality conditions
- CPDAG semantics
- graph transformation logic
- score decomposition behavior

The implementation focuses specifically on improving scalability through:
- locality-aware scheduling
- bounded parent-set exploration
- reduced graph materialization
- lightweight parallel score evaluation

By extending the current estimator instead of introducing a separate implementation, the proposal minimizes maintenance complexity while preserving correctness and implementation consistency.