## [Title]

Contributors: @anusa-saha (Anusa Saha, NIT Agartala)

### Introduction

**Extension of Score Based Causal Discovery Algorithms**

Score-based causal discovery algorithms aim to recover causal structure by optimizing a scoring criterion over the space of graphical models. While pgmpy currently provides support for Greedy Equivalence Search (GES), recent research has produced several scalable and theoretically stronger alternatives that address limitations in runtime, model assumptions, and search strategy.

This proposal extends pgmpy's score-based causal discovery framework through the implementation of a progression of modern algorithms: FGES, Nonparametric GES, SP, GRaSP, and BOSS.

These algorithms represent three major directions in causal discovery research:

1. Improving the scalability of equivalence-class search (FGES).
2. Extending score-based methods beyond linear Gaussian assumptions (Nonparametric GES).
3. Reformulating causal discovery as permutation optimization (SP, GRaSP, BOSS).

Together these implementations provide a comprehensive framework for modern score-based causal discovery while remaining consistent with pgmpy's existing API and architecture.

The current repository mainly supports the GES (Greedy Equivalence Search) algorithm for causal discovery. Although GES is an important score-based method, it becomes computationally expensive for large and dense datasets because it directly searches in DAG/CPDAG space. Modern causal discovery research has introduced more scalable and efficient approaches to overcome these limitations.

This proposal aims to extend the repository by implementing a sequence of advanced algorithms:
- **FGES (Fast Greedy Equivalence Search):**
  Improves the scalability of traditional GES through optimized candidate evaluation and efficient graph-space search.

- **Nonparametric GES:**
  Extends score-based causal discovery beyond linear Gaussian assumptions, enabling the modeling of nonlinear and non-Gaussian causal relationships.

- **SP (Sparsest Permutation):**
  Reformulates causal discovery as a search over variable permutations, selecting the sparsest DAG consistent with the observed conditional independencies.

- **GRaSP (Greedy Relaxations of the Sparsest Permutation):**
  Improves the practicality of SP by replacing exhaustive permutation search with efficient greedy local search over variable orderings.

- **BOSS (Best Order Score Search):**
  Further improves permutation-based causal discovery through efficient order optimization and Grow-Shrink Tree (GST) caching for scalable structure learning.

Together, these implementations will provide a complete progression of modern score-based causal discovery algorithms within a unified and research-oriented framework.

### Proposed Solution

1. The proposed solution aims to significantly extend the current repository, which presently supports only the Greedy Equivalence Search (GES) algorithm, into a comprehensive framework for modern score-based causal discovery. The extension will focus on implementing a sequence of scalable and research-oriented algorithms that represent the methodological evolution of causal structure learning.

2. The first stage of the implementation will introduce **FGES (Fast Greedy Equivalence Search)**, a scalable extension of the traditional Greedy Equivalence Search (GES) algorithm. Like GES, FGES performs score-based causal discovery by searching over equivalence classes of Directed Acyclic Graphs (DAGs) represented as CPDAGs. The algorithm proceeds through three phases: a forward phase, where edges are greedily added to maximize score improvement; a backward phase, where unnecessary edges are removed; and a turning phase, where edge orientations are modified to further improve the score. At each step, FGES evaluates legal graph operations and applies the operation that yields the largest increase in the chosen scoring criterion.

The primary difference between FGES and standard GES lies in its focus on scalability. FGES reduces the computational overhead of candidate evaluation through optimized search strategies and parallelization, making it significantly more practical for larger and denser causal graphs while preserving the correctness guarantees of GES.

This proposal introduces a lightweight FGES-inspired optimization layer into the existing `pgmpy/causal_discovery/GES.py` implementation. Rather than creating a separate class, the implementation extends the current GES framework through an optional execution mode while preserving the existing legality checks, CPDAG operations, scoring framework, and public API. The proposed work focuses on parallel candidate-score evaluation and degree-bounded search to improve runtime performance and scalability without modifying the underlying mathematical formulation of the algorithm.

**Link to paper**: https://link.springer.com/article/10.1007/s41060-016-0032-z
**Other implementations in pacakges**: 
- https://github.com/salesforce/PyRCA/blob/d85512b2/pyrca/graphs/causal/fges.py 
- https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/Fges.java 
**Documentation**: https://www.phil.cmu.edu/tetrad-javadocs/7.6.3/edu/cmu/tetrad/search/Fges.html

3. After implementing FGES, the next stage of the project will extend score-based causal discovery to nonparametric graphical models. Classical implementations of GES and FGES are typically analyzed under parametric assumptions, such as linear Gaussian or multinomial distributions, where scoring criteria like BIC are well-defined and theoretically justified. However, many real-world systems exhibit nonlinear relationships, complex conditional distributions, and non-Gaussian noise that cannot be adequately captured by these assumptions. Recent theoretical work on nonparametric GES demonstrates that the greedy equivalence search framework can be extended beyond parametric models while preserving consistency guarantees. Instead of restricting the underlying data-generating process to specific functional forms, the algorithm assumes only that the conditional distributions in the DAG factorization satisfy mild smoothness conditions. This allows the framework to model a much broader class of causal mechanisms, including nonlinear and non-additive relationships.

The core search procedure remains identical to standard GES. Starting from an empty graph, the algorithm performs a forward phase that greedily adds edges to improve the model fit, followed by a backward phase that removes unnecessary edges. The search is performed over Markov equivalence classes represented as CPDAGs rather than individual DAGs. The primary difference lies in how candidate models are evaluated. Rather than relying solely on classical parametric scores such as BIC, candidate DAG models are compared using nonparametric model-selection criteria derived from Bayesian model comparison. This enables the same greedy search framework to operate on general nonparametric DAG models while retaining the decomposability and consistency properties required by GES. 

Within pgmpy, this extension naturally builds upon the FGES infrastructure already developed in the first stage of the project. The existing search procedure, legality conditions, CPDAG operations, and optimization strategies can be reused, while the scoring component is generalized to support nonparametric model evaluation. This separation between search and scoring provides a flexible framework that can later accommodate kernel-based scores, Bayesian nonparametric likelihoods, and other expressive model classes without requiring modifications to the underlying graph search algorithm.
 

**Link to the paper**: https://arxiv.org/abs/2406.17228


4. Following the implementation of Nonparametric GES, the next stage of development will focus on the Sparsest Permutation (SP) algorithm, which introduces a fundamentally different perspective on causal structure learning. Unlike GES and FGES, which perform optimization directly in graph space through edge additions, deletions, and orientation operations, SP reformulates causal discovery as a search over variable permutations. This alternative formulation is motivated by theoretical results showing that the true Markov equivalence class can be recovered under assumptions that are strictly weaker than the faithfulness conditions required by many classical causal discovery algorithms.

The central idea of the SP algorithm is to associate every permutation of the variables with a minimal I-MAP (Independence Map). Given an ordering of variables, conditional independence tests are used to construct the sparsest DAG consistent with that ordering. The algorithm then evaluates all candidate permutations and selects the DAG corresponding to the permutation that produces the fewest edges. In other words, rather than directly optimizing graph structure, SP searches for the variable ordering whose induced DAG is maximally sparse while remaining consistent with the observed conditional independence relationships.

Formally, the optimization objective can be viewed as finding the permutation that minimizes the number of edges in the resulting minimal I-MAP. In the Gaussian setting, the paper further shows that this objective is equivalent to finding the sparsest Cholesky factorization of the inverse covariance matrix, establishing a direct connection between SP, sparse matrix factorization, and ℓ₀-penalized maximum likelihood estimation.

The significance of SP lies primarily in its theoretical properties rather than computational efficiency. The paper proves that SP is consistent under the Sparsest Markov Representation (SMR) assumption, which is strictly weaker than restricted faithfulness. As a result, SP can correctly recover causal structure in settings where traditional constraint-based and score-based approaches may fail. Although exhaustive permutation search is computationally expensive, the SP framework serves as the conceptual foundation for later scalable permutation-based methods, including GRaSP and BOSS, which replace exhaustive search with efficient greedy optimization strategies while retaining the key idea of searching over permutations rather than graph structures.

Implementing SP within pgmpy therefore represents an important intermediate step toward a modern permutation-based causal discovery framework. It introduces the core concepts of permutation search, minimal I-MAP construction, sparsity-based optimization, and ordering-based structure learning that underpin several state-of-the-art causal discovery algorithms.

**Link to the paper**: https://par.nsf.gov/servlets/purl/10064193
**Other implementation**: 
- https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/Sp.java

4. The next stage of the extension will focus on implementing the Greedy Relaxations of the Sparsest Permutation (GRaSP) algorithm, which was developed to overcome the computational limitations of the Sparsest Permutation (SP) framework while retaining its permutation-based perspective on causal discovery. Although SP enjoys strong theoretical guarantees under assumptions weaker than faithfulness, its practical applicability is severely limited because identifying the optimal solution requires examining an exponential number of variable permutations. As the number of variables increases, exhaustive permutation search quickly becomes computationally infeasible.

GRaSP addresses this challenge by replacing exhaustive permutation optimization with an efficient greedy search procedure over the space of variable orderings. Starting from an initial permutation, the algorithm incrementally improves the ordering through a sequence of local permutation transformations known as *tuck operations*. These operations modify the ordering while preserving much of its structure, allowing the algorithm to efficiently explore promising regions of the permutation space without evaluating all possible permutations. Through iterative application of these local improvements, GRaSP searches for permutations that induce increasingly sparse causal graphs.

A key contribution of GRaSP is the introduction of a hierarchy of increasingly relaxed search strategies, namely GRaSP0, GRaSP1, and GRaSP2. The lower tiers are shown to be logically equivalent to previously proposed permutation-search algorithms such as Triangle SP (TSP) and Edge SP (ESP), while higher tiers expand the search space through more general tuck operations. This relaxation enables GRaSP2 to recover correct causal structures under weaker assumptions than its lower-tier counterparts and improves its ability to handle violations of faithfulness.

For practical structure construction, GRaSP combines permutation search with the Grow-Shrink (GS) procedure for estimating local Markov boundaries. Rather than evaluating all possible parent sets for every variable, the GS procedure efficiently identifies relevant parents relative to a given ordering and constructs the corresponding DAG. This substantially reduces the computational burden of DAG construction while maintaining consistency with the permutation-induced structure.

The significance of implementing GRaSP extends beyond improving the scalability of SP. The algorithm establishes a practical framework for permutation-based causal discovery that scales to substantially larger graphs while preserving strong theoretical guarantees. Experimental results reported in the original paper demonstrate that GRaSP2 achieves high accuracy on dense graphs with more than one hundred variables and frequently outperforms traditional methods such as PC and FGES, particularly in settings involving near-violations of faithfulness. Consequently, GRaSP represents a critical step toward building a modern, scalable, and theoretically grounded causal discovery framework within the repository.

**Link to the paper**: https://proceedings.mlr.press/v180/lam22a.html
**Other Implementation**:
- https://github.com/py-why/causal-learn/blob/main/causallearn/search/PermutationBased/GRaSP.py
- https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/Grasp.java
- https://disco-coders.github.io/causalDisco/reference/grasp.html\
- https://rdrr.io/cran/causalDisco/src/R/grasp.R

5. The final stage of the extension will focus on implementing the Best Order Score Search (BOSS) algorithm, a modern permutation-based causal discovery method designed to improve the scalability and efficiency of previous approaches such as the Sparsest Permutation (SP) algorithm and its successor, Greedy Relaxations of the Sparsest Permutation (GRaSP). While GRaSP demonstrated strong accuracy on dense causal graphs through greedy exploration of permutation space, its computational requirements increase substantially as the number of variables grows. Large-scale searches require repeated execution of grow-shrink procedures and repeated evaluation of similar variable orderings, resulting in significant computational overhead.

BOSS addresses these limitations through a more efficient order-search framework that directly optimizes variable permutations using a best-move strategy. Starting from an initial ordering, the algorithm repeatedly selects variables and greedily relocates them to positions that maximize the overall score. This process continues until no further score improvement is possible, producing a high-scoring permutation from which a causal graph can be constructed. Compared to the more complex relaxation procedures used in GRaSP, this approach is simpler, requires fewer tuning parameters, and scales more effectively to high-dimensional datasets.

A major innovation introduced by BOSS is the use of Grow-Shrink Trees (GSTs), a specialized caching data structure designed for permutation-based causal discovery. Instead of repeatedly executing grow-shrink parent-selection procedures for every candidate permutation, GSTs store and reuse previously computed grow-shrink results. By caching parent-set evaluations and associated scores, GSTs eliminate a large amount of redundant computation and significantly accelerate permutation search. The same GST framework can also be used to speed up existing permutation-based algorithms such as GRaSP.

The significance of implementing BOSS lies in its ability to maintain the strong accuracy characteristics of GRaSP while achieving substantially better scalability. Experimental evaluations reported in the original work demonstrate that BOSS achieves performance comparable to or better than GRaSP while scaling to graphs containing hundreds or even thousands of variables with high average degree. The authors report successful applications to dense simulated networks and large-scale fMRI datasets, highlighting BOSS as one of the most scalable permutation-based causal discovery algorithms currently available. Consequently, BOSS represents the final optimization stage of the proposed causal discovery framework, providing an efficient, theoretically grounded, and practically scalable solution for large-scale structure learning.

**Link to the paper**: https://pmc.ncbi.nlm.nih.gov/articles/PMC11393735/
**Other implementations**:
- https://github.com/cmu-phil/boss
- https://github.com/cmu-phil/tetrad/blob/development/tetrad-lib/src/main/java/edu/cmu/tetrad/search/Boss.java
- https://github.com/py-why/causal-learn/blob/main/causallearn/search/PermutationBased/BOSS.py

### Repository Architecture

The proposed algorithms will be implemented within pgmpy/causal_discovery and integrated with the existing
causal discovery framework.
```
pgmpy/
    causal_discovery/
        GES.py
        SP.py
        GRaSP.py
        BOSS.py
        NonParametricGES.py

    tests/
        test_causal_discovery/
            test_GES.py
            test_SP.py
            test_GRaSP.py
            test_BOSS.py
            test_NonParametricGES.py
```

### Algorthmic Design of proposed algorithms

**Algorithm 1: FGES Optimizations for GES:**
**Problem Description**: The current GES implementation evaluates candidate insert, delete, and turn operations sequentially during each search phase. As the number of variables increases, candidate scoring becomes the dominant computational cost of structure learning. This proposal introduces a lightweight FGES-inspired execution mode that improves scalability while preserving the existing legality conditions, CPDAG semantics, graph operators, and scoring framework already present in 
`pgmpy/causal_discovery/GES.py`.

The implementation focuses on accelerating candidate evaluation through parallel execution and reducing search-space growth through optional degree constraints.

**Algorithm Overview**: The implementation preserves the standard three-phase GES search procedure:
- Forward Equivalence Search
- Backward Equivalence Search
- Turning Phase

*Algorithm Steps:*
1. Initialize the search with an empty CPDAG.
2. Execute the Forward Equivalence Search (FES) phase by generating all legal edge insertion operations.
3. Evaluate the score improvement associated with each legal insertion.
4. Apply the insertion yielding the largest positive score improvement.
5. Update the CPDAG and repeat until no insertion improves the score.
6. Execute the Backward Equivalence Search (BES) phase by generating all legal edge deletion operations.
7. Evaluate candidate deletions and apply the deletion with the largest positive score improvement.
8. Continue until no deletion improves the score.
9. Execute the Turning Phase by considering legal edge orientation modifications.
10. Apply the highest-scoring turning operation and repeat until convergence.
11. Return the final CPDAG representing the learned Markov equivalence class.

**Solution and Implementation Details**:

The implementation extends the existing class through:
```python
GES(
    variant="ges"
)

GES(
    variant="fges"
)
```
instead of introducing a separate classes. This allows both variants to share:
- insert operator
- delete operator
- turn operator
- clique legality conditions
- semi-directed path checks
- CPDAG calibration
- scoring functions
while minimizing code duplication.

**Optimization 1:** Parallel Candidate Evaluation
The largest computational bottleneck in GES is candidate score evaluation.
For a fixed graph state, candidate operations are independent and can therefore be evaluated concurrently.
The implementation uses Joblib:
```python
results = Parallel(
    n_jobs=n_workers,
    backend="threading"
)(
    delayed(self._score_insert)(
        current_model,
        score_fn,
        ordered_tuple,
        u,
        v,
    )
    for u, v in potential_edges
)
```
The same strategy is applied to:
- Forward Search
- Backward Search
- Turning Phase

**Optimization 2:** Degree-Bounded Search
An optional degree constraint is introduced through:
`max_neighbors`
Before evaluating a candidate insertion:
```python
if (
    len(current_model.all_neighbors(u))
    >= self.max_neighbors
    or
    len(current_model.all_neighbors(v))
    >= self.max_neighbors
):
    continue
```
This prevents exploration of excessively dense graphs and reduces the number of candidate operations considered during the forward phase.

**API**
```python
GES(
    scoring_method=None,
    return_type="pdag",
    min_improvement=1e-6,

    variant="ges",
    n_jobs=1,
    max_neighbors=None,
)
```
| Parameter         | Description                                        |
|-------------------|----------------------------------------------------|
| `variant`         | Search mode: `"ges"` or `"fges"`.                  |
| `n_jobs`          | Number of parallel workers for candidate scoring.  |
| `max_neighbors`   | Maximum allowed node degree during forward search. |

**Test Plan**:
1. Correctness Tests: If FGES returns returns valid CPDAGs, preserves legality conditions as Insert/delete/turn operators remain unchanged.
GES and FGES produce identical results when n_jobs=1 for both:
`variant="ges"`
`variant="fges"`

2. Parallelization Tests: Verify that: n_jobs={1, 2, 4, 8} produce identical score improvements and final graphs.

3. Degree Constraint Tests: Verify that: `max_neighbors=3` never allows a node degree greater than three.

4. Benchmark Tests: Compare GES and FGES on example models containing:
- 50 variables
- 100 variables
- 500 variables
**Metrics:** Runtime, Candidate evaluations, SHD

**Unit Tests**: `pgmpy/tests/test_causal_discovery/test_GES.py`

Comprehensive unit tests will be developed for all FGES-specific functionality and integrated into the existing GES test suite. The tests will verify:
- Correct execution of the variant="fges" mode
- Correct behavior of parallel candidate evaluation across different values of n_jobs
- Deterministic outputs between sequential and parallel execution
- Correct enforcement of the max_neighbors degree constraint
- Preservation of DAG acyclicity and CPDAG validity
- Consistency between GES and FGES outputs on benchmark datasets
- Compatibility with the existing scikit-learn estimator interface

The new tests will be added alongside the current GES test suite to ensure that all FGES optimizations preserve the correctness guarantees of the original implementation while maintaining full regression coverage for future development.

**Algorithm 2 : Non Paramteric GES**
API design to be decided

**Algorithm 3: SP**
API design to be decided

**Algorithm 4: GRaSP**
API design to be decided

**Algorithm 5: BOSS**
API design to be decided