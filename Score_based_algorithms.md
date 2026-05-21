## [Title]

Contributors: anusa-saha

### Introduction

#### Extension of Score Based Causal Discovery Algorithms
The current repository mainly supports the GES (Greedy Equivalence Search) algorithm for causal discovery. Although GES is an important score-based method, it becomes computationally expensive for large and dense datasets because it directly searches in DAG/CPDAG space. Modern causal discovery research has introduced more scalable and efficient approaches to overcome these limitations.

This proposal aims to extend the repository by implementing a sequence of advanced algorithms:

    - FGES (Fast Greedy Equivalence Search):
    FGES improves the scalability of traditional GES through optimized graph search.
    - SP (Sparsest Permutation):
    SP introduces permutation-based search instead of direct DAG-space optimization.
    - GRaSP (Greedy Relaxations of the Sparsest Permutation):
    SP introduces permutation-based search instead of direct DAG-space optimization.
    - BOSS (Best Order Score Search):
    BOSS further improves scalability using efficient caching and optimized search strategies.

Together, these implementations will provide a complete progression of modern score-based causal discovery algorithms within a unified and research-oriented framework.

### Proposed Solution

1. The proposed solution aims to significantly extend the current repository, which presently supports only the Greedy Equivalence Search (GES) algorithm [https://proceedings.mlr.press/v244/nazaret24a.html], into a comprehensive framework for modern score-based causal discovery. The extension will focus on implementing a sequence of scalable and research-oriented algorithms that represent the methodological evolution of causal structure learning.
2. The first stage of the implementation will introduce FGES (Fast Greedy Equivalence Search) [https://link.springer.com/article/10.1007/s41060-016-0032-z], an optimized and scalable extension of the traditional GES algorithm designed for high-dimensional causal discovery. FGES improves the efficiency of equivalence-class search through local score decomposition, score caching, parallelized candidate evaluation, and optimized graph update mechanisms. The implementation will include:
    - forward phase edge insertion for greedy graph expansion
    - backward phase edge deletion for graph refinement and sparsity optimization
    - local decomposable score computation using scoring functions such as BIC or AIC
    - candidate edge evaluation and score-difference computation
    - legality checks to maintain acyclicity during graph updates
    - CPDAG construction and equivalence-class update utilities
    - score caching mechanisms to avoid redundant local score recomputation
    - parallelizable candidate edge scoring for scalability on large datasets
    - graph orientation utilities for maintaining valid causal structures during search iterations
3. After establishing the FGES framework, the work will be extended with the Sparsest Permutation (SP) algorithm to introduce a fundamentally different and more scalable approach to causal discovery. While FGES significantly improves the efficiency of classical GES through optimized graph-space search, it still performs structure learning directly in DAG/CPDAG space. As the number of variables increases, this approach becomes computationally expensive due to repeated CPDAG updates, legality checks for acyclicity, and large candidate graph search spaces. 
    These limitations motivate the need for a more structured and efficient optimization strategy. The SP algorithm addresses these challenges by reformulating causal discovery as a permutation optimization problem instead of direct graph optimization. Rather than searching over all possible graph structures, SP searches over variable orderings and constructs DAGs consistent with those orderings. This naturally enforces acyclicity and reduces the complexity associated with direct DAG-space operations. 
Note: This extension is particularly important because SP forms the theoretical foundation for later algorithms such as GRaSP and BOSS, both of which build upon permutation-search principles introduced by SP. Therefore, implementing SP serves as the next sensible step toward developing a complete and modern scalable causal discovery framework within the repository.

4. The next stage of the extension will implement GRaSP (Greedy Relaxations of the Sparsest Permutation), which improves the practicality and efficiency of permutation-based causal discovery introduced by the SP algorithm. Although SP provides an elegant reformulation of causal discovery through permutation optimization, evaluating large numbers of permutations remains computationally expensive for high-dimensional datasets. This creates scalability challenges when searching for optimal variable orderings in complex causal systems.
    GRaSP addresses these limitations by introducing a greedy local search strategy that efficiently explores permutation space instead of exhaustively evaluating possible orderings. Rather than performing expensive global permutation optimization, GRaSP incrementally improves candidate permutations using local relaxation operations and score-guided updates. This makes permutation-based causal discovery significantly more practical for larger graphs and dense causal structures.
Another major contribution of GRaSP is the introduction of grow-shrink based parent optimization, which improves parent selection efficiency during DAG construction. Instead of evaluating all possible parent combinations, GRaSP dynamically expands and prunes parent sets to optimize sparsity and local score improvement.

5. Finally, the last stage will be the BOSS (Best Order Score Search), which represents a scalable and optimized refinement of permutation-based causal discovery methods such as GRaSP. While GRaSP significantly improves the practicality of permutation search through greedy local relaxations and grow-shrink based parent optimization, it still suffers from repeated score computations and redundant parent-set evaluations during large-scale searches. These repeated computations become increasingly expensive for high-dimensional and dense causal graphs, limiting scalability and runtime performance.
    BOSS addresses these limitations by introducing more efficient order-search strategies along with reusable computation structures known as Grow-Shrink Trees (GSTs). Instead of repeatedly recomputing parent evaluations for similar permutations, GSTs cache and reuse grow-shrink computation results across multiple search iterations. This substantially reduces redundant score calculations and improves overall search efficiency.
Another important improvement introduced by BOSS is its simplified and efficient order-based hill climbing strategy. Rather than performing expensive global permutation relaxations, BOSS greedily updates variable orderings using score-guided local moves, making large-scale permutation optimization significantly faster and more scalable.

### Details of proposed solution



### User journeys with the solution

