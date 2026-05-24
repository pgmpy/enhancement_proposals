# FPR and Branch-and-cut algorithm

## **False Positive Rate of Edges (FPR)**

When we have no prior knowledge about the causal structure, choosing an appropriate cutoff becomes a difficult problem. Instead of using a naive or randomly chosen threshold, this algorithm focuses on controlling the false positive rate of edges.

The underlying idea of the algorithm is that the edge frequencies obtained across multiple bootstrap iterations can be assumed to follow a mixture of two Beta distributions:

- $\mathrm{Beta}(\alpha, 1)$, if the edge truly exists.
- $\mathrm{Beta}(1, \beta)$, if no real biological interaction exists between the nodes.

Correspondingly, the overall edge frequency distribution can be modeled as:

$$
\text{Edge Frequency} \sim p \cdot \mathrm{Beta}(\alpha, 1) + (1-p) \cdot \mathrm{Beta}(1, \beta)
$$

where:

- $p$ represents the probability that an edge truly exists,
- $\alpha$ controls the distribution of true edges,
- $\beta$ controls the distribution of false edges.

The parameters $\alpha$ and $\beta$ can be estimated using the `Expectation-Maximization` (EM) algorithm. After estimating these parameters, we can compute the posterior log odds ratio for each edge:

$$ edge[i,j] = \log \left( \frac{P(\text{edge is true})}{P(\text{edge is false})} \right) $$

If an edge has a positive posterior log odds ratio, then the probability of the edge being a true interaction is greater than the probability of it being a false interaction. Hence, such edges are more likely to represent true causal relationships.

However, instead of simply using a cutoff of `edge > 0`, the algorithm allows the selection of a dynamic threshold based on a desired false positive rate. Specifically, a cutoff $\tau$ is selected such that:

$$
\int_{\tau}^{1} \mathrm{Beta}(x,1,\beta)\,dx = q
$$

where $q$ is the prescribed false positive rate.

## **Branch-and-cut algorithm**

After computing the confidence score for each edge, the next objective is to construct a final causal graph that is both high-scoring and acyclic. In other words, we want to select the set of edges with maximum total confidence while ensuring that the resulting graph does not contain any directed cycles.

To formulate this problem, let \(W\) denote the weight matrix where each entry represents the confidence score of a directed edge. For every possible directed edge \(a \in V \times V\), a binary variable is introduced:

$$
x_a =
\begin{cases}
1 & \text{if edge } a \text{ is selected} \\
0 & \text{otherwise}
\end{cases}
$$

The optimization objective is to maximize the total weight of the selected edges:

$$
\max \sum_{a \in V \times V} w_a x_a
$$

subject to the constraint that the selected edges must not form any directed cycle.

For every directed cycle \(C\), the following constraint is imposed:

$$
\sum_{a \in C} x_a \leq |C| - 1
$$

This constraint guarantees that at least one edge from every possible cycle is removed, thereby ensuring that the final graph remains acyclic.

However, the number of possible directed cycles grows exponentially with the number of nodes, making it computationally infeasible to include all cycle constraints directly in the Integer Linear Program (ILP). To address this issue, the algorithm uses a **cutting-plane approach**.

Initially, the cycle constraints are ignored and the relaxed linear program is solved. The algorithm then searches for violated cycle constraints in the current fractional solution. If a directed cycle violating the acyclicity condition is found, the corresponding constraint is added back into the optimization problem. This process is repeated iteratively until no violated cycle constraints remain.

## Example of Consensus DAG Construction

Suppose we have four variables:

$$
V = \{A, B, C, D\}
$$

We generate many bootstrap samples from the dataset and run a causal discovery algorithm on each sample.

After 100 bootstrap iterations, suppose we obtain the following edge frequencies:

| Edge | Frequency |
|---|---|
| A -> B | 0.91 |
| B -> C | 0.88 |
| C -> A | 0.82 |
| A -> D | 0.20 |
| D -> C | 0.15 |

Using the Beta mixture model and EM algorithm, the confidence score for each edge is computed:

$$
r_{ij} = \log \left( \frac{P(\text{true edge})}{P(\text{false edge})} \right)
$$

Suppose the scores are:

| Edge | Score |
|---|---|
| A -> B | 4.2 |
| B -> C | 3.7 |
| C -> A | 3.1 |
| A -> D | -1.2 |
| D -> C | -2.0 |

Assume the false positive cutoff is:

$$
\tau = 1.0
$$

The adjusted score becomes:

$$
w(i,j) = r_{ij} - \tau
$$

Resulting adjusted scores:

| Edge | Adjusted Score |
|---|---|
| A -> B | 3.2 |
| B -> C | 2.7 |
| C -> A | 2.1 |
| A -> D | -2.2 |
| D -> C | -3.0 |

Only edges with positive adjusted scores are selected:

- A -> B
- B -> C
- C -> A

However, these edges form a directed cycle:

$$
A \rightarrow B \rightarrow C \rightarrow A
$$

Since a valid causal graph must be acyclic, the ILP optimization step is applied.

The ILP tries to maximize the total edge weight while removing enough edges to break all cycles.

The total score with all edges is:

| Edge | Weight |
|---|---|
| A -> B | 3.2 |
| B -> C | 2.7 |
| C -> A | 2.1 |

The cycle constraint says that not all edges in the cycle can be selected together.

The ILP therefore removes the edge with the smallest weight:

- C -> A

because removing it breaks the cycle while losing the least confidence.

The final DAG becomes:

- A -> B
- B -> C

This graph is acyclic and has the maximum possible total confidence score under the DAG constraint.
