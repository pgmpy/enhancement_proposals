# Bootstrap Estimator for Causal Discovery

Contributors: [Jatin bhardwaj](https://github.com/jatinbhardwaj-093)

## Description

Causal discovery algorithms often produce a single estimated causal graph from observational data. However, in practical settings, the discovered structure can be sensitive to sampling variability, noise, limited sample size, or small perturbations in the dataset. As a result, it becomes difficult to determine how reliable or stable the inferred causal relations actually are.

This proposal focuses on introducing non-parametric bootstrap estimators for causal discovery algorithms to provide a measure of confidence and stability for the discovered causal structure.

The core idea behind non-parametric bootstrap is to repeatedly generate new datasets by resampling the observed training data with replacement. The causal discovery algorithm is then re-executed on each resampled dataset. By analyzing how frequently particular edges or edge orientations appear across bootstrap iterations, we can estimate the robustness and consistency of the inferred causal relations.

The motivation for this approach is that causal discovery methods frequently operate in settings where multiple graph structures may explain the data nearly equally well. Small variations in the dataset can therefore lead to different inferred structures. Bootstrap-based estimation helps quantify this uncertainty by identifying relationships that remain stable under repeated resampling.

## Non-parametric Bootstrap Algorithm 

Step 1: Resample the dataset with replacement to generate a new dataset of size N for each iteration.

Step 2: Apply the causal discovery algorithm to the newly generated dataset.

Step 3: Repeat the process for M bootstrap iterations.

Step 4: Compute the statistics of interest (e.g., edge existence probabilities, edge orientation frequencies).

Step 5: Form a optimal consensus graph using the computed statistics.

## Consensus Graph Construction

To construct a final consensus graph, we apply a threshold to the estimated stability scores. However, naive thresholding introduces a challenge: even if every bootstrap graph is a valid Directed Acyclic Graph (DAG), independently aggregating highly frequent edges can create cycles.

To address this and construct a valid consensus DAG, we support two primary consensus strategies:

### 1. Greedy Edge-Addition (Kruskal-style)

This approach ensures that the highest confidence edges survive, subject to DAG constraints.
- Sort all candidate edges in descending order by their stability statistics.
- Iteratively add edges one-by-one to the final graph.
- If adding an edge creates a cycle, it is skipped.

### 2. False Positive Rate (FPR) & Branch-and-Cut (Recommended)

This method decouples thresholding from heuristic cycle resolution by introducing a statistically rigorous approach. This approach come to consensus by two steps:
- **False Positive Rate of Edges (FPR):** Fits a Beta mixture model using the Expectation-Maximization (EM) algorithm to model the distribution of true and false edges. It calculates a posterior log odds ratio for each candidate edge, allowing a dynamic confidence threshold tailored to a desired false positive rate.
- **Branch-and-Cut Acyclicity Optimization:** Models consensus DAG construction as an Integer Linear Program (ILP) that maximizes the total confidence score of selected edges while enforcing acyclicity using cutting-plane constraints.

For a comprehensive breakdown, mathematical formulation, and examples of this methodology, please refer to `consensus_fpr_method.md`.

## Architecture

The bootstrap framework is implemented using a single unified class:

`BootstrapEstimator` is the central interface responsible for:
- Executing the bootstrap estimation pipeline.
- Computing stability statistics (e.g., edge confidence and orientation scores).
- Managing iteration-level results.
- Constructing consensus graphs.

## API

```python
class BootstrapEstimator:
    """
    Main bootstrap estimation interface for causal discovery algorithms.
    """

    def __init__(...):
        """
        Parameters
        ----------
        estimator:
            Instance of the causal discovery estimator.

        n_resamples:
            Number of bootstrap iterations.

        sample_frac:
            Fraction of samples used in each bootstrap dataset.

        seed:
            Random seed used for reproducibility.

        n_jobs:
            Number of parallel processes.
        """

    def fit(self, data):
        """
        Execute the bootstrap estimation pipeline.

        Pipeline
        --------
        1. Generate bootstrap resampled datasets.
        2. Execute the causal discovery algorithm.
        3. Store iteration-level graph and sample index information.
        4. Compute stability statistics.

        Returns
        -------
        self

        Attributes Created
        ------------------
        edge_prob_: dict or DataFrame
            Edge existence probabilities estimated across bootstrap iterations.

        direction_prob_: dict or DataFrame
            Orientation stability scores estimated across bootstrap iterations.

        bootstrap_results_: dict or DataFrame
            Iteration-level structural summary table (history of all bootstrap runs).

        """

    # --------------------------------------------------
    # Consensus Graph Construction
    # --------------------------------------------------

    def consensus_graph(self, threshold=None):
        """
        Construct and return a valid consensus graph.
        """

    def adjacency_matrix(self, threshold=None, technique=None):
        """
        Return adjacency matrix representation of the consensus graph.
        """

    # --------------------------------------------------
    # Iteration-Level Access
    # --------------------------------------------------

    def nth_edges(self, n):
        """
        Return the edge existence probabilities for the nth bootstrap iteration.
        """
    
    def nth_directions(self, n):
        """
        Return the orientation frequencies for the nth bootstrap iteration.
        """
    
    def resampled_data(self, n):
        """
        Reconstruct and return the bootstrap dataset used during the nth
        bootstrap iteration.

        Notes
        -----
        Only sampled row indices are stored internally for memory efficiency.
        """
```


## Example Usage

```python
>>> from pgmpy.estimators import BootstrapEstimator
>>> from pgmpy.estimators import HillClimbSearch

# Initialize bootstrap estimator

>>> est = BootstrapEstimator(
        estimator=HillClimbSearch(),
        n_resamples=10,
        sample_frac=0.8,
        seed=42,
        n_jobs=-1,
    )

# Run bootstrap estimation

>>> est.fit(data)

# Access fitted stability statistics directly as attributes

>>> est.edge_prob_
>>> est.direction_prob_
>>> est.bootstrap_results_


# Construct consensus graph and adjacency matrix

>>> est.consensus_graph()
>>> est.adjacency_matrix()

# Access iteration-level details

>>> est.nth_edges(10)
>>> est.nth_directions(10)
>>> est.resample_data(10)

```

### References

- Original Issue: https://github.com/pgmpy/pgmpy/issues/3326
- Reference Paper: https://arxiv.org/pdf/1301.6695
- FPR Method: https://doi.org/10.4230/OASICS.GCB.2013.46
