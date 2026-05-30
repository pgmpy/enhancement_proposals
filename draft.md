# Bootstrap Estimator for Causal Discovery

Contributors: [Jatin bhardwaj](https://github.com/jatinbhardwaj-093)

## Description

Causal discovery algorithms often produce a single estimated causal graph from observational data. However, in practical settings, the discovered structure can be sensitive to sampling variability, noise, limited sample size, or small perturbations in the dataset. As a result, it becomes difficult to determine how reliable or stable the inferred causal relations actually are.

This proposal focuses on introducing non-parametric bootstrap estimators for causal discovery algorithms to provide a measure of confidence and stability for the discovered causal structure.

The core idea behind non-parametric bootstrap is to repeatedly generate new datasets by resampling the observed training data with replacement. The causal discovery algorithm is then re-executed on each resampled dataset. By analyzing how frequently particular edges or edge orientations appear across bootstrap iterations, we can estimate the robustness and consistency of the inferred causal relations.

The motivation for this approach is that causal discovery methods frequently operate in settings where multiple graph structures may explain the data nearly equally well. Small variations in the dataset can therefore lead to different inferred structures. Bootstrap-based estimation helps quantify this uncertainty by identifying relationships that remain stable under repeated resampling.

## Non-parametric Bootstrap Algorithm 

Step 1: Generate a resampled dataset of size N by sampling with replacement from the original observational data.

Step 2: Run the target causal discovery estimator on the resampled dataset.

Step 3: Repeat Steps 1 and 2 for M independent bootstrap iterations.

Step 4: Compute the statistics of interest (e.g., edge existence probabilities, edge orientation frequencies).

Step 5: Construct an optimal consensus graph from the aggregated bootstrap statistics.

### Consensus Graph Construction

From all the bootstrap graphs, we construct a consensus graph. The simplest way is to filter edges based on their bootstrap presence probabilities, retaining only those that exceed a specified threshold. 
The issue with construction of this graph is that this can introduce cyclicity in our consensus graph. So we need to make sure that our graph remains a DAG.

To avoid cyclicity, we can use a greedy approach. We arrange the edges in descending order of their probabilities and then add the edges that do not lead to a cycle.


## Architecture

The bootstrap framework is implemented using a single unified class:

`BootstrapEstimator` is the central interface responsible for:
- Executing the bootstrap estimation pipeline.
- Computing stability statistics (e.g., edge confidence and orientation scores).
- Managing iteration-level results.
- Constructing consensus graphs.

## API

```python
class BootstrapEstimator(_BaseCausalDiscovery):
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

        threshold:
            Threshold for the edge existence probabilities across bootstrap iterations.

        show_progress:
            Display a progress bar during bootstrap estimation.

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
        5. Construct consensus graph using class threshold.

        Returns
        -------
        self

        Attributes Created
        ------------------
        causal_graph_: DAG
            Consensus graph causal structure estimated across bootstrap iterations. 

        adjacency_matrix_: DataFrame
            Adjacency matrix representation of the consensus graph.

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

    def get_consensus_graph(self, threshold=None):
        """
        Construct and return a valid consensus graph.
        """

    def get_adjacency_matrix(self, threshold=None):
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

    def nth_causual_graph(self, n):
        """
        Return the causal graph for the nth bootstrap iteration.
        """
    
    def nth_adjacency_matrix(self, n):
        """
        Return the adjacency matrix for the nth bootstrap iteration.
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
>>> from pgmpy.causal_discovery import HillClimbSearch

# Initialize bootstrap estimator

>>> est = BootstrapEstimator(
        estimator=HillClimbSearch(),
        n_resamples=10,
        threshold=0.5,
        show_progress=True,
        sample_frac=0.8,
        seed=42,
        n_jobs=-1,
    )

# Run bootstrap estimation

>>> est.fit(data)

# Access fitted stability statistics directly as attributes

>>> est.causal_graph_
>>> est.adjacency_matrix_
>>> est.edge_prob_
>>> est.direction_prob_
>>> est.bootstrap_results_

# Construct consensus graph and adjacency matrix

>>> est.get_consensus_graph()
>>> est.get_adjacency_matrix()

# Access iteration-level details

>>> est.nth_edges(10)
>>> est.resampled_data(10)

```

### References

- Original Issue: https://github.com/pgmpy/pgmpy/issues/3326
- Reference Paper: https://arxiv.org/pdf/1301.6695
