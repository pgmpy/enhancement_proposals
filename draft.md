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

Step 3: Compute the stability score for each feature of interest (e.g., edge existence or edge orientation).

Step 4: Repeat the process for M bootstrap iterations.

Step 5: Calculate the mean stability score across all bootstrap iterations for each feature of interest.

## Consensus Graph Construction

To construct a final consensus graph, we apply a threshold to the estimated stability scores. However, naive thresholding introduces a challenge: even if every bootstrap graph is a valid Directed Acyclic Graph (DAG), independently aggregating highly frequent edges can create cycles.

To address this and construct a valid consensus DAG, the following consensus methods can be used:

### 1. Greedy Edge-Addition (Kruskal-style)

This approach ensures that the highest confidence edges survive, subject to DAG constraints.
- Sort all candidate edges in descending order by their stability score.
- Iteratively add edges one-by-one to the final graph.
- If adding an edge creates a cycle, it is skipped.

### 2. Direction Thresholding Separately

This method decouples the existence of a causal link from its specific orientation.
- First, compute the stability score for the *existence* of an edge between two nodes (ignoring its direction). Apply a threshold to form a consensus skeleton.
- Second, for the edges that survive, compare the relative frequency of the orientations. Assign the direction that was most frequent across the bootstrap iterations.
- *(Note: Cycle resolution steps may still be required if the resulting orientations create a cycle).*

## Architecture

The bootstrap framework constucted using two main classes:

- `BootstrapEstimator`
- `TabularBootstrap`

`BootstrapEstimator` is the main entry point responsible for:
- bootstrap estimation,
- scoring,
- and consensus graph construction.

`TabularBootstrap` is a result container class responsible for:
- storing iteration-level graph information,
- aggregated bootstrap statistics,
- and analysis utilities.

This separation keeps the estimation logic independent from the result representation layer.


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

        threshold:
            Default confidence threshold used during consensus graph construction.

        return_type:
            Type of graph returned by the estimator.
            Inferred automatically from the estimator instance.

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
        3. Store iteration-level graph information.
        4. Compute stability statistics.

        Returns
        -------
        self

        Attributes Created
        ------------------
        results_:
            Instance of TabularBootstrap containing bootstrap statistics.

        graph_:
            Final consensus graph.

        adjacency_matrix_:
            Adjacency matrix representation of the consensus graph.
        """

    def _resample(self, data):
        """
        Generate a bootstrap resampled dataset.
        """

    def _score(self):
        """
        Compute bootstrap stability statistics.
        """

    def consensus_graph_(self, threshold=None, technique=None):
        """
        Construct a valid consensus graph.

        Parameters
        ----------
        threshold:
            Confidence threshold used for edge selection.
            Defaults to self.threshold.

        technique:
            Consensus graph construction strategy.
        """

    def adjacency_matrix_(self, threshold=None, technique=None):
        """
        Return adjacency matrix representation of the consensus graph.
        """

    @property
    def results_(self):
        """
        Return the TabularBootstrap result container.
        """

class TabularBootstrap:
    """
    Result container for bootstrap estimation.

    Stores iteration-level graph information, aggregated stability
    statistics, and utilities for structural analysis.
    """

    # --------------------------------------------------
    # Aggregated Statistics
    # --------------------------------------------------

    def edge_scores_(self):
        """
        Return empirical edge stability scores.

        Returns
        -------
        dict or DataFrame
            Edge existence probabilities estimated across
            bootstrap iterations.
        """

    def direction_scores_(self):
        """
        Return edge orientation stability statistics.

        Returns
        -------
        dict or DataFrame
            Orientation frequencies estimated across
            bootstrap iterations.
        """

    def bootstrap_table_(self):
        """
        Return iteration-level structural summary table.

        Example
        -------
        | iter | AB | BC | CD |
        | ---- | -- | -- | -- |
        | 1    | 1  | 0  | 1  |
        | 2    | 1  | 1  | 0  |
        | 3    | 0  | 1  | 1  |
        """

    # --------------------------------------------------
    # Iteration-Level Access
    # --------------------------------------------------

    def nth_graph_(self, n):
        """
        Return the graph generated during the nth
        bootstrap iteration.
        """

    def resample_data_(self, n):
        """
        Reconstruct and return the bootstrap dataset used
        during the nth bootstrap iteration.

        Notes
        -----
        Only sampled row indices are stored internally
        for memory efficiency.
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

# Construct consensus graph

>>> est.consensus_graph_(
        threshold=0.8,
        technique="greedy",
    )

# Adjacency matrix representation

>>> est.adjacency_matrix_(
        threshold=0.8,
        technique="greedy",
    )

# Access bootstrap results

>>> results = est.results_

# Aggregated statistics

>>> results.edge_scores_()
>>> results.direction_scores_()
>>> results.bootstrap_table_()

# Iteration-level access

>>> results.nth_graph_(10)
>>> results.resample_data_(10)

```

### References

- Original Issue: https://github.com/pgmpy/pgmpy/issues/3326
- Reference Paper: https://arxiv.org/pdf/1301.6695
