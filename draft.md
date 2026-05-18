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

## Scoring

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

## TabularBootstrap

This is going to be a result container class named `TabularBootstrap` which stores aggregated bootstrap statistics and iteration-level graph information generated during bootstrap estimation.

The class acts as a centralized interface for inspecting bootstrap-derived structural properties such as:

- edge stability scores,
- edge orientation frequencies,
- edge existence probabilities,
- bootstrap iteration summaries,
- consensus structural statistics.

After fitting the estimator, the bootstrap results can be accessed through:

```python
>>> est = BootstrapEstimator(
        estimator=HillClimbSearch(),
        n_resamples=10,
        sample_frac=0.8,
        random_state=42,
        n_jobs=-1,
    )

>>> est.fit(data)
>>> table = est.table_
```

The `TabularBootstrap` instance exposes user-facing attributes and methods for analyzing the stability of the discovered causal structure.

```python
>>> table.edge_scores_
```

Returns the empirical stability score of each edge across bootstrap iterations.

```python
>>> table.direction_scores_
```

Returns the orientation stability statistics for directed edges.
In addition to aggregated statistics, the class also stores iteration-level structural information. To maintain memory efficiency with large datasets, `TabularBootstrap` stores only the row indices used for each iteration rather than full data copies. The `resample_data(n)` method reconstructs the data on the fly:

```python
>>> table.bootstrap_table_
```

This returns a tabular representation where:

- each row corresponds to a bootstrap iteration,
- each column corresponds to a structural feature such as edge existence or orientation,
- each entry indicates whether the corresponding feature was present in that bootstrap graph.

Example:

| iter | AB | BC | CD |
| ---- | ---- | ---- | ---- |
| 1    | 1    | 0    | 1    |
| 2    | 1    | 1    | 0    |
| 3    | 0    | 1    | 1    |


The main idea behind `TabularBootstrap` is to provide a manageable interface for accessing iteration-level and aggregated bootstrap statistics from the fitted estimator.

This also separates the bootstrap estimation logic from the result representation and analysis utilities.

## API 

```python
BootstrapEstimator(
    estimator: Instance of the causal discovery algorithm,
    n_resamples: Number of bootstrap iterations,
    sample_frac: Fraction of samples to use in each bootstrap dataset (e.g., 0.8 for 80%),
    threshold: Default confidence threshold for stability scores,
    return_type: Resultant graph type,
    random_state: Random seed for reproducibility,
    n_jobs: Number of parallel processes to use
)
```

## Class Skeleton and functions

```python
class BootstrapEstimator:

    def __init__(...):
        ...

    def fit(self, data):
        """
        Functionality
        -------------

        1. Generate bootstrap datasets using `_resample`.
        2. Apply the causal discovery algorithm on each bootstrap dataset.
        3. Store iteration-level structural information in `TabularBootstrap`.
        4. Repeat for `n_resamples` iterations.
        5. Compute structural stability statistics using `_score`.
        6. Construct the final valid causal graph using `consensus_graph`.

        Returns
        -------

        Self object exposing:
        - table_: TabularBootstrap object
        - graph_: Final consensus causal graph
        - adjacency_matrix_: Adjacency matrix representation

        """

    def _score(self):
        """
        Compute edge and orientation stability statistics from bootstrap iterations.
        """

    def _resample(self, data):
        """
        Generate a bootstrap resampled dataset.
        """

    def consensus_graph(self, threshold=None):
        """
        Construct and return a valid consensus graph using the computed bootstrap stability scores.
        If threshold is not provided, uses the default threshold set during initialization.
        This allows users to dynamically experiment with different thresholds without refitting.
        """
```

## Example

```python
>>> from pgmpy.estimators import BootstrapEstimator
>>> from pgmpy.causal_discovery import HillClimbSearch

>>> est = BootstrapEstimator(
        estimator=HillClimbSearch(),
        n_resamples=10,
        sample_frac=0.8, # 80% of the original dataset
        threshold=0.8,
        return_type='pdag',
        random_state=42,
        n_jobs=-1
    )

>>> est.fit(data)

# Final consensus structure

>>> est.graph_
>>> est.adjacency_matrix_

# Access bootstrap statistics

>>> table = est.table_
>>> table.edge_scores_
>>> table.direction_scores_
>>> table.bootstrap_table_

# Access iteration-level information

>>> table.nth_graph(n)
>>> table.resample_data(n) # Reconstructs dataset on the fly using stored indices

# Dynamically generate consensus graphs with different thresholds
>>> strict_graph = est.consensus_graph(threshold=0.9)
```

### References

- Original Issue: https://github.com/pgmpy/pgmpy/issues/3326
- Reference Paper: https://arxiv.org/pdf/1301.6695
