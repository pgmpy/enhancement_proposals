"""
Prototype implementation of the _SimulationMixin and LinearGaussianSCM simulator.

This file demonstrates how a concrete simulator dataset class would look
using the proposed _SimulationMixin architecture. It is included alongside
the proposal for reference and is not intended to be merged as-is into pgmpy.

Changes from v1 (based on @ankurankan's review):
  - Dropped class-level caching. _simulate() is now a pure function.
  - Ground truth type widened from DAG to CausalGraph (Union of DAG, PDAG, ADMG, MAG).
  - n_samples and seed are explicit parameters (not buried in **kwargs).

See proposal.md for the full design discussion.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from pgmpy.base import DAG, ADMG, MAG, PDAG
from pgmpy.datasets._base import _BaseDataset

CausalGraph = Union[DAG, PDAG, ADMG, MAG]


# ---------------------------------------------------------------------------
# This mixin would live in pgmpy/datasets/_base.py alongside _CovarianceMixin
# ---------------------------------------------------------------------------


class _SimulationMixin:
    """
    Mixin for datasets where data is generated on-the-fly via a simulation
    process. Subclasses must implement ``_simulate()``.

    This mixin is stateless. It does not cache simulation results.
    ``load_dataset()`` is the intended entry point: it calls ``_simulate()``
    once and passes both the data and ground-truth graph directly to the
    ``Dataset`` constructor.

    If users call ``_simulate()`` directly, they are responsible for passing
    the same ``seed`` to get reproducible results.

    When using this mixin, it should be the first parent class so that its
    methods take precedence in the MRO (same convention as
    ``_CovarianceMixin``).
    """

    @classmethod
    def _simulate(
        cls,
        n_samples: int | None = None,
        seed: int | None = None,
        **sim_kwargs,
    ) -> tuple[pd.DataFrame, CausalGraph]:
        """
        Generate simulated data and the ground-truth causal graph.

        Must be implemented by each simulator dataset class.

        Parameters
        ----------
        n_samples : int or None
            Number of samples to generate.
        seed : int or None
            Seed for reproducibility.
        **sim_kwargs
            Simulator-specific hyperparameters.

        Returns
        -------
        tuple[pd.DataFrame, CausalGraph]
            (data, ground_truth_graph)
        """
        raise NotImplementedError(f"{cls.__name__} must implement _simulate().")


# ---------------------------------------------------------------------------
# This concrete class would live in pgmpy/datasets/linear_gaussian_scm.py
# ---------------------------------------------------------------------------


class LinearGaussianSCM(_SimulationMixin, _BaseDataset):
    """
    Simulator for linear Gaussian structural causal models.

    Generates a random DAG with Gaussian noise and samples from it.
    Uses ``LinearGaussianBayesianNetwork.get_random()`` to build the model
    and ``.simulate()`` to draw samples.
    """

    _tags = {
        "name": "linear_gaussian_scm",
        "n_variables": None,
        "n_samples": None,
        "has_ground_truth": True,
        "has_expert_knowledge": False,
        "has_missing_data": False,
        "has_index_col": False,
        "is_simulated": True,
        "is_interventional": False,
        "is_discrete": False,
        "is_continuous": True,
        "is_mixed": False,
        "is_ordinal": False,
    }

    @classmethod
    def _simulate(
        cls,
        n_samples: int | None = 1000,
        seed: int | None = None,
        n_nodes: int = 5,
        edge_prob: float = 0.3,
        noise_scale: float = 1.0,
        **kwargs,
    ) -> tuple[pd.DataFrame, DAG]:
        from pgmpy.models import LinearGaussianBayesianNetwork as LGBN

        model = LGBN.get_random(
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            scale=noise_scale,
            seed=seed,
        )
        data = model.simulate(n_samples=n_samples, seed=seed)
        ground_truth = DAG(model.edges())

        return data, ground_truth


# ---------------------------------------------------------------------------
# Quick smoke test (run with: python prototype_linear_gaussian_scm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default simulation
    df, dag = LinearGaussianSCM._simulate()
    print(f"Default: {df.shape[0]} samples, {df.shape[1]} variables, {dag.number_of_edges()} edges")

    # Custom hyperparameters
    df, dag = LinearGaussianSCM._simulate(n_samples=500, seed=42, n_nodes=8, edge_prob=0.4)
    print(f"Custom:  {df.shape[0]} samples, {df.shape[1]} variables, {dag.number_of_edges()} edges")

    # Reproducibility check (stateless: same seed = same output)
    df1, dag1 = LinearGaussianSCM._simulate(n_samples=100, seed=42)
    df2, dag2 = LinearGaussianSCM._simulate(n_samples=100, seed=42)
    assert df1.equals(df2), "Reproducibility failed: DataFrames differ"
    assert set(dag1.edges()) == set(dag2.edges()), "Reproducibility failed: DAGs differ"
    print("Reproducibility: OK")

    # Verify ground truth type
    assert isinstance(dag, DAG), f"Expected DAG, got {type(dag)}"
    print(f"Ground truth type: {type(dag).__name__}")
