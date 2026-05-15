"""
Prototype implementation of the _SimulationMixin and LinearGaussianSCM simulator.

This file demonstrates how a concrete simulator dataset class would look
using the proposed _SimulationMixin architecture. It is included alongside
the proposal for reference and is not intended to be merged as-is into pgmpy.

The mixin overrides load_dataframe() and load_ground_truth() so that
load_dataset() works polymorphically — no if-else branching needed.
Both methods call _simulate() directly. Since _simulate() is deterministic
for a given seed, both calls produce consistent results without shared state.

See proposal.md for the full design discussion.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from pgmpy.base import DAG, ADMG, MAG, PDAG
from pgmpy.datasets._base import _BaseDataset

# MAG is included for completeness; PAG does not exist in pgmpy yet.
# Current simulators will return DAG, PDAG, or ADMG.
CausalGraph = Union[DAG, PDAG, ADMG, MAG]


# ---------------------------------------------------------------------------
# This mixin would live in pgmpy/datasets/_base.py alongside _CovarianceMixin
# ---------------------------------------------------------------------------


class _SimulationMixin:
    """
    Mixin for datasets where data is generated on-the-fly via a simulation
    process. Overrides ``load_dataframe`` and ``load_ground_truth`` so that
    ``load_dataset()`` works uniformly across static and simulated datasets.

    Subclasses must implement ``_simulate(**kwargs) -> tuple[pd.DataFrame, CausalGraph]``.

    Both ``load_dataframe()`` and ``load_ground_truth()`` call ``_simulate()``
    independently. Since ``_simulate()`` is deterministic for a given seed,
    both calls produce consistent results without any shared mutable state.

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

    @classmethod
    def load_dataframe(cls, n_samples=None, seed=None, **sim_kwargs) -> pd.DataFrame:
        """Override of _BaseDataset.load_dataframe. Generates data via _simulate()."""
        data, _ = cls._simulate(n_samples=n_samples, seed=seed, **sim_kwargs)
        return data

    @classmethod
    def load_ground_truth(cls, n_samples=None, seed=None, **sim_kwargs) -> CausalGraph:
        """Override of _BaseDataset.load_ground_truth. Returns the graph from _simulate()."""
        _, gt = cls._simulate(n_samples=n_samples, seed=seed, **sim_kwargs)
        return gt


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
        "sim_params": {
            "n_nodes": {"default": 5, "desc": "Number of variables in the DAG"},
            "edge_prob": {"default": 0.3, "desc": "Probability of edge between any two nodes"},
            "noise_scale": {"default": 1.0, "desc": "Std dev of additive Gaussian noise"},
        },
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
    # Test 1: load_dataframe and load_ground_truth produce consistent results
    df = LinearGaussianSCM.load_dataframe(n_samples=100, seed=42, n_nodes=6)
    dag = LinearGaussianSCM.load_ground_truth(n_samples=100, seed=42, n_nodes=6)
    print(f"Consistent call: {df.shape[0]} samples, {df.shape[1]} variables, {dag.number_of_edges()} edges")
    assert df.shape[1] == 6, f"Expected 6 variables, got {df.shape[1]}"
    assert df.shape[0] == 100, f"Expected 100 samples, got {df.shape[0]}"

    # Test 2: Different kwargs produce different results (no stale state)
    df2 = LinearGaussianSCM.load_dataframe(n_samples=200, seed=7, n_nodes=4)
    dag2 = LinearGaussianSCM.load_ground_truth(n_samples=200, seed=7, n_nodes=4)
    print(f"Different kwargs: {df2.shape[0]} samples, {df2.shape[1]} variables, {dag2.number_of_edges()} edges")
    assert df.shape != df2.shape, "Different kwargs should produce different results"

    # Test 3: Reproducibility (same seed = same output)
    df3 = LinearGaussianSCM.load_dataframe(n_samples=100, seed=42, n_nodes=6)
    dag3 = LinearGaussianSCM.load_ground_truth(n_samples=100, seed=42, n_nodes=6)
    assert df.equals(df3), "Reproducibility failed: DataFrames differ"
    assert set(dag.edges()) == set(dag3.edges()), "Reproducibility failed: DAGs differ"
    print("Reproducibility: OK")

    # Test 4: Ground truth type
    assert isinstance(dag, DAG), f"Expected DAG, got {type(dag)}"
    print(f"Ground truth type: {type(dag).__name__}")
