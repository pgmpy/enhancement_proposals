"""
Prototype implementation of the _SimulationMixin and LinearGaussianSCM simulator.

This file demonstrates how a concrete simulator dataset class would look
using the proposed _SimulationMixin architecture. It is included alongside
the proposal for reference and is not intended to be merged as-is into pgmpy.

The mixin declares the contract: concrete classes must override
load_dataframe() and load_ground_truth(). How each class structures
its internals is up to it — the mixin doesn't enforce any particular
decomposition.

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
    Mixin for simulated datasets. Concrete classes must override
    ``load_dataframe()`` and ``load_ground_truth()`` with their own
    simulation logic.

    When using this mixin, it should be the first parent class so that its
    methods take precedence in the MRO (same convention as
    ``_CovarianceMixin``).
    """

    @classmethod
    def load_dataframe(cls, n_samples=None, seed=None, **sim_kwargs) -> pd.DataFrame:
        """Generate and return simulated data. Must be implemented by each simulator."""
        raise NotImplementedError(f"{cls.__name__} must implement load_dataframe().")

    @classmethod
    def load_ground_truth(cls, n_samples=None, seed=None, **sim_kwargs) -> CausalGraph:
        """Construct and return the ground-truth graph. Must be implemented by each simulator."""
        raise NotImplementedError(f"{cls.__name__} must implement load_ground_truth().")


# ---------------------------------------------------------------------------
# This concrete class would live in pgmpy/datasets/linear_gaussian_scm.py
# ---------------------------------------------------------------------------


class LinearGaussianSCM(_SimulationMixin, _BaseDataset):
    """
    Simulator for linear Gaussian structural causal models.

    Generates a random LGBN model (graph + CPDs) via a shared _build_model()
    helper, then extracts the DAG or simulates data from it.
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
    def _build_model(cls, seed=None, n_nodes=5, edge_prob=0.3, noise_scale=1.0, **kwargs):
        """Shared helper: builds the LGBN model (graph + CPDs)."""
        from pgmpy.models import LinearGaussianBayesianNetwork as LGBN

        return LGBN.get_random(
            n_nodes=n_nodes, edge_prob=edge_prob, scale=noise_scale, seed=seed
        )

    @classmethod
    def load_ground_truth(cls, n_samples=None, seed=None, n_nodes=5, edge_prob=0.3, **kwargs):
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob, **kwargs)
        return DAG(model.edges())

    @classmethod
    def load_dataframe(cls, n_samples=1000, seed=None, n_nodes=5,
                       edge_prob=0.3, noise_scale=1.0, **kwargs):
        model = cls._build_model(
            seed=seed, n_nodes=n_nodes, edge_prob=edge_prob,
            noise_scale=noise_scale, **kwargs
        )
        return model.simulate(n_samples=n_samples, seed=seed)


# ---------------------------------------------------------------------------
# Quick smoke test (run with: python prototype_linear_gaussian_scm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: load_dataframe and load_ground_truth produce consistent results
    df = LinearGaussianSCM.load_dataframe(n_samples=100, seed=42, n_nodes=6)
    dag = LinearGaussianSCM.load_ground_truth(seed=42, n_nodes=6)
    print(f"Consistent call: {df.shape[0]} samples, {df.shape[1]} variables, {dag.number_of_edges()} edges")
    assert df.shape[1] == 6, f"Expected 6 variables, got {df.shape[1]}"
    assert df.shape[0] == 100, f"Expected 100 samples, got {df.shape[0]}"
    assert set(df.columns) == set(dag.nodes()), "Column names should match DAG nodes"

    # Test 2: Different kwargs produce different results
    df2 = LinearGaussianSCM.load_dataframe(n_samples=200, seed=7, n_nodes=4)
    dag2 = LinearGaussianSCM.load_ground_truth(seed=7, n_nodes=4)
    print(f"Different kwargs: {df2.shape[0]} samples, {df2.shape[1]} variables, {dag2.number_of_edges()} edges")
    assert df.shape != df2.shape, "Different kwargs should produce different results"

    # Test 3: Reproducibility (same seed = same output)
    df3 = LinearGaussianSCM.load_dataframe(n_samples=100, seed=42, n_nodes=6)
    dag3 = LinearGaussianSCM.load_ground_truth(seed=42, n_nodes=6)
    assert df.equals(df3), "Reproducibility failed: DataFrames differ"
    assert set(dag.edges()) == set(dag3.edges()), "Reproducibility failed: DAGs differ"
    print("Reproducibility: OK")

    # Test 4: Ground truth type
    assert isinstance(dag, DAG), f"Expected DAG, got {type(dag)}"
    print(f"Ground truth type: {type(dag).__name__}")
