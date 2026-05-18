"""
Prototype implementation of the _SimulationMixin and LinearGaussianSCM simulator.

This file demonstrates how a concrete simulator dataset class would look
using the proposed _SimulationMixin architecture. It is included alongside
the proposal for reference and is not intended to be merged as-is into pgmpy.

See proposal.md for the full design discussion.
"""

# type: ignore

from __future__ import annotations

from typing import Union

import pandas as pd

from pgmpy.base import DAG, ADMG, MAG, PDAG
from pgmpy.datasets._base import _BaseDataset  # type: ignore

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
    def load_ground_truth(cls, **sim_kwargs) -> CausalGraph:
        """Construct and return the ground-truth graph. Must be implemented by each simulator."""
        raise NotImplementedError(f"{cls.__name__} must implement load_ground_truth().")


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

    _tags: dict = {
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
            n_nodes=n_nodes,
            edge_prob=edge_prob,
            scale=noise_scale,
            seed=seed,
        )

    @classmethod
    def load_ground_truth(cls, seed=None, n_nodes=5, edge_prob=0.3, **kwargs):
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob, **kwargs)
        raw_edges = model.edges()
        edges = [(str(e[0]), str(e[1])) for e in raw_edges]
        return DAG(edges)

    @classmethod
    def load_dataframe(cls, n_samples=1000, seed=None, n_nodes=5,
                       edge_prob=0.3, noise_scale=1.0, **kwargs):
        model = cls._build_model(seed=seed, n_nodes=n_nodes, edge_prob=edge_prob,
                                  noise_scale=noise_scale, **kwargs)
        actual_samples = 1000 if n_samples is None else n_samples
        return model.simulate(n_samples=actual_samples, seed=seed)


# ---------------------------------------------------------------------------
# Quick smoke test (run with: python prototype_linear_gaussian_scm.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default simulation
    df = LinearGaussianSCM.load_dataframe()
    dag = LinearGaussianSCM.load_ground_truth()
    print(f"Default: {df.shape[0]} samples, {df.shape[1]} variables, {dag.number_of_edges()} edges")

    # Custom hyperparameters
    df = LinearGaussianSCM.load_dataframe(n_samples=500, seed=42, n_nodes=8, edge_prob=0.4)
    dag = LinearGaussianSCM.load_ground_truth(seed=42, n_nodes=8, edge_prob=0.4)
    print(f"Custom:  {df.shape[0]} samples, {df.shape[1]} variables, {dag.number_of_edges()} edges")

    # Reproducibility check
    df1 = LinearGaussianSCM.load_dataframe(n_samples=100, seed=42)
    df2 = LinearGaussianSCM.load_dataframe(n_samples=100, seed=42)
    dag1 = LinearGaussianSCM.load_ground_truth(seed=42)
    dag2 = LinearGaussianSCM.load_ground_truth(seed=42)
    assert df1.equals(df2), "Reproducibility failed: DataFrames differ"
    assert set(dag1.edges()) == set(dag2.edges()), "Reproducibility failed: DAGs differ"
    print("Reproducibility: OK")

    # Verify ground truth type
    assert isinstance(dag, DAG), f"Expected DAG, got {type(dag)}"
    print(f"Ground truth type: {type(dag).__name__}")
