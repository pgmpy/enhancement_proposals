"""
Prototype implementation of the LinearGaussianSCM simulator dataset.

This file demonstrates how a concrete simulator dataset class would look
using the proposed _SimulationMixin architecture. It is included alongside
the proposal for reference and is not intended to be merged as-is into pgmpy.

See proposal.md for the full design discussion.
"""

from __future__ import annotations

import pandas as pd

from pgmpy.base import DAG
from pgmpy.datasets._base import _BaseDataset


# ---------------------------------------------------------------------------
# This mixin would live in pgmpy/datasets/_base.py alongside _CovarianceMixin
# ---------------------------------------------------------------------------


class _SimulationMixin:
    """
    Mixin for datasets where data is generated on-the-fly via a simulation
    process.  Overrides ``load_dataframe`` and ``load_ground_truth`` to call
    the subclass's ``_simulate`` method.

    Subclasses must implement ``_simulate(**kwargs) -> tuple[pd.DataFrame, DAG]``.

    When using this mixin it should be the first parent class so that its
    ``load_dataframe`` and ``load_ground_truth`` take precedence in the MRO
    (same convention as ``_CovarianceMixin``).
    """

    _cached_data: pd.DataFrame | None = None
    _cached_ground_truth: DAG | None = None

    @classmethod
    def _simulate(cls, **kwargs) -> tuple[pd.DataFrame, DAG]:
        raise NotImplementedError(f"{cls.__name__} must implement _simulate().")

    @classmethod
    def load_dataframe(cls, **kwargs) -> pd.DataFrame:
        data, ground_truth = cls._simulate(**kwargs)
        cls._cached_data = data
        cls._cached_ground_truth = ground_truth
        return data

    @classmethod
    def load_ground_truth(cls) -> DAG:
        if cls._cached_ground_truth is None:
            cls.load_dataframe()
        return cls._cached_ground_truth


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
        n_samples: int = 1000,
        n_nodes: int = 5,
        edge_prob: float = 0.3,
        noise_scale: float = 1.0,
        seed: int | None = None,
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
    df, dag = LinearGaussianSCM._simulate(n_samples=500, n_nodes=8, edge_prob=0.4, seed=42)
    print(f"Custom:  {df.shape[0]} samples, {df.shape[1]} variables, {dag.number_of_edges()} edges")

    # Reproducibility check
    df1, dag1 = LinearGaussianSCM._simulate(n_samples=100, seed=42)
    df2, dag2 = LinearGaussianSCM._simulate(n_samples=100, seed=42)
    assert df1.equals(df2), "Reproducibility failed: DataFrames differ"
    assert set(dag1.edges()) == set(dag2.edges()), "Reproducibility failed: DAGs differ"
    print("Reproducibility: OK")

    # Mixin flow (load_dataframe -> load_ground_truth)
    data = LinearGaussianSCM.load_dataframe(n_samples=200, seed=7)
    gt = LinearGaussianSCM.load_ground_truth()
    assert data.shape[0] == 200
    assert gt is not None
    print(f"Mixin flow: {data.shape[0]} samples, ground_truth has {gt.number_of_edges()} edges")
