"""
Proof of Concept (POC): DataFrame-Agnostic Causal Discovery via Narwhals
Target: Chow-Liu

Starting with BaseCausalDiscovery ?
    As this the class that is the base class that every causal discovery algorithm inherits

"""

import warnings
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator
from sklearn.metrics import mutual_info_score
from sklearn.utils.validation import validate_data

import narwhals.stable.v2 as nw
from pgmpy.base import DAG


class BaseCausalDiscovery(BaseEstimator):

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.categorical = True
        tags.input_tags.allow_nan = False
        tags.target_tags.required = False
        return tags

    def _check_fit_data(self, X) -> nw.DataFrame:
        """
        DataFrame-agnostic validation via Narwhals.
        Accepts: pandas, Polars, PyArrow, Modin, cuDF, or 2D numpy arrays.
        """
        # Run scikit-learn validate_data for NumPy and Pandas inputs
        if isinstance(X, (np.ndarray, pd.DataFrame)):
            validate_data(
                self,
                X=X,
                dtype=None,
                accept_sparse=False,
                ensure_all_finite=True,
                reset=True,
            )
            if isinstance(X, np.ndarray):
                cols = [f"x{i}" for i in range(X.shape[1])]
                X = pd.DataFrame(X, columns=cols)

        nw_df = X if isinstance(X, nw.DataFrame) else nw.from_native(X, eager_only=True)

        n_samples, n_features = nw_df.shape
        if n_features < 1 or n_samples < 2:
            raise ValueError(
                f"Invalid shape: {nw_df.shape}. Expected at least 2 samples and 1 feature."
            )

        for col in nw_df.columns:
            s = nw_df.get_column(col)
            null_count = s.is_null().sum()
            try:
                nan_count = s.is_nan().sum() if s.dtype.is_float() else 0
            except (TypeError, AttributeError):
                nan_count = 0
            if (null_count + nan_count) > 0:
                raise ValueError(f"Input contains NaN/null values in column '{col}'.")

        self.feature_names_in_ = np.asarray(nw_df.columns, dtype=object)
        self.n_features_in_ = n_features
        return nw_df

    def fit(self, X, y=None):
        """
        Public fit entry point. Validates data, runs zero-variance check,
        and delegates to the algorithm's _fit implementation.
        """
        nw_df = self._check_fit_data(X)

        for col in nw_df.columns:
            if nw_df.get_column(col).n_unique() == 1:
                warnings.warn(
                    f"Variable '{col}' is constant (zero variance).",
                    UserWarning,
                    stacklevel=2,
                )

        return self._fit(nw_df)


class _TreeSearchMixin:
    """
    Mixin providing tree-search algorithms (ChowLiu and TAN) with shared
    mutual information weighting and MST graph construction.
    """

    @staticmethod
    def _get_weights(data: nw.DataFrame, edge_weights_fn=mutual_info_score):
        """Compute pairwise edge weights via Narwhals column extraction."""
        cols = data.columns
        n_vars = len(cols)
        weights = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            u_vec = data.get_column(cols[i]).to_numpy()
            for j in range(i + 1, n_vars):
                v_vec = data.get_column(cols[j]).to_numpy()
                w = edge_weights_fn(u_vec, v_vec)
                weights[i, j] = weights[j, i] = w
        return weights

    @staticmethod
    def _create_tree_and_dag(weights, columns, root_node):
        """Construct NetworkX MST directly without pandas adjacency."""
        n_vars = len(columns)
        G = nx.Graph()
        G.add_nodes_from(columns)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                G.add_edge(columns[i], columns[j], weight=weights[i, j])

        mst = nx.maximum_spanning_tree(G)
        directed_tree = nx.bfs_tree(mst, root_node)
        return DAG(directed_tree)


class ChowLiuNew(_TreeSearchMixin, BaseCausalDiscovery):
    """
    Chow-Liu algorithm for learning tree-structured Bayesian networks.
    Directly inherits from _TreeSearchMixin and BaseCausalDiscovery.
    """

    def __init__(self, root_node=None):
        self.root_node = root_node

    def _fit(self, nw_df: nw.DataFrame):
        columns = nw_df.columns

        weights = self._get_weights(nw_df)

        root_node = self.root_node
        if root_node is None:
            root_node = columns[np.argmax(weights.sum(axis=0))]

        self.causal_graph_ = self._create_tree_and_dag(weights, columns, root_node)
        return self
