# GraN-DAG: Implementation Details

**Algorithm steps:**
1. For each variable $X_i$, parameterize $p(X_i|X_{-i})$ using a separate MLP.
2. The weighted adjacency matrix $W$ is obtained from the first-layer weights of the MLPs via a masking scheme.
3. Optimize the negative log-likelihood of the data subject to the continuous acyclicity constraint.
4. Use an augmented Lagrangian method to turn the hard constraint into a soft penalty.
5. After convergence, threshold the learned adjacency matrix to recover the DAG skeleton.
6. Optionally apply a Preliminary Neighborhood Selection (PNS) step.

**API:**
```python
from pgmpy.causal_discovery.base import _BaseCausalDiscovery
import pandas as pd
import numpy as np
import torch

class GraNDAG(_BaseCausalDiscovery):
    def __init__(
        self,
        hidden_dim: int = 16,
        n_layers: int = 2,
        dist_type: str = "gauss",
        lr: float = 1e-3,
        iterations: int = 25000,
        pns: bool = False,
        pns_thresh: float = 0.75,
        lambda_init: float = 0.0,
        mu_init: float = 1e-3,
        omega_lambda: float = 1e-4,
        omega_mu: float = 0.9,
        h_threshold: float = 1e-8,
        edge_clamp_range: float = 1e-4,
        device: str = "cpu",
    ):
        ...

    def fit(self, X: pd.DataFrame) -> "GraNDAG":
        ...

    # Internal methods
    def _build_model(self, n_nodes: int): ...
    def _compute_h(self, W: torch.Tensor) -> torch.Tensor: ... 
    def _augmented_lagrangian_step(self, X, lam, mu): ...
    def _threshold_graph(self, W: np.ndarray) -> np.ndarray: ...
    def _to_dag(self, adj: np.ndarray, nodes: list) -> "DAG": ...
```

**Tests (`pgmpy/tests/test_causaldiscovery/test_grandag.py`):** Synthetic DAGs with linear-Gaussian and non-linear (MLP-generated) SCMs. SHD (Structural Hamming Distance) is checked to be below a permissive threshold on small graphs.
