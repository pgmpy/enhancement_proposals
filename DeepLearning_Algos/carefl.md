# CAREFL: Implementation Details

**Algorithm steps:**
1. Fit an affine autoregressive flow (IAF/MAF) for each candidate causal ordering.
2. Compare log-likelihoods on a held-out validation set to assess causal direction.
3. For the multivariate case, search over causal orderings.
4. Optionally prune spurious edges after a skeleton is obtained.

**API:**
```python
from pgmpy.causal_discovery.base import _BaseCausalDiscovery
import pandas as pd
import numpy as np

class CAREFL(_BaseCausalDiscovery):
    def __init__(
        self,
        n_layers: int = 3,
        n_hidden: int = 100,
        epochs: int = 300,
        lr: float = 1e-3,
        batch_size: int = 256,
        flow_type: str = "affine",
        mode: str = "multivariate",
        alpha: float = 0.05,
        val_fraction: float = 0.2,
    ):
        ...

    def fit(self, X: pd.DataFrame) -> "CAREFL":
        ...

    # Internal methods
    def _build_flow(self, input_dim: int) -> "nn.Module": ...
    def _fit_flow(self, X: np.ndarray, Y: np.ndarray) -> float: ... 
    def _bivariate_direction(self, X, Y) -> tuple: ...
    def _multivariate_ordering(self, data: np.ndarray) -> list: ...
    def _to_dag(self, order: list, data: np.ndarray, nodes: list) -> "DAG": ...
```

**Tests (`pgmpy/tests/test_causaldiscovery/test_carefl.py`):** Bivariate tests use synthetic ANM pairs where the true direction is known. Multivariate tests use small synthetically generated DAGs. Correct direction recovery rate is asserted to exceed chance level.
