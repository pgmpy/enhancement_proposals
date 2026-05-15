# CASTLE: Implementation Details

**Algorithm steps:**
1. Formulate a supervised prediction task for a target variable.
2. Initialize an internal masked-autoencoder network (`_CASTLEModel`) that prevents any feature from causing itself.
3. Optimize the joint objective minimizing supervised loss (MSE), data reconstruction loss, and continuous acyclicity penalty.
4. Calculate the weighted adjacency matrix $W$ from the input-layer weights of the internal network.
5. Add a DAG acyclicity penalty $h(W) = tr(e^{W \odot W}) - d = 0$.
6. Final inference zeroes out entries below a threshold `w_threshold` to return the resulting causal DAG.

**Key design decisions:**
* Implementation includes an internal pytorch model `_CASTLEModel`.
* Uses `torch.linalg.matrix_exp` to perform trace calculation for DAG constraints efficiently.
* Provides a mechanism for handling both unregularised predictive inputs and graph discovery.

**API:**
```python
from pgmpy.causal_discovery.base import _BaseCausalDiscovery
import pandas as pd
import numpy as np

class CASTLE(_BaseCausalDiscovery):
    def __init__(
        self,
        reg_lambda: float = 1.0,
        reg_beta: float = 5.0,
        rho: float = 1.0,
        lr: float = 0.001,
        batch_size: int = 32,
        n_hidden: int = 32,
        w_threshold: float = 0.3,
        max_epochs: int = 200,
        random_state: int | None = None,
    ):
        ...

    def fit(self, X: pd.DataFrame, y=None, **kwargs) -> "CASTLE":
        ...

    def _fit(self, X: pd.DataFrame, target_col=None) -> "CASTLE":
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    def get_adjacency_matrix(self) -> pd.DataFrame:
        ...

    def get_dag(self) -> "DAG":
        ...
```

**Usage:**
```python
from pgmpy.causal_discovery import CASTLE
import pandas as pd

df = pd.read_csv("data.csv")
model = CASTLE(max_epochs=200, random_state=42)
model.fit(df, target_col="Target")
print(model.causal_graph_.edges())
```

**Tests (`pgmpy/tests/test_causaldiscovery/test_castle.py`):** Synthetic DAGs (e.g. from bnlearn simulating the Asia network) will be generated. Tests will assert whether the learned DAG captures the structural relationships accurately according to the simulated DAG, and checks the predictive MSE on unseen data.
