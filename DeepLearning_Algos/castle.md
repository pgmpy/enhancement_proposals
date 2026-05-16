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

## API

### `_CASTLEModel(nn.Module)` (Internal)
PyTorch masked autoencoder for feature reconstruction and target prediction.
* **`__init__(num_inputs, n_hidden)`**: Initializes masked input layers and scalar output layers.
* **`forward(X)`**: Returns full reconstruction (`Out`) and target prediction (`out_0`).
* **`get_W()`**: Computes adjacency matrix from current weights (L2 norm of column $j$ in sub-network $i$).

### `_CASTLETrainer` (Internal)
Handles the training loop and Augmented Lagrangian updates.
* **`__init__(...)`**: Configures hyperparameters (`rho`, `reg_lambda`, `reg_beta`, `w_threshold`) and optimizer.
* **`train(X_tensor)`**: Runs epochs (batch updates, Lagrangian multiplier adjustments) and returns the thresholded adjacency matrix (`W_final`).

### `CASTLE(_BaseCausalDiscovery)` (Public API)
Validates data, orchestrates training, and builds the `pgmpy.DAG`.
* **`__init__(reg_lambda=1.0, reg_beta=5.0, rho=1.0, lr=0.001, batch_size=32, n_hidden=32, w_threshold=0.3, max_epochs=200, random_state=None)`**
* **`fit(X, target_col=None)`**: Trains the model and builds the DAG. `target_col` defaults to the first column.
* **`predict(X)`**: Predicts the target column using the learned sub-network.
* **`get_dag()`**: Returns the learned causal graph as a `pgmpy.base.DAG`.
* **`get_adjacency_matrix()`**: Returns a Pandas DataFrame of the edge weights.

## Usage

```python
from pgmpy.causal_discovery import CASTLE
import pandas as pd
import numpy as np

# Load data
df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'Target'])

# Initialize and fit
model = CASTLE(max_epochs=200, w_threshold=0.3, random_state=42)
model.fit(df, target_col="Target")

# Access results
dag = model.get_dag()
print("Edges:", dag.edges())

# Predict
preds = model.predict(df[['A', 'B']])
```

