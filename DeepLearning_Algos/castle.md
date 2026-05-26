# CASTLE: Implementation Details

**Algorithm steps:**
1. Formulate a supervised prediction task for a target variable.
2. Initialize an internal masked-autoencoder network (`_CASTLEModel`) that prevents any feature from causing itself.
3. Optimize the joint objective minimizing supervised loss (MSE), data reconstruction loss, and continuous acyclicity penalty.
4. Calculate the weighted adjacency matrix $W$ from the input-layer weights of the internal network.
5. Add a DAG acyclicity penalty $h(W) = \text{tr}(e^{W \odot W}) - d = 0$.
6. Final inference zeroes out entries below a threshold to return the resulting causal DAG.

**Key design decisions:**
- Implementation includes an internal PyTorch model `_CASTLEModel`.
- Uses `torch.linalg.matrix_exp` to perform trace calculation for DAG constraints efficiently.
- Internal hyperparameters are grouped into dataclasses (`RegularizationConfig`, `TrainingConfig`, `NetworkConfig`) for readability — the public API signature is flat and unchanged.
- Accepts a PyTorch optimizer object directly; defaults to Adam internally.

---

## API

### `_CASTLEModel(nn.Module)` (Internal)
PyTorch masked autoencoder for feature reconstruction and target prediction.

- **`__init__(num_inputs, network_cfg: NetworkConfig)`**: Initializes masked input layers and scalar output layers.
- **`forward(X)`**: Returns full reconstruction (`Out`) and target prediction (`out_0`).
- **`get_W()`**: Computes adjacency matrix from current weights (L2 norm of column $j$ in sub-network $i$).

### `_CASTLETrainer` (Internal)
Handles the training loop and Augmented Lagrangian updates. Receives `RegularizationConfig` and `TrainingConfig` objects rather than individual parameters.

- **`__init__(model, train_cfg: TrainingConfig, reg_cfg: RegularizationConfig)`**: Configures grouped hyperparameters and optimizer.
- **`train(X_tensor)`**: Runs epochs (batch updates, Lagrangian multiplier adjustments) and returns the thresholded adjacency matrix (`W_final`).

### `CASTLE(_BaseCausalDiscovery)` (Public API)
Validates data, orchestrates training, and builds the `pgmpy.DAG`.

```python
CASTLE(
    dag_weight=1.0,
    sparsity_weight=5.0,
    dag_penalty=1.0,
    optimizer=None,
    batch_size=32,
    hidden_dim=32,
    edge_threshold=0.3,
    max_epochs=200,
    random_state=None,
)
```

Joint training objective being minimized:

$$\min_\Theta \frac{1}{N}\|Y - [f_\Theta(\tilde{X})]_{:,1}\|^2 + \lambda \underbrace{\left(L_N(f_\Theta) + (\text{tr}(e^{M \odot M}) - d - 1)^2 + \beta V_{\Theta_1}\right)}_{R_{\text{DAG}}}$$

- `dag_weight` (λ): Weight on the entire DAG regularization loss $R_{\text{DAG}}$.
- `sparsity_weight` (β): Group-lasso penalty on input-layer weights $V_{\Theta_1}$, promoting edge sparsity.
- `dag_penalty` (ρ): Initial penalty coefficient in the Augmented Lagrangian for the acyclicity constraint.
- `optimizer`: Any `torch.optim.Optimizer` instance. Defaults to `Adam(lr=1e-3)` if `None`. Note: `weight_decay` is not recommended when passing a custom optimizer as CASTLE has its own sparsity mechanism via `dag_weight` and `sparsity_weight`.
- `batch_size`: Number of samples per mini-batch.
- `hidden_dim` (h): Hidden layer width in each sub-network $f_k$.
- `edge_threshold`: Edges with weight below this value are zeroed in the final DAG.
- `max_epochs`: Maximum number of training epochs.
- `random_state`: Seed for reproducibility.

**Methods:**
- **`fit(X, target_col=None)`**: Trains the model and builds the DAG. `target_col` defaults to the first column.
- **`predict(X)`**: Predicts the target column using the learned sub-network.
- **`get_dag()`**: Returns the learned causal graph as a `pgmpy.base.DAG`.
- **`get_adjacency_matrix()`**: Returns a Pandas DataFrame of the edge weights.

---

## Usage

```python
import torch
from pgmpy.causal_discovery import CASTLE
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'Target'])

# Default: Adam with lr=1e-3
model = CASTLE(max_epochs=200, edge_threshold=0.3, random_state=42)

# Custom optimizer — user controls all optimizer params directly
opt = torch.optim.Adam(lr=1e-4, weight_decay=1e-5)
model = CASTLE(max_epochs=200, edge_threshold=0.3, optimizer=opt, random_state=42)

model.fit(df, target_col="Target")
dag = model.get_dag()
preds = model.predict(df[['A', 'B']])
```

---

## Test Plan

All tests use fixed `random_state=42` and small synthetic DAGs generated via `utils.gen_data_nonlinear` (mirroring the reference implementation) so results are deterministic and fast. A shared 5-node linear DAG fixture with 500 samples is the default dataset unless stated otherwise.

### 1. Basic Input / Output Tests

- **Fit returns self and sets attributes**: `CASTLE.fit(df)` returns the estimator instance; `causal_graph_`, `adjacency_matrix_`, `model_`, `scaler_`, `predictor_names_`, and `cols_` are all set after the call.
- **Output shapes are correct**: `adjacency_matrix_` is `(d, d)`; `predict(X_test)` returns a 1-D array of length `N`; `get_W()` inside `_CASTLEModel` returns `(d, d)`.
- **Target column selection**: Fit with `target_col` as a string name, as an integer index, and omitted (defaults to first column) — all three produce identical `cols_[0]`.
- **Invalid target raises `ValueError`**: Passing a column name not in `X` or an out-of-range integer index raises `ValueError`.
- **Predict column alignment**: `predict` called with a DataFrame whose columns match `predictor_names_` returns the same result as calling it with the equivalent numpy array in the same column order.
- **DataFrame and numpy inputs**: `fit` accepts both `pd.DataFrame` and validates that non-numeric input raises.
- **Single-column input is rejected gracefully**: A DataFrame with one column raises `ValueError` before any training begins, with a clear message.
- **Hyperparameter edge cases do not crash**: `max_epochs=1`, `batch_size=1`, `batch_size > N`, `hidden_dim=1` — all complete without error and produce a valid DAG.

### 2. Network Correctness Tests

- **Self-masking is enforced**: After initialisation and after training, `model_.get_W().diagonal()` is all zeros — no feature reconstructs itself.
- **Mask buffers survive a round-trip**: Save and reload `model_.state_dict()`; verify all `mask_{i}` buffers are unchanged (persistent buffer registration).
- **`forward` output shapes**: `Out.shape == (B, d)` and `out_0.shape == (B, 1)` for a random batch of size `B`.

### 3. DAG Validity Tests

- **Threshold is applied**: All entries in `adjacency_matrix_` are either zero or `≥ edge_threshold`; no values fall in `(0, edge_threshold)`.
- **Varying `edge_threshold` changes graph density monotonically**: Fit once; apply three thresholds `[0.1, 0.3, 0.5]` to `adjacency_matrix_`; verify edge count is non-increasing.

### 4. Additional Tests

- **`_CASTLEModel` and `_CASTLETrainer` are independently instantiable**: Both internal classes can be constructed and used without going through `CASTLE.fit`, confirming the three-class separation holds at the unit level.
- **Scaler is fit on training data only**: `scaler_.mean_` and `scaler_.scale_` match those of a separately fitted `StandardScaler` on the same training split, confirming test data has not leaked into the scaler.
- **Missing `torch` raises a clean error**: Importing `CASTLE` without `torch` installed raises `ImportError` with an actionable install message, not a bare `ModuleNotFoundError`.
- **`fit` → `predict` is stateless across calls**: Calling `predict` twice on the same input returns identical results — no stochastic behaviour during inference.

### 5. Benchmarking Tests

> These tests require sufficient data and epochs to observe meaningful trends. Run separately from the main test suite.

- **Supervised loss decreases**: Record supervised loss at epoch 1 and epoch `max_epochs`; assert final < initial.
- **Reconstruction loss decreases**: Same check on the MSE reconstruction term across epochs.
- **`h(W)` trends downward within a tolerance**: Record `_dag_constraint(model_.get_W())` at epoch 1 and final epoch. Assert decrease of at least 50% from initial value. An absolute floor of `h < 0.5` is logged as a soft check but not a hard failure.
- **`dag_penalty` doubles correctly**: When `h(W)` fails to decrease by 75% between epochs, the trainer's internal `dag_penalty` doubles. Tested by patching the `h` value returned inside the epoch loop to a fixed constant.
- **`sparsity_weight` drives edge sparsity**: Train with high `sparsity_weight` (e.g. 50) and low (e.g. 0.1); assert that high `sparsity_weight` yields strictly fewer non-zero entries in `W_final`.
- **Output is a valid DAG**: `nx.is_directed_acyclic_graph(causal_graph_)` is `True` after every fit, across 5 random seeds.
- **No self-loops**: `causal_graph_` contains no edge `(v, v)` for any node `v`.
- **Predictions are in original scale**: `predict()` applies `scaler_.inverse_transform` before returning. Assert that mean and std of `predict(X_test)` are within a reasonable range of the target column's original (unscaled) mean and std.
- **Reproducibility**: Two `CASTLE` instances with the same `random_state` and hyperparameters, fit on the same data, produce byte-identical `adjacency_matrix_` values and identical `predict` outputs.
