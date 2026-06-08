# GSoC Proposal: Deep Learning-Based Causal Discovery Algorithms for pgmpy

## Personal Details
**Contributors**: @Manas-7854  
**Mentors**: ankurankan, DARHWOLF

## Project Goals

### Problem Description
This proposal aims to implement four deep learning-based causal discovery algorithms in pgmpy: CASTLE, DiffAN, GraN-DAG, and CAREFL. Each algorithm leverages neural networks to go beyond classical score-based or constraint-based methods, enabling discovery on non-linear, non-Gaussian data. All four algorithms will be implemented inside `pgmpy/causal_discovery`, similar to the existing causal discovery algorithms. This enables easy extension using the current base class and its built-in functionality, while keeping soft dependencies (PyTorch, diffusers, nflows) isolated.

### Algorithm Overviews

**CASTLE**
Overview: CASTLE is a regularization method that improves supervised learning by jointly learning a DAG and a predictive model over the data.
Paper: [CASTLE: Regularization via Auxiliary Causal Graph Discovery](https://arxiv.org/pdf/2009.13180)
Reference codebase: [trentkyono/CASTLE](https://github.com/trentkyono/CASTLE)

**DiffAN**
Overview: DiffAN (Diffusion-based Acyclicity Notears) trains a Diffusion Probabilistic Model (DPM) over the dataset to estimate the score function of the joint distribution. Under an Additive Noise Model (ANM) assumption, the score function uniquely identifies leaf nodes via diagonal Hessian variance analysis. Leaf nodes are pruned iteratively to produce a topological ordering, which is then converted to a DAG using CAM pruning.
Paper: [Diffusion Models for Causal Discovery via Topological Ordering](https://arxiv.org/abs/2210.06201)
Reference codebase: [vios-s/DiffAN](https://github.com/vios-s/DiffAN)

**GraN-DAG**
Overview: GraN-DAG (Gradient-based Neural DAG Learning) parameterizes each node’s conditional distribution using a neural network that takes all other variables as input. A continuous differentiable acyclicity constraint (adapted from NOTEARS) is imposed so that the entire structure learning problem can be solved end-to-end with gradient descent. A Lagrangian augmentation scheme is used to enforce the acyclicity constraint as a hard constraint at convergence. Unlike linear NOTEARS, GraN-DAG captures non-linear relationships without requiring explicit functional form assumptions.
Paper: [Gradient-Based Neural DAG Learning](https://arxiv.org/abs/1906.02226)
Reference codebase: [kurowasan/GraN-DAG](https://github.com/kurowasan/GraN-DAG)

**CAREFL**
Overview: CAREFL (Causal Autoregressive Flows) identifies causal direction using normalizing flows — specifically, affine autoregressive flows. The core idea is that in the true causal direction X → Y, a flow-based model can achieve a higher log-likelihood when the residuals (noise terms) are modelled as independent of the causes, compared to the anti-causal direction. This leverages a connection to independent component analysis (ICA): in the correct causal ordering, the model exhibits a higher marginal likelihood. CAREFL works for both bivariate causal discovery and multivariate settings via a permutation-based search over variable orderings.
Paper: [Causal Autoregressive Flows](https://arxiv.org/abs/2011.02268)
Reference codebase: [piomonti/CAREFL](https://github.com/piomonti/CAREFL)

## Solution and Implementation Details

All four algorithms will be implemented inside `pgmpy/causal_discovery/`, similar to the existing causal discovery algorithms. This enables easy extension using the current base class (`_BaseCausalDiscovery`) and its built-in functionality, while keeping soft dependencies (PyTorch, diffusers, nflows) isolated. Test files will be located in `pgmpy/tests/test_causaldiscovery/`.

```python
pgmpy/
  causal_discovery/
    base.py       # _BaseCausalDiscovery (existing)
    castle.py     # (new)
    diffan.py     # (new)
    grandag.py    # (new)
    carefl.py     # (new)
  tests/
    test_causaldiscovery/
      test_castle.py   # (new)
      test_diffan.py   # (new)
      test_grandag.py  # (new)
      test_carefl.py   # (new)
```

# CASTLE: Implementation Details

**Algorithm steps:**
1. Formulate a supervised prediction task for a target variable.
2. Initialize an internal masked-autoencoder network (`_CASTLEModel`) that prevents any feature from causing itself.
3. Optimize the joint objective minimizing supervised loss (MSE), data reconstruction loss, and continuous acyclicity penalty, with optional early stopping.
4. Calculate the weighted adjacency matrix $W$ from the input-layer weights of the internal network.
5. Add a DAG acyclicity penalty $h(W) = \text{tr}(e^{W \odot W}) - d = 0$.
6. Final inference zeroes out entries below a threshold to return the resulting causal DAG.

**Key design decisions:**
- Implementation includes an internal PyTorch model `_CASTLEModel`.
- Uses `torch.linalg.matrix_exp` to perform trace calculation for DAG constraints efficiently.
- Internal hyperparameters are grouped into dataclasses (`RegularizationConfig`, `ModelConfig`) for readability — the public API signature is flat and unchanged.
- Accepts a PyTorch optimizer object directly; defaults to Adam internally.
- Accepts a user-provided scaler for input normalization (for example, `sklearn.preprocessing.StandardScaler`); applied in `CASTLE.fit` and stored as `scaler_` after fitting.
- Logs training metrics to TensorBoard when `tensorboard_log_dir` is provided; no verbose printing.

---

## API

### `_CASTLEModel(nn.Module)` (Internal)
PyTorch masked autoencoder for feature reconstruction, target prediction, and training.

- **`__init__(num_inputs, model_cfg: ModelConfig, reg_cfg: RegularizationConfig)`**: Initializes masked input layers, scalar output layers, grouped hyperparameters, and optimizer.
- **`forward(X)`**: Returns full reconstruction (`Out`) and target prediction (`out_0`).
- **`train(X_tensor)`**: Runs epochs with early stopping (batch updates, Lagrangian multiplier adjustments), logs to TensorBoard if enabled, and returns the thresholded adjacency matrix (`W_final`).
- **`get_W()`**: Computes adjacency matrix from current weights (L2 norm of column $j$ in sub-network $i$).

### Internal dataclasses
Grouped configuration objects used by `_CASTLEModel`.

- **`ModelConfig`**: `hidden_dim`, `batch_size`, `max_epochs`, `optimizer`, `seed`, `min_loss_improvement`, `early_stop_patience`, `tensorboard_log_dir`, `scaler`, `target_col`
- **`RegularizationConfig`**: `dag_weight`, `sparsity_weight`, `dag_penalty`, `edge_threshold`

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
    target_col=None,
    max_epochs=200,
    min_loss_improvement=1e-4,
    early_stop_patience=10,
    scaler=None,
    tensorboard_log_dir=None,
    seed=42,
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
- `target_col`: Target column name or index. Defaults to the first column when `None`.
- `max_epochs`: Maximum number of training epochs.
- `min_loss_improvement`: Minimum decrease in total loss per epoch to count as an improvement for early stopping.
- `early_stop_patience`: Number of consecutive epochs without improvement before stopping early.
- `scaler`: Optional scaler for input normalization (for example, `sklearn.preprocessing.StandardScaler`). It must implement `fit(X)` and `transform(X)`; `inverse_transform(X)` is optional for user-side de-scaling. If provided, it is fit on training data and stored as `scaler_`.
- `tensorboard_log_dir`: If set, logs training metrics via `SummaryWriter(log_dir=...)`. If `None`, TensorBoard logging is disabled.
- `seed`: Seed for reproducibility.

**Methods:**
- **`fit(X)`**: Trains the model and builds the DAG.

**Attributes set after `fit`:**
- `n_features_in_`: Number of input features seen during fit.
- `feature_names_in_`: Feature names from the input DataFrame.
- `causal_graph_`: Learned causal DAG as a `pgmpy.base.DAG`.
- `adjacency_matrix_`: Weighted adjacency matrix as a Pandas DataFrame.
- `model_`: Internal neural network instance used for training.
- `scaler_`: Fitted scaler used to normalize inputs during training (when provided).

---

## Usage

```python
import torch
from pgmpy.causal_discovery import CASTLE
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.DataFrame(np.random.randn(100, 3), columns=['A', 'B', 'Target'])

# Default: Adam with lr=1e-3
model = CASTLE(max_epochs=200, edge_threshold=0.3, seed=42)

# Custom optimizer — user controls all optimizer params directly
opt = torch.optim.Adam(lr=1e-4, weight_decay=1e-5)
model = CASTLE(max_epochs=200, edge_threshold=0.3, optimizer=opt, seed=42)

# User-provided scaler and early stopping defaults
scaler = StandardScaler()
model = CASTLE(
    max_epochs=200,
    edge_threshold=0.3,
    target_col="Target",
    scaler=scaler,
    min_loss_improvement=1e-4,
    early_stop_patience=10,
    tensorboard_log_dir="runs/castle",
    seed=42,
)
model.fit(df)
dag = model.causal_graph_

# User-side scaling and inverse-scaling for predictions (example)
scaler_fitted = model.scaler_
X_scaled = scaler_fitted.transform(df.values)
y_pred_scaled = some_model(X_scaled)
y_pred = scaler_fitted.inverse_transform(y_pred_scaled)
```

View logs with: `tensorboard --logdir runs/castle`

---

## Test Plan

All tests use fixed `seed=42` and small synthetic DAGs generated via `utils.gen_data_nonlinear` (mirroring the reference implementation) so results are deterministic and fast. A shared 5-node linear DAG fixture with 500 samples is the default dataset unless stated otherwise.

### 1. Basic Input / Output Tests

- **Fit returns self and sets attributes**: `CASTLE.fit(df)` returns the estimator instance; `n_features_in_`, `feature_names_in_`, `causal_graph_`, `adjacency_matrix_`, and `model_` are all set after the call.
- **Output shapes are correct**: `adjacency_matrix_` is `(d, d)`; `get_W()` inside `_CASTLEModel` returns `(d, d)`.
- **Target column selection**: Construct with `target_col` as a string name, as an integer index, and omitted (defaults to first column) — all three produce identical results.
- **Invalid target raises `ValueError`**: Passing a column name not in `X` or an out-of-range integer index raises `ValueError`.
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

- **`_CASTLEModel` is independently instantiable**: The internal class can be constructed and used without going through `CASTLE.fit`, confirming the internal class separation holds at the unit level.
- **Scaler is fit on training data only**: When `scaler` is provided, `scaler_` matches a separately fitted scaler on the same training split, confirming test data has not leaked into the scaler.
- **Scaler interface validation**: Passing a scaler without `fit` or `transform` raises a clear error; a valid scaler (e.g., `StandardScaler`) works end-to-end.
- **Missing `torch` raises a clean error**: Importing `CASTLE` without `torch` installed raises `ImportError` with an actionable install message, not a bare `ModuleNotFoundError`.
- **Early stopping triggers as expected**: With `min_loss_improvement` set high and `early_stop_patience` small, training halts before `max_epochs` and reports the shorter epoch count.
- **TensorBoard logging is optional**: When `tensorboard_log_dir=None`, no SummaryWriter is created; when set, a valid event file is written.

### 5. Benchmarking Tests

> These tests require sufficient data and epochs to observe meaningful trends. Run separately from the main test suite.

- **Supervised loss decreases**: Record supervised loss at epoch 1 and epoch `max_epochs`; assert final < initial.
- **Reconstruction loss decreases**: Same check on the MSE reconstruction term across epochs.
- **`h(W)` trends downward within a tolerance**: Record `_dag_constraint(model_.get_W())` at epoch 1 and final epoch. Assert decrease of at least 50% from initial value. An absolute floor of `h < 0.5` is logged as a soft check but not a hard failure.
- **`dag_penalty` doubles correctly**: When `h(W)` fails to decrease by 75% between epochs, the trainer's internal `dag_penalty` doubles. Tested by patching the `h` value returned inside the epoch loop to a fixed constant.
- **`sparsity_weight` drives edge sparsity**: Train with high `sparsity_weight` (e.g. 50) and low (e.g. 0.1); assert that high `sparsity_weight` yields strictly fewer non-zero entries in `W_final`.
- **Output is a valid DAG**: `nx.is_directed_acyclic_graph(causal_graph_)` is `True` after every fit, across 5 random seeds.
- **No self-loops**: `causal_graph_` contains no edge `(v, v)` for any node `v`.
- **Reproducibility**: Two `CASTLE` instances with the same `seed` and hyperparameters, fit on the same data, produce byte-identical `adjacency_matrix_` values.

# GraN-DAG: Implementation Details
---
## TO BE DECIDED
---
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

# CAREFL: Implementation Details
---
## TO BE DECIDED
---
**Algorithm steps:**
1*Algorithm steps:**
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

# DiffAN: Implementation Details
---
## TO BE DECIDED
---

**Algorithm steps:**
1. Assume the data is generated by an ANM: $X_i = f_i(Pa(X_i)) + \epsilon_i$.
2. Train a DPM over the dataset X to obtain an estimate of the score $
abla_x \log p(x)$.
3. Compute the Jacobian of the neural network output at sampled timesteps t to approximate the diagonal of the Hessian.
4. The variable with the lowest variance of the diagonal Hessian entry is identified as the leaf node.
5. Apply residue correction (the “deciduous score”) to subtract the removed leaf’s contribution, enabling subsequent iterations without retraining.
6. Repeat steps 3–5 to obtain a full topological ordering.
7. Apply CAM pruning on the ordering to produce the final DAG.

**Key design decisions:**
* Uses HuggingFace diffusers (`DDPMScheduler`), reducing complexity.
* The DPM backbone is a small MLP (not U-Net), which is appropriate for tabular data.
* CAM pruning is implemented using a GLM with a significance threshold.

**API:**
```python
from pgmpy.causal_discovery.base import _BaseCausalDiscovery
import pandas as pd
import numpy as np

class DiffAN(_BaseCausalDiscovery):
    def __init__(
        self,
        epochs: int = 300,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        residue: bool = True,
        masking: bool = True,
        n_votes: int = 3,
        cutoff: float = 1e-3,
        hidden_dim: int = 64,
        n_diffusion_steps: int = 1000,
        device: str = "cpu",
    ):
        ...

    def fit(self, X: pd.DataFrame) -> "DiffAN":
        ...

    # Internal methods
    def _normalize(self, X: np.ndarray) -> np.ndarray: ...
    def _build_model(self, n_nodes: int): ... 
    def _train_score_model(self, X: np.ndarray): ... 
    def _topological_ordering(self, X: np.ndarray) -> list: ... 
    def _cam_pruning(self, order: list, X: np.ndarray) -> np.ndarray: ...
    def _to_dag(self, adj: np.ndarray, nodes: list) -> "DAG": ...
```

**Tests (`pgmpy/tests/test_causaldiscovery/test_diffan.py`):** Data will be synthetically generated from a known ANM (e.g., linear Gaussian or non-linear with sigmoid activations). The test asserts that the recovered ordering is consistent with the true topological ordering on small graphs (n ≤ 6 nodes).
