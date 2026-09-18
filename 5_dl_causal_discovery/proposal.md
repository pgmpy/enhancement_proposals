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

**Algorithm steps:**
1. For each variable $X_i$, parameterize $p(X_i|X_{-i})$ using a neural network.
2. *(Optional — PNS pre-filter)* If `pns_threshold` is set, fit a tree-based sklearn estimator (default `ExtraTreesRegressor`; any estimator exposing `feature_importances_` is accepted) per variable to compute feature importances; mask out parent candidates with importance below `pns_threshold × mean_importance`, reducing the effective input dimension before optimization.
3. The weighted adjacency matrix $A_\phi$ is obtained from the internal connectivity of the NNs via a masking scheme (self-masking). For each NN $j$, the connectivity matrix $C^{(j)} = |W_L^{(j)}| \cdots |W_1^{(j)}|$ (product of absolute weight matrices across all layers) yields one row of $A_\phi$; the diagonal is forced to zero.
4. Optimize the negative log-likelihood of the data subject to the continuous acyclicity constraint $h(\phi) = \text{tr}(e^{A_\phi}) - d = 0$ using an augmented Lagrangian method (outer loop over subproblems, mini-batch SGD inner loop with early stopping on a held-out validation split, $\lambda$/$\mu$ updates between subproblems). Note: unlike CASTLE's $h(W) = \text{tr}(e^{W \odot W}) - d$, GraN-DAG does **not** element-wise-square $A_\phi$ because the connectivity product $C^{(j)} = |W_L| \cdots |W_1|$ already produces non-negative entries.
5. After convergence, compute the expected absolute Jacobian matrix $J_{ij} = \mathbb{E}\!\left[\left|\tfrac{\partial f_j}{\partial x_i}\right|\right]$ over the training data via `torch.autograd` and threshold it at `edge_threshold` to recover the DAG skeleton. The Jacobian is preferred over $A_\phi$ because it measures the actual functional sensitivity of each output to each input, accounting for weight cancellations and non-linear saturation that the raw connectivity product cannot capture.
6. *(Optional — CAM pruning)* If `pruning_cutoff` is set, apply CAM pruning using pgmpy's CAM score implementation (available via an existing PR); drop any parent whose score has a p-value exceeding `pruning_cutoff`, removing statistically insignificant edges.
7. Build the final `pgmpy.DAG` from the surviving edges.

**Key design decisions:**
- Implementation includes an internal PyTorch model `_GraNDAGModel` (ensemble of per-variable NNs).
- Internal hyperparameters are grouped into dataclasses (`NetworkConfig`, `TrainingConfig`, `RegularizationConfig`) for readability — the public API signature is flat and unchanged.
- Accepts an optimizer as a **string name** (`"rmsprop"`, `"adam"`, etc.) plus an optional `optimizer_params` dict; the string is resolved to the corresponding `torch.optim` class internally and instantiated with `optimizer_params`. This avoids the complexity of passing an unbound optimizer object and keeps the API serializable.
- Accepts a user-provided scaler for input normalization (for example, `sklearn.preprocessing.StandardScaler`); it must implement `fit(X)` and `transform(X)`. Applied in `GraNDAG.fit`, fit on training data only, and stored as `scaler_` after fitting.
- Logs training metrics to TensorBoard when `tensorboard_log_dir` is provided; no verbose printing.
- Optional **Preliminary Neighbourhood Selection (PNS)** via `pns_threshold` reduces the search space before training, following the GraN-DAG paper's recommended pipeline. The estimator used for feature importance is configurable — any sklearn estimator exposing a `feature_importances_` attribute (e.g., `ExtraTreesRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor`) is accepted; defaults to `ExtraTreesRegressor`.
- Optional **CAM pruning** via `pruning_cutoff` removes statistically insignificant edges after structure recovery, as described in the paper's post-processing. Will reuse pgmpy's existing CAM score implementation (open PR) if possible, avoiding a standalone reimplementation.

---

## API

### Helper functions (module-level, private)
Standalone utilities used by `_GraNDAGModel` and `GraNDAG`. Defined at module scope to keep the classes focused.

- **`_validate_optimizer(optimizer: str, optimizer_params: dict) -> None`** — Validates that `optimizer` resolves to a `torch.optim` class (case-insensitive) and that `optimizer_params` is a dict. Raises `ValueError` with a message listing valid optimizer names on failure. Reused as-is from CASTLE.
- **`_dag_constraint(W: Tensor, square: bool = True) -> Tensor`** — Computes the acyclicity constraint via `torch.linalg.matrix_exp`. When `square=True` (CASTLE), computes $h(W) = \text{tr}(e^{W \odot W}) - d$ because $W$ can contain negative entries. When `square=False` (GraN-DAG), computes $h(W) = \text{tr}(e^{W}) - d$ directly since $A_\phi$ is already non-negative. Shared between CASTLE and GraN-DAG; GraN-DAG calls with `square=False`.
- **`_run_pns(X: np.ndarray, pns_threshold: float, seed: int, estimator=None) -> np.ndarray`** — Preliminary Neighbourhood Selection: fits the provided estimator (default `ExtraTreesRegressor(random_state=seed)`) per variable on all other variables. The estimator must expose a `feature_importances_` attribute after fitting; a `TypeError` is raised otherwise. Returns a boolean mask of shape `(d, d)` where `True` indicates a surviving parent candidate (importance $\ge$ `pns_threshold` $\times$ mean importance). Diagonal is always `False`.
- **`_run_cam_pruning(X: np.ndarray, adj: np.ndarray, pruning_cutoff: float) -> np.ndarray`** — CAM pruning: delegates to pgmpy's CAM score implementation (open PR) to score each parent; drops any parent whose score has a p-value exceeding `pruning_cutoff`. Returns the pruned adjacency matrix. Falls back to a standalone OLS implementation if the pgmpy CAM module is not yet available.
- **`_threshold_to_dag(J: Tensor, edge_threshold: float) -> Tensor`** — Iterative edge removal following Appendix A.2 of the paper: threshold entries of $J$ below `edge_threshold` to zero, then iteratively remove the lowest-weight edge that participates in a cycle until the graph is a DAG. Returns a binary adjacency tensor.

### `_GraNDAGModel(nn.Module)` (Internal)
PyTorch NN ensemble for per-variable conditional distribution learning and DAG structure recovery.

- **`__init__(self, num_vars: int, network_cfg: NetworkConfig, train_cfg: TrainingConfig, reg_cfg: RegularizationConfig)`**: Clones user-provided NN template `num_vars` times (via `copy.deepcopy`), registers self-masks and optional PNS mask as buffers, initializes Lagrangian coefficients ($\lambda$, $\mu$), instantiates the optimizer from `train_cfg.optimizer` + `train_cfg.optimizer_params`, and sets the random seed via `train_cfg.seed`.
- **`forward(self, X) -> Tensor`**: Applies self-masking per variable (and PNS mask if active), runs each sub-network, returns distribution parameters $\theta \in \mathbb{R}^{N \times d \times \text{output\_dim}}$ for all $d$ NNs.
- **`train(self, X_tensor=True, val_tensor=None)`**: Overloaded following the same pattern as CASTLE's `_CASTLEModel.train()` — when called with a bool (`True`/`False`), delegates to `nn.Module.train(mode)` for train/eval mode switching; when called with a `Tensor` as the first argument, runs the actual training loop: augmented Lagrangian outer loop, where each subproblem runs a mini-batch SGD inner loop computing validation NLL each epoch, stopping when no improvement exceeding `min_loss_improvement` for `early_stop_patience` consecutive epochs. After each subproblem, calls `_update_lagrangian`. Returns thresholded adjacency matrix `W_final`.
- **`get_A(self) -> Tensor`**: Computes weighted adjacency matrix $A_\phi$ via connectivity matrix $C^{(j)} = |W_L^{(j)}| \cdots |W_1^{(j)}|$ for each NN $j$; row $j$ of $A_\phi$ is the sum across output neurons of $C^{(j)}$; diagonal forced to zero.
- **`get_jacobian(self, X) -> Tensor`**: Computes expected absolute Jacobian matrix $J_{ij} = \mathbb{E}[|\partial f_j / \partial x_i|]$ over the full dataset using `torch.autograd.grad` with `create_graph=False`; used for final thresholding instead of $A_\phi$.
- **`_update_lagrangian(self, h_val: float, h_prev: float) -> None`** — Updates the Lagrangian coefficients after each subproblem per Eq. 14 of the paper: $\lambda \leftarrow \lambda + \mu \cdot h_{\text{val}}$; if $h_{\text{val}} > \omega_\mu \cdot h_{\text{prev}}$ then $\mu \leftarrow \eta \cdot \mu$.

### Internal dataclasses
Grouped configuration objects used by `_GraNDAGModel`.

- **`NetworkConfig`**: `net`, `output_dim`, `log_likelihood`, `scaler`
- **`TrainingConfig`**: `optimizer`, `optimizer_params`, `batch_size`, `val_size`, `min_loss_improvement`, `early_stop_patience`, `max_subproblems`, `seed`
- **`RegularizationConfig`**: `dag_multiplier_init`, `dag_penalty`, `dag_penalty_growth_factor`, `dag_penalty_growth_threshold`, `dag_constraint_tol`, `edge_threshold`

### `GraNDAG(_BaseCausalDiscovery)` (Public API)
Validates data, orchestrates PNS → training → thresholding → CAM pruning, and builds the `pgmpy.DAG`. Calls `_check_soft_dependencies` on import to verify `torch` availability.

```python
GraNDAG(
    # Network
    net=None,
    output_dim=2,
    log_likelihood=None,
    scaler=None,

    # Augmented Lagrangian
    dag_multiplier_init=0.0,
    dag_penalty=1e-3,
    dag_penalty_growth_factor=10.0,
    dag_penalty_growth_threshold=0.9,
    dag_constraint_tol=1e-8,
    max_subproblems=None,

    # Optimization
    optimizer="rmsprop",
    optimizer_params=None,
    batch_size=64,
    val_size=0.1,
    min_loss_improvement=1e-4,
    early_stop_patience=5,
    seed=42,

    # Thresholding & post-processing
    edge_threshold=1e-4,
    pns_threshold=None,
    pruning_cutoff=None,
)
```

Augmented Lagrangian objective being minimized per subproblem:

$$\min_\phi \; \underbrace{-\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{d} \log p\!\left(x_{ij} \mid \theta_j(\mathbf{x}_i;\,\phi)\right)}_{\text{negative log-likelihood}} \;+\; \lambda\, h(A_\phi) \;+\; \frac{\mu}{2}\, h(A_\phi)^2$$

where the acyclicity constraint (Eq. 9 of the paper) is:

$$h(\phi) = \text{tr}\!\left(e^{A_\phi}\right) - d = 0$$

Note: $A_\phi$ is already entry-wise non-negative (built from the connectivity product $C^{(j)} = |W_L| \cdots |W_1|$), so no element-wise squaring is needed — unlike CASTLE's $h(W) = \text{tr}(e^{W \odot W}) - d$ where $W$ can be negative.

Between subproblems the Lagrangian coefficients are updated as:
- $\lambda^{(k+1)} = \lambda^{(k)} + \mu^{(k)} \cdot h\!\left(A_\phi^{(k)}\right)$
- If $h\!\left(A_\phi^{(k)}\right) > \omega_\mu \cdot h\!\left(A_\phi^{(k-1)}\right)$: $\;\mu^{(k+1)} = \eta \cdot \mu^{(k)}$, else $\mu$ is unchanged.

---

- `net`: `nn.Module` template, cloned $d$ times via `copy.deepcopy`; defaults to a built-in 2-layer MLP with sigmoid activations and `output_dim` outputs. The template must satisfy: (a) all learnable layers are `nn.Linear`, (b) the first layer accepts `d` inputs, (c) the last layer produces `output_dim` outputs. These constraints are required so that the connectivity matrix $C^{(j)} = |W_L| \cdots |W_1|$ can be computed from the `Linear` layer weights.
- `output_dim`: Number of output neurons per sub-network. Determines the number of distribution parameters produced per variable: 2 for Gaussian ($\mu$, $\log\sigma$), 1 for a distribution parameterized by its mean alone. Must match the expectation of the `log_likelihood` function.
- `log_likelihood`: Callable with signature `fn(x_j: Tensor[batch_size], theta: Tensor[batch_size, output_dim]) -> Tensor[batch_size]` that returns the per-sample log-probability of observed values `x_j` given distribution parameters `theta`. Defaults to Gaussian: $\log \mathcal{N}(x_j;\, \mu,\, e^{\log\sigma})$.
- `scaler`: Optional scaler for input normalization (for example, `sklearn.preprocessing.StandardScaler`). It must implement `fit(X)` and `transform(X)`; `inverse_transform(X)` is optional for user-side de-scaling. If provided, it is fit on training data only and stored as `scaler_` after fitting.
- `dag_multiplier_init` ($\lambda^{(0)}$): Initial Lagrangian multiplier for the acyclicity equality constraint. Starts at 0; increased automatically each outer iteration via $\lambda \leftarrow \lambda + \mu \cdot h$.
- `dag_penalty` ($\mu^{(0)}$): Initial quadratic penalty coefficient on the acyclicity violation $h(A_\phi)^2$. Same role as CASTLE's `dag_penalty` ($\rho$) — understood to be a starting value that is increased automatically during training.
- `dag_penalty_growth_factor` ($\eta$): Multiplier applied to the penalty when $h(A_\phi)$ has not decreased by at least `dag_penalty_growth_threshold` relative to the previous subproblem.
- `dag_penalty_growth_threshold` ($\gamma$): Ratio threshold for triggering a penalty increase. If $h^{(k)} > \gamma \cdot h^{(k-1)}$, then the penalty is multiplied by $\eta$.
- `dag_constraint_tol`: Stop the outer loop when $h(A_\phi) \le$ `dag_constraint_tol`, indicating the acyclicity constraint is approximately satisfied.
- `max_subproblems`: Hard cap on the number of augmented Lagrangian outer iterations. `None` means no cap — the loop runs until $h(A_\phi) \le$ `dag_constraint_tol`.
- `optimizer`: String name of a `torch.optim` optimizer class (`"rmsprop"`, `"adam"`, `"sgd"`, etc.), resolved internally via `getattr(torch.optim, name)`. Defaults to `"rmsprop"` as in the paper. Case-insensitive matching is applied.
- `optimizer_params`: Dict of keyword arguments forwarded to the optimizer constructor (e.g., `{"lr": 1e-3, "weight_decay": 1e-5}`). Defaults to `{"lr": 1e-3}` when `None`.
- `batch_size`: Number of samples per mini-batch during the inner optimization loop.
- `val_size`: Fraction of the data held out for early stopping within each subproblem. The split is random and controlled by `seed`. If `0.0`, early stopping is disabled and each subproblem runs for a fixed number of iterations determined by other stopping criteria.
- `min_loss_improvement`: Minimum decrease in validation NLL per epoch to count as an improvement for early stopping within a subproblem.
- `early_stop_patience`: Number of consecutive epochs without improvement (exceeding `min_loss_improvement`) before the current subproblem is stopped early and the outer loop advances.
- `seed`: Seed for reproducibility (controls weight initialization, data shuffling, and validation split).
- `edge_threshold`: Entries in the Jacobian matrix $J$ below this value are zeroed in the final adjacency matrix.
- `pns_threshold`: If not `None`, a Preliminary Neighbourhood Selection step is run before training: a tree-based sklearn estimator (default `ExtraTreesRegressor` with `random_state=seed`) is fit per variable on all other variables; parent candidates whose feature importance falls below `pns_threshold × mean_importance` for that variable are masked out (their input connections are permanently zeroed). Any estimator exposing `feature_importances_` after fitting is accepted (e.g., `RandomForestRegressor`, `GradientBoostingRegressor`). This reduces the search space for high-dimensional problems. Requires `scikit-learn`.
- `pruning_cutoff`: If not `None`, a CAM pruning post-processing step is applied after Jacobian thresholding, using pgmpy's CAM score implementation (open PR) where available. For each node, parents whose score has a p-value exceeding `pruning_cutoff` are dropped, removing statistically insignificant edges.

**Methods:**
- **`__init__(self, **20 params)`** — Mirrors CASTLE's flat signature. Stores all hyperparameters as instance attributes and calls `_check_soft_dependencies` to verify `torch` (and optionally `sklearn` if `pns_threshold` or `pruning_cutoff` are set).
- **`_fit(self, X: pd.DataFrame)`** — Called by the base class `fit(X)`. Executes the full pipeline:
  - **Step 0 — Validate**: Call `_validate_optimizer(optimizer, optimizer_params)`, validate `X` (numeric, ≥ 2 columns), run `_run_pns(X, pns_threshold, seed)` if `pns_threshold` is set.
  - **Step 1 — Build configs**: Construct `NetworkConfig`, `TrainingConfig`, `RegularizationConfig` dataclasses from stored hyperparameters.
  - **Step 2 — Train**: Convert `X` to tensors (apply `scaler` if provided, split off `val_size` fraction), instantiate `_GraNDAGModel(num_vars, ...)`, call `.train(X_tensor, val_tensor)` to run the augmented Lagrangian loop.
  - **Step 3 — Post-process**: Compute Jacobian via `model_.get_jacobian(X_tensor)`, apply `_threshold_to_dag(J, edge_threshold)`, run `_run_cam_pruning(X, adj, pruning_cutoff)` if `pruning_cutoff` is set, build `adjacency_matrix_` and `causal_graph_`.

**Attributes set after `fit`:**
- `causal_graph_`: Learned causal DAG as a `pgmpy.base.DAG`.
- `adjacency_matrix_`: Thresholded (and optionally pruned) adjacency matrix as a Pandas DataFrame indexed by feature names.
- `model_`: Internal `_GraNDAGModel` instance used for training.
- `scaler_`: Fitted scaler used to normalize inputs during training (set only when `scaler` is provided).
- `n_features_in_`: Number of input features (variables) seen during fit.
- `feature_names_in_`: Feature names from the input DataFrame.
- `cols_`: Alias for `feature_names_in_`; used internally for column-order bookkeeping.
- `network_config_`: The `NetworkConfig` dataclass instance constructed during fit.
- `train_config_`: The `TrainingConfig` dataclass instance constructed during fit.
- `reg_config_`: The `RegularizationConfig` dataclass instance constructed during fit.
- `pns_mask_`: Boolean mask array of shape `(d, d)` indicating which parent candidates survived PNS (set only when `pns_threshold` is provided).

---

## Usage

```python
import pandas as pd
import numpy as np
from pgmpy.causal_discovery import GraNDAG

df = pd.DataFrame(np.random.randn(500, 4), columns=["X1", "X2", "X3", "X4"])

# Minimal — all defaults (RMSprop, lr=1e-3, Gaussian likelihood)
model = GraNDAG(seed=42)
model.fit(df)
dag = model.causal_graph_

# Custom optimizer
model = GraNDAG(
    optimizer="adam",
    optimizer_params={"lr": 5e-4},
    seed=42,
)
model.fit(df)

# Custom neural network template
import torch.nn as nn
custom_net = nn.Sequential(
    nn.Linear(4, 16),    # first layer: d inputs
    nn.Sigmoid(),
    nn.Linear(16, 16),
    nn.Sigmoid(),
    nn.Linear(16, 2),    # last layer: output_dim outputs
)
model = GraNDAG(net=custom_net, output_dim=2, seed=42)
model.fit(df)

# Full pipeline: PNS + training + CAM pruning + scaler
from sklearn.preprocessing import StandardScaler

model = GraNDAG(
    optimizer="rmsprop",
    optimizer_params={"lr": 1e-3},
    batch_size=64,
    val_size=0.2,
    min_loss_improvement=1e-4,
    early_stop_patience=5,
    max_subproblems=20,
    edge_threshold=0.05,
    pns_threshold=0.75,
    pruning_cutoff=0.001,
    scaler=StandardScaler(),
    seed=42,
)
model.fit(df)
dag = model.causal_graph_
adj = model.adjacency_matrix_
```

---

## Test Plan

All tests use fixed `seed=42` and small synthetic DAGs generated via non-linear SCMs so results are deterministic and fast. A shared 4-node non-linear DAG fixture with 500 samples is the default dataset unless stated otherwise.

### 1. Basic Input / Output Tests

- **Fit returns self and sets attributes**: `GraNDAG.fit(df)` returns the estimator instance; `n_features_in_`, `feature_names_in_`, `causal_graph_`, `adjacency_matrix_`, and `model_` are all set after the call.
- **Output shapes are correct**: `adjacency_matrix_` is `(d, d)`; `get_A()` inside `_GraNDAGModel` returns `(d, d)`; `get_jacobian(X)` returns `(d, d)`.
- **DataFrame and numpy inputs**: `fit` accepts `pd.DataFrame` and validates that non-numeric input raises.
- **Single-column input is rejected gracefully**: A DataFrame with one column raises `ValueError` before any training begins, with a clear message.
- **Hyperparameter edge cases do not crash**: `max_subproblems=1`, `batch_size=1`, `batch_size > N`, `early_stop_patience=1` — all complete without error and produce a valid DAG.
- **`val_size=0.0` disables early stopping**: When `val_size=0.0`, training runs without computing validation loss, and no early stopping occurs within subproblems.

### 2. Network Correctness Tests

- **Self-masking is enforced**: After initialisation and after training, `model_.get_A().diagonal()` is all zeros — no variable predicts itself.
- **Custom `net` cloning**: Provide a custom `nn.Module`; verify that after fitting, the $d$ sub-networks have diverged (different weights) but share the same architecture.
- **Custom `net` validation**: Passing a net whose first `Linear` layer has wrong input dim raises `ValueError`. Passing a net whose last `Linear` layer has wrong output dim raises `ValueError`.
- **`forward` output shapes**: For a random batch of size `B`, output tensor has shape `(B, d, output_dim)`.
- **PNS mask shape and content**: When `pns_threshold` is set, `pns_mask_` is `(d, d)` boolean with `True` diagonal zeroed out and at least one `True` per row.

### 3. DAG Validity Tests

- **Threshold is applied**: All entries in `adjacency_matrix_` are either zero or `≥ edge_threshold`; no values fall in `(0, edge_threshold)`.
- **Varying `edge_threshold` changes graph density monotonically**: Fit once; apply three thresholds `[1e-5, 1e-3, 0.1]` to `adjacency_matrix_`; verify edge count is non-increasing.
- **Output is a valid DAG**: `nx.is_directed_acyclic_graph(causal_graph_)` is `True` after every fit, across 3 random seeds.
- **No self-loops**: `causal_graph_` contains no edge `(v, v)` for any node `v`.

### 4. Optimizer & Early Stopping Tests

- **Optimizer string resolution**: `"rmsprop"`, `"adam"`, `"sgd"` all resolve successfully; `"nonexistent_optimizer"` raises `ValueError` with a clear message listing valid options.
- **Case-insensitive resolution**: `"Adam"`, `"ADAM"`, `"adam"` all resolve to `torch.optim.Adam`.
- **`optimizer_params` are forwarded**: Pass `{"lr": 0.1}` and verify the instantiated optimizer's `param_groups[0]["lr"]` equals `0.1`.
- **Default `optimizer_params`**: When `None`, the optimizer is instantiated with `lr=1e-3`.
- **Early stopping triggers**: With `min_loss_improvement` set high and `early_stop_patience=2`, the first subproblem terminates before exhausting iterations.
- **Early stopping is per-subproblem**: Verify that patience counter resets at the start of each new subproblem.

### 5. PNS & CAM Pruning Tests

- **PNS reduces parent set**: With `pns_threshold=1.0` (aggressive), at least one input is masked out per variable (for a dataset with redundant features).
- **PNS `None` skips the step**: When `pns_threshold=None`, no estimator is fitted and `pns_mask_` is not set.
- **PNS accepts custom estimator**: Passing a `RandomForestRegressor` as the PNS estimator works end-to-end; passing an estimator without `feature_importances_` (e.g., `LinearRegression`) raises `TypeError`.
- **PNS mask is applied during training**: After PNS masking, the masked entries in `get_A()` remain zero throughout training.
- **CAM pruning removes edges**: With `pruning_cutoff=0.5` (permissive) on a sparse ground-truth DAG, the pruned graph has fewer or equal edges compared to the unpruned Jacobian-thresholded graph.
- **CAM `None` skips the step**: When `pruning_cutoff=None`, no pruning regression is run.
- **PNS requires scikit-learn**: When `pns_threshold` is set but `sklearn` is not installed, raise `ImportError` with an actionable message.
- **CAM pruning requires scikit-learn**: When `pruning_cutoff` is set but `sklearn` is not installed, raise `ImportError` with an actionable message.

### 6. Additional Tests

- **`_GraNDAGModel` is independently instantiable**: The internal class can be constructed and used without going through `GraNDAG.fit`.
- **Scaler is fit on training data only**: When `scaler` is provided, `scaler_` matches a separately fitted scaler on the same training split (excluding the validation portion).
- **Scaler interface validation**: Passing a scaler without `fit` or `transform` raises a clear error; a valid scaler (e.g., `StandardScaler`) works end-to-end.
- **Missing `torch` raises a clean error**: Importing `GraNDAG` without `torch` installed raises `ImportError` with an actionable install message, not a bare `ModuleNotFoundError`.
- **Custom `log_likelihood` end-to-end**: Provide a custom log-likelihood (e.g., Laplace) with `output_dim=2`; verify training completes and produces a valid DAG.
- **Reproducibility**: Two `GraNDAG` instances with the same `seed` and hyperparameters, fit on the same data, produce byte-identical `adjacency_matrix_` values.

### 7. Benchmarking Tests

> These tests require sufficient data and epochs to observe meaningful trends. Run separately from the main test suite.

- **Negative log-likelihood decreases**: Record NLL at start and end of training; assert final < initial.
- **$h(A_\phi)$ trends toward zero**: Record $h(A_\phi)$ at the first and last subproblem; assert final $h < 0.1 \times$ initial $h$.
- **`dag_penalty` increases correctly**: When $h(A_\phi)$ fails to decrease by `dag_penalty_growth_threshold` ($\gamma$) between subproblems, verify that `dag_penalty` ($\mu$) is multiplied by `dag_penalty_growth_factor` ($\eta$).
- **`dag_multiplier_init` update is correct**: After each subproblem, verify $\lambda^{(k+1)} = \lambda^{(k)} + \mu^{(k)} \cdot h(A_\phi^{(k)})$.
- **SHD is below threshold**: On a 4-node ground-truth DAG with 500 samples, assert SHD ≤ 4 (permissive threshold for a non-linear SCM).
- **Reproducibility**: Two identical runs produce byte-identical adjacency matrices.

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
