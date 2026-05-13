# GSoC Proposal: Deep Learning-Based Causal Discovery Algorithms for pgmpy

## Personal Details
**Contributors**: @Manas-7854  
**Mentors**: ankurankan, DARHWOLF

## Project Goals

### Problem Description
This proposal aims to implement four deep learning-based causal discovery algorithms in pgmpy: CASTLE, DiffAN, GraN-DAG, and CAREFL. Each algorithm leverages neural networks to go beyond classical score-based or constraint-based methods, enabling discovery on non-linear, non-Gaussian data. All four will be implemented inside `pgmpy/causal_discovery` to isolate the soft dependencies (PyTorch, diffusers, nflows) from the rest of the library.

### Algorithm Overviews

**CASTLE**
Overview: CASTLE is a regularization method that improves supervised learning by jointly learning a DAG and a predictive model over the data.
Paper: [Kyono, T., Zhang, Y., & van der Schaar, M. (2020). CASTLE: Regularization via Auxiliary Causal Graph Discovery. NeurIPS 2020.](https://arxiv.org/pdf/2009.13180)
Reference codebase: [trentkyono/CASTLE](https://github.com/trentkyono/CASTLE)

**DiffAN**
Overview: DiffAN (Diffusion-based Acyclicity Notears) trains a Diffusion Probabilistic Model (DPM) over the dataset to estimate the score function of the joint distribution. Under an Additive Noise Model (ANM) assumption, the score function uniquely identifies leaf nodes via diagonal Hessian variance analysis. Leaf nodes are pruned iteratively to produce a topological ordering, which is then converted to a DAG using CAM pruning.
Paper: [Sanchez, P. & Tsaftaris, S.A. (2022). Diffusion Models for Causal Discovery via Topological Ordering. arXiv:2210.06201](https://arxiv.org/abs/2210.06201)
Reference codebase: [vios-s/DiffAN](https://github.com/vios-s/DiffAN)

**GraN-DAG**
Overview: GraN-DAG (Gradient-based Neural DAG Learning) parameterizes each node’s conditional distribution using a neural network that takes all other variables as input. A continuous differentiable acyclicity constraint (adapted from NOTEARS) is imposed so that the entire structure learning problem can be solved end-to-end with gradient descent. A Lagrangian augmentation scheme is used to enforce the acyclicity constraint as a hard constraint at convergence. Unlike linear NOTEARS, GraN-DAG captures non-linear relationships without requiring explicit functional form assumptions.
Paper: [Lachapelle, S., Brouillard, P., Deleu, T. & Lacoste-Julien, S. (2020). Gradient-Based Neural DAG Learning. ICLR 2020. arXiv:1906.02226](https://arxiv.org/abs/1906.02226)
Reference codebase: [kurowasan/GraN-DAG](https://github.com/kurowasan/GraN-DAG)

**CAREFL**
Overview: CAREFL (Causal Autoregressive Flows) identifies causal direction using normalizing flows — specifically, affine autoregressive flows. The core idea is that in the true causal direction X → Y, a flow-based model can achieve a higher log-likelihood when the residuals (noise terms) are modelled as independent of the causes, compared to the anti-causal direction. This leverages a connection to independent component analysis (ICA): in the correct causal ordering, the model exhibits a higher marginal likelihood. CAREFL works for both bivariate causal discovery and multivariate settings via a permutation-based search over variable orderings.
Paper: [Khemakhem, I., Monti, R., Leech, R. & Hyvärinen, A. (2021). Causal Autoregressive Flows. AISTATS 2021. arXiv:2011.02268](https://arxiv.org/abs/2011.02268)
Reference codebase: [piomonti/CAREFL](https://github.com/piomonti/CAREFL)

## Solution and Implementation Details

All four algorithms will be implemented inside `pgmpy/causal_discovery/` and will inherit from the `_BaseCausalDiscovery` base class. Since these algorithms depend on PyTorch (and others like `diffusers` / `nflows`), they are kept as an isolated sub-module with soft dependencies.

```python
pgmpy/
  causal_discovery/
    base.py       # _BaseCausalDiscovery (existing)
    CASTLE.py     # (new)
    DiffAN.py     # (new)
    GraNDAG.py    # (new)
    CAREFL.py     # (new)
```

### CASTLE: Implementation Details

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

**Tests:** Synthetic DAGs (e.g. from bnlearn simulating the Asia network) will be generated. Tests will assert whether the learned DAG captures the structural relationships accurately according to the simulated DAG, and checks the predictive MSE on unseen data.

### DiffAN: Implementation Details

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

**Tests:** Data will be synthetically generated from a known ANM (e.g., linear Gaussian or non-linear with sigmoid activations). The test asserts that the recovered ordering is consistent with the true topological ordering on small graphs (n ≤ 6 nodes).

### GraN-DAG: Implementation Details

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

**Tests:** Synthetic DAGs with linear-Gaussian and non-linear (MLP-generated) SCMs. SHD (Structural Hamming Distance) is checked to be below a permissive threshold on small graphs.

### CAREFL: Implementation Details

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
        device: str = "cpu",
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

**Tests:** Bivariate tests use synthetic ANM pairs where the true direction is known. Multivariate tests use small synthetically generated DAGs. Correct direction recovery rate is asserted to exceed chance level.
