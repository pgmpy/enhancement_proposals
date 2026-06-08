## Continuous Optimization for Causal Discovery

**Contributors**: Syed Abdullah Abrar

### Introduction

This proposal aims to expand `pgmpy`'s causal discovery capabilities by integrating state-of-the-art continuous
optimization algorithms. Classical score-based (e.g., HillClimbSearch) and constraint-based methods face significant
scalability challenges and often struggle with high-dimensional or complex data distributions.

### Proposed Solution

By framing causal discovery as a continuous optimization problem, we can learn DAGs using gradient descent. While early
continuous methods (like NOTEARS) relied on matrix exponentials that suffer from vanishing gradients, this project
focuses on implementing the highly efficient **DAGMA** (Directed Acyclic Graphs via M-matrices for Acyclicity) family
of algorithms.
The project will implement the foundational DAGMA Linear model (currently underway), scale it to DAGMA Non-linear using
neural networks (MLP) and implement advanced variants like DAGMA-DCE.


#### DAGMA Base Class Abstraction Levels

To integrate `DAGMALinear`, `DAGMANonlinear`, and `DAGMADce` without code duplication while adhering strictly to `scikit-learn` guidelines, this proposal will use a **Mixin/Logic Container** architecture.

* **`_BaseDAGMAMixin` in `_base.py`:** This introduces a stateless `_BaseDAGMAMixin` class in `pgmpy/causal_discovery/_base.py` without an `__init__` method (as the proposed algorithms do not share common hyperparameters). It holds the shared mathematical logic (`_log_det_barrier`, `_convert_to_dag`) and a centralized, unified optimization loop (`_optimize`) that accepts an uninstantiated PyTorch optimizer class.
* **Model-Agnostic Loop:** Subclasses act as wrapper estimators. They define their unique hyperparameters in their `__init__` methods, construct their specific model architectures and loss objectives (e.g. least squares for Linear, log-likelihood on standard MLPs for Nonlinear, and Jacobian-based causal effects for DCE), and pass the parameters along with their chosen `torch.optim` optimizer to the unified `_optimize` loop from the mixin.
* **Hardcoded Bias:** The non-linear models (Nonlinear, DCE) hardcode the bias parameter of MLP layers to `True`. This aids neural network capacity (helps activate dead neurons, particularly under L1/L2 regularization on input weights/Jacobians), simplifying the public API by removing a hyperparameter that should always be True in practice.

### Algorithm Overviews

#### DAGMA Linear (Currently Under Implementation)
*   **Overview:** The foundational algorithm. It learns a DAG from observational data by optimizing a continuous score
function (Least Squares) subject to a log-determinant acyclicity constraint. It relies on the properties of M-matrices
to create a barrier that prevents cycles without suffering from vanishing gradients.
*   **Current Status:** Currently under implementation. The base PyTorch architecture, L-BFGS optimizer integration,
and the core log-det barrier logic are being actively developed and refined to ensure numerical stability and `sklearn`
API compliance.
*   **Ref** : DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization.
Kevin Bello, Bryon Aragam, Pradeep Ravikumar. Booth School of Business, University of Chicago, Chicago, IL 60637.
Machine Learning Department, Carnegie Mellon University, Pittsburgh, PA 15213
    * **Paper:** <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/36e2967f87c3362e37cf988781a887ad-Paper
    -Conference.pdf"> Neurips-dagma </a>
    * **Code ref:** <a href="https://github.com/kevinsbello/dagma/blob/main/src/dagma/linear.py">Original Dagma Linear 
    code </a>


#### DAGMA Non-linear
*   **Overview:** Extends the DAGMA acyclicity characterization to non-linear relationships. Instead of a simple weight
matrix , it parameterizes the conditional distributions using Multilayer Perceptrons (MLPs). The adjacency matrix is
then extracted from the input-layer weights of the MLPs. 
*   **Significance:** Allows to discover causal graphs in complex, real-world datasets where the additive noise model
is highly non-linear, optimizing a log-likelihood.
*   **Ref** : DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization.
Kevin Bello, Bryon Aragam, Pradeep Ravikumar. Booth School of Business, University of Chicago, Chicago, IL 60637.
Machine Learning Department, Carnegie Mellon University, Pittsburgh, PA 15213
    * **Paper:** <a href="https://proceedings.neurips.cc/paper_files/paper/2022/file/36e2967f87c3362e37cf988781a887ad-Paper
    -Conference.pdf"> Neurips-dagma </a>
    * **Code ref:** <a href="https://github.com/kevinsbello/dagma/blob/main/src/dagma/nonlinear.py">Original Dagma
    Non-Linear code </a>

#### DAGMA-DCE
*   **Overview:** Implementing advanced extensions of the DAGMA framework (such as DAGMA-DCE) adapted for specific
challenging data regimes or interventional data.
*   **Significance:** Ensures `pgmpy` is not just catching up to basic continuous optimization, but providing
state-of-the-art variants capable of handling specific causal inference edge cases.
*   **Ref** : DAGMA-DCE: Interpretable, Non-Parametric Differentiable Causal Discovery. DANIEL WAXMAN (Graduate Student
Member, IEEE), KURT BUTLER (Graduate Student Member, IEEE), AND PETAR M. DJURIC (Life Fellow, IEEE) Department of
Electrical and Computer Engineering, Stony Brook University, Stony Brook, NY 11794 USA

    * **Paper:** <a href="https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10384714"> IEEE dagma-DCE </a>
    * **Code ref:** <a href="https://github.com/DanWaxman/DAGMA-DCE">Original Dagma-DCE
    code</a>


### Details of proposed solution

### Implementation Details

All algorithms will be implemented inside `pgmpy/causal_discovery` to isolate the soft dependency on PyTorch from the rest of the library, while inheriting from both `_BaseDAGMAMixin` and `_BaseCausalDiscovery`.

The internal MLP modules will be organized as follows:
* **`LocallyConnected` & `DagmaMLP`**: Will be defined within `pgmpy/causal_discovery/DAGMANonLinear.py`.
* **`DagmaMLP_DCE`**: Will be defined within `pgmpy/causal_discovery/DAGMAdce.py` (importing `LocallyConnected` from `DAGMANonLinear.py` to prevent code duplication).

```text
pgmpy/
  causal_discovery/
    _base.py            # (Modified) Will Include _BaseDAGMAMixin for DAGMA and its variants
    DAGMA.py            # (New) DAGMALinear class
    DAGMANonLinear.py   # (New) DAGMANonlinear class and internal PyTorch MLPs
    DAGMAdce.py         # (New) DAGMADce class

pgmpy/
  tests/
    test_causal_discovery/
      test_DAGMA.py             # (New) Tests for DAGMALinear
      test_DAGMANonLinear.py    # (New) Tests for DAGMANonlinear
      test_DAGMAdce.py          # (New) Tests for DAGMADce
```

#### Shared Architectural Decisions:
*   **Pure PyTorch Backend:** All DAGMA variants will utilize PyTorch for automatic differentiation.
*   **Scikit-Learn Compatibility via Mixin:** All algorithms will be strictly Scikit-Learn compatible. The heavy optimization loops and PyTorch tensors will be contained within a stateless `_BaseDAGMA` Mixin, preserving strict `__init__` parameter declarations in the leaf estimators.
*   **Testing & CI/CD Safety:** Algorithms will be tested using `scikit-learn`'s API checks. Linear versions use `LinearGaussianBN` synthetic data, datasets such as `sachs_continuous` from pgmpy, with strict seeding (`torch.manual_seed`) to prevent flakiness in automated CI/CD pipelines.

### API Design:

**Proposed DAGMA Architecture (Scikit-Learn Compliance + Mixin Base)**
```python
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
import torch
import pandas as pd
import numpy as np
from typing import Type

class _BaseDAGMAMixin:
    """
    Mixin class implementing shared acyclicity constraint, optimization and graph reconstruction logic for DAGMA and its
    variants.
    """

    def _log_det_barrier(self, W: torch.Tensor, s: float) -> torch.Tensor:
        """
        Computes the log-determinant acyclicity barrier function:
        h(W) = -log det(sI - W o W) + d log s
        """
        ...

    def _convert_to_dag(self, W: np.ndarray, feature_names: list, w_threshold: float, return_type: str):
        """
        Thresholds the estimated weight matrix and converts it into a pgmpy DAG or CPDAG.
        """
        ...

    def _optimize(
        self,
        model: torch.nn.Module,
        optimizer_cls: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict,
        s: float,
        mu_init: float,
        mu_factor: float,
        max_iter: int,
    ) -> np.ndarray:
        """
        Unified optimization loop executing the dual-loop DAGMA optimization.

        - Outer loop: Decays the penalty parameter mu (mu = mu * mu_factor).
        - Inner loop: Minimizes the augmented Lagrangian objective function (loss + h(W)/mu)
          using the passed PyTorch optimizer class (e.g., Adam or L-BFGS).
        """
        ...

    def _resolve_device_and_dtype(self) -> tuple[torch.device, torch.dtype]:
        """
        Queries the global pgmpy configurations to resolve the PyTorch device and 
        tensor float precision (mapping string representations to torch.dtype objects).
        """
        ...

class DAGMALinear(_BaseDAGMAMixin, _BaseCausalDiscovery):
    def __init__(
        self, 
        s: float = 1.0, 
        lambda1: float = 0.05, 
        mu_init: float = 1.0, 
        mu_factor: float = 0.1, 
        max_iter: int = 100, 
        w_threshold: float = 0.3, 
        return_type: str = "dag", 
        optimizer: Type[torch.optim.Optimizer] = torch.optim.LBFGS
    ) -> None:
        """
        DAGMA Linear causal discovery estimator.
        
        Parameters
        ----------
        s : float, default=1.0
            M-matrix barrier scaling parameter. Must be greater than or equal to 1.0. 
            Ensures that (s*I - W o W) is an M-matrix by dominating the spectral radius of W o W.
        lambda1 : float, default=0.05
            L1 regularization penalty coefficient. Controls the sparsity of the learned graph 
            by penalizing non-zero edge weights.
        mu_init : float, default=1.0
            Initial penalty coefficient for the acyclicity barrier term. 
            Sets the initial weight of the acyclicity constraint in the augmented Lagrangian.
        mu_factor : float, default=0.1
            Decay factor for the penalty coefficient mu. At each outer iteration, mu is updated 
            as mu = mu * mu_factor, forcing the acyclicity constraint to become stricter.
        max_iter : int, default=100
            Maximum number of iterations allowed for the optimization.
        w_threshold : float, default=0.3
            Weight threshold for edge pruning. Any edge with absolute weight less than 
            w_threshold is removed in the final learned graph.
        return_type : str, default="dag"
            The format of the returned graph. Can be "dag" (pgmpy.base.DAG) or "cpdag" (pgmpy.base.CPDAG).
        optimizer : Type[torch.optim.Optimizer], default=torch.optim.LBFGS
            The uninstantiated PyTorch optimizer class to use for minimizing the continuous loss.
            LBFGS is highly recommended for the linear case due to the log-det barrier curvature.
        """
        self.s = s
        self.lambda1 = lambda1
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.return_type = return_type
        self.optimizer = optimizer

    def _fit(self, X: pd.DataFrame) -> DAGMALinear:
        # 1. Resolve device & dtype, convert X to PyTorch tensors
        # 2. Extract covariance matrix
        # 3. Call self._optimize(model, self.optimizer, ...)
        # 4. Return self._convert_to_dag(W_est) and assign to self.causal_graph_
        ...

class DAGMANonlinear(_BaseDAGMAMixin, _BaseCausalDiscovery):
    def __init__(
        self, 
        hidden_dims: tuple[int, ...] = (10,), 
        s: float = 1.0, 
        lambda1: float = 0.02, 
        lambda2: float = 0.005, 
        mu_init: float = 0.1, 
        mu_factor: float = 0.1, 
        max_iter: int = 80000, 
        lr: float = 0.0002, 
        w_threshold: float = 0.3, 
        return_type: str = "dag", 
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    ) -> None:
        """
        DAGMA Non-linear causal discovery estimator using neural networks (MLPs).
        
        The internal MLP layers are configured with bias=True (hardcoded).
        Hardcoding bias to True ensures that the MLP layers have the capacity to fit off-center data and prevents dead
        neurons during gradient backpropagation, which is critical for learning non-linear structures.
        
        Parameters
        ----------
        hidden_dims : tuple of int, default=(10,)
            Dimensions of the hidden layers in the Multilayer Perceptrons (MLPs).
            Each variable is modeled using a separate MLP with these hidden dimensions.
        s : float, default=1.0
            M-matrix barrier scaling parameter. Must be greater than or equal to 1.0. 
            Ensures that the acyclicity constraint is properly defined relative to the MLP input layer weights.
        lambda1 : float, default=0.02
            L1 regularization penalty coefficient on the input weights of the MLPs. 
            Controls the sparsity of the graph by penalizing connections from other variables.
        lambda2 : float, default=0.005
            L2 regularization penalty coefficient (weight decay) on all network weights. 
            Helps prevent overfitting and stabilizes training.
        mu_init : float, default=0.1
            Initial penalty coefficient for the acyclicity barrier term.
        mu_factor : float, default=0.1
            Decay factor for the penalty coefficient mu. At each outer step, mu = mu * mu_factor.
        max_iter : int, default=80000
            Maximum number of optimization steps/iterations.
        lr : float, default=0.0002
            Learning rate for the optimizer.
        w_threshold : float, default=0.3
            Weight threshold for edge pruning. Any edge with absolute weight less than 
            w_threshold is removed in the final learned graph.
        return_type : str, default="dag"
            The format of the returned graph. Can be "dag" (pgmpy.base.DAG) or "cpdag" (pgmpy.base.CPDAG).
        optimizer : Type[torch.optim.Optimizer], default=torch.optim.Adam
            The uninstantiated PyTorch optimizer class to use for minimizing the continuous loss.
            Adam is recommended for neural network optimization.
        """
        self.hidden_dims = hidden_dims
        self.s = s
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.lr = lr
        self.w_threshold = w_threshold
        self.return_type = return_type
        self.optimizer = optimizer

    def _fit(self, X: pd.DataFrame) -> DAGMANonlinear:
        # 1. Resolve device & dtype, convert X to PyTorch tensors
        # 2. Instantiate standard DagmaMLP using self.hidden_dims (with bias=True)
        # 3. Call self._optimize(model, self.optimizer, ...)
        # 4. Return self._convert_to_dag(W_est) and assign to self.causal_graph_
        ...

class DAGMADce(_BaseDAGMAMixin, _BaseCausalDiscovery):
    def __init__(
        self, 
        hidden_dims: tuple[int, ...] = (10,), 
        s: float = 1.0, 
        lambda1: float = 0.02, 
        lambda2: float = 0.005, 
        mu_init: float = 1.0, 
        mu_factor: float = 0.1, 
        max_iter: int = 8000, 
        lr: float = 1e-3, 
        w_threshold: float = 0.3, 
        return_type: str = "dag", 
        optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    ) -> None:
        """
        DAGMA-DCE (Differentiable Causal Effects) causal discovery estimator.
        
        Note: The internal MLP layers are configured with bias=True (hardcoded).
        Hardcoding bias to True ensures that the MLP layers have the capacity to fit off-center data 
        and prevents dead neurons during gradient backpropagation, which is critical for learning non-linear structures.
        
        Parameters
        ----------
        hidden_dims : tuple of int, default=(10,)
            Dimensions of the hidden layers in the Multilayer Perceptrons (MLPs).
        s : float, default=1.0
            M-matrix barrier scaling parameter. Must be greater than or equal to 1.0.
        lambda1 : float, default=0.02
            L1 regularization penalty coefficient on the Jacobian of the MLPs.
            Enforces structural sparsity by penalizing the norms of the partial derivatives (causal effects).
        lambda2 : float, default=0.005
            L2 regularization penalty coefficient (weight decay) on all network weights.
        mu_init : float, default=1.0
            Initial penalty coefficient for the acyclicity barrier term.
        mu_factor : float, default=0.1
            Decay factor for the penalty coefficient mu. At each outer step, mu = mu * mu_factor.
        max_iter : int, default=8000
            Maximum number of optimization steps/iterations.
        lr : float, default=1e-3
            Learning rate for the optimizer.
        w_threshold : float, default=0.3
            Weight threshold for edge pruning.
        return_type : str, default="dag"
            The format of the returned graph. Can be "dag" (pgmpy.base.DAG) or "cpdag" (pgmpy.base.CPDAG).
        optimizer : Type[torch.optim.Optimizer], default=torch.optim.Adam
            The uninstantiated PyTorch optimizer class to use for minimizing the continuous loss.
        """
        self.hidden_dims = hidden_dims
        self.s = s
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.lr = lr
        self.w_threshold = w_threshold
        self.return_type = return_type
        self.optimizer = optimizer

    def _fit(self, X: pd.DataFrame) -> DAGMADce:
        # 1. Resolve device & dtype, convert X to PyTorch tensors
        # 2. Instantiate DagmaMLP_DCE (computes Jacobian, with bias=True)
        # 3. Call self._optimize(model, self.optimizer, ...)
        # 4. Return self._convert_to_dag(W_est) and assign to self.causal_graph_
        ...

# Internal PyTorch Neural Network Modules for Non-linear Estimators

class LocallyConnected(torch.nn.Module):
    """
    Implements a local linear layer, batching independent linear transformations
    for each variable in parallel. Equivalent to 1D local convolution with filter size 1.
    """
    def __init__(self, num_linear: int, input_features: int, output_features: int, bias: bool = True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.Tensor(num_linear, input_features, output_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(input.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            out += self.bias
        return out


class DagmaMLP(torch.nn.Module):
    """
    Neural network module that models structural equations for standard Non-linear DAGMA.
    Uses LocallyConnected layers for hidden representations.
    """
    def __init__(self, dims: list[int], bias: bool = True):
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.fc1 = torch.nn.Linear(self.d, self.d * dims[1], bias=bias)
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        return x.squeeze(dim=2)

    def get_w(self) -> torch.Tensor:
        """
        Computes the continuous weighted adjacency matrix from the first linear layer weights.
        W[i, j] is the L2 norm of the weights mapping input i to variable j.
        """
        fc1_weight = self.fc1.weight.view(self.d, -1, self.d)
        A = torch.sum(fc1_weight ** 2, dim=1).t()
        return torch.sqrt(A)

    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """Computes log MSE loss (likelihood proxy for continuous variables)."""
        n, d = X.shape
        return 0.5 * d * torch.log(1 / n * torch.sum((X_hat - X) ** 2))

    def get_l1_reg(self) -> torch.Tensor:
        """Computes L1 norm of the first fully-connected layer weights."""
        return torch.sum(torch.abs(self.fc1.weight))


class DagmaMLP_DCE(torch.nn.Module):
    """
    Neural network module that models structural equations for DAGMA-DCE.
    Computes the adjacency matrix dynamically using the Jacobian over sample inputs.
    """
    def __init__(self, dims: list[int], bias: bool = True):
        super().__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        self.dims, self.d = dims, dims[0]
        self.fc1 = torch.nn.Linear(self.d, self.d * dims[1], bias=bias)
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(self.d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = torch.nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = x.view(-1, self.dims[0], self.dims[1])
        for fc in self.fc2:
            x = torch.sigmoid(x)
            x = fc(x)
        return x.squeeze(dim=2)

    def get_graph(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the root-mean-square (RMS) Jacobian matrix over the data points.
        Returns:
            W (torch.Tensor): RMS Jacobian adjacency matrix of shape (d, d).
            observed_deriv (torch.Tensor): Batched Jacobian evaluations of shape (n, d, d).
        """
        x_dummy = x.detach().requires_grad_()
        observed_deriv = torch.func.vmap(torch.func.jacrev(self.forward))(x_dummy).view(-1, self.d, self.d)
        W = torch.sqrt(torch.mean(observed_deriv ** 2, axis=0).T)
        return W, observed_deriv

    def loss(self, X: torch.Tensor, X_hat: torch.Tensor) -> torch.Tensor:
        """Computes standard sum-squared MSE loss divided by 2N."""
        n, d = X.shape
        return 0.5 / n * torch.sum((X_hat - X) ** 2)

    def get_l1_reg(self, observed_derivs: torch.Tensor) -> torch.Tensor:
        """Computes structural L1 norm over the mean Jacobian derivatives."""
        return torch.sum(torch.abs(torch.mean(observed_derivs, axis=0)))
```

**Testing Non-Linear Estimators (e.g., test_DAGMANonLinear.py)**
To ensure CI/CD stability and align with `pgmpy`'s testing conventions, the non-linear test suites will be structured as follows:

*   **Scikit-Learn Compatibility Check:** Use `@parametrize_with_checks` along with an `expected_failed_checks` helper function to enforce API consistency, matching standard `pgmpy` discovery tests.
*   **Core Estimation & Accuracy:** Combine functional and accuracy tests into a single test (e.g., `test_estimate_dag`). This test will load a standardized dataset (like `sachs_continuous`), enforce strict `torch.manual_seed()` for CI/CD determinism, run the `fit()` method, and assert that the Structural Hamming Distance (SHD) against the known ground truth is within acceptable bounds.
*   **Configuration & Variant Testing:** A dedicated test to verify that custom arguments (`hidden_dims`, `lambda2`, `optimizer`) map cleanly to the estimator instance.
*   **Failure Modes & Edge Cases:** Tests to explicitly trigger and catch expected failures .

