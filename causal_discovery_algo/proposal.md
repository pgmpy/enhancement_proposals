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

To integrate `DAGMALinear` (L-BFGS), `DAGMANonlinear` (Adam), and `DAGMADce` (Adam) without code duplication while adhering strictly to `scikit-learn` guidelines, this proposal will use a **Mixin/Logic Container** architecture.

* **`_BaseDAGMA` as a Mixin:** This introduces a `_BaseDAGMA` class without an `__init__` (As the proposed algorithms do not share common arguments). It holds the shared mathematical logic (`_log_det_barrier`, `_convert_to_dag`) and implements centralized optimization loops (`_optimize_lbfgs` and `_optimize_adam`). 
* **Model-Agnostic Loops:** The individual algorithms (Linear, Nonlinear, DCE) act as wrappers. They define their unique parameters in `__init__`, construct their specific objective functions (e.g., standard MLP vs DCE Jacobian), and call the appropriate optimization loop from the base class.

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
All algorithms will be implemented inside `pgmpy/causal_discovery` to isolate the soft dependency on PyTorch from the
rest of the library, while fully inheriting from `_BaseCausalDiscovery`.

```text
pgmpy/
  causal_discovery/
    _base.py            # _BaseCausalDiscovery (existing)
    DAGMA.py            # _BaseDAGMA Mixin and DAGMALinear class
    DAGMANonLinear.py   # DAGMANonlinear class and internal PyTorch MLPs
    DAGMAdce.py         # DAGMADce class

pgmpy/
  tests/
    test_causal_discovery/
      test_DAGMA.py             # Tests for DAGMALinear
      test_DAGMANonLinear.py    # Tests for DAGMANonlinear
      test_DAGMAdce.py          # Tests for DAGMADce
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

class _BaseDAGMA(_BaseCausalDiscovery):
    """
    Acts as a Logic Container/Mixin. 
    Does NOT have an __init__ to preserve strict scikit-learn compatibility in subclasses.
    """
    # Method to implement $h(W) = -\log \det(sI - W \circ W) + d \log s$ function
        # Shared mathematical logic for h(W) acyclicity constraint
        ...

    # method to convert the weightd from different algorithms into valid DAG
        # Shared logic for pruning below w_threshold and casting to pgmpy.base.DAG
        ...

    def _optimize_lbfgs(self, cov_tensor, n_features, ...):
        # Centralized PyTorch L-BFGS loop over mu decay
        ...

    def _optimize_adam(self, model: torch.nn.Module, X_tensor, ...):
        # Centralized Adam loop over mu decay. 
        # Relies on standardized methods in the model (e.g., model.get_l1_reg(), model.get_h_val())
        ...

class DAGMALinear(_BaseDAGMA):
    def __init__(
        self, s=1.0, lambda1=0.05, mu_init=1.0, mu_factor=0.1, 
        max_iter=100, w_threshold=0.3, return_type="dag", optimizer="lbfgs"
    ) -> None:
        self.s = s
        self.lambda1 = lambda1
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.return_type = return_type
        self.optimizer = optimizer

    def _fit(self, X: pd.DataFrame):
        # 1. Resolve device & dtype, convert X to PyTorch tensors
        # 2. Extract covariance matrix
        # 3. Call self._optimize_lbfgs() (or self._optimize_adam() if requested)
        # 4. Return self._convert_to_dag(W_est)
        ...

class DAGMANonlinear(_BaseDAGMA):
    def __init__(
        self, hidden_dims=(10,), bias=True, s=1.0, lambda1=0.02, lambda2=0.005, 
        mu_init=0.1, mu_factor=0.1, max_iter=80000, lr=0.0002, 
        w_threshold=0.3, return_type="dag", optimizer="adam"
    ) -> None:
        self.hidden_dims = hidden_dims
        self.bias = bias
        self.s = s
        ...
        self.optimizer = optimizer

    def _fit(self, X: pd.DataFrame):
        # 1. Resolve device & dtype, convert X to PyTorch tensors
        # 2. Instantiate standard DagmaMLP using self.hidden_dims
        # 3. Call self._optimize_adam(model, X)
        # 4. Return self._convert_to_dag(W_est)
        ...

class DAGMADce(_BaseDAGMA):
    def __init__(
        self, hidden_dims=(10,), bias=True, s=1.0, lambda1=0.02, lambda2=0.005, 
        mu_init=1.0, mu_factor=0.1, max_iter=8000, lr=1e-3, 
        w_threshold=0.3, return_type="dag", optimizer="adam"
    ) -> None:
        self.hidden_dims = hidden_dims
        ...
        self.optimizer = optimizer

    def _fit(self, X: pd.DataFrame):
        # 1. Resolve device & dtype, convert X to PyTorch tensors
        # 2. Instantiate DagmaMLP_DCE (computes Jacobian instead of FC1 weights)
        # 3. Call self._optimize_adam(model, X)
        # 4. Return self._convert_to_dag(W_est)
        ...
```

**Testing Non-Linear Estimators (e.g., test_DAGMANonLinear.py)**
To ensure CI/CD stability and align with `pgmpy`'s testing conventions, the non-linear test suites will be structured as follows:

*   **Scikit-Learn Compatibility Check:** Use `@parametrize_with_checks` along with an `expected_failed_checks` helper function to enforce API consistency, matching standard `pgmpy` discovery tests.
*   **Core Estimation & Accuracy:** Combine functional and accuracy tests into a single test (e.g., `test_estimate_dag`). This test will load a standardized dataset (like `sachs_continuous`), enforce strict `torch.manual_seed()` for CI/CD determinism, run the `fit()` method, and assert that the Structural Hamming Distance (SHD) against the known ground truth is within acceptable bounds.
*   **Configuration & Variant Testing:** A dedicated test to verify that custom arguments (`hidden_dims`, `lambda2`, `optimizer`) map cleanly to the estimator instance.
*   **Failure Modes & Edge Cases:** Tests to explicitly trigger and catch expected failures .

