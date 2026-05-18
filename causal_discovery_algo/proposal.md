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

### Alternative Solutions

#### DAGMA Base Class Abstraction Levels

When considering to integrate both `DAGMALinear` (which uses L-BFGS (The original still uses Adam)) and `DAGMANonLinear` (which uses Adam) without code duplication, three levels of abstraction could be thought of:

1. **Approach A: Shared Boilerplate + Math Utilities**
   * **Description:** Create a `_BaseDAGMA` class that handles `__init__` parameters, data validation, tensor conversion, device config, the core `_log_det_barrier` mathematical logic, and final `DAG` object creation. Subclasses only implement their specific `_fit` optimization loops (L-BFGS vs. Adam) and neural network/matrix setups.
   * **Pros:** Clean, pythonic, avoids forcing incompatible optimizers together, respects DRY, and centralizes `sklearn` parameter getters and setters.
   * **Cons:** The outer `mu` decay loop is still written twice.

2. **Approach B: Strict Template Method (Full Loop Abstraction)**
   * **Description:** Abstract the entire outer `mu` decay loop into `_BaseDAGMA._fit`. The base class loops over `mu` and calls a subclass-implemented `_optimize_inner_step()`.
   * **Pros:** Zero duplication of the optimization flow.
   * **Cons:** Very brittle. Adam requires completely different stopping logic, scheduling, and state tracking (e.g. `obj_prev`) compared to L-BFGS.

3. **Approach C: Stick to `_BaseCausalDiscovery` Only**
   * **Description:** No specific DAGMA base class. Both `DAGMALinear` and `DAGMANonlinear` inherit directly from `_BaseCausalDiscovery`.
   * **Pros:** Maximum flexibility for each algorithm.
   * **Cons:** Significant code duplication of parameter ingestion, PyTorch tensor conversion, the log-det barrier math, and thresholding logic.

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
    _base.py        # _BaseCausalDiscovery (existing)
    DAGMA.py        # DAGMALinear class (Ongoing) + DAGMANonLinear class(New)
    DAGMAdce.py     # DAGMA-DCE (New)

pgmpy/
  tests/
    test_causal_discovery/
      test_DAGMA.py       # Tests for DAGMALinear (Ongoing) and DAGMANonLinear classes (New)
      test_DAGMAdce.py    # Tests for DAGMAdce (New)
```

#### Shared Architectural Decisions:
*   **Pure PyTorch Backend:** All DAGMA variants will utilize PyTorch.
*   **Scikit-Learn Compatibility:** All algorithms will be implemented to be Scikit-Learn compatible as required by
pgmpy.
*   **Test**: All algorithms will be tested to be Scikit-Learn compatible, tested on synthetic data using
LinearGaussiaBN and against the known implementation of the algorithm.

### API Design:

**DAGMA Architecture (Approach A: Shared BaseDAGMA class)**
```python
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
import torch
import pandas as pd

class _BaseDAGMA(_BaseCausalDiscovery):
    def __init__(
        self,
        s=1.0,
        lambda1=0.05,
        mu_init=1.0,
        mu_factor=0.1,
        max_iter=100,
        w_threshold=0.3,
        return_type: str = "dag",
    ) -> None:

        self.s = s
        self.lambda1 = lambda1
        self.mu_init = mu_init
        self.mu_factor = mu_factor
        self.max_iter = max_iter
        self.w_threshold = w_threshold
        self.return_type = return_type
        self.device = config.get_device()
        dtype = config.get_dtype()
        if isinstance(dtype, str):
            self.dtype = getattr(torch, dtype)
        else:
            self.dtype = dtype

        # Handles hyperparameters and pgmpy config resolution (device, dtype)
        ...

    def _log_det_barrier(self, W: torch.Tensor, s: float) -> torch.Tensor:
        # Shared mathematical logic for h(W) acyclicity constraint
        ...

    def _convert_to_dag(self, W_est) -> 'DAG':
        # Shared logic for pruning below w_threshold and casting to pgmpy.base.DAG
        ...

class DAGMALinear(_BaseDAGMA):
    def _fit(self, X: pd.DataFrame):
        # Converts X to Tensor and extracts covariance
        # Executes mu decay loop using PyTorch L-BFGS optimizer
        # Calls self._log_det_barrier() and builds the objective
        # Returns self._convert_to_dag(W_est)
        ...

class DAGMANonlinear(_BaseDAGMA):
    def __init__(self, model_dims, **kwargs):
        super().__init__(**kwargs)
        # Initializes the Multilayer Perceptron (MLP) for structural equations
        ...

    def _fit(self, X: pd.DataFrame):
        # Executes mu decay loop using Adam optimizer (with scheduling/tol stopping)
        # Computes log-MSE score and calls self._log_det_barrier() via MLP weights
        # Returns self._convert_to_dag(W_est)
        ...
```

**test_DAGMA**
```python
from pgmpy.base import DAG
from pgmpy.causal_discovery.DAGMA import DAGMALinear
from pgmpy.metrics import SHD
from pgmpy.models import LinearGaussianBayesianNetwork

def expected_failed_checks(estimator):
    ...

@parametrize_with_checks(
    [DAGMALinear()],
    expected_failed_checks=expected_failed_checks,
)

def test_dagma_compatibility(estimator, check):
    ...

def continuous_data():

class TestDagmaLinearCore:

  def test_estimate_returns_dag(self, continuous_data):
    ...

  def test_against_linear_gaussian_bn(self):

    ...

  def test_custom_hyperparameters(self):
    ...

  def test_compare_with_official_dagma(self):
    ...

```

### User Journeys with the Solution

#### 1. High-Dimensional Linear Discovery (DAGMALinear/DAGMANonLinear)
**Scenario:** A bio engineer is analyzing protien structure data (like pgmpy's sachs dataset) containing 100+ variables.
Traditional combinatorial methods like `HillClimbSearch` fail to converge or take days to run.<br>
**Journey:** The user imports `DAGMALinear` (or `DAGMANonLinear` for more accuracy and precision) from
`pgmpy.causal_discovery`. Since the data is continuous and assumed to follow a linear Gaussian structural model,
they pass the pandas DataFrame directly to the `fit` method. Because `DAGMALinear` uses PyTorch internally, the user
can set `pgmpy.config.set_backend("torch", device="cuda")` to accelerate the L-BFGS optimization on their GPU. Within
minutes, the algorithm returns a `pgmpy.base.DAG` object representing the protein signaling network, ready for
downstream causal inference.

#### 2. Interpretable Causal Discovery with DAGMA-DCE
**Scenario:** A data scientist in the healthcare sector needs to discover a causal graph from patient health records.
Strict regulatory requirements demand that the learned relationships be completely transparent and quantifiable,
meaning standard black-box causal discovery tools are non-compliant because their internal scoring matrices lack
real-world physical meaning.<br>
**Journey:** The user applies DAGMA-DCE (Differentiable Causal Effects). While the framework still allows the
underlying data patterns to be captured using flexible, non-linear neural networks (MLPs), it replaces standard
uninterpretable edge-weight proxies with a non-parametric formulation based on expected derivatives. The user fits the
model to their data. Not only do they receive a valid DAG, but the learned weighted adjacency matrix provides direct,
human-interpretable estimates of the causal strengths between health indicators. This fulfills their medical compliance
requirements while leveraging state-of-the-art continuous optimization.

---

### Empirical Validation: Adam vs. L-BFGS

As documented during initial planning, `DAGMALinear` in `pgmpy` utilizes PyTorch's **L-BFGS** optimizer instead of the original **Adam** implementation. 

Below is the benchmark testing script that empirically validates why L-BFGS is preferred for the linear case. Adam is highly sensitive to feature scaling and frequently gets trapped in cyclic local minima when applied to raw or mean-centered variance. L-BFGS, being a second-order optimizer, uses curvature information to navigate the sharp log-det barrier constraints more robustly.

#### Benchmark Test Script
```python
import pandas as pd
import numpy as np
import networkx as nx
from pgmpy.datasets import load_dataset
from pgmpy.causal_discovery.DAGMA import DAGMALinear
import sys
import os

# Add the official dagma (used local clone here) / pip install dagma
sys.path.append(os.path.abspath('PATH_TO_DAGMA-OFFICIAL'))
from dagma.linear import DagmaLinear as OfficialDagmaLinear

def run_tests():
    data = load_dataset('sachs_continuous').data
    
    data_raw = data.to_numpy().copy()
    data_centred = (data - data.mean()).to_numpy().copy()
    data_std = ((data - data.mean()) / data.std()).to_numpy().copy()
    
    scenarios = {
        "Raw Data": data_raw,
        "Mean-Centered": data_centered,
        "Standardized": data_std
    }
    
    for name, X in scenarios.items():
        print(f"\n--- Scenario: {name} ---")
        
        # pgmpy (L-BFGS)
        print("pgmpy (L-BFGS):")
        try:
            df_X = pd.DataFrame(X, columns=data.columns)
            est_pgmpy = DAGMALinear(lambda1=0.05, w_threshold=0.3, s=1.0)
            est_pgmpy.fit(df_X)
            
            is_dag_pgmpy = nx.is_directed_acyclic_graph(est_pgmpy.causal_graph_)
            print(f"  Valid DAG: {is_dag_pgmpy}")
            if not is_dag_pgmpy:
                cycles = list(nx.simple_cycles(est_pgmpy.causal_graph_))
                print(f"  Failed (Cyclic). Found {len(cycles)} cycles.")
            else:
                print(f"  Edge count: {len(list(est_pgmpy.causal_graph_.edges()))}")
        except Exception as e:
            print(f"  Failed: {e}")
            
        # Official (Adam)
        print("Official (Adam):")
        try:
            model_official = OfficialDagmaLinear(loss_type='l2')
            W_official = model_official.fit(X, lambda1=0.05, w_threshold=0.3, s=1.0, T=5)
            
            df_adj = pd.DataFrame(W_official, index=data.columns, columns=data.columns)
            nx_official = nx.from_pandas_adjacency(df_adj, create_using=nx.DiGraph)
            is_dag_off = nx.is_directed_acyclic_graph(nx_official)
            
            print(f"  Valid DAG: {is_dag_off}")
            if not is_dag_off:
                cycles = list(nx.simple_cycles(nx_official))
                print(f"  Failed (Cyclic). Found {len(cycles)} cycles.")
            else:
                print(f"  Edge count: {len(nx_official.edges())}")
        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == '__main__':
    run_tests()
```

#### Results
```text
--- Scenario: Raw Data ---
pgmpy (L-BFGS):
  Valid DAG: True
  Edge count: 8
Official (Adam):
  Valid DAG: False
  Failed (Cyclic). Found 5 cycles.

--- Scenario: Mean-Centered ---
pgmpy (L-BFGS):
  Valid DAG: True
  Edge count: 8
Official (Adam):
  Valid DAG: False
  Failed (Cyclic). Found 5 cycles.

--- Scenario: Standardized ---
pgmpy (L-BFGS):
  Valid DAG: True
  Edge count: 2
Official (Adam):
  Valid DAG: True
  Edge count: 6
```

**Conclusion:** `pgmpy`'s L-BFGS implementation successfully captures 8 valid edges on the natural raw and mean-centered data without falling into cycles. The Adam optimizer fail on the same natural data, only succeeding when the variance is heavily damped via unit standardization (which in turn crushes L-BFGS edge discovery to just 2).
