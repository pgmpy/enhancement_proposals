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

The detailed implementation steps, key design decisions, API, and tests for each algorithm are provided in separate files:
- [CASTLE Implementation Details](castle.md)
- [DiffAN Implementation Details](diffan.md)
- [GraN-DAG Implementation Details](gran-dag.md)
- [CAREFL Implementation Details](carefl.md)
