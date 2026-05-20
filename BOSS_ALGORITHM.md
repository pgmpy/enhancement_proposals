## [Title]

Contributors: anusa-saha

### Introduction

Extension of Causal Discovery Algorithms

### Proposed Solution

Proposal for adding the Best Order Score Search (BOSS) algorithm to pgmpy as a new score-based causal discovery method.

BOSS is a permutation-based causal discovery algorithm that searches over variable orderings instead of directly searching over graph structures. For a given permutation of variables, BOSS constructs a DAG by selecting parents from predecessor variables using a Grow-Shrink procedure and evaluates the resulting structure using a decomposable score such as BIC.

Implementing the core BOSS search framework first provides a strong foundation for future extensions mentioned below

### Alternative Solutions

Several alternative approaches and extensions related to BOSS already exist in the causal discovery literature.
The current proposal focuses on implementing the core BOSS algorithm first because it provides the foundational permutation-search framework on top of which these extensions are built.

1. Efficient Latent Variable Causal Discovery: Combining Score Search and Targeted Testing
https://arxiv.org/abs/2510.04263

2. Scalable Causal Discovery from Recursive Nonlinear Data via Truncated Basis Function Scores and Tests
https://arxiv.org/abs/2510.04276

3. Learning Causal Structure of Time Series using Best Order Score Search
https://arxiv.org/abs/2603.05370


### Details of proposed solution

The proposed way of implementing this
The algorithm will be added to: `pgmpy/causal_discovery/BOSS.py`

Since BOSS performs permutation-based score-based causal discovery, it should inherit from:

    - `_BaseCausalDiscovery`
    - `_ScoreMixin`

both of which are defined in:
`pgmpy/causal_discovery/_base.py`

`_ScoreMixin` already provides functionality for score computation and score deltas, which can be reused during Grow-Shrink parent selection and BES refinement.

#### Public API:

`class BOSS(_ScoreMixin, _BaseCausalDiscovery)`

#### Constructor used:
```
def __init__(
    self,
    scoring_method: Optional[str] = "bic",
    start_perm: Optional[List[str]] = None,
    max_iter: int = 100,
    random_state: Optional[int] = None,
):
```
| Parameter        | Description                                                   |
| ---------------- | ------------------------------------------------------------- |
| `scoring_method` | Score function used for evaluating structures (e.g., `"bic"`) |
| `start_perm`     | Optional starting permutation of variables                    |
| `max_iter`       | Maximum number of search iterations                           |
| `random_state`   | Random seed for reproducibility                               |

If `start_perm` is not provided:
    a random permutation of variables will be initialized internally using random_state.

BOSS always performs:
    - permutation search
    - permutation projection
    - BES refinement

Therefore BES will not be a user argument.

#### Algorithmic Intuition:

BOSS searches over permutations of variables and greedily improves them using a best-move operator.

For each permutation:
    - Variables are processed in permutation order.
    - Parent sets are selected using Grow-Shrink search over predecessors.
    - The permutation is projected into a DAG.
    - The DAG is refined using BES.

The search iteratively improves the permutation until convergence.

The implementation intentionally follows the linear structure of the original BOSS paper:
    - initialize permutation
    - optimize permutation
    - project to DAG
    - run BES
    - return graph

The implementation avoids deeply nested helper methods and keeps the main algorithmic flow inside `_fit()`.

```
BOSS
 ├── __init__()
 ├── _fit()
 ├── _best_move()
 ├── _score_permutation()
 └── _run_bes()
```


### User journeys with the solution

1. Learning a causal DAG from tabular data

A user has observational tabular data and wants to recover the underlying causal structure using permutation-based causal discovery.

2. Recovering dense causal graphs

BOSS is particularly useful in settings where:
    - graphs are dense
    - variable count is high
    - permutation-based methods outperform local edge search

such as: neuroscience, genomics, financial systems

Users can directly apply BOSS to high-dimensional tabular datasets using the same API as other pgmpy discovery methods.
