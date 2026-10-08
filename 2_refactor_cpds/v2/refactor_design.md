## CPD Refactor Design Overview

Contributors: @ankurankan, @daehyun

> **Companion documents:**
> - `class_contracts.md` — combined API/contracts reference across all sub-proposals.
> - `01_cpd_boundary.md` — local CPD contract, adapters, tags, and lifecycle.
> - `02_parameterized_dag.md` — DAG-owned CPD registry, schema, inference migration, and compatibility.
> - `03_scm_counterfactuals.md` — optional SCM / intervention / counterfactual layer.
> - `prototype/FINDINGS.md` — prototype results and gaps.

### Why split this proposal?

The previous `v2` draft bundled three distinct architectural decisions into a single design:

1. A new framework-agnostic CPD boundary built on sklearn/skpro-style estimators.
2. A DAG-level refactor that centralizes parameter registration, schema ownership, and inference dispatch.
3. A new SCM layer for interventions, counterfactuals, diagnostics, and bootstrap-style uncertainty.

Those pieces do not carry the same risk.

- The **CPD boundary** is the lowest-level architectural change and is valuable on its own.
- The **parameterized DAG** work is large, but still closely tied to existing pgmpy responsibilities.
- The **SCM layer** is the most opinionated part of the design and introduces new semantics, especially for discrete counterfactuals.

Splitting the design allows them to be reviewed, accepted, and implemented independently.

### Shared motivation

A pgmpy Bayesian network is currently parameterized by three unrelated CPD families:
`TabularCPD`, `LinearGaussianCPD`, and `FunctionalCPD`. In the current codebase:

- CPDs carry graph identity (`variable`, `evidence`, or `parents`).
- model classes duplicate CPD management, `fit`, and `simulate` logic across discrete, Gaussian, and functional paths.
- inference and sampling rely on concrete CPD classes and factor-specific methods in many places.
- state metadata is spread across CPDs, models, and serialization formats.

The split docs keep the same overall goal: make CPDs more reusable, reduce model duplication, and enable broader parameterization shapes without requiring bespoke pgmpy wrappers.

### Proposal map

| Document | Primary question | Depends on |
|---|---|---|
| `01_cpd_boundary.md` | What is a CPD in the new design? | None |
| `02_parameterized_dag.md` | How does a parameterized `DAG` own and consume those CPDs? | `01` |
| `03_scm_counterfactuals.md` | Should pgmpy add a first-class SCM/counterfactual layer on top? | `01`, optionally `02` |

### Shared design principles

- **Identity-free local models.** CPDs should not need to know which node they parameterize.
- **DAG-owned structure metadata.** Parent order and schema belong to the graph layer, not to individual CPDs.
- **Protocol and capability-based dispatch.** Operations should check for required capabilities, not hard-code concrete classes wherever possible.
- **Additive 1.x migration before any 2.0 deletions.** Core compatibility shims should land before major removals.
- **Separable acceptance.** Design 3 should not block acceptance of Designs 1 and 2.

### Recommended review order

1. Review `01_cpd_boundary.md` first.
2. Review `02_parameterized_dag.md` second, using Design 1 as its substrate.
3. Review `03_scm_counterfactuals.md` last, as an optional extension rather than part of the core refactor.

### What remains intentionally unsplit

`class_contracts.md` is still a combined reference document. It spans all three sub-proposals so reviewers can compare the full surface in one place. If the split designs stabilize, the contracts doc can be split later along the same boundaries.

### What this overview does not do

- It does not preserve the old monolithic narrative in full.
- It does not define an implementation plan.
- It does not force the three sub-proposals to land in the same release.

The detailed design now lives in the three scoped docs below.
