# Debug Recorder and Interactive Viewer for Causal Discovery Algorithms

## Contributors

- @ankurankan

## Introduction

Causal discovery algorithms in pgmpy (`PC`, `GES`, `HillClimbSearch`, plus the newer in-progress `TOPIC`,
`ExpertInLoop`, `DAGMA`, `LiNGAM`, `CASTLE`) currently expose only the final estimated graph. There is no built-in way
to inspect what the algorithm did along the way: which CI tests were performed, which operations were scored, which
edges were added / removed / oriented at each iteration, what the optimizer's loss and weighted-matrix state looked like
across iterations.

This makes three workflows painful:

1. **Debugging** algorithm behavior on real data: When an unexpected edge is or isn't present in the output, the only
   recourse today is to add ad-hoc `print` statements or step through the algorithm in a debugger.
2. **Teaching**: Learners cannot watch the graph evolve step by step or see the decisions (CI tests, score evaluations)
   that drove each change. Static final graphs hide the pedagogically interesting parts of these algorithms.
3. **Implementing new algorithms**: there is no scaffolded way to introspect intermediate state during development of
   new algorithms.

The goal is a small debugging subsystem: a `DebugRecorder` that algorithm emit events into via a single function call,
plus an interactive viewer that turns a recorded run into a self-contained HTML view (notebook, browser, or embedded in
webpages). The design is deliberately scoped to **small graphs** (≤ ~50 nodes), since this is a development / teaching
tool, not for use on massive graphs/datasets.

## Proposed Solution

Add a single new module `pgmpy/causal_discovery/debug.py` containing three pieces (kept inside `causal_discovery/`
because this is a causal-discovery-specific feature; consolidated into one file because the components are tightly
coupled and the module stays manageable):

1. **`Event`** — a frozen dataclass capturing one moment in algorithm execution: event type, graph snapshot at that
   moment, optional phase tag, free-form metadata dict, and a monotonically increasing step index.

   ```python
   @dataclass(frozen=True)
   class Event:
       event_type: str           # e.g. "ci_test", "edge_removed", "optimizer_iter"
       graph: nx.Graph           # deep copy of the graph at record time
       phase: str | None         # advisory marker, e.g. "skeleton-depth-1"
       data: dict                # algorithm-specific extras (p_value, score_delta, weights, ...)
       step: int                 # auto-assigned, monotonic
   ```

2. **`DebugRecorder`** — append-only event log with serialization and viewer convenience methods.

   ```python
   class DebugRecorder:
       events: list[Event]

       def __init__(self) -> None: ...

       def record(
           self,
           event_type: str,
           graph: nx.Graph,
           *,
           phase: str | None = None,
           **data: Any,
       ) -> None:
           """Snapshot `graph` and append an Event with the given type/phase/data."""

       def save(self, path: str | Path) -> None:
           """Write events + graphs (networkx node-link) + ndarrays to JSON."""

       @classmethod
       def load(cls, path: str | Path) -> "DebugRecorder":
           """Inverse of save()."""

       def show(self, **plot_opts: Any) -> None:
           """output_notebook() + show(plot(self, **plot_opts))."""

       def save_html(self, path: str | Path, **plot_opts: Any) -> None:
           """output_file(path) + save(plot(self, **plot_opts))."""
   ```

3. **`plot()`** — pure function that consumes a recorder and returns a Bokeh layout containing a graph view, per-event
   metadata panel, timeseries panel for auto-detected scalars, and slider / play controls / filter widgets.

   ```python
   def plot(
       recorder: DebugRecorder,
       *,
       layout: Callable | dict | None = None,    # nx layout fn or precomputed pos dict
       show_timeseries: bool = True,
       initial_filter: list[str] | None = None,  # event types visible initially
       title: str | None = None,
       width: int = 1000,
       height: int = 700,
   ) -> "bokeh.layouts.LayoutDOM": ...
   ```

Algorithms thread a recorder through their helper methods. A null-object stand-in
keeps call sites free of `if recorder is not None:` guards when recording is off:

```python
class _NullRecorder:
    """No-op recorder used when debug=False. Same shape as DebugRecorder; .record(...)
    is a no-op and there is no .events list. _resolve_recorder returns this when
    recording is disabled, so algorithm code can call recorder.record(...) unconditionally."""

    def record(self, *args, **kwargs) -> None: ...
    # show / save / save_html raise: this instance is not a real recording.

def _resolve_recorder(
    debug: bool | DebugRecorder,
) -> DebugRecorder | _NullRecorder:
    """Coerce the user-supplied debug value.

    debug=False (or None) → _NullRecorder()  — recording disabled, calls are no-ops.
    debug=True            → fresh DebugRecorder().
    debug=DebugRecorder   → returned unchanged.
    """
```

In each instrumented algorithm:

```python
def _fit(self, data, debug=False, ...):
    recorder = _resolve_recorder(debug)
    self._build_skeleton(data, recorder)
    self._orient_colliders(skeleton, recorder)
    self._apply_meek_rules(pdag, recorder)
    self.record_ = recorder if isinstance(recorder, DebugRecorder) else None
```

Helper methods accept `recorder` as a parameter and call `recorder.record(...)` directly —
no contextvar, no module-level state, no per-call guards.

The `debug` argument is unified — it accepts `False` (default, no recording), `True` (auto-create a recorder, expose as
`pc.record_`), or a `DebugRecorder` instance (use the caller's recorder, lets users pre-configure or compare runs).

The viewer is built on Bokeh. Bokeh ships a first-class graph-rendering primitive (`bokeh.plotting.from_networkx`),
supports fully self-contained HTML output, and supports client-side interactivity via `CustomJS` callbacks — meaning
filtering, scrubbing, play / pause, and threshold sliders all run in the browser without a Python kernel. The recorded
events are serialized once at plot time and embedded as a JS-side data array.

The recorder and the plotter are decoupled at the import level: recording works without bokeh installed, and bokeh is
only imported lazily inside the `plot()` function (a deferred `import bokeh.*` at the top of the function body, not at
module load). Bokeh becomes a new optional dependency in an `extras_require['debug']` group, alongside the existing
optional `daft-pgm` and `pygraphviz` viz extras.

## Alternative Solutions

Several design choices were considered. The three with meaningful trade-offs:

### 1. Recording strategy: snapshot per event vs. mutation log + replay

| Aspect | Snapshot per event (chosen) | Mutation log + replay |
|---|---|---|
| Implementation complexity | Low — events are self-contained | Higher — needs replay logic + cache |
| Memory | O(events × graph size) | O(events) for events + O(1) base graph |
| Random-access scrub speed | Fast (already materialized) | Fast with cache, slower without |
| Handles weighted matrices? | Yes, naturally | Awkward — every iteration is a "mutation" |

Mutation log is more efficient for large graphs and long runs, but the use case is small
graphs and modest run lengths. Snapshotting eliminates an entire class of bugs (replay
correctness, cache invalidation) and keeps the data model trivially serializable.

### 2. Plotting backend: Bokeh vs. Plotly vs. Pyvis / Cytoscape.js vs. ipywidgets / Streamlit

Bokeh is the best fit because it combines native graph rendering, self-contained HTML, and client-side filtering. It is
the only library where all three are first-class. Plotly is a close second but requires precomputing multiple frame sets
to support filtering (HTML grows multiplicatively in event count). Pyvis / Cytoscape produce the prettiest graphs but
lack native timeline / scrubbing. ipywidgets and Streamlit fail the "embed in webpages / standalone HTML" requirement.

### 3. Recording API: `debug` parameter vs. context manager vs. global config

| API style | Pros | Cons |
|---|---|---|
| **`debug=False \| True \| Recorder` parameter (chosen)** | Per-call control; explicit; natural way to pass a pre-made recorder | Each algorithm grows one parameter |
| External context manager (`with DebugRecorder() as rec: ...`) | No API change to algorithms | Implicit; harder to discover in docstrings |
| Global config flag (`config.set_recorder(...)`) | No API change; consistent with `config.SHOW_PROGRESS` | Hidden global state; bad for parallel runs |
| Algorithm-flag + getter only (no recorder injection) | Simplest UX | No way to pre-configure or share a recorder across runs |

The unified `debug` parameter wins on explicitness and on supporting both "I just want something" (`debug=True`) and "I
want to control the recorder" (`debug=Recorder(...)`). A context manager could be added later for grouping runs without
breaking the API, but isn't needed in v1.

### 4. Granularity decision: producer captures everything; consumer filters

The recorder always captures full granularity (every CI test, score eval, optimizer iteration). The plotter handles
verbosity via UI filters. This separation of concerns keeps algorithm-author instrumentation simple (no "level"
parameter to think about) and gives the viewer authoritative access to all decisions when debugging. The cost is larger
recordings, which is acceptable given the small-graph scoping.

## Details of Proposed Solution

### Public API

```python
from pgmpy.causal_discovery.debug import DebugRecorder, plot
from pgmpy.causal_discovery import PC

# (1) default: zero overhead, no recording
pc = PC(data)
pc.estimate()

# (2) auto-create a recorder; retrieve via sklearn-style trailing-underscore attribute
pc = PC(data)
pc.estimate(debug=True)
pc.record_.show()                    # bokeh viewer in notebook
pc.record_.save_html("run.html")     # standalone HTML
pc.record_.save("run.json")          # serialized run for offline analysis

# (3) bring your own recorder (pre-configured, reusable, comparable across runs)
rec = DebugRecorder()
pc = PC(data)
pc.estimate(debug=rec)
rec.show()
```

Resolution helper, shared by all algorithms:

| Input | Output |
|---|---|
| `False` (or omitted) | `_NullRecorder()` (no-op stand-in) |
| `True` | fresh `DebugRecorder()` |
| `DebugRecorder` instance | the instance, unchanged |

After `_fit`, the algorithm sets `self.record_ = recorder` if it's a real `DebugRecorder`,
else `self.record_ = None`. Users only ever see real recorders on `record_`; the null
object is internal.

### Data model

```python
@dataclass(frozen=True)
class Event:
    event_type: str           # "ci_test", "edge_removed", "optimizer_iter", ...
    graph: nx.Graph           # deep-copied at record time
    phase: str | None         # advisory marker, e.g. "skeleton-depth-1"
    data: dict                # algorithm-specific extras
    step: int                 # auto-assigned, monotonic

class DebugRecorder:
    events: list[Event]

    def record(self, event_type: str, graph, *, phase: str | None = None, **data) -> None: ...
    def save(self, path) -> None: ...               # JSON
    @classmethod
    def load(cls, path) -> "DebugRecorder": ...
    def show(self, **plot_opts) -> None: ...        # delegates to plot()
    def save_html(self, path, **plot_opts) -> None: ...
```

Weighted-matrix algorithms (DAGMA / LiNGAM / CASTLE) record events whose `graph` is a
`networkx.DiGraph` with `weight` edge attributes set from the current `W`. The raw matrix
may also be stashed in `data["weights"]` for the heatmap / threshold-slider panel.

Serialization uses networkx `node_link_data` for graphs and converts numpy arrays in
`data` to nested lists. `load(path)` is the inverse.

### Author-facing instrumentation

The recorder is created once in `_fit` and threaded through helper methods as a parameter.
Helpers call `recorder.record(...)` directly. When `debug=False` the recorder is a
`_NullRecorder` whose `.record(...)` is a no-op, so call sites need no `if` guards.

```python
from pgmpy.causal_discovery.debug import DebugRecorder, _resolve_recorder

class PC:
    def _fit(self, data, debug=False, ...):
        recorder = _resolve_recorder(debug)
        skeleton = self._build_skeleton(data, recorder)
        pdag = self._orient_colliders(skeleton, recorder)
        pdag = self._apply_meek_rules(pdag, recorder)
        self.record_ = recorder if isinstance(recorder, DebugRecorder) else None
        return pdag

    def _build_skeleton(self, data, recorder):
        for l in range(...):
            recorder.record("phase_start", graph=current_pdag,
                            phase=f"skeleton-depth-{l}")
            for u, v in candidate_pairs:
                # ... CI test ...
                recorder.record(
                    "ci_test", graph=current_pdag,
                    phase=f"skeleton-depth-{l}",
                    u=u, v=v, sep_set=S, p_value=p,
                )
                if independent:
                    current_pdag.remove_edge(u, v)
                    recorder.record("edge_removed", graph=current_pdag,
                                    phase=f"skeleton-depth-{l}", u=u, v=v)
```

Cost when `debug=False`: each `recorder.record(...)` call dispatches to
`_NullRecorder.record`, which is `pass`. Effectively zero overhead, no contextvar lookup,
no module-level mutable state.

### v1 instrumentation hook points

- **PC**: `phase_start("skeleton-depth-{l}")` → per-pair `ci_test` (with `u`, `v`,
  `sep_set`, `p_value`, `independent`) → `edge_removed` when applicable →
  `phase_start("colliders")` → `edge_oriented` → `phase_start("meek-r{1..4}")` →
  `edge_oriented`.
- **GES**: `phase_start("forward")` → per-candidate `score_eval` (with `op`, `score_delta`)
  → `edge_added` on accept; analogous for `backward` and `turning`.
- **HillClimbSearch**: per-iteration `score_eval` for each candidate operation → one of
  `edge_added` / `edge_removed` / `edge_reversed` on accept; `tabu_skip` events when
  applicable.

The newer algorithms (TOPIC, ExpertInLoop, DAGMA, LiNGAM, CASTLE) get instrumented in
follow-up PRs by their respective authors as the algorithms stabilize. No new abstraction
is required — just thread `recorder` through `_fit` (and any helpers) and sprinkle
`recorder.record(...)` calls.

### Viewer (Bokeh)

`plot(recorder, ...)` returns a `bokeh.layouts.LayoutDOM`. Output is fully self-contained
client-side HTML — identical experience in notebooks, standalone HTML files, and embedded
webpages.

```
┌──────────────────────────────────┬──────────────────┐
│   Bokeh figure with              │  Div with        │
│   GraphRenderer                  │  per-event       │
│   (nodes + directed/undirected   │  metadata        │
│    edges + highlight layer)      │  (HTML, updated  │
│                                  │   via CustomJS)  │
├──────────────────────────────────┴──────────────────┤
│   Timeseries figure (auto-detected scalars)         │
├─────────────────────────────────────────────────────┤
│  ▶  ⏸   ━━━━━━●━━━━━━  Step 47 / 132                │
│  Phase: skeleton-depth-2                            │
│  Filter: [event-types MultiChoice]  [Threshold ━●━] │
└─────────────────────────────────────────────────────┘
```

Graph rendering details:

- `bokeh.plotting.from_networkx` is the structural layer; layout function defaults to
  `nx.kamada_kawai_layout` for ≤ 20 nodes, else `nx.spring_layout`. Layout is computed
  **once on the union graph** (every node + edge that ever appears across all events) so
  positions are stable across scrubbing.
- Directed arrows are drawn via a separate `Arrow` annotation layer.
- A highlight layer renders edges/nodes referenced in the current event (e.g., the `(u,v)`
  pair of a CI test) with a distinct color outline.
- Weighted edges use line width and a diverging-palette color bound to the weight; the
  threshold slider filters which edges are drawn.
- `HoverTool` shows variable name, current adjacency, and per-event annotations.

Animation and interaction (all client-side via `CustomJS`):

- A `Slider` (range `0..len(events)-1`) drives every layer. One CustomJS callback on
  `slider.value` rewrites `ColumnDataSource.data` for nodes / edges / highlights / metadata
  Div / timeseries cursor.
- `Play` / `Pause` / `Speed` are `Button` and `Select` widgets with CustomJS using
  `setInterval` to step the slider.
- An event-type filter (`MultiChoice`) rebinds the slider's index → event mapping (a JS
  filtered-indices array) so scrubbing skips filtered events.
- A phase-jump `Select` widget jumps the slider to the first event of a chosen phase.
- A weight-threshold slider is added only when any event carries weights.

The metadata `Div` is updated per slider change, formatting `event.data` as a small
key/value list. Long fields are summarized
(`weights: 10×10 ndarray, max=2.31, min=−1.05`).

The timeseries panel auto-detects scalar fields in `event.data` (e.g., `p_value`,
`score_delta`, `loss`, `h`, `mu`) and plots each as a line glyph; a `Span` annotation marks
the current step. If no numeric scalars are detected, the panel is omitted.

### Plot signature

```python
def plot(
    recorder: DebugRecorder,
    *,
    layout: callable | dict | None = None,    # nx layout fn or precomputed pos dict
    show_timeseries: bool = True,
    initial_filter: list[str] | None = None,  # event types visible initially
    title: str | None = None,
    width: int = 1000,
    height: int = 700,
) -> bokeh.layouts.LayoutDOM: ...
```

### File layout

A single new module + a single test module:

```
pgmpy/causal_discovery/debug.py
    # All components live here:
    #   - Event (frozen dataclass)
    #   - DebugRecorder (event log, save/load, show/save_html convenience)
    #   - _NullRecorder (no-op stand-in returned when debug=False)
    #   - _resolve_recorder (False / True / Recorder → DebugRecorder | _NullRecorder)
    #   - _serialize / _deserialize helpers (JSON, networkx node_link, ndarray coercion)
    #   - plot(recorder, ...) — bokeh imports done lazily inside this function

pgmpy/tests/test_causal_discovery/test_debug.py
    # All tests in one file, organized by class:
    #   - TestRecorder: record(), event ordering, deepcopy semantics
    #   - TestNullRecorder: .record(...) is a no-op; .events does not exist
    #   - TestResolveRecorder: False/True/instance handling
    #   - TestSerialization: save/load round-trip with mixed event types
    #   - TestPlot: bokeh-gated smoke test (skipped if bokeh missing)
    #   - TestIntegration: debug=True on PC / GES / HillClimbSearch on tiny seeds
```

Public exports are reachable via `pgmpy.causal_discovery.debug`. The
`pgmpy/causal_discovery/__init__.py` is left alone (no top-level re-export) so the debug
surface stays out of the way of users who never opt in.

`pyproject.toml` gains `bokeh>=3.0` in a new `extras_require['debug']` group.

### Out of scope for v1 (deferred)

- Instrumentation of TOPIC, ExpertInLoop, DAGMA, LiNGAM, CASTLE — done by their authors as
  the algorithms stabilize.
- Comparison view (two recorders side by side).
- Recording for inference algorithms (VariableElimination, GibbsSampling, etc.).
- Plotly / pyvis / cytoscape backend alternatives.
- Diff-based storage (revisit only if memory becomes an issue on a real use case).

## User Journeys with the Solution

### Journey 1 — Debugging "why didn't PC remove this edge?"

A user runs PC on real data and the output graph contains an edge they expected to be
removed during the skeleton phase. Today they would re-run with print statements or a
debugger. With the recorder:

```python
pc = PC(data)
pc.estimate(debug=True)
pc.record_.show()
```

In the viewer they filter event types to `ci_test` only, scrub through the skeleton phase,
and find the CI test for that edge. The metadata panel shows the conditioning set used and
the p-value. They can see: "the test was run with separating set `{X}`, p = 0.04, just
above the threshold". They adjust the significance level and re-run.

### Journey 2 — Teaching how PC builds a graph

An instructor demonstrating PC in a lecture wants students to watch the algorithm evolve.
They run PC on a small synthetic dataset, save the run as standalone HTML, and embed it on
the course website:

```python
pc = PC(data)
pc.estimate(debug=True)
pc.record_.save_html("lecture/pc_demo.html")
```

Students open the page, hit play, and watch the skeleton phase eliminate edges, then
collider detection orient v-structures, then Meek rules propagate orientations. The phase
dropdown lets them jump back to any stage. The metadata panel shows the CI test or
orientation rule firing at each step. No Python kernel is required at the view time — the
HTML is fully self-contained and embeds anywhere.

### Journey 3 — Implementing a new score-based algorithm

An author is implementing a variant of GES. They want to confirm the forward phase is
adding the highest-scoring edges and not getting stuck. They thread `recorder` through
the forward-phase helper and sprinkle `recorder.record(...)` calls at the candidate-scoring
loop and at acceptance:

```python
recorder.record("score_eval", graph=current_dag, phase="forward",
                op=("add", u, v), score_delta=delta)
# ...
recorder.record("edge_added", graph=current_dag, phase="forward", u=u, v=v,
                score_delta=delta)
```

They run with `debug=True` and view the timeseries panel: `score_delta` over time tells
them whether the algorithm is making meaningful progress or thrashing. They filter to
`score_eval` events and scrub to find iterations where the chosen operator was a poor
choice.

### Journey 4 — Debugging a weighted-matrix algorithm (DAGMA)

A DAGMA implementer wants to see how the weighted matrix `W` evolves under the central-path
optimizer. After instrumenting:

```python
recorder.record("optimizer_iter", graph=W_to_digraph(W), phase=f"epoch-{k}",
                weights=W.copy(), loss=loss_value, h=h_value, mu=mu_value)
```

The viewer shows the graph (edges with line widths proportional to `|W_ij|`) updating
across iterations. The timeseries panel auto-detects `loss`, `h`, and `mu` and plots them.
A weight-threshold slider lets the user explore which edges survive at different cutoffs.
They notice that `h(W)` is plateauing while `loss` is still falling, which suggests the
acyclicity constraint is being violated unequally across iterations.

### Journey 5 — Sharing a recorded run for code review

A reviewer asks "what does PC do on this dataset?" without wanting to install pgmpy or run
the algorithm themselves. The author saves the recording:

```python
pc.record_.save("review/pc_run.json")          # serialized run
pc.record_.save_html("review/pc_run.html")     # interactive standalone viewer
```

The reviewer opens `pc_run.html` in any browser. They can scrub through the run, filter by
event type, and inspect every CI test the algorithm performed — without setting up a Python
environment.
