## Time-series graph representation: `_TimeseriesCoreGraph`, `TimeseriesDAG`, `TimeseriesPDAG`

Contributors: [@Theavinash02](https://github.com/Theavinash02)

Related: pgmpy#3575

### Introduction

pgmpy has no class that can hold a stationary time-series causal graph. The only time-aware model is
`DynamicBayesianNetwork`, which does not fit:

1. It stores `(variable, time_slice)` nodes, and `add_edge` materialises the slice-1 copy of every lag-0 edge
   rather than deriving it, so one link occupies two stored edges.
2. It is limited to lag 1. There is no `tau_max`.
3. It has no way to represent an unoriented link.
4. There is no way to read in the output format these algorithms use: tigramite's `(N, N, tau_max + 1)` array, where `graph[i, j, tau]` describes the link between `var_i` at lag `tau` and `var_j` in the present.

### Proposed Solution

A private base class `_TimeseriesCoreGraph(_CoreGraph)` would hold the lag handling: `tau_max`, node and edge normalisation, and the adjacency conversion. Two public classes inherit it and declare which edge types they
allow.

<img width="693" height="556" alt="image" src="https://github.com/user-attachments/assets/00fc5e02-dd3f-4420-b263-4d26e0532edd" />

`_CoreGraph`'s subclasses differ in which edge types they allow. The time-series classes differ in what a node is. Those are independent, which is why `_TimeseriesCoreGraph` can inherit `_CoreGraph` instead of being a standalone class. `_CoreGraph` is not modified.

**Storage.** The graph is stationary, so only the difference between an edge's endpoints is stored:

```
add_edge(u, v, edge_type)  ->  lag = v_time - u_time,  stored as ((u_var, -lag), (v_var, 0))
```

The caller's time convention does not matter. `(("A", -1), ("B", 0), "->")` and `(("A", 0), ("B", 1), "->")` are
the same link and reach the same stored edge. An edge written backwards in time is stored with its endpoints
swapped and its `edge_type` inverted, using the markers `_CoreGraph` already maintains, so an unoriented link
keeps its endpoint marks. Shifted copies follow from stationarity and are not stored.

This is tigramite's layout, so `graph[i, j, tau]` maps directly onto one stored edge.

Endpoints come back exactly as stored, so the present slice is always time 0 and a lag-1 link reads as `("A", -1) -> ("B", 0)`. An edge carries a lag between its two endpoints, so it can be written in any frame; a single node has no lag to reduce, so `get_parents(("A", 0))` means the node at offset 0.

Where a method takes an optional edge type, the "any type" value is `None` rather than a literal edge type.
`"--"` would be the obvious sentinel but it is not in `TimeseriesDAG.SUPPORTED_EDGE_TYPES`, so a query would be
passing that class an edge type it does not accept. `None` is carried through an endpoint swap unchanged.

#### Adjacency: `"tigramite"` as the only encoding

`_CoreGraph` defines `from_adjacency` / `to_adjacency`, with an `encoding` argument selecting the
format - `edge_type`, `binary`, `bnlearn`, `causal-learn`, `pcalg` - and documents the round-trip invariant
`cls.from_adjacency(g.to_adjacency(e), encoding=e) == g`. None of those five has a third axis, so none can carry a lag.

A time-series graph needs a third dimension for the lag. And `_TimeseriesCoreGraph` overrides both the methods, and `tigramite` is the only encoding:

```python
from_adjacency(cls, adj, encoding="tigramite", variables=None)
to_adjacency(self, encoding="tigramite", variables=None)
```

- `encoding` stays in the signature, defaulting to `"tigramite"`, and any other value raises naming the class: `TimeseriesDAG only supports encoding='tigramite'; got encoding='pcalg'. The two-dimensional encodings have no axis for the lag.`
- `nodelist` is not accepted. It orders the rows of a two-dimensional matrix; `variables` does that job here, so it leaves the signature rather than lingering as a parameter that must be rejected.
- There is nothing left to auto-detect. With one encoding, the dimensionality check is validation rather than selection: a 2-D array is an error, not a different format.
- The round-trip invariant becomes single-encoding: `cls.from_adjacency(g.to_adjacency(), variables=g.variables) == g`. `variables` is load-bearing for equality, not merely for ordering — the array carries no names, so without it the variables come back as the integer indices `0 … N-1` and the graph, while structurally identical, is not equal to the one it came from.

This diverges from the base class's contract, which is the cost: a `TimeseriesDAG` cannot stand in for a `_CoreGraph` wherever one of the five encodings is requested. The alternative — adding `tigramite` as a sixth encoding and delegating the rest to `super()` — preserves that substitutability, at the price of five reachable formats that serialise a time-series graph into a shape nothing consumes.

The table translating tigramite’s three-character marks to and from pgmpy’s two-character edge types is local
to `from_adjacency` and `to_adjacency`, exactly as `_CoreGraph` already keeps its `pcalg` and `causal-learn`
code tables inside the two methods that use them. It covers the whole tigramite vocabulary, and
`SUPPORTED_EDGE_TYPES` does the filtering: a mark is accepted only if the table translates it and the
translation is one the class admits. Mark rejection therefore needs no per-class code and no per-class table -
`TimeseriesDAG` turns away `o-o` because `--`is not in its frozenset, through the same validation path as any
other edge. tigramite’s `x-x` has no pgmpy edge type and so is in no table at all.

#### Why inherit `_CoreGraph`

- `_CoreGraph` stores orientation as per-endpoint markers (`-`, `>`, `o`) on an `nx.MultiGraph`, not as directed
  edges. That covers what time-series discovery returns: `->` oriented, `--` unoriented, and `o>` / `oo` / `<>`
  for the latent-confounder algorithms. None of it assumes contemporaneity.
- Subclasses restrict themselves with `SUPPORTED_EDGE_TYPES`, as `PDAG`, `ADMG` and `MAG` already do.
  `TimeseriesDAG` and `TimeseriesPDAG` differ only in that frozenset.


### Alternative Solutions

**1. `TimeseriesDAG(DAG)`** - the original direction in pgmpy#3575. `DAG` still inherits `nx.DiGraph` rather
than `_CoreGraph` (pgmpy#3585 is pending), so it does not store endpoint markers: `TimeseriesPDAG` could not
share a base with it, and unoriented links would have nowhere to go. Every lag-handling method would also have
to be written against a second storage model rather than reusing the one `PDAG`, `ADMG` and `MAG` already use.

### Details of proposed solution

Class purpose and method signatures only.

#### `pgmpy/base/_timeseries_base.py`

```python
class _TimeseriesCoreGraph(_CoreGraph):
    """
    Base class for time-series graph types. A node is a (variable, time) pair, and what gets stored for an
    edge is the gap between its two times, not the times themselves. Subclasses say which edge types they
    accept, the same way the non-time-series classes do; none of the time handling here depends on that
    choice.
    """

    def __init__(self, edge_list=None, tau_max=None, exposures=None, outcomes=None, latents=None, roles=None):
        """
        Builds the graph from (u, v, edge_type) triples.

        `tau_max` has to be set before calling the parent constructor, because the parent adds the edges by
        calling `add_edge`, and this class's `add_edge` checks the lag against `tau_max`.
        """

    # tau_max

    def _set_tau_max(self, tau_max):
        """
        Records the maximum lag, and whether the caller asked for one.

        Those are two different situations. If a maximum was given, exceeding it is an error. If none was
        given, it starts at 0 and rises to fit whatever is added. The difference has to be remembered because
        an (N, N, tau_max + 1) array states a maximum even when no edge actually reaches it, and converting
        the graph back has to produce an array of the same shape.
        """

    @property
    def tau_max(self):
        """The largest lag this graph will accept."""

    def _check_lag(self, lag):
        """Rejects a lag above a maximum the caller set, or raises the maximum when the caller set none."""

    # normalisation

    @staticmethod
    def _normalize_node(node):
        """
        Turns whatever was passed into a (variable, time) pair.

        A bare name counts as time 0. A `DynamicNode`, or anything else with a `to_tuple` method, is unwrapped
        first.
        """

    def _normalize_edge(self, u, v, edge_type):
        """
        Works out the lag and puts the edge in one fixed position: head at time 0, tail at minus the lag.

        An edge written backwards in time is not refused. Its two ends are swapped and its edge type inverted,
        which gives the same link written the other way round. `edge_type=None` means "any type" and stays
        None through the swap.
        """

    # mutation

    def add_node(self, node, **attr):
        """
        Adds a variable, placed at time 0.

        A node already in the graph keeps the time it has. This matters because inherited code such as
        `with_role` calls `add_node` to attach an attribute to an existing node; moving it would attach the
        attribute to a different node.
        """

    def add_nodes_from(self, nodes, **attr):
        """Several at once, each following the rule above."""

    def add_edge(self, u, v, edge_type):
        """
        Adds a link.

        The lag is checked against `tau_max`. The times themselves are not checked, so any time frame works.
        """

    def remove_edge(self, u, v, edge_type):
        """
        Removes a link, written in any time frame.

        The edge goes through the same normalising as `add_edge`, so a link can be removed using different
        times than the ones it was added with.
        """

    def has_edge(self, u, v, edge_type=None):
        """Whether a link is present, in any time frame. `None` means any edge type will do."""

    # accessors

    @property
    def variables(self):
        """
        The variable names, with times dropped and duplicates removed.

        This also fixes the row and column order of the tigramite array.
        """

    def get_edges(self, data=True, edge_types=None):
        """The edges, as stored: a (variable, time) pair at each end, the head at time 0."""

    def get_parents(self, nodes) / get_children(self, nodes):
        """As inherited, with the node argument coerced into a (variable, time) pair first."""

    def get_links(self):
        """
        Every edge as (tail_variable, head_variable, lag, edge_type), with no times at all. One tuple
        corresponds to one cell of the tigramite array.

        The earlier end is always put first. That is not automatic: for a symmetric edge type such as `"--"`
        the inherited edge view may return the two ends either way round, and the lag has to be measured from
        the earlier one.
        """

    def get_lag(self, u, v):
        """How far apart in time the two ends of a link are, whatever type of link it is."""

    def get_intra_edges(self) / get_inter_edges(self):
        """
        Links within one time point, and links across time points.

        Named to match the methods DBN already has.
        """

    def get_parents_by_lag(self, variable):
        """
        The parents of a variable, grouped by how far back they sit: {1: ["A"], 2: ["A", "C"]}.

        The lag has to come back alongside the name, because that pair is what a conditioning set is built
        from.
        """

    # adjacency

    @classmethod
    def from_adjacency(cls, adj, encoding="tigramite", variables=None):
        """
        Overrides the inherited method. "tigramite" is the only encoding, and the default: an array of shape
        (N, N, tau_max + 1), the third axis being the lag. Both shapes tigramite produces work: text marks
        like "-->", and the plain 0/1 array it gives before conversion.

        Any other encoding raises, naming the class. The five two-dimensional encodings have no axis for a
        lag, so a 2-D array is a shape error rather than another format.

        The mark table is defined here, covering the whole tigramite vocabulary; a mark is kept only if its
        translation is in the class’s SUPPORTED_EDGE_TYPES. A subclass declares nothing for this to work.
        """

    def to_adjacency(self, encoding="tigramite", variables=None):
        """
        The reverse, with the inverse table defined here. `nodelist` is not accepted: it orders a
        two-dimensional matrix, which `variables` does here.

        A graph converted out and back comes back equal to itself, for every one of them.
        """

    # identity

    def get_subgraph(self, nodes=None, edge_types=None):
        """
        As inherited, but the result keeps the original `tau_max`.

        The inherited version builds the result by calling `self.__class__()` with no arguments, which would
        reset it to 0. `copy` is built on this method, so this one override fixes both.
        """

    def __eq__(self, other) / __hash__(self):
        """
        Two graphs are equal when the inherited comparison says so and their `tau_max` agrees.

        Same links but different maximum lags means different graphs, because they accept different things.
        """
```

#### `pgmpy/base/TimeseriesDAG.py`

```python
class TimeseriesDAG(_TimeseriesCoreGraph):
    """
    Directed acyclic graph over a stationary time series. Every link is oriented. An algorithm that leaves
    some same-time links unoriented produces a `TimeseriesPDAG` instead.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-"])
```

A DAG has only directed links, so why list both `"->"` and `"<-"`? Because an edge written backwards in time
is turned round rather than refused, and turning it round inverts its type, so the link arrives as `"<-"`. If
`"<-"` were not accepted, that rewrite would fail. `PDAG` lists both for the same reason.

#### `pgmpy/base/TimeseriesPDAG.py`

```python
class TimeseriesPDAG(_TimeseriesCoreGraph):
    """
    The partially directed counterpart. A same-time link the algorithm could not orient is held as "--".
    Lagged links are always oriented, because time decides the direction.
    """

    SUPPORTED_EDGE_TYPES = frozenset(["->", "<-", "--"])
```

`"--"` means the algorithm found a link but could not say which way it points. That can only happen at lag 0.
Across time there is nothing to decide: the earlier event is the cause, because the later one cannot be.

Neither subclass defines a method, and neither declares anything but its frozenset. That one line is also what
decides which tigramite marks the class will accept.

### User journeys with the solution
**1. Building a graph**
 
```python
import numpy as np
from pgmpy.base import TimeseriesDAG
 
dag_edges = [
    (("A", -1), ("A", 0), "->"),
    (("A", -1), ("B", 0), "->"),
    (("B", 0), ("C", 0), "->"),
    (("C", -2), ("A", 0), "->"),
]
dag = TimeseriesDAG(edge_list=dag_edges, tau_max=2)
dag.tau_max
```
 
```
2
```
 
A maximum the caller sets is kept even when no edge reaches it, so converting to an array produces the shape
they asked for. Left out, it starts at 0 and rises to fit whatever is added:
 
```python
g = TimeseriesDAG(edge_list=[(("A", -1), ("B", 0), "->")])
g.tau_max
```
 
```
1
```
 
```python
g.add_edge(("C", -3), ("B", 0), "->")
g.tau_max
```

```
3
```
Node roles are accepted at construction, as on any other graph:

```python
TimeseriesDAG(
    edge_list=[(("A", -1), ("B", 0), "->")],
    latents={("A", -1)},
).latents
```

```
{('A', -1)}
```

**2. Adding and removing**

```python
g = dag.copy()
g.add_edge(("B", -1), ("C", 0), "->")
g.has_edge(("B", -1), ("C", 0))
```

```
True
```

The same link in any other frame reaches the same edge, so it can be removed with times it was never added
with:

```python
g.remove_edge(("B", 4), ("C", 5), "->")
g.has_edge(("B", -1), ("C", 0))
```

```
False
```

`has_edge` narrows to one type, and `None` means any type will do:

```python
g.has_edge(("A", -1), ("B", 0), "->")
```

```
True
```

A bare name is a variable at time 0:

```python
g.add_node("D")
g.add_nodes_from(["E", "F"])
```

`add_edges_from` and `remove_edges_from` are inherited and go through the same normalising. A lag above
`tau_max` is refused:

```python
g.add_edge(("A", -5), ("B", 0), "->")
```

```
ValueError: Edge lag 5 exceeds tau_max=2. Build the graph with a larger tau_max to allow it.
```

**3. Reading the structure**

Two forms. `get_edges` gives a `(variable, time)` pair at each end, with the present slice at 0:

```python
sorted(dag.get_edges(data=True))
```

```
[(('A', -1), ('A', 0), '->'),
 (('A', -1), ('B', 0), '->'),
 (('B', 0), ('C', 0), '->'),
 (('C', -2), ('A', 0), '->')]
```

`get_links` drops times entirely and gives the lag - one tuple per cell of the tigramite array:

```python
sorted(dag.get_links())
```

```
[('A', 'A', 1, '->'), ('A', 'B', 1, '->'), ('B', 'C', 0, '->'), ('C', 'A', 2, '->')]
```

For a single pair:

```python
dag.get_lag(("A", -1), ("B", 0))
```

```
1
```

The variable names, times dropped - this also fixes the row and column order of the array:

```python
dag.variables
```

```
['A', 'B', 'C']
```
**4. Parents and children**

Inherited, returning the same `(variable, time)` pairs `get_edges` reports:

```python
sorted(dag.get_parents(("A", 0)))
```

```
[('A', -1), ('C', -2)]
```

```python
sorted(dag.get_children(("A", -1)))
```

```
[('A', 0), ('B', 0)]
```

**5. Parents grouped by lag**

What a conditioning set is built from. The lag comes back alongside the name, because the pair is what identifies the parent:

```python
dag.get_parents_by_lag("A")
```

```
{1: ['A'], 2: ['C']}
```

A variable with only a contemporaneous parent groups under lag 0:

```python
dag.get_parents_by_lag("C")
```

```
{0: ['B']}
```

**6. Same-time and across-time links**

Named to match the methods DBN already has. Both return the lag-relative form:

```python
sorted(dag.get_intra_edges())
```

```
[('B', 'C', 0, '->')]
```

```python
sorted(dag.get_inter_edges())
```

```
[('A', 'A', 1, '->'), ('A', 'B', 1, '->'), ('C', 'A', 2, '->')]
```

**7. Array conversion - from_adjacency and to_adjacency**

This is how a PCMCI estimator in pgmpy#3588 hands its result over. Out first - `tigramite` is the default, so no `encoding` argument:

```python
arr = dag.to_adjacency()
arr.shape
```

```
(3, 3, 3)
```

```python
arr
```

```
array([[['', '-->', ''],
        ['', '-->', ''],
        ['', '', '']],

       [['', '', ''],
        ['', '', ''],
        ['-->', '', '']],

       [['', '', '-->'],
        ['<--', '', ''],
        ['', '', '']]], dtype='<U3')
```

Row and column order is `dag.variables`, which is `['A', 'B', 'C']`. `arr[0, 0, 1]` is `A` at lag 1 into `A` now; `arr[2, 0, 2]` is `C` at lag 2 into `A` now. The contemporaneous `B -> C` occupies both `arr[1, 2, 0]` as `'-->'` and `arr[2, 1, 0]` as `'<--'`, which is tigramite's own convention for a lag-0 link.

And back in:

```python
TimeseriesDAG.from_adjacency(arr, variables=dag.variables) == dag
```

```
True
```

`variables` is load-bearing for equality and ordering. A bare array carries no names, so without it the variables come back as the integer indices and the graph, while structurally identical, is not equal to the one it came from:

```python
TimeseriesDAG.from_adjacency(arr) == dag
```

```
False
```

```python
TimeseriesDAG.from_adjacency(arr).variables
```

```
[0, 1, 2]
```

Any other encoding is refused, naming the class:

```python
dag.to_adjacency(encoding="pcalg")
```

```
ValueError: TimeseriesDAG only supports encoding='tigramite'; got encoding='pcalg'. The two-dimensional encodings have no axis for the lag.
```

A two-dimensional array is a shape error rather than another format, since there is no longer anything to select between:

```python
TimeseriesDAG.from_adjacency(np.zeros((3, 3), dtype=int))
```

```
ValueError: Expected a three-dimensional array of shape (N, N, tau_max + 1), the third axis being the lag.
Got shape (3, 3).
```

The plain 0/1 array tigramite produces before it converts marks also decodes, as long as every link is lagged:

```python
raw = np.zeros((3, 3, 3), dtype=int)
raw[0, 1, 1] = 1          # A at lag 1 -> B
raw[2, 0, 2] = 1          # C at lag 2 -> A
raw
```

```
array([[[0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]],

       [[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]],

       [[0, 0, 1],
        [0, 0, 0],
        [0, 0, 0]]])
```

```python
sorted(TimeseriesDAG.from_adjacency(raw, variables=["A", "B", "C"]).get_links())
```

```
[('A', 'B', 1, '->'), ('C', 'A', 2, '->')]
```

A symmetric pair at lag 0 in a 0/1 array is an unoriented link, which a `TimeseriesDAG` cannot hold. The same array again, with that pair added between `B` and `C` in the present:

```python
raw_with_contemporaneous = raw.copy()
raw_with_contemporaneous[1, 2, 0] = 1
raw_with_contemporaneous[2, 1, 0] = 1

TimeseriesDAG.from_adjacency(raw_with_contemporaneous, variables=["A", "B", "C"])
```

```
ValueError: adj[1, 2, 0] and adj[2, 1, 0] are both 1, which marks an unoriented link.
TimeseriesDAG supports ['->', '<-'].

**8. Subgraph - get_subgraph and copy**

`get_subgraph` carries `tau_max` across, which the inherited version would have reset to 0:

```python
sub = dag.get_subgraph(nodes=[("A", -1), ("A", 0), ("B", 0)])
sorted(sub.get_edges(data=True))
```

```
[(('A', -1), ('A', 0), '->'), (('A', -1), ('B', 0), '->')]
```

```python
sub.tau_max
```

```
2
```
`copy` is built on `get_subgraph`, so the one override fixes both:

```python
dag.copy() == dag
```

```
True
```

Equality takes `tau_max` into account. The same links under a different maximum are a different graph, because the two accept different things:

```python
TimeseriesDAG(edge_list=dag_edges, tau_max=3) == dag
```

```
False
```


### How the data is stored internally
 `_CoreGraph` is an `nx.MultiGraph` subclass, so a `TimeseriesDAG` is one: there is no separate store, and no storage function. For the `dag` built in journey 1:

```python
from pgmpy.base import TimeseriesDAG
 
dag_edges = [
    (("A", -1), ("A", 0), "->"),
    (("A", -1), ("B", 0), "->"),
    (("B", 0), ("C", 0), "->"),
    (("C", -2), ("A", 0), "->"),
]
dag = TimeseriesDAG(edge_list=dag_edges, tau_max=2)
dict(dag.nodes(data=True))
```

```
{('A', -1): {}, ('A', 0): {}, ('B', 0): {}, ('C', 0): {}, ('C', -2): {}}
```
Five nodes for three variables. A variable appears once per distinct offset it is actually used at: `A` at the present and at -1, `C` at the present and at -2, `B` only at the present. These are offsets, not time points - nothing is allocated per time step, and `tau_max` does not decide how many nodes exist.

```python
list(dag.edges(keys=True, data=True))
```

```
[(('A', -1), ('A', 0), 0, {('A', -1): '-', ('A', 0): '>'}),
 (('A', -1), ('B', 0), 0, {('A', -1): '-', ('B', 0): '>'}),
 (('A',  0), ('C', -2), 0, {('C', -2): '-', ('A', 0): '>'}),
 (('B',  0), ('C', 0), 0, {('B', 0): '-', ('C', 0): '>'})]
```

- Four stored edges for four links - one each, with no shifted copies. The pair order in the tuple is `nx`'s, not the link's: the third edge is listed as `('A', 0), ('C', -2)` even though the link runs `C` at -2 into `A` now, which is why `get_links` puts the earlier end first itself.
-  **Every head sits at offset 0.** : The tail carries the whole lag as a negative offset. That is the normalisation `_normalize_edge` applies, and it is what makes `(("A", -1), ("B", 0))` and `(("A", 0), ("B", 1))` the same stored edge. 