"""Web-preserving transformations of SEDs (see docs/transformations.md).

Each transformation takes an ``SEDiagram`` and returns a new one, with every
web carried across by the rule's bijection ``tau``. The new SED re-checks
the mapped webs (see ``CHECK``). Edges untouched by a rule keep their labels;
the rule only supplies labels on the edges it creates.

The rules are methods of ``SEDEditor``, which applies any number of them in
place and touches only the edges and webs each rewrites; the functions here
run one editor session each.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Hashable, Iterable, Sequence

import networkx as nx
import numpy as np

from dizx._f2 import sp_canonical
from dizx.zx_diagram import (
    PAULI, PAULI_BITS, SPIDERS, Edge, Node, NodeType, SEDiagram, column_transport,
    webs_from_matrix,
)

Bits = tuple[int, int]

_SWAP = {NodeType.Z: NodeType.X, NodeType.X: NodeType.Z}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _matching(t: NodeType, xz: Bits) -> tuple[int, int]:
    """Split ``(x, z)`` into (matching, non-matching) components at a spider of type t."""
    return xz if t is NodeType.Z else (xz[1], xz[0])


def _from_matching(t: NodeType, m: int, n: int) -> Bits:
    return (m, n) if t is NodeType.Z else (n, m)


def _fresh(types: dict[Node, NodeType], base: Hashable) -> Node:
    if base not in types:
        return base
    k = 1
    while (base, k) in types:
        k += 1
    return (base, k)


def _legs(sed: SEDiagram, v: Node) -> list[Edge]:
    """Legs of v as edges oriented ``(v, u, key)``; rejects self-loops."""
    legs = list(sed.graph.edges(v, keys=True))
    if any(u == v for _, u, _ in legs):
        raise ValueError(f"{v!r} has a self-loop")
    return legs


def _spider(sed: SEDiagram, v: Node) -> NodeType:
    t = sed.types.get(v)
    if t not in SPIDERS:
        raise ValueError(f"{v!r} is not a spider")
    return t


CHECK = "local"
"""How transformations check their output: ``"local"`` (the default) checks
the web rules at the nodes the step touched, and each web's class;
``"full"`` re-checks every web at every node, and independence; ``"none"``
trusts the rule. Set it here or use :func:`checking`."""


@contextmanager
def checking(mode: str):
    """Use check ``mode`` for transformations inside the ``with`` block."""
    global CHECK
    if mode not in ("local", "full", "none"):
        raise ValueError("mode must be 'local', 'full' or 'none'")
    old, CHECK = CHECK, mode
    try:
        yield
    finally:
        CHECK = old


def _toward(positions: dict, a: Node, b: Node, f: float):
    pa, pb = np.asarray(positions[a], float), np.asarray(positions[b], float)
    return tuple(pa + f * (pb - pa))


# -- layout --------------------------------------------------------------------
#
# Transformations never move existing nodes. A split puts its two halves
# SPACING apart, symmetric about where the original node was; a loop puts its
# four spiders on a rectangle whose centre is where the original node was.

SPACING = 0.25
"""Distance between the two halves of a split (and the width of a loop)."""


def _unit(d: np.ndarray, fallback: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else np.asarray(fallback, float)


def _mean(positions: dict, nodes: Iterable[Node]) -> np.ndarray | None:
    nodes = list(nodes)
    return np.mean([positions[n] for n in nodes], axis=0) if nodes else None


def _separation(positions: dict, a_nodes: Iterable[Node], b_nodes: Iterable[Node], at: Node) -> np.ndarray:
    """Unit vector from where ``a_nodes`` lie towards where ``b_nodes`` lie
    (using ``at`` for an empty side)."""
    p = np.asarray(positions[at], float)
    pa, pb = _mean(positions, a_nodes), _mean(positions, b_nodes)
    return _unit((p if pb is None else pb) - (p if pa is None else pa))


def _carry_hadamards(graph: nx.MultiGraph, types: dict, positions: dict, old_positions: dict,
                     origin: dict[Node, np.ndarray], skip: Iterable[Node] = ()) -> None:
    """H nodes are functionally edges: re-place each H node next to a moved
    spider at the same fraction along its edge as before the move.

    ``origin`` maps each moved spider to the position it (or the node it was
    split from) had before; ``old_positions`` holds the others' old positions.
    """
    done = set(skip)
    for n, o in origin.items():
        for h in list(graph.neighbors(n)):
            if h in done or types.get(h) is not NodeType.H or h not in old_positions:
                continue
            done.add(h)
            ys = [x for x in graph.neighbors(h) if x != n]
            if len(ys) != 1:
                continue
            (y,) = ys
            a_old = np.asarray(o, float)
            y_old = np.asarray(origin[y] if y in origin else old_positions.get(y, positions[y]), float)
            seg = y_old - a_old
            L = float(np.dot(seg, seg))
            t = float(np.clip(np.dot(np.asarray(old_positions[h], float) - a_old, seg) / L, 0, 1)) if L > 1e-12 else 0.5
            a_new, y_new = np.asarray(positions[n], float), np.asarray(positions[y], float)
            positions[h] = tuple(a_new + t * (y_new - a_new))


def _perpendicular(d: np.ndarray) -> np.ndarray:
    """A unit vector perpendicular to d, in the (x, y) plane when possible."""
    return _unit(np.cross(d, [0.0, 0.0, 1.0]), _unit(np.cross(d, [1.0, 0.0, 0.0])))


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------

class SEDEditor:
    """Applies the rules below to an SED in place; ``freeze`` returns the
    resulting ``SEDiagram``, checked as ``CHECK`` (``"local"``: at every node
    some rule touched).

    Webs are held as an index from each edge to the webs on it and their
    ``(x, z)`` labels, so a rule only visits the webs on the edges it
    rewrites. The graph's adjacency is kept in the order a fresh copy of the
    graph has (each node's earlier neighbours first, in node order, then the
    later ones as they were added), which is the order of the SED it freezes
    to: rules applied in turn to one editor give exactly the SED that applying
    them one at a time gives.

    ``graph``, ``types``, ``positions``, ``inputs``, ``outputs`` and ``edges``
    read like an SED's (the graph must only be changed through the rules), so
    ``cut_edges``, ``loopable`` and the like accept either.
    """

    def __init__(self, sed: SEDiagram) -> None:
        self.graph = nx.MultiGraph(sed.graph)
        self.types = dict(sed.types)
        self.positions = dict(sed.positions)
        self.inputs, self.outputs = tuple(sed.inputs), tuple(sed.outputs)
        self.closed = dict(sed.closed)
        self.carried = list(sed.carried)
        self._rank = {v: i for i, v in enumerate(self.graph.nodes)}
        self._next = len(self._rank)
        self._family = [f for f, ws in enumerate((sed.detectors, sed.stabilisers,
                                                   sed.co_stabilisers, sed.logicals))
                        for _ in ws]
        self._at: dict[Edge, dict[int, Bits]] = {}
        n = sed.n_edges
        W = sed._matrix(sed.webs).tocoo()
        for wid, c in zip(W.row.tolist(), W.col.tolist()):
            d = self._at.setdefault(self._key(*sed.edges[c % n]), {})
            x, z = d.get(wid, (0, 0))
            d[wid] = (1, z) if c < n else (x, 1)
        self._tracked: dict[int, set[Edge]] = {}  # support of webs made here
        self._grown: set[Node] = set()
        self._touched: set[Node] = set()

    # -- state ----------------------------------------------------------------

    @property
    def edges(self) -> list[Edge]:
        """All edges, in the order of the SED this freezes to."""
        return list(self.graph.edges(keys=True))

    def freeze(self) -> SEDiagram:
        new = SEDiagram(self.graph, self.types, self.inputs, self.outputs, positions=self.positions,
                        check="none", closed=self.closed)
        n = new.n_edges
        rows, cols = [], []
        for key, d in self._at.items():
            j = new._index[key]
            for wid, (x, z) in d.items():
                if x:
                    rows.append(wid)
                    cols.append(j)
                if z:
                    rows.append(wid)
                    cols.append(n + j)
        W = sp_canonical(column_transport(np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64),
                                          len(self._family), 2 * n))
        fam = np.array(self._family, dtype=np.int64)
        new.detectors, new.stabilisers, new.co_stabilisers, new.logicals = (
            webs_from_matrix(new, W[np.flatnonzero(fam == f)]) for f in range(4))
        new.carried = list(self.carried)
        if CHECK == "full":
            new._check_webs()
        elif CHECK == "local":
            new._check_webs(nodes=[v for v in self._touched if v in new.types], independence=False)
        return new

    # -- primitives -------------------------------------------------------------

    def _key(self, u: Node, v: Node, k: int) -> Edge:
        return (u, v, k) if self._rank[u] <= self._rank[v] else (v, u, k)

    def _on(self, e: Edge) -> dict[int, Bits]:
        """The webs on edge e and their labels (live)."""
        return self._at.get(self._key(*e), {})

    def _set(self, e: Edge, wid: int, xz: Bits) -> None:
        """Label a fresh edge."""
        if xz[0] or xz[1]:
            key = self._key(*e)
            self._at.setdefault(key, {})[wid] = (int(xz[0]), int(xz[1]))
            if wid in self._tracked:
                self._tracked[wid].add(key)

    def _add_edge(self, u: Node, v: Node) -> Edge:
        for x in (u, v):  # the order networkx adds new nodes in
            if x not in self._rank:
                self._rank[x] = self._next
                self._next += 1
        k = self.graph.add_edge(u, v)
        self._grown.update((u, v))
        self._touched.update((u, v))
        return (u, v, k)

    def _pop_edge(self, e: Edge) -> dict[int, Bits]:
        """Forget e's labels (the caller removes e from the graph)."""
        key = self._key(*e)
        self._touched.update(e[:2])
        d = self._at.pop(key, {})
        for wid in d:
            if wid in self._tracked:
                self._tracked[wid].discard(key)
        return d

    def _remove_edge(self, u: Node, v: Node, k: int) -> dict[int, Bits]:
        d = self._pop_edge((u, v, k))
        self.graph.remove_edge(u, v, k)
        return d

    def _remove_nodes(self, nodes: Iterable[Node]) -> None:
        nodes = list(nodes)
        for v in nodes:
            for e in list(self.graph.edges(v, keys=True)):
                self._pop_edge(e)
        self.graph.remove_nodes_from(nodes)
        for v in nodes:
            del self._rank[v]
            del self.types[v]
            self.positions.pop(v, None)

    def _new_detector(self, labels: dict[Edge, str]) -> int:
        wid = len(self._family)
        self._family.append(0)
        self._tracked[wid] = set()
        for e, p in labels.items():
            self._set(e, wid, PAULI_BITS[p])
        return wid

    def _settle(self) -> None:
        """Put the neighbours of nodes that gained some back in copy order."""
        r = self._rank
        for v in self._grown:
            if v not in r:
                continue
            adj = self.graph._adj[v]
            items = list(adj.items())
            early = sorted((it for it in items if r[it[0]] < r[v]), key=lambda it: r[it[0]])
            late = [it for it in items if r[it[0]] >= r[v]]
            if [a for a, _ in early + late] != [a for a, _ in items]:
                adj.clear()
                adj.update(early + late)
        self._grown.clear()

    def _edges_at(self, S: set[Node]) -> list[Edge]:
        """The edges with an end in S, in ``edges`` order and orientation."""
        r, adj = self._rank, self.graph._adj
        found = {}
        for v in S:
            for u, kd in adj[v].items():
                a, b = (v, u) if r[v] <= r[u] else (u, v)
                for k in kd:
                    found[(a, b, k)] = None
        place: dict[Node, dict[tuple, tuple[int, int]]] = {}

        def where(e: Edge) -> tuple:
            a, b, k = e
            if a not in place:
                place[a] = {(u, kk): (j, i) for j, (u, kd) in enumerate(adj[a].items())
                            for i, kk in enumerate(kd)}
            return (r[a], *place[a][(b, k)])
        return sorted(found, key=where)

    def _positions_near(self, nodes: Iterable[Node], radius: int) -> dict[Node, tuple]:
        """Positions of the nodes within ``radius`` steps of ``nodes``."""
        seen = set(nodes)
        frontier = set(seen)
        for _ in range(radius):
            frontier = {u for v in frontier for u in self.graph.neighbors(v)} - seen
            seen |= frontier
        return {v: self.positions[v] for v in seen if v in self.positions}

    # -- rules ------------------------------------------------------------------

    def dualise(self, nodes: Iterable[Node]) -> None:
        """Apply D_v to every spider in ``nodes`` at once (see :func:`dualise`)."""
        S = set(nodes)
        for v in S:
            _spider(self, v)
            _legs(self, v)  # rejects self-loops
        edges = self._edges_at(S)
        for v in S:
            self.types[v] = _SWAP[self.types[v]]
        self._touched |= S
        for e in edges:
            a, b, k = e
            if a in S and b in S:
                d = self._on(e)
                for wid, (x, z) in d.items():
                    d[wid] = (z, x)
                continue
            v, u = (a, b) if a in S else (b, a)
            h = _fresh(self.types, ("H", v, u, k))
            self.types[h] = NodeType.H
            old = self._remove_edge(a, b, k)
            e1 = self._add_edge(v, h)
            e2 = self._add_edge(h, u)
            for wid, (x, z) in old.items():
                self._set(e1, wid, (z, x))
                self._set(e2, wid, (x, z))
            if self.positions:
                self.positions[h] = _toward(self.positions, v, u, 0.35)
        self._settle()

    def insert_identity(self, e: Edge, colour: NodeType | str) -> Node:
        """I_e^c (see :func:`insert_identity`); returns the new spider."""
        colour = NodeType(colour)
        if colour not in SPIDERS:
            raise ValueError("identity spider must be Z or X")
        u, v, k = e
        s = _fresh(self.types, ("id", u, v, k))
        self.types[s] = colour
        old = self._remove_edge(u, v, k)
        e1 = self._add_edge(u, s)
        e2 = self._add_edge(s, v)
        if self.positions:
            self.positions[s] = _toward(self.positions, u, v, 0.5)
        for wid, xz in old.items():
            self._set(e1, wid, xz)
            self._set(e2, wid, xz)
        self._settle()
        return s

    def split(self, v: Node, A: Iterable[Edge], name: Hashable | None = None,
              direction: Sequence[float] | None = None) -> Node:
        """S_v(A, B) (see :func:`split`); returns the new half."""
        t = _spider(self, v)
        legs = _legs(self, v)
        keys = [self._key(*e) for e in legs]
        a_set = {self._key(*e) for e in A}
        if not a_set <= set(keys):
            raise ValueError(f"A must be legs of {v!r}")
        old_pos = self._positions_near([v], 2) if self.positions else {}
        vb = _fresh(self.types, name if name is not None else ("split", v))
        self.types[vb] = t
        on = [dict(self._on(e)) for e in legs]
        moved: list[tuple[Edge, Edge]] = []
        for e, key, d in zip(legs, keys, on):
            if key not in a_set:
                _, u, k = e
                self._remove_edge(v, u, k)
                new = self._add_edge(vb, u)
                moved.append((e, new))
                for wid, xz in d.items():
                    self._set(new, wid, xz)
        f = self._add_edge(v, vb)
        # On f: the matching component common to v's legs, and the
        # non-matching parity over A.
        a_on = [d for key, d in zip(keys, on) if key in a_set]
        for wid in sorted(set().union(*on)) if on else ():
            m = _matching(t, on[0].get(wid, (0, 0)))[0]
            n = sum(_matching(t, d.get(wid, (0, 0)))[1] for d in a_on) % 2
            self._set(f, wid, _from_matching(t, m, n))
        if self.positions:
            d = _unit(np.asarray(direction, float)) if direction is not None else _separation(
                self.positions, [e[1] for e, key in zip(legs, keys) if key in a_set],
                [e[1] for e, _ in moved], v)
            p = np.asarray(self.positions[v], float)
            self.positions[v], self.positions[vb] = tuple(p - SPACING / 2 * d), tuple(p + SPACING / 2 * d)
            _carry_hadamards(self.graph, self.types, self.positions, old_pos, {v: p, vb: p})
        self._settle()
        return vb

    def add_parallel_edge(self, u: Node, v: Node) -> int:
        """See :func:`add_parallel_edge`; returns the loop web's index."""
        tu, tv = _spider(self, u), _spider(self, v)
        path = _joining_path(self, u, v)
        if (tu is tv) != (len(path) == 2):
            raise ValueError("same-colour spiders need a plain edge, opposite colours an H node")
        old = [(a, b, next(iter(self.graph[a][b]))) for a, b in zip(path, path[1:])]
        on = [dict(self._on(e)) for e in old]
        if len(path) == 2:
            new = [self._add_edge(u, v)]
        else:
            h2 = _fresh(self.types, ("H", u, v))
            self.types[h2] = NodeType.H
            new = [self._add_edge(u, h2), self._add_edge(h2, v)]
            if self.positions:
                f = _perpendicular(np.asarray(self.positions[v], float) - np.asarray(self.positions[u], float))
                p = np.asarray(self.positions[path[1]], float)
                self.positions[path[1]], self.positions[h2] = tuple(p - SPACING / 2 * f), tuple(p + SPACING / 2 * f)
        # The leg at u of each copy, and at v (the same edge on a plain path).
        ends = [(tu, on[0], old[0], new[0]), (tv, on[-1], old[-1], new[-1])]
        for wid in sorted(set().union(*on)):
            labels = {e2: _from_matching(t, _matching(t, d.get(wid, (0, 0)))[0], 0) for t, d, _, e2 in ends}
            for e2, xz in labels.items():
                self._set(e2, wid, xz)
        loop = {e: PAULI[_from_matching(t, 0, 1)] for t, _, e, e2 in ends for e in (e, e2)}
        wid = self._new_detector(loop)
        self._settle()
        return wid

    def loop_edge(self, u: Node, v: Node, keep_u: Node | None = None, keep_v: Node | None = None) -> None:
        """See :func:`loop_edge`."""
        if not loopable(self, u, v):
            raise ValueError(f"{u!r} and {v!r} must be joined by one edge or H node, "
                             "with exactly two other legs each")
        path = _joining_path(self, u, v)
        others = {a: sorted(set(self.graph.neighbors(a)) - set(path), key=repr) for a in (u, v)}
        keep = {u: keep_u, v: keep_v}
        has_positions = bool(self.positions)
        P = self._positions_near([u, v], 2) if has_positions else {}
        if has_positions:
            pu, pv = np.asarray(P[u], float), np.asarray(P[v], float)
            e = _unit(pv - pu, (1.0, 0.0, 0.0))
            # The loop opens across e, towards the legs the new spiders take.
            f = None
            for a in (u, v):
                n1, n2 = (np.asarray(P[n], float) for n in others[a])
                w = (n2 - n1) - np.dot(n2 - n1, e) * e
                if np.linalg.norm(w) > 1e-9:
                    f = _unit(w)
                    break
            f = f if f is not None else _perpendicular(e)
            for a in (u, v):
                if keep[a] is None:  # keep the leg on the near side of the loop
                    keep[a] = min(others[a], key=lambda n: np.dot(np.asarray(P[n], float), f))
            corners = {u: pu, v: pv} | ({path[1]: np.asarray(P[path[1]], float)} if len(path) == 3 else {})
        for a in (u, v):
            keep[a] = others[a][0] if keep[a] is None else keep[a]
            if keep[a] not in others[a]:
                raise ValueError(f"keep must be a neighbour of {a!r} off the joining path")

        loop = self.add_parallel_edge(u, v)
        if len(path) == 3:
            h = path[1]
            h2 = next(x for x in self.graph.neighbors(u) if x != h and self.types[x] is NodeType.H
                      and set(self.graph.neighbors(x)) == {u, v})
            orig = {u: (u, h, 0), v: (v, h, 0)}
        else:
            k0 = min(self.graph[u][v])
            orig = {u: (u, v, k0), v: None}
        for a in (u, v):
            legs = _legs(self, a)
            o = orig[a] or (v, u, min(self.graph[v][u]))
            o = next(e_ for e_ in legs if e_[1] == o[1] and e_[2] == o[2])
            self.split(a, [e_ for e_ in legs if e_[1] == keep[a] or e_ == o], name=("loop", a))

        if has_positions:
            half = SPACING / 2
            new = {u: ("loop", u), v: ("loop", v)} | ({h: h2} if len(path) == 3 else {})
            for old, p in corners.items():
                self.positions[old] = tuple(p - half * f)
                self.positions[new[old]] = tuple(p + half * f)
            _carry_hadamards(self.graph, self.types, self.positions, P,
                             {u: pu, ("loop", u): pu, v: pv, ("loop", v): pv},
                             skip=set(new) - {u, v} | set(new.values()))
        self._shortest_way_round(loop)

    def loop_generator(self, g: Node, order: Sequence[Node] | None = None) -> None:
        """See :func:`loop_generator`."""
        legs = _legs(self, g)
        order = list(order) if order is not None else _cyclic(self, g, [e[1] for e in legs])
        if len(legs) != 4 or sorted(map(repr, order)) != sorted(repr(e[1]) for e in legs):
            raise ValueError(f"order must list the 4 distinct neighbours of {g!r}")
        a, b, c, d = order
        p = np.asarray(self.positions[g], float) if self.positions else None
        P0 = self._positions_near([g], 2) if self.positions else {}
        # g(a, b) = h(c, d); then g(a) - h(d) - ("loop", g, 2)(c) - ("loop", g, 1)(b) - g.
        self.split(g, [e for e in _legs(self, g) if e[1] in (a, b)], name=("loop", g, 3))
        h = ("loop", g, 3)
        loop = self.add_parallel_edge(g, h)
        f, f2 = sorted(self.graph[g][h])
        # g keeps a and f; its new half takes b and f'. h keeps d and f; its half takes c and f'.
        self.split(g, [e for e in _legs(self, g) if e[1] == a or (e[1] == h and e[2] == f)],
                   name=("loop", g, 1))
        self.split(h, [e for e in _legs(self, h) if e[1] == d or (e[1] == g and e[2] == f)],
                   name=("loop", g, 2))
        if p is not None:
            # Each spider towards its leg; the four are centred where g was.
            ring = [g] + [("loop", g, k) for k in (1, 2, 3)]
            offs = [SPACING / np.sqrt(2) * _unit(np.asarray(self.positions[n], float) - p)
                    for n in order]
            mean = np.mean(offs, axis=0)
            for node, o in zip(ring, offs):
                self.positions[node] = tuple(p + o - mean)
            _carry_hadamards(self.graph, self.types, self.positions, P0, {n: p for n in ring})
        self._shortest_way_round(loop)

    def _shortest_way_round(self, loop: int) -> None:
        """Reroute every web the short way round a loop: replace w by w + loop
        when that has less support on the loop's edges. The loop web has
        trivial boundary, so classes and boundary operators are unchanged,
        and the webs stay independent."""
        ring = {key: self._at[key][loop] for key in self._tracked[loop]}
        others = sorted(set().union(*(self._at[key] for key in ring)) - {loop})
        for wid in others:
            mine = {key: self._at[key].get(wid, (0, 0)) for key in ring}
            flip = {key: (b[0] ^ ring[key][0], b[1] ^ ring[key][1]) for key, b in mine.items()}
            if sum(map(any, flip.values())) < sum(map(any, mine.values())):
                for key, b in flip.items():
                    if any(b):
                        self._at[key][wid] = b
                    else:
                        self._at[key].pop(wid, None)

    def fuse(self, u: Node, v: Node) -> None:
        """See :func:`fuse`."""
        if u == v or _spider(self, u) is not _spider(self, v):
            raise ValueError("fuse needs two distinct spiders of the same colour")
        if self.graph.number_of_edges(u, v) != 1:
            raise ValueError("fuse needs exactly one edge between the spiders")
        for e in _legs(self, v):
            _, x, k = e
            old = self._remove_edge(v, x, k)
            if x != u:
                new = self._add_edge(u, x)
                for wid, xz in old.items():
                    self._set(new, wid, xz)
        self._remove_nodes([v])
        self._settle()

    def remove_identity(self, s: Node) -> None:
        """See :func:`remove_identity`."""
        _spider(self, s)
        legs = _legs(self, s)
        if len(legs) != 2:
            raise ValueError(f"{s!r} does not have degree 2")
        (e1, e2) = legs
        old = dict(self._on(e1))
        self._remove_nodes([s])
        e = self._add_edge(e1[1], e2[1])
        for wid, xz in old.items():
            self._set(e, wid, xz)
        self._settle()

    def cancel_hadamards(self, h1: Node, h2: Node) -> None:
        """See :func:`cancel_hadamards`."""
        if self.types.get(h1) is not NodeType.H or self.types.get(h2) is not NodeType.H:
            raise ValueError("cancel_hadamards needs two H nodes")
        if self.graph.number_of_edges(h1, h2) != 1:
            raise ValueError("H nodes must be joined by exactly one edge")
        (eu,) = [e for e in _legs(self, h1) if e[1] != h2]
        (ev,) = [e for e in _legs(self, h2) if e[1] != h1]
        old = dict(self._on(eu))
        self._remove_nodes([h1, h2])
        e = self._add_edge(eu[1], ev[1])
        for wid, xz in old.items():
            self._set(e, wid, xz)
        self._settle()

    # -- regions ------------------------------------------------------------------

    def split_boundary(self, S: Iterable[Node]) -> None:
        """See :func:`split_boundary`."""
        S = set(S)
        for v in sorted(cut_nodes(self, S), key=repr):
            self.split(v, [e for e in _legs(self, v) if e[1] in S])

    def split_by_side(self, side: Callable[[Node], Hashable], spiders: Iterable[Node]) -> None:
        """See :func:`split_by_side`."""
        for v in sorted(spiders, key=repr):
            home = side(v)
            away = sorted({side(u) for _, u, _ in _legs(self, v)} - {home}, key=repr)
            for s in away:
                keep = [e for e in _legs(self, v) if side(e[1]) != s]
                self.split(v, keep, name=("split", v, s))

    def prepare_cluster_region(self, S: Iterable[Node]) -> set[Node]:
        """See :func:`prepare_cluster_region`; returns the region."""
        S = set(S)
        S |= {p for p in self.inputs + self.outputs if set(self.graph.neighbors(p)) <= S}
        self.split_boundary(S)
        xs = [v for v in S if self.types.get(v) is NodeType.X]
        if xs:
            self.dualise(xs)
        # H nodes created between two region spiders belong to the region.
        S |= {h for h, t in self.types.items()
              if t is NodeType.H and set(self.graph.neighbors(h)) <= S}
        return S

    def loop_boundary(self, S: Iterable[Node]) -> tuple[set[Node], list[frozenset]]:
        """See :func:`loop_boundary`; returns the region and the rungs."""
        S = set(S)
        pairs = []
        for a, b, _ in cut_edges(self, S):
            inside, out = (a, b) if a in S else (b, a)
            if self.types[out] is NodeType.H:
                out = next(x for x in self.graph.neighbors(out) if x != inside)
            if loopable(self, out, inside):
                pairs.append((out, inside))
        rungs = []
        for out, inside in pairs:
            self.loop_edge(out, inside)
            S.add(("loop", inside))
            rungs += [frozenset((out, ("loop", out))), frozenset((inside, ("loop", inside)))]
        return S, rungs


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------
#
# Each takes an SED and returns a new one (one editor session); to apply many
# rules, use an SEDEditor directly.

def duality(sed: SEDiagram, v: Node) -> SEDiagram:
    """D_v: swap the colour of spider v and put an H node on each of its legs.

    The H-neighbour edge keeps the old label; the v-H edge has x and z swapped.
    """
    return dualise(sed, [v])


def dualise(sed: SEDiagram, nodes: Iterable[Node]) -> SEDiagram:
    """Apply D_v to every spider in ``nodes`` at once, cancelling the H pair
    that would appear on an edge between two of them (that edge just has its
    label's x and z swapped)."""
    ed = SEDEditor(sed)
    ed.dualise(nodes)
    return ed.freeze()


def insert_identity(sed: SEDiagram, e: Edge, colour: NodeType | str) -> SEDiagram:
    """I_e^c: put a degree-2 spider of colour c on edge e; both halves keep e's label."""
    ed = SEDEditor(sed)
    ed.insert_identity(e, colour)
    return ed.freeze()


def split(sed: SEDiagram, v: Node, A: Iterable[Edge], name: Hashable | None = None,
          direction: Sequence[float] | None = None) -> SEDiagram:
    """S_v(A, B): v keeps the legs in A; a new spider ``name`` of the same colour
    takes the rest (B); the two are joined by a single new edge f.

    On f, the matching component is the common value at v and the
    non-matching component is the parity over A.

    Layout: the two halves sit SPACING apart along ``direction`` (by default
    from A's neighbours towards B's), with their midpoint where v was. No
    other node moves.
    """
    ed = SEDEditor(sed)
    ed.split(v, A, name, direction)
    return ed.freeze()


# -- not web-preserving ------------------------------------------------------

def _joining_path(sed: SEDiagram | SEDEditor, u: Node, v: Node) -> list[Node]:
    """The single path joining u and v: ``[u, v]`` for one plain edge, or
    ``[u, h, v]`` for one degree-2 H node h (and no plain edge)."""
    hs = [h for h in sed.graph.neighbors(u)
          if sed.types[h] is NodeType.H and set(sed.graph.neighbors(h)) == {u, v}
          and sed.graph.degree(h) == 2]
    n = sed.graph.number_of_edges(u, v) if sed.graph.has_edge(u, v) else 0
    if n + len(hs) != 1:
        raise ValueError(f"{u!r} and {v!r} must be joined by exactly one edge or one H node")
    return [u, v] if n else [u, hs[0], v]


def add_parallel_edge(sed: SEDiagram, u: Node, v: Node) -> SEDiagram:
    """Join spiders u and v, already joined by one edge f (or one path f
    through an H node, if they have opposite colours), by a second copy f'.
    The linear map is unchanged (up to scalar), but the web space grows by
    the **loop web**: each spider's non-matching colour on its legs in f and
    f' (swapped across an H node).

    Every web takes the matching components of f on f'; the loop web is added
    as a detector (it has trivial boundary).

    Layout: on an H path, the old and new H nodes sit SPACING apart,
    perpendicular to u - v, with their midpoint where the old one was. No
    other node moves."""
    ed = SEDEditor(sed)
    ed.add_parallel_edge(u, v)
    return ed.freeze()


def loopable(sed: SEDiagram | SEDEditor, u: Node, v: Node) -> bool:
    """Whether :func:`loop_edge` applies: u and v are joined by one edge (or
    one H node) and each has exactly two other legs."""
    try:
        path = _joining_path(sed, u, v)
    except ValueError:
        return False
    return all(sed.graph.degree(a) == 3 and sed.graph.number_of_edges(a, b) == 1
               for a, b in ((u, path[1]), (v, path[-2])))


def loop_edge(sed: SEDiagram, u: Node, v: Node,
              keep_u: Node | None = None, keep_v: Node | None = None) -> SEDiagram:
    """Turn the edge (or H path) joining u and v, each with exactly two other
    legs, into a loop of four spiders with one external leg each.

    Adds a parallel copy (:func:`add_parallel_edge`), then splits u so that it
    keeps its leg to ``keep_u`` and the original copy, while ``("loop", u)``
    takes its other leg and the new copy; likewise for v. The cycle is
    u - v - ("loop", v) - ("loop", u) - u. Adds one detector, the loop web,
    and reroutes every web the short way round it.

    Layout: the four spiders sit on a rectangle centred on the midpoint of u
    and v (where the generator they were split from was): u - v keeps its
    length, and the loop is SPACING wide, opening towards the legs that move
    to the new spiders. By default each of u and v keeps the leg on its own
    side, so legs do not cross. No other node moves."""
    ed = SEDEditor(sed)
    ed.loop_edge(u, v, keep_u, keep_v)
    return ed.freeze()


def loop_boundary(sed: SEDiagram, S: Iterable[Node]
                  ) -> tuple[SEDiagram, set[Node], list[frozenset]]:
    """Loop every loopable boundary edge of S (:func:`loopable`,
    :func:`loop_edge`), e.g. after ``prepare_cluster_region``; other boundary
    edges are left as they are.

    Returns the new SED, the region (with the inside loop halves), and the
    two same-colour edges of each loop off the boundary, (u, ("loop", u)) and
    (v, ("loop", v)), e.g. to realise as Bell pairs. The boundary then
    crosses each loop twice."""
    ed = SEDEditor(sed)
    S, rungs = ed.loop_boundary(S)
    return ed.freeze(), S, rungs


def loop_generator(sed: SEDiagram, g: Node, order: Sequence[Node] | None = None) -> SEDiagram:
    """Turn a generator spider into a loop of spiders, one per leg.

    Split ``g`` into two halves, join them by a second edge
    (:func:`add_parallel_edge`), then split each half so that every spider
    keeps one leg: the four spiders form a cycle with the legs to ``order``
    (neighbours of ``g``) in cyclic order. ``g`` keeps ``order[0]``; the others
    are ``("loop", g, k)`` for leg ``order[k]``. Needs exactly 4 legs, each to
    a distinct neighbour. Adds one detector, the loop web."""
    ed = SEDEditor(sed)
    ed.loop_generator(g, order)
    return ed.freeze()


def _cyclic(sed: SEDiagram | SEDEditor, g: Node, nbs: list[Node]) -> list[Node]:
    """Neighbours of g sorted by angle around it in the (x, y) plane."""
    p = np.asarray(sed.positions[g][:2], float)
    return sorted(nbs, key=lambda n: np.arctan2(*(np.asarray(sed.positions[n][:2], float) - p)[::-1]))


# -- inverses ----------------------------------------------------------------

def fuse(sed: SEDiagram, u: Node, v: Node) -> SEDiagram:
    """Inverse of split: fuse spiders u and v (same colour, exactly one edge
    between them) into u. The joining edge is dropped; other labels are kept."""
    ed = SEDEditor(sed)
    ed.fuse(u, v)
    return ed.freeze()


def remove_identity(sed: SEDiagram, s: Node) -> SEDiagram:
    """Inverse of identity insertion: remove a degree-2 spider s."""
    ed = SEDEditor(sed)
    ed.remove_identity(s)
    return ed.freeze()


def cancel_hadamards(sed: SEDiagram, h1: Node, h2: Node) -> SEDiagram:
    """Replace u - h1 - h2 - v (adjacent H nodes) by a single edge u - v."""
    ed = SEDEditor(sed)
    ed.cancel_hadamards(h1, h2)
    return ed.freeze()


# ---------------------------------------------------------------------------
# Regions and boundary splitting
# ---------------------------------------------------------------------------

def cut_edges(sed: SEDiagram | SEDEditor, S: Iterable[Node]) -> list[Edge]:
    """dE(S): edges with exactly one end in S."""
    S = set(S)
    return [(u, v, k) for u, v, k in sed.edges if (u in S) != (v in S)]


def cut_nodes(sed: SEDiagram | SEDEditor, S: Iterable[Node]) -> set[Node]:
    """dV(S): nodes of S with a neighbour outside S."""
    S = set(S)
    return {u if u in S else v for u, v, _ in cut_edges(sed, S)}


def has_matching_boundary(sed: SEDiagram | SEDEditor, S: Iterable[Node]) -> bool:
    """Whether dE(S) is a matching: no two cut edges share an endpoint."""
    ends = [x for u, v, _ in cut_edges(sed, S) for x in (u, v)]
    return len(ends) == len(set(ends))


def split_boundary(sed: SEDiagram, S: Iterable[Node]) -> SEDiagram:
    """Split each spider of dV(S) into an inside half (keeping its label and its
    legs into S) and an outside half ``("split", v)``, joined by one edge.
    Afterwards S has a matching boundary."""
    ed = SEDEditor(sed)
    ed.split_boundary(S)
    return ed.freeze()


def split_by_side(sed: SEDiagram, side: Callable[[Node], Hashable],
                  spiders: Iterable[Node]) -> SEDiagram:
    """Split each of ``spiders`` so that every half only touches neighbours on
    one side: v keeps its legs to neighbours with ``side(u) == side(v)``, and a
    new spider ``("split", v, s)`` takes the legs to side s, for each other s."""
    ed = SEDEditor(sed)
    ed.split_by_side(side, spiders)
    return ed.freeze()


def prepare_cluster_region(sed: SEDiagram, S: Iterable[Node]) -> tuple[SEDiagram, set[Node]]:
    """Web-preserving steps of clusterisation: split the boundary of S (so it
    is a matching) and dualise S's X spiders (so D[S] is graph-like).

    Ports attached only to S join S, so a strand that starts or ends inside the
    region keeps its port. Returns the new SED and the region (inside halves
    keep their labels; outside halves are ``("split", v)``).
    """
    ed = SEDEditor(sed)
    S = ed.prepare_cluster_region(S)
    return ed.freeze(), S
