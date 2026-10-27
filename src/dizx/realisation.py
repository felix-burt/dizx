"""Realisations of SEDs and QEC experiments (see docs/realisations.md).

A ``RealisedSED`` adds classical outputs and world-lines to an SED. A
``QECExperiment`` additionally closes every port with a preparation or a
measurement.

Closing a port with a degree-1 preparation spider, or with a measurement spider
plus classical output, forces the non-matching component on its boundary edge
to vanish. Only webs satisfying all of these constraints survive:

    basis 'Z' (|0> / Z measurement, X spider):  x_e = 0  (boundary in {I, Z})
    basis 'X' (|+> / X measurement, Z spider):  z_e = 0  (boundary in {I, X})

With all ports closed, every surviving web has trivial boundary. Those built
only from SED detectors, stabilisers and co-stabilisers are **detectors**;
those involving a logical are **observables**.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable, Literal, Mapping, Sequence

import networkx as nx
import numpy as np

from dizx._f2 import nullspace, sp_matmul, sp_rows
from dizx.zx_diagram import (
    SPIDERS, Edge, Node, NodeType, Outcome, PauliWeb, SEDiagram, ZXDiagram, column_transport,
    draw_diagram, port_cols_bits, web_matrix, webs_from_matrix,
)


# ---------------------------------------------------------------------------
# Interactions and links
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Interaction:
    """A connection between spiders ``nodes[0]`` and ``nodes[-1]`` on different
    world-lines, through zero or more H nodes, along ``edges``."""
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]

    @property
    def ends(self) -> tuple[Node, Node]:
        return self.nodes[0], self.nodes[-1]

    @property
    def hadamards(self) -> int:
        return len(self.nodes) - 2

    def same_colour(self, types: Mapping[Node, NodeType]) -> bool:
        """Effective colours agree: a link (Bell pair / fusion), not a gate."""
        u, v = self.ends
        return (types[u] is types[v]) != (self.hadamards % 2 == 1)


def interactions(diagram: ZXDiagram, worldlines: Sequence[Sequence[Node]]) -> list[Interaction]:
    """Every connection between spiders that is not part of a world-line,
    following chains of H nodes that are not on world-lines. Classical outputs
    and ports are not spiders here."""
    on_line = {v for p in worldlines for v in p}
    line_edges = set()
    for p in worldlines:
        for a, b in zip(p, p[1:]):
            line_edges.add(frozenset((a, b)))
    outputs = set(diagram.classical_outputs())
    spiders = [v for v, t in diagram.types.items() if t in SPIDERS and v not in outputs]
    found: dict[tuple, Interaction] = {}
    for u in spiders:
        for e in diagram.graph.edges(u, keys=True):
            if frozenset(e[:2]) in line_edges or e[1] in outputs:
                continue
            nodes, edges, prev, cur = [u], [e], u, e[1]
            while diagram.types[cur] is NodeType.H and cur not in on_line:
                (nxt,) = [f for f in diagram.graph.edges(cur, keys=True) if f[1] != prev]
                nodes.append(cur)
                edges.append(nxt)
                prev, cur = cur, nxt[1]
            if diagram.types[cur] not in SPIDERS:
                continue
            nodes.append(cur)
            key = tuple(sorted(repr(tuple(sorted(f[:2], key=repr)) + (f[2],)) for f in edges))
            if key not in found:
                found[key] = Interaction(tuple(nodes), tuple(edges))
    return list(found.values())


LinkKind = Literal["bell", "fusion"]


@dataclass(frozen=True)
class Link:
    """An interaction realised as a Bell-pair preparation (both world-lines
    start here) or a fusion (both end here). Same-colour interactions are
    always links; an opposite-colour one is a link only when asked for (e.g.
    ``u -H- v`` between Z spiders as an ``XZ``/``ZX`` fusion), else a gate.

    A fusion measures ``ops[0]`` and ``ops[1]`` on the qubits at ``ends``
    (``("XX", "ZZ")``, or ``("XZ", "ZX")`` across one H node), with outcome
    labels ``outcomes``. A Bell preparation prepares the +1 eigenstate of both.
    """
    interaction: Interaction
    kind: LinkKind

    @property
    def ops(self) -> tuple[str, str]:
        return ("XZ", "ZX") if self.interaction.hadamards % 2 else ("XX", "ZZ")

    @property
    def outcomes(self) -> tuple[Hashable, Hashable]:
        u, v = self.interaction.ends
        return tuple(("fusion", u, v, op) for op in self.ops)  # type: ignore[return-value]


def _records(diagram: ZXDiagram, links: Sequence[Link],
             webs: Sequence[PauliWeb]) -> list[set[Hashable]]:
    """Each web's record (see ``RealisedSED.record``), all at once: the labels
    of the classical outputs whose leg it highlights and of the fusion
    outcomes it picks up (the first operator's outcome when the edge leaving
    ``ends[0]`` has an x component, the second's for z)."""
    n = diagram.n_edges
    cols: list[int] = []
    labels: list[Hashable] = []
    outs, legs = diagram.classical_legs()
    for c, i in zip(outs, legs.tolist()):
        p = diagram.phases[c]
        if isinstance(p, Outcome):
            cols += [i, n + i]
            labels += [p.label, p.label]
    for link in links:
        if link.kind == "fusion":
            i = diagram.edge_index(link.interaction.edges[0])
            cols += [i, n + i]
            labels += list(link.outcomes)
    if not cols:
        return [set() for _ in webs]
    order = np.argsort(cols, kind="stable")
    cols_s = np.asarray(cols, dtype=np.int64)[order]
    out = []
    for w in webs:
        # Positions of the web's ones among the (sorted) record columns.
        lo = np.searchsorted(cols_s, w.idx, side="left")
        hi = np.searchsorted(cols_s, w.idx, side="right")
        out.append({labels[order[k]] for a, b in zip(lo.tolist(), hi.tolist()) for k in range(a, b)})
    return out


def _resolve_links(
    diagram: ZXDiagram,
    worldlines: Sequence[Sequence[Node]],
    measured: set[Node],
    kinds: Mapping[frozenset, LinkKind],
) -> list[Link]:
    """Same-colour interactions as links; a kind not given in ``kinds`` (keyed
    by ``frozenset(ends)``) is inferred from whether both world-lines start or
    both end there."""
    starts = {p[0] for p in worldlines}
    ends = {p[-1] for p in worldlines}
    links = []
    for it in interactions(diagram, worldlines):
        u, v = it.ends
        # An opposite-colour interaction is a gate unless a link is asked for
        # (a fusion's operators depend only on the H nodes between its ends).
        if not it.same_colour(diagram.types) and frozenset((u, v)) not in kinds:
            continue
        kind = kinds.get(frozenset((u, v)))
        if kind is None:
            if {u, v} <= starts and not {u, v} & ends:
                kind = "bell"
            elif {u, v} <= ends and not {u, v} & starts:
                kind = "fusion"
            else:
                raise ValueError(f"link {u!r} - {v!r}: specify 'bell' or 'fusion'")
        if kind == "bell" and not {u, v} <= starts:
            raise ValueError(f"Bell link {u!r} - {v!r}: both world-lines must start there")
        if kind == "fusion" and not ({u, v} <= ends and not {u, v} & measured):
            raise ValueError(f"fusion {u!r} - {v!r}: both world-lines must end there, unmeasured")
        links.append(Link(it, kind))
    return links


# ---------------------------------------------------------------------------
# RealisedSED
# ---------------------------------------------------------------------------

class RealisedSED(ZXDiagram):
    """An SED with classical outputs and world-lines.

    Attributes
    ----------
    sed : SEDiagram
        The underlying (unrealised) SED.
    classical : dict[Node, Node]
        Measured spider ``v`` -> its classical output ``("meas", v)``, whose
        phase is ``Outcome(v)``.
    worldlines : list[tuple[Node, ...]]
        Vertex-disjoint directed paths, earliest node first. A path may pass
        through H nodes, which do not count as spiders.
    links : list[Link]
        Same-colour interactions between world-lines, realised as Bell-pair
        preparations or fusions. Kinds may be given in ``links`` (keyed by
        ``frozenset`` of the two end spiders) and are otherwise inferred.
    detectors, stabilisers, co_stabilisers, logicals : list[PauliWeb]
        The SED's webs, each extended uniquely onto the classical outputs.
    graph_region : set[Node]
        Spiders realised as resource-state (graph) qubits, e.g. the region S
        of ``hybrid``; empty for a purely circuit-based realisation. Only used
        to label circuit operations (see ``circuit_extractor``).
    """

    def __init__(
        self,
        sed: SEDiagram,
        measured: Iterable[Node],
        worldlines: Iterable[Sequence[Node]],
        links: Mapping[frozenset, LinkKind] | None = None,
        graph_region: Iterable[Node] = (),
    ) -> None:
        graph = nx.MultiGraph(sed.graph)
        types = dict(sed.types)
        phases: dict[Node, Outcome] = {}
        self.classical: dict[Node, Node] = {}
        for v in dict.fromkeys(measured):
            if sed.types.get(v) not in SPIDERS or sed.graph.degree(v) < 1:
                raise ValueError(f"can only measure spiders with at least one leg, got {v!r}")
            c = ("meas", v)
            if c in types:
                raise ValueError(f"node label {c!r} already in use")
            graph.add_edge(v, c)
            types[c] = types[v]
            phases[c] = Outcome(v)
            self.classical[v] = c
        super().__init__(graph, types, phases, sed.inputs, sed.outputs)

        self.sed = sed
        self.graph_region = set(graph_region)
        self.worldlines = [tuple(p) for p in worldlines]
        self._check_worldlines()
        self.links = _resolve_links(self, self.worldlines, set(self.classical), links or {})
        self.detectors = self._extend(sed.detectors)
        self.stabilisers = self._extend(sed.stabilisers)
        self.co_stabilisers = self._extend(sed.co_stabilisers)
        self.logicals = self._extend(sed.logicals)

        self.positions = dict(sed.positions)
        if self.positions:
            for v, c in self.classical.items():
                x, y, z = self.positions[v]
                self.positions[c] = (x, y, z + 0.35)

    @classmethod
    def circuit_based(cls, sed: SEDiagram, links: LinkKind = "fusion") -> RealisedSED:
        """Data world-lines along the shortest path from each input to its
        output (the one with the same qubit for ``("in", q)`` / ``("out", q)``
        ports, else the one in the same position); every other spider is an
        ancilla on its own singleton world-line. A closed wire (see
        ``close_ports``) runs between its end spiders and is measured at the
        last; its bases follow from the colours of its first and last spiders.

        Same-colour edges between ancillas (e.g. split generators) are realised
        as ``links``: ``"bell"`` prepares the two ancillas as a Bell pair and
        measures both; ``"fusion"`` fuses them at the end, without a classical
        output. Every other ancilla gets a classical output.
        """
        data = [nx.shortest_path(sed.graph, i, o) for i, o in _wire_ends(sed)]
        on_data = {v for p in data for v in p}
        ancillas = [v for v, t in sed.types.items() if t in SPIDERS and v not in on_data]
        worldlines = data + [(v,) for v in ancillas]
        linked = [it.ends for it in interactions(sed, worldlines) if it.same_colour(sed.types)]
        kinds = {frozenset(e): links for e in linked}
        fused = {v for e in linked for v in e} if links == "fusion" else set()
        ends = [p[-1] for p in data if sed.types[p[-1]] in SPIDERS]  # closed wires
        return cls(sed, [v for v in ancillas if v not in fused] + ends, worldlines, kinds)

    @classmethod
    def measurement_based(cls, sed: SEDiagram) -> RealisedSED:
        """Duality on every X spider makes the diagram graph-like; then every
        spider is a resource-state qubit on its own world-line, measured unless
        it is next to an output port. Ports stay open legs of the resource
        state, on no world-line; ``QECExperiment`` closes them onto graph
        qubits."""
        from dizx.transformations import dualise

        xs = [v for v, t in sed.types.items() if t is NodeType.X]
        sed = dualise(sed, xs) if xs else sed
        spiders = [v for v, t in sed.types.items() if t is NodeType.Z]
        at_output = {_port_spider(sed, p)[1] for p in sed.outputs}
        return cls(sed, [v for v in spiders if v not in at_output], [(v,) for v in spiders],
                   graph_region=spiders)

    @classmethod
    def hybrid(cls, sed: SEDiagram, S: Iterable[Node],
               links: Mapping[frozenset, LinkKind] | None = None) -> RealisedSED:
        """Clusterise region S and realise the rest circuit-based.

        ``sed`` and ``S`` should come from ``prepare_cluster_region``, so S has
        a matching boundary and is graph-like. Inside S every spider is its own
        world-line (continued to any port it touches) with a classical output,
        except boundary spiders, whose boundary edge is realised as a fusion.
        Outside S, qubits whose ports lie outside S get data world-lines and
        every other spider is an ancilla, measured unless it is fused.

        Other same-colour edges between world-lines need their kind in
        ``links`` (keyed by ``frozenset`` of the two spiders), e.g. the Bell
        pairs of a loop (``transformations.loop_edge``). An opposite-colour
        boundary edge is allowed when ``links`` names it a fusion.
        """
        S = set(S)
        spiders = {v for v, t in sed.types.items() if t in SPIDERS}
        inside = spiders & S
        boundary = [it for it in interactions(sed, [])
                    if (it.ends[0] in S) != (it.ends[1] in S)]
        if not all(it.same_colour(sed.types) or (links or {}).get(frozenset(it.ends)) == "fusion"
                   for it in boundary):
            raise ValueError("every boundary edge of S must be a split (same-colour) edge "
                             "or a fusion given in links")
        fused = {v for it in boundary for v in it.ends}
        fused |= {v for e, kind in (links or {}).items() if kind == "fusion" for v in e}

        worldlines, measured = [], []
        for v in sorted(inside, key=repr):
            path = [v]
            for port in sed.inputs:
                route = _port_route(sed, port, v)
                if route is not None:
                    path = route + path
            for port in sed.outputs:
                route = _port_route(sed, port, v)
                if route is not None:
                    path = path + route[::-1]
            worldlines.append(tuple(path))
            if path[-1] == v and v not in fused:
                measured.append(v)

        outside = sed.graph.subgraph(set(sed.types) - S)
        data = [nx.shortest_path(outside, i, o) for i, o in _wire_ends(sed)
                if i not in S and o not in S]
        measured += [p[-1] for p in data if sed.types[p[-1]] in SPIDERS]  # closed wires
        on_data = {v for p in data for v in p}
        # In node order (not set order), so the circuit is hash-seed independent.
        ancillas = [v for v in sed.types if v in spiders and v not in inside and v not in on_data]
        worldlines += data + [(v,) for v in ancillas]
        measured += [v for v in ancillas if v not in fused]
        kinds = {frozenset(it.ends): "fusion" for it in boundary}
        kinds.update(links or {})
        return cls(sed, measured, worldlines, kinds, graph_region=inside)

    # -- construction -------------------------------------------------------

    def _check_worldlines(self) -> None:
        seen: set[Node] = set()
        classical = set(self.classical.values())
        for path in self.worldlines:
            if not path:
                raise ValueError("empty world-line")
            for v in path:
                if v not in self.types:
                    raise ValueError(f"world-line node {v!r} not in diagram")
                if v in classical:
                    raise ValueError(f"classical output {v!r} on a world-line")
                if v in seen:
                    raise ValueError(f"world-lines share node {v!r}")
                seen.add(v)
            for u, v in zip(path, path[1:]):
                if not self.graph.has_edge(u, v):
                    raise ValueError(f"world-line step {u!r} -> {v!r} is not an edge")
                if self.types[u] is NodeType.P and u not in self.inputs:
                    raise ValueError(f"world-line leaves port {u!r}, which is not an input")
                if self.types[v] is NodeType.P and v not in self.outputs:
                    raise ValueError(f"world-line enters port {v!r}, which is not an output")

    def _extend(self, webs: Sequence[PauliWeb]) -> list[PauliWeb]:
        """Carry webs of ``sed`` onto ``self``: classical-output legs take the
        matching component at their spider and no non-matching component."""
        if not webs:
            return []
        sed, n, m = self.sed, self.n_edges, self.sed.n_edges
        dst = np.array([self.edge_index(e) for e in sed.edges], dtype=np.int64)
        src_c = [np.arange(2 * m)]
        dst_c = [np.concatenate([dst, n + dst])]
        for v, c in self.classical.items():
            leg = sed.legs(v)[0]
            j = self.edge_index((v, c, 0))
            if sed.types[v] is NodeType.Z:
                src_c.append([leg])
                dst_c.append([j])
            else:
                src_c.append([m + leg])
                dst_c.append([n + j])
        P = column_transport(np.concatenate(src_c), np.concatenate(dst_c), 2 * m, 2 * n)
        out = sp_matmul(web_matrix(webs, m), P)
        assert not any(self.violations(out)), "outcome extension failed"
        return webs_from_matrix(self, out)

    # -- queries ------------------------------------------------------------

    @property
    def webs(self) -> list[PauliWeb]:
        return self.detectors + self.stabilisers + self.co_stabilisers + self.logicals

    def record(self, w: PauliWeb) -> set[Hashable]:
        """Outcome labels whose parity is ``mu(w)``: a parity check for a
        detector, the sign of a stabiliser or logical. Includes fusion outcomes."""
        return _records(self, self.links, [w])[0]

    def records(self, webs: Sequence[PauliWeb]) -> list[set[Hashable]]:
        """``record`` of each web."""
        return _records(self, self.links, webs)

    def draw(self, web: PauliWeb | None = None, ax=None, title: str | None = None):
        """Draw at ``self.positions`` with world-line arrows; see ``draw_diagram``."""
        return draw_diagram(self, self.positions, web, ax, title, self.worldlines,
                            _fusion_edges(self.links))


def _fusion_edges(links: Iterable[Link]) -> list[Edge]:
    return [e for link in links if link.kind == "fusion" for e in link.interaction.edges]


def resource_states(realised: RealisedSED, S: Iterable[Node]) -> list[set[Node]]:
    """The graph states of a hybrid realisation of region S: S's spiders,
    joined by every interaction between them that is not a link (each H edge
    is a CZ). Links (fusions, Bell pairs) join different resource states."""
    S = set(S)
    linked = {frozenset(link.interaction.ends) for link in realised.links}
    g = nx.Graph()
    g.add_nodes_from(v for v in S if realised.types[v] in SPIDERS
                     and v not in realised.classical.values())
    for it in interactions(realised, realised.worldlines):
        u, v = it.ends
        if u in g and v in g and frozenset((u, v)) not in linked:
            g.add_edge(u, v)
    return [set(c) for c in nx.connected_components(g)]


def _wire_ends(sed: SEDiagram) -> list[tuple[Node, Node]]:
    """The two ends of each qubit wire: an input port (or, for a closed input,
    the spider now starting its wire) and the matching output port (or closing
    spider). Paired by qubit for ``("in", q)`` / ``("out", q)`` labels,
    otherwise by position."""
    def label(p: Node, tag: str):
        return p[1] if isinstance(p, tuple) and len(p) == 2 and p[0] == tag else None

    closed = getattr(sed, "closed", {})
    starts = {label(p, "in"): p for p in sed.inputs}
    starts.update({label(p, "in"): v for p, v in closed.items() if label(p, "in") is not None})
    ends = {label(p, "out"): p for p in sed.outputs}
    ends.update({label(p, "out"): v for p, v in closed.items() if label(p, "out") is not None})
    if None not in starts and None not in ends and starts.keys() == ends.keys():
        return [(starts[q], ends[q]) for q in starts]
    if closed:
        raise ValueError("closed wires need ('in', q) / ('out', q) port labels")
    if len(sed.inputs) != len(sed.outputs):
        raise ValueError("circuit-based realisation needs as many inputs as outputs")
    return list(zip(sed.inputs, sed.outputs))


def _port_route(sed: SEDiagram, port: Node, v: Node) -> list[Node] | None:
    """``[port, H nodes...]`` if ``port`` reaches spider ``v`` through H nodes only."""
    route, prev, cur = [port], None, port
    while True:
        nxt = [u for u in sed.graph.neighbors(cur) if u != prev]
        if len(nxt) != 1:
            return None
        prev, cur = cur, nxt[0]
        if cur == v:
            return route
        if sed.types[cur] is not NodeType.H:
            return None
        route.append(cur)


def _port_spider(diagram: ZXDiagram, port: Node) -> tuple[list[Node], Node]:
    """``([port, H nodes...], spider)``: the walk from a port to its spider."""
    route, prev, cur = [port], None, port
    while True:
        (cur, prev), = [(u, cur) for u in diagram.graph.neighbors(cur) if u != prev]
        if diagram.types[cur] is not NodeType.H:
            if diagram.types[cur] not in SPIDERS:
                raise ValueError(f"port {port!r} does not lead to a spider")
            return route, cur
        route.append(cur)


# ---------------------------------------------------------------------------
# QECExperiment
# ---------------------------------------------------------------------------

Basis = str  # 'Z' or 'X'


def _natural_basis(diagram: ZXDiagram, port: Node) -> Basis:
    """The basis a port's wire end spider fixes: walk from the port through
    H nodes to the first spider; Z (X) gives basis X (Z), swapped by each H."""
    prev, cur, flips = None, port, 0
    while True:
        (nxt,) = [u for u in diagram.graph.neighbors(cur) if u != prev] or [None]
        if nxt is None:
            raise ValueError(f"port {port!r} does not lead to a spider")
        prev, cur = cur, nxt
        t = diagram.types[cur]
        if t is NodeType.H:
            flips += 1
        elif t in SPIDERS:
            colour = t if flips % 2 == 0 else (NodeType.X if t is NodeType.Z else NodeType.Z)
            return "X" if colour is NodeType.Z else "Z"
        else:
            raise ValueError(f"port {port!r} does not lead to a spider")
Origin = tuple[tuple[str, int], ...]  # ((family, index), ...) of realised-SED webs

_CLOSER = {"Z": NodeType.X, "X": NodeType.Z}  # spider type closing a port in a basis
_OTHER = {NodeType.Z: NodeType.X, NodeType.X: NodeType.Z}
FAMILIES = ("detector", "stabiliser", "co-stabiliser", "logical")


class QECExperiment(ZXDiagram):
    """A realised SED with every input prepared and every output measured.

    Attributes
    ----------
    realised : RealisedSED
    preps, measurements : dict[Node, Basis]
        Basis for each input / output port of ``realised``. A port not given
        follows the colour of the spider its wire starts or ends at (counting
        H nodes on the way): Z gives X, X gives Z.
    worldlines : list[tuple[Node, ...]]
        ``realised.worldlines`` with each input port replaced by its
        preparation spider ``("prep", p)`` and each output port by its
        measurement spider ``("readout", p)``. A port on no world-line is an
        open leg of a resource state: its closing merges into the port's
        spider when the colours match, and is otherwise a measured graph qubit
        ``("prep", p)`` / ``("readout", p)`` of that spider's colour behind an
        H node, on a world-line of its own.
    graph_region : set[Node]
        ``realised.graph_region`` plus the graph qubits added by closings.
    classical : dict[Node, Node]
        Measured spider -> classical output, including the readouts.
    detectors, observables : list[PauliWeb]
        A basis of the surviving webs, split as described in the module docstring.
    detector_origins, observable_origins : list[Origin]
        Which realised-SED webs each one is the sum of.
    lost : list[tuple[str, int]]
        Realised-SED webs that do not survive on their own.
    """

    def __init__(
        self,
        realised: RealisedSED,
        preps: Mapping[Node, Basis] | None = None,
        measurements: Mapping[Node, Basis] | None = None,
    ) -> None:
        preps, measurements = dict(preps or {}), dict(measurements or {})
        if not set(preps) <= set(realised.inputs) or not set(measurements) <= set(realised.outputs):
            raise ValueError("preparations must be on inputs and measurements on outputs")
        for p in realised.inputs:
            preps.setdefault(p, _natural_basis(realised, p))
        for p in realised.outputs:
            measurements.setdefault(p, _natural_basis(realised, p))
        if not set(preps.values()) | set(measurements.values()) <= set(_CLOSER):
            raise ValueError("bases must be 'Z' or 'X'")
        self.realised = realised
        self.preps = dict(preps)
        self.measurements = dict(measurements)

        graph = nx.MultiGraph(realised.graph)
        types = dict(realised.types)
        phases = dict(realised.phases)
        self.classical: dict[Node, Node] = dict(realised.classical)
        self.graph_region = set(realised.graph_region)
        positions = dict(realised.positions)
        on_line = {v for path in realised.worldlines for v in path}
        closers: dict[Node, Node] = {}
        new_lines: list[tuple[Node, ...]] = []
        # How a realised edge's labels reach a new edge: (realised edge index,
        # new edge, "copy" | "swap" (x <-> z) | "match" (the matching component
        # of the spider, as on a classical leg)).
        moves: list[tuple[int, tuple[Node, Node], str]] = []
        dropped: set[Node] = set()

        def measure(v: Node) -> Node:
            c = ("meas", v)
            graph.add_edge(v, c)
            types[c] = types[v]
            phases[c] = Outcome(v)
            self.classical[v] = c
            if positions:
                x, y, z = positions[v]
                positions[c] = (x, y, z + 0.35)
            return c

        for p, basis, closes_output in ([(p, b, False) for p, b in self.preps.items()]
                                        + [(p, b, True) for p, b in self.measurements.items()]):
            pos = positions.pop(p, None)
            if p in on_line:
                # The port's world-line starts at a preparation spider or ends
                # at a readout spider.
                (u,) = realised.graph.neighbors(p)
                (i,) = realised.legs(p)
                s = ("readout" if closes_output else "prep", p)
                graph.remove_node(p)
                graph.add_edge(u, s)
                del types[p]
                types[s] = _CLOSER[basis]
                closers[p] = s
                dropped.add(p)
                moves.append((i, (s, u), "copy"))
                if pos is not None:
                    positions[s] = pos
                if closes_output:
                    moves.append((i, (s, measure(s)), "copy"))
                continue
            # An open leg of a resource state (see RealisedSED.measurement_based):
            # a closing spider of v's colour merges into v; one of the other
            # colour is a graph qubit of v's colour behind an H node, measured.
            route, v = _port_spider(realised, p)
            (e,) = [e for e in realised.graph.edges(route[-1], keys=True) if e[1] == v]
            i = realised.edge_index(e)
            graph.remove_nodes_from(route)
            for u in route:
                del types[u]
                positions.pop(u, None)
            dropped |= set(route)
            colour = _CLOSER[basis] if len(route) % 2 else _OTHER[_CLOSER[basis]]
            if closes_output and v not in self.classical:
                moves.append((i, (v, measure(v)), "match"))
            if colour is types[v]:
                continue
            g = ("readout" if closes_output else "prep", p)
            h = ("H", g)
            types[g], types[h] = types[v], NodeType.H
            graph.add_edge(v, h)
            graph.add_edge(h, g)
            if positions:
                positions[g] = pos
                positions[h] = tuple((np.asarray(positions[v]) + np.asarray(pos)) / 2)
            moves += [(i, (v, h), "copy"), (i, (h, g), "swap"), (i, (g, measure(g)), "swap")]
            new_lines.append((g,))
            if v in self.graph_region:
                self.graph_region.add(g)
        super().__init__(graph, types, phases)
        self.positions = positions
        self._moves = moves
        self._dropped = dropped
        self.worldlines = [tuple(closers.get(v, v) for v in p) for p in realised.worldlines]
        self.worldlines += new_lines
        self.links = realised.links
        self._check_worldlines()
        self._find_webs()

    @classmethod
    def memory(cls, realised: RealisedSED, basis: Basis = "Z") -> QECExperiment:
        """Prepare every input and measure every output in the same basis."""
        return cls(realised, {p: basis for p in realised.inputs},
                   {p: basis for p in realised.outputs})

    # -- construction -------------------------------------------------------

    def _check_worldlines(self) -> None:
        """Every qubit is initialised and measured: every spider lies on a
        world-line, and each world-line ends at a measured or fused spider."""
        on = {v for p in self.worldlines for v in p}
        outputs = set(self.classical.values())
        on |= outputs
        missing = [v for v, t in self.types.items() if t in SPIDERS and v not in on]
        if missing:
            raise ValueError(f"spiders on no world-line: {missing[:5]}")
        fused = {v for link in self.links if link.kind == "fusion" for v in link.interaction.ends}
        for path in self.worldlines:
            if path[-1] not in self.classical and path[-1] not in fused:
                raise ValueError(f"world-line ending at {path[-1]!r} is never measured")

    def _port_constraints(self, V) -> np.ndarray:
        """For each row of web matrix ``V``, one bit per closed port: the
        component its closing spider forbids."""
        r = self.realised
        cols = []
        for p in r.inputs + r.outputs:
            basis = self.preps.get(p) or self.measurements[p]
            (i,) = r.legs(p)
            cols.append(i if basis == "Z" else r.n_edges + i)
        return port_cols_bits(V, cols)

    def _close(self, V) -> list[PauliWeb]:
        """Carry surviving webs of ``realised`` (rows of web matrix V) onto ``self``."""
        r = self.realised
        n, m = self.n_edges, r.n_edges
        src, dst = [], []
        for i, e in enumerate(r.edges):
            if not ({e[0], e[1]} & self._dropped):
                j = self.edge_index(e)
                src += [i, m + i]
                dst += [j, n + j]
        for i, (a, b), how in self._moves:
            j = self.edge_index((a, b, 0))
            if how == "copy":
                src += [i, m + i]
                dst += [j, n + j]
            elif how == "swap":
                src += [i, m + i]
                dst += [n + j, j]
            else:
                k = 0 if self.types[a] is NodeType.Z else 1
                src.append(k * m + i)
                dst.append(k * n + j)
        P = column_transport(np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64),
                             2 * m, 2 * n)
        out = sp_matmul(V, P)
        assert not any(self.violations(out)), "closing a surviving web failed"
        return webs_from_matrix(self, out)

    def _find_webs(self) -> None:
        r = self.realised
        origin = [(f, i) for f, ws in zip(FAMILIES, (r.detectors, r.stabilisers,
                                                     r.co_stabilisers, r.logicals))
                  for i in range(len(ws))]
        webs = r.webs
        V = web_matrix(webs, r.n_edges)
        C = self._port_constraints(V)
        survives = ~C.any(axis=1)
        self.lost = [o for o, s in zip(origin, survives) if not s]

        # Combinations of realised webs that survive (as sorted web indices):
        # individual survivors first, then combinations of the others that
        # complete the surviving subspace (a nullspace basis of their
        # constraints, which is what a basis over all webs adds).
        lost = np.flatnonzero(~survives)
        combos = [np.array([i]) for i in np.flatnonzero(survives)]
        if lost.size:
            combos += [lost[np.flatnonzero(a)] for a in nullspace(C[lost].T)]

        self.detectors, self.observables = [], []
        self.detector_origins, self.observable_origins = [], []
        closed = self._close(sp_matmul(sp_rows(combos, len(webs)), V)) if combos else []
        for idx, web in zip(combos, closed):
            orig = tuple(origin[i] for i in idx)
            if any(f == "logical" for f, _ in orig):
                self.observables.append(web)
                self.observable_origins.append(orig)
            else:
                self.detectors.append(web)
                self.detector_origins.append(orig)

    # -- queries ------------------------------------------------------------

    @property
    def webs(self) -> list[PauliWeb]:
        return self.detectors + self.observables

    def webs_complete(self) -> bool:
        """Whether detectors and observables span the closed diagram's web space."""
        return len(self.webs) == len(self.web_space())

    def record(self, w: PauliWeb) -> set[Hashable]:
        """Outcome labels whose parity is ``mu(w)``, including fusion outcomes."""
        return _records(self, self.links, [w])[0]

    def records(self, webs: Sequence[PauliWeb]) -> list[set[Hashable]]:
        """``record`` of each web."""
        return _records(self, self.links, webs)

    def draw(self, web: PauliWeb | None = None, ax=None, title: str | None = None):
        return draw_diagram(self, self.positions, web, ax, title, self.worldlines,
                            _fusion_edges(self.links))
