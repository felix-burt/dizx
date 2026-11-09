"""ZX diagrams and Pauli webs.

A ZX diagram is a multigraph with a type map ``t : V -> {Z, X, H, P}`` and a
phase map on spiders. A Pauli web is an F2 vector ``(x | z)`` of length
``2|E|`` over the diagram's fixed edge ordering, stored sparsely (the
positions of its ones); a set of webs is a sparse matrix.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Hashable, Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
import scipy.sparse as sp

from dizx._f2 import (
    Echelon, SparseEchelon, nullspace, rank, rref, solve, sp_canonical, sp_matmul, sp_rank, sp_row,
    sp_rows, sp_solver,
)
from dizx.code import Code, Qubit

Node = Hashable
Edge = tuple[Node, Node, int]  # (u, v, key), as in nx.MultiGraph


class NodeType(Enum):
    Z = "Z"
    X = "X"
    H = "H"
    P = "P"


@dataclass(frozen=True)
class Outcome:
    """Phase ``m_c * pi`` set by the measurement outcome named ``label``."""
    label: Hashable


Phase = int | Outcome  # int in {0, 1}, i.e. a fixed phase 0 or pi

SPIDERS = (NodeType.Z, NodeType.X)
PAULI = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
PAULI_BITS = {p: xz for xz, p in PAULI.items()}


# ---------------------------------------------------------------------------
# ZXDiagram
# ---------------------------------------------------------------------------

class ZXDiagram:
    """A ZX diagram on a (frozen) multigraph.

    Attributes
    ----------
    graph : nx.MultiGraph
        Topology only; a frozen copy of the graph passed in.
    types : dict[Node, NodeType]
    phases : dict[Node, Phase]
        Defined on every Z/X node (missing spiders default to 0).
    inputs, outputs : tuple[Node, ...]
        Ordered ports; together they partition the P nodes.
    edges : tuple[Edge, ...]
        Fixed edge ordering defining the web vector layout.
    """

    def __init__(
        self,
        graph: nx.MultiGraph,
        types: Mapping[Node, NodeType | str],
        phases: Mapping[Node, Phase] | None = None,
        inputs: Iterable[Node] = (),
        outputs: Iterable[Node] = (),
    ) -> None:
        self.graph: nx.MultiGraph = nx.freeze(nx.MultiGraph(graph))
        self.types = {v: NodeType(t) for v, t in types.items()}
        phases = dict(phases or {})
        self.phases: dict[Node, Phase] = {
            v: phases.pop(v, 0) for v, t in self.types.items() if t in SPIDERS
        }
        if phases:
            raise ValueError(f"phases given on non-spider nodes: {sorted(map(str, phases))}")
        self.inputs = tuple(inputs)
        self.outputs = tuple(outputs)

        self.edges: tuple[Edge, ...] = tuple(self.graph.edges(keys=True))
        self._index: dict[Edge, int] = {}
        for i, (u, v, k) in enumerate(self.edges):
            self._index[(u, v, k)] = i
            self._index[(v, u, k)] = i
        self._constraints: tuple[sp.csr_matrix, list[Node]] | None = None
        self._constraints_int: sp.csr_matrix | None = None
        self._classical_legs: tuple[list[Node], np.ndarray] | None = None
        self.validate()

    # -- structure ----------------------------------------------------------

    def validate(self) -> None:
        """Check the conditions of the ZX diagram definition."""
        if set(self.types) != set(self.graph.nodes):
            raise ValueError("type map must cover exactly the graph's nodes")
        for v, t in self.types.items():
            d = self.graph.degree(v)
            if t is NodeType.H and d != 2:
                raise ValueError(f"H node {v!r} has degree {d}, expected 2")
            if t is NodeType.P and d != 1:
                raise ValueError(f"port {v!r} has degree {d}, expected 1")
        for v, a in self.phases.items():
            if not isinstance(a, Outcome) and a not in (0, 1):
                raise ValueError(f"phase of {v!r} must be 0, 1 or an Outcome, got {a!r}")
        ports = [v for v, t in self.types.items() if t is NodeType.P]
        io = self.inputs + self.outputs
        if len(io) != len(set(io)) or set(io) != set(ports):
            raise ValueError("inputs and outputs must partition the ports")

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def edge_index(self, e: Edge) -> int:
        """Index of edge ``(u, v, key)`` in either orientation."""
        return self._index[e]

    def legs(self, v: Node) -> list[int]:
        """Edge indices of E(v); a self-loop appears twice."""
        return [self._index[e] for e in self.graph.edges(v, keys=True)] + [
            self._index[e] for e in self.graph.edges(v, keys=True) if e[0] == e[1]
        ]

    def boundary_edges(self, side: str) -> list[int]:
        """Edge indices of B_in (``side='in'``) or B_out (``side='out'``), port order."""
        ports = {"in": self.inputs, "out": self.outputs}[side]
        return [self.legs(p)[0] for p in ports]

    def classical_legs(self) -> tuple[list[Node], np.ndarray]:
        """``classical_outputs`` and the edge index of each one's leg (cached)."""
        if self._classical_legs is None:
            outs = self.classical_outputs()
            self._classical_legs = outs, np.array([self.legs(c)[0] for c in outs], dtype=np.int64)
        return self._classical_legs

    def classical_outputs(self) -> list[Node]:
        """Degree-one spiders with an Outcome phase, same type as their neighbour."""
        out = []
        for v, a in self.phases.items():
            if not isinstance(a, Outcome) or self.graph.degree(v) != 1:
                continue
            (u,) = self.graph.neighbors(v)
            if self.types[u] is self.types[v]:
                out.append(v)
        return out

    # -- webs ---------------------------------------------------------------

    def web_constraints(self) -> tuple[sp.csr_matrix, list[Node]]:
        """F2 matrix ``A`` (sparse, uint8) with ``A w = 0`` iff ``w`` is a
        valid web.

        Also returns the node each row comes from. Columns are laid out as
        ``(x_0..x_{n-1} | z_0..z_{n-1})``. Cached, since diagrams are immutable.
        """
        if self._constraints is None:
            self._constraints = self.constraints_at(self.types)
        return self._constraints

    def constraints_at(self, nodes: Iterable[Node]) -> tuple[sp.csr_matrix, list[Node]]:
        """The rows of ``web_constraints`` that come from ``nodes``."""
        n = self.n_edges
        indptr, indices = [0], []
        owners: list[Node] = []

        def add(cols: Iterable[int], v: Node) -> None:
            odd = sorted(c for c, k in Counter(cols).items() if k % 2)
            if odd:
                indices.extend(odd)
                indptr.append(len(indices))
                owners.append(v)

        for v in nodes:
            t = self.types[v]
            legs = self.legs(v)
            if t in SPIDERS:
                # Z: x constant on legs, z even parity. X: swap x and z.
                const, parity = (0, n) if t is NodeType.Z else (n, 0)
                for e in legs[1:]:
                    add([const + legs[0], const + e], v)
                add([parity + e for e in legs], v)
            elif t is NodeType.H:
                e1, e2 = legs
                add([e1, n + e2], v)
                add([n + e1, e2], v)

        A = sp.csr_matrix((np.ones(len(indices), dtype=np.uint8), indices, indptr),
                          shape=(len(owners), 2 * n))
        return A, owners

    def _web_constraints_int(self) -> sp.csr_matrix:
        """``web_constraints`` as int32, for products with web matrices."""
        if self._constraints_int is None:
            self._constraints_int = self.web_constraints()[0].astype(np.int32)
        return self._constraints_int

    def violations(self, webs, nodes: Iterable[Node] | None = None) -> list[list[Node]]:
        """For each row of ``webs`` (a ``k x 2|E|`` matrix, dense or sparse),
        the nodes at which a web rule fails, checking only the rules at
        ``nodes`` if given."""
        if nodes is None:
            A, owners = self._web_constraints_int(), self.web_constraints()[1]
        else:
            A, owners = self.constraints_at(nodes)
            A = A.astype(np.int32)
        W = sp.csr_matrix(webs).astype(np.int32)
        bad = sp_canonical(A @ W.T).tocsc()
        out: list[list[Node]] = [[] for _ in range(W.shape[0])]
        for j in np.flatnonzero(np.diff(bad.indptr)):
            rows = bad.indices[bad.indptr[j]:bad.indptr[j + 1]]
            out[j] = list(dict.fromkeys(owners[r] for r in sorted(rows)))
        return out

    def web_space(self) -> list[PauliWeb]:
        """A basis of W(D). Arbitrary choice of basis; for checks, not construction."""
        A, _ = self.web_constraints()
        return [PauliWeb(self, w) for w in nullspace(A.toarray())]

    def detector_space(self) -> list[PauliWeb]:
        """A basis of the detecting-region subspace {w in W(D) : O_w = I -> I}."""
        A, _ = self.web_constraints()
        n = self.n_edges
        bnd = self.boundary_edges("in") + self.boundary_edges("out")
        fix = np.zeros((2 * len(bnd), 2 * n), dtype=np.uint8)
        for r, i in enumerate(bnd):
            fix[2 * r, i] = fix[2 * r + 1, n + i] = 1
        return [PauliWeb(self, w) for w in nullspace(np.vstack([A.toarray(), fix]))]


# ---------------------------------------------------------------------------
# PauliWeb
# ---------------------------------------------------------------------------

class WebKind(Enum):
    STABILISER = "stabiliser"        # P_in = I,  P_out != I
    CO_STABILISER = "co-stabiliser"  # P_in != I, P_out = I
    LOGICAL = "logical"              # P_in != I, P_out != I
    DETECTOR = "detector"            # P_in = I,  P_out = I


class PauliWeb:
    """An assignment ``E -> F2^2`` on a diagram, as the vector ``(x | z)``.

    Stored sparsely: ``idx`` holds the sorted positions of the vector's ones
    (``i`` for x on edge i, ``|E| + i`` for z); ``vec`` gives the dense vector.
    Construction does not require validity; use ``is_valid``.
    """

    __slots__ = ("diagram", "idx")

    def __init__(self, diagram: ZXDiagram, vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.uint8) % 2
        if vec.shape != (2 * diagram.n_edges,):
            raise ValueError(f"expected shape {(2 * diagram.n_edges,)}, got {vec.shape}")
        self.diagram = diagram
        self.idx = np.flatnonzero(vec).astype(np.int64)

    @classmethod
    def from_indices(cls, diagram: ZXDiagram, idx: np.ndarray) -> PauliWeb:
        """From the sorted, distinct positions of the vector's ones."""
        w = cls.__new__(cls)
        w.diagram = diagram
        w.idx = np.asarray(idx, dtype=np.int64)
        return w

    @classmethod
    def from_paulis(cls, diagram: ZXDiagram, paulis: Mapping[Edge, str]) -> PauliWeb:
        """Build from ``{edge: 'X' | 'Y' | 'Z' | 'I'}``; unlisted edges carry I."""
        n = diagram.n_edges
        idx = []
        for e, p in paulis.items():
            i = diagram.edge_index(e)
            x, z = PAULI_BITS[p]
            if x:
                idx.append(i)
            if z:
                idx.append(n + i)
        return cls.from_indices(diagram, np.unique(np.array(idx, dtype=np.int64)))

    @property
    def vec(self) -> np.ndarray:
        v = np.zeros(2 * self.diagram.n_edges, dtype=np.uint8)
        v[self.idx] = 1
        return v

    def _has(self, c: int) -> int:
        k = np.searchsorted(self.idx, c)
        return int(k < len(self.idx) and self.idx[k] == c)

    def bits(self, e: Edge | int) -> tuple[int, int]:
        """``(x, z)`` on edge e."""
        i = e if isinstance(e, (int, np.integer)) else self.diagram.edge_index(e)
        return self._has(i), self._has(self.diagram.n_edges + i)

    def pauli(self, e: Edge | int) -> str:
        return PAULI[self.bits(e)]

    def support(self) -> dict[Edge, str]:
        n, edges = self.diagram.n_edges, self.diagram.edges
        x = set(self.idx[self.idx < n].tolist())
        z = set((self.idx[self.idx >= n] - n).tolist())
        return {edges[i]: PAULI[int(i in x), int(i in z)] for i in sorted(x | z)}

    # -- validity -----------------------------------------------------------

    def violations(self) -> list[Node]:
        """Nodes at which a web rule fails (empty iff valid)."""
        return self.diagram.violations(web_matrix([self], self.diagram.n_edges))[0]

    def is_valid(self) -> bool:
        return not self.violations()

    # -- boundary -----------------------------------------------------------

    def boundary(self) -> tuple[str, str]:
        """``(P_in, P_out)`` as Pauli strings in port order."""
        d = self.diagram
        return (
            "".join(self.pauli(i) for i in d.boundary_edges("in")),
            "".join(self.pauli(i) for i in d.boundary_edges("out")),
        )

    def kind(self) -> WebKind:
        p_in, p_out = (set(p) - {"I"} for p in self.boundary())
        if p_in:
            return WebKind.LOGICAL if p_out else WebKind.CO_STABILISER
        return WebKind.STABILISER if p_out else WebKind.DETECTOR

    # -- outcomes -----------------------------------------------------------

    def outcomes(self) -> list[Node]:
        """C(w): classical outputs whose leg is highlighted."""
        outs, legs = self.diagram.classical_legs()
        if not outs:
            return []
        lit = np.isin(legs, self.idx) | np.isin(legs + self.diagram.n_edges, self.idx)
        return [outs[k] for k in np.flatnonzero(lit)]

    # -- algebra ------------------------------------------------------------

    def __add__(self, other: PauliWeb) -> PauliWeb:
        if other.diagram is not self.diagram:
            raise ValueError("cannot add webs on different diagrams")
        return PauliWeb.from_indices(self.diagram, np.setxor1d(self.idx, other.idx, assume_unique=True))

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, PauliWeb)
            and other.diagram is self.diagram
            and np.array_equal(self.idx, other.idx)
        )

    def __hash__(self) -> int:
        return hash((id(self.diagram), self.idx.tobytes()))

    def __repr__(self) -> str:
        return f"PauliWeb({self.support()})"


def web_matrix(webs: Sequence[PauliWeb], n_edges: int) -> sp.csr_matrix:
    """The webs as the rows of a ``k x 2 n_edges`` CSR matrix (uint8)."""
    return sp_rows([w.idx for w in webs], 2 * n_edges)


def webs_from_matrix(diagram: ZXDiagram, M) -> list[PauliWeb]:
    """The rows of ``M`` (over ``diagram``'s columns) as webs."""
    M = sp_canonical(M)
    if M.shape[1] != 2 * diagram.n_edges:
        raise ValueError(f"expected {2 * diagram.n_edges} columns, got {M.shape[1]}")
    return [PauliWeb.from_indices(diagram, sp_row(M, r)) for r in range(M.shape[0])]


def edge_transport(src: Sequence[int] | np.ndarray, dst: Sequence[int] | np.ndarray,
                   n_src: int, n_dst: int) -> sp.csr_matrix:
    """The ``2 n_src x 2 n_dst`` 0/1 matrix carrying the x and z columns of
    edge ``src[k]`` to those of edge ``dst[k]`` (``W @ P`` moves webs)."""
    src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
    return column_transport(np.concatenate([src, n_src + src]), np.concatenate([dst, n_dst + dst]),
                            2 * n_src, 2 * n_dst)


def column_transport(src: np.ndarray, dst: np.ndarray, n_src: int, n_dst: int) -> sp.csr_matrix:
    """The ``n_src x n_dst`` 0/1 matrix with a 1 at each ``(src[k], dst[k])``."""
    return sp.csr_matrix((np.ones(len(src), dtype=np.int32), (np.asarray(src), np.asarray(dst))),
                         shape=(n_src, n_dst))


def _port_cols(d: ZXDiagram, ports: Sequence[Node]) -> list[int]:
    """The x columns, then the z columns, of the ports' edges."""
    idx = [d.legs(p)[0] for p in ports]
    return idx + [d.n_edges + i for i in idx]


def port_bits(d: ZXDiagram, W, ports: Sequence[Node]) -> np.ndarray:
    """Dense ``(x | z)`` bits of the rows of web matrix ``W`` on ``ports``."""
    return port_cols_bits(W, _port_cols(d, ports))


# ---------------------------------------------------------------------------
# SEDiagram
# ---------------------------------------------------------------------------

WebLike = PauliWeb | Mapping[Edge, str] | np.ndarray


class SEDiagram(ZXDiagram):
    """A syndrome extraction diagram.

    A phase-free ZX diagram with no classical outputs, i.e. a codespace
    projector with no fixed realisation, together with a chosen generating set
    of detecting regions, stabilisers, co-stabilisers and logical operators.

    Webs may be given as ``PauliWeb`` (on any diagram with the same edges),
    ``{edge: pauli}`` maps, or dense vectors in this diagram's edge order. They
    are rebuilt on ``self`` and checked for validity, class and independence
    (``check="full"``), or not at all (``check="none"``, for callers that
    guarantee them, e.g. transformations, which check locally).

    ``positions`` optionally maps nodes to 3D coordinates for ``draw``.
    """

    def __init__(
        self,
        graph: nx.MultiGraph,
        types: Mapping[Node, NodeType | str],
        inputs: Iterable[Node] = (),
        outputs: Iterable[Node] = (),
        detectors: Iterable[WebLike] = (),
        stabilisers: Iterable[WebLike] = (),
        co_stabilisers: Iterable[WebLike] = (),
        logicals: Iterable[WebLike] = (),
        positions: Mapping[Node, tuple[float, float, float]] | None = None,
        check: str = "full",
        closed: Mapping[Node, Node] | None = None,
        carried: Iterable[int] = (),
    ) -> None:
        if check not in ("full", "none"):
            raise ValueError("check must be 'full' or 'none'")
        super().__init__(graph, types, None, inputs, outputs)
        self.positions = dict(positions or {})
        # Closed port -> the spider now ending its wire (see close_ports).
        self.closed: dict[Node, Node] = {p: v for p, v in (closed or {}).items() if v in self.types}
        self.detectors = self._adopt_all(detectors)
        self.stabilisers = self._adopt_all(stabilisers)
        self.co_stabilisers = self._adopt_all(co_stabilisers)
        self.logicals = self._adopt_all(logicals)
        # Logicals that are really stabilisers carried through (class S -> S,
        # from a partial round): not logical when combined (see _file).
        self.carried: list[int] = sorted(carried)
        if check == "full":
            self._check_webs()

    @classmethod
    def from_code(
        cls,
        code: Code,
        rounds: int,
        qubit_positions: Mapping[Qubit, tuple[float, float]] | None = None,
        first_round: int = 0,
        first: str = "Z",
        last: str = "X",
        z0: float | None = None,
    ) -> SEDiagram:
        """The SED of ``rounds`` alternating Z- and X-type layers of a CSS code,
        built as ``B ; B ; ... ; B`` from a single-round body ``B``.

        Layers sit one unit apart in the time coordinate, the first at height
        ``z0`` (default ``2 * first_round``, plus one if ``first="X"``), so
        phases built with consecutive ``z0`` stack without gaps.

        Rounds are numbered from ``first_round`` (in node labels and in the
        time coordinate), so SEDs of consecutive phases have distinct labels
        and can be composed directly.

        ``first`` and ``last`` are the types of the first and last layers.
        Each round is a Z layer then an X layer, so ``first="X"`` adds a round
        with only an X layer before the ``rounds`` full rounds, and
        ``last="Z"`` adds a round with only a Z layer after them. A partial
        round carries the stabilisers it does not measure straight through
        (filed with the logicals: their class is ``S -> S``).

        Nodes, webs and positions are as described in ``_from_code_unrolled``
        (which builds the same diagram directly).
        """
        if first not in ("Z", "X") or last not in ("Z", "X"):
            raise ValueError("first and last must be 'Z' or 'X'")
        plan = (["X"] if first == "X" else []) + ["ZX"] * rounds + (["Z"] if last == "Z" else [])
        if not plan:
            raise ValueError("no layers")
        bodies = {lv: cls._from_code_unrolled(code, 1, qubit_positions, levels=lv) for lv in set(plan)}
        sed = None
        z = 2.0 * first_round + (first == "X") if z0 is None else float(z0)
        for k, lv in enumerate(plan):
            r = first_round + k
            # A body's Z layer is at height 0 and its X layer at 1.
            dz = z - (1.0 if lv == "X" else 0.0)
            part = relabel(bodies[lv], lambda v, r=r: _shift_round(v, r), shift=(0.0, 0.0, dz))
            sed = part if sed is None else compose(sed, part, check="none")[0]
            z += len(lv)
        for p in sed.inputs + sed.outputs if sed.positions else ():
            # Ports one unit beyond their wire's end.
            (n,) = sed.graph.neighbors(p)
            zn = sed.positions[n][2]
            sed.positions[p] = (*sed.positions[p][:2], zn - 1.0 if p in sed.inputs else zn + 1.0)
        sed._check_webs()
        return sed

    @classmethod
    def _from_code_unrolled(
        cls,
        code: Code,
        rounds: int,
        qubit_positions: Mapping[Qubit, tuple[float, float]] | None = None,
        levels: str = "ZX",
    ) -> SEDiagram:
        """The SED of ``rounds`` alternating Z- and X-type layers of a CSS code,
        built directly (used for the body of ``from_code``, and as a reference).

        ``levels`` restricts every round to those layer types (e.g. ``"X"``);
        the stabilisers of a missing type are then carried straight through,
        as webs of class ``S -> S`` filed after the logicals.

        Each qubit ``q`` is a wire ``("in", q) -> ... -> ("out", q)``. In round
        ``r`` it passes through a Z data spider ``("data", r, "Z", q)`` joined
        to its Z-type generators, then an X data spider ``("data", r, "X", q)``
        joined to its X-type generators. Neighbouring data spiders of the same
        colour are fused (keeping the earliest label), so a qubit acted on by
        one generator type only has a single data spider. Generator ``i`` of
        type P in round ``r`` is a generator spider ``("gen", r, i)`` of the
        opposite colour.

        Webs, each ordered by generator ``i``:
          - detectors: generator ``i`` in rounds ``r - 1`` and ``r``, ordered
            by ``(r, i)`` for ``1 <= r < rounds``;
          - stabilisers: generator ``i`` in the last round, carried to the outputs;
          - co-stabilisers: generator ``i`` in the first round, carried to the inputs;
          - logicals: ``a_0, b_0, a_1, b_1, ...`` from ``code.logicals``,
            carried from inputs to outputs.

        Node positions for ``draw`` put qubit ``q`` at ``qubit_positions[q]``
        (default: ``q`` itself, if it is an ``(x, y)`` pair) with time upwards;
        generator spiders sit at ``code.stabiliser_positions`` if given, else
        at the centroid of their support.
        """
        if not code.is_css():
            raise ValueError("from_code supports CSS codes only")
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        other = {"X": "Z", "Z": "X"}
        kinds = [next(iter(s.values())) for s in code.stabilisers]

        types: dict[Node, str] = {}
        edges: list[tuple[Node, Node]] = []
        gen_legs: dict[Node, list[Node]] = {}  # data spider -> its generator spiders
        data: dict[tuple[int, str, Qubit], Node] = {}  # (r, P, q) -> data spider
        wires: dict[Qubit, list[Node]] = {q: [("in", q)] for q in code.qubits}
        has = {q: {p for s, p in zip(code.stabilisers, kinds) if q in s and p in levels}
               for q in code.qubits}
        present = [i for i, p in enumerate(kinds) if p in levels]

        for r in range(rounds):
            for q in code.qubits:
                for p in "ZX":
                    if p not in has[q]:
                        continue
                    v = wires[q][-1]
                    if types.get(v) != p:  # otherwise fuse into the previous spider
                        v = ("data", r, p, q)
                        types[v] = p
                        gen_legs[v] = []
                        wires[q].append(v)
                    data[r, p, q] = v
            for i, (s, p) in enumerate(zip(code.stabilisers, kinds)):
                if p not in levels:
                    continue
                g = ("gen", r, i)
                types[g] = other[p]
                for q in s:
                    edges.append((data[r, p, q], g))
                    gen_legs[data[r, p, q]].append(g)
        for q, wire in wires.items():
            wire.append(("out", q))
            types[("in", q)] = types[("out", q)] = "P"
            edges += zip(wire, wire[1:])

        graph = nx.MultiGraph()
        graph.add_nodes_from(types)
        graph.add_edges_from(edges)

        def toggle(web: dict[Edge, str], u: Node, v: Node, p: str) -> None:
            e = min((u, v, 0), (v, u, 0), key=repr)
            a, b = PAULI_BITS[web.get(e, "I")], PAULI_BITS[p]
            web[e] = PAULI[a[0] ^ b[0], a[1] ^ b[1]]

        def along(web: dict[Edge, str], q: Qubit, a: Node, b: Node, p: str) -> None:
            """Carry Pauli p along wire q between nodes a and b. Data spiders of
            p's colour (X spiders for Z, Z spiders for X) strictly between are
            highlighted on all their generator legs."""
            w = wires[q]
            i, j = sorted((w.index(a), w.index(b)))
            for u, v in zip(w[i:j], w[i + 1 : j + 1]):
                toggle(web, u, v, p)
            for u in w[i + 1 : j]:
                if types[u] == other[p]:
                    for g in gen_legs[u]:
                        toggle(web, u, g, p)

        def generator(web: dict[Edge, str], r: int, i: int) -> None:
            """Highlight all legs of generator spider (r, i) in its own colour."""
            p = kinds[i]
            for q in code.stabilisers[i]:
                toggle(web, data[r, p, q], ("gen", r, i), p)

        def occurrence(r: int, i: int, end: Node | None, r2: int | None = None):
            """Generator i in round r, carried to port ``end`` or to round r2."""
            web: dict[Edge, str] = {}
            p = kinds[i]
            generator(web, r, i)
            if r2 is not None:
                generator(web, r2, i)
            for q in code.stabilisers[i]:
                target = data[r2, p, q] if r2 is not None else (end, q)
                along(web, q, data[r, p, q], target, p)
            return web

        detectors = [occurrence(r - 1, i, None, r) for r in range(1, rounds) for i in present]
        stabilisers = [occurrence(rounds - 1, i, "out") for i in present]
        co_stabilisers = [occurrence(0, i, "in") for i in present]

        def through(op: Mapping[Qubit, str]) -> dict[Edge, str]:
            web: dict[Edge, str] = {}
            for q, c in op.items():
                for p in "XZ":
                    if c in (p, "Y"):
                        along(web, q, ("in", q), ("out", q), p)
            return web

        logicals = [through(op) for pair in code.logicals for op in pair]
        k = len(logicals)
        logicals += [through(s) for i, s in enumerate(code.stabilisers) if i not in present]

        positions = _code_layout(code, rounds, kinds, wires, qubit_positions)
        positions = {v: p for v, p in positions.items() if v in types}
        return cls(graph, types, [("in", q) for q in code.qubits],
                   [("out", q) for q in code.qubits],
                   detectors=detectors, stabilisers=stabilisers,
                   co_stabilisers=co_stabilisers, logicals=logicals,
                   positions=positions, carried=range(k, len(logicals)))

    def _adopt_all(self, webs) -> list[PauliWeb]:
        """A family given as webs, or as a sparse matrix (one web per row)."""
        if sp.issparse(webs):
            return webs_from_matrix(self, webs)
        return [self._adopt(w) for w in webs]

    def _adopt(self, w: WebLike) -> PauliWeb:
        if isinstance(w, PauliWeb):
            return w if w.diagram is self else PauliWeb.from_paulis(self, w.support())
        if isinstance(w, np.ndarray):
            return PauliWeb(self, w)
        return PauliWeb.from_paulis(self, w)

    def _families(self) -> list[tuple[WebKind, list[PauliWeb]]]:
        return [(WebKind.DETECTOR, self.detectors), (WebKind.STABILISER, self.stabilisers),
                (WebKind.CO_STABILISER, self.co_stabilisers), (WebKind.LOGICAL, self.logicals)]

    def _check_webs(self, nodes: Iterable[Node] | None = None, independence: bool = True) -> None:
        """Validity (at ``nodes`` only, if given), class and, optionally,
        independence of every web, in one matrix product each."""
        W = self._matrix(self.webs)
        bad = self.violations(W, nodes)
        has_in = port_bits(self, W, self.inputs).any(axis=1)
        has_out = port_bits(self, W, self.outputs).any(axis=1)
        kinds = {(False, False): WebKind.DETECTOR, (False, True): WebKind.STABILISER,
                 (True, False): WebKind.CO_STABILISER, (True, True): WebKind.LOGICAL}
        j = 0
        for kind, webs in self._families():
            for i in range(len(webs)):
                if bad[j]:
                    raise ValueError(f"{kind.value} {i} is invalid at nodes {bad[j]}")
                got = kinds[bool(has_in[j]), bool(has_out[j])]
                # A logical may have one side trivial (I -> L, see _file).
                if got is not kind and (kind is not WebKind.LOGICAL or got is WebKind.DETECTOR):
                    raise ValueError(f"{kind.value} {i} is a {got.value}")
                j += 1
        if independence and sp_rank(W) < W.shape[0]:
            raise ValueError("webs are not linearly independent")

    def _matrix(self, webs: list[PauliWeb]) -> sp.csr_matrix:
        """The webs as rows of a CSR matrix over this diagram's columns."""
        return web_matrix(webs, self.n_edges)

    @property
    def webs(self) -> list[PauliWeb]:
        return self.detectors + self.stabilisers + self.co_stabilisers + self.logicals

    def detectors_complete(self) -> bool:
        """Whether the given detectors span the full detecting-region subspace."""
        return len(self.detectors) == len(self.detector_space())

    def webs_complete(self) -> bool:
        """Whether the given webs span the full web space W(D)."""
        return len(self.webs) == len(self.web_space())

    def draw(self, web: PauliWeb | None = None, ax=None, title: str | None = None):
        """Draw the diagram in 3D at ``self.positions``; see ``draw_diagram``."""
        return draw_diagram(self, self.positions, web, ax, title)


def _shift_round(v: Node, r: int) -> Node:
    """Relabel a node of the single-round body of ``from_code`` into round r."""
    if isinstance(v, tuple) and v and v[0] in ("data", "gen"):
        return (v[0], v[1] + r, *v[2:])
    return v


def relabel(
    sed: SEDiagram,
    f: Callable[[Node], Node],
    shift: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> SEDiagram:
    """Rename every node by ``f`` (injective), carrying webs across, and shift positions."""
    mapping = {v: f(v) for v in sed.types}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("relabelling must be injective")
    graph = nx.relabel_nodes(nx.MultiGraph(sed.graph), mapping, copy=True)

    def web(w: PauliWeb) -> dict[Edge, str]:
        return {(mapping[u], mapping[v], k): p for (u, v, k), p in w.support().items()}

    positions = {mapping[v]: tuple(np.add(p, shift)) for v, p in sed.positions.items()}
    return SEDiagram(
        graph, {mapping[v]: t for v, t in sed.types.items()},
        [mapping[v] for v in sed.inputs], [mapping[v] for v in sed.outputs],
        detectors=[web(w) for w in sed.detectors],
        stabilisers=[web(w) for w in sed.stabilisers],
        co_stabilisers=[web(w) for w in sed.co_stabilisers],
        logicals=[web(w) for w in sed.logicals],
        positions=positions,
        closed={p: mapping[v] for p, v in sed.closed.items()},
        carried=sed.carried,
    )


def _default_join(d1: SEDiagram, d2: SEDiagram) -> list[tuple[Node, Node]]:
    def qubit(p: Node, tag: str):
        return p[1] if isinstance(p, tuple) and len(p) == 2 and p[0] == tag else None

    outs = {qubit(o, "out"): o for o in d1.outputs}
    ins = {qubit(i, "in"): i for i in d2.inputs}
    if None not in outs and None not in ins:
        return [(outs[q], ins[q]) for q in outs if q in ins]
    if len(d1.outputs) != len(d2.inputs):
        raise ValueError("d1 must have as many outputs as d2 has inputs")
    return list(zip(d1.outputs, d2.inputs))


def compose(
    d1: SEDiagram,
    d2: SEDiagram,
    fuse: bool = True,
    join: Sequence[tuple[Node, Node]] | None = None,
    check: str = "full",
) -> tuple[SEDiagram, dict[Node, Node]]:
    """Sequential composition ``d1 ; d2``: each pair ``(output of d1, input of
    d2)`` in ``join`` is joined and both ports are removed. Unjoined ports stay
    open: unjoined inputs of d2 become inputs and unjoined outputs of d1
    become outputs of the composite.

    By default, ports labelled ``("out", q)`` and ``("in", q)`` (as made by
    ``from_code``) are joined when they share the qubit ``q``; otherwise output
    ``k`` is joined to input ``k``.

    Apart from those ports, the two diagrams must have disjoint node labels.
    If ``fuse``, two same-colour spiders joined by a single new edge are fused
    into the d1 spider, as in the SED definition.

    Webs of the composite are pairs of webs of d1 and d2 that agree on every
    joined edge. A generating set is built from the given ones: webs of d1
    with ``P_out = I`` and of d2 with ``P_in = I`` carry over; a d1 web and a d2
    web with ``P_out = P_in`` are glued (a stabiliser and the matching
    co-stabiliser give a detector, two logicals give a logical). The rest are
    found by ``_complete_glue``: unglued logicals carried straight through,
    then webs glued to partners that consume their boundary, then general
    solutions, each reduced to a detector if its boundary duplicates others'.

    A web built from a logical of d1 or d2 is a logical (it may have one
    trivial side: a merge's outcome ``I -> Z Z`` is the merge's stabilisers
    carried out by the patches' logicals); any other is filed by which
    boundary is trivial (see ``_file``).

    ``check`` is passed to the composite's constructor (``"none"`` for
    intermediate steps whose result is checked later).

    Returns the composite and ``{d2 node: d1 node}`` for fused spiders.
    """
    if join is None:
        join = _default_join(d1, d2)
    join = list(join)
    outs, ins = {o for o, _ in join}, {i for _, i in join}
    if not outs <= set(d1.outputs) or not ins <= set(d2.inputs) or len(outs) != len(join) \
            or len(ins) != len(join):
        raise ValueError("join must pair distinct outputs of d1 with distinct inputs of d2")
    open_in = [i for i in d2.inputs if i not in ins]
    open_out = [o for o in d1.outputs if o not in outs]
    clash = (set(d1.types) - outs) & (set(d2.types) - ins)
    if clash:
        raise ValueError(f"d1 and d2 share node labels: {sorted(map(repr, clash))[:5]}")

    graph = nx.MultiGraph()
    types = {v: t for v, t in d1.types.items() if v not in outs}
    types.update({v: t for v, t in d2.types.items() if v not in ins})
    graph.add_nodes_from(types)
    positions = {v: p for v, p in d1.positions.items() if v not in outs}
    positions.update({v: p for v, p in d2.positions.items() if v not in ins})

    # (side, original edge) -> its edge in the composite (None if fused away),
    # and the reverse, keyed by an orientation-free form of the edge.
    where: dict[tuple[int, Edge], Edge | None] = {}
    owners: dict[tuple, list[tuple[int, Edge]]] = {}

    def canon(e: Edge) -> tuple:
        u, v, k = e
        return (u, v, k) if repr(u) <= repr(v) else (v, u, k)

    def place(origs: list[tuple[int, Edge]], e: Edge) -> None:
        for o in origs:
            where[o] = e
        owners.setdefault(canon(e), []).extend(origs)

    for side, d, ports in ((1, d1, outs), (2, d2, ins)):
        for e in d.edges:
            if not ({e[0], e[1]} & ports):
                graph.add_edge(e[0], e[1], key=e[2])
                place([(side, e)], e)

    def boundary_edge(d: SEDiagram, p: Node) -> tuple[Edge, Node]:
        (i,) = d.legs(p)
        e = d.edges[i]
        return e, (e[1] if e[0] == p else e[0])

    pairs = [(boundary_edge(d1, o), boundary_edge(d2, i)) for o, i in join]
    joins = Counter((u, v) for (_, u), (_, v) in pairs)
    rename: dict[Node, Node] = {}
    for (e1, u), (e2, v) in pairs:
        if (fuse and types[u] in SPIDERS and types[u] is types[v]
                and joins[u, v] == 1 and v not in rename):
            for _, x, k in list(graph.edges(v, keys=True)):
                origs = owners.pop(canon((v, x, k)))
                place(origs, (u, x, graph.add_edge(u, x)))
            graph.remove_node(v)
            del types[v]
            positions.pop(v, None)
            rename[v] = u
            where[(1, e1)] = where[(2, e2)] = None
        else:
            u, v = rename.get(u, u), rename.get(v, v)
            place([(1, e1), (2, e2)], (u, v, graph.add_edge(u, v)))

    # Solve for combinations of d1 and d2 webs agreeing on the joined edges.
    # A combination is a sorted array of web indices: d1's, then n1 + d2's.
    V1, V2 = d1._matrix(d1.webs), d2._matrix(d2.webs)
    n1, n2 = V1.shape[0], V2.shape[0]
    G = n1 + n2
    out1 = port_bits(d1, V1, [o for o, _ in join])
    in2 = port_bits(d2, V2, [i for _, i in join])

    combos = [np.array([i]) for i in range(n1) if not out1[i].any()]
    combos += [np.array([n1 + j]) for j in range(n2) if not in2[j].any()]
    by_in: dict[bytes, list[int]] = {}
    for j, b in enumerate(in2):
        if b.any():
            by_in.setdefault(b.tobytes(), []).append(j)
    glued1, glued2 = set(), set()
    for i, b in enumerate(out1):
        if b.any() and by_in.get(b.tobytes()):
            j = by_in[b.tobytes()].pop(0)
            combos.append(np.array([i, n1 + j]))
            glued1.add(i)
            glued2.add(j)
    M = np.vstack([out1, in2])
    dim = G - rank(M.T)
    if len(combos) < dim:
        combos = _complete_glue(d1, d2, join, open_in, open_out, V1, V2, out1, in2,
                                combos, glued1, glued2, dim)

    inputs, outputs = list(d1.inputs) + open_in, open_out + list(d2.outputs)
    logical, carried = _logical_masks([(d1, 0), (d2, n1)], G)
    # Every combination at once, moved into the composite's edge order. On a
    # joined edge both sides agree, so the two parts are combined by OR.
    sed = SEDiagram(
        graph, types, inputs, outputs, positions=positions, check="none",
        closed={**d1.closed, **{p: rename.get(v, v) for p, v in d2.closed.items()}},
    )
    cindex = sed._index
    nE = sed.n_edges
    C = sp_rows(combos, G)
    W1, W2 = sp_matmul(C[:, :n1], V1), sp_matmul(C[:, n1:], V2)
    vecs = sp.csr_matrix((len(combos), 2 * nE), dtype=np.int32)
    for side, d, W in ((1, d1, W1), (2, d2, W2)):
        pairs_ = [(d.edge_index(e), cindex[new]) for (sd, e), new in where.items()
                  if sd == side and new is not None]
        if pairs_:
            src, dst = zip(*pairs_)
            vecs = vecs + W.astype(np.int32) @ edge_transport(src, dst, d.n_edges, nE)
    vecs = sp.csr_matrix(vecs)
    vecs.data = (vecs.data > 0).astype(np.int32)
    vecs = sp_canonical(vecs)
    has_in = port_bits(d1, W1, d1.inputs).any(axis=1) | port_bits(d2, W2, open_in).any(axis=1)
    has_out = port_bits(d1, W1, open_out).any(axis=1) | port_bits(d2, W2, d2.outputs).any(axis=1)
    from_logical = np.asarray(C.astype(np.int32) @ logical.astype(np.int32)).ravel() > 0
    families, still_carried = _sort_families(has_in, has_out, from_logical)
    _fill(sed, vecs, families, still_carried, check)
    return sed, rename


def _fill(sed: SEDiagram, vecs, families: dict[WebKind, list[int]], carried: list[int],
          check: str) -> None:
    """Give a freshly built SED (made with ``check="none"`` and no webs) the
    rows of ``vecs`` as its webs, by family, then check them as ``check``."""
    sed.detectors = webs_from_matrix(sed, vecs[families[WebKind.DETECTOR]])
    sed.stabilisers = webs_from_matrix(sed, vecs[families[WebKind.STABILISER]])
    sed.co_stabilisers = webs_from_matrix(sed, vecs[families[WebKind.CO_STABILISER]])
    sed.logicals = webs_from_matrix(sed, vecs[families[WebKind.LOGICAL]])
    sed.carried = sorted(carried)
    if check == "full":
        sed._check_webs()


def _sort_families(has_in: np.ndarray, has_out: np.ndarray, from_logical: np.ndarray
                   ) -> tuple[dict[WebKind, list[int]], list[int]]:
    """Each combined web's family (``_file``), as row lists per family, and
    the positions within the logicals of the carried stabilisers."""
    families: dict[WebKind, list[int]] = {k: [] for k in WebKind}
    still_carried: list[int] = []
    for j in range(len(has_in)):
        kind = _file(bool(has_in[j]), bool(has_out[j]), bool(from_logical[j]))
        if kind is WebKind.LOGICAL and not from_logical[j]:
            still_carried.append(len(families[kind]))
        families[kind].append(j)
    return families, still_carried


def _straight_web(d: SEDiagram, bits: Mapping[Node, tuple[int, int]]) -> np.ndarray | None:
    """The web of ``d`` with the given (x, z) bits on the given ports, I on
    every other port, confined to the wires of the qubits those bits are on
    (their data spiders and ports, labelled ``(..., q)``) and the legs leaving
    them: a straight sheet, or None if there is none."""
    qubits = {p[1] for p, (x, z) in bits.items() if x or z}
    wire = {v for v in d.types if isinstance(v, tuple) and len(v) >= 2 and v[-1] in qubits
            and v[0] in ("data", "in", "out")}
    cols = sorted({i for v in wire for i in d.legs(v)})
    if not cols:
        return None
    n = d.n_edges
    A, _ = d.web_constraints()
    sub = A[:, cols + [n + i for i in cols]]
    sub = sub[np.flatnonzero(sub.getnnz(axis=1))].toarray()
    where = {i: k for k, i in enumerate(cols)}
    fix, target = [], []
    for p in d.inputs + d.outputs:
        (i,) = d.legs(p)
        if i in where:
            x, z = bits.get(p, (0, 0))
            for comp, val in ((where[i], x), (len(cols) + where[i], z)):
                r = np.zeros(2 * len(cols), dtype=np.uint8)
                r[comp] = 1
                fix.append(r)
                target.append(val)
    M = np.vstack([sub] + fix)
    t = np.concatenate([np.zeros(len(sub), np.uint8), np.array(target, np.uint8)])
    w = solve(M.T, t)
    if w is None:
        return None
    vec = np.zeros(2 * n, dtype=np.uint8)
    vec[cols], vec[[n + i for i in cols]] = w[:len(cols)], w[len(cols):]
    return vec


def _logical_masks(parts: Sequence[tuple[SEDiagram, int]], G: int) -> tuple[np.ndarray, np.ndarray]:
    """Over the concatenated webs of ``parts`` (each at its offset): which are
    genuine logicals, and which are carried stabilisers."""
    logical, carried = np.zeros(G, dtype=bool), np.zeros(G, dtype=bool)
    for d, off in parts:
        start = off + len(d.webs) - len(d.logicals)
        logical[start:start + len(d.logicals)] = True
        idx = [start + i for i in d.carried]
        logical[idx], carried[idx] = False, True
    return logical, carried


def _file(p_in: bool, p_out: bool, from_logical: bool) -> WebKind:
    """Family of a combined web: a detector if both boundaries are trivial,
    a logical if built from a genuine logical (so ``I -> Z Z``, made from a
    merge's stabilisers and the patches' logicals, is one), otherwise by
    which boundary is trivial (``S -> S`` is filed with the logicals, as a
    carried stabiliser)."""
    if not (p_in or p_out):
        return WebKind.DETECTOR
    if from_logical or (p_in and p_out):
        return WebKind.LOGICAL
    return WebKind.CO_STABILISER if p_in else WebKind.STABILISER


def _complete_glue(d1, d2, join, open_in, open_out, V1, V2, out1, in2,
                   combos, glued1, glued2, dim) -> list[np.ndarray]:
    """Complete ``combos`` (webs passed through or glued exactly, as sorted
    arrays of web indices) to a basis of the composite's webs, ``dim`` of
    them, by these steps in turn:

    1. carry each unglued web straight through, logicals first: a d1 web
       ``A -> B`` continues as the straight sheet ``B -> B`` across d2 (on
       the same qubits' wires only, ``_straight_web``), and a d2 web
       ``B -> C`` likewise back across d1 (so a merge's outcome leaves its
       lid along the inner boundaries);
    2. glue each other unglued web to a combination of the other side's
       webs that consume its boundary (non-trivial on the joined ports);
    3. general solutions.

    A candidate whose boundary is already a sum of the chosen webs'
    boundaries is reduced by them to a detector, so no web is a duplicate of
    others up to detectors (e.g. ``Z Z -> I`` beside ``I -> Z Z``, ``Z I -> Z I``
    and ``I Z -> I Z``)."""
    n1, n2 = V1.shape[0], V2.shape[0]
    G = n1 + n2
    first_logical1 = n1 - len(d1.logicals)
    first_logical2 = n2 - len(d2.logicals)
    zero1 = lambda ports: np.zeros((n1, 2 * len(ports)), dtype=np.uint8)
    zero2 = lambda ports: np.zeros((n2, 2 * len(ports)), dtype=np.uint8)
    in1, out2 = port_bits(d1, V1, d1.inputs), port_bits(d2, V2, d2.outputs)
    oin2, oout1 = port_bits(d2, V2, open_in), port_bits(d1, V1, open_out)
    # Composite boundary of each web: inputs (d1's, then d2's open ones), then
    # outputs (d1's open ones, then d2's).
    Bnd = np.vstack([np.hstack([in1, zero1(open_in), oout1, zero1(d2.outputs)]),
                     np.hstack([zero2(d1.inputs), oin2, zero2(open_out), out2])]).astype(np.uint8)

    def boundary(a: np.ndarray) -> np.ndarray:
        return np.bitwise_xor.reduce(Bnd[a], axis=0) if len(a) else np.zeros(Bnd.shape[1], np.uint8)

    def dense(a: np.ndarray) -> np.ndarray:
        v = np.zeros(G, dtype=np.uint8)
        v[a] = 1
        return v

    span = SparseEchelon()
    for a in combos:
        span.add(a)
    bnd = Echelon()
    for a in combos:
        b = boundary(a)
        if b.any():
            bnd.add(b, dense(a))
    combos = list(combos)

    def add(a: np.ndarray) -> None:
        b = boundary(a)
        if b.any():
            rest, tag = bnd.reduce(b, dense(a))
            if not rest.any():
                a, b = np.flatnonzero(tag), rest
        if span.add(a):
            combos.append(a)
            if b.any():
                bnd.add(b, dense(a))

    def qubit(p: Node):
        return p[1] if isinstance(p, tuple) and len(p) == 2 and p[0] in ("in", "out") else None

    def moved(b: np.ndarray, src: Sequence[Node], dst: Sequence[Node]) -> np.ndarray | None:
        """Bits ``b`` on ports ``src`` moved to the ports of ``dst`` on the same qubits."""
        where = {qubit(p): k for k, p in enumerate(dst)}
        t = np.zeros(2 * len(dst), dtype=np.uint8)
        for k, p in enumerate(src):
            if b[k] or b[len(src) + k]:
                if qubit(p) not in where:
                    return None
                t[where[qubit(p)]], t[len(dst) + where[qubit(p)]] = b[k], b[len(src) + k]
        return t

    joined_out, joined_in = [o for o, _ in join], [i for _, i in join]
    solvers: dict[int, Callable] = {}

    def carry(side: int, d: SEDiagram, V, near: Sequence[Node], b: np.ndarray,
              far: Sequence[Node]) -> np.ndarray | None:
        """Combination (web indices) of d's webs ``V`` that is the straight
        carry of ``b`` (bits on d's ports ``near``) across d to its ports ``far``."""
        t = moved(b, near, far)
        if t is None:
            return None
        bits = {p: (b[k], b[len(near) + k]) for k, p in enumerate(near)}
        bits.update({p: (t[k], t[len(far) + k]) for k, p in enumerate(far)})
        w = _straight_web(d, bits)
        if w is None:
            return None
        if side not in solvers:
            solvers[side] = sp_solver(V)
        return solvers[side](np.flatnonzero(w))

    # 1. Carry unglued webs straight through, logicals first.
    for i in [*range(first_logical1, n1), *range(first_logical1)]:
        if i in glued1 or not out1[i].any():
            continue
        y = carry(2, d2, V2, joined_in, out1[i], d2.outputs)
        if y is not None:
            add(np.concatenate([[i], n1 + y]))
    for j in [*range(first_logical2, n2), *range(first_logical2)]:
        if j in glued2 or not in2[j].any():
            continue
        x = carry(1, d1, V1, joined_out, in2[j], d1.inputs)
        if x is not None:
            add(np.concatenate([x, [n1 + j]]))
    # 2. Glue the rest to webs that consume their boundary.
    if len(combos) < dim:
        live2 = np.array([j for j in range(n2) if in2[j].any()], dtype=np.int64)
        for i in range(n1):
            if out1[i].any() and i not in glued1:
                y = solve(in2[live2], out1[i])
                if y is not None:
                    add(np.concatenate([[i], n1 + live2[np.flatnonzero(y)]]))
    if len(combos) < dim:
        live1 = np.array([i for i in range(n1) if out1[i].any()], dtype=np.int64)
        for j in range(n2):
            if in2[j].any() and j not in glued2:
                x = solve(out1[live1], in2[j])
                if x is not None:
                    add(np.concatenate([live1[np.flatnonzero(x)], [n1 + j]]))
    # 3. Anything left. Webs off the joined ports are already in, as units:
    # only combinations of the others can be new.
    if len(combos) < dim:
        M = np.vstack([out1, in2])
        live = np.flatnonzero(M.any(axis=1))
        for a in nullspace(M[live].T):
            add(live[np.flatnonzero(a)])
    return combos


_CLOSING = {"X": NodeType.Z, "Z": NodeType.X}  # spider colour preparing / measuring a basis


def _complete(pairs: list[np.ndarray], C: np.ndarray) -> list[np.ndarray]:
    """Combinations (sorted web indices) of the webs failing a constraint
    (nonzero rows of C) that, with the webs passing every constraint, span
    {a : a C = 0}: the first maximal independent subset of ``pairs`` (each
    a combination of failing webs) followed by a nullspace basis."""
    live = np.flatnonzero(C.any(axis=1))
    if not live.size:
        return []
    pos = {int(k): r for r, k in enumerate(live)}
    P = np.zeros((len(pairs), len(live)), dtype=np.uint8)
    for r, a in enumerate(pairs):
        P[r, [pos[int(k)] for k in a]] = 1
    cand = np.vstack([P, nullspace(C[live].T)])
    if not len(cand):
        return []
    _, keep = rref(cand.T)
    return [live[np.flatnonzero(cand[k])] for k in keep]



def close_ports(sed: SEDiagram, bases: Mapping[Node, str]) -> SEDiagram:
    """Remove ports: each port ``p`` in ``bases`` is prepared (an input) or
    measured (an output) in basis ``bases[p]``, so its wire must end at a
    spider of colour c = Z for basis X, X for basis Z.

    If p's neighbour is already of colour c, the port is simply removed (the
    neighbour now ends the wire). Otherwise p becomes an explicit degree-one
    spider of colour c, ``("init", p)`` for an input or ``("proj", p)`` for an
    output, at p's position. ``closed`` records the spider now ending each
    closed wire.

    A web survives if it is free of the component c forbids on p's edge (Z
    for colour Z, X for colour X). A generating set of the new diagram's webs
    is built from the given ones: survivors, then pairs whose forbidden parts
    cancel, then any remaining combinations; each is filed by its new class.
    """
    for p, b in bases.items():
        if p not in sed.inputs + sed.outputs:
            raise ValueError(f"{p!r} is not a port")
        if b not in _CLOSING:
            raise ValueError("bases must be 'Z' or 'X'")
    graph = nx.MultiGraph(sed.graph)
    types = dict(sed.types)
    positions = dict(sed.positions)
    closed = dict(sed.closed)
    rename: dict[Node, Node] = {}       # converted port -> its spider
    removed: set[Node] = set()
    forbid: list[tuple[int, int]] = []  # (edge index, 0 for x / 1 for z) per closed port
    for p, b in bases.items():
        c = _CLOSING[b]
        (i,) = sed.legs(p)
        u, v, _ = sed.edges[i]
        n = v if u == p else u
        if sed.types[n] not in SPIDERS:
            raise ValueError(f"port {p!r} must be attached to a spider")
        forbid.append((i, 1 if c is NodeType.Z else 0))
        if sed.types[n] is c:
            graph.remove_node(p)
            del types[p]
            positions.pop(p, None)
            removed.add(p)
            closed[p] = n
        else:
            s = ("init", p) if p in sed.inputs else ("proj", p)
            del types[p]
            types[s] = c
            if p in positions:
                positions[s] = positions.pop(p)
            rename[p] = s
            closed[p] = s

    if rename:
        graph = nx.relabel_nodes(graph, rename)
    webs = sed.webs
    V = sed._matrix(webs)
    n = sed.n_edges
    G = len(webs)
    cols = [i + n * comp for i, comp in forbid]
    C = port_cols_bits(V, cols)

    combos = [np.array([k]) for k in range(G) if not C[k].any()]
    by_bits: dict[bytes, list[int]] = {}
    for k in range(G):
        if C[k].any():
            by_bits.setdefault(C[k].tobytes(), []).append(k)
    pairs = [np.array([a, b]) for ks in by_bits.values() for a, b in zip(ks[::2], ks[1::2])]
    combos += _complete(pairs, C)

    inputs = [rename.get(p, p) for p in sed.inputs if p not in bases]
    outputs = [rename.get(p, p) for p in sed.outputs if p not in bases]
    logical, _ = _logical_masks([(sed, 0)], G)
    A = sp_rows(combos, G)
    W = sp_matmul(A, V)
    # The webs moved into the new diagram's edge order.
    new = SEDiagram(graph, types, inputs, outputs, positions=positions, closed=closed, check="none")
    cindex = new._index
    src, dst = [], []
    for i, (x, y, k) in enumerate(sed.edges):
        if x not in removed and y not in removed:
            src.append(i)
            dst.append(cindex[(rename.get(x, x), rename.get(y, y), k)])
    vecs = sp_matmul(W, edge_transport(src, dst, n, new.n_edges))
    has_in = port_bits(sed, W, [p for p in sed.inputs if p not in bases]).any(axis=1)
    has_out = port_bits(sed, W, [p for p in sed.outputs if p not in bases]).any(axis=1)
    from_logical = np.asarray(A.astype(np.int32) @ logical.astype(np.int32)).ravel() > 0
    families, still_carried = _sort_families(has_in, has_out, from_logical)
    _fill(new, vecs, families, still_carried, "full")
    return new


def port_cols_bits(W, cols: Sequence[int]) -> np.ndarray:
    """Dense bits of the rows of web matrix ``W`` on columns ``cols``."""
    if not len(cols):
        return np.zeros((W.shape[0], 0), dtype=np.uint8)
    return (sp.csr_matrix(W)[:, list(cols)].toarray() % 2).astype(np.uint8)


def draw_diagram(
    diagram: ZXDiagram,
    positions: Mapping[Node, Sequence[float]],
    web: PauliWeb | None = None,
    ax=None,
    title: str | None = None,
    worldlines: Iterable[Sequence[Node]] = (),
    bold: Iterable[Edge] = (),
):
    """Draw a diagram in 3D, optionally highlighting a web and world-lines.

    Spiders are green (Z) and red (X), ports black, and classical outputs
    small. A Hadamard edge (through an H node) is a dashed blue line. Web
    edges are green (X), red (Z) or purple (Y), dashed on Hadamard edges.
    World-line steps are black arrows. Edges in ``bold`` (e.g. fusions) are
    drawn thicker.
    """
    import matplotlib.pyplot as plt

    if not positions:
        raise ValueError("diagram has no positions to draw with")
    if ax is None:
        ax = plt.figure(figsize=(7, 7)).add_subplot(projection="3d")
    colour = {"X": "tab:green", "Z": "tab:red", "Y": "tab:purple"}
    thick = {diagram.edge_index(e) for e in bold}
    seen: dict[frozenset, int] = {}
    for i, (u, v, _) in enumerate(diagram.edges):
        # The k-th of several parallel edges bows out, alternating sides.
        k = seen[frozenset((u, v))] = seen.get(frozenset((u, v)), -1) + 1
        pu, pv = np.asarray(positions[u], float), np.asarray(positions[v], float)
        if k and u != v:
            d = pv - pu
            n = np.cross(d, [0, 0, 1] if abs(d[2]) < 0.9 * np.linalg.norm(d) else [1, 0, 0])
            n *= 0.15 * ((k + 1) // 2) * (1 if k % 2 else -1) * np.linalg.norm(d) / np.linalg.norm(n)
            xs, ys, zs = zip(pu, (pu + pv) / 2 + n, pv)
        else:
            xs, ys, zs = zip(pu, pv)
        p = web.pauli(i) if web is not None else "I"
        hadamard = NodeType.H in (diagram.types[u], diagram.types[v])
        style = "--" if hadamard else "-"
        if p == "I":
            ax.plot(xs, ys, zs, color="tab:blue" if hadamard else "0.2" if i in thick else "0.7",
                    ls=style, lw=3 if i in thick else 1.2 if hadamard else 0.8)
        else:
            ax.plot(xs, ys, zs, color=colour[p], ls=style, lw=5 if i in thick else 3)
    for path in worldlines:
        for u, v in zip(path, path[1:]):
            a, b = np.asarray(positions[u], float), np.asarray(positions[v], float)
            ax.quiver(*a, *(b - a), color="black", lw=1, arrow_length_ratio=0.25)
    node_colour = {NodeType.Z: "tab:green", NodeType.X: "tab:red", NodeType.P: "black"}
    classical = set(diagram.classical_outputs())
    for t, c in node_colour.items():
        for small in (False, True):
            vs = [v for v, tv in diagram.types.items() if tv is t and (v in classical) == small]
            if vs:
                xs, ys, zs = zip(*(positions[v] for v in vs))
                size = 8 if small else 12 if t is NodeType.P else 30
                ax.scatter(xs, ys, zs, color=c, s=size, depthshade=False,
                           edgecolors="black", linewidths=0.4)
    # Equal scale on all axes (3D axes default to a cube, which stretches
    # plaquettes): box sides proportional to the data ranges.
    spans = [max(hi - lo, 1e-9) for lo, hi in (ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d())]
    ax.set_box_aspect(spans)
    ax.set_axis_off()
    if title is None and web is not None:
        p_in, p_out = web.boundary()
        title = f"{web.kind().value}: {p_in or '-'} -> {p_out or '-'}"
    if title:
        ax.set_title(title)
    return ax


def _code_layout(
    code: Code,
    rounds: int,
    kinds: list[str],
    wires: Mapping[Qubit, list[Node]],
    qubit_positions: Mapping[Qubit, tuple[float, float]] | None,
) -> dict[Node, tuple[float, float, float]]:
    """3D positions for the nodes of ``SEDiagram.from_code``; time runs along z."""
    if qubit_positions is None:
        if not all(isinstance(q, tuple) and len(q) == 2 for q in code.qubits):
            return {}
        qubit_positions = {q: q for q in code.qubits}
    xy = {q: np.asarray(qubit_positions[q], dtype=float) for q in code.qubits}
    level = {"Z": 0.0, "X": 1.0}  # height within a round

    pos: dict[Node, tuple[float, float, float]] = {}
    for q, wire in wires.items():
        for v in wire:
            if v[0] == "in":
                z = -1.0
            elif v[0] == "out":
                z = 2.0 * rounds
            else:
                z = 2.0 * v[1] + level[v[2]]
            pos[v] = (*xy[q], z)
    for i, (s, p) in enumerate(zip(code.stabilisers, kinds)):
        if code.stabiliser_positions is not None:
            c = np.asarray(code.stabiliser_positions[i], dtype=float)
        else:
            c = np.mean([xy[q] for q in s], axis=0)
        for r in range(rounds):
            pos[("gen", r, i)] = (*c, 2.0 * r + level[p])
    return pos
