"""Two-patch lattice surgery, built by composing memory, merge and split SEDs.

The patches are ``d x d`` rotated surface codes at origins ``(0, 0)`` and
``(d + 1, 0)``, with a seam column at ``x = d``; the merged patch covers both
and the seam. Side by side in the default orientation the merge measures
``XX``; with both patches rotated by 90 degrees it measures ``ZZ``.

A ``ZZ`` merge prepares and measures the seam in X, an ``XX`` merge in Z.
The seam has no ports: its wires start and end at spiders whose colour fixes
that basis (Z for X, X for Z; see ``close_ports``). By default the merge
starts and ends with a layer of the joint's type (Z X ... X Z for ``ZZ``,
X Z ... Z X for ``XX``: one layer more than the merged rounds), so the seam's
first and last spiders already have the right colour. Otherwise
(``matching=False``, the merge is Z X ... Z X) an explicit ``("init", p)`` or
``("proj", p)`` spider closes the seam end that does not.

To realise the merge across two units joined by Bell pairs, an **interface**
assigns every position (x, y) to a unit; ``Surgery.split_seam`` splits each
generator touching another unit's qubits, and the halves are linked. The
seam qubits' units are the free choice: ``straight_interface`` puts them all
on the right (2-2 splits); ``zigzag_interface`` alternates them along the seam
(3-1 splits, arXiv:2504.08891 App. B, arXiv:2503.04968 Fig. 2(c)).

``surgery_hybrid_linear`` instead clusterises the merge region and splits its
graph state into linear resource states joined by fusions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Hashable

from dizx.code import Code, Qubit, disjoint_union, rotated_surface_code
from dizx.transformations import SEDEditor, loop_boundary, prepare_cluster_region, split_by_side
from dizx.realisation import LinkKind, QECExperiment, RealisedSED
from dizx.zx_diagram import Node, NodeType, SEDiagram, close_ports, compose

SEAM_BASIS = {"ZZ": "X", "XX": "Z"}

Interface = Callable[[float, float], Hashable]
"""Unit of the qubit or generator at (x, y)."""


def straight_interface(d: int) -> Interface:
    """Left patch | seam and right patch."""
    return lambda x, y: "left" if x < d else "right"


def zigzag_interface(d: int, phase: int = 1) -> Interface:
    """Seam qubits alternate units along the seam, so each weight-4 generator
    beside it has one seam qubit on each side. The seam qubit at y is on the
    right iff ``y + phase`` is even.

    Which phase keeps full distance depends on the links and the schedule;
    at d = 3 with ``RotatedSurfaceCodeSchedule``, in both sectors of both
    joints:

    ==============  ===============  ===========
    links           not alternating  alternating
    ==============  ===============  ===========
    ``"bell"``      phase 1          neither
    ``"fusion"``    phase 0          neither
    ==============  ===============  ===========

    ``phase=0`` is the rule of the previous pipeline (``_zigzag_seam_side``)."""
    def side(x, y):
        if x == d:
            return "right" if (round(y) + phase) % 2 == 0 else "left"
        return "left" if x < d else "right"
    return side


@dataclass
class Surgery:
    """A two-patch surgery SED with the codes it was built from."""
    sed: SEDiagram
    joint: str                # "ZZ" or "XX"
    patches: tuple[Code, Code]
    merged: Code
    seam: list[Qubit]
    rounds: tuple[int, int, int]
    merge_rounds: range       # round labels of the merge phase

    def seam_nodes(self, sed: SEDiagram | None = None) -> set[Node]:
        """Spiders on the seam's wires (data spiders and any init/proj spiders)."""
        sed = sed or self.sed
        seam = set(self.seam)
        return {v for v in sed.types if isinstance(v, tuple) and (
            (v[0] == "data" and v[-1] in seam) or v[0] in ("init", "proj"))}

    def merge_region(self) -> set[Node]:
        """The seam's spiders and the merge-round generators touching them."""
        seam = self.seam_nodes()
        return seam | {g for v in seam for g in self.sed.graph.neighbors(v)
                       if g[0] == "gen" and g[1] in self.merge_rounds}

    def split_seam(self, interface: Interface) -> SEDiagram:
        """The SED with every generator split along ``interface``; each half
        only touches qubits of its own unit."""
        pos = self.sed.positions
        gens = [v for v in self.sed.types if isinstance(v, tuple) and v[0] == "gen"]
        return split_by_side(self.sed, lambda v: interface(*pos[v][:2]), gens)

    def split_seam_linear(self, sed: SEDiagram) -> tuple[SEDiagram, dict[frozenset, LinkKind]]:
        """Split each seam spider of ``sed`` (graph-like, from
        ``prepare_cluster_region``) into a left half, which keeps its own label,
        and a right half ``("half", v)``, joined by a fusion.

        Each half takes the generators on its side of the seam. The wire legs
        alternate up the seam: at the k-th merge layer the left half takes the
        leg up if k is even and the leg down if k is odd, so the left halves
        pair layers (0, 1), (2, 3), ... and the right halves (1, 2), (3, 4), ...
        A spider with no legs for one half is not split.

        Returns the new SED and the fusions, for ``RealisedSED.hybrid``.
        """
        x_seam = self.seam[0][0]
        wires: dict[Qubit, list[Node]] = {}
        for v in self.seam_nodes(sed):
            if v[0] == "data":
                wires.setdefault(v[-1], []).append(v)
        fusions: dict[frozenset, LinkKind] = {}
        ed = SEDEditor(sed)
        for wire in wires.values():
            wire.sort(key=lambda v: ed.positions[v][2])
            for k, v in enumerate(wire):
                z = ed.positions[v][2]
                left = []
                for e in ed.graph.edges(v, keys=True):
                    u = _through_h(ed, v, e[1])
                    x, _, zu = ed.positions[u]
                    if u[0] == "gen" and x < x_seam or u[0] != "gen" and (zu > z) == (k % 2 == 0):
                        left.append(e)
                if 0 < len(left) < ed.graph.degree(v):
                    half = ("half", v)
                    ed.split(v, left, name=half, direction=(1, 0, 0))
                    fusions[frozenset((v, half))] = "fusion"
        return ed.freeze(), fusions

    def cut_time_edges(self, sed: SEDiagram, S: set[Node], period: int,
                       ) -> tuple[SEDiagram, dict[frozenset, LinkKind]]:
        """Put a fusion on every ``period``-th time edge of the region S.

        A time edge joins two seam spiders of S on one qubit at different
        heights through one H node (a wire step of a resource chain). Ranking
        the edges' qubit positions ``P`` and lower heights ``L`` over the whole
        region, an edge is cut when ``(P + L + 1) % period == 0``, which
        staggers the cuts along each chain (the previous pipeline's
        ``subdivide_time_edges_periodic``).

        A cut ``u -H- v`` becomes ``u -H- c -H- c' -H- v``: two new spiders,
        each gated to one end and fused to each other across an H (an
        ``XZ``/``ZX`` fusion), since u and v may already be fused elsewhere.

        Returns the new SED and the fusions.
        """
        if period < 1:
            raise ValueError("period must be a positive integer")
        seam = set(self.seam)

        def qubit(v: Node):
            while isinstance(v, tuple) and v and v[0] == "half":
                v = v[1]
            return v[-1] if isinstance(v, tuple) and v and v[0] == "data" and v[-1] in seam else None

        edges = []
        for v in S:
            if qubit(v) is None:
                continue
            for _, h, _ in sed.graph.edges(v, keys=True):
                if sed.types[h] is not NodeType.H:
                    continue
                u = _through_h(sed, v, h)
                if u in S and qubit(u) == qubit(v) and sed.positions[u][2] < sed.positions[v][2]:
                    edges.append((round(sed.positions[v][1], 6), round(sed.positions[u][2], 6), u, h, v))
        P = {y: i for i, y in enumerate(sorted({e[0] for e in edges}))}
        L = {z: i for i, z in enumerate(sorted({e[1] for e in edges}))}
        fusions: dict[frozenset, LinkKind] = {}
        ed = SEDEditor(sed)
        for y, z, u, h, v in sorted(edges, key=lambda t: (t[1], t[0], repr(t[4]))):
            if (P[y] + L[z] + 1) % period:
                continue
            # u -h- v  ->  u -h- x - v  ->  u -h- H - x(Z) - H - v  ->  u -h- c -H- x -H- v
            x = ed.insert_identity(next(e for e in ed.graph.edges(h, keys=True) if e[1] == v), "X")
            ed.dualise([x])
            (h1,) = [w for w in ed.graph.neighbors(x) if w != h and _through_h(ed, x, w) == u]
            c = ed.insert_identity(next(e for e in ed.graph.edges(h, keys=True) if e[1] == h1), "Z")
            fusions[frozenset((c, x))] = "fusion"
        return ed.freeze(), fusions

    def experiment(self, patch_basis: str, realised: RealisedSED | None = None) -> QECExperiment:
        """Prepare and measure the patches in ``patch_basis`` (the seam is
        closed in the SED). Uses a circuit-based realisation unless one is given."""
        real = realised or RealisedSED.circuit_based(self.sed)
        return QECExperiment(real, {p: patch_basis for p in real.inputs},
                             {p: patch_basis for p in real.outputs})


def two_patch_surgery(d: int, joint: str = "ZZ", rounds: tuple[int, int, int] | None = None,
                      matching: bool = True, layers: tuple[int, int, int] | None = None,
                      ) -> Surgery:
    """``rounds`` = (before, merged, after), each ``d`` by default. With
    ``matching`` (the default) the merge starts and ends with a layer of the
    joint's type, so it has one layer more than ``2 * merged``.

    ``layers`` overrides ``rounds`` with the number of layers in each phase:
    the patches' layers end, and resume, with the type opposite to the merge's
    first and last layers; the merge's last layer follows from its count.
    """
    if joint not in SEAM_BASIS:
        raise ValueError("joint must be 'ZZ' or 'XX'")
    rot = 90 if joint == "ZZ" else 0
    # Each patch's joint-type logical on its outer boundary, away from the seam.
    left = _outer_logical(rotated_surface_code(d, rotation=rot), joint[0], 0)
    right = _outer_logical(rotated_surface_code(d, rotation=rot, origin=(d + 1, 0)), joint[0], 2 * d)
    merged = rotated_surface_code(*((2 * d + 1, d) if rot == 0 else (d, 2 * d + 1)), rotation=rot)
    both = disjoint_union(left, right)
    first = joint[0] if matching else "Z"
    if layers is None:
        r1, r2, r3 = rounds or (d, d, d)
        layers = (2 * r1, 2 * r2 + matching, 2 * r3)
    l1, l2, l3 = layers
    if min(layers) < 1:
        raise ValueError("every phase needs at least one layer")
    # Layers alternate across the phases.
    before, n1 = _phase(both, l1, _other(first) if l1 % 2 else first, first_round=0, z0=0)
    merge, span = _phase(merged, l2, first, first_round=n1, z0=l1)
    last = first if l2 % 2 else _other(first)
    seam = sorted(set(merged.qubits) - set(both.qubits))
    # Closing the seam first leaves the merge's webs seam-free, so composing
    # carries the patches' logicals through and the merge outcome I -> J J
    # (a product of the merge's stabilisers) out to the patches' own logicals.
    basis = SEAM_BASIS[joint]
    merge = close_ports(merge, {p: basis for p in merge.inputs + merge.outputs if p[1] in set(seam)})
    after, _ = _phase(both, l3, _other(last), first_round=n1 + span, z0=l1 + l2)
    sed, _ = compose(before, merge, check="none")
    sed, _ = compose(sed, after)
    rounds = tuple(n // 2 for n in layers)
    return Surgery(sed, joint, (left, right), merged, seam, rounds, range(n1, n1 + span))


def surgery_hybrid_linear(d: int, joint: str = "ZZ", rounds: tuple[int, int, int] | None = None,
                          matching: bool = True, looped: bool = False,
                          layers: tuple[int, int, int] | None = None,
                          period: int | None = None,
                          ) -> tuple[Surgery, RealisedSED, set[Node]]:
    """Hybrid surgery whose merge region is a chain of linear resource states
    (the decomposition of the previous pipeline's
    ``build_surgery_hybrid_linear_long``).

    Starts from the basic hybrid realisation of ``merge_region`` and splits
    its single graph state with fusions (``Surgery.split_seam_linear``): each
    resource state is one column of boundary generators over two consecutive
    merge layers, threaded through that side's seam halves. With ``looped``,
    the region's boundary is first looped (``loop_boundary``, Bell-pair
    rungs), which restores full distance in the non-joint sector.

    ``period`` (``k``) cuts the chains further with fusions on their time
    edges (``Surgery.cut_time_edges``): ``1`` cuts every one, ``k`` every
    k-th, staggered from layer to layer, as ``max_ancillas_per_resource`` with
    ``unfuse_time_edges`` did in the previous pipeline.

    Returns the surgery, the realisation and the clusterised region.
    """
    s = two_patch_surgery(d, joint, rounds, matching, layers)
    sed, S = prepare_cluster_region(s.sed, s.merge_region())
    links: dict[frozenset, LinkKind] = {}
    if looped:
        sed, S, rungs = loop_boundary(sed, S)
        links.update({frozenset(r): "bell" for r in rungs})
    sed, fusions = s.split_seam_linear(sed)
    if period is not None:
        sed, cuts = s.cut_time_edges(sed, S | {v for e in fusions for v in e}, period)
        fusions.update(cuts)
    links.update(fusions)
    S |= {v for e in fusions for v in e}
    return s, RealisedSED.hybrid(sed, S, links), S


def _outer_logical(code: Code, pauli: str, x: int) -> Code:
    """``code`` with its ``pauli`` logical (a column of qubits) moved to column ``x``."""
    ((lx, lz),) = code.logicals
    op = lz if pauli == "Z" else lx
    if len({q[0] for q in op}) != 1:
        raise ValueError("expected the logical to be a column")
    moved = {(x, q[1]): pauli for q in op}
    pair = (lx, moved) if pauli == "Z" else (moved, lz)
    return Code(code.stabilisers, code.qubits, [pair], code.stabiliser_positions)


def _through_h(sed: SEDiagram, v: Node, u: Node) -> Node:
    """The spider reached from v along the leg to u, past any H nodes."""
    prev = v
    while sed.types[u] is NodeType.H:
        prev, u = u, next(w for w in sed.graph.neighbors(u) if w != prev)
    return u


def _other(p: str) -> str:
    return "X" if p == "Z" else "Z"


def _phase(code: Code, n_layers: int, start: str, first_round: int, z0: float
           ) -> tuple[SEDiagram, int]:
    """``n_layers`` alternating layers starting with ``start``, the first at
    height ``z0``, and the number of round labels used."""
    last = start if n_layers % 2 else _other(start)
    rounds = (n_layers - (start == "X") - (last == "Z")) // 2
    sed = SEDiagram.from_code(code, rounds, first_round=first_round, first=start, last=last, z0=z0)
    return sed, rounds + (start == "X") + (last == "Z")
