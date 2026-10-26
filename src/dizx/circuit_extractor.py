"""Compile a ``QECExperiment`` into a Stim circuit.

The extractor reads only the annotated diagram, so circuit-based,
measurement-based and hybrid realisations go through the same code:

- each world-line is a qubit; spiders along it are events on that qubit, and
  H nodes on it are ``H`` gates;
- the first spider prepares the qubit in its colour (Z: |+>, X: |0>) and the
  last one, if it has a classical output, measures it in its colour;
- an interaction between spiders ``u``, ``v`` on different world-lines is a
  two-qubit gate when their effective colours differ (Z-X, or Z-H-Z / X-H-X),
  and a link otherwise: a Bell-pair preparation where both world-lines start,
  a fusion (``MPP``) where both end.

Gates are emitted as ``CX``, ``CZ`` and ``H``; with ``fusions="gates"`` a
fusion is a ``CX`` (``XX``/``ZZ``) or ``CZ`` (``XZ``/``ZX``) followed by
single-qubit measurements, as in the previous synthesiser. An X spider with a gate through
an H node is put in the Hadamard frame for its duration, so all its gates
become ``CZ``/``CX``; at the ends of a world-line the frame is absorbed into
the preparation or measurement basis.

Every operation carries a JSON tag in the format of the previous synthesiser
(see ``noise_models.apply_tagged_noise_model``): ``type`` is ``"circuit"`` or
``"graph"`` (spiders of ``exp.graph_region``), or ``"circuit-graph"`` for
a fusion between the two; ``operation`` is ``reset``, ``single_qubit_gate``,
``two_qubit_gate``, ``single_qubit_measurement``, ``fusion_gate``,
``fusion_measurement``, ``bell_prep_reset``, ``bell_prep_two_qubit``,
``state_prep`` (an ``I`` on each resource-state qubit before its readout) or
``idle`` (an ``I`` on each live data qubit after each layer); ``qubit_type`` is ``data``/``ancilla`` for circuit qubits and
``matter`` (touching a circuit qubit) / ``flying`` for graph qubits; ``t`` is
the layer.

Ordering: spiders sharing a scheduled generator (or any unscheduled gate) form
a **step**; steps are ordered by the world-lines. Consecutive steps at one
height on disjoint world-lines (e.g. the same round on two patches) form a
**layer**, and run side by side: their preparations, gates and measurements
share ``TICK``s. Within a step, gates of scheduled generators run in ``TICK`` =
slot (shifted to avoid clashes), and other gates are packed greedily. Qubits
are prepared just before their first layer and measured right after their
last, and physical qubits are reused per (x, y) position.
"""
from __future__ import annotations

import heapq
import json
from collections import defaultdict
from typing import Hashable, Iterable

import numpy as np
import stim

from dizx.realisation import Interaction, QECExperiment, interactions
from dizx.schedules import Schedule
from dizx.zx_diagram import SPIDERS, Node, NodeType


def to_stim(exp: QECExperiment, schedule: Schedule | None = None,
            fusions: str = "mpp") -> stim.Circuit:
    """The Stim circuit of ``exp``, with a ``DETECTOR`` per detector web and an
    ``OBSERVABLE_INCLUDE`` per observable web. ``fusions`` is ``"mpp"`` (one
    ``MPP`` per fusion) or ``"gates"`` (an entangling gate and two
    single-qubit measurements)."""
    if fusions not in ("mpp", "gates"):
        raise ValueError("fusions must be 'mpp' or 'gates'")
    return _Extractor(exp, schedule, fusions).run()


def _tag(**kwargs) -> str:
    return json.dumps(kwargs)


class _Text:
    """Builds a circuit as Stim text, parsed once at the end: much faster than
    ``stim.Circuit.append`` per instruction, with the same result (parsing
    merges consecutive instructions just as appending does)."""

    _ESCAPES = str.maketrans({"\\": "\\B", "]": "\\C", "\r": "\\r", "\n": "\\n"})

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append(self, name: str, targets: Iterable = (), arg=None, tag: str = "") -> None:
        head = name
        if tag:
            head += f"[{tag.translate(self._ESCAPES)}]"
        if arg is not None:
            args = list(arg) if isinstance(arg, (list, tuple)) else [arg]
            if args:
                head += "(" + ", ".join(repr(float(a)) for a in args) + ")"
        self.lines.append(" ".join([head, *map(self._target, targets)]))

    @staticmethod
    def _target(t) -> str:
        if isinstance(t, stim.GateTarget):
            if t.is_combiner:
                return "*"
            if t.is_measurement_record_target:
                return f"rec[{t.value}]"
            if t.is_x_target:
                return f"X{t.value}"
            if t.is_y_target:
                return f"Y{t.value}"
            if t.is_z_target:
                return f"Z{t.value}"
            return str(t.value)
        return str(int(t))

    def circuit(self) -> stim.Circuit:
        return stim.Circuit("\n".join(self.lines))


class _Extractor:
    def __init__(self, exp: QECExperiment, schedule: Schedule | None, fusions: str = "mpp") -> None:
        self.exp = exp
        self.schedule = schedule
        self.fusion_gates = fusions == "gates"
        self.types = exp.types
        self.pos = exp.positions
        self.lines = [tuple(p) for p in exp.worldlines]
        self.line_of = {v: i for i, p in enumerate(self.lines) for v in p}
        self.index_in = {v: k for p in self.lines for k, v in enumerate(p)}
        # Per line (keyed by id): number of H nodes before each position.
        self._h_before = {id(p): np.concatenate([[0], np.cumsum([self.types[v] is NodeType.H for v in p])])
                          for p in self.lines}
        self.n_spiders = [sum(self.types[v] in SPIDERS for v in p) for p in self.lines]
        self.singleton = {i for i, k in enumerate(self.n_spiders) if k == 1}

        for v in exp.classical:
            line = self.lines[self.line_of[v]]
            if line[-1] != v:
                raise ValueError(f"classical output on {v!r}, which is not the end of its world-line")

        links = {frozenset(link.interaction.ends): link for link in exp.links}
        self.gates: list[Interaction] = []
        for it in interactions(exp, self.lines):
            if frozenset(it.ends) in links:
                continue
            if it.same_colour(self.types):
                raise ValueError(f"same-colour edge {it.ends} is not a Bell pair or fusion")
            self.gates.append(it)
        self.links = list(links.values())
        self.gates_at: dict[Node, list[Interaction]] = defaultdict(list)
        for g in self.gates:
            for v in g.ends:
                self.gates_at[v].append(g)
        # X spiders with a gate through an H node work in the Hadamard frame.
        self.framed = {v for v, gs in self.gates_at.items()
                       if self.types[v] is NodeType.X and any(g.hadamards % 2 for g in gs)}

        # Roles for tags: graph (resource-state) or circuit world-lines.
        region = exp.graph_region
        self.graph = {i for i, p in enumerate(self.lines) if any(v in region for v in p)}
        touching = {(self.line_of[g.ends[0]], self.line_of[g.ends[1]]) for g in self.gates}
        touching |= {tuple(self.line_of[v] for v in link.interaction.ends) for link in self.links}
        self.matter = {i for a, b in touching for i, j in ((a, b), (b, a))
                       if i in self.graph and j not in self.graph}
        self._gated = {id(p): [v for v in p if v in self.gates_at] for p in self.lines}

    # -- tags ---------------------------------------------------------------

    def kind(self, line: int) -> str:
        return "graph" if line in self.graph else "circuit"

    def qubit_type(self, line: int) -> str:
        if line in self.graph:
            return "matter" if line in self.matter else "flying"
        return "data" if self.n_spiders[line] > 1 else "ancilla"

    def pair_kind(self, a: int, b: int) -> str:
        """Type of an operation on two world-lines: a gate touching a graph
        qubit is a graph gate; a fusion between the two kinds is circuit-graph."""
        kinds = {self.kind(a), self.kind(b)}
        return kinds.pop() if len(kinds) == 1 else "mixed"

    # -- helpers ------------------------------------------------------------

    def z(self, v: Node) -> float:
        return float(self.pos[v][2]) if v in self.pos else 0.0

    def xy(self, v: Node) -> tuple | None:
        return tuple(np.round(self.pos[v][:2], 6)) if v in self.pos else None

    def coupling(self, v: Node) -> NodeType:
        return NodeType.Z if v in self.framed else self.types[v]

    def gated(self, line: tuple[Node, ...]) -> list[Node]:
        """The spiders of a world-line (one of ``self.lines``) with gates."""
        return self._gated[id(line)]

    def flips(self, line: tuple[Node, ...], a: int, b: int) -> int:
        """Number of H nodes strictly between positions a and b of a world-line."""
        h = self._h_before[id(line)]
        return int(h[b] - h[a + 1]) if b > a + 1 else 0

    # -- steps --------------------------------------------------------------

    def steps(self) -> tuple[list[list[list[Node]]], dict[Node, int]]:
        """The layers, each a list of steps (blocks of spiders), in order, and
        the layer of every gated spider."""
        parent: dict[Node, Node] = {}

        def find(v: Node) -> Node:
            parent.setdefault(v, v)
            while parent[v] != v:
                parent[v] = parent[parent[v]]
                v = parent[v]
            return v

        for g in self.gates:
            u, v = g.ends
            parent[find(u)] = find(v)
        groups: dict[Node, list[Node]] = defaultdict(list)
        for v in self.gates_at:
            groups[find(v)].append(v)
        blocks = list(groups.values())
        block_of = {v: i for i, b in enumerate(blocks) for v in b}
        for b in blocks:
            lines = [self.line_of[v] for v in b]
            if len(lines) != len(set(lines)):
                raise ValueError("a step contains two spiders of one world-line; cannot order its gates")

        succ: dict[int, set[int]] = defaultdict(set)
        indeg = [0] * len(blocks)
        for line in self.lines:
            seq = [block_of[v] for v in self.gated(line)]
            for a, b in zip(seq, seq[1:]):
                if a != b and b not in succ[a]:
                    succ[a].add(b)
                    indeg[b] += 1
        key = [(min(self.z(v) for v in b), min(repr(v) for v in b)) for b in blocks]
        heap = [(key[i], i) for i in range(len(blocks)) if indeg[i] == 0]
        heapq.heapify(heap)
        order = []
        while heap:
            _, i = heapq.heappop(heap)
            order.append(i)
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    heapq.heappush(heap, (key[j], j))
        if len(order) != len(blocks):
            raise ValueError("steps are cyclically ordered by the world-lines; not a circuit")
        # Consecutive steps at one height on disjoint world-lines share a layer.
        # Any path between two steps runs through a step ordered between them,
        # so consecutive steps sharing no world-line are independent.
        layers: list[list[int]] = []
        used: set[int] = set()
        for i in order:
            lines = {self.line_of[v] for v in blocks[i]}
            if layers and key[i][0] == key[layers[-1][0]][0] and not lines & used:
                layers[-1].append(i)
                used |= lines
            else:
                layers.append([i])
                used = lines
        rank = {b: r for r, layer in enumerate(layers) for b in layer}
        return [[blocks[i] for i in layer] for layer in layers], {v: rank[block_of[v]] for v in block_of}

    def ticks(self, block: list[Node], layers: dict[NodeType, set[float]],
              ) -> tuple[list[list[Interaction]], set[tuple[NodeType, float]]]:
        """Gates of one step, grouped into ticks, and the (type, height) of its
        scheduled generators. ``layers[t]`` holds the heights of earlier
        scheduled generators of type ``t``; a generator's layer index counts
        those below it, so the steps of one layer (e.g. two separate patches)
        share an index."""
        singleton = self.singleton
        tick_of: dict[Interaction, int] = {}
        ancillas: list[Node] = []
        if self.schedule is not None:
            # An ancilla is scheduled on its gates to data qubits. Its gates to
            # other single-spider qubits (e.g. the photon of a Bell-prep
            # gadget) are pre-gates, run before the scheduled ones.
            def data_gates(v: Node) -> list[Interaction]:
                return [g for g in self.gates_at[v]
                        if all(self.line_of[w] not in singleton for w in g.ends if w != v)]

            def schedulable(v: Node) -> bool:
                line = self.line_of[v]
                if line not in singleton or not data_gates(v):
                    return False
                others = [w for g in self.gates_at[v] if g not in data_gates(v)
                          for w in g.ends if w != v]
                return not others or (line not in self.graph
                                      and all(self.line_of[w] in self.graph for w in others))

            ancillas = sorted((v for v in block if schedulable(v)), key=lambda v: (self.z(v), repr(v)))
            n = self.schedule.n_slots
            substeps: list[set[tuple[int, int]]] = []
            for a in ancillas:
                partner = {g: (g.ends[1] if g.ends[0] == a else g.ends[0]) for g in data_gates(a)}
                layer = sum(h < self.z(a) for h in layers.get(self.types[a], ()))
                slots = self.schedule.slots(a, list(partner.values()), self.pos, self.types[a], layer)
                need = {(self.line_of[w], slots[w]) for w in partner.values()}
                need |= {(self.line_of[a], k) for k in slots.values()}
                s = next((i for i, used in enumerate(substeps) if not used & need), len(substeps))
                if s == len(substeps):
                    substeps.append(set())
                substeps[s] |= need
                for g, w in partner.items():
                    tick_of[g] = s * n + slots[w]
        rest = sorted({g for v in block for g in self.gates_at[v]} - set(tick_of),
                      key=lambda g: (self.pair_kind(*(self.line_of[v] for v in g.ends)) == "mixed",
                                     max(self.z(v) for v in g.ends), repr(g.ends)))
        scheduled_lines = {self.line_of[a] for a in ancillas}
        busy: dict[int, set[int]] = defaultdict(set)
        if any(self.line_of[v] in scheduled_lines for g in rest for v in g.ends):
            # Pre-gates: pack the rest first and shift the scheduled gates after them.
            pre: dict[Interaction, int] = {}
            for g in rest:
                q = {self.line_of[v] for v in g.ends}
                t = 0
                while busy[t] & q:
                    t += 1
                busy[t] |= q
                pre[g] = t
            shift = max(pre.values()) + 1
            tick_of = {g: t + shift for g, t in tick_of.items()} | pre
            rest = []
            busy = defaultdict(set)
        for g, t in tick_of.items():
            busy[t] |= {self.line_of[v] for v in g.ends}
        for g in rest:
            q = {self.line_of[v] for v in g.ends}
            t = 0
            while busy[t] & q:
                t += 1
            tick_of[g] = t
            busy[t] |= q
        by_tick: dict[int, list[Interaction]] = defaultdict(list)
        for g, t in tick_of.items():
            by_tick[t].append(g)
        scheduled = {(self.types[a], self.z(a)) for a in ancillas}
        return [sorted(by_tick[t], key=lambda g: repr(g.ends)) for t in sorted(by_tick)], scheduled

    # -- emission -----------------------------------------------------------

    def run(self) -> stim.Circuit:
        layer_steps, step_of = self.steps()
        n_steps = max(len(layer_steps), 1)
        first, last = {}, {}
        for i, line in enumerate(self.lines):
            g = self.gated(line)
            first[i] = step_of[g[0]] if g else 0
            last[i] = step_of[g[-1]] if g else 0

        bell = {}   # line -> link, for both ends of each Bell pair
        fusion = {}
        for link in self.links:
            for v in link.interaction.ends:
                i = self.line_of[v]
                links = bell if link.kind == "bell" else fusion
                if i in links:  # a fusion is destructive; a Bell pair starts the line
                    raise ValueError(f"world-line of {v!r} is in two {link.kind} links")
                links[i] = link

        # What happens at each step, gathered once (in the order a scan of
        # every line or link at that step would find it).
        starts_at, ends_at = defaultdict(list), defaultdict(list)
        for i in range(len(self.lines)):
            starts_at[first[i]].append(i)
            ends_at[last[i]].append(i)
        bells_at, fusions_at = defaultdict(dict), defaultdict(dict)
        for i, link in bell.items():
            bells_at[min(first[self.line_of[v]] for v in link.interaction.ends)][id(link)] = link
        for i, link in fusion.items():
            fusions_at[max(last[self.line_of[v]] for v in link.interaction.ends)][id(link)] = link
        steps_at = defaultdict(list)  # step -> (line, u, w) for gated u, w consecutive on line
        for i, p in enumerate(self.lines):
            g = self.gated(p)
            for u, w in zip(g, g[1:]):
                steps_at[step_of[u]].append((i, p, u, w))

        circ = _Text()
        coords: dict[int, tuple] = {}
        phys: dict[int, int] = {}
        pool: dict[tuple | None, list[int]] = defaultdict(list)
        n_qubits = 0
        meas_index: dict[Hashable, int] = {}
        n_meas = 0

        def alloc(line: int) -> int:
            nonlocal n_qubits
            head = self.gated(self.lines[line]) or [self.lines[line][0]]
            key = self.xy(head[0])
            if pool[key]:
                q = pool[key].pop(0)
            else:
                q, n_qubits = n_qubits, n_qubits + 1
                if key is not None:
                    coords[q] = key
            phys[line] = q
            return q

        def release(line: int) -> None:
            q = phys.pop(line)
            head = self.gated(self.lines[line]) or [self.lines[line][0]]
            pool[self.xy(head[0])].append(q)
            pool[self.xy(head[0])].sort()

        def record(labels: list[Hashable]) -> None:
            nonlocal n_meas
            for lab in labels:
                if lab not in meas_index:
                    for k in waiting_on.get(lab, ()):
                        unmet[k] -= 1
                        if not unmet[k]:
                            ready.append(k)
                meas_index[lab] = n_meas
                n_meas += 1

        def emit(name: str, targets: list, tick: bool = True) -> None:
            if targets:
                circ.append(name, targets)
                if tick:
                    circ.append("TICK")

        def emit_tagged(name: str, items: list[tuple[str, list]]) -> None:
            """One instruction per tag, in order of first appearance; each item
            is (tag, targets of one operation)."""
            groups: dict[str, list] = {}
            for tag, targets in items:
                groups.setdefault(tag, []).extend(targets)
            for tag, targets in groups.items():
                circ.append(name, targets, tag=tag)

        def op_tag(line: int, operation: str, s: int, with_type: bool = True) -> str:
            kw = dict(type=self.kind(line), operation=operation, t=s)
            if with_type:
                kw["qubit_type"] = self.qubit_type(line)
            return _tag(**kw)

        detectors = [sorted(rec, key=repr) for rec in self.exp.records(self.exp.detectors)]
        observables = self.exp.records(self.exp.observables)
        emitted = [False] * len(detectors)
        # Detectors waiting on each outcome, and how many outcomes each still lacks.
        waiting_on: dict[Hashable, list[int]] = defaultdict(list)
        for k, rec in enumerate(detectors):
            for lab in rec:
                waiting_on[lab].append(k)
        unmet = [len(rec) for rec in detectors]
        ready: list[int] = []
        obs_seen: list[set[Hashable]] = [set() for _ in observables]
        layers: dict[NodeType, set[float]] = {}

        def start_basis(line: int) -> str:
            """'Z' (|0>) or 'X' (|+>) after the prep spider and any H up to the
            first gated spider, in that spider's frame."""
            p = self.lines[line]
            g = self.gated(p)
            end = self.index_in[g[0]] if g else len(p) - 1
            flips = self.flips(p, 0, end) + (g[0] in self.framed if g else 0)
            basis = "X" if self.types[p[0]] is NodeType.Z else "Z"
            return basis if flips % 2 == 0 else {"X": "Z", "Z": "X"}[basis]

        def tail_flips(line: int) -> int:
            p = self.lines[line]
            g = self.gated(p)
            a = self.index_in[g[-1]] if g else 0
            return self.flips(p, a, len(p) - 1) + (g[-1] in self.framed if g else 0)

        for s in range(n_steps):
            # -- preparations --
            starting = [i for i in starts_at[s] if i not in bell]
            new_bell = sorted(bells_at[s].values(), key=lambda l: repr(l.interaction.ends))
            r, rx = [], []
            for i in starting:
                q = alloc(i)
                (r if start_basis(i) == "Z" else rx).append((q, i))
            bell_pairs = []
            for link in new_bell:
                a, b = (self.line_of[v] for v in link.interaction.ends)
                bell_pairs.append((link, alloc(a), alloc(b), a, b))
            # A Bell pair is |+>|0> and a CX.
            bell_tag = lambda i, op: (op_tag(i, op, s) if i in self.graph else
                                      _tag(type="circuit", operation=op, t=s, qubit_type="ancilla"))
            r_items = [(op_tag(i, "reset", s), [q]) for q, i in sorted(r)]
            r_items += [(bell_tag(b, "bell_prep_reset"), [qb]) for _, _, qb, _, b in bell_pairs]
            rx_items = [(op_tag(i, "reset", s), [q]) for q, i in sorted(rx)]
            rx_items += [(bell_tag(a, "bell_prep_reset"), [qa]) for _, qa, _, a, _ in bell_pairs]
            if r_items or rx_items:
                emit_tagged("R", r_items)
                emit_tagged("RX", rx_items)
                circ.append("TICK")
            if bell_pairs:
                emit_tagged("CX", [(bell_tag(a, "bell_prep_two_qubit"), [qa, qb])
                                   for _, qa, qb, a, _ in bell_pairs])
                circ.append("TICK")
                fix = []
                for link, qa, qb, a, b in bell_pairs:
                    for line, q, extra in ((a, qa, 0), (b, qb, link.interaction.hadamards % 2)):
                        p = self.lines[line]
                        g = self.gated(p)
                        flips = self.flips(p, 0, self.index_in[g[0]] if g else len(p) - 1)
                        flips += (g[0] in self.framed if g else 0) + extra
                        if flips % 2:
                            fix.append((q, line))
                if fix:
                    emit_tagged("H", [(bell_tag(line, "single_qubit_gate"), [q]) for q, line in sorted(fix)])
                    circ.append("TICK")

            # -- gates --
            # The steps of a layer touch disjoint qubits: their k-th ticks run together.
            ticks: list[list[Interaction]] = []
            scheduled: set[tuple[NodeType, float]] = set()
            for block in (layer_steps[s] if s < len(layer_steps) else []):
                block_ticks, block_scheduled = self.ticks(block, layers)
                ticks += [[] for _ in range(len(block_ticks) - len(ticks))]
                for k, tick in enumerate(block_ticks):
                    ticks[k] += tick
                scheduled |= block_scheduled
            ticks = [sorted(tick, key=lambda g: repr(g.ends)) for tick in ticks]
            for t, h in scheduled:
                layers.setdefault(t, set()).add(h)
            for tick in ticks:
                cz, cx = [], []
                for g in tick:
                    u, v = g.ends
                    lu, lv = self.line_of[u], self.line_of[v]
                    qu, qv = phys[lu], phys[lv]
                    kind = self.pair_kind(lu, lv)
                    tag = _tag(type="graph" if kind == "mixed" else kind,
                               operation="two_qubit_gate", t=s)
                    cu, cv = self.coupling(u), self.coupling(v)
                    if cu is NodeType.Z and cv is NodeType.Z:
                        cz.append((tag, [qu, qv]))
                    elif cu is NodeType.Z:
                        cx.append((tag, [qu, qv]))
                    elif cv is NodeType.Z:
                        cx.append((tag, [qv, qu]))
                    else:
                        raise AssertionError("X-X coupling should be framed")
                emit_tagged("CZ", cz)
                emit_tagged("CX", cx)
                circ.append("TICK")

            # -- basis changes between steps, and before fusions --
            hs = []
            for i, p, u, w in steps_at[s]:
                flips = (u in self.framed) + self.flips(p, self.index_in[u], self.index_in[w]) + (w in self.framed)
                if flips % 2:
                    hs.append((phys[i], i))
            ending = ends_at[s]
            fusions = sorted(fusions_at[s].values(), key=lambda l: repr(l.interaction.ends))
            for link in fusions:
                for v in link.interaction.ends:
                    if tail_flips(self.line_of[v]) % 2:
                        hs.append((phys[self.line_of[v]], self.line_of[v]))
            if hs:
                emit_tagged("H", [(op_tag(i, "single_qubit_gate", s, with_type=False), [q])
                                  for q, i in sorted(hs)])
                circ.append("TICK")

            # -- measurements and fusions --
            # One marker per finished resource-state qubit, before its readout.
            done = sorted(i for i in ending if i in self.graph)
            if done:
                emit_tagged("I", [(op_tag(i, "state_prep", s), [phys[i]]) for i in done])
            if self.fusion_gates and fusions:
                # CX: XX on qa (MX), ZZ on qb (M). CZ: XZ on qa, ZX on qb (both MX).
                for link in fusions:
                    la, lb = (self.line_of[v] for v in link.interaction.ends)
                    kind = self.pair_kind(la, lb)
                    circ.append("CZ" if link.ops == ("XZ", "ZX") else "CX", [phys[la], phys[lb]],
                                tag=_tag(type="circuit-graph" if kind == "mixed" else kind,
                                         operation="fusion_gate", t=s))
                circ.append("TICK")
            m, mx = [], []
            for i in sorted(ending, key=lambda i: phys[i]):
                if i in fusion:
                    continue
                end = self.lines[i][-1]
                basis = "X" if self.types[end] is NodeType.Z else "Z"
                if tail_flips(i) % 2:
                    basis = {"X": "Z", "Z": "X"}[basis]
                label = self.exp.phases[self.exp.classical[end]].label
                tag = op_tag(i, "single_qubit_measurement", s,
                             with_type=self.qubit_type(i) == "data")
                (m if basis == "Z" else mx).append((tag, phys[i], label))
            for name, items in (("M", m), ("MX", mx)):
                emit_tagged(name, [(tag, [q]) for tag, q, _ in items])
                order: dict[str, list] = {}
                for tag, _, label in items:
                    order.setdefault(tag, []).append(label)
                for labels in order.values():
                    record(labels)
            for link in fusions:
                la, lb = (self.line_of[v] for v in link.interaction.ends)
                qa, qb = phys[la], phys[lb]
                kind = self.pair_kind(la, lb)
                kind = "circuit-graph" if kind == "mixed" else kind
                if self.fusion_gates:
                    hadamard = link.ops == ("XZ", "ZX")
                    for q, op, name in ((qa, link.ops[0], "MX"),
                                        (qb, link.ops[1], "MX" if hadamard else "M")):
                        circ.append(name, [q], tag=_tag(type=kind, operation="fusion_measurement",
                                                         t=s, basis=op))
                else:
                    targets = []
                    for op in link.ops:  # e.g. "XZ": X on qa times Z on qb
                        pa = stim.target_x(qa) if op[0] == "X" else stim.target_z(qa)
                        pb = stim.target_x(qb) if op[1] == "X" else stim.target_z(qb)
                        targets += [pa, stim.target_combiner(), pb]
                    circ.append("MPP", targets, tag=_tag(type=kind, operation="fusion_measurement", t=s))
                record(list(link.outcomes))
            if m or mx or fusions:
                circ.append("TICK")
            for i in ending:
                if i not in fusion:  # fused qubits wait for their partner
                    release(i)
            for link in fusions:
                for v in link.interaction.ends:
                    release(self.line_of[v])

            # -- one idle marker per live data qubit after each layer --
            height = (min(self.z(v) for b in layer_steps[s] for v in b)
                      if s < len(layer_steps) else None)
            following = (min(self.z(v) for b in layer_steps[s + 1] for v in b)
                         if s + 1 < len(layer_steps) else None)
            if height is not None and following != height:
                live = sorted(phys[i] for i in phys
                              if i not in self.graph and self.qubit_type(i) == "data")
                if live:
                    circ.append("I", live, tag=_tag(type="circuit", operation="idle", t=s,
                                                     qubit_type="data"))

            # -- detectors and observables --
            for k in sorted(ready):
                rec = detectors[k]
                emitted[k] = True
                targets = [stim.target_rec(meas_index[lab] - n_meas) for lab in rec]
                circ.append("DETECTOR", targets, self._coords(rec, s))
            ready.clear()
            for k, rec in enumerate(observables):
                new = sorted((lab for lab in rec if lab in meas_index and lab not in obs_seen[k]),
                             key=repr)
                if new:
                    obs_seen[k] |= set(new)
                    circ.append("OBSERVABLE_INCLUDE",
                                [stim.target_rec(meas_index[lab] - n_meas) for lab in new], k)

        missing = [k for k, done in enumerate(emitted) if not done and detectors[k]]
        if missing:
            raise ValueError(f"detectors {missing[:5]} reference outcomes never measured")
        head = _Text()
        for q in sorted(coords):
            head.append("QUBIT_COORDS", [q], list(coords[q]))
        head.lines += circ.lines
        return head.circuit()

    def _coords(self, labels: list[Hashable], step: int) -> list[float]:
        pts = []
        for lab in labels:
            nodes = [lab] if lab in self.types else list(lab[1:3])
            pts += [self.pos[v][:2] for v in nodes if v in self.pos]
        xy = list(np.mean(pts, axis=0)) if pts else []
        return [float(c) for c in xy] + [float(step)]
