"""Gate schedules for generator spiders.

A schedule assigns each leg of a generator spider (one two-qubit gate between
its ancilla and a data qubit) a **slot** in ``range(n_slots)``. During
extraction, all gates in the same slot of a layer run in one TICK.

Slots may be left empty: a weight-2 boundary plaquette of the rotated surface
code keeps the slots of its missing corners, so it stays in step with the bulk.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np

from dizx.code import Code, Qubit
from dizx.zx_diagram import Node, NodeType

Point = Sequence[float]


class Schedule(ABC):
    """Assigns slots to the legs of a generator spider. Subclass to define your own."""

    n_slots: int

    @abstractmethod
    def slots(self, gen: Node, neighbours: Sequence[Node], positions: Mapping[Node, Point],
              gen_type: NodeType, layer: int = 0) -> dict[Node, int]:
        """Slot of each neighbour of ``gen``; distinct, in ``range(n_slots)``.

        ``layer`` counts the earlier layers of generators of the same spider
        type, so a schedule may vary between layers."""


class RotatedSurfaceCodeSchedule(Schedule):
    """Hook-safe orders for a rotated surface code, in any orientation.

    Slot ``k`` is the gate to the data qubit in the quadrant of
    ``orders[type][k]`` around the generator, comparing (x, y) positions;
    missing corners of weight-2 plaquettes leave their slots empty.

    A type-P check's hook error is a P-type pair on its last two qubits, which
    must lie parallel to the type-P boundary (perpendicular to the type-P
    logical). The boundary axis is read from the code: the support of each
    weight-2 type-P generator lies along it. The order ends on a row pair for
    a horizontal boundary, ``(+,+), (-,+), (+,-), (-,-)``, and on a column
    pair for a vertical one, ``(+,+), (+,-), (-,+), (-,-)``. These match the
    gate order of the previous DISQEC pipeline.

    The order is already hook-safe, so every layer uses it. With ``alternate``
    (as in the previous pipeline), odd layers of each type run it backwards;
    the last two gates are then the first two of the forward order, which lie
    along the same axis. Split generators may lose distance with ``alternate``,
    since a dropped leg moves from early to late in the order.

    Qubit labels are used as positions unless ``qubit_positions`` is given.
    """

    ROW_LAST = [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    COLUMN_LAST = [(0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (-0.5, -0.5)]
    n_slots = 4

    def __init__(self, code: Code, qubit_positions: Mapping[Qubit, Point] | None = None,
                 alternate: bool = False) -> None:
        self.alternate = alternate
        pos = qubit_positions or {q: q for q in code.qubits}
        self.boundary_axis: dict[str, str] = {}
        for p in "XZ":
            axes = set()
            for s in code.stabilisers:
                if len(s) == 2 and p in s.values():
                    (x0, y0), (x1, y1) = (pos[q][:2] for q in s)
                    axes.add("row" if np.isclose(y0, y1) else "column" if np.isclose(x0, x1) else "?")
            if len(axes) > 1 or "?" in axes:
                raise ValueError(f"cannot read a single {p}-boundary axis from the weight-2 generators")
            if axes:
                self.boundary_axis[p] = axes.pop()
        # The X and Z boundaries are perpendicular; infer one from the other if
        # a type has no weight-2 generators (e.g. d = 2).
        other = {"row": "column", "column": "row"}
        for p, q in (("X", "Z"), ("Z", "X")):
            if p not in self.boundary_axis and q in self.boundary_axis:
                self.boundary_axis[p] = other[self.boundary_axis[q]]
        if len(self.boundary_axis) != 2:
            raise ValueError("cannot read the boundary axes from the weight-2 generators")
        order = {"row": self.ROW_LAST, "column": self.COLUMN_LAST}
        # X-type generators are Z spiders and vice versa.
        self.orders = {NodeType.Z: order[self.boundary_axis["X"]],
                       NodeType.X: order[self.boundary_axis["Z"]]}

    def slots(self, gen, neighbours, positions, gen_type, layer=0):
        """Match each neighbour to the corner in the same quadrant, so slightly
        moved generators (e.g. split halves) keep their bulk slots."""
        g = np.asarray(positions[gen][:2], dtype=float)
        order = [tuple(np.sign(d)) for d in self.orders[gen_type]]
        if self.alternate and layer % 2:
            order = order[::-1]
        out = {}
        for v in neighbours:
            quadrant = tuple(np.sign(np.asarray(positions[v][:2], dtype=float) - g))
            if quadrant not in order:
                raise ValueError(f"neighbour {v!r} of {gen!r} is not at a plaquette corner")
            out[v] = order.index(quadrant)
        if len(set(out.values())) != len(out):
            raise ValueError(f"two neighbours of {gen!r} share a corner")
        return out


class UnrotatedSurfaceCodeSchedule(Schedule):
    """Hook-safe orders for an unrotated surface code, in any orientation.

    Slot ``k`` is the gate to the data qubit in direction ``orders[type][k]``
    from the generator (the sign of the larger of the x and y offsets);
    missing neighbours of weight-3 generators leave their slots empty.

    Every order ends on one horizontal and one vertical neighbour, so a hook
    error is a diagonal pair. A minimum-weight logical runs straight along
    one of the axes and gains nothing from it, so the orders need not depend
    on the boundaries: E N S W for X-type generators and E S N W for Z-type
    ones.
    """

    X_ORDER = [(1, 0), (0, 1), (0, -1), (-1, 0)]
    Z_ORDER = [(1, 0), (0, -1), (0, 1), (-1, 0)]
    n_slots = 4

    def __init__(self, code: Code | None = None) -> None:
        # X-type generators are Z spiders and vice versa.
        self.orders = {NodeType.Z: self.X_ORDER, NodeType.X: self.Z_ORDER}

    def slots(self, gen, neighbours, positions, gen_type, layer=0):
        g = np.asarray(positions[gen][:2], dtype=float)
        order = self.orders[gen_type]
        out = {}
        for v in neighbours:
            dx, dy = np.asarray(positions[v][:2], dtype=float) - g
            step = (int(np.sign(dx)), 0) if abs(dx) > abs(dy) else (0, int(np.sign(dy)))
            if np.isclose(abs(dx), abs(dy)):
                raise ValueError(f"neighbour {v!r} of {gen!r} is not on an axis")
            out[v] = order.index(step)
        if len(set(out.values())) != len(out):
            raise ValueError(f"two neighbours of {gen!r} share a direction")
        return out
