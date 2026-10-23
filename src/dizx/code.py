"""Stabiliser codes defined by their generators.

Pauli operators are sparse maps ``{qubit: 'X' | 'Y' | 'Z'}``; internally they
are symplectic F2 vectors ``(x | z)`` over the code's qubit ordering.
"""
from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np

from dizx._f2 import nullspace, rank

Qubit = Hashable
Pauli = dict[Qubit, str]

_BITS = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}
_CHARS = {xz: p for p, xz in _BITS.items()}


def symplectic(a: np.ndarray, b: np.ndarray) -> int:
    """Symplectic product of ``(x | z)`` vectors: 0 iff they commute."""
    n = len(a) // 2
    return int(a[:n] @ b[n:] + a[n:] @ b[:n]) % 2


class Code:
    """A stabiliser code on labelled qubits.

    Attributes
    ----------
    qubits : tuple[Qubit, ...]
        Fixed qubit ordering for the symplectic representation.
    stabilisers : list[Pauli]
        Independent, commuting generators.
    logicals : list[tuple[Pauli, Pauli]]
        ``k`` anticommuting pairs, each commuting with the stabilisers and
        with every other pair. Computed if not given.
    stabiliser_positions : list[tuple[float, float]] | None
        Optional 2D position of each stabiliser, used for layouts.
    """

    def __init__(
        self,
        stabilisers: Sequence[Mapping[Qubit, str]],
        qubits: Sequence[Qubit] | None = None,
        logicals: Sequence[tuple[Mapping[Qubit, str], Mapping[Qubit, str]]] | None = None,
        stabiliser_positions: Sequence[tuple[float, float]] | None = None,
    ) -> None:
        self.stabilisers = [_clean(s) for s in stabilisers]
        if stabiliser_positions is not None and len(stabiliser_positions) != len(self.stabilisers):
            raise ValueError("need one position per stabiliser")
        self.stabiliser_positions = (
            None if stabiliser_positions is None else [tuple(p) for p in stabiliser_positions]
        )
        if qubits is None:
            qubits = sorted({q for s in self.stabilisers for q in s}, key=repr)
        self.qubits = tuple(qubits)
        self._index = {q: i for i, q in enumerate(self.qubits)}
        if len(self._index) != len(self.qubits):
            raise ValueError("duplicate qubits")

        S = self.stabiliser_matrix
        if any(symplectic(a, b) for i, a in enumerate(S) for b in S[i + 1:]):
            raise ValueError("stabilisers do not commute")
        if rank(S) < len(S):
            raise ValueError("stabilisers are not independent")

        if logicals is None:
            self.logicals = [(self.to_pauli(a), self.to_pauli(b)) for a, b in self._find_logicals()]
        else:
            self.logicals = [(_clean(a), _clean(b)) for a, b in logicals]
            self._check_logicals()

    # -- representation -----------------------------------------------------

    @property
    def n(self) -> int:
        return len(self.qubits)

    @property
    def k(self) -> int:
        return self.n - len(self.stabilisers)

    def to_vec(self, p: Mapping[Qubit, str]) -> np.ndarray:
        v = np.zeros(2 * self.n, dtype=np.uint8)
        for q, c in p.items():
            i = self._index[q]
            v[i], v[self.n + i] = _BITS[c]
        return v

    def to_pauli(self, v: np.ndarray) -> Pauli:
        p = {q: _CHARS[int(v[i]), int(v[self.n + i])] for i, q in enumerate(self.qubits)}
        return {q: c for q, c in p.items() if c != "I"}

    @property
    def stabiliser_matrix(self) -> np.ndarray:
        """``m x 2n`` symplectic matrix, one row per stabiliser."""
        rows = [self.to_vec(s) for s in self.stabilisers]
        return np.array(rows, dtype=np.uint8).reshape(len(rows), 2 * self.n)

    def is_css(self) -> bool:
        return all(len(set(s.values())) == 1 and "Y" not in s.values() for s in self.stabilisers)

    # -- logicals -----------------------------------------------------------

    def _find_logicals(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Symplectic basis of the normaliser modulo the stabiliser group."""
        S = self.stabiliser_matrix
        n = self.n
        normaliser = nullspace(np.hstack([S[:, n:], S[:, :n]]))
        # Complete the stabilisers to a basis of the normaliser.
        span, rest = S, []
        for v in normaliser:
            if rank(np.vstack([span, v])) > len(span):
                span = np.vstack([span, v])
                rest.append(v.copy())
        # Symplectic Gram-Schmidt.
        pairs = []
        while rest:
            a = rest.pop(0)
            j = next(j for j, b in enumerate(rest) if symplectic(a, b))
            b = rest.pop(j)
            rest = [c ^ (symplectic(c, b) * a) ^ (symplectic(c, a) * b) for c in rest]
            pairs.append((a, b))
        return pairs

    def _check_logicals(self) -> None:
        if len(self.logicals) != self.k:
            raise ValueError(f"expected {self.k} logical pairs, got {len(self.logicals)}")
        S = self.stabiliser_matrix
        L = [self.to_vec(p) for pair in self.logicals for p in pair]
        for i, a in enumerate(L):
            if any(symplectic(a, s) for s in S):
                raise ValueError(f"logical {i // 2}{'ab'[i % 2]} anticommutes with a stabiliser")
            for j, b in enumerate(L):
                if symplectic(a, b) != (i // 2 == j // 2 and i != j):
                    raise ValueError("logicals are not a symplectic basis")

    def __repr__(self) -> str:
        return f"Code(n={self.n}, k={self.k}, m={len(self.stabilisers)})"


def _clean(p: Mapping[Qubit, str]) -> Pauli:
    for c in p.values():
        if c not in _BITS:
            raise ValueError(f"unknown Pauli {c!r}")
    return {q: c for q, c in p.items() if c != "I"}


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------

def rotated_surface_code(
    dx: int,
    dy: int | None = None,
    rotation: int = 0,
    origin: tuple[float, float] = (0, 0),
) -> Code:
    """Rotated surface code on a ``dx`` x ``dy`` grid of data qubits ``(x, y)``.

    Before rotation, the patch has ``0 <= x < dx``, ``0 <= y < dy``. Plaquette
    ``(x, y)`` for ``-1 <= x < dx``, ``-1 <= y < dy`` acts on the data qubits at
    its corners ``(x..x+1, y..y+1)``, with type X if ``x + y`` is even and Z
    otherwise. Weight-2 X plaquettes sit on the top/bottom boundaries and
    weight-2 Z plaquettes on the left/right. Logical X runs down column 0
    (weight ``dy``), logical Z along row 0 (weight ``dx``).

    Each stabiliser is positioned at its plaquette centre ``(x + 0.5, y + 0.5)``,
    weight-2 ones included, so positions stay consistent under tiling.

    ``rotation`` (a multiple of 90 degrees) turns the patch a quarter turn at a
    time via ``(x, y) -> (h - 1 - y, x)``, where ``h`` is the current height, so
    the rotated patch again starts at ``(0, 0)``. Finally everything is shifted
    by ``origin``; since the checkerboard is fixed before the shift, patches
    placed at even offsets agree on plaquette types wherever they overlap.
    """
    dy = dx if dy is None else dy
    if dx < 2 or dy < 2:
        raise ValueError("dx and dy must be at least 2")
    if rotation % 90:
        raise ValueError("rotation must be a multiple of 90")
    stabilisers, positions = [], []
    for y in range(-1, dy):
        for x in range(-1, dx):
            support = [(x + ex, y + ey) for ey in (0, 1) for ex in (0, 1)
                       if 0 <= x + ex < dx and 0 <= y + ey < dy]
            p = "X" if (x + y) % 2 == 0 else "Z"
            on_tb = y in (-1, dy - 1)
            if len(support) == 4 or (len(support) == 2 and (p == "X") == on_tb):
                stabilisers.append({q: p for q in support})
                positions.append((x + 0.5, y + 0.5))
    logicals: list[tuple[Pauli, Pauli]] = [
        ({(0, y): "X" for y in range(dy)}, {(x, 0): "Z" for x in range(dx)})
    ]

    turns = (rotation // 90) % 4

    def place(pt):
        x, y = pt
        w, h = dx, dy
        for _ in range(turns):
            x, y = h - 1 - y, x
            w, h = h, w
        return (x + origin[0], y + origin[1])

    qubits = sorted((place((x, y)) for x in range(dx) for y in range(dy)), key=lambda q: (q[1], q[0]))
    stabilisers = [{place(q): c for q, c in s.items()} for s in stabilisers]
    logicals = [tuple({place(q): c for q, c in op.items()} for op in pair) for pair in logicals]
    positions = [place(p) for p in positions]
    return Code(stabilisers, qubits, logicals, positions)


def disjoint_union(*codes: Code) -> Code:
    """Codes on disjoint qubits side by side: stabilisers, logicals and
    positions concatenated."""
    qubits = [q for c in codes for q in c.qubits]
    if len(set(qubits)) != len(qubits):
        raise ValueError("codes share qubits")
    positions = None
    if all(c.stabiliser_positions is not None for c in codes):
        positions = [p for c in codes for p in c.stabiliser_positions]
    return Code([s for c in codes for s in c.stabilisers], qubits,
                [pair for c in codes for pair in c.logicals], positions)
