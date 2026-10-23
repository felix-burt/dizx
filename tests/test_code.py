from itertools import combinations, product

import numpy as np
import pytest

from dizx.code import Code, rotated_surface_code, symplectic


def min_logical_weight(code, max_w):
    S = code.stabiliser_matrix
    L = [code.to_vec(p) for pair in code.logicals for p in pair]
    for w in range(1, max_w + 1):
        for qs in combinations(code.qubits, w):
            for ps in product("XYZ", repeat=w):
                v = code.to_vec(dict(zip(qs, ps)))
                if not any(symplectic(v, s) for s in S) and any(symplectic(v, l) for l in L):
                    return w
    return None


@pytest.mark.parametrize("d", [2, 3, 4, 5, 7])
def test_rotated_surface_code_parameters(d):
    code = rotated_surface_code(d)
    assert (code.n, code.k, len(code.stabilisers)) == (d * d, 1, d * d - 1)
    assert code.is_css()
    weights = sorted(len(s) for s in code.stabilisers)
    assert weights.count(2) == 2 * (d - 1) and weights.count(4) == (d - 1) ** 2


@pytest.mark.parametrize("d", [2, 3])
def test_rotated_surface_code_distance(d):
    assert min_logical_weight(rotated_surface_code(d), d) == d


def test_logicals_found_when_not_given():
    given = rotated_surface_code(3)
    code = Code(given.stabilisers, given.qubits)
    (a, b), = code.logicals
    va, vb = code.to_vec(a), code.to_vec(b)
    assert symplectic(va, vb) == 1
    assert not any(symplectic(v, s) for v in (va, vb) for s in code.stabiliser_matrix)


def test_five_qubit_code_logicals():
    gens = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    code = Code([{i: c for i, c in enumerate(g) if c != "I"} for g in gens], qubits=range(5))
    assert code.k == 1 and not code.is_css()
    assert min_logical_weight(code, 3) == 3


def test_rejects_anticommuting_stabilisers():
    with pytest.raises(ValueError, match="commute"):
        Code([{0: "X"}, {0: "Z"}])


def test_rejects_dependent_stabilisers():
    with pytest.raises(ValueError, match="independent"):
        Code([{0: "Z", 1: "Z"}, {1: "Z", 2: "Z"}, {0: "Z", 2: "Z"}])


def test_rejects_bad_logicals():
    with pytest.raises(ValueError, match="anticommutes"):
        Code([{0: "Z", 1: "Z"}], logicals=[({0: "X"}, {0: "Z"})])
    with pytest.raises(ValueError, match="expected 1"):
        Code([{0: "Z", 1: "Z"}], logicals=[])


def test_rotated_surface_code_stabiliser_positions():
    code = rotated_surface_code(3)
    for s, (px, py) in zip(code.stabilisers, code.stabiliser_positions):
        # Every plaquette sits at the centre of its full 2x2 cell, even if truncated.
        assert all(abs(qx - px) == 0.5 and abs(qy - py) == 0.5 for qx, qy in s)
    assert (1.5, -0.5) in code.stabiliser_positions  # the top-boundary weight-2 X check


def test_stabiliser_positions_length_checked():
    with pytest.raises(ValueError, match="one position per stabiliser"):
        Code([{0: "Z", 1: "Z"}], stabiliser_positions=[])


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_patch(rotation):
    code = rotated_surface_code(3, rotation=rotation)
    assert (code.n, code.k, len(code.stabilisers)) == (9, 1, 8)
    assert set(code.qubits) == {(x, y) for x in range(3) for y in range(3)}
    assert min_logical_weight(code, 3) == 3
    x_bar = sorted(code.logicals[0][0])
    vertical = len({x for x, _ in x_bar}) == 1
    assert vertical == (rotation % 180 == 0)
    for s, (px, py) in zip(code.stabilisers, code.stabiliser_positions):
        assert all(abs(qx - px) == 0.5 and abs(qy - py) == 0.5 for qx, qy in s)


@pytest.mark.parametrize("dx,dy", [(2, 3), (3, 2), (3, 4)])
@pytest.mark.parametrize("rotation", [0, 90])
def test_asymmetric_patch(dx, dy, rotation):
    code = rotated_surface_code(dx, dy, rotation=rotation)
    assert (code.n, code.k, len(code.stabilisers)) == (dx * dy, 1, dx * dy - 1)
    w, h = (dy, dx) if rotation == 90 else (dx, dy)
    assert set(code.qubits) == {(x, y) for x in range(w) for y in range(h)}
    x_bar, z_bar = code.logicals[0]
    assert (len(x_bar), len(z_bar)) == (dy, dx)
    assert min_logical_weight(code, min(dx, dy)) == min(dx, dy)


def test_origin_shifts_everything():
    a, b = rotated_surface_code(3), rotated_surface_code(3, origin=(4, 0))
    shift = lambda q: (q[0] + 4, q[1])
    assert b.qubits == tuple(map(shift, a.qubits))
    assert b.stabilisers == [{shift(q): c for q, c in s.items()} for s in a.stabilisers]
    assert b.stabiliser_positions == [shift(p) for p in a.stabiliser_positions]
