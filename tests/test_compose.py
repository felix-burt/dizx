import numpy as np
import pytest

from dizx.code import Code, rotated_surface_code
from dizx.zx_diagram import NodeType, SEDiagram, close_ports, compose, relabel

REP = Code([{0: "Z", 1: "Z"}, {1: "Z", 2: "Z"}], qubits=range(3))
C422_Y = Code([{q: "X" for q in range(4)}, {q: "Z" for q in range(4)}], range(4),
              [({0: "Y", 1: "Y"}, {0: "Z", 2: "Z"}), ({0: "Y", 2: "Y"}, {0: "Z", 1: "Z"})])


def edge_set(sed):
    return sorted(sorted(map(repr, (u, v))) for u, v, _ in sed.edges)


def supports(webs):
    return [{frozenset((u, v)): p for (u, v, _), p in w.support().items()} for w in webs]


@pytest.mark.parametrize("code,rounds", [
    (rotated_surface_code(3), 1), (rotated_surface_code(3), 3), (rotated_surface_code(3, rotation=90), 2),
    (REP, 3), (C422_Y, 2),
])
def test_composed_matches_direct(code, rounds):
    composed = SEDiagram.from_code(code, rounds)
    direct = SEDiagram._from_code_unrolled(code, rounds)
    assert composed.types == direct.types
    assert composed.inputs == direct.inputs and composed.outputs == direct.outputs
    assert edge_set(composed) == edge_set(direct)
    for fam in ("detectors", "stabilisers", "co_stabilisers", "logicals"):
        assert supports(getattr(composed, fam)) == supports(getattr(direct, fam)), fam
    if direct.positions:
        assert composed.positions.keys() == direct.positions.keys()
        for v, pos in direct.positions.items():
            assert np.allclose(composed.positions[v], pos), v


def test_repetition_record():
    sed = SEDiagram.from_code(rotated_surface_code(3), 3)
    rep = sed.repetition
    assert len(rep.copies) == 3
    for r, copy in enumerate(rep.copies):
        for v, u in copy.items():
            assert u == (v[0], r, *v[2:])
    # Every spider of the composite is the image of exactly one body spider.
    images = [u for copy in rep.copies for u in copy.values()]
    assert sorted(map(repr, images)) == sorted(repr(v) for v, t in sed.types.items() if t.value in "ZX")


def test_repetition_record_with_fusion():
    # Repetition-code data spiders fuse across rounds: every copy maps to round 0.
    rep = SEDiagram.from_code(REP, 3).repetition
    for copy in rep.copies:
        for q in range(3):
            assert copy[("data", 0, "Z", q)] == ("data", 0, "Z", q)


def test_compose_without_fusion():
    body = SEDiagram._from_code_unrolled(REP, 1)
    two, rename = compose(body, relabel(body, lambda v: ("b", v)), fuse=False)
    assert rename == {}
    assert len(two.detectors) == len(REP.stabilisers)
    assert two.webs_complete() and two.detectors_complete()


def test_compose_mismatched_logicals_uses_combinations():
    # d2's logical Z is Z0 times the stabiliser Z0Z1: no single d2 web matches
    # d1's Z0, so the logical must be found as a combination.
    a = Code([{0: "Z", 1: "Z"}], [0, 1], [({0: "X", 1: "X"}, {0: "Z"})])
    b = Code([{0: "Z", 1: "Z"}], [0, 1], [({0: "X", 1: "X"}, {1: "Z"})])
    d1 = SEDiagram._from_code_unrolled(a, 1)
    d2 = relabel(SEDiagram._from_code_unrolled(b, 1), lambda v: ("b", v))
    sed, _ = compose(d1, d2)
    assert len(sed.logicals) == 2 and sed.webs_complete() and sed.detectors_complete()


def test_compose_checks():
    body = SEDiagram._from_code_unrolled(REP, 1)
    with pytest.raises(ValueError, match="share node labels"):
        compose(body, body)
    small = SEDiagram._from_code_unrolled(Code([{0: "Z", 1: "Z"}], [0, 1]), 1)
    with pytest.raises(ValueError, match="as many outputs"):
        compose(body, relabel(small, lambda v: ("b", v)))


@pytest.mark.parametrize("first, last", [("Z", "X"), ("Z", "Z"), ("X", "X"), ("X", "Z")])
def test_from_code_first_last(first, last):
    code = rotated_surface_code(3)
    sed = SEDiagram.from_code(code, 2, first_round=1, first=first, last=last)
    sed._check_webs()
    assert sed.webs_complete() and sed.detectors_complete()
    # Layer sequence: full rounds are Z then X; partial rounds add one layer.
    layers = sorted({(g[1], "X" if sed.types[g] is NodeType.Z else "Z") for g in sed.types if g[0] == "gen"},
                    key=lambda t: (t[0], t[1] == "X"))
    seq = "".join(p for _, p in layers)
    assert seq[0] == first and seq[-1] == last and len(seq) == 4 + (first == "X") + (last == "Z")
    assert all(a != b for a, b in zip(seq, seq[1:]))
    assert layers[0][0] == 1


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_close_ports(basis):
    """Closing every port in a basis: removed where the end spider already has
    the closing colour (Z for X), an explicit spider otherwise."""
    code = rotated_surface_code(3)
    sed = SEDiagram.from_code(code, 2)  # wires start at Z spiders, end at X spiders
    out = close_ports(sed, {p: basis for p in sed.inputs + sed.outputs})
    assert not out.inputs and not out.outputs and len(out.closed) == 2 * 9
    colour = NodeType.Z if basis == "X" else NodeType.X
    for p, v in out.closed.items():
        assert out.types[v] is colour
        explicit = v[0] in ("init", "proj")
        natural = NodeType.Z if p[0] == "in" else NodeType.X
        assert explicit == (natural is not colour)
    out._check_webs()
    assert out.webs_complete() and out.detectors_complete()
    # No ports: every web is a detecting region, including the closed logical.
    assert not out.stabilisers and not out.co_stabilisers and not out.logicals
    assert len(out.detectors) == len(out.web_space())


@pytest.mark.parametrize("first,last", [("Z", "X"), ("X", "Z"), ("X", "X"), ("Z", "Z")])
def test_partial_rounds_keep_stabilisers(first, last):
    """Stabilisers carried through a partial round (S -> S) combine into
    stabiliser webs, not logicals."""
    code = rotated_surface_code(3)
    sed = SEDiagram.from_code(code, 2, first=first, last=last)
    assert len(sed.stabilisers) == len(sed.co_stabilisers) == len(code.stabilisers)
    assert len(sed.logicals) == 2 * code.k and not sed.carried
    assert sed.webs_complete() and sed.detectors_complete()
