import networkx as nx
import numpy as np
import pytest

from dizx.code import rotated_surface_code
import dizx.transformations as T
from dizx.transformations import (
    add_parallel_edge, cancel_hadamards, cut_nodes, dualise, fuse, insert_identity,
    loop_edge, loop_generator, remove_identity, split, split_boundary,
)
from dizx.zx_diagram import NodeType, SEDiagram, ZXDiagram
from helpers import has_matching_boundary


@pytest.fixture
def sed():
    return SEDiagram.from_code(rotated_surface_code(3), 2)


def assert_preserved(a, b):
    """Same families, same boundary operators, still a complete generating set."""
    for fa, fb in [(a.detectors, b.detectors), (a.stabilisers, b.stabilisers),
                   (a.co_stabilisers, b.co_stabilisers), (a.logicals, b.logicals)]:
        assert [w.boundary() for w in fa] == [w.boundary() for w in fb]
    assert len(b.web_space()) == len(a.web_space())
    assert b.webs_complete() and b.detectors_complete()


def assert_same_shape(a, b):
    assert set(a.types) == set(b.types) and a.types == b.types
    ga = nx.MultiGraph(a.graph)
    gb = nx.MultiGraph(b.graph)
    assert sorted(map(sorted, map(lambda e: map(repr, e), ga.edges()))) == \
        sorted(map(sorted, map(lambda e: map(repr, e), gb.edges())))


DATA = ("data", 0, "Z", (0, 0))
GEN = next(v for v in SEDiagram.from_code(rotated_surface_code(3), 2).graph.neighbors(DATA)
           if v[0] == "gen")


@pytest.mark.parametrize("v", [GEN, DATA, ("data", 1, "X", (1, 1))])
def test_duality(sed, v):
    new = dualise(sed, [v])
    assert new.types[v] is not sed.types[v]
    assert sum(t is NodeType.H for t in new.types.values()) == sed.graph.degree(v)
    assert_preserved(sed, new)


def test_duality_twice_then_cancel_restores(sed):
    new = dualise(dualise(sed, [GEN]), [GEN])
    for h in list(new.graph.neighbors(GEN)):  # inner H of each H-H pair
        (h2,) = [x for x in new.graph.neighbors(h) if x != GEN]
        new = cancel_hadamards(new, h, h2)
    assert_same_shape(sed, new)
    assert_preserved(sed, new)


@pytest.mark.parametrize("colour", ["Z", "X"])
@pytest.mark.parametrize("e", [(("in", (0, 0)), DATA, 0), (DATA, GEN, 0)])
def test_insert_and_remove_identity(sed, e, colour):
    new = insert_identity(sed, e, colour)
    assert len(new.types) == len(sed.types) + 1
    assert_preserved(sed, new)
    (s,) = set(new.types) - set(sed.types)
    back = remove_identity(new, s)
    assert_same_shape(sed, back)
    assert_preserved(sed, back)


@pytest.mark.parametrize("v", [GEN, ("gen", 1, 3), DATA])
def test_split_and_fuse(sed, v):
    legs = list(sed.graph.edges(v, keys=True))
    for cut in range(len(legs) + 1):
        new = split(sed, v, legs[:cut])
        assert new.graph.degree(v) == cut + 1
        assert_preserved(sed, new)
        back = fuse(new, v, ("split", v))
        assert_same_shape(sed, back)
        assert_preserved(sed, back)


def test_double_edge_split_adds_a_web(sed):
    # Non-example: joining the two halves by a second edge adds a loop web.
    new = split(sed, GEN, list(sed.graph.edges(GEN, keys=True))[:2])
    g = nx.MultiGraph(new.graph)
    g.add_edge(GEN, ("split", GEN))
    doubled = ZXDiagram(g, new.types, inputs=new.inputs, outputs=new.outputs)
    assert len(doubled.web_space()) == len(new.web_space()) + 1


def test_split_boundary_gives_matching_boundary(sed):
    q = (1, 1)
    data = {v for v in sed.types if v[0] == "data" and v[-1] == q}
    S = data | {g for v in data for g in sed.graph.neighbors(v) if g[0] == "gen"}
    assert not has_matching_boundary(sed, S)
    new = split_boundary(sed, S)
    assert has_matching_boundary(new, S)
    assert cut_nodes(new, S) == cut_nodes(sed, S)
    assert_preserved(sed, new)


def test_rules_reject_bad_targets(sed):
    with pytest.raises(ValueError, match="not a spider"):
        dualise(sed, [("in", (0, 0))])
    with pytest.raises(ValueError, match="legs"):
        split(sed, GEN, [(("in", (0, 0)), DATA, 0)])
    with pytest.raises(ValueError, match="exactly one edge"):
        fuse(sed, DATA, ("data", 0, "Z", (1, 1)))
    with pytest.raises(ValueError, match="degree 2"):
        remove_identity(sed, DATA)


def weight4_generator(sed):
    return next(v for v in sed.types if v[0] == "gen" and sed.graph.degree(v) == 4)


def test_add_parallel_edge_adds_loop_web(sed):
    g = weight4_generator(sed)
    legs = list(sed.graph.edges(g, keys=True))
    s1 = split(sed, g, legs[:2])
    s2 = add_parallel_edge(s1, g, ("split", g))
    assert s2.graph.number_of_edges(g, ("split", g)) == 2
    # Old webs keep their boundaries; one new detector, the loop web.
    for fa, fb in [(s1.detectors, s2.detectors[:-1]), (s1.stabilisers, s2.stabilisers),
                   (s1.co_stabilisers, s2.co_stabilisers), (s1.logicals, s2.logicals)]:
        assert [w.boundary() for w in fa] == [w.boundary() for w in fb]
    assert len(s2.web_space()) == len(s1.web_space()) + 1
    assert s2.webs_complete() and s2.detectors_complete()
    loop = s2.detectors[-1]
    # The spiders' own colour: opposite to the generator's stabiliser type.
    assert set(loop.support().values()) == {sed.types[g].value}
    assert {frozenset(e[:2]) for e in loop.support()} == {frozenset((g, ("split", g)))}
    with pytest.raises(ValueError):
        add_parallel_edge(s2, g, ("split", g))  # already two edges


def test_loop_generator(sed):
    g = weight4_generator(sed)
    data = set(sed.graph.neighbors(g))
    out = loop_generator(sed, g)
    ring = [g] + [("loop", g, k) for k in (1, 2, 3)]
    assert len(out.detectors) == len(sed.detectors) + 1
    assert out.webs_complete() and out.detectors_complete()
    # One data leg per spider, and the four spiders form a cycle.
    assert {u for v in ring for u in out.graph.neighbors(v) if u not in ring} == data
    for v in ring:
        assert out.types[v] is sed.types[g] and out.graph.degree(v) == 3
        assert sum(u in ring for u in out.graph.neighbors(v)) == 2
    assert nx.is_connected(out.graph.subgraph(ring)) and out.graph.subgraph(ring).number_of_edges() == 4
    # The loop web runs once around the cycle in the non-matching colour.
    loop = out.detectors[-1]
    assert {frozenset(e[:2]) for e in loop.support()} == {frozenset(e[:2]) for e in out.graph.subgraph(ring).edges}
    # Data legs follow the cyclic order around the plaquette.
    corner = {v: next(u for u in out.graph.neighbors(v) if u not in ring) for v in ring}
    pos = lambda v: sed.positions[corner[v]][:2]
    for u, v in out.graph.subgraph(ring).edges():
        (x0, y0), (x1, y1) = pos(u), pos(v)
        assert abs(x0 - x1) + abs(y0 - y1) == 1  # neighbouring corners, not diagonal


def test_loop_edge_plain(sed):
    g = weight4_generator(sed)
    legs = list(sed.graph.edges(g, keys=True))
    s1 = split(sed, g, legs[:2])
    h = ("split", g)
    s2 = loop_edge(s1, g, h)
    ring = [g, h, ("loop", h), ("loop", g)]
    assert len(s2.detectors) == len(s1.detectors) + 1
    assert s2.webs_complete() and s2.detectors_complete()
    for v in ring:  # each spider keeps one data leg
        assert sum(u not in ring for u in s2.graph.neighbors(v)) == 1
    assert s2.graph.subgraph(ring).number_of_edges() == 4


def test_loop_edge_through_hadamard(sed):
    g = weight4_generator(sed)
    legs = list(sed.graph.edges(g, keys=True))
    s1 = dualise(split(sed, g, legs[:2]), [("split", g)])  # g -H- ("split", g)
    h = ("split", g)
    assert s1.types[g] is not s1.types[h]
    s2 = loop_edge(s1, g, h)
    assert len(s2.detectors) == len(s1.detectors) + 1
    assert s2.webs_complete() and s2.detectors_complete()
    # Two H nodes cross between {g, ("loop", g)} and {h, ("loop", h)}.
    hs = [x for x, t in s2.types.items() if t is NodeType.H and
          {*s2.graph.neighbors(x)} & {g, ("loop", g)} and {*s2.graph.neighbors(x)} & {h, ("loop", h)}]
    assert len(hs) == 2
    # The loop web: g's colour on g's side, h's colour on h's side.
    loop = s2.detectors[-1]
    for (a, b, _), p in loop.support().items():
        end = a if a in (g, ("loop", g), h, ("loop", h)) else b
        assert p == s2.types[end].value


def broken_identity(sed):
    """insert_identity with a wrong label on one half: invalid at the new spider."""
    e = next(e for e in sed.edges if sed.types[e[0]] is NodeType.Z and sed.types[e[1]] is NodeType.X)
    ed = T.SEDEditor(sed)
    s = ed.insert_identity(e, "Z")
    d = ed._on(next(f for f in ed.graph.edges(s, keys=True) if f[1] == e[1]))
    for wid, (x, z) in d.items():
        d[wid] = (x, 1 - z)
    return ed.freeze()


@pytest.mark.parametrize("mode", ["local", "full"])
def test_checks_catch_a_broken_rule(sed, mode):
    with T.checking(mode), pytest.raises(ValueError, match="invalid at nodes .*'id'"):
        broken_identity(sed)


def test_checks_off(sed):
    with T.checking("none"):
        out = broken_identity(sed)
    assert not all(w.is_valid() for w in out.webs)
    assert T.CHECK == "local"  # restored


@pytest.mark.parametrize("mode", ["local", "full", "none"])
def test_check_modes_agree(sed, mode):
    g = weight4_generator(sed)
    with T.checking(mode):
        out = loop_generator(sed, g)
    ref = loop_generator(sed, g)
    assert [w.vec.tolist() for w in out.webs] == [w.vec.tolist() for w in ref.webs]
    assert out.webs_complete() and out.detectors_complete()
    out._check_webs()  # full check passes


def assert_hadamards_straight(sed):
    """Every H node lies strictly between its two neighbours, on the line."""
    for h, t in sed.types.items():
        if t is NodeType.H:
            a, b = (np.asarray(sed.positions[x], float) for x in sed.graph.neighbors(h))
            p = np.asarray(sed.positions[h], float)
            s = np.dot(p - a, b - a) / np.dot(b - a, b - a)
            assert 0 < s < 1 and np.allclose(a + s * (b - a), p), h


def assert_layout(before, after, moved):
    """Only ``moved`` (the transformed nodes) and H nodes, which are edges,
    change position; no two nodes share a position; H edges stay straight."""
    for v in before.types:
        if v in after.types and v not in moved and after.types[v] is not NodeType.H:
            assert after.positions[v] == before.positions[v], v
    pts = [tuple(round(float(c), 9) for c in after.positions[v]) for v in after.types]
    assert len(set(pts)) == len(pts)
    assert_hadamards_straight(after)


def test_split_layout(sed):
    g = weight4_generator(sed)
    p = np.asarray(sed.positions[g], float)
    right = [e for e in sed.graph.edges(g, keys=True) if sed.positions[e[1]][0] > p[0]]
    out = split(sed, g, [e for e in sed.graph.edges(g, keys=True) if e not in right])
    a, b = (np.asarray(out.positions[n], float) for n in (g, ("split", g)))
    assert np.allclose((a + b) / 2, p)  # midpoint of the new edge is where g was
    assert np.isclose(np.linalg.norm(b - a), T.SPACING) and (b - a)[0] > 0  # towards B
    assert_layout(sed, out, {g})


def test_loop_generator_layout(sed):
    g = weight4_generator(sed)
    out = loop_generator(sed, g)
    ring = [g] + [("loop", g, k) for k in (1, 2, 3)]
    assert np.allclose(np.mean([out.positions[n] for n in ring], axis=0), sed.positions[g])
    assert_layout(sed, out, {g})
    for n in ring:  # each spider sits towards its own data leg
        (leg,) = [x for x in out.graph.neighbors(n) if x not in ring]
        d = np.subtract(out.positions[n], sed.positions[g])
        assert np.dot(d, np.subtract(sed.positions[leg], sed.positions[g])) > 0


def test_loop_edge_is_a_rectangle(sed):
    g = weight4_generator(sed)
    legs = list(sed.graph.edges(g, keys=True))
    s1 = dualise(split(sed, g, legs[:2]), [("split", g)])
    h = ("split", g)
    out = loop_edge(s1, g, h)
    P = {v: np.asarray(p, float) for v, p in out.positions.items()}
    ring = [g, h, ("loop", h), ("loop", g)]
    # Centred on the original generator; a rectangle SPACING wide.
    assert np.allclose(np.mean([P[n] for n in ring], axis=0), sed.positions[g])
    d1, d2 = P[("loop", g)] - P[g], P[("loop", h)] - P[h]
    assert np.allclose(d1, d2) and np.isclose(np.linalg.norm(d1), T.SPACING)
    assert np.isclose(np.dot(d1, P[h] - P[g]), 0)
    hs = [x for x, t in out.types.items() if t is NodeType.H and
          {*out.graph.neighbors(x)} & {g, ("loop", g)} and {*out.graph.neighbors(x)} & {h, ("loop", h)}]
    assert len(hs) == 2
    assert_layout(s1, out, set(ring) | set(hs))
    # Each spider keeps the external leg on its own side: legs do not cross.
    for a in (g, h):
        (ea,) = [x for x in out.graph.neighbors(a) if x not in ring and x not in hs]
        (eb,) = [x for x in out.graph.neighbors(("loop", a)) if x not in ring and x not in hs]
        assert np.dot(P[eb] - P[ea], d1) >= 0


def assert_short_way_round(sed, loops):
    """No web gets smaller support on a loop by adding that loop's web."""
    n = sed.n_edges
    for ell in loops:
        cols = np.nonzero(ell.vec[:n] | ell.vec[n:])[0]
        weight = lambda v: np.count_nonzero(v[cols] | v[n + cols])
        for w in sed.webs:
            if w is not ell:
                assert weight(w.vec) <= weight(w.vec ^ ell.vec)


def test_loop_generator_short_way_round(sed):
    out = loop_generator(sed, weight4_generator(sed))
    assert_short_way_round(out, out.detectors[-1:])
    out._check_webs()  # still valid, right classes, independent


# -- editor sessions ------------------------------------------------------------

def assert_identical(a, b):
    """Same SED exactly: edge order, node order, positions and web vectors."""
    assert a.edges == b.edges
    assert list(a.types.items()) == list(b.types.items())
    assert a.positions == b.positions
    for fa, fb in [(a.detectors, b.detectors), (a.stabilisers, b.stabilisers),
                   (a.co_stabilisers, b.co_stabilisers), (a.logicals, b.logicals)]:
        assert [w.idx.tolist() for w in fa] == [w.idx.tolist() for w in fb]
    assert a.carried == b.carried


def _sequence(sed, ed=None):
    """A mixed sequence of rules, applied one at a time (ed None) or to ``ed``."""
    def step(name, *args):
        nonlocal sed
        if ed is None:
            sed = getattr(T, name)(sed, *args)
        else:
            getattr(ed, name)(*args)

    def view():
        return sed if ed is None else ed

    step("dualise", [GEN])
    e = next(e for e in view().graph.edges(DATA, keys=True) if view().types[e[1]] is NodeType.H)
    step("insert_identity", e, "X")
    legs = list(view().graph.edges(DATA, keys=True))
    step("split", DATA, legs[:2])
    g4 = next(v for v in view().types if v[0] == "gen" and view().graph.degree(v) == 4
              and len(set(view().graph.neighbors(v))) == 4)
    step("loop_generator", g4)
    return sed if ed is None else ed.freeze()


def test_editor_session_equals_rules_one_at_a_time(sed):
    one_by_one = _sequence(sed)
    session = _sequence(sed, T.SEDEditor(sed))
    assert_identical(one_by_one, session)
    assert all(w.is_valid() for w in session.webs)


def test_editor_regions_equal_rules_one_at_a_time(sed):
    S = {v for v in sed.types if v[0] in ("data", "gen") and v[1] == 1}
    ref = sed
    for v in sorted(cut_nodes(ref, S), key=repr):
        ref = split(ref, v, [e for e in ref.graph.edges(v, keys=True) if e[1] in S])
    assert_identical(ref, split_boundary(sed, S))
