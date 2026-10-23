import networkx as nx
import numpy as np
import pytest

from dizx._f2 import rank
from dizx.zx_diagram import (
    NodeType, Outcome, PauliWeb, SEDiagram, WebKind, ZXDiagram,
)


def diagram(edges, types, **kw):
    g = nx.MultiGraph()
    g.add_nodes_from(types)
    g.add_edges_from(edges)
    return ZXDiagram(g, types, **kw)


def dim(d):
    return rank(np.array([w.vec for w in d.web_space()]).reshape(-1, 2 * d.n_edges))


# -- structure ---------------------------------------------------------------

def test_degree_rules():
    with pytest.raises(ValueError, match="H node"):
        diagram([("h", "a")], {"h": "H", "a": "Z"})
    with pytest.raises(ValueError, match="port"):
        diagram([("p", "a"), ("p", "b")], {"p": "P", "a": "Z", "b": "Z"}, outputs=["p"])


def test_ports_must_be_partitioned():
    with pytest.raises(ValueError, match="partition"):
        diagram([("p", "a")], {"p": "P", "a": "Z"})


def test_phases_only_on_spiders():
    with pytest.raises(ValueError, match="non-spider"):
        diagram([("p", "a")], {"p": "P", "a": "Z"}, phases={"p": 1}, outputs=["p"])


def test_edge_index_either_orientation():
    d = diagram([("a", "b"), ("a", "b")], {"a": "Z", "b": "X"})
    assert d.edge_index(("a", "b", 1)) == d.edge_index(("b", "a", 1))
    assert d.edge_index(("a", "b", 0)) != d.edge_index(("a", "b", 1))


def test_self_loop_counts_twice():
    d = diagram([("a", "a"), ("a", "p")], {"a": "Z", "p": "P"}, outputs=["p"])
    assert len(d.legs("a")) == 3


# -- webs --------------------------------------------------------------------

def test_single_z_spider_state():
    # |+> = Z spider with one output: stabiliser X, and Z is not allowed.
    d = diagram([("a", "p")], {"a": "Z", "p": "P"}, outputs=["p"])
    (w,) = d.web_space()
    assert w.boundary() == ("", "X")
    assert w.kind() is WebKind.STABILISER
    assert not PauliWeb.from_paulis(d, {("a", "p", 0): "Z"}).is_valid()


def test_bell_pair():
    # Z spider with two outputs: stabilisers XX and ZZ.
    d = diagram([("a", "p"), ("a", "q")], {"a": "Z", "p": "P", "q": "P"}, outputs=["p", "q"])
    assert dim(d) == 2
    for paulis in ("XX", "ZZ", "YY"):
        w = PauliWeb.from_paulis(d, {("a", "p", 0): paulis[0], ("a", "q", 0): paulis[1]})
        assert w.is_valid() and w.kind() is WebKind.STABILISER
    assert PauliWeb.from_paulis(d, {("a", "p", 0): "X", ("a", "q", 0): "Z"}).violations() == ["a"]


def test_hadamard_wire_maps_x_to_z():
    d = diagram([("i", "h"), ("h", "o")], {"i": "P", "h": "H", "o": "P"}, inputs=["i"], outputs=["o"])
    assert dim(d) == 2
    w = PauliWeb.from_paulis(d, {("i", "h", 0): "X", ("h", "o", 0): "Z"})
    assert w.is_valid() and w.boundary() == ("X", "Z") and w.kind() is WebKind.LOGICAL
    w = PauliWeb.from_paulis(d, {("i", "h", 0): "Y", ("h", "o", 0): "Y"})
    assert w.is_valid()


def test_web_addition():
    d = diagram([("a", "p"), ("a", "q")], {"a": "Z", "p": "P", "q": "P"}, outputs=["p", "q"])
    xx = PauliWeb.from_paulis(d, {("a", "p", 0): "X", ("a", "q", 0): "X"})
    zz = PauliWeb.from_paulis(d, {("a", "p", 0): "Z", ("a", "q", 0): "Z"})
    assert (xx + zz).boundary() == ("", "YY")
    assert (xx + xx).kind() is WebKind.DETECTOR and not (xx + xx).vec.any()


def test_hopf_parallel_edges():
    # Z and X spiders joined by two parallel edges: closed webs where both
    # edges carry the same Pauli (X, Z or Y), and nothing else.
    d = diagram([("a", "b"), ("a", "b")], {"a": "Z", "b": "X"})
    assert dim(d) == 2
    assert PauliWeb.from_paulis(d, {("a", "b", 0): "X", ("a", "b", 1): "X"}).is_valid()
    assert not PauliWeb.from_paulis(d, {("a", "b", 0): "X"}).is_valid()


def measured_repetition_code():
    """Two rounds of a ZZ measurement on qubits 0, 1 (X-spider ancilla readouts).

    Each round: Z spiders z{r}{q} on the data lines, joined to an X spider
    m{r} carrying a classical output c{r}.
    """
    types = {"i0": "P", "i1": "P", "o0": "P", "o1": "P"}
    edges, phases = [], {}
    for r in range(2):
        types |= {f"z{r}0": "Z", f"z{r}1": "Z", f"m{r}": "X", f"c{r}": "X"}
        phases[f"c{r}"] = Outcome(r)
        edges += [(f"z{r}0", f"m{r}"), (f"z{r}1", f"m{r}"), (f"m{r}", f"c{r}")]
    edges += [("i0", "z00"), ("z00", "z10"), ("z10", "o0"),
              ("i1", "z01"), ("z01", "z11"), ("z11", "o1")]
    g = nx.MultiGraph()
    g.add_nodes_from(types)
    g.add_edges_from(edges)
    return ZXDiagram(g, types, phases=phases, inputs=["i0", "i1"], outputs=["o0", "o1"])


# The product of the two ZZ outcomes.
DET = {
    ("m0", "c0", 0): "Z", ("z00", "m0", 0): "Z", ("z01", "m0", 0): "Z",
    ("z00", "z10", 0): "Z", ("z01", "z11", 0): "Z",
    ("z10", "m1", 0): "Z", ("z11", "m1", 0): "Z", ("m1", "c1", 0): "Z",
}
def test_repetition_code_detector():
    d = measured_repetition_code()
    assert set(d.classical_outputs()) == {"c0", "c1"}
    det = PauliWeb.from_paulis(d, DET)
    assert det.is_valid(), det.violations()
    assert det.kind() is WebKind.DETECTOR
    assert set(det.outcomes()) == {"c0", "c1"}
    assert det.outcome_parity({0: 1, 1: 1}) == 0
    assert det.outcome_parity({0: 1, 1: 0}) == 1
    (only,) = d.detector_space()
    assert only == det


# -- SEDiagram --------------------------------------------------------------

from dizx.code import Code, rotated_surface_code  # noqa: E402

REP = Code([{0: "Z", 1: "Z"}], qubits=[0, 1])


def gens_touched(w):
    """Generator spiders highlighted in their own colour (other generators the
    web passes see an even number of legs in the other colour)."""
    d = w.diagram
    own = {NodeType.X: "ZY", NodeType.Z: "XY"}
    return {g for g, t in d.types.items()
            if g[0] == "gen" and all(w.pauli(i) in own[t] for i in d.legs(g))}


def test_se_diagram_is_phase_free():
    with pytest.raises(ValueError, match="phase-free"):
        SEDiagram.from_diagram(measured_repetition_code())


def test_se_diagram_classes():
    se = SEDiagram.from_code(REP, 2)
    assert [len(x) for x in (se.detectors, se.stabilisers, se.co_stabilisers, se.logicals)] == [1, 1, 1, 2]
    assert se.stabilisers[0].boundary() == ("II", "ZZ")
    assert se.co_stabilisers[0].boundary() == ("ZZ", "II")
    assert se.detectors_complete() and se.webs_complete()
    assert all(w.diagram is se for w in se.webs)
    assert not se.classical_outputs()


def test_se_diagram_adopts_webs_from_other_diagram():
    se = SEDiagram.from_code(REP, 2)
    copy = SEDiagram.from_diagram(se, detectors=se.detectors)
    assert copy.detectors[0].diagram is copy
    assert copy.detectors[0].support() == se.detectors[0].support()


def test_se_diagram_rejects_wrong_class():
    se = SEDiagram.from_code(REP, 2)
    with pytest.raises(ValueError, match="detector 0 is a stabiliser"):
        SEDiagram.from_diagram(se, detectors=se.stabilisers)


def test_se_diagram_rejects_invalid_web():
    se = SEDiagram.from_code(REP, 2)
    with pytest.raises(ValueError, match="invalid"):
        SEDiagram.from_diagram(se, detectors=[{se.edges[0]: "Z"}])


def test_se_diagram_rejects_dependent_webs():
    se = SEDiagram.from_code(REP, 2)
    with pytest.raises(ValueError, match="independent"):
        SEDiagram.from_diagram(se, logicals=[se.logicals[0]] * 2)


def test_se_diagram_incomplete_detectors():
    se = SEDiagram.from_code(REP, 2)
    assert not SEDiagram.from_diagram(se).detectors_complete()


# -- SEDiagram.from_code -----------------------------------------------------

@pytest.mark.parametrize("d,rounds", [(2, 1), (3, 1), (3, 3), (5, 2)])
def test_from_code_rotated_surface_code(d, rounds):
    code = rotated_surface_code(d)
    se = SEDiagram.from_code(code, rounds)
    m = len(code.stabilisers)
    assert len(se.detectors) == (rounds - 1) * m
    assert len(se.stabilisers) == len(se.co_stabilisers) == m and len(se.logicals) == 2
    assert se.detectors_complete() and se.webs_complete()
    assert all(a == 0 for a in se.phases.values())
    # Logical X on column 0 passes straight from inputs to outputs.
    p_in, p_out = se.logicals[0].boundary()
    assert p_in == p_out
    assert {q for q, c in zip(code.qubits, p_in) if c != "I"} == {(0, y) for y in range(d)}


def test_from_code_detectors_are_local():
    code = rotated_surface_code(3)
    se = SEDiagram.from_code(code, 3)
    m = len(code.stabilisers)
    for k, det in enumerate(se.detectors):
        r, i = 1 + k // m, k % m
        assert gens_touched(det) == {("gen", r - 1, i), ("gen", r, i)}


def test_from_code_stabiliser_boundaries_match_code():
    code = rotated_surface_code(3)
    se = SEDiagram.from_code(code, 2)
    for s, w, cw in zip(code.stabilisers, se.stabilisers, se.co_stabilisers):
        expected = "".join(s.get(q, "I") for q in code.qubits)
        assert w.boundary() == ("I" * code.n, expected)
        assert cw.boundary() == (expected, "I" * code.n)
        assert gens_touched(w) == {("gen", 1, code.stabilisers.index(s))}
        assert gens_touched(cw) == {("gen", 0, code.stabilisers.index(s))}


def test_from_code_fuses_single_type_qubits():
    # A repetition code has Z generators only: one data spider per qubit for all rounds.
    code = Code([{0: "Z", 1: "Z"}, {1: "Z", 2: "Z"}], qubits=range(3))
    se = SEDiagram.from_code(code, 3)
    assert sorted(v for v in se.types if v[0] == "data") == [("data", 0, "Z", q) for q in range(3)]
    assert se.detectors_complete() and se.webs_complete()


def test_from_code_mixed_logicals():
    # [[4,2,2]] with a symplectic basis containing Y logicals, giving Y edges.
    logicals = [({0: "Y", 1: "Y"}, {0: "Z", 2: "Z"}), ({0: "Y", 2: "Y"}, {0: "Z", 1: "Z"})]
    code = Code([{q: "X" for q in range(4)}, {q: "Z" for q in range(4)}], range(4), logicals)
    se = SEDiagram.from_code(code, 2)
    assert len(se.logicals) == 4 and se.detectors_complete() and se.webs_complete()
    assert "Y" in se.logicals[0].support().values()
    for w, op in zip(se.logicals, [p for pair in code.logicals for p in pair]):
        p_in, p_out = w.boundary()
        assert p_in == p_out == "".join(op.get(q, "I") for q in code.qubits)


def test_from_code_rejects_non_css():
    gens = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]
    code = Code([{i: c for i, c in enumerate(g) if c != "I"} for g in gens], qubits=range(5))
    with pytest.raises(ValueError, match="CSS"):
        SEDiagram.from_code(code, 1)


def test_layout_uses_stabiliser_positions():
    code = rotated_surface_code(3)
    se = SEDiagram.from_code(code, 2)
    for i, (px, py) in enumerate(code.stabiliser_positions):
        for r in range(2):
            assert se.positions[("gen", r, i)][:2] == (px, py)
