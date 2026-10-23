import pytest

from dizx.code import Code, rotated_surface_code
from dizx.realisation import RealisedSED
from dizx.zx_diagram import SEDiagram

REP = Code([{0: "Z", 1: "Z"}], qubits=[0, 1])


@pytest.mark.parametrize("d,rounds", [(3, 1), (3, 3), (5, 2)])
def test_circuit_based_rotated_surface_code(d, rounds):
    code = rotated_surface_code(d)
    sed = SEDiagram.from_code(code, rounds)
    real = RealisedSED.circuit_based(sed)
    m = len(code.stabilisers)

    # Data world-lines follow the qubit wires; generators are singletons.
    data = [p for p in real.worldlines if len(p) > 1]
    assert len(data) == code.n
    assert all(p[0] == ("in", q) and p[-1] == ("out", q) for p, q in zip(data, code.qubits))
    assert all(v[0] != "gen" for p in data for v in p)
    singles = [p for p in real.worldlines if len(p) == 1]
    assert {p[0] for p in singles} == set(real.classical) == {v for v in sed.types if v[0] == "gen"}

    # Webs carry over with the same classes and boundaries.
    for a, b in zip(sed.webs, real.webs):
        assert b.is_valid() and a.kind() is b.kind() and a.boundary() == b.boundary()

    # Detectors compare two consecutive outcomes of the same generator.
    for k, w in enumerate(real.detectors):
        r, i = 1 + k // m, k % m
        assert real.record(w) == {("gen", r - 1, i), ("gen", r, i)}
    # Stabilisers / co-stabilisers take their sign from the last / first round.
    for i, (s, cs) in enumerate(zip(real.stabilisers, real.co_stabilisers)):
        assert real.record(s) == {("gen", rounds - 1, i)}
        assert real.record(cs) == {("gen", 0, i)}
    # Logicals pass generators in the non-matching colour only: no outcomes.
    assert all(real.record(w) == set() for w in real.logicals)


def test_classical_outputs_added():
    sed = SEDiagram.from_code(REP, 2)
    real = RealisedSED.circuit_based(sed)
    assert set(real.classical_outputs()) == {("meas", ("gen", r, 0)) for r in range(2)}
    assert real.n_edges == sed.n_edges + 2
    assert not real.positions  # integer qubits: no layout


def test_positions_cover_classical_outputs():
    real = RealisedSED.circuit_based(SEDiagram.from_code(rotated_surface_code(2), 2))
    assert set(real.positions) == set(real.types)


def test_worldline_validation():
    sed = SEDiagram.from_code(REP, 1)
    data = ("in", 0), ("data", 0, "Z", 0), ("out", 0)
    with pytest.raises(ValueError, match="share"):
        RealisedSED(sed, [], [data, (("data", 0, "Z", 0),)])
    with pytest.raises(ValueError, match="not an edge"):
        RealisedSED(sed, [], [(("in", 0), ("out", 0))])
    with pytest.raises(ValueError, match="not an input"):
        RealisedSED(sed, [], [data[::-1]])
    with pytest.raises(ValueError, match="classical output"):
        RealisedSED(sed, [("gen", 0, 0)], [(("gen", 0, 0), ("meas", ("gen", 0, 0)))])


def test_measure_only_spiders():
    sed = SEDiagram.from_code(REP, 1)
    with pytest.raises(ValueError, match="spiders"):
        RealisedSED(sed, [("in", 0)], [])
