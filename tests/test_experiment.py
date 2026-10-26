import pytest

from dizx.code import Code, rotated_surface_code
from dizx.realisation import QECExperiment, RealisedSED
from dizx.zx_diagram import SPIDERS, NodeType, SEDiagram


def realised(code, rounds):
    return RealisedSED.circuit_based(SEDiagram.from_code(code, rounds))


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("d,rounds", [(3, 1), (3, 3)])
def test_memory_experiment(d, rounds, basis):
    code = rotated_surface_code(d)
    exp = QECExperiment.memory(realised(code, rounds), basis)
    m = len(code.stabilisers)
    same = [i for i, s in enumerate(code.stabilisers) if basis in s.values()]

    assert not exp.inputs and not exp.outputs
    # Bulk detectors, plus first- and last-round detectors of the matching type.
    assert len(exp.detectors) == (rounds - 1) * m + 2 * len(same)
    assert len(exp.observables) == 1
    assert exp.webs_complete()
    # The other type's stabilisers / co-stabilisers and the other logical are lost.
    other = [i for i in range(m) if i not in same]
    assert sorted(exp.lost) == sorted(
        [("stabiliser", i) for i in other] + [("co-stabiliser", i) for i in other]
        + [("logical", 0 if basis == "Z" else 1)])


def test_memory_records():
    code = rotated_surface_code(3)
    rounds = 2
    exp = QECExperiment.memory(realised(code, rounds), "Z")
    readout = lambda q: ("readout", ("out", q))
    for w, orig in zip(exp.detectors, exp.detector_origins):
        (fam, i), = orig
        if fam == "co-stabiliser":
            assert exp.record(w) == {("gen", 0, i)}
        elif fam == "stabiliser":
            assert exp.record(w) == {("gen", rounds - 1, i)} | {readout(q) for q in code.stabilisers[i]}
    (obs,) = exp.observables
    assert exp.observable_origins == [(("logical", 1),)]
    assert exp.record(obs) == {readout(q) for q in code.logicals[0][1]}


def test_worldlines_closed():
    exp = QECExperiment.memory(realised(rotated_surface_code(3), 1))
    data = [p for p in exp.worldlines if len(p) > 1]
    assert all(p[0][0] == "prep" and p[-1][0] == "readout" for p in data)
    assert all(p[-1] in exp.classical for p in exp.worldlines)
    assert set(exp.positions) == set(exp.types)


def test_mixed_bases_need_combinations():
    # [[4,2,2]], qubits 0, 1 in X and 2, 3 in Z. The logical Z0Z1 web dies on its
    # own, but times the ZZZZ stabiliser and co-stabiliser it gives the
    # representative Z2Z3 -> Z2Z3, which survives.
    code = Code([{q: "X" for q in range(4)}, {q: "Z" for q in range(4)}], range(4),
                [({0: "X", 1: "X"}, {0: "Z", 2: "Z"}), ({0: "X", 2: "X"}, {0: "Z", 1: "Z"})])
    bases = {0: "X", 1: "X", 2: "Z", 3: "Z"}
    exp = QECExperiment(realised(code, 2), {("in", q): b for q, b in bases.items()},
                        {("out", q): b for q, b in bases.items()})
    assert exp.webs_complete() and all(w.is_valid() for w in exp.webs)
    assert ("logical", 3) in exp.lost
    assert exp.observable_origins == [
        (("logical", 0),),
        (("stabiliser", 1), ("co-stabiliser", 1), ("logical", 3)),
    ]
    readouts = {("readout", ("out", q)) for q in (2, 3)}
    assert exp.record(exp.observables[1]) == {("gen", 0, 1), ("gen", 1, 1)} | readouts


def test_infers_unspecified_bases():
    """A port with no basis given follows the colour of its wire's end spider."""
    r = realised(rotated_surface_code(2), 1)
    exp = QECExperiment(r, {}, {p: "Z" for p in r.outputs})
    # from_code wires start at Z data spiders: prepared in X.
    assert set(exp.preps.values()) == {"X"}
    with pytest.raises(ValueError, match="preparations must be on inputs"):
        QECExperiment(r, {r.outputs[0]: "Z"})


@pytest.mark.parametrize("prep,meas,closer", [("Z", "Z", "prep"), ("X", "X", "readout")])
def test_measurement_based_experiment_is_one_resource_state(prep, meas, closer):
    """Ports close onto graph qubits: a closing that the port's spider absorbs
    merges into it; the other basis (here Z on the inputs, X on the outputs)
    adds a measured Z spider behind an H node."""
    real = RealisedSED.measurement_based(SEDiagram.from_code(rotated_surface_code(3), 3))
    assert not any(v in real.inputs + real.outputs for p in real.worldlines for v in p)
    exp = QECExperiment(real, {p: prep for p in real.inputs}, {p: meas for p in real.outputs})
    outputs = set(exp.classical.values())
    spiders = {v for v, t in exp.types.items() if t in SPIDERS and v not in outputs}
    assert all(exp.types[v] is NodeType.Z for v in spiders)
    assert spiders == exp.graph_region == {v for p in exp.worldlines for v in p}
    assert set(exp.classical) == spiders
    added = [v for v in spiders if v[0] in ("prep", "readout")]
    assert len(added) == 9 and all(v[0] == closer for v in added)
    for g in added:
        (h,) = [u for u in exp.graph.neighbors(g) if u != exp.classical[g]]
        assert exp.types[h] is NodeType.H and exp.graph.degree(h) == 2
    assert exp.webs_complete() and len(exp.observables) == 1
