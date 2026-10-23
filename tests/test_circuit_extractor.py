import pytest
import stim

from dizx.circuit_extractor import to_stim
from dizx.code import rotated_surface_code
from dizx.realisation import QECExperiment, RealisedSED
from dizx.schedules import RotatedSurfaceCodeSchedule
from dizx.transformations import dualise, split_boundary
from dizx.zx_diagram import NodeType, SEDiagram
from helpers import check_deterministic, distance

ALLOWED = {"QUBIT_COORDS", "R", "RX", "H", "CX", "CZ", "M", "MX", "MPP", "TICK",
           "DETECTOR", "OBSERVABLE_INCLUDE", "I"}  # I: tagged idle / state-prep markers


def check_valid(c: stim.Circuit, exp: QECExperiment) -> None:
    assert {inst.name for inst in c.flattened()} <= ALLOWED
    assert c.num_detectors == len(exp.detectors)
    assert c.num_observables == len(exp.observables)
    check_deterministic(c)
    # No qubit is used twice within a tick.
    for layer in str(c.flattened()).split("TICK"):
        used = []
        for line in layer.strip().splitlines():
            name, *targets = line.split()
            if name in ("CX", "CZ", "H", "R", "RX", "M", "MX"):
                used += targets
        assert len(used) == len(set(used)), layer


def surface(d, rounds, rotation=0, basis="Z", mode="circuit"):
    code = rotated_surface_code(d, rotation=rotation)
    sed = SEDiagram.from_code(code, rounds)
    if mode == "circuit":
        real = RealisedSED.circuit_based(sed)
        schedule = RotatedSurfaceCodeSchedule(code)
    else:
        real = RealisedSED.measurement_based(sed)
        schedule = None
    exp = QECExperiment.memory(real, basis)
    return code, exp, to_stim(exp, schedule)


@pytest.mark.parametrize("d,rotation,basis", [
    (3, 0, "Z"), (3, 0, "X"), (3, 90, "Z"), (3, 90, "X"), (5, 0, "Z"), (5, 90, "X"),
])
def test_circuit_based_memory(d, rotation, basis):
    code, exp, c = surface(d, d, rotation, basis)
    check_valid(c, exp)
    # Data qubits plus one reused ancilla per plaquette.
    assert c.num_qubits == code.n + len(code.stabilisers)
    assert distance(c) == d


def test_circuit_based_rounds_use_four_cx_ticks_per_layer():
    _, _, c = surface(3, 2)
    layers = str(c).split("TICK")
    assert sum(layer.strip().startswith("CX") for layer in layers) == 2 * 2 * 4


@pytest.mark.parametrize("d", [3, 5])
@pytest.mark.parametrize("basis", ["Z", "X"])
def test_measurement_based_memory(d, basis):
    _, exp, c = surface(d, d, basis=basis, mode="mb")
    check_valid(c, exp)
    assert {inst.name for inst in c.flattened()} <= ALLOWED - {"CX", "MPP"}
    # Resource-state preparation is not fault tolerant, so only measurement noise.
    assert distance(c, gates=()) == d


def split_sed(dual: bool):
    code = rotated_surface_code(3)
    sed = SEDiagram.from_code(code, 3)
    data = {v for v in sed.types if v[0] == "data" and v[-1] == (1, 1)}
    S = data | {g for v in data for g in sed.graph.neighbors(v) if g[0] == "gen"}
    split = split_boundary(sed, S)
    if dual:
        split = dualise(split, [g for g in S if g[0] == "gen" and split.types[g] is NodeType.X])
    return code, split


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("links", ["fusion", "bell"])
def test_split_generators_as_links(links, dual):
    code, sed = split_sed(dual)
    real = RealisedSED.circuit_based(sed, links=links)
    assert real.links and all(link.kind == links for link in real.links)
    ops = {link.ops for link in real.links}
    assert ops == ({("XX", "ZZ"), ("XZ", "ZX")} if dual else {("XX", "ZZ")})
    exp = QECExperiment.memory(real, "Z")
    c = to_stim(exp, RotatedSurfaceCodeSchedule(code))
    check_valid(c, exp)
    assert ("MPP" in str(c)) == (links == "fusion")
    if links == "fusion" and dual:
        mpp = [inst.targets_copy() for inst in c.flattened() if inst.name == "MPP"]
        assert any(t[0].is_x_target and t[2].is_z_target for t in mpp)
    assert distance(c) == 3


def test_fusion_outcomes_in_records():
    code, sed = split_sed(dual=False)
    real = RealisedSED.circuit_based(sed, links="fusion")
    exp = QECExperiment.memory(real, "Z")
    fusion_labels = {lab for link in real.links for lab in link.outcomes}
    assert any(exp.record(w) & fusion_labels for w in exp.detectors)


def test_classical_output_must_end_world_line():
    code = rotated_surface_code(2)
    sed = SEDiagram.from_code(code, 1)
    real = RealisedSED.circuit_based(sed)
    data = [p for p in real.worldlines if len(p) > 1]
    mid = data[0][1]
    bad = RealisedSED(sed, list(real.classical) + [mid], real.worldlines)
    with pytest.raises(ValueError, match="not the end of its world-line"):
        to_stim(QECExperiment.memory(bad), RotatedSurfaceCodeSchedule(code))


def test_alternation_counts_layers_not_steps():
    """Two separate patches are scheduled as separate steps of one layer;
    alternation must still give both the same layer index."""
    from dizx.code import disjoint_union

    left = rotated_surface_code(3)
    right = rotated_surface_code(3, origin=(4, 0))
    sed = SEDiagram.from_code(disjoint_union(left, right), 2)
    seen: dict[float, set[int]] = {}

    class Recording(RotatedSurfaceCodeSchedule):
        def slots(self, gen, neighbours, positions, gen_type, layer=0):
            seen.setdefault(float(positions[gen][2]), set()).add(layer)
            return super().slots(gen, neighbours, positions, gen_type, layer)

    to_stim(QECExperiment.memory(RealisedSED.circuit_based(sed)),
            Recording(disjoint_union(left, right), alternate=True))
    assert all(len(ls) == 1 for ls in seen.values())
    # Each type's layers count 0, 1, ...: Z layers at even z, X layers at odd z.
    assert [seen[z].pop() for z in sorted(seen)] == [0, 0, 1, 1]
