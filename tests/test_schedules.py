import pytest

from dizx.code import Code, rotated_surface_code
from dizx.realisation import QECExperiment, RealisedSED
from dizx.schedules import (
    RotatedSurfaceCodeSchedule, Schedule, assign_slots, conflicts, generator_legs,
)
from dizx.zx_diagram import NodeType, SEDiagram


def experiment(d=3, rounds=2, rotation=0):
    code = rotated_surface_code(d, rotation=rotation)
    exp = QECExperiment.memory(RealisedSED.circuit_based(SEDiagram.from_code(code, rounds)))
    return exp, RotatedSurfaceCodeSchedule(code)


def axis(a, b):
    return "row" if a[1] == b[1] else "column" if a[0] == b[0] else None


ROTATIONS = [0, 90, 180, 270]


def gens(exp):
    return [v for v in exp.types if v[0] == "gen"]


def layers(exp):
    out = {}
    for g in gens(exp):
        out.setdefault((g[1], exp.types[g]), []).append(g)
    return list(out.values())


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("d", [3, 5])
def test_rotated_surface_code_schedule(d, rotation):
    exp, schedule = experiment(d, rotation=rotation)
    slots = assign_slots(exp, schedule, gens(exp))
    assert len(slots) == sum(len(generator_legs(exp, g)) for g in gens(exp))
    assert set(slots.values()) == {0, 1, 2, 3}
    assert conflicts(slots, layers(exp)) == []


def test_boundary_axis_follows_rotation():
    assert RotatedSurfaceCodeSchedule(rotated_surface_code(3)).boundary_axis == {"X": "row", "Z": "column"}
    assert RotatedSurfaceCodeSchedule(rotated_surface_code(3, rotation=90)).boundary_axis == {"X": "column", "Z": "row"}


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_hook_errors_are_perpendicular_to_logicals(rotation):
    code = rotated_surface_code(5, rotation=rotation)
    exp, schedule = experiment(5, rotation=rotation)
    slots = assign_slots(exp, schedule, gens(exp))
    # Axis along which each logical runs, read from the code (not the schedule).
    x_bar, z_bar = (sorted(op) for op in code.logicals[0])
    logical_axis = {NodeType.Z: axis(x_bar[0], x_bar[1]),   # X generators (Z spiders)
                    NodeType.X: axis(z_bar[0], z_bar[1])}   # Z generators (X spiders)
    for g in gens(exp):
        legs = sorted(generator_legs(exp, g), key=lambda e: slots[e])
        if len(legs) == 4:
            hook = axis(*(exp.positions[e[1]][:2] for e in legs[2:]))
            assert hook is not None and hook != logical_axis[exp.types[g]]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_boundary_generators_keep_bulk_slots(rotation):
    exp, schedule = experiment(3, rotation=rotation)
    slots = assign_slots(exp, schedule, gens(exp))
    weight2 = [g for g in gens(exp) if len(generator_legs(exp, g)) == 2]
    assert weight2
    # A boundary check's two corners are the first or last pair of its order.
    for g in weight2:
        ks = sorted(slots[e] for e in generator_legs(exp, g))
        assert ks in ([0, 1], [2, 3])


class Sorted(Schedule):
    """A user-defined schedule: legs in sorted order."""
    n_slots = 2

    def slots(self, gen, neighbours, positions, gen_type, layer=0):
        return {v: k for k, v in enumerate(sorted(neighbours, key=repr))}


class AllZero(Schedule):
    n_slots = 1

    def slots(self, gen, neighbours, positions, gen_type, layer=0):
        return {v: 0 for v in neighbours}


def test_custom_schedule():
    rep = Code([{0: "Z", 1: "Z"}, {1: "Z", 2: "Z"}], qubits=range(3))
    exp = QECExperiment.memory(RealisedSED.circuit_based(SEDiagram.from_code(rep, 1)))
    slots = assign_slots(exp, Sorted(), gens(exp))
    assert sorted(slots.values()) == [0, 0, 1, 1]
    with pytest.raises(ValueError, match="same slot"):
        assign_slots(exp, AllZero(), gens(exp))


def test_conflicts_detected():
    exp, _ = experiment(3)
    # Two generators in one layer acting on the same data spider in the same slot.
    layer = layers(exp)[0]
    a, b, shared = next((a, b, v) for a in layer for b in layer if a != b
                        for v in exp.graph.neighbors(a) if v in set(exp.graph.neighbors(b)))
    slots = {(a, shared, 0): 0, (b, shared, 0): 0}
    assert conflicts(slots, [layer]) == [(shared, 0)]
    assert conflicts(slots, [[a], [b]]) == []


def test_d2_infers_missing_boundary_axis():
    # d = 2 has weight-2 generators of one type only.
    assert RotatedSurfaceCodeSchedule(rotated_surface_code(2)).boundary_axis == {"X": "row", "Z": "column"}
