import pytest

from dizx.code import rotated_surface_code
from dizx.realisation import QECExperiment, RealisedSED
from dizx.schedules import RotatedSurfaceCodeSchedule
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


def leg_slots(exp, schedule, g):
    """``{neighbour: slot}`` for the gate legs of generator g (not its readout)."""
    outputs = set(exp.classical_outputs())
    nbs = [v for v in exp.graph.neighbors(g) if v not in outputs]
    return schedule.slots(g, nbs, exp.positions, exp.types[g])


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_hook_errors_are_perpendicular_to_logicals(rotation):
    code = rotated_surface_code(5, rotation=rotation)
    exp, schedule = experiment(5, rotation=rotation)
    # Axis along which each logical runs, read from the code (not the schedule).
    x_bar, z_bar = (sorted(op) for op in code.logicals[0])
    logical_axis = {NodeType.Z: axis(x_bar[0], x_bar[1]),   # X generators (Z spiders)
                    NodeType.X: axis(z_bar[0], z_bar[1])}   # Z generators (X spiders)
    for g in gens(exp):
        slots = leg_slots(exp, schedule, g)
        if len(slots) == 4:
            hook = axis(*(exp.positions[v][:2] for v in sorted(slots, key=slots.get)[2:]))
            assert hook is not None and hook != logical_axis[exp.types[g]]


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_boundary_generators_keep_bulk_slots(rotation):
    exp, schedule = experiment(3, rotation=rotation)
    slots = [leg_slots(exp, schedule, g) for g in gens(exp)]
    weight2 = [sorted(s.values()) for s in slots if len(s) == 2]
    assert weight2
    # A boundary check's two corners are the first or last pair of its order.
    assert all(ks in ([0, 1], [2, 3]) for ks in weight2)


def test_d2_infers_missing_boundary_axis():
    # d = 2 has weight-2 generators of one type only.
    assert RotatedSurfaceCodeSchedule(rotated_surface_code(2)).boundary_axis == {"X": "row", "Z": "column"}
