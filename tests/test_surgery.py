import pytest
import stim

from dizx.circuit_extractor import to_stim
from dizx.schedules import RotatedSurfaceCodeSchedule
from dizx.realisation import RealisedSED
from dizx.code import disjoint_union
from dizx.surgery import _other, straight_interface, two_patch_surgery, zigzag_interface


def distance(c: stim.Circuit, p: float = 1e-3) -> int:
    noisy = stim.Circuit()
    for inst in c.flattened():
        if inst.name in ("CX", "CZ"):
            noisy.append(inst)
            noisy.append("DEPOLARIZE2", inst.targets_copy(), p)
        elif inst.name in ("M", "MX", "MPP"):
            noisy.append(inst.name, inst.targets_copy(), p)
        else:
            noisy.append(inst)
    return len(noisy.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4, dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False))


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
@pytest.mark.parametrize("matching", [True, False])
def test_seam_is_closed(joint, matching):
    """The seam has no ports; its wires end at spiders of the seam basis's
    colour, explicit init / proj spiders only where the merge's end layers do
    not match the joint."""
    s = two_patch_surgery(3, joint, matching=matching)
    assert len(s.seam) == 3
    assert len(s.sed.inputs) == len(s.sed.outputs) == 2 * 9
    assert not [p for p in s.sed.inputs + s.sed.outputs if p[1] in set(s.seam)]
    colour = {"ZZ": "Z", "XX": "X"}[joint]  # Z spiders prepare / measure X
    assert len(s.sed.closed) == 6
    for p, v in s.sed.closed.items():
        assert s.sed.types[v].value == colour
        assert (v[0] in ("init", "proj")) == (not matching and (p[0] == "out") == (joint == "ZZ"))
    assert s.sed.webs_complete() and s.sed.detectors_complete()
    # The merge has one layer more than 2 * rounds by default.
    layers = {(g[1], s.sed.types[g]) for g in s.sed.types if g[0] == "gen" and g[1] in s.merge_rounds}
    assert len(layers) == 2 * s.rounds[1] + matching


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
@pytest.mark.parametrize("matching", [True, False])
def test_surgery_layer_modes(joint, matching):
    s = two_patch_surgery(3, joint, matching=matching)
    c = to_stim(s.experiment("Z"), RotatedSurfaceCodeSchedule(s.merged))
    dets, obs = c.compile_detector_sampler().sample(200, separate_observables=True)
    assert not dets.any() and not obs.any()
    assert distance(c) == 3


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
@pytest.mark.parametrize("patch_basis", ["Z", "X"])
def test_surgery_experiment(joint, patch_basis):
    s = two_patch_surgery(3, joint)
    exp = s.experiment(patch_basis)
    c = to_stim(exp, RotatedSurfaceCodeSchedule(s.merged))
    dets, obs = c.compile_detector_sampler().sample(200, separate_observables=True)
    assert not dets.any() and not obs.any()
    # Measuring in the basis of the joint operator gives the merge outcome
    # (I -> J J) and both single-patch logicals; otherwise only K K -> K K.
    assert c.num_observables == (3 if patch_basis == joint[0] else 1)
    assert c.num_detectors == to_stim(s.experiment(_other(patch_basis)),
                                      RotatedSurfaceCodeSchedule(s.merged)).num_detectors
    assert distance(c) == 3


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
@pytest.mark.parametrize("matching", [True, False])
def test_surgery_logicals(joint, matching):
    """One logical web per logical correlation of the merge, each a minimal
    sheet: J_1 -> J_1 and J_2 -> J_2 straight up the patches' outer
    boundaries, I -> J J from the merge's lid up the inner boundaries, and
    K K -> K K."""
    d = 3
    s = two_patch_surgery(d, joint, matching=matching)
    sed = s.sed
    both = disjoint_union(*s.patches)
    J, K = joint[0], _other(joint[0])

    def column(x, p):
        return {(x, y): p for y in range(d)}

    row = {(x, 0): K for x in range(2 * d + 1) if x != d}
    expected = [(column(0, J), column(0, J), {0}), (column(2 * d, J), column(2 * d, J), {2 * d}),
                ({}, {**column(d - 1, J), **column(d + 1, J)}, {d - 1, d, d + 1}),
                (row, row, None)]

    def side(w, ports):
        return {p[1]: c for p in ports if (c := w.pauli(sed.edges[sed.legs(p)[0]])) != "I"}

    def key(p_in, p_out):
        return frozenset(p_in.items()), frozenset(p_out.items())

    got = {key(side(w, sed.inputs), side(w, sed.outputs)): w for w in sed.logicals}
    assert set(got) == {key(i, o) for i, o, _ in expected} and len(sed.logicals) == 4
    for p_in, p_out, columns in expected:
        if columns is not None:
            w = got[key(p_in, p_out)]
            touched = {a[-1][0] for e in w.support() for a in e[:2] if a[0] == "data"}
            assert touched == columns
    assert len(sed.co_stabilisers) == len(sed.stabilisers) == len(both.stabilisers)


@pytest.fixture(scope="module", params=["ZZ", "XX"])
def surgery(request):
    return two_patch_surgery(3, request.param)


def test_split_seam_units(surgery):
    d = 3
    for iface in (straight_interface(d), zigzag_interface(d)):
        sed = surgery.split_seam(iface)
        assert sed.webs_complete() and sed.detectors_complete()
        halves = [v for v in sed.types if v[0] == "split"]
        assert halves and all(h[1][1] in surgery.merge_rounds for h in halves)
        # Every split half only touches qubits of one unit.
        for v in [h for h in halves] + [h[1] for h in halves]:
            # The interface is in the unsplit diagram's coordinates.
            sides = {iface(*surgery.sed.positions[u][:2]) for u in sed.graph.neighbors(v) if u[0] == "data"}
            assert len(sides) == 1


@pytest.mark.parametrize("patch_basis", ["Z", "X"])
@pytest.mark.parametrize("links, phase", [("bell", 1), ("fusion", 0)])
def test_split_surgery_interfaces(surgery, patch_basis, links, phase):
    """The zigzag interface keeps full distance in both sectors; the straight
    one loses one sector."""
    d = 3
    out = {}
    for name, iface in (("straight", straight_interface(d)), ("zigzag", zigzag_interface(d, phase))):
        real = RealisedSED.circuit_based(surgery.split_seam(iface), links=links)
        assert {link.kind for link in real.links} == {links}
        c = to_stim(surgery.experiment(patch_basis, realised=real), RotatedSurfaceCodeSchedule(surgery.merged))
        dets, obs = c.compile_detector_sampler().sample(200, separate_observables=True)
        assert not dets.any() and not obs.any()
        out[name] = distance(c)
    assert out["zigzag"] == 3
    hurt = {"ZZ": "X", "XX": "Z"}[surgery.joint]
    assert out["straight"] == (2 if patch_basis == hurt else 3)


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
@pytest.mark.parametrize("matching", [True, False])
def test_layers_evenly_spaced(joint, matching):
    """Every layer sits one unit above the last, across all three phases."""
    s = two_patch_surgery(3, joint, (2, 2, 2), matching=matching)
    zs = sorted({float(s.sed.positions[g][2]) for g in s.sed.types if g[0] == "gen"})
    assert zs == list(range(len(zs))) and len(zs) == 12 + matching
