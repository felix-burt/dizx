from collections import Counter

import pytest

from dizx.circuit_extractor import to_stim
from dizx.schedules import RotatedSurfaceCodeSchedule, UnrotatedSurfaceCodeSchedule
from dizx.realisation import RealisedSED, interactions
from dizx.code import disjoint_union
from dizx.transformations import prepare_cluster_region
from dizx.zx_diagram import SPIDERS, NodeType
from dizx.surgery import (_other, seam_column_interface, straight_interface, two_patch_surgery,
                          zigzag_interface)
from helpers import check_deterministic, distance, has_matching_boundary


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
def test_surgery_without_matching_layers(joint):
    s = two_patch_surgery(3, joint, matching=False)
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


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
def test_seam_column_interface_unrotated(joint):
    """Joint-type generators on the seam split into three pieces, the other
    type's generators on the boundary columns split once; every piece touches
    one unit's qubits."""
    d = 3
    s = two_patch_surgery(d, joint, unrotated=True)
    x_seam = s.seam[0][0]
    iface = seam_column_interface(x_seam)
    sed = s.split_seam(iface)
    assert sed.webs_complete() and sed.detectors_complete()
    halves = [v for v in sed.types if v[0] == "split"]
    pieces = Counter(h[1] for h in halves)
    pos = s.sed.positions
    joint_spider = sed.types[next(g for g in pieces if pos[g][0] == x_seam)]
    for g, n in pieces.items():
        on_seam = pos[g][0] == x_seam
        assert n == (2 if on_seam else 1)
        assert (sed.types[g] is joint_spider) == on_seam
        assert abs(pos[g][0] - x_seam) <= 1
    # d seam generators per joint-type merge layer, 2 (d - 1) boundary ones
    # per other layer.
    n_joint = len({pos[g][2] for g in pieces if pos[g][0] == x_seam})
    n_other = len({pos[g][2] for g in pieces if pos[g][0] != x_seam})
    assert len(pieces) == d * n_joint + 2 * (d - 1) * n_other
    for v in halves + list(pieces):
        sides = {iface(*pos[u][:2]) for u in sed.graph.neighbors(v) if u[0] == "data"}
        assert len(sides) == 1


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
def test_seam_column_region_clusterises(joint):
    """The seam unit clusterises into a graph state whose every boundary edge
    is a fusion, and the hybrid realisation keeps full distance."""
    d = 3
    s = two_patch_surgery(d, joint, unrotated=True)
    sed, region, fusions = s.seam_column_region()
    ids = [v for v in region if v[0] == "id"]
    three_way = {v[1] for v in sed.types if v[0] == "split" and v[2] != "seam"}
    assert len(ids) == 2 * len(three_way)  # one per boundary leg
    hsed, S = prepare_cluster_region(sed, region, fusions)
    assert has_matching_boundary(hsed, S)
    # Each fusion joins a cluster node directly to a generator piece: the
    # boundary adds no outside copies, in either layer type.
    assert not any(v[0] == "split" and len(v) == 2 for v in hsed.types)
    boundary = {frozenset(it.ends) for it in interactions(hsed, []) if (it.ends[0] in S) != (it.ends[1] in S)}
    assert boundary == set(fusions)
    assert hsed.webs_complete() and hsed.detectors_complete()
    assert all(hsed.types[v] is NodeType.Z for v in S if hsed.types[v] in SPIDERS)
    assert not any(u in S and v in S and hsed.types[u] in SPIDERS and hsed.types[v] in SPIDERS
                   for u, v, _ in hsed.edges)
    real = RealisedSED.hybrid(hsed, S, fusions)
    for basis in "XZ":
        c = to_stim(s.experiment(basis, realised=real), UnrotatedSurfaceCodeSchedule())
        check_deterministic(c)
        assert distance(c) == d


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
