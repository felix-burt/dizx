import networkx as nx
import pytest
import stim

from dizx.circuit_extractor import to_stim
from dizx.code import rotated_surface_code
from dizx.realisation import QECExperiment, RealisedSED, interactions, resource_states
from dizx.schedules import RotatedSurfaceCodeSchedule
from dizx.surgery import surgery_hybrid_linear, two_patch_surgery
from dizx.transformations import (
    cut_edges, loop_boundary, prepare_cluster_region,
)
from dizx.zx_diagram import SPIDERS, NodeType, SEDiagram
import helpers
from helpers import check_deterministic as check, has_matching_boundary


def distance(c: stim.Circuit) -> int:
    """Noise on CX, measurements and fusions (none on cluster CZs)."""
    return helpers.distance(c, gates=("CX",))


def memory_region():
    code = rotated_surface_code(3)
    sed = SEDiagram.from_code(code, 3)
    data = {v for v in sed.types if v[0] == "data" and v[-1] == (1, 1)}
    return code, sed, data | {g for v in data for g in sed.graph.neighbors(v) if g[0] == "gen"}


def test_prepare_cluster_region():
    _, sed, S = memory_region()
    hsed, S2 = prepare_cluster_region(sed, S)
    assert has_matching_boundary(hsed, S2)
    # Graph-like inside: only Z spiders, joined through H nodes.
    assert all(hsed.types[v] in (NodeType.Z, NodeType.H, NodeType.P) for v in S2)
    assert hsed.webs_complete() and hsed.detectors_complete()
    assert all(len({u, v} & S2) == 1 for u, v, _ in cut_edges(hsed, S2))


def test_hybrid_memory():
    code, sed, S = memory_region()
    hsed, S2 = prepare_cluster_region(sed, S)
    real = RealisedSED.hybrid(hsed, S2)
    assert {link.kind for link in real.links} == {"fusion"}
    assert {link.ops for link in real.links} == {("XX", "ZZ"), ("XZ", "ZX")}
    # Every clusterised Z spider is a world-line of its own.
    for v in S2:
        if real.types[v] is NodeType.Z:
            (path,) = (p for p in real.worldlines if v in p)
            assert [u for u in path if real.types[u] in SPIDERS] == [v]
    exp = QECExperiment.memory(real)
    c = to_stim(exp, RotatedSurfaceCodeSchedule(code))
    check(c)
    assert distance(c) == 3


@pytest.fixture(scope="module")
def surgery_region():
    s = two_patch_surgery(3, "ZZ")
    return (s, *prepare_cluster_region(s.sed, s.merge_region()))


@pytest.fixture(scope="module")
def looped_surgery(surgery_region):
    s, hsed, S2 = surgery_region
    lsed, S3, rungs = loop_boundary(hsed, S2)
    return s, hsed, S2, lsed, S3, rungs


def test_loop_boundary(looped_surgery):
    _, hsed, S2, lsed, S3, rungs = looped_surgery
    cut = len(cut_edges(hsed, S2))
    n = len(rungs) // 2  # loopable boundary edges
    assert 0 < n <= cut
    assert len(lsed.detectors) == len(hsed.detectors) + n  # one loop web each
    assert lsed.webs_complete() and lsed.detectors_complete()
    assert len(cut_edges(lsed, S3)) == cut + n  # each loop crosses twice
    # Each loop node has its two loop edges and exactly one external leg.
    for rung in rungs:
        for x in rung:
            assert lsed.graph.degree(x) == 3
    for rung in rungs:
        u, v = rung
        assert lsed.graph.number_of_edges(u, v) == 1 and lsed.types[u] is lsed.types[v]
        assert (u in S3) == (v in S3)
    for w in lsed.detectors[len(hsed.detectors):]:
        assert len(w.support()) in (4, 6)  # 4 edges, plus 2 more through H nodes
    pts = [tuple(round(float(c), 9) for c in lsed.positions[v]) for v in lsed.types]
    assert len(set(pts)) == len(pts)  # no overlapping nodes


@pytest.mark.parametrize("patch_basis", ["Z", "X"])
def test_hybrid_surgery_looped_boundary(looped_surgery, patch_basis):
    """Option B: each boundary edge becomes a loop of two Bell pairs (one
    outside, one inside the cluster) and two fusions across the boundary.
    The loop webs restore full distance in both sectors."""
    s, _, _, lsed, S3, rungs = looped_surgery
    real = RealisedSED.hybrid(lsed, S3, {r: "bell" for r in rungs})
    assert sum(link.kind == "bell" for link in real.links) == len(rungs)
    c = to_stim(s.experiment(patch_basis, realised=real), RotatedSurfaceCodeSchedule(s.merged))
    check(c)
    assert distance(c) == 3


@pytest.fixture(scope="module")
def central_column():
    """The central column of a d=3 memory, clusterised, then its boundary looped."""
    sed = SEDiagram.from_code(rotated_surface_code(3), 2)
    data = {v for v in sed.types if v[0] == "data" and v[-1][0] == 1}
    hsed, S2 = prepare_cluster_region(
        sed, data | {g for v in data for g in sed.graph.neighbors(v) if g[0] == "gen"})
    return hsed, S2, loop_boundary(hsed, S2)


def test_central_column_memory_looped(central_column):
    """Clusterise the central column of a d=3 memory: boundary fusions on both
    sides. The loops restore full distance (unlooped, one basis drops to 2)."""
    code = rotated_surface_code(3)
    hsed, S2, (lsed, S3, rungs) = central_column
    assert len(rungs) // 2 == 8  # the weight-2 boundary generators split 1-1, unloopable
    out = {}
    for name, real in (("plain", RealisedSED.hybrid(hsed, S2)),
                       ("looped", RealisedSED.hybrid(lsed, S3, {r: "bell" for r in rungs}))):
        ds = []
        for b in "ZX":
            c = to_stim(QECExperiment.memory(real, b), RotatedSurfaceCodeSchedule(code))
            check(c)
            ds.append(distance(c))
        out[name] = ds
    assert out == {"plain": [2, 3], "looped": [3, 3]}


def test_loop_boundary_short_way_round(central_column):
    """Webs through a looped node use one side of the loop, not three."""
    from test_transformations import assert_short_way_round
    hsed, _, (lsed, _, _) = central_column
    assert_short_way_round(lsed, lsed.detectors[len(hsed.detectors):])
    lsed._check_webs()
    for a, b in [(hsed.stabilisers, lsed.stabilisers), (hsed.co_stabilisers, lsed.co_stabilisers),
                 (hsed.logicals, lsed.logicals)]:
        assert [w.boundary() for w in a] == [w.boundary() for w in b]


@pytest.mark.parametrize("joint", ["ZZ", "XX"])
@pytest.mark.parametrize("matching", [True, False])
def test_surgery_hybrid_linear(joint, matching):
    """The merge region's graph state splits into linear resource states, as
    in the previous pipeline's build_surgery_hybrid_linear_long: one column of
    generators over two merge layers each, 9 qubits at d = 3."""
    s, real, S = surgery_hybrid_linear(3, joint, matching=matching)
    states = resource_states(real, S)
    assert all(nx.is_tree(g) and max(d for _, d in g.degree()) <= 2
               for g in (_cluster(real, c) for c in states if len(c) > 1))
    sizes = sorted(len(c) for c in states)
    assert sizes == ([2, 2, 3, 3] + [9] * 6 if matching else [2, 2, 3, 5] + [9] * 5)
    assert {link.ops for link in real.links if link.kind == "fusion"} >= {("XX", "ZZ")}
    for patch_basis in "ZX":
        c = to_stim(s.experiment(patch_basis, realised=real), RotatedSurfaceCodeSchedule(s.merged))
        check(c)
        if matching:
            # Same as the undecomposed hybrid: full distance in the joint's sector.
            assert distance(c) == (3 if patch_basis == joint[0] else 2)


def test_surgery_hybrid_linear_looped():
    s, real, _ = surgery_hybrid_linear(3, "ZZ", looped=True)
    for patch_basis in "ZX":
        c = to_stim(s.experiment(patch_basis, realised=real), RotatedSurfaceCodeSchedule(s.merged))
        check(c)
        assert distance(c) == 3


def _cluster(real, nodes):
    g = nx.Graph()
    g.add_nodes_from(nodes)
    linked = {frozenset(link.interaction.ends) for link in real.links}
    g.add_edges_from(it.ends for it in interactions(real, real.worldlines)
                     if set(it.ends) <= nodes and frozenset(it.ends) not in linked)
    return g
