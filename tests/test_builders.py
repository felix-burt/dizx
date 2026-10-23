"""The SED pipeline's builders with the previous pipeline's interface and tags."""
from collections import Counter

import pytest

from dizx import builders as B
from dizx.noise_models import NoiseSpec, apply_tagged_noise_model, parse_op_tag

D = 3
BUILDS = {
    "circuit": lambda o: B.build_surgery_circuit(D, D, "ZZ", o),
    "bell": lambda o: B.build_surgery_bell_prep_fusion(D, D, "ZZ", o),
    "bell_zigzag": lambda o: B.build_surgery_bell_prep_fusion_zigzag(D, D, "ZZ", o),
    "linear_kinf": lambda o: B.build_surgery_hybrid_linear_3dchain(D, D, "ZZ", o),
    "linear_k1": lambda o: B.build_surgery_hybrid_linear_5chain(D, D, "ZZ", o),
    "bell_meas_fusion": lambda o: B.build_surgery_bell_meas_fusion(D, D, "ZZ", o),
    "bell_meas_fusion_zigzag": lambda o: B.build_surgery_bell_meas_fusion_zigzag(D, D, "ZZ", o),
}
# Measurements and detectors of the previous pipeline's circuits at d = 3
# (merge ZZ, either observable).
OLD = {"circuit": (213, 188), "bell": (236, 156), "bell_zigzag": (271, 156),
       "linear_kinf": (242, 156), "linear_k1": (278, 156),
       "bell_meas_fusion": (236, 156), "bell_meas_fusion_zigzag": (271, 156)}
OPS = {"reset", "single_qubit_gate", "two_qubit_gate", "single_qubit_measurement", "fusion_gate",
       "fusion_measurement", "bell_prep_reset", "bell_prep_two_qubit", "state_prep", "idle"}


@pytest.fixture(scope="module", params=list(BUILDS))
def build(request):
    return request.param, {o: BUILDS[request.param](o) for o in ("ZZ", "XX")}


def test_matches_previous_pipeline(build):
    name, builds = build
    for o, b in builds.items():
        c = b.circuit
        dets, obs = c.compile_detector_sampler().sample(100, separate_observables=True)
        assert not dets.any() and not obs.any()
        assert c.num_observables == 1
        assert (c.num_measurements, c.num_detectors) == OLD[name]


def test_every_operation_is_tagged(build):
    name, builds = build
    kinds = Counter()
    for inst in builds["ZZ"].circuit.flattened():
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "TICK"):
            continue
        tag = parse_op_tag(inst.tag)
        assert tag.get("type") in ("circuit", "graph", "circuit-graph"), inst
        assert tag.get("operation") in OPS, inst
        kinds[tag["type"], tag["operation"], inst.name] += 1
    assert kinds["circuit", "two_qubit_gate", "CX"] > 0
    if name == "circuit":
        assert not any(t != "circuit" for t, *_ in kinds)
    else:
        # Fusions are a gate and two measurements; resource-state qubits are graph.
        assert kinds["graph", "fusion_gate", "CX"] + kinds["graph", "fusion_gate", "CZ"] > 0
        assert kinds["graph", "reset", "R"] + kinds["graph", "reset", "RX"] > 0
    if name.startswith("linear"):
        assert kinds["circuit-graph", "fusion_gate", "CX"] + kinds["circuit-graph", "fusion_gate", "CZ"] > 0


def test_config_noise_attaches(build):
    name, builds = build
    model = {("circuit", "two_qubit_gate"): [NoiseSpec("DEPOLARIZE2", 1e-3)],
             ("graph", "fusion_measurement"): [NoiseSpec("HERALDED_ERASE", 1e-2)],
             ("circuit-graph", "fusion_measurement"): [NoiseSpec("MEASUREMENT_FLIP", 1e-3)]}
    noisy = apply_tagged_noise_model(builds["XX"].circuit, noise_by_type=model)
    names = Counter(inst.name for inst in noisy.flattened())
    assert names["DEPOLARIZE2"] > 0
    assert (names["HERALDED_ERASE"] > 0) == (name != "circuit")
    noisy.detector_error_model(allow_gauge_detectors=False, approximate_disjoint_errors=True)


# Previous pipeline's memory circuits at d = 3: (measurements, detectors,
# fusions) per observable. The fusion memory here adds one readout photon per
# split output spider (6 at d = 3), see memory.line_pattern.
OLD_MEMORY = {
    ("circuit", "X"): (61, 52, 0), ("circuit", "Z"): (61, 52, 0),
    ("cluster", "X"): (169, 52, 0), ("cluster", "Z"): (169, 52, 0),
    ("fusion", "X"): (320 + 6, 52, 151), ("fusion", "Z"): (318 + 6, 52, 149),
}


@pytest.mark.parametrize("name,basis", list(OLD_MEMORY))
def test_memory_matches_previous_pipeline(name, basis):
    c = B.BUILDERS[f"memory_{name}"](D, D, basis).circuit
    dets, obs = c.compile_detector_sampler().sample(100, separate_observables=True)
    assert not dets.any() and not obs.any()
    assert c.num_observables == 1
    fusions = sum(len(i.targets_copy()) for i in c.flattened()
                  if parse_op_tag(i.tag).get("operation") == "fusion_measurement") // 2
    assert (c.num_measurements, c.num_detectors, fusions) == OLD_MEMORY[name, basis]


# Graphlike distances of the previous pipeline's circuits at d = 3 under
# depolarising noise on every circuit operation (observable ZZ, XX). The
# straight seams lose the X sector (ceil(d/2)), as before.
OLD_DISTANCE = {"bell_meas_fusion": (3, 2), "bell_meas_fusion_zigzag": (3, 3)}


@pytest.mark.parametrize("name", list(OLD_DISTANCE))
def test_distance_matches_previous_pipeline(name):
    model = {(kind, op): [NoiseSpec(chan, 1e-3)] for kind in ("circuit", "circuit-graph", "graph")
             for op, chan in (("reset", "X_ERROR"), ("single_qubit_gate", "DEPOLARIZE1"),
                              ("two_qubit_gate", "DEPOLARIZE2"), ("fusion_gate", "DEPOLARIZE2"),
                              ("single_qubit_measurement", "X_ERROR"),
                              ("fusion_measurement", "X_ERROR"))}
    got = []
    for o in ("ZZ", "XX"):
        noisy = apply_tagged_noise_model(BUILDS[name](o).circuit, noise_by_type=model)
        got.append(len(noisy.shortest_graphlike_error(ignore_ungraphlike_errors=True,
                                                     canonicalize_circuit_errors=False)))
    assert tuple(got) == OLD_DISTANCE[name]

