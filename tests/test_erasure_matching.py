"""Tests for the pure-pymatching heralded-erasure decoding pipeline."""

from __future__ import annotations

import stim

from dizx.erasure_matching import (
    _meas_count,
    build_erasure_matching_data,
    make_matching_circuit,
)
from dizx.noise_models import MEASUREMENT_GATES as MEAS_OPS


def add_heralded_erasure(circuit: stim.Circuit, p_erasure: float) -> stim.Circuit:
    """``HERALDED_ERASE(p_erasure)`` (plus one herald detector per herald) before
    every measurement, on its qubits, with DETECTOR / OBSERVABLE_INCLUDE lookbacks
    remapped to the same original measurements."""
    def qubits(inst):
        return list(dict.fromkeys(t.qubit_value for t in inst.targets_copy() if not t.is_combiner))

    flat = list(circuit.flattened())
    orig_to_new, orig, new = {}, 0, 0
    for inst in flat:
        n = _meas_count(inst)
        if n and inst.name in MEAS_OPS:
            new += len(qubits(inst))
        for k in range(n):
            orig_to_new[orig + k] = new + k
        orig += n
        new += n
    out, orig, new = stim.Circuit(), 0, 0
    for inst in flat:
        if inst.name in MEAS_OPS:
            qs = qubits(inst)
            if qs:
                out.append("HERALDED_ERASE", qs, p_erasure)
                for i in range(len(qs)):
                    out.append("DETECTOR", [stim.target_rec(-(i + 1))], tag="herald")
                new += len(qs)
            out.append(inst)
            orig += _meas_count(inst)
            new += _meas_count(inst)
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            ts = [stim.target_rec(orig_to_new[orig + t.value] - new) if t.is_measurement_record_target else t
                  for t in inst.targets_copy()]
            out.append(inst.name, ts, inst.gate_args_copy())
        else:
            out.append(inst)
    return out

def _memory(distance: int = 3, rounds: int = 3, p: float = 0.0) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=p,
        before_measure_flip_probability=p,
    )


def test_make_matching_circuit_strips_heralds():
    circuit = add_heralded_erasure(_memory(p=0.01), 0.05)
    matching_circuit, n_erasures, erase_to_herald_meas = make_matching_circuit(
        circuit
    )

    herald_dets = build_erasure_matching_data(circuit).herald_det_ids
    # add_heralded_erasure puts one herald detector per herald bit.
    assert n_erasures == len(herald_dets) == len(erase_to_herald_meas)
    # The matching circuit has the herald detectors removed.
    assert matching_circuit.num_detectors == circuit.num_detectors - len(herald_dets)
    # No HERALDED_ERASE survives; replaced by tagged DEPOLARIZE1.
    names = {inst.name for inst in matching_circuit.flattened()}
    assert "HERALDED_ERASE" not in names
    assert "DEPOLARIZE1" in names
    # Its DEM builds as an ordinary graphlike model (no approximate_disjoint).
    dem = matching_circuit.detector_error_model(decompose_errors=True)
    assert sum(1 for i in dem if i.type == "error") > 0


def test_erasure_data_alignment():
    circuit = add_heralded_erasure(_memory(p=0.01), 0.05)
    data = build_erasure_matching_data(circuit)
    # The matching graph keeps the sampling circuit's detector numbering.
    assert data.base_dem.num_detectors == circuit.num_detectors
    # Every erasure is linked to an actual herald detector.
    assert all(h is not None for h in data.herald_det_for_erase)
    assert len(data.herald_det_for_erase) == data.n_erasures


def test_make_matching_circuit_preserves_exact_gate_args():
    """The text-based rewrite must not round noise probabilities.

    ``str(instruction)`` prints gate args at 6 significant digits, so a naive
    text round-trip would silently perturb sweep probabilities such as the
    ones ``np.logspace`` produces.
    """
    p = 0.0012589254117941673  # 17 significant digits
    circuit = add_heralded_erasure(_memory(p=p), p)
    matching_circuit, _, _ = make_matching_circuit(circuit)

    probs = {
        inst.gate_args_copy()[0]
        for inst in matching_circuit.flattened()
        if inst.name in ("DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR")
    }
    assert probs == {p}


def test_make_matching_circuit_handles_awkward_tags():
    """Tags may contain ``]`` and text that looks like a rec target."""
    tag = r'{"note": "b\Cc", "looks_like": "rec[-9\C"}'
    circuit = stim.Circuit(
        f"""
        RX 0 1
        HERALDED_ERASE(0.03) 0
        M 0
        HERALDED_ERASE(0.03) 1
        M 1
        DETECTOR rec[-4]
        DETECTOR rec[-2]
        DETECTOR[{tag}] rec[-3] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-3]
        """
    )
    matching_circuit, n_erasures, _ = make_matching_circuit(circuit)

    assert n_erasures == 2
    insts = list(matching_circuit.flattened())
    # Both herald detectors are dropped; the surviving one keeps its tag and
    # has its rec offsets shifted to skip the removed herald bits.
    dets = [i for i in insts if i.name == "DETECTOR"]
    assert len(dets) == 1
    assert [t.value for t in dets[0].targets_copy()] == [-2, -1]
    assert dets[0].tag == stim.Circuit(f"I[{tag}] 0")[0].tag
    obs = [i for i in insts if i.name == "OBSERVABLE_INCLUDE"]
    assert [t.value for t in obs[0].targets_copy()] == [-2]
    erase = [i for i in insts if i.tag.startswith("erase_")]
    assert len(erase) == 2 and all(i.name == "DEPOLARIZE1" for i in erase)
