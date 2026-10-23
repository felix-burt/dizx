"""Tests for the sinter heralded-erasure pymatching decoder."""

from __future__ import annotations

import numpy as np
import pymatching
import pytest
import sinter
import stim

from dizx.memory import build_memory_fusion
from dizx.noise_models import NoiseSpec, apply_tagged_noise_model
from dizx.erasure_matching import (
    build_erasure_matching_data,
    simulate_logical_error_rate_heralded,
    _reweighted_dem,
)
import dizx.sinter_decoders as sinter_decoders
from dizx.sinter_decoders import PyMatchingHeraldedDecoder


@pytest.fixture(params=["table", "list", "legacy"])
def reweight_path(request, monkeypatch):
    """Run a test through each way the decoder can apply erasure weights."""
    if request.param == "table" and not sinter_decoders.SUPPORTS_REWEIGHT_TABLE:
        pytest.skip("installed pymatching lacks edge_reweight_table")
    if request.param == "list":
        if not sinter_decoders.SUPPORTS_EDGE_REWEIGHTS:
            pytest.skip("installed pymatching lacks edge_reweights")
        monkeypatch.setattr(sinter_decoders, "SUPPORTS_REWEIGHT_TABLE", False)
    if request.param == "legacy":
        monkeypatch.setattr(sinter_decoders, "SUPPORTS_EDGE_REWEIGHTS", False)
    return request.param


def _mem_fusion(distance: int, p_erase: float, instruction: str = "HERALDED_ERASE"):
    base = build_memory_fusion(distance, distance, observable="Z").circuit
    return apply_tagged_noise_model(
        base,
        noise_by_type={("graph", "fusion_measurement"): NoiseSpec(instruction, p_erase)},
    )


@pytest.mark.parametrize("enable_correlations", [False, True])
def test_compiled_matches_per_shot_rebuild(enable_correlations, reweight_path):
    """The compiled decoder must reproduce a fresh per-shot DEM rebuild.

    For ``enable_correlations=False`` this validates the fast in-place edge
    swap; for ``True`` it validates the per-shot rebuild used under correlated
    matching.  Both must match a clean reference exactly.
    """
    circuit = _mem_fusion(3, 0.10)
    data = build_erasure_matching_data(circuit, enable_correlations=enable_correlations)

    n = 1500
    dets_bp = circuit.compile_detector_sampler().sample(n, bit_packed=True)

    compiled = (
        PyMatchingHeraldedDecoder(enable_correlations=enable_correlations)
        .with_circuit(circuit=circuit)
        .compile_decoder_for_dem(
            dem=circuit.detector_error_model(approximate_disjoint_errors=True)
        )
    )
    pred_bp = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets_bp)
    pred = np.unpackbits(pred_bp, axis=1, count=circuit.num_observables, bitorder="little")

    unpacked = np.unpackbits(dets_bp, axis=1, count=circuit.num_detectors, bitorder="little")
    cols = np.asarray(data.data_det_ids)
    herald = data.herald_det_for_erase
    for i in range(n):
        row = unpacked[i]
        fired = {k for k, h in enumerate(herald) if h is not None and row[h]}
        if fired:
            m = pymatching.Matching.from_detector_error_model(
                _reweighted_dem(data.base_dem, fired, 0.5),
                enable_correlations=enable_correlations,
            )
        else:
            m = data.off_matching
        expected = m.decode(row[cols], enable_correlations=enable_correlations)
        np.testing.assert_array_equal(pred[i], expected)


def test_no_erasure_matches_plain_pymatching(reweight_path):
    """With no heralds (plain DEPOLARIZE1 noise) the decoder == plain pymatching."""
    circuit = _mem_fusion(3, 0.05, instruction="DEPOLARIZE1")
    data = build_erasure_matching_data(circuit, enable_correlations=False)
    assert data.n_erasures == 0

    n = 2000
    dets_bp = circuit.compile_detector_sampler().sample(n, bit_packed=True)

    compiled = (
        PyMatchingHeraldedDecoder(enable_correlations=False)
        .with_circuit(circuit=circuit)
        .compile_decoder_for_dem(dem=circuit.detector_error_model())
    )
    pred_bp = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets_bp)
    pred = np.unpackbits(pred_bp, axis=1, count=circuit.num_observables, bitorder="little")

    plain = pymatching.Matching.from_detector_error_model(
        circuit.detector_error_model(decompose_errors=True)
    )
    expected = plain.decode_batch(np.unpackbits(
        dets_bp, axis=1, count=circuit.num_detectors, bitorder="little"
    ))
    np.testing.assert_array_equal(pred, expected)


def test_sinter_collect_runs_and_matches_standalone():
    """End-to-end via sinter.collect; LER agrees with the standalone simulator."""
    circuit = _mem_fusion(3, 0.10)
    task = sinter.Task(circuit=circuit, decoder="pymatching_heralded")
    results = sinter.collect(
        num_workers=2,
        tasks=[task],
        custom_decoders={
            "pymatching_heralded": PyMatchingHeraldedDecoder().with_circuit(circuit=circuit)
        },
        max_shots=8000,
        max_errors=8000,
        print_progress=False,
    )
    stat = results[0]
    assert stat.shots >= 8000
    sinter_ler = stat.errors / stat.shots
    standalone_ler = simulate_logical_error_rate_heralded(circuit, num_shots=8000)
    # Independent draws -> allow a few sigma of Monte-Carlo spread.
    assert abs(sinter_ler - standalone_ler) < 0.03
