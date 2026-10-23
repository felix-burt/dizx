import matplotlib

matplotlib.use("Agg")

import pytest

from dizx.builders import BUILDERS
from dizx.dem_view import (
    dem_mechanisms, graphlike_distance, noisy_circuit, plot_dem, shortest_logical,
)
from dizx.erasure_matching import make_matching_circuit
from dizx.noise_models import NoiseSpec

ERASURE = {("graph", "fusion_measurement"): NoiseSpec("HERALDED_ERASE", 0.1)}
CIRCUIT = {("circuit", "two_qubit_gate"): NoiseSpec("DEPOLARIZE2", 1e-3),
           ("circuit", "single_qubit_measurement"): NoiseSpec("MEASUREMENT_FLIP", 1e-3)}


@pytest.fixture(scope="module")
def bell():
    return BUILDERS["surgery_bell_prep_fusion"](distance=3, n_cycles=1).circuit


def test_erasure_dem_keeps_detector_numbering(bell):
    # Erasure edges are drawn at the base circuit's coordinates, so removing the
    # herald detectors must restore the original numbering.
    matching = make_matching_circuit(noisy_circuit(bell, ERASURE))[0]
    assert matching.get_detector_coordinates() == bell.get_detector_coordinates()


def test_mechanisms_are_labelled_and_placed(bell):
    mechs = dem_mechanisms(bell, {**ERASURE, **CIRCUIT})
    coords = bell.get_detector_coordinates()
    assert all(len(coords[d]) == 3 for m in mechs for d in m.dets)
    sources = {m.source for m in mechs}
    assert "graph/fusion_measurement HERALDED_ERASE" in sources
    assert "circuit/two_qubit_gate DEPOLARIZE2" in sources


def test_plot_dem_draws_shortest_logical(bell):
    noise = {**ERASURE, **CIRCUIT}
    fig, ax, mechs = plot_dem(bell, noise)
    chain = shortest_logical(bell, noise)
    assert len(chain) == graphlike_distance(bell, noise) > 0
    assert any("shortest logical" in t.get_text() for t in ax.get_legend().get_texts())
    matplotlib.pyplot.close(fig)


@pytest.mark.parametrize("builder, observable, expected", [
    ("surgery_bell_prep_fusion", "ZZ", 4), ("surgery_bell_prep_fusion_zigzag", "ZZ", 4),  # d + 1
    # The straight seam halves XX: (d + 1) / 2 vs d.
    ("surgery_bell_prep_fusion", "XX", 2), ("surgery_bell_prep_fusion_zigzag", "XX", 3),
])
def test_erasure_distance(builder, observable, expected):
    c = BUILDERS[builder](distance=3, n_cycles=3, observable=observable).circuit
    assert graphlike_distance(c, ERASURE) == expected
