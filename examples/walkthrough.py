"""Helpers shared by the ``builder_*`` walkthrough notebooks."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import stim

from dizx.noise_models import NoiseSpec, apply_tagged_noise_model
from dizx.sinter_decoders import PyMatchingHeraldedDecoder

FUSION_ERASURE = ("graph", "fusion_measurement")


def erasure(key: tuple[str, ...], p: float) -> dict:
    """Heralded erasure at rate p on the operations tagged ``key``."""
    return {key: NoiseSpec("HERALDED_ERASE", p)}


def logical_error_rate(circuit: stim.Circuit, shots: int, seed: int | None = None) -> float:
    """Sample ``circuit`` (noisy) and decode with correlated heralded pymatching,
    as the paper's runs do."""
    dets, obs = circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True, bit_packed=True)
    decoder = PyMatchingHeraldedDecoder(decompose=True, enable_correlations=True).with_circuit(circuit)
    compiled = decoder.compile_decoder_for_dem(dem=circuit.detector_error_model(
        approximate_disjoint_errors=True))
    pred = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=dets)
    return float(np.mean(np.any(pred != obs, axis=1)))


def erasure_sweep(build, key: tuple[str, ...], distances, ps, shots: int = 5_000):
    """LER vs erasure rate on ``key``-tagged operations, one curve per
    distance; ``build(d)`` returns the noiseless tagged circuit."""
    rates = {}
    for d in distances:
        circuit = build(d)
        rates[d] = [logical_error_rate(apply_tagged_noise_model(circuit, noise_by_type=erasure(key, p)),
                                       shots, seed=d) for p in ps]
    return rates


def full_and_seam(diagram, d: int, title: str, web=None):
    """``diagram`` drawn whole and zoomed onto the seam column (x = d)."""
    fig = plt.figure(figsize=(14, 6))
    diagram.draw(web, ax=fig.add_subplot(1, 2, 1, projection="3d"), title=title)
    ax = diagram.draw(web, ax=fig.add_subplot(1, 2, 2, projection="3d"), title="seam")
    ax.set_xlim(d - 1.5, d + 1.5)
    plt.show()


def draw_resource_states(real, S, d: int, ax=None, title: str | None = None):
    """The resource states in S flattened onto (y, t), one colour each, with
    the two sides of the seam pulled apart; fusions black, Bell pairs dashed."""
    from dizx.realisation import interactions, resource_states

    ax = ax or plt.figure(figsize=(5, 6)).add_subplot()
    states = resource_states(real, S)
    linked = {frozenset(link.interaction.ends): link.kind for link in real.links}

    def xy(v):
        x, y, z = real.positions[v]
        return y + 0.6 * (x - d), z

    cmap = plt.get_cmap("tab10")
    for k, c in enumerate(states):
        for it in interactions(real, real.worldlines):
            if set(it.ends) <= c and frozenset(it.ends) not in linked:
                ax.plot(*np.array([xy(v) for v in it.ends]).T, color=cmap(k % 10), lw=2)
        ax.scatter(*np.array([xy(v) for v in c]).T, color=cmap(k % 10), s=30, zorder=3)
    inside = set().union(*states)
    for ends, kind in linked.items():
        if ends & inside:
            ax.plot(*np.array([xy(v) for v in ends]).T, "k" + ("--" if kind == "bell" else "-"), lw=2.5)
    ax.set(xlabel="y (sides pulled apart)", ylabel="t",
           title=title or f"{len(states)} resource states")
    return ax


def plot_sweep(rates: dict, ps, ax=None, title: str | None = None):
    ax = ax or plt.figure(figsize=(5, 4)).add_subplot()
    for d, r in rates.items():
        ax.plot(ps, r, "o-", label=f"d={d}")
    ax.set(xlabel="erasure rate", ylabel="logical error rate", yscale="log", title=title or "")
    ax.legend()
    return ax
