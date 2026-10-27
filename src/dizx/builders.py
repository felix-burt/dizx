"""Builders of the new SED pipeline with the previous pipeline's interface.

Each ``build_*`` function takes the arguments of its namesake in
the previous pipeline's ``disqec.experiments`` (as the threshold runner passes them) and returns a
``SEDBuild`` whose ``circuit`` is tagged for
``noise_models.apply_tagged_noise_model`` (see ``circuit_extractor``), with
fusions emitted as an entangling gate and two single-qubit measurements, so
the existing noise-model configs apply unchanged.

Experiment conventions follow the previous builders: for a ``merge_type``
``JJ`` merge both patches are prepared in the other basis K, and measured in
the basis of ``observable``. Observable ``JJ`` is then the merge outcome
(``I -> J J``) and ``KK`` the product of the patches' K logicals; each circuit
has that single observable.

Layers: the patches run ``2 n`` layers before the merge and ``2 n - 1`` after
it; the merge has ``2 n + 1``, starting and ending with the joint's type.

Each variant (seam shape, chain truncation) is its own builder with no
options for it, so a builder name alone identifies a circuit family.
``BUILDERS`` keys them by function name without ``build_``, e.g.
``surgery_hybrid_linear_5chain``; configs and cached circuit files use that name.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from dizx import memory
from dizx.circuit_extractor import to_stim
from dizx.realisation import QECExperiment, RealisedSED
from dizx.schedules import RotatedSurfaceCodeSchedule, Schedule, UnrotatedSurfaceCodeSchedule
from dizx.surgery import (
    Surgery, straight_interface, surgery_hybrid_linear, two_patch_surgery, zigzag_interface,
)
from dizx.transformations import SEDEditor
from dizx.zx_diagram import Node, NodeType, SEDiagram


@dataclass
class SEDBuild:
    """A built experiment: ``circuit`` (tagged, noiseless) and how it was made."""
    circuit: stim.Circuit
    experiment: QECExperiment
    surgery: Surgery


def surgery_layers(n_cycles: int) -> tuple[int, int, int]:
    """Layers before, during and after the merge."""
    return 2 * n_cycles, 2 * n_cycles + 1, 2 * n_cycles - 1


def _check(merge_type: str, observable: str) -> None:
    if merge_type not in ("XX", "ZZ"):
        raise ValueError(f"merge_type must be 'XX' or 'ZZ', got {merge_type!r}")
    if observable not in ("XX", "ZZ"):
        raise ValueError(f"observable must be 'XX' or 'ZZ', got {observable!r}")


def _finish(s: Surgery, real: RealisedSED, observable: str,
            schedule: Schedule | None) -> SEDBuild:
    prep = "X" if s.joint == "ZZ" else "Z"
    exp = QECExperiment(real, {p: prep for p in real.inputs},
                        {p: observable[0] for p in real.outputs})
    if len(exp.observables) != 1:
        raise AssertionError(f"expected one observable, got {len(exp.observables)}")
    circuit = to_stim(exp, schedule or RotatedSurfaceCodeSchedule(s.merged), fusions="gates")
    return SEDBuild(circuit, exp, s)


def build_surgery_circuit(distance: int, n_cycles: int, merge_type: str = "ZZ",
                          observable: str = "ZZ",
                          schedule: Schedule | None = None) -> SEDBuild:
    """Circuit-based two-patch surgery (``experiments.build_surgery_circuit``)."""
    _check(merge_type, observable)
    s = two_patch_surgery(distance, merge_type,
                          layers=surgery_layers(n_cycles))
    return _finish(s, RealisedSED.circuit_based(s.sed), observable, schedule)


def build_surgery_unrotated_circuit(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                    observable: str = "ZZ",
                                    schedule: Schedule | None = None) -> SEDBuild:
    """``build_surgery_circuit`` with unrotated surface code patches
    (``two_patch_surgery(..., unrotated=True)``). No counterpart in the
    previous pipeline."""
    _check(merge_type, observable)
    s = two_patch_surgery(distance, merge_type,
                          layers=surgery_layers(n_cycles), unrotated=True)
    return _finish(s, RealisedSED.circuit_based(s.sed), observable,
                   schedule or UnrotatedSurfaceCodeSchedule())


def _gadget_chains(sed: SEDiagram, colours) -> tuple[SEDEditor, list[tuple[Node, Node, list[Node]]]]:
    """Insert identity spiders on the edge from each split generator's
    original half ``n1`` to its ``split`` half (``Surgery.split_seam``), with
    colours ``colours(c)`` for a half of colour c, evenly spaced between the
    halves. Returns the editor and ``(n1, half, chain)`` for each split
    generator."""
    halves = [v for v in sed.types if isinstance(v, tuple) and v[0] == "split"]
    ed = SEDEditor(sed)
    out = []
    for half in halves:
        n1 = half[1]
        chain = []
        u = n1
        for colour in colours(ed.types[n1]):
            chain.append(u := ed.insert_identity(
                next(f for f in ed.graph.edges(u, keys=True) if f[1] == half), colour))
        a, b = np.asarray(ed.positions[n1], float), np.asarray(ed.positions[half], float)
        for i, v in enumerate(chain, 1):
            ed.positions[v] = tuple(a + i / (len(chain) + 1) * (b - a))
        out.append((n1, half, chain))
    return ed, out


def bell_prep_gadgets(sed: SEDiagram) -> tuple[SEDiagram, set[Node]]:
    """Replace each edge between the halves of a split generator by a Bell
    pair made from two emissions and a fusion

        n1 -- f1 == f2 -- n2

    Each half ``n`` is the emitter: it entangles a photon ``f`` of the other
    colour with a CX, and the two photons are fused (``XX``/``ZZ``).

    Returns the SED and the gadget spiders (resource-state qubits).
    """
    other = {NodeType.Z: "X", NodeType.X: "Z"}
    ed, chains = _gadget_chains(sed, lambda c: (other[c], other[c]))
    return ed.freeze(), {v for _, _, chain in chains for v in chain}


def resource_pair_gadgets(sed: SEDiagram) -> tuple[SEDiagram, set[Node]]:
    """Replace each edge between the halves of a split generator by two
    two-qubit resource states and three fusions

        n1 == g1 -- g2 == g3 -- g4 == n2

    as in the previous pipeline's ``unfuse_resource_fusion_pair``: ``g1, g4``
    have the halves' colour c and ``g2, g3`` the other, so each resource pair
    ``(g1, g2)``, ``(g3, g4)`` is a Bell pair made with a CX, and every
    fusion is between like colours (``XX``/``ZZ``).

    Returns the SED and the gadget spiders (resource-state qubits).
    """
    other = {NodeType.Z: "X", NodeType.X: "Z"}
    ed, chains = _gadget_chains(sed, lambda c: (c.value, other[c], other[c], c.value))
    return ed.freeze(), {v for _, _, chain in chains for v in chain}


def _seam_gadget_surgery(distance: int, n_cycles: int, merge_type: str, observable: str,
                         gadgets, zigzag_phase: int | None, schedule: Schedule | None,
                         unrotated: bool = False) -> SEDBuild:
    """Two units joined across each split generator by ``gadgets`` (e.g.
    ``bell_prep_gadgets``); the seam is ``zigzag_interface`` with
    ``zigzag_phase``, or straight if ``None`` (the only choice for
    ``unrotated`` patches)."""
    _check(merge_type, observable)
    if unrotated and zigzag_phase is not None:
        raise ValueError("unrotated patches only have a straight interface")
    s = two_patch_surgery(distance, merge_type,
                          layers=surgery_layers(n_cycles), unrotated=unrotated)
    iface = (straight_interface(s.seam[0][0]) if zigzag_phase is None
             else zigzag_interface(distance, zigzag_phase))
    sed, region = gadgets(s.split_seam(iface))
    real = RealisedSED.circuit_based(sed, links="fusion")
    real.graph_region = region
    if unrotated:
        schedule = schedule or UnrotatedSurfaceCodeSchedule()
    return _finish(s, real, observable, schedule)


def build_surgery_bell_prep_fusion(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                   observable: str = "ZZ",
                                   schedule: Schedule | None = None) -> SEDBuild:
    """Two units joined by Bell pairs made from photon emissions and a
    photon fusion (``bell_prep_gadgets``), across a straight seam. The
    previous pipeline's ``experiments.build_surgery_bell_prep_fusion`` put a
    separate matter qubit between each half and its photon; that circuit has
    the same detector error model under every noise model the paper uses."""
    return _seam_gadget_surgery(distance, n_cycles, merge_type, observable,
                                bell_prep_gadgets, None, schedule)


def build_surgery_bell_prep_fusion_zigzag(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                          observable: str = "ZZ",
                                          schedule: Schedule | None = None) -> SEDBuild:
    """``build_surgery_bell_prep_fusion`` across a zigzag seam of phase 1, which
    keeps full distance in both sectors with the default schedule, as the
    previous pipeline's zigzag did (phase 0 loses the X sector)."""
    return _seam_gadget_surgery(distance, n_cycles, merge_type, observable,
                                bell_prep_gadgets, 1, schedule)


def build_surgery_unrotated_bell_prep_fusion(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                             observable: str = "ZZ",
                                             schedule: Schedule | None = None) -> SEDBuild:
    """``build_surgery_bell_prep_fusion`` with unrotated surface code patches,
    across a straight seam (the seam column on the right). Each row of the
    seam has one split generator, d of the joint's type and d - 1 of the
    other, and with ``UnrotatedSurfaceCodeSchedule`` both sectors keep full
    distance. No counterpart in the previous pipeline."""
    return _seam_gadget_surgery(distance, n_cycles, merge_type, observable,
                                bell_prep_gadgets, None, schedule, unrotated=True)


def build_surgery_bell_meas_fusion(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                   observable: str = "ZZ",
                                   schedule: Schedule | None = None) -> SEDBuild:
    """Two units joined, across each split generator, by two Bell-pair
    resource states and three fusions (``resource_pair_gadgets``; the previous
    pipeline's ``experiments.build_surgery_fusion_resource_split``), across a
    straight seam. That has distance ceil(d/2) in the X sector, as before."""
    return _seam_gadget_surgery(distance, n_cycles, merge_type, observable,
                                resource_pair_gadgets, None, schedule)


def build_surgery_bell_meas_fusion_zigzag(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                          observable: str = "ZZ",
                                          schedule: Schedule | None = None) -> SEDBuild:
    """``build_surgery_bell_meas_fusion`` across a zigzag seam of phase 0, which
    keeps full distance in both sectors, as the previous pipeline's did; phase
    1 loses the X sector (ceil(d/2)). That is the reverse of the Bell-prep
    gadget, whose halves gate their photons instead of fusing with them (see
    ``zigzag_interface``)."""
    return _seam_gadget_surgery(distance, n_cycles, merge_type, observable,
                                resource_pair_gadgets, 0, schedule)


def _hybrid_linear(distance: int, n_cycles: int, merge_type: str, observable: str,
                   k: int | None, looped: bool, schedule: Schedule | None) -> SEDBuild:
    """Hybrid surgery with linear resource states
    (``experiments.build_surgery_hybrid_linear_long`` with
    ``unfuse_time_edges``); ``k`` cuts every k-th time edge of the chains,
    ``None`` leaves them whole. ``looped`` loops the clusterised region's
    boundary (Bell-pair rungs on the interface)."""
    _check(merge_type, observable)
    s, real, _ = surgery_hybrid_linear(distance, merge_type,
                                       layers=surgery_layers(n_cycles),
                                       period=k, looped=looped)
    return _finish(s, real, observable, schedule)


def build_surgery_hybrid_linear_3dchain(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                        observable: str = "ZZ",
                                        schedule: Schedule | None = None) -> SEDBuild:
    """Linear-chain hybrid surgery with untruncated chains (the 3d-chain)."""
    return _hybrid_linear(distance, n_cycles, merge_type, observable,
                          None, False, schedule)


def build_surgery_hybrid_linear_5chain(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                       observable: str = "ZZ",
                                       schedule: Schedule | None = None) -> SEDBuild:
    """Linear-chain hybrid surgery with every time edge cut (the 5-chain)."""
    return _hybrid_linear(distance, n_cycles, merge_type, observable,
                          1, False, schedule)


def build_surgery_hybrid_linear_8chain(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                       observable: str = "ZZ",
                                       schedule: Schedule | None = None) -> SEDBuild:
    """Linear-chain hybrid surgery with every second time edge cut (the 8-chain)."""
    return _hybrid_linear(distance, n_cycles, merge_type, observable,
                          2, False, schedule)


def build_surgery_hybrid_linear_looped(distance: int, n_cycles: int, merge_type: str = "ZZ",
                                       observable: str = "ZZ",
                                       schedule: Schedule | None = None) -> SEDBuild:
    """``build_surgery_hybrid_linear_3dchain`` with the clusterised region's
    boundary looped, which restores full distance in the non-joint sector. No
    counterpart in the previous pipeline."""
    return _hybrid_linear(distance, n_cycles, merge_type, observable,
                          None, True, schedule)


def _registry(*builders) -> dict:
    """``{name: builder}``, the name being the function's without ``build_``."""
    return {b.__name__.removeprefix("build_"): b for b in builders}


BUILDERS = {
    **memory.BUILDERS,
    **_registry(build_surgery_circuit, build_surgery_unrotated_circuit,
                build_surgery_bell_prep_fusion, build_surgery_bell_prep_fusion_zigzag,
                build_surgery_unrotated_bell_prep_fusion, build_surgery_bell_meas_fusion, build_surgery_bell_meas_fusion_zigzag,
                build_surgery_hybrid_linear_3dchain, build_surgery_hybrid_linear_5chain,
                build_surgery_hybrid_linear_8chain, build_surgery_hybrid_linear_looped),
}
