"""Single-patch memory builders with the previous pipeline's interface.

Each ``build_memory_*`` takes the arguments of its namesake in the previous
pipeline's ``experiments`` module and returns a ``MemoryBuild`` whose
``circuit`` is tagged for ``noise_models.apply_tagged_noise_model``.

Conventions follow the previous builders: the patch is prepared and measured
in the basis of ``observable`` and runs ``4 n + 1`` stabiliser layers (``4 n
+ 3`` with ``add_noiseless_rounds``), starting and ending with the layer of
the other type. The patch is ``rotated_surface_code(d, rotation=90)``, which
puts the weight-2 X plaquettes on the left and right boundaries, as the
previous patches had them.

- ``build_memory_circuit``: circuit-based.
- ``build_memory_cluster``: measurement-based, every spider its own photon.
- ``build_memory_fusion``: the cluster cut into line-shaped resource states
  joined by fusions (``line_pattern``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import stim

from dizx.circuit_extractor import to_stim
from dizx.code import Code, rotated_surface_code
from dizx.realisation import QECExperiment, RealisedSED
from dizx.schedules import RotatedSurfaceCodeSchedule, Schedule
from dizx.surgery import _through_h
from dizx.transformations import SEDEditor, prepare_cluster_region
from dizx.zx_diagram import SPIDERS, Node, NodeType, SEDiagram


@dataclass
class MemoryBuild:
    """A built memory experiment: ``circuit`` (tagged, noiseless) and how it was made."""
    circuit: stim.Circuit
    experiment: QECExperiment
    code: Code


def memory_code(distance: int) -> Code:
    return rotated_surface_code(distance, rotation=90)


def memory_sed(code: Code, n_cycles: int, observable: str,
               add_noiseless_rounds: bool = False) -> SEDiagram:
    """``4 n + 1`` layers (``4 n + 3`` with ``add_noiseless_rounds``),
    starting and ending with the type other than ``observable``."""
    if observable not in ("X", "Z"):
        raise ValueError(f"observable must be 'X' or 'Z', got {observable!r}")
    if n_cycles < 1:
        raise ValueError("n_cycles must be >= 1")
    other = "Z" if observable == "X" else "X"
    rounds = 2 * n_cycles + (1 if add_noiseless_rounds else 0)
    return SEDiagram.from_code(code, rounds, first=other, last=other)


def _observable(observable: str, merge_type: str | None) -> str:
    # The runner passes the sweep's merge type; for memory it names the basis.
    return merge_type if merge_type is not None else observable


def build_memory_circuit(distance: int, n_cycles: int, observable: str = "X",
                         add_noiseless_rounds: bool = False,
                         merge_type: str | None = None,
                         schedule: Schedule | None = None) -> MemoryBuild:
    """Circuit-based memory (``experiments.build_memory_circuit``)."""
    observable = _observable(observable, merge_type)
    code = memory_code(distance)
    sed = memory_sed(code, n_cycles, observable, add_noiseless_rounds)
    exp = QECExperiment.memory(RealisedSED.circuit_based(sed), observable)
    return MemoryBuild(to_stim(exp, schedule or RotatedSurfaceCodeSchedule(code), fusions="gates"),
                       exp, code)


def build_memory_cluster(distance: int, n_cycles: int, observable: str = "X",
                         add_noiseless_rounds: bool = False,
                         merge_type: str | None = None) -> MemoryBuild:
    """Measurement-based memory on one cluster state
    (``experiments.build_memory_cluster``); ``merge_type`` overrides
    ``observable``, as the runner's sweep axis.

    Same circuit up to the order of the CZs, which the previous builder fixed
    differently: with noise on the graph CZs the X sector here has distance
    ceil(d/2) (d before). No config puts noise on graph CZs."""
    observable = _observable(observable, merge_type)
    code = memory_code(distance)
    sed = memory_sed(code, n_cycles, observable, add_noiseless_rounds)
    real = RealisedSED.measurement_based(sed)
    real.graph_region = {v for v, t in real.types.items() if t in SPIDERS}
    exp = QECExperiment.memory(real, observable)
    return MemoryBuild(to_stim(exp, None, fusions="gates"), exp, code)


def line_pattern(sed: SEDiagram, x_min: float = 0.0, tol: float = 1e-9
                 ) -> tuple[SEDiagram, dict[frozenset, str]]:
    """Cut a graph-like SED into line-shaped resource states joined by
    fusions, as the previous pipeline's ``unfuse_line_pattern``.

    In the previous frame (data qubits at even ``x``, plaquettes at odd ``x``,
    so ``x`` here is twice the patch coordinate; layers one apart in ``z``)
    every spider at ``x >= x_min`` is split in two: a data spider's legs
    towards ``+x`` or ``+z`` go to the new half, a plaquette spider's legs
    towards ``+x`` do. A spider with every leg (ports aside) on one side is
    left whole. Directions are those of the original positions throughout.

    A fused half cannot also be read out, so a split spider's output leg gets
    a readout photon of its own (an identity spider before the port). The
    previous pipeline folded that readout into the fusion outcomes instead, so
    its circuits have one qubit fewer per such output.

    Returns the SED and the fusions (pairs of halves).
    """
    pos = dict(sed.positions)
    origin: dict[Node, Node] = {}
    outputs = set(sed.outputs)

    def frame(n: Node) -> tuple[float, float]:
        x, _, z = pos[origin.get(n, n)]
        return 2 * x, z

    spiders = sorted((v for v, t in sed.types.items() if t in SPIDERS),
                     key=lambda v: (frame(v), repr(v)))
    sed = SEDEditor(sed)  # edited in place from here
    readouts: set[Node] = set()
    fusions: dict[frozenset, str] = {}
    for v in spiders:
        x, z = frame(v)
        if x < x_min - tol:
            continue
        data = round(x - x_min) % 2 == 0

        def ahead(n: Node) -> bool:
            if sed.types[n] is NodeType.P or n in readouts:
                return False
            fx, fz = frame(n)
            return fx > x + tol or (data and fz > z + tol)

        ends = [_through_h(sed, v, e[1]) for e in sed.graph.edges(v, keys=True)]
        real = [n for n in ends if sed.types[n] is not NodeType.P]
        n_ahead = sum(map(ahead, real))
        if n_ahead == 0 or n_ahead == len(real):
            continue
        for port in [n for n in ends if n in outputs]:
            # The readout photon is a Z spider on a Hadamard edge from v: on
            # a plain leg, insert an X identity and dualise it.
            plain = next(iter(sed.graph.neighbors(port))) == v
            r = sed.insert_identity(next(iter(sed.graph.edges(port, keys=True))),
                                    NodeType.X if plain else NodeType.Z)
            if plain:
                sed.dualise([r])
            readouts.add(r)
            pos[r] = sed.positions[r]
        keep = [e for e in sed.graph.edges(v, keys=True) if not ahead(_through_h(sed, v, e[1]))]
        half: Hashable = ("line", v)
        sed.split(v, keep, name=half)
        origin[half] = v
        fusions[frozenset((v, half))] = "fusion"
    return sed.freeze(), fusions


def build_memory_fusion(distance: int, n_cycles: int, observable: str = "X",
                        add_noiseless_rounds: bool = False,
                        merge_type: str | None = None) -> MemoryBuild:
    """Measurement-based memory on line-shaped resource states joined by
    fusions (``experiments.build_memory_fusion``)."""
    observable = _observable(observable, merge_type)
    code = memory_code(distance)
    sed = memory_sed(code, n_cycles, observable, add_noiseless_rounds)
    sed, S = prepare_cluster_region(sed, set(sed.types))
    sed, fusions = line_pattern(sed)
    S = set(sed.types)
    real = RealisedSED.hybrid(sed, S, fusions)
    exp = QECExperiment.memory(real, observable)
    return MemoryBuild(to_stim(exp, None, fusions="gates"), exp, code)


# Keyed like builders.BUILDERS: the function name without ``build_``.
BUILDERS = {
    "memory_circuit": build_memory_circuit,
    "memory_cluster": build_memory_cluster,
    "memory_fusion": build_memory_fusion,
}
