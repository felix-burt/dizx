"""Look at a circuit's detector error model in 3D.

stim finds the DEM; this module only labels and draws it. Each noise channel
(one key of ``noise_by_type``) gets its own DEM, so every mechanism knows which
channel caused it. Heralded erasures go through
``erasure_matching.make_matching_circuit``: they become ordinary edges between
real detectors, as the decoder sees them, and the coordinate-less herald
detectors disappear. Detectors are placed at the ``(x, y, t)`` coordinates the
circuit extractor writes.

>>> fig, ax, mechs = plot_dem(build.circuit, {("graph", "fusion_measurement"): NoiseSpec("HERALDED_ERASE", 0.1)})
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import stim
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from dizx.erasure_matching import make_matching_circuit
from dizx.noise_models import HERALDING_INSTRUCTIONS, NoiseSpec, apply_tagged_noise_model

ERASURE_COLOUR = "#e34948"
PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#8a5cd1", "#e87ba4", "#008300", "#52514e"]


@dataclass(frozen=True)
class Mechanism:
    """One graphlike piece of a DEM error: detectors it flips (1 = boundary, 2 = edge,
    more = undecomposed hyperedge), observables it flips, and its noise channel."""
    dets: tuple[int, ...]
    obs: tuple[int, ...]
    p: float
    source: str


def _kinds(value) -> list[str]:
    values = value if isinstance(value, list) else [value]
    return [v.instruction if isinstance(v, NoiseSpec) else v[0] if isinstance(v, tuple) else "?"
            for v in values]


def _label(key, value) -> str:
    key = key if isinstance(key, tuple) else (key,)
    return f"{'/'.join(key)} {'+'.join(_kinds(value))}"


def _is_erasure(source: str) -> bool:
    return any(h in source for h in HERALDING_INSTRUCTIONS)


def _dem(noisy: stim.Circuit) -> stim.DetectorErrorModel:
    """DEM with heralded erasures turned into plain edges (herald detectors removed)."""
    if any(inst.name in HERALDING_INSTRUCTIONS for inst in noisy.flattened()):
        noisy = make_matching_circuit(noisy)[0]
    try:
        return noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    except ValueError:
        return noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True,
                                          ignore_decomposition_failures=True)


def _pieces(dem: stim.DetectorErrorModel, source: str) -> list[Mechanism]:
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        dets: list[int] = []
        obs: list[int] = []
        for t in inst.targets_copy() + [stim.DemTarget.separator()]:
            if t.is_separator():
                if dets or obs:
                    out.append(Mechanism(tuple(dets), tuple(obs), p, source))
                dets, obs = [], []
            elif t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs.append(t.val)
    return out


def noisy_circuit(circuit: stim.Circuit, noise_by_type: Mapping) -> stim.Circuit:
    return apply_tagged_noise_model(circuit, noise_by_type=dict(noise_by_type))


def dem_mechanisms(circuit: stim.Circuit, noise_by_type: Mapping) -> list[Mechanism]:
    """Every graphlike mechanism, labelled by the noise channel that causes it."""
    mechs = []
    for key, value in noise_by_type.items():
        mechs += _pieces(_dem(noisy_circuit(circuit, {key: value})), _label(key, value))
    return mechs


def graphlike_distance(circuit: stim.Circuit, noise_by_type: Mapping) -> int:
    """Size of stim's shortest graphlike logical error (0 if there is none)."""
    try:
        return len(_dem(noisy_circuit(circuit, noise_by_type)).shortest_graphlike_error())
    except ValueError:
        return 0


def shortest_logical(circuit: stim.Circuit, noise_by_type: Mapping) -> list[Mechanism]:
    """stim's shortest graphlike logical error under all of ``noise_by_type``."""
    try:
        err = _dem(noisy_circuit(circuit, noise_by_type)).shortest_graphlike_error()
    except ValueError:
        return []
    return _pieces(err, "logical")


def _segments(mechs: list[Mechanism], pos: dict[int, np.ndarray], centre: np.ndarray,
              boundary_len: float) -> tuple[list, list]:
    """Line segments for edges and boundary stubs, and spokes for hyperedges.

    Positions end in time; a boundary stub points away from ``centre`` in space."""
    lines, spokes = [], []
    for m in mechs:
        pts = [pos[d] for d in m.dets if d in pos]
        if len(pts) == 2:
            lines.append(pts)
        elif len(pts) == 1:
            out = pts[0][:-1] - centre[:-1]
            norm = np.linalg.norm(out)
            out = out / norm if norm > 1e-9 else np.eye(len(out))[0]
            lines.append([pts[0], pts[0] + boundary_len * np.array([*out, 0.0])])
        elif len(pts) > 2:
            hub = np.mean(pts, axis=0)
            spokes += [[hub, q] for q in pts]
    return lines, spokes


def _add_lines(ax, segments, **kwargs) -> None:
    if not segments:
        return
    if ax.name == "3d":
        ax.add_collection3d(Line3DCollection(segments, **kwargs))
    else:
        ax.add_collection(LineCollection(segments, **kwargs))


def _sheet(mechs: list[Mechanism], pos: dict[int, np.ndarray]) -> tuple[dict, dict]:
    """The interface flattened to a plane: ``({det: (along, t)}, {det: side})``.

    The interface is every detector an erasure can flip (else any fusion
    channel's detectors). It is a thin sheet around the seam, so the spatial
    axis it spreads least along is dropped; ``side`` says which side of the seam
    a detector sits on."""
    iface = {d for m in mechs if _is_erasure(m.source) for d in m.dets}
    iface = iface or {d for m in mechs if "graph" in m.source for d in m.dets}
    iface = sorted(d for d in iface if d in pos)
    if not iface:
        raise ValueError("no interface: the noise model has no erasure or fusion channel")
    xyz = np.array([pos[d] for d in iface])
    drop = int(np.argmin(np.ptp(xyz[:, :2], axis=0)))
    keep, seam = 1 - drop, np.median(xyz[:, drop])
    flat = {d: np.array([pos[d][keep], pos[d][2]]) for d in iface}
    return flat, {d: pos[d][drop] > seam for d in iface}


def plot_dem(circuit: stim.Circuit, noise_by_type: Mapping, *, ax=None, sources=None,
             logical: bool = True, sheet: bool = False, boundary_len: float = 0.6,
             title: str | None = None):
    """Draw the DEM of ``circuit`` under ``noise_by_type`` (time vertical).

    Edges are coloured by noise channel (erasure always red), observable-flipping
    ones are thicker, and with ``logical`` stim's shortest graphlike logical error
    is drawn in black. ``sources`` restricts which channel labels are drawn.

    By default the view is 3D. ``sheet=True`` instead draws the interface alone
    in 2D: the detectors erasures can flip, flattened onto (position along the
    seam, t), with circles and squares for the two sides of the seam. Only
    mechanisms lying entirely in the interface are drawn.
    Returns ``(fig, ax, mechanisms)``.
    """
    coords = circuit.get_detector_coordinates()
    pos = {d: np.array(c[:3], dtype=float) for d, c in coords.items() if len(c) >= 3}
    mechs = dem_mechanisms(circuit, noise_by_type)
    side = None
    if sheet:
        pos, side = _sheet(mechs, pos)
    xyz = np.array(list(pos.values()))
    centre = xyz.mean(axis=0)

    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection=None if sheet else "3d")
    fig = ax.figure

    labels = list(dict.fromkeys(m.source for m in mechs))
    colours, i = {}, 0
    for s in labels:
        colours[s] = ERASURE_COLOUR if _is_erasure(s) else PALETTE[i % len(PALETTE)]
        i += not _is_erasure(s)

    inside = lambda m: all(d in pos for d in m.dets)  # noqa: E731
    shown = [m for m in mechs if (sources is None or m.source in sources) and inside(m)]
    full_chain = shortest_logical(circuit, noise_by_type) if logical else []
    chain = [m for m in full_chain if inside(m)]
    # Only detectors some drawn mechanism can flip; the rest are trivial here.
    live = sorted({d for m in shown + chain for d in m.dets})
    if sheet:
        for flag, marker in ((False, "o"), (True, "s")):
            pts = [pos[d] for d in live if side[d] == flag]
            if pts:
                ax.scatter(*np.array(pts).T, s=22, marker=marker, c="#84837c", zorder=3)
    elif live:
        ax.scatter(*np.array([pos[d] for d in live]).T, s=10, c="#84837c", depthshade=False)
    # Erasure last, so it sits on top of the (much denser) circuit noise.
    for s in sorted(labels, key=_is_erasure):
        if sources is not None and s not in sources:
            continue
        group = [m for m in shown if m.source == s]
        # Many gates give the same edge; draw each (detectors, observables) once.
        distinct = {(m.dets, m.obs): m for m in group}.values()
        for flips, width in ((False, 0.9), (True, 2.2)):
            if _is_erasure(s):
                width += 0.8
            lines, spokes = _segments([m for m in distinct if bool(m.obs) == flips], pos, centre,
                                      boundary_len)
            _add_lines(ax, lines, colors=colours[s], linewidths=width, alpha=0.8)
            _add_lines(ax, spokes, colors=colours[s], linewidths=width, linestyles="dashed",
                       alpha=0.6)
        ax.plot([], [], color=colours[s], label=f"{s} ({len(distinct)} edges)")

    if logical:
        _add_lines(ax, _segments(chain, pos, centre, boundary_len)[0], colors="black",
                   linewidths=3.5, zorder=2)
        n_in = f", {len(chain)} in the sheet" if sheet else ""
        ax.plot([], [], color="black", lw=3, label=f"shortest logical ({len(full_chain)}{n_in})")

    ax.plot([], [], color="#84837c", lw=2.2, label="thick: flips an observable")
    # The frame spans every (interface) detector, so views of one circuit
    # filtered to different sources line up.
    lo, hi = xyz.min(axis=0) - boundary_len, xyz.max(axis=0) + boundary_len
    if sheet:
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel="along the seam", ylabel="t")
        ax.scatter([], [], marker="o", c="#84837c", label="one side of the seam")
        ax.scatter([], [], marker="s", c="#84837c", label="other side")
    else:
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), zlim=(lo[2], hi[2]),
               xlabel="x", ylabel="y", zlabel="t")
        # True proportions make time (one unit per layer) a tall needle; squash
        # it to at most 1.5x the footprint.
        dx, dy, dt = hi - lo
        ax.set_box_aspect((dx, dy, min(dt, 1.5 * max(dx, dy))))
    if sheet:  # below the axes: a sheet fills its frame
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), fontsize=8)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8)
    ax.set_title(title or f"{len(live)} of {len(coords)} detectors, {len(shown)} mechanisms")
    return fig, ax, mechs
