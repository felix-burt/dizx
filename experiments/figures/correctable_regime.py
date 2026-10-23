"""The correctable-regime figures: erasure threshold vs local noise level.

Trimmed port of DISQEC's scripts/correctable_regime_to_pgfplots.py, in the
mode the paper used (thresholds from the scaling-window fit, measured points
joined by straight lines). Each curve is one crg config: its points are the
valid `scaling_fit.crg_points`, plus the p_l = 0 point from the matching
pure-erasure sweep (`scaling_fit.erasure_point`). With `extend_to_axis` a
curve that stops above zero is carried on to the x axis, dashed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from _regime_common import load_stats_by_local_prob
from figures.common import plain_label, write_figure_data
from figures.scaling_fit import crg_points, erasure_point


@dataclass
class _Series:
    """One plotted config: a line plus a real-data overlay.

    `line_points`/`line_is_real` describe the drawn curve -- solid wherever
    `line_is_real` is True (a measured point), dashed elsewhere (the
    extend-to-axis extrapolation). `scatter_points`/`scatter_errs` are the
    measurements themselves with their error bars. `is_flat` marks a
    horizontal reference line (unused by the paper's figures).
    """
    label: str
    color: str
    marker: str
    is_flat: bool
    line_points: list[tuple[float, float]]
    line_is_real: list[bool]
    scatter_points: list[tuple[float, float]] = field(default_factory=list)
    scatter_errs: list[float | None] = field(default_factory=list)


def _decimal_places_for(values: list[float]) -> int:
    """Decimal digits needed so consecutive distinct x values render as
    distinct fixed-point tick labels (e.g. a 0.0005 step needs 4 places).
    """
    distinct = sorted(set(values))
    if len(distinct) < 2:
        return 2
    step = min(b - a for a, b in zip(distinct, distinct[1:]))
    if step <= 0:
        return 2
    return max(2, -math.floor(math.log10(step)))


def _closed_at(stats_by_local_prob: dict, distances: list[int], after: float) -> float | None:
    """The first local_prob above `after` measured with the correctable region
    already closed: at its lowest sampled erasure rate the larger of the two
    smallest distances is no better than the smaller (above threshold
    everywhere)."""
    d_lo, d_hi = sorted(distances)[:2]
    for lp in sorted(lp for lp in stats_by_local_prob if lp > after):
        tot: dict[tuple[int, float], list[int]] = {}
        for st in stats_by_local_prob[lp]:
            key = (int(st.json_metadata["distance"]), float(st.json_metadata["p"]))
            t = tot.setdefault(key, [0, 0])
            t[0] += st.shots - st.discards
            t[1] += st.errors
        ps = sorted(p for d, p in tot if d == d_lo and (d_hi, p) in tot and tot[(d_lo, p)][0] > 0)
        if not ps:
            continue
        (n_lo, e_lo), (n_hi, e_hi) = tot[(d_lo, ps[0])], tot[(d_hi, ps[0])]
        return lp if e_hi / n_hi >= e_lo / n_lo else None
    return None


def _extend_to_axis(points: list[tuple[float, float]], is_real: list[bool],
                    closed_by: float | None = None) -> None:
    """Carry a curve that stops above zero on to the x axis: a straight line
    through its last three points, to where it reaches threshold zero (the
    correctable region closing) -- but no further than `closed_by`, a noise
    level measured with the region already closed (the curves steepen at the
    end, so the straight line can overshoot it). Appended as a non-real point,
    so it is drawn dashed, unshaded and unmarked, and never reaches the data
    file."""
    tail = points[-3:]
    if len(tail) < 2 or tail[-1][1] <= 0:
        return
    slope, intercept = np.polyfit([x for x, _ in tail], [y for _, y in tail], 1)
    if slope >= 0:
        return
    x0 = -intercept / slope
    if closed_by is not None:
        x0 = min(x0, closed_by)
    if x0 > tail[-1][0]:
        points.append((float(x0), 0.0))
        is_real.append(False)


def _style_runs(
    points: list[tuple[float, float]], is_real: list[bool],
) -> list[tuple[bool, list[tuple[float, float]]]]:
    """Splits an x-ordered curve into maximal contiguous runs of matching
    `is_real`, each run after the first prefixed with the previous run's
    last point so solid/dashed segments visually connect with no gap at the
    style transition (a curve can alternate more than once -- e.g. real data,
    a gap, real data again, then a trailing extrapolation).
    """
    runs: list[tuple[bool, list[tuple[float, float]]]] = []
    i = 0
    n = len(points)
    while i < n:
        j = i
        while j + 1 < n and is_real[j + 1] == is_real[i]:
            j += 1
        seg = list(points[i:j + 1])
        if runs:
            seg = [points[i - 1]] + seg
        runs.append((is_real[i], seg))
        i = j + 1
    return runs


def _render_axis(
    series: list[_Series], title: str | None, xlabel: str, ylabel: str,
    *, tick_xs: list[float] | None = None, y_floor: float = 0.5, legend_pos: str = "north east",
    shade_extrapolated: bool = False,
) -> str:
    all_xs = [x for s in series for x, _ in s.line_points]
    # Tick-label decimal precision reflects the configured local_prob step
    # (e.g. 0.0005), not whichever points happen to be drawn.
    precision_xs = tick_xs if tick_xs is not None else all_xs
    # y_floor (default 0.5, the natural ceiling for a non-local erasure
    # threshold) is only a minimum: a pure-erasure local_prob=0 intercept can
    # sit just above it (e.g. 0.505) -- and `restrict y to domain` below would
    # silently drop that point -- so the axis grows to fit the data.
    all_ys = [y for s in series for _, y in s.line_points + s.scatter_points]
    y_top = max(y_floor, math.ceil(max(all_ys, default=0.0) * 20) / 20)
    # One tick per real sweep step, thinned to at most ~10 ticks: a range
    # starting at 0 with a 0.001 step otherwise gets pgfplots' own 0.0005
    # ticks, whose 4-decimal labels overrun each other.
    distinct = sorted(set(precision_xs))
    steps = [b - a for a, b in zip(distinct, distinct[1:]) if b - a > 1e-12]
    xtick_opt = []
    if steps:
        step = round(min(steps), 12)
        n_ticks = (max(all_xs) - min(all_xs)) / step
        xtick_opt = [f"xtick distance={step * max(1, math.ceil(n_ticks / 10)):.6g}"]
    lines: list[str] = []
    lines.append(r"\begin{tikzpicture}")
    axis_opts = [
        r"width=\linewidth",
        f"xlabel={{{xlabel}}}",
        f"ylabel={{{ylabel}}}",
        "grid=both",
        "minor grid style={gray!15}",
        "major grid style={gray!30}",
        f"legend pos={legend_pos}",
        "legend cell align=left",
        "legend style={font=\\footnotesize}",
        "every axis plot/.append style={thick}",
        "ymin=0",
        f"ymax={y_top:.6g}",
        # Some finite-size-scaling fits (esp. with only 2 distances) produce a
        # degenerate stderr many orders of magnitude too large -- the error
        # bar's raw coordinate then overflows TeX's dimension registers
        # before ymin/ymax ever gets a chance to clip it on-screen. Clipping
        # the data itself to the axis domain avoids that crash and just
        # truncates the bar at the axis edge instead.
        f"restrict y to domain=0:{y_top:.6g}",
        f"xmin={min(all_xs):.6g}",
        f"xmax={max(all_xs):.6g}",
        *xtick_opt,
        "scaled x ticks=false",
        # Plain decimals on y too: below 0.1 pgfplots otherwise switches to
        # `8 \cdot 10^{-2}` labels, which a low ymax (e.g. 0.2) exposes.
        "scaled y ticks=false",
        "y tick label style={/pgf/number format/fixed}",
        "x tick label style={/pgf/number format/fixed, "
        f"/pgf/number format/precision={_decimal_places_for(precision_xs)}}}",
    ]
    if title:
        axis_opts.append(f"title={{{title}}}")
    lines.append(r"\begin{axis}[" + ",\n  ".join(axis_opts) + "]")

    for s in series:
        runs = _style_runs(s.line_points, s.line_is_real)
        # Markers live on the line itself only when there's no separate real-
        # data overlay (flat references, and the no-fit raw-points fallback,
        # where the line already *is* the real data).
        # (When the overlay is the line's own points -- raw points with error
        # bars -- the line keeps its marks too, so the legend shows them.)
        separate_overlay = s.scatter_points and not set(s.scatter_points) <= set(s.line_points)
        mark_opt = "" if (s.is_flat or separate_overlay) else f", mark={s.marker}"

        for idx, (is_real, pts) in enumerate(runs):
            coords = "\n        ".join(f"({x:.6g},{y:.6g})" for x, y in pts)
            # Extrapolated runs carry no marks: their points aren't measurements.
            dash_opt = "" if is_real else ", dashed, mark=none"
            shade = is_real or shade_extrapolated
            name_path_opt = ", name path=upper" if shade else ""
            forget_opt = "" if idx == 0 else ", forget plot"
            lines.append(
                rf"  \addplot[color={s.color}{mark_opt}{dash_opt}{name_path_opt}{forget_opt}] coordinates {{"
            )
            lines.append("        " + coords)
            lines.append("  };")
            if idx == 0:
                lines.append(rf"  \addlegendentry{{{s.label}}}")
            if shade and pts[-1][0] > pts[0][0]:
                # (A single-point run -- e.g. a lone local_prob=0 intercept
                # followed by fitted-only grid points -- has nothing to shade,
                # and a zero-width `fill between` is a TeX "Missing number".)
                # Shade only under real-backed segments -- an extrapolated
                # stretch hasn't earned a "confirmed correctable" claim --
                # unless asked to (--extend-to-axis: the region runs on to
                # where the dashed line closes it).
                # `lower` is bounded to *this run's* x-range, not the full
                # axis: a `lower` spanning the whole axis width makes
                # `fill between` close the polygon across the entire domain
                # for every run, which for a short, isolated interior run
                # (real data resuming briefly after a gap) produces a
                # criss-crossing wedge instead of a plain trapezoid under
                # that segment.
                x_lo, x_hi = pts[0][0], pts[-1][0]
                lines.append(rf"  \path[name path=lower] (axis cs:{x_lo:.6g},0) -- (axis cs:{x_hi:.6g},0);")
                lines.append(rf"  \addplot[opacity=0.2, {s.color}, forget plot] fill between [of=upper and lower];")

        if s.scatter_points:
            has_err = any(e is not None for e in s.scatter_errs)
            if has_err:
                # A degenerate finite-size-scaling fit (esp. with only 2
                # distances) can report a stderr many orders of magnitude too
                # large. `restrict y to domain` only filters the marker
                # itself, not the error-bar delta, so an unclipped huge delta
                # still asks pgfplots for a coordinate that overflows TeX's
                # dimension registers ("Dimension too large") -- clip it here
                # to something comfortably outside any real threshold value
                # but safely representable.
                coords = "\n        ".join(
                    f"({x:.6g},{y:.6g}) +- (0,{min(e or 0.0, 1.0):.6g})"
                    for (x, y), e in zip(s.scatter_points, s.scatter_errs)
                )
                err_opt = ", error bars/.cd, y dir=both, y explicit"
            else:
                coords = "\n        ".join(f"({x:.6g},{y:.6g})" for x, y in s.scatter_points)
                err_opt = ""
            lines.append(
                rf"  \addplot[only marks, mark={s.marker}, color={s.color}, forget plot{err_opt}] coordinates {{"
            )
            lines.append("        " + coords)
            lines.append("  };")

    lines.append(r"\end{axis}")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)



def render(curves: list[dict], *, name: str, out_dir: Path, results_dir: Path,
           xlabel: str, ylabel: str, legend_pos: str = "north east",
           extend_to_axis: bool = False, ymax: float = 0.5) -> Path:
    """Write `<out_dir>/<name>.tex` and its `_data.csv`.

    curves: [{"label", "color", "mark", "config": crg config, "erasure":
    pure-erasure config (its threshold is the p_l = 0 point)}]. The x axis is
    in percent. A curve with no valid point is left out."""
    series: list[_Series] = []
    figure_rows = []
    # The configured local_probs set the x tick step, not the plotted points.
    tick_xs: list[float] = []
    for curve in curves:
        config = curve["config"]
        known = {lp: (f.p_th, f.stderr, "scaling")
                 for lp, f in crg_points(config, results_dir=results_dir).items() if f.valid}
        if not known:
            print(f"Warning: skipping {config['name']} -- no valid threshold")
            continue
        tick_xs.extend(config["sweep"]["local_probs"])
        fit = erasure_point(curve["erasure"], results_dir=results_dir)
        if fit is not None and fit.valid:
            known[0.0] = (fit.p_th, fit.stderr, "scaling")
            tick_xs.append(0.0)
        else:
            print(f"Warning: {config['name']}: no valid local_prob=0 point from "
                  f"{curve['erasure']['name']} ({fit.reason if fit is not None else 'no data'})")
        xs = sorted(known)
        line_points = [(lp, known[lp][0]) for lp in xs]
        line_is_real = [True] * len(line_points)
        # The same points again as the marker overlay, each with its fit's
        # stderr as an error bar.
        scatter_points = list(line_points)
        scatter_errs = [known[lp][1] for lp in xs]
        if extend_to_axis:
            closed_by = _closed_at(load_stats_by_local_prob(config, results_dir=results_dir),
                                   config["sweep"]["distances"], after=xs[-1])
            _extend_to_axis(line_points, line_is_real, closed_by)
        if len(line_points) < 2:
            print(f"Warning: skipping {config['name']} -- fewer than 2 points")
            continue
        figure_rows += [(plain_label(curve["label"]), lp, known[lp][0], known[lp][1]) for lp in xs]
        series.append(_Series(curve["label"], curve["color"], curve["mark"], False,
                              line_points, line_is_real, scatter_points, scatter_errs))
    if not series:
        raise ValueError(f"{name}: no curve has data")

    # Percent axis. round() drops float noise (0.07 * 100 = 7.000000000000001)
    # that would otherwise inflate the inferred tick-label precision.
    def pct(x: float) -> float:
        return round(100 * x, 10)
    for s in series:
        s.line_points = [(pct(x), y) for x, y in s.line_points]
        s.scatter_points = [(pct(x), y) for x, y in s.scatter_points]
    body = _render_axis(series, title="", xlabel=xlabel, ylabel=ylabel, legend_pos=legend_pos,
                        shade_extrapolated=extend_to_axis, tick_xs=[pct(x) for x in tick_xs],
                        y_floor=ymax)
    out = out_dir / f"{name}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + "\n")
    print(f"Wrote {out} ({len(series)} series)")
    write_figure_data(out, ["series", "local_prob", "threshold", "stderr"], figure_rows)
    return out
