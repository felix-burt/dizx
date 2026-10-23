"""The suppression-factor grid: Lambda vs fusion erasure rate.

Trimmed port of DISQEC's scripts/lambda_plot_to_pgfplots.py. Each source is
one ``subthreshold_lambda`` config (a dict) at one local-noise level, with a
photon-loss grid in ``sweep.probs``. At every loss rate the LER-vs-distance
points are fitted to

    LER(d) = A * Lambda^(-(d+1)/2),

i.e. Lambda is the factor the LER drops by per d -> d+2 step. The fit is a
weighted least squares of ln LER on (d+1)/2 with weights = error counts (the
Poisson variance of ln LER is ~1/errors), so a handful-of-errors point barely
pulls it. Points with fewer than `min_errors` errors are left out.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import sinter

from _regime_common import csv_path_for
from figures.common import plain_label, write_figure_data
from run_threshold_experiment import boosted_erasure_probability


def _stats_by_p_distance(config: dict, local_prob: float, results_dir: Path,
                         extra_distances: list[int] | None = None,
                         ) -> tuple[dict, Path] | None:
    """{p: {d: (shots, errors)}} from the config's sinter CSV at local_prob, or
    None if missing.

    Only the config's own sweep.distances: a companion config (e.g. the d=9
    subthreshold_lambda_d9_* point) shares the CSV, and one loss rate fitted
    over more distances than its neighbours would put a kink in the curve.
    extra_distances opts in to more (a fit that asks for d=9 by name).
    """
    path = csv_path_for(config, local_prob, results_dir=results_dir)
    if not path.exists():
        return None
    distances = set(config["sweep"]["distances"]) | set(extra_distances or [])
    out: dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for s in sinter.read_stats_from_csv_files(path):
        md = s.json_metadata or {}
        if int(md["distance"]) not in distances:
            continue
        cell = out[float(md["p"])][int(md["distance"])]
        cell[0] += s.shots - s.discards
        cell[1] += s.errors
    return out, path


def loss_for_erasure(p_e: float, n: int) -> float:
    """Photon loss rate at which boost level n gives fusion erasure rate p_e
    (inverse of boosted_erasure_probability; needs p_e >= its n's floor)."""
    p_fail = 2.0 ** -n
    return 1.0 - ((1.0 - p_e) / (1.0 - p_fail / 2.0)) ** p_fail


def _n_boost(config: dict) -> int:
    """Boost level of the config's photon-loss (`probability: p_e`) channel."""
    (level,) = {ch["n_boost"] for ch in config["noise_model"] if ch.get("probability") == "p_e"}
    return level


_CLASSES: dict = {}


def _leading_order_lambda(config: dict, observable: str, d1: int, d2: int,
                          x_lo: float, x_hi: float, points: int = 60) -> list[tuple[float, float]]:
    """Leading-order loss-only Lambda over a log grid of erasure rates."""
    from figures.leading_order import erasure_edges, min_logical
    sums = {}
    for d in (d1, d2):
        key = (config["name"], observable, d)
        if key not in _CLASSES:
            classes, warnings = erasure_edges(config, d, observable)
            if warnings:
                raise ValueError(f"{config['name']} d={d}: {sorted(set(warnings))}")
            _CLASSES[key] = classes
        sums[d] = _CLASSES[key]
    curve = []
    for k in range(points):
        pe = math.exp(math.log(x_lo) + k * (math.log(x_hi) - math.log(x_lo)) / (points - 1))
        s1 = min_logical(sums[d1], lambda m: 1 - (1 - pe) ** m)[1]
        s2 = min_logical(sums[d2], lambda m: 1 - (1 - pe) ** m)[1]
        curve.append((pe, (s1 / s2) ** (2.0 / (d2 - d1))))
    return curve


def fit_lambda(points: dict[int, tuple[int, int]], min_errors: int,
               distances: list[int] | None = None,
               ) -> tuple[str, float, float, list[int]] | None:
    """('fit', Lambda, stderr, distances used) or ('lower', bound, 0, [d_lo, d_hi]).

    points: {d: (shots, errors)}. None when neither a fit nor a bound exists.
    distances restricts the fit to those distances.
    """
    if distances is not None:
        points = {d: v for d, v in points.items() if d in distances}
    usable = sorted((d, s, e) for d, (s, e) in points.items() if s > 0 and e >= min_errors)
    # A LER near 1/2 carries no distance information (above threshold / saturated).
    usable = [(d, s, e) for d, s, e in usable if e / s < 0.3]
    if len(usable) >= 2:
        xs = [(d + 1) / 2 for d, _, _ in usable]
        ys = [math.log(e / s) for _, s, e in usable]
        ws = [e for _, _, e in usable]
        w_sum = sum(ws)
        x_bar = sum(w * x for w, x in zip(ws, xs)) / w_sum
        y_bar = sum(w * y for w, y in zip(ws, ys)) / w_sum
        sxx = sum(w * (x - x_bar) ** 2 for w, x in zip(ws, xs))
        slope = sum(w * (x - x_bar) * (y - y_bar) for w, x, y in zip(ws, xs, ys)) / sxx
        lam = math.exp(-slope)
        return "fit", lam, lam / math.sqrt(sxx), [d for d, _, _ in usable]
    if len(usable) == 1:
        d_lo, s_lo, e_lo = usable[0]
        above = sorted((d, s) for d, (s, e) in points.items() if d > d_lo and e == 0 and s > 0)
        if above:
            d_hi, s_hi = above[0]
            ratio = (e_lo / s_lo) / (3.0 / s_hi)
            if ratio > 1:
                return "lower", ratio ** (2.0 / (d_hi - d_lo)), 0.0, [d_lo, d_hi]
    return None


def _series_label(spec: dict, series: dict) -> str:
    """The legend label of a series: the legend entry with its colour and mark."""
    if "label" in series:
        return plain_label(series["label"])
    for entry in spec.get("legend", []):
        if entry["color"] == series["color"] and entry["mark"] == series["mark"]:
            return plain_label(entry["label"])
    return series["color"]


def _render(spec: dict, *, results_dir: Path, min_errors: int, legend_name: str,
            data_rows: list | None = None) -> tuple[str, list[str], int]:
    """A groupplot of Lambda-vs-loss panels sharing one legend below the grid.

    spec::

        {"cols": 2, "width": "...", "height": "...",
         "xlabel": "...", "x": "loss",                # or "erasure"
         "cells": [{"title": "...", "ylabel": "...",
                    "series": [{"config": <dict>", "local_prob": 0.0, "color": "...",
                                "mark": "...", "line": "solid"}]}],
         "legend": [{"label": "...", "color": "...", "mark": "..."}],   # families
         "style_legend": [{"line": "dashed", "label": "..."}],          # line styles
         "legend_columns": 5,
         "fit_distances": [5, 7],           # optional: fit every point over these only
         "asymptote_distances": [3, 7],     # optional, erasure axis only
         "asymptote_x_max": 0.2,            # optional: leading order is a small-p_e result
         "asymptote_to_data": true,         # optional: end each curve at its family's first point
         "asymptote_overlap": 2,            # optional: ...or this many points further into the data
         "x_floor": 0.04,                   # optional: extend the x axis down to this
         "draw_bounds": true,               # optional: false leaves the lower bounds out
         "x_log": false,                    # optional, loss axis: logarithmic
         "asymptote_style": "semithick, densely dotted",  # optional
         "x_marks": [{"x": 0.125, "left": "$n=3$", "right": "$n=2$"}]}  # optional
                                            # labelled vertical lines, e.g. a switchover

    A source may also carry "pe_range": [lo, hi], a filter on the point's
    erasure rate on either axis, and "x_boost": n (loss axis only), which
    places each point at the loss rate boost level n needs to reach its
    erasure rate instead of the loss rate it was run at. With loss the only
    non-local noise a point depends on p_e alone, so one run gives a point on
    every boost level whose floor is at or below its p_e.

    A series may also carry "line_regions": [{"x_range": [lo, hi], "line":
    "dashed"}, ...]: its marks and error bars are then drawn once and its line
    in one piece per region (lo <= x < hi), in that region's style, e.g. the
    boost level that reaches that erasure rate.

    A series may instead carry "sources": [{"config", "local_prob",
    "x_range": [lo, hi]}, ...] (plus color/mark/line): the points of each
    source inside its x range (lo <= x < hi) are joined into one curve. On the
    erasure axis this stitches boost levels together: with loss the only
    noise, n only sets p_e, so equal-p_e points of different n are the same
    experiment.

    Without fit_distances, each point is fitted over every resolved,
    unsaturated distance, so a series whose smallest distance crosses the
    saturation cut changes distance set partway along and jumps wherever
    LER(d) is not a pure exponential (Bell XX: Lambda 3->5 is 3-9x Lambda
    5->7). A fixed pair keeps every point on the same step.

    A cell may also carry "asymptotes": [{"config": <dict>, "observable": "ZZ",
    "color": "..."}], each drawn as a leading-order Lambda curve (asymptote_style):
    (LER_lo(d1) / LER_lo(d2))^(2/(d2-d1)) with LER_lo(d) = 1/2 * sum over the
    minimum-weight erasure logicals of prod q (figures.leading_order),
    the same per-step Lambda the points are fitted to. They don't set the y
    range, so their steep low-p_e ends are clipped rather than squashing the data.

    x "loss" is the photon loss rate in percent; "erasure" is the boosted
    fusion erasure rate it maps to at the config's n_boost, which puts
    different boost levels on one axis; its coordinates stay rates (as do
    x_floor, asymptote_x_max and x_marks) but the tick labels read in percent,
    like the loss axis. Returns (body, header notes, curves
    drawn); every plotted point is also appended to `data_rows` (see main).
    """
    data_rows = [] if data_rows is None else data_rows
    cols = spec["cols"]
    cells = spec["cells"]
    rows = math.ceil(len(cells) / cols)
    erasure_x = spec.get("x", "loss") == "erasure"
    fit_distances = spec.get("fit_distances")
    notes: list[str] = []
    drawn = 0
    lines = [r"\begin{tikzpicture}",
             r"\begin{groupplot}[",
             f"  group style={{group size={cols} by {rows}, horizontal sep=0.35cm, "
             "vertical sep=0.9cm, xlabels at=edge bottom, ylabels at=edge left, "
             "xticklabels at=edge bottom, yticklabels at=edge left},",
             f"  width={spec['width']}, height={spec['height']}, scale only axis,",
             f"  xlabel={{{spec['xlabel']}}}, ymode=log, grid=both,",
             "  minor grid style={gray!12}, major grid style={gray!30},",
             "  title style={font=\\small}, every tick label/.append style={font=\\footnotesize},",
             "  error bars/error bar style={solid},",
             "]"]
    panels: list[tuple[list[str], list[str], list[dict], dict[str, float]]] = []
    all_values: list[float] = []
    x_min, x_max = math.inf, 0.0
    for i, cell in enumerate(cells):
        opts = []
        if cell.get("title"):
            opts.append(f"title={{{cell['title']}}}")
        if cell.get("ylabel"):
            opts.append(f"ylabel={{{cell['ylabel']}}}")
        body: list[str] = []
        # Plotted x per series colour, to find where each family's data starts.
        first_x: dict[str, list[float]] = {}
        if i == 0:
            opts.append(f"legend to name={legend_name}")
            opts.append(f"legend columns={spec.get('legend_columns', len(spec['legend']))}")
            opts.append("legend style={draw=none, /tikz/every even column/.append style={column sep=1em}}")
            for entry in spec["legend"]:
                body.append(rf"  \addlegendimage{{{entry['color']}, mark={entry['mark']}, thick}}")
                body.append(rf"  \addlegendentry{{{entry['label']}}}")
            for entry in spec.get("style_legend", []):
                body.append(rf"  \addlegendimage{{black, thick, {entry['line']}, line legend}}")
                body.append(rf"  \addlegendentry{{{entry['label']}}}")
        for series in cell["series"]:
            fits, bounds = [], []
            for source in series.get("sources", [series]):
                config = source["config"]
                loaded = _stats_by_p_distance(config, source["local_prob"], results_dir,
                                              fit_distances)
                if loaded is None:
                    print(f"skipping {config['name']}: no data at local_prob={source['local_prob']:g}")
                    continue
                tag = f"{config['name']} p_l={source['local_prob']:g}"
                by_p, csv = loaded
                x_lo_src, x_hi_src = source.get("x_range", [-math.inf, math.inf])
                pe_lo_src, pe_hi_src = source.get("pe_range", [-math.inf, math.inf])
                n_fits, n_bounds = len(fits), len(bounds)
                for p in sorted(by_p):
                    pe = boosted_erasure_probability(p, _n_boost(config))
                    if not pe_lo_src <= pe < pe_hi_src:
                        continue
                    if erasure_x:
                        x = pe
                    elif "x_boost" in source:
                        x = 100 * loss_for_erasure(pe, source["x_boost"])
                    else:
                        x = 100 * p
                    if not x_lo_src <= x < x_hi_src:
                        continue
                    result = fit_lambda(by_p[p], min_errors, fit_distances)
                    if result is None:
                        continue
                    kind, lam, err, ds = result
                    if kind == "fit":
                        data_rows.append([_series_label(spec, series), source["local_prob"], pe, lam, err])
                    x_min, x_max = min(x_min, x), max(x_max, x)
                    if kind == "fit":
                        fits.append((x, lam, err))
                        if len(ds) == 2 and fit_distances is None:
                            notes.append(f"% NOTE: {tag} loss={p:g}: Lambda from d={ds} only\n")
                    else:
                        if not spec.get("draw_bounds", True):
                            notes.append(f"% NOTE: {tag} loss={p:g}: Lambda >= {lam:.3g} "
                                         f"(d={ds[0]} vs zero errors at d={ds[1]}), not drawn\n")
                            continue
                        bounds.append((x, lam))
                        notes.append(f"% NOTE: {tag} loss={p:g}: lower bound from d={ds[0]} "
                                     f"vs zero errors at d={ds[1]}\n")
                    all_values.append(lam)
                print(f"{tag}: {len(fits) - n_fits} fitted, {len(bounds) - n_bounds} bounded "
                      f"loss rate(s) from {csv.name}")
            fits.sort()
            if fits:
                first_x.setdefault(series["color"], []).extend(x for x, _, _ in fits)
            if fits and series.get("line_regions"):
                # Marks and error bars in one plot, the line in one piece per
                # region, each carried on to the next region's first point so
                # the curve stays connected across a style change.
                body.append(rf"  \addplot[{series['color']}, mark={series['mark']}, only marks, "
                            r"mark options={solid}, forget plot,")
                body.append(r"    error bars/.cd, y dir=both, y explicit]")
                body.append("  coordinates {")
                body.extend(f"    ({x:g},{lam:.6g}) +- (0,{err:.6g})" for x, lam, err in fits)
                body.append("  };")
                for region in series["line_regions"]:
                    lo, hi = region["x_range"]
                    idx = [i for i, (x, _, _) in enumerate(fits) if lo <= x < hi]
                    if not idx:
                        continue
                    piece = fits[idx[0]:min(idx[-1] + 2, len(fits))]
                    if len(piece) < 2:
                        continue
                    body.append(rf"  \addplot[{series['color']}, thick, {region['line']}, "
                                r"no marks, forget plot] coordinates {"
                                + " ".join(f"({x:g},{lam:.6g})" for x, lam, _ in piece) + "};")
            elif fits:
                body.append(rf"  \addplot[{series['color']}, mark={series['mark']}, thick, "
                            rf"{series['line']}, mark options={{solid}}, forget plot,")
                body.append(r"    error bars/.cd, y dir=both, y explicit]")
                body.append("  coordinates {")
                body.extend(f"    ({x:g},{lam:.6g}) +- (0,{err:.6g})" for x, lam, err in fits)
                body.append("  };")
            if bounds and spec.get("draw_bounds", True):
                body.append(rf"  \addplot[{series['color']}, only marks, mark=triangle, "
                            r"mark options={solid, fill=white}, forget plot]")
                body.append("  coordinates {" + " ".join(f"({x:g},{lam:.6g})" for x, lam in bounds) + "};")
            if fits or bounds:
                drawn += 1
        for extra in cell.get("extra_points", []):
            # Points from outside the CSVs (e.g. rare-event estimates where
            # sampling has no errors): open marks in the series' colour.
            pts = [pt for pt in extra["coords"] if pt[1] > 0]
            if not pts:
                continue
            data_rows += [[extra.get("label", extra["color"]), None, x, lam, err] for x, lam, err in pts]
            body.append(rf"  \addplot[{extra['color']}, only marks, mark={extra['mark']}, "
                        r"mark options={solid, fill=white}, forget plot,")
            body.append(r"    error bars/.cd, y dir=both, y explicit]")
            body.append("  coordinates {" + " ".join(
                f"({x:g},{lam:.6g}) +- (0,{err:.6g})" for x, lam, err in pts) + "};")
            all_values.extend(lam for _, lam, _ in pts)
            x_min = min([x_min] + [x for x, _, _ in pts])
        panels.append((opts, body, cell.get("asymptotes", []), first_x))

    y_lo = min([v for v in all_values if v > 0] + [1.0]) / 1.5
    y_hi = max(all_values + [10.0]) * 1.5
    if any(asym for _, _, asym, _ in panels):
        # Headroom for the steep leading-order curves (Lambda ~ p_e^-4 for the
        # weight-2d families), so their crossings stay on the panel.
        y_hi *= 5
    # The erasure axis is logarithmic: loss-only Lambda ~ p_e^-(w(d+2)-w(d))
    # deep below threshold, so each curve's slope there reads off its
    # erasure-distance increment (2 for weight-d families, 4 for weight-2d).
    loss_log = not erasure_x and spec.get("x_log", False)
    if erasure_x:
        x_lo, x_hi = min(x_min / 1.1, spec.get("x_floor", math.inf)), x_max * 1.1
    elif loss_log:
        x_lo, x_hi = x_min / 1.3, x_max * 1.2
    else:
        x_lo, x_hi = 0.0, x_max * 1.04
    d1, d2 = spec.get("asymptote_distances", [3, 7])
    asym_style = spec.get("asymptote_style", "semithick, densely dotted")
    for opts, body, asymptotes, first_x in panels:
        opts.append(f"xmin={x_lo:.3g}, xmax={x_hi:.3g}, ymin={y_lo:.3g}, ymax={y_hi:.3g}")
        if erasure_x:
            # Only ticks inside the range: pgfplots pairs xticklabels with the
            # ticks it actually draws, so out-of-range entries shift the labels.
            ticks = [t for t in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5) if x_lo <= t <= x_hi]
            opts.append("xmode=log, log basis x=10, "
                        f"xtick={{{','.join(f'{t:g}' for t in ticks)}}}, "
                        f"xticklabels={{{','.join(f'{100 * t:g}' for t in ticks)}}}")
        elif loss_log:
            ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20)
                     if x_lo <= t <= x_hi]
            opts.append("xmode=log, log basis x=10, "
                        f"xtick={{{','.join(f'{t:g}' for t in ticks)}}}, "
                        f"xticklabels={{{','.join(f'{t:g}' for t in ticks)}}}")
        lines.append(r"\nextgroupplot[" + ", ".join(opts) + "]")
        lines.extend(body)
        for asym in asymptotes:
            asym_hi = min(x_hi, spec.get("asymptote_x_max", x_hi))
            if spec.get("asymptote_to_data") and asym["color"] in first_x:
                # Hand over to the data exactly where it starts.
                # (optionally a few points in, so the two visibly overlap).
                xs = sorted(first_x[asym["color"]])
                asym_hi = xs[min(spec.get("asymptote_overlap", 0), len(xs) - 1)]
            curve = _leading_order_lambda(asym["config"], asym["observable"], d1, d2,
                                          x_lo, asym_hi)
            lines.append(rf"  \addplot[{asym['color']}, {asym_style}, forget plot] coordinates {{"
                         + " ".join(f"({x:.4g},{lam:.5g})" for x, lam in curve) + "};")
            notes.append(f"% leading-order Lambda (d={d1}->{d2}): {asym['config']['name']} "
                         f"{asym['observable']}\n")
        # Lambda = 1 is threshold: below it, distance no longer helps. Spans
        # exactly the x range: a fixed wide one overflows TeX's dimensions on
        # the erasure axis, whose units are ~20x smaller than the loss axis's.
        lines.append(rf"  \addplot[black!60, densely dashdotted, forget plot] "
                     rf"coordinates {{({x_lo:.3g},1) ({x_hi:.3g},1)}};")
        for mark in spec.get("x_marks", []):
            lines.append(rf"  \draw[black!45, thin, dashed] (axis cs:{mark['x']:g},{y_lo:.3g}) "
                         rf"-- (axis cs:{mark['x']:g},{y_hi:.3g});")
            for side, anchor in (("left", "north east"), ("right", "north west")):
                if mark.get(side):
                    lines.append(rf"  \node[anchor={anchor}, font=\scriptsize, text=black!60, "
                                 rf"inner sep=2pt] at (axis cs:{mark['x']:g},{y_hi:.3g}) "
                                 rf"{{{mark[side]}}};")
    lines.append(r"\end{groupplot}")
    # Centred under the bottom row: the middle panel's foot, or with an even
    # column count the middle of the gap (half the 0.35cm sep) between the two
    # central panels.
    anchor = (f"[yshift=-0.9cm]group c{(cols + 1) // 2}r{rows}.south" if cols % 2
              else f"[xshift=0.175cm, yshift=-0.9cm]group c{cols // 2}r{rows}.south east")
    lines.append(rf"\node[anchor=north] at ({anchor}) "
                 rf"{{\pgfplotslegendfromname{{{legend_name}}}}};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines), notes, drawn


def render(spec: dict, *, name: str, out_dir: Path, results_dir: Path, min_errors: int = 5) -> Path:
    """Write `<out_dir>/<name>.tex` and its `_data.csv` (see _render for the spec)."""
    data_rows: list = []
    body, _, drawn = _render(spec, results_dir=results_dir, min_errors=min_errors,
                             legend_name=f"legend_{name}", data_rows=data_rows)
    if drawn == 0:
        raise ValueError(f"{name}: no panel has any data")
    out = out_dir / f"{name}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + "\n")
    print(f"Wrote {out} ({drawn} curve(s))")
    write_figure_data(out, ["series", "local_prob", "p_e", "Lambda", "stderr"], data_rows)
    return out
