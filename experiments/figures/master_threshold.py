"""The master erasure-threshold figure: LER vs erasure rate, one cell per family.

Trimmed port of DISQEC's scripts/master_threshold_pgf.py (its --grid-spec
mode). Each cell is one resource-state family; colour encodes the distance
(shared across cells), line style the observable, and a small mark on the
bottom axis the fitted threshold. Each row shares one y range; every cell has
the same x width.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import sinter

from figures.common import plain_label, write_figure_data

# quantity 'threshold' (distance and p empty) or 'ler'; errors are the drawn
# bars (stderr for a threshold, the binomial interval for an LER).
_DATA_COLUMNS = ["series", "quantity", "distance", "p", "value", "err_minus", "err_plus"]

# Distance -> colour, shared across every series so d=3 is always the same
# colour whichever CSV/mark it appears under.
_COLOR_CYCLE = [
    "blue", "red", "OliveGreen", "purple!80", "Purple",
    "Brown", "Cyan", "Magenta", "Gray", "Olive",
]
_MLF = 1000.0   # sinter.fit_binomial max_likelihood_factor for the error bars


def _stat_p(s: sinter.TaskStats) -> float:
    md = s.json_metadata or {}
    if "p" not in md:
        raise ValueError(
            f"TaskStats {s.strong_id} missing 'p' in json_metadata; cannot plot."
        )
    return float(md["p"])


def _stat_d(s: sinter.TaskStats) -> int:
    md = s.json_metadata or {}
    return int(md.get("d", md.get("distance", 0)))


def _merge_by_p(stats: list[sinter.TaskStats]) -> list[sinter.TaskStats]:
    by_p: dict[float, sinter.TaskStats] = {}
    for s in stats:
        p = _stat_p(s)
        if p not in by_p:
            by_p[p] = s
        else:
            prev = by_p[p]
            by_p[p] = sinter.TaskStats(
                strong_id=prev.strong_id,
                decoder=prev.decoder,
                json_metadata=prev.json_metadata,
                shots=prev.shots + s.shots,
                errors=prev.errors + s.errors,
                discards=prev.discards + s.discards,
                seconds=prev.seconds + s.seconds,
                custom_counts=prev.custom_counts,
            )
    return [by_p[p] for p in sorted(by_p)]


def _group_by_distance(stats: list[sinter.TaskStats]) -> dict[int, list[sinter.TaskStats]]:
    grouped: dict[int, list[sinter.TaskStats]] = defaultdict(list)
    for s in stats:
        grouped[_stat_d(s)].append(s)
    return {d: _merge_by_p(items) for d, items in sorted(grouped.items())}


def _ler_with_bars(
    s: sinter.TaskStats, *, include_discards: bool, mlf: float,
) -> tuple[float, float, float] | None:
    shots = s.shots - (0 if include_discards else s.discards)
    hits = s.errors + (s.discards if include_discards else 0)
    if shots <= 0:
        return None
    fit = sinter.fit_binomial(num_shots=shots, num_hits=hits, max_likelihood_factor=mlf)
    return fit.best, fit.low, fit.high


# ---------------------------------------------------------------------------
# TeX emission
# ---------------------------------------------------------------------------

# Relabels each x tick as 100 * p, leaving coordinates (and every p window
# computed from them) in p. Fine for the erasure-rate scale: TeX's fixed-point
# error on \tick is far below the printed precision.
_PERCENT_XTICKLABEL = r"xticklabel={\pgfmathparse{100*\tick}\pgfmathprintnumber{\pgfmathresult}}"

def _coordinates_block(
    points: list[tuple[float, float, float, float]],
    *, min_p: float | None, max_p: float | None,
) -> str:
    lines = []
    for p, y, lo, hi in points:
        plus = max(hi - y, 0.0)
        minus = max(y - lo, 0.0)
        text = f"({p:.6g},{y:.6g}) +- ({plus:.6g},{minus:.6g})"
        if (min_p is not None and p < min_p) or (max_p is not None and p > max_p):
            text = f"% {text}"
        lines.append(text)
    return "\n        ".join(lines)


def _panel_points(
    panel: list[tuple[str, str, dict[int, list[sinter.TaskStats]], float | None, float | None]],
    *, mlf: float, include_discards: bool,
) -> list[tuple[str, str, int, str, list[tuple[float, float, float, float]], float | None, float | None]]:
    """(label, mark, d, upper-label, points, min_p, max_p) for each
    (series, distance) pair in a panel that has at least one plottable point.
    """
    out = []
    for label, mark, by_d, min_p, max_p in panel:
        for d, stats in by_d.items():
            points: list[tuple[float, float, float, float]] = []
            for s in stats:
                r = _ler_with_bars(s, include_discards=include_discards, mlf=mlf)
                if r is None:
                    continue
                best, low, high = r
                points.append((_stat_p(s), best, low, high))
            if points:
                out.append((label, mark, d, label.upper(), points, min_p, max_p))
    return out


def _shared_x_range(
    panels: list[list[tuple[str, str, dict[int, list[sinter.TaskStats]], float | None, float | None]]],
    *, mlf: float, include_discards: bool,
) -> tuple[float, float] | None:
    """Min/max p across every *included* (non-windowed-out) point in every
    panel, so multi-panel output can force the same x-axis range on each row.
    """
    included_ps = []
    for panel in panels:
        for _, _, _, _, points, min_p, max_p in _panel_points(panel, mlf=mlf, include_discards=include_discards):
            for p, _, _, _ in points:
                if (min_p is not None and p < min_p) or (max_p is not None and p > max_p):
                    continue
                included_ps.append(p)
    return (min(included_ps), max(included_ps)) if included_ps else None


def _crossing_y(
    by_d: dict[int, list[sinter.TaskStats]], p_th: float, *, mlf: float, include_discards: bool,
) -> float | None:
    """Where a series' distance curves meet at `p_th`: each curve's LER
    interpolated there (geometrically, the y axis being logarithmic), averaged
    over the distances. None if `p_th` falls outside the collected points.
    """
    import math
    ys = []
    for stats in by_d.values():
        pts = []
        for s in stats:
            r = _ler_with_bars(s, include_discards=include_discards, mlf=mlf)
            if r is not None and r[0] > 0:
                pts.append((_stat_p(s), r[0]))
        pts.sort()
        for (p_a, y_a), (p_b, y_b) in zip(pts, pts[1:]):
            if p_a <= p_th <= p_b:
                f = 0.0 if p_b == p_a else (p_th - p_a) / (p_b - p_a)
                ys.append(math.log(y_a) + f * (math.log(y_b) - math.log(y_a)))
                break
    return math.exp(sum(ys) / len(ys)) if ys else None


def _render_grid(
    cells: list[tuple[str, list[tuple[str, dict[int, list[sinter.TaskStats]], str, list | None]]] | None],
    *, cols: int, style_legend: list[dict], legend_position: str,
    min_p: float | None, max_p: float | None,
    xlabel: str, ylabel: str, xmode: str, ymode: str,
    width: str, height: str, mlf: float, include_discards: bool, legend_suffix: str,
    x_percent: bool = False, center_last_row: bool = False,
) -> str:
    """cells: row-by-row (title, [(mark, by_distance, line style, threshold)])
    or None for an empty cell; threshold is an optional [estimate, stderr]
    drawn as a small mark on the bottom axis. See --grid-spec.

    Rows share a y range so their curves compare directly across a row, with
    y tick labels on the left column only. Each
    cell gets its own x window, but every window is as wide (in p) as the
    widest cell, so with `scale only axis` equal-size boxes the x scale is
    the same everywhere -- a sweep over 11 points at step 0.01 occupies the
    same fraction of every cell. Each cell is titled with its resource state,
    which stands in for the marker legend of the stacked layout.

    center_last_row drops the last row's trailing empty cells and centres
    its panels under the rows above.

    The legend is a single-row strip centred above (legend_position="top") or
    below ("bottom") the whole grid, so the cells can use the full text width.
    """
    n_rows = -(-len(cells) // cols)
    all_distances = sorted({d for c in cells if c is not None for _, by_d, _, _ in c[1] for d in by_d})
    color_of = {d: _COLOR_CYCLE[i % len(_COLOR_CYCLE)] for i, d in enumerate(all_distances)}
    dlegend = f"dlegend_{legend_suffix}"
    group_name = f"group{legend_suffix}"

    def cell_curves(cell):
        """(mark, line, d, all points, points inside the p window)."""
        out = []
        for mark, by_d, line, _ in cell[1]:
            for _, _, d, _, points, _, _ in _panel_points(
                [("", mark, by_d, min_p, max_p)], mlf=mlf, include_discards=include_discards
            ):
                kept = [pt for pt in points
                        if not ((min_p is not None and pt[0] < min_p) or (max_p is not None and pt[0] > max_p))]
                out.append((mark, line, d, points, kept))
        return out

    curves = [cell_curves(c) if c is not None else [] for c in cells]
    # (mark, (threshold, stderr)) per cell, for the series that carry one.
    thresholds = [[(thr[0], _crossing_y(by_d, thr[0], mlf=mlf, include_discards=include_discards))
                   for _, by_d, _, thr in c[1] if thr] if c is not None else []
                  for c in cells]
    x_lo: list[float | None] = []
    spans = []
    for cell_c in curves:
        ps = [pt[0] for *_, kept in cell_c for pt in kept]
        x_lo.append(min(ps) if ps else None)
        if ps:
            spans.append(max(ps) - min(ps))
    span = max(spans) if spans else 1.0
    pad = 0.05 * span

    y_ranges = []
    for r in range(n_rows):
        ys = [v for cell_c in curves[r * cols:(r + 1) * cols] for *_, kept in cell_c
              for _, y, lo, hi in kept for v in (lo, y, hi) if ymode != "log" or v > 0]
        if not ys:
            y_ranges.append(None)
        elif ymode == "log":
            y_ranges.append((min(ys) / 1.5, max(ys) * 1.5))
        else:
            ypad = 0.05 * (max(ys) - min(ys))
            y_ranges.append((min(ys) - ypad, max(ys) + ypad))

    lines = [r"\begin{tikzpicture}", r"\begin{groupplot}["]
    lines.append(
        f"  group style={{group name={group_name}, group size={cols} by {n_rows}, "
        "yticklabels at=edge left, ylabels at=edge left, "
        "xlabels at=edge bottom, horizontal sep=8pt, vertical sep=1.4cm},"
    )
    lines.append("  " + ",\n  ".join([
        "scale only axis",
        f"width={width}",
        f"height={height}",
        f"xmode={xmode}",
        f"ymode={ymode}",
        f"xlabel={{{xlabel}}}",
        f"ylabel={{{ylabel}}}",
        "grid=both",
        "minor grid style={gray!15}",
        "major grid style={gray!30}",
        "title style={font=\\small, yshift=-4pt}",
        "legend cell align=left",
        # Space between entries of the one-row legend strip.
        "legend style={font=\\footnotesize, /tikz/every even column/.append style={column sep=8pt}}",
        "every axis plot/.append style={line width=0.5pt, mark size=1.5pt}",
        "error bars/y dir=both",
        "error bars/y explicit",
        "scaled x ticks=false",
        "x tick label style={/pgf/number format/fixed, font=\\footnotesize}",
        *([_PERCENT_XTICKLABEL] if x_percent else []),
    ]) + ",")
    lines.append(r"]")
    lines.append("")

    # Centring the last row: shift each of its panels (groupplots places
    # every cell from the grid, not from its neighbour) by half the width of
    # the trailing empty cells.
    last_start = (n_rows - 1) * cols
    last_row = cells[last_start:]
    trailing = cols - len(last_row) + next(
        (i for i, c in enumerate(reversed(last_row)) if c is not None), len(last_row))
    center_shift = (rf"{{{trailing / 2:g}*({width}+8pt)}}"
                    if center_last_row and 0 < trailing < cols else None)

    legend_placed = False
    for index, cell in enumerate(cells):
        y_range = y_ranges[index // cols]
        if cell is None and center_shift and index >= last_start:
            continue  # the trailing empties of a centred last row
        if cell is None:
            lines.append(r"\nextgroupplot[group/empty plot]")
            lines.append("")
            continue
        title = cell[0]
        opts = [f"title={{{title}}}"]
        if center_shift and index >= last_start:
            opts.append(f"xshift={center_shift}")
        if x_lo[index] is not None:
            opts.append(f"xmin={x_lo[index] - pad:.6g}, xmax={x_lo[index] + span + pad:.6g}")
        if y_range is not None:
            opts.append(f"ymin={y_range[0]:.6g}, ymax={y_range[1]:.6g}")
        if not legend_placed:
            columns = 1 if legend_position == "side" else -1
            opts.append(f"legend to name={dlegend}, legend columns={columns}, legend style={{draw=none}}")
        lines.append(rf"\nextgroupplot[{', '.join(opts)}]")
        for p_th, y_cross in thresholds[index]:
            # The series' threshold estimate: a dotted drop line from where its
            # distance curves cross down to the axis.
            lines.append(f"  % {title.upper()} threshold estimate p* = {p_th:.6g}")
            top = y_cross if y_cross is not None else (y_range[1] if y_range else 1.0)
            lines.append(rf"  \addplot[color=black, dotted, mark=none, forget plot] coordinates {{"
                         rf"({p_th:.6g},{y_range[0]:.6g}) ({p_th:.6g},{top:.6g})}};")
        for mark, line, d, points, _ in curves[index]:
            lines.append(f"  % {title.upper()} d={d} (mark={mark}, {line})")
            # `mark options={solid}` keeps a dashed curve's marks undashed.
            lines.append(rf"  \addplot[color={color_of[d]}, mark={mark}, {line}, mark options={{solid}}, forget plot] coordinates {{")
            lines.append("        " + _coordinates_block(points, min_p=min_p, max_p=max_p))
            lines.append("  };")
        if not legend_placed:
            lines.append(r"  % Legend: d values (colour encodes d)")
            for d in all_distances:
                lines.append(rf"  \addlegendimage{{color={color_of[d]}, line legend, mark=none}}")
                lines.append(rf"  \addlegendentry{{$d={d}$}}")
            legend_placed = True
        lines.append("")

    lines.append(r"\end{groupplot}")
    lines.append("")

    # Style legend: each group of entries (e.g. the observables, the
    # interfaces) stacked in its own column, beside the one-row d legend.
    # `transpose legend` fills column by column, so shorter groups are padded
    # with blank entries to keep every group in a single column.
    slegend = f"stylelegend_{legend_suffix}"
    groups = [g if isinstance(g, list) else [g] for g in style_legend]
    if groups:
        n_rows = max(len(g) for g in groups)
        lines.append(r"% Hidden axis collecting the style legend")
        lines.append(r"\begin{axis}[")
        lines.append(rf"  hide axis, overlay, at={{({group_name} c1r1.center)}}, anchor=center,")
        lines.append(r"  xmin=0, xmax=1, ymin=0, ymax=1,")
        lines.append(rf"  legend to name={slegend}, legend columns={len(groups)}, transpose legend,")
        lines.append(r"  legend cell align=left,")
        lines.append(r"  legend style={font=\footnotesize, draw=none, /tikz/every even column/.append style={column sep=8pt}},")
        lines.append(r"]")
        for group in groups:
            for entry in group + [None] * (n_rows - len(group)):
                image, text = (entry["image"], entry["text"]) if entry else ("empty legend", "")
                lines.append(rf"  \addlegendimage{{{image}}}")
                lines.append(rf"  \addlegendentry{{{text}}}")
        lines.append(r"\end{axis}")
        lines.append("")

    if legend_position == "side":
        # Beside the last panel (in the space a short, centred last row
        # leaves), d legend over style legend, framed together.
        last = max(i for i, c in enumerate(cells) if c is not None)
        cell_name = f"{group_name} c{last % cols + 1}r{last // cols + 1}"
        legends = rf"\pgfplotslegendfromname{{{dlegend}}}"
        if groups:
            legends += rf"\\[4pt]\pgfplotslegendfromname{{{slegend}}}"
        lines.append(rf"\node[draw, inner sep=3pt, align=center, anchor=west, xshift=6pt] "
                     rf"at ({cell_name}.east)")
        lines.append(rf"  {{{legends}}};")
        lines.append(r"\end{tikzpicture}")
        return "\n".join(lines)

    # One framed strip, centred on everything drawn so far (cells, titles,
    # tick and axis labels); the legends inside are vertically centred.
    legends = rf"$\vcenter{{\hbox{{\pgfplotslegendfromname{{{dlegend}}}}}}}$"
    if groups:
        legends += rf"\hspace{{12pt}}$\vcenter{{\hbox{{\pgfplotslegendfromname{{{slegend}}}}}}}$"
    if legend_position == "top":
        lines.append(r"\node[draw, inner sep=3pt, anchor=south, yshift=4pt] at (current bounding box.north)")
    else:
        lines.append(r"\node[draw, inner sep=3pt, anchor=north, yshift=-4pt] at (current bounding box.south)")
    lines.append(rf"  {{{legends}}};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines)



def _data_rows(series: str, by_d: dict[int, list[sinter.TaskStats]],
               threshold: list | None) -> list[list]:
    """The figure-data rows for one series (see _DATA_COLUMNS)."""
    rows = []
    if threshold:
        err = threshold[1] if len(threshold) > 1 else None
        rows.append([series, "threshold", None, None, threshold[0], err, err])
    for d in sorted(by_d):
        for s in sorted(by_d[d], key=_stat_p):
            r = _ler_with_bars(s, include_discards=False, mlf=_MLF)
            if r is None:
                continue
            best, low, high = r
            rows.append([series, "ler", d, _stat_p(s), best, best - low, high - best])
    return rows


def render(spec: dict, *, name: str, out_dir: Path, xlabel: str, width: str, height: str,
           ylabel: str = "Logical error rate", x_percent: bool = True) -> Path:
    """Write `<out_dir>/<name>.tex` and its `_data.csv` from a grid spec.

    spec: {"cols": N, "cells": [None | {"title": str, "series": [{"csv": [paths],
    "mark": str, "line": "solid" | "dashed", "label": str, "threshold": [p, err]}]}],
    "style_legend": [[{"image": ..., "text": ...}]], "legend_position": "bottom" | "top" |
    "side", "center_last_row": bool}. Cells fill row by row; None leaves one empty.
    """
    cells = []
    data_rows: list[list] = []
    for cell in spec["cells"]:
        if cell is None:
            cells.append(None)
            continue
        cell_series = []
        for entry in cell["series"]:
            stats = sinter.stats_from_csv_files(*[Path(csv) for csv in entry["csv"]])
            if not stats:
                raise ValueError(f"no TaskStats in {entry['csv']} (cell '{cell['title']}')")
            by_d = _group_by_distance(stats)
            cell_series.append((entry["mark"], by_d, entry.get("line", "solid"), entry.get("threshold")))
            data_rows += _data_rows(plain_label(entry.get("label", cell["title"])),
                                    by_d, entry.get("threshold"))
        cells.append((cell["title"], cell_series))
    body = _render_grid(
        cells, cols=spec["cols"], style_legend=spec.get("style_legend", []),
        legend_position=spec.get("legend_position", "bottom"),
        min_p=None, max_p=None, xlabel=xlabel, ylabel=ylabel, xmode="linear", ymode="log",
        width=width, height=height, mlf=_MLF, include_discards=False,
        legend_suffix=re.sub(r"[^a-zA-Z0-9]", "", name),
        x_percent=x_percent, center_last_row=spec.get("center_last_row", False),
    )
    out = out_dir / f"{name}.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + "\n")
    print(f"Wrote {out}")
    write_figure_data(out, _DATA_COLUMNS, data_rows)
    return out
