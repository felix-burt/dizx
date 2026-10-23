#!/usr/bin/env python
"""Reproduce the paper's figures: collect each curve's data, then draw.

    poetry run python experiments/reproduce.py list
    poetry run python experiments/reproduce.py crg_prep_ZZ                  # one figure
    poetry run python experiments/reproduce.py crg_prep/bell/ZZ             # one curve
    poetry run python experiments/reproduce.py 'lambda/linear_k1/*'         # glob over curve ids
    poetry run python experiments/reproduce.py all --workers 8
    poetry run python experiments/reproduce.py crg_local_XX --plot-only

Targets are figure keys, curve ids or shell-style globs over either (`list`
shows them all). Every targeted curve is collected, then every figure holding
a targeted curve is redrawn from whatever data exists, into
`<out-dir>/<paper file name>.tex` plus its `_data.csv` (and a standalone PDF
under `<out-dir>/pdf/` with --pdf).

Collection resumes: sinter CSVs are keyed by strong_id (circuit + noise +
decoder), so an interrupted or repeated run, or a curve two figures share,
never collects a shot twice. `--budget-scale 0.01` gives a quick low-statistics
preview; a later full run tops the same points up. The plan itself (which
points, which budgets) lives in paper.py.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import math
import os
import sys
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
_REPO = _EXPERIMENTS.parent
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_REPO / "scripts"))

import paper  # noqa: E402
from paper import FAMILIES, FIGURES, CURVES, OBSERVABLES, Curve, Figure  # noqa: E402

# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def _scaled(config: dict, *, workers: int, budget_scale: float, results_dir: str) -> dict:
    config = copy.deepcopy(config)
    config["results_dir"] = results_dir
    si = config["sinter"]
    si["num_workers"] = workers
    si["max_shots"] = max(1, math.ceil(si["max_shots"] * budget_scale))
    si["max_errors"] = max(1, math.ceil(si["max_errors"] * budget_scale))
    return config


def _tasks(run: dict) -> int:
    return len(run["sweep"]["distances"]) * len(run["sweep"]["probs"] if "probs" in run["sweep"] else
                                               _sweep_probs(run["sweep"]))


def _sweep_probs(sweep: dict) -> list[float]:
    from run_threshold_experiment import sweep_probs
    return sweep_probs(sweep)


def collect(curves: list[Curve], *, workers: int, budget_scale: float, results_dir: str,
            dry_run: bool = False) -> None:
    from run_threshold_experiment import run_threshold
    seen: set[str] = set()
    for curve in curves:
        print(f"\n===== {curve.id} =====", flush=True)
        for run in curve.runs():
            key = repr(sorted(run.items(), key=str))
            if key in seen:   # e.g. the erasure sweep under several crg curves
                continue
            seen.add(key)
            run = _scaled(run, workers=workers, budget_scale=budget_scale, results_dir=results_dir)
            noise = {ch["probability"] for ch in run["noise_model"]
                     if isinstance(ch.get("probability"), float)}
            print(f"  {run['name']} local={sorted(noise) or '-'} d={run['sweep']['distances']} "
                  f"{_tasks(run)} task(s), <= {run['sinter']['max_shots']:.0e} shots / "
                  f"{run['sinter']['max_errors']} errors each", flush=True)
            if not dry_run:
                run_threshold(run)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

_PC_XLABEL = r"Circuit noise probability $p_c$ (\%)"
_PR_XLABEL = r"Resource state error $p_{r}$ (\%)"
_PE_XLABEL = r"Erasure rate $p_e$ (\%)"
_ERASURE_THRESHOLD_YLABEL = r"Erasure threshold $p^{*}_{e}$"


def _master(fig: Figure, out_dir: Path, results_dir: Path) -> Path:
    from _regime_common import flat_csv_path_for
    from figures import master_threshold
    from figures.scaling_fit import erasure_point
    grid = [["bell", "bell_zigzag", "linear_kinf"], ["linear_k1", "linear_k2", None]]
    line = {"ZZ": "solid", "XX": "dashed"}
    wanted = {(c.family, c.observable) for c in fig.curves}
    cells = []
    for family in (f for row in grid for f in row):
        series = []
        for obs in OBSERVABLES:
            if family is None or (family, obs) not in wanted:
                continue
            config = paper.erasure_config(family, obs)
            csv = flat_csv_path_for(config, results_dir=results_dir)
            label = f"{FAMILIES[family].label} ({obs})"
            if not csv.exists():
                print(f"  no data: {label}")
                continue
            entry = {"csv": [str(csv)], "mark": "*, mark size=1pt", "line": line[obs], "label": label}
            fit = erasure_point(config, results_dir=results_dir)
            if fit is not None and fit.valid:
                entry["threshold"] = [fit.p_th, fit.stderr]
            else:
                print(f"  no valid threshold: {label}")
            series.append(entry)
        cells.append({"title": FAMILIES[family].label, "series": series} if series else None)
    style_legend = [[{"image": f"black, {line[obs]}, line legend, mark=none",
                      "text": rf"$\bar{{{obs[0]}}}\bar{{{obs[1]}}}$"} for obs in OBSERVABLES]]
    spec = {"cols": 3, "cells": cells, "style_legend": style_legend, "center_last_row": True,
            "legend_position": "side"}
    return master_threshold.render(spec, name=fig.stem, out_dir=out_dir, xlabel=_PE_XLABEL,
                                   width=rf"{0.87 / 3:.3f}\linewidth", height=r"0.2\textheight")


def _crg(fig: Figure, out_dir: Path, results_dir: Path) -> Path:
    from figures import correctable_regime
    kind = fig.curves[0].kind
    curves = [{"label": FAMILIES[c.family].label, "color": FAMILIES[c.family].color,
               "mark": FAMILIES[c.family].mark,
               "config": paper.crg_config(kind, c.family, c.observable),
               "erasure": paper.erasure_config(c.family, c.observable)} for c in fig.curves]
    if kind == "crg_local":
        # Bottom left: at the paper's figure size the top-right legend covers
        # the Bell/3d-Chain curves.
        opts = {"xlabel": _PC_XLABEL, "legend_pos": "south west"}
    else:
        opts = {"xlabel": _PR_XLABEL, "extend_to_axis": True}
    return correctable_regime.render(curves, name=fig.stem, out_dir=out_dir, results_dir=results_dir,
                                     ylabel=_ERASURE_THRESHOLD_YLABEL, **opts)


# Stitched erasure-axis curves: line style = the lowest boost level that
# reaches that erasure rate (n=1 from 0.25 up, n=2 from 0.125, n=3 from
# 0.0625). n=1 has no runs of its own: its range is covered by the n=2 data,
# and an equal-p_e point is the same experiment whichever n produced it.
_REGION_LINES = {1: ("solid", r"$n=1$"), 2: ("dashed", r"$n=2$"), 3: ("densely dotted", r"$n=3$")}
# Leading order as a translucent band, clear of the region line styles.
_ASYMPTOTE_STYLE = "line width=2.2pt, opacity=0.3"


def _lambda(fig: Figure, out_dir: Path, results_dir: Path) -> Path:
    from _regime_common import csv_path_for
    from figures import lambda_grid
    floors = {n: paper.boosted_erasure_floor(n) for n in _REGION_LINES}
    regions = [{"x_range": [floors[n], floors.get(n - 1, math.inf)], "line": line}
               for n, (line, _) in _REGION_LINES.items()]
    wanted = {(c.family, c.observable, c.local_prob) for c in fig.curves}
    cells = []
    for row, lp in enumerate(paper.LAMBDA_LOCAL_PROBS):
        for col, obs in enumerate(OBSERVABLES):
            series = []
            for family, fam in FAMILIES.items():
                if (family, obs, lp) not in wanted:
                    continue
                sources = []
                for n in sorted(paper.LAMBDA_POINTS[(family, obs, lp)]):
                    config = paper.lambda_config(family, obs, n)
                    if not csv_path_for(config, lp, results_dir=results_dir).exists():
                        print(f"  no data: {family} {obs} n={n} p_l={lp:g}")
                        continue
                    sources.append({"config": config, "local_prob": lp, "line": "solid",
                                    "label": f"{fam.label} ({obs}, n={n}, p_l={lp:g})",
                                    # Each boost level takes the erasure rates from its
                                    # own floor up to the next lower level's.
                                    "x_range": [floors[n], floors[n - 1] if n > 2 else math.inf]})
                if sources:
                    series.append({"color": fam.color, "mark": fam.mark, "line": "solid",
                                   "sources": sources, "line_regions": regions,
                                   "label": f"{fam.label} ({obs})"})
            cell = {"title": rf"$\bar{{{obs[0]}}}\bar{{{obs[1]}}}$" if row == 0 else "",
                    "ylabel": (rf"$\Lambda$ ($p_c=p_r={paper.p_l_tex(lp) if lp else 0}$)"
                               if col == 0 else ""),
                    "series": series}
            if not lp:
                # Leading-order loss-only Lambda per family; any of its configs
                # gives the circuit (the boost level only sets p_e).
                cell["asymptotes"] = [{"config": paper.lambda_config(f, obs, 2), "observable": obs,
                                       "color": FAMILIES[f].color}
                                      for f in FAMILIES if (f, obs, lp) in wanted]
            cells.append(cell)
    spec = {
        "cols": len(OBSERVABLES),
        "cells": cells,
        "width": rf"{0.9 / len(OBSERVABLES):.3f}\linewidth",
        "height": r"0.2\textheight",
        "x": "erasure",
        "xlabel": r"Fusion erasure rate $p_e$ (\%)",
        "legend": [{"label": f.label, "color": f.color, "mark": f.mark} for f in FAMILIES.values()],
        "style_legend": ([{"line": line, "label": text} for line, text in _REGION_LINES.values()]
                         + [{"line": _ASYMPTOTE_STYLE, "label": "leading order"}]),
        "legend_columns": len(FAMILIES),
        # One step for every point: d=5->7, the closest to the asymptotic Lambda.
        "fit_distances": paper.LAMBDA_DISTANCES,
        "draw_bounds": False,
        "asymptote_style": _ASYMPTOTE_STYLE,
        "asymptote_distances": paper.LAMBDA_DISTANCES,
        # Down to where the 5-chain's leading-order Lambda overtakes Bell's (~0.075).
        "x_floor": 0.04,
        # Leading order stands in for the data below where it runs out: each
        # family's band stops two points into its data, and at 10% erasure
        # for a family with no points.
        "asymptote_to_data": True,
        "asymptote_overlap": 2,
        "asymptote_x_max": 0.1,
    }
    return lambda_grid.render(spec, name=fig.stem, out_dir=out_dir, results_dir=results_dir)


_BUILDERS = {"master_threshold": _master, "crg_local_ZZ": _crg, "crg_local_XX": _crg,
             "crg_prep_ZZ": _crg, "crg_prep_XX": _crg, "lambda_grid": _lambda}


def plot(fig: Figure, *, out_dir: Path, results_dir: Path, pdf: bool) -> Path | None:
    """Draw `fig` from the CSVs under `<results_dir>/sinter`."""
    print(f"\n===== figure {fig.key} -> {fig.stem} =====", flush=True)
    try:
        snippet = _BUILDERS[fig.key](fig, out_dir, results_dir / "sinter")
    except ValueError as ex:   # nothing collected yet
        print(f"  skipped: {ex}")
        return None
    if pdf:
        from figures.common import compile_standalone
        out = compile_standalone(snippet, out_dir / "pdf")
        if out is not None:
            print(f"  pdf: {out}")
    return snippet


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resolve_targets(targets: list[str]) -> tuple[list[Curve], list[Figure]]:
    """(curves to collect, figures to draw) for figure keys / curve ids / globs."""
    curves: dict[str, Curve] = {}
    if "all" in targets:
        targets = list(FIGURES)
    for target in targets:
        figs = [f for k, f in FIGURES.items() if fnmatch.fnmatchcase(k, target)]
        ids = [i for i in CURVES if fnmatch.fnmatchcase(i, target)]
        if not figs and not ids:
            raise SystemExit(f"unknown target {target!r} (see `reproduce.py list`)")
        for fig in figs:
            curves.update((c.id, c) for c in fig.curves)
        curves.update((i, CURVES[i]) for i in ids)
    figures = [f for f in FIGURES.values() if any(c.id in curves for c in f.curves)]
    return list(curves.values()), figures


def _list() -> None:
    for fig in FIGURES.values():
        print(f"{fig.key}  ->  {fig.stem}.tex")
        for curve in fig.curves:
            runs = curve.runs()
            print(f"    {curve.id:32s} {len(runs)} run(s), {sum(_tasks(r) for r in runs)} task(s)")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("targets", nargs="+", help="`list`, `all`, figure keys, curve ids or globs.")
    p.add_argument("--workers", type=int, default=2,
                   help="sinter worker processes (default: %(default)s, light enough to keep "
                        "using the machine; raise it on a dedicated one).")
    p.add_argument("--budget-scale", type=float, default=1.0,
                   help="Multiply every point's max_shots / max_errors (e.g. 0.01 for a preview).")
    p.add_argument("--results-dir", default=paper.RESULTS_DIR,
                   help="Root of the sinter CSVs (default: %(default)s/).")
    p.add_argument("--out-dir", type=Path, default=_REPO / "figures" / "paper",
                   help="Where the .tex snippets and _data.csv files go (default: %(default)s).")
    p.add_argument("--collect-only", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print the runs without collecting.")
    p.add_argument("--pdf", action="store_true", help="Also compile standalone PDFs (latexmk).")
    p.add_argument("--no-nice", action="store_true", help="Don't lower this process's priority.")
    args = p.parse_args(argv)

    if args.targets == ["list"]:
        _list()
        return
    curves, figures = resolve_targets(args.targets)
    results_dir, out_dir = Path(args.results_dir).resolve(), args.out_dir.resolve()
    # The runner's circuit cache and result paths are relative to the repo root.
    os.chdir(_REPO)
    if not args.no_nice and hasattr(os, "nice"):
        os.nice(10)   # sinter's workers inherit it
    if not args.plot_only:
        collect(curves, workers=args.workers, budget_scale=args.budget_scale,
                results_dir=str(results_dir), dry_run=args.dry_run)
    if not args.collect_only and not args.dry_run:
        for fig in figures:
            plot(fig, out_dir=out_dir, results_dir=results_dir, pdf=args.pdf)


if __name__ == "__main__":
    main()
