"""The paper catalogue (experiments/paper.py) and its front end (reproduce.py)."""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pytest
import sinter

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import _regime_common as rc  # noqa: E402
import paper  # noqa: E402
import reproduce  # noqa: E402
import run_threshold_experiment as rte  # noqa: E402
from dizx.builders import BUILDERS  # noqa: E402


@pytest.fixture(autouse=True)
def _base_plan():
    """These test the paper's base plan, not the frozen crg refinement rounds."""
    with paper.refinement_rounds([]):
        yield

# The figure files the paper \input's (DLEFBQC sections/7_results.tex).
PAPER_STEMS = {
    "master_threshold_pm_corr_mergeZZ_obsZZ-XX",
    "circuit_and_erasure_correctable_regime_pm_corr_mergeZZ_obsZZ",
    "circuit_and_erasure_correctable_regime_pm_corr_mergeZZ_obsXX",
    "prep_and_erasure_correctable_regime_pm_corr_mergeZZ_obsZZ",
    "prep_and_erasure_correctable_regime_pm_corr_mergeZZ_obsXX",
    "subthreshold_lambda_erasure_grid_pm_corr_mergeZZ_obsZZ-XX",
}
ALL_RUNS = [(curve, run) for curve in paper.CURVES.values() for run in curve.runs()]


def test_figures_are_the_papers():
    assert {f.stem for f in paper.FIGURES.values()} == PAPER_STEMS
    sizes = {k: len(f.curves) for k, f in paper.FIGURES.items()}
    assert sizes == {"master_threshold": 10, "crg_local_ZZ": 5, "crg_local_XX": 5,
                     "crg_prep_ZZ": 5, "crg_prep_XX": 5, "lambda_grid": 30}
    assert len(paper.CURVES) == 60


def test_families_name_builders():
    for family in paper.FAMILIES.values():
        assert family.builder in BUILDERS, family.key


def test_crg_window():
    assert paper.crg_window(0.284) == [round(0.255 + 0.005 * k, 6) for k in range(13)]
    # Never below zero: a threshold near zero keeps the points above it.
    assert paper.crg_window(0.006) == [round(0.005 * k, 6) for k in range(8)]
    assert paper.crg_window(None) == paper.CRG_CLOSING_PROBE


def test_crg_windows_bracket_every_paper_threshold():
    for (kind, family, obs), thresholds in paper.CRG_THRESHOLDS.items():
        config = paper.crg_config(kind, family, obs)
        windows = config["sweep"]["local_prob_sweeps"]
        assert set(thresholds) <= set(config["sweep"]["local_probs"])
        for lp, p_th in thresholds.items():
            probs = windows[str(lp)]["probs"]
            assert min(probs) <= p_th <= max(probs), (kind, family, obs, lp)


def test_lambda_points_fall_in_their_stitch_range():
    """Each boost level supplies the erasure rates from its own floor up to the
    next lower level's floor, so every collected point is plotted."""
    floor2, floor3 = paper.boosted_erasure_floor(2), paper.boosted_erasure_floor(3)
    for key, by_n in paper.LAMBDA_POINTS.items():
        for loss in by_n.get(2, []):
            assert rte.boosted_erasure_probability(loss, 2) >= floor2, key
        for loss in by_n.get(3, []):
            assert floor3 <= rte.boosted_erasure_probability(loss, 3) < floor2, key


def _read_path(curve: paper.Curve, run: dict, sinter_dir: Path) -> Path:
    """The CSV the figure code reads for this run."""
    obs = curve.observable
    if run["name"].startswith("erasure_"):
        return rc.flat_csv_path_for(paper.erasure_config(curve.family, obs), results_dir=sinter_dir)
    local_prob = rte.extract_local_prob(run["noise_model"])
    if curve.kind == "lambda":
        n_boost = int(run["name"][-1])
        return rc.csv_path_for(paper.lambda_config(curve.family, obs, n_boost), local_prob,
                               results_dir=sinter_dir)
    return rc.csv_path_for(paper.crg_config(curve.kind, curve.family, obs), local_prob,
                           results_dir=sinter_dir)


def test_runs_write_where_figures_read(tmp_path):
    for curve, run in ALL_RUNS:
        written = rte.sinter_csv_path(run, run["noise_model"], paper.MERGE, curve.observable,
                                      out_dir=str(tmp_path))
        assert Path(written) == _read_path(curve, run, tmp_path / "sinter"), (curve.id, run["name"])


# What may differ between two runs writing one CSV: which points, how hard.
_POINT_KEYS = {"sweep.distances", "sweep.probs", "sinter.max_errors", "sinter.max_shots",
               "sinter.num_workers"}


def _experiment(config: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_experiment(v, key + "."))
        elif key not in _POINT_KEYS:
            out[key] = v
    return out


def test_runs_sharing_a_csv_are_one_experiment(tmp_path):
    by_csv = defaultdict(list)
    for curve, run in ALL_RUNS:
        by_csv[rte.sinter_csv_path(run, run["noise_model"], paper.MERGE, curve.observable,
                                   out_dir=str(tmp_path))].append(run)
    for path, runs in by_csv.items():
        assert all(_experiment(r) == _experiment(runs[0]) for r in runs), path
    # One CSV per erasure sweep, per crg (curve, p_l) and per Lambda (curve, n).
    n_crg = sum(len(paper.CRG_LOCAL_PROBS[k][f]) for k in paper.CRG_LOCAL_PROBS
                for f in paper.FAMILIES) * len(paper.OBSERVABLES)
    n_lambda = sum(len(by_n) for by_n in paper.LAMBDA_POINTS.values())
    assert len(by_csv) == 10 + n_crg + n_lambda


def test_resolve_targets():
    curves, figs = reproduce.resolve_targets(["crg_prep_ZZ"])
    assert [c.id for c in curves] == [f"crg_prep/{f}/ZZ" for f in paper.FAMILIES]
    assert [f.key for f in figs] == ["crg_prep_ZZ"]
    curves, figs = reproduce.resolve_targets(["lambda/linear_k1/*"])
    assert len(curves) == 6 and [f.key for f in figs] == ["lambda_grid"]
    curves, figs = reproduce.resolve_targets(["all"])
    assert len(curves) == 60 and len(figs) == 6
    with pytest.raises(SystemExit):
        reproduce.resolve_targets(["no_such_figure"])


def test_dry_run_lists_runs(capsys):
    curves, _ = reproduce.resolve_targets(["crg_prep/bell/ZZ", "erasure/bell/ZZ"])
    reproduce.collect(curves, workers=1, budget_scale=0.5, results_dir="unused", dry_run=True)
    out = capsys.readouterr().out
    # The crg curve's p_l = 0 point is the erasure sweep: collected once.
    assert out.count("erasure_bell_corr_obsZZ") == 1
    assert out.count("crg_prep_bell_corr_obsZZ") == len(paper.CRG_LOCAL_PROBS["crg_prep"]["bell"])
    assert "5e+05 shots / 50000 errors" in out


def _write_stats(path: Path, rows: list[tuple[int, float, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [sinter.CSV_HEADER]
    for d, p, shots, errors in rows:
        lines.append(sinter.TaskStats(strong_id=f"{d}_{p}", decoder="pymatching_heralded",
                                      json_metadata={"distance": d, "p": p},
                                      shots=shots, errors=errors).to_csv_line())
    path.write_text("\n".join(lines) + "\n")


def test_lambda_grid_renders(tmp_path):
    from figures import lambda_grid
    config = paper.lambda_config("bell", "ZZ", 2)
    sinter_dir = tmp_path / "sinter"
    _write_stats(rc.csv_path_for(config, 0.001, results_dir=sinter_dir),
                 [(5, 0.01, 10**6, 1000), (7, 0.01, 10**6, 200)])
    spec = {"cols": 1, "width": "5cm", "height": "4cm", "x": "erasure", "xlabel": "x",
            "cells": [{"title": "", "ylabel": "", "series": [
                {"color": "blue", "mark": "*", "line": "solid", "label": "Bell (ZZ)",
                 "sources": [{"config": config, "local_prob": 0.001, "x_range": [0.125, 1.0]}]}]}],
            "legend": [{"label": "Bell", "color": "blue", "mark": "*"}],
            "fit_distances": [5, 7]}
    lambda_grid.render(spec, name="l", out_dir=tmp_path, results_dir=sinter_dir)
    rows = (tmp_path / "l_data.csv").read_text().splitlines()
    assert len(rows) == 2 and float(rows[1].split(",")[3]) == pytest.approx(5.0)
