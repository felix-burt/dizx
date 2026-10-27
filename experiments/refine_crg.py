#!/usr/bin/env python
"""Refine the crg thresholds in rounds until their scaling fits converge.

    poetry run python experiments/refine_crg.py status --results-dir results/cluster
    poetry run python experiments/refine_crg.py next --dry-run --results-dir results/cluster
    poetry run python experiments/refine_crg.py next --results-dir results/cluster
    poetry run python experiments/cluster.py submit 'crg_*' --results-dir results/cluster

Each crg point (curve, p_l) is fitted with `scaling_fit.scaling_threshold`,
the estimator the figures plot. A point is converged when its fit is valid,
its error is <= TARGET_STDERR and, after a refinement round, p_th moved by at
most half that error since the fit that planned the round. Otherwise `next`
plans one more round for it, inside its scaling window

    |p - p_th| d^(1/nu) <= WINDOW_X     (the fit's own window, x_max),

per distance: every point of the CRG_WINDOW_STEP grid in it (re-centring the
window, or widening it where nu is large and the paper's fixed window cut it
short), on the FINE_STEP grid where that leaves fewer than MIN_PER_DISTANCE,
all collected to a budget (stderr / target)^2 times the largest one already
there, clamped to [MIN_FACTOR, MAX_FACTOR] per round. Points stop after
MAX_ROUNDS rounds whatever their state.

`next` appends the round to paper.REFINEMENT_FILE; from then on the round is
part of the fixed plan (`paper.Curve.runs`), so `reproduce.py` and
`cluster.py` collect it like any other run. The pure-erasure p_l = 0 points
are not refined.

`next --add-distance D` plans a different kind of round: distance D over its
own scaling window, at the base budget, for every point with a valid fit, so
each curve keeps one distance set. The fits then use every distance any round
added (`paper.crg_fit_distances`); 2026-10-27 that was d = 17, chosen over a
2x round as more stat-error reduction per core-hour (Fisher estimate: 33% for
~1.5k core-h vs 29% for ~2.2k) and a check on the crossing's drift with d.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_EXPERIMENTS.parent / "scripts"))

import paper  # noqa: E402
from figures import scaling_fit  # noqa: E402
from figures.scaling_fit import ScalingFit  # noqa: E402

TARGET_STDERR = 0.0005
MAX_ROUNDS = 4
# The fit window, not the widest systematic one (0.20): that one costs ~45%
# more per round, mostly new points outside the paper's +-0.03 data when nu ~ 2.
WINDOW_X = scaling_fit.X_MAX
FINE_STEP = paper.CRG_WINDOW_STEP / 2
MIN_PER_DISTANCE = 4
# MAX_FACTOR was 4 (2026-10-27): round 1 then priced at ~10.9k core-hours, twice
# the whole original crg plan, so rounds go 2x at a time instead.
MIN_FACTOR, MAX_FACTOR = 2.0, 2.0


def crg_curves() -> list[paper.Curve]:
    return [c for c in paper.CURVES.values() if c.kind in paper.CRG_LOCAL_PROBS]


def current_fits(results_dir: Path) -> dict[str, ScalingFit]:
    """{point key: fit} for every crg point with data."""
    out = {}
    for curve in crg_curves():
        config = paper.crg_config(curve.kind, curve.family, curve.observable)
        for lp, fit in scaling_fit.crg_points(config, results_dir=results_dir / "sinter",
                                              distances=paper.crg_fit_distances()).items():
            out[paper.point_key(curve.id, lp)] = fit
    return out


def history(rounds: list[dict], key: str) -> list[dict]:
    return [r["points"][key] for r in rounds if key in r["points"]]


def collected_scale(rounds: list[dict], key: str, d: int, p: float) -> float:
    """The budget scale (d, p) has been planned to so far (1 = the base plan)."""
    return max([pt["scale"] for pt in history(rounds, key) if p in pt["probs"].get(str(d), [])],
               default=1.0)


def verdict(fit: ScalingFit, past: list[dict]) -> str:
    """"converged", "refine" or why the point is left alone."""
    if fit.p_th is None or not fit.valid:
        return f"no valid fit ({fit.reason})"
    moved = abs(fit.p_th - past[-1]["fit"]["p_th"]) if past else 0.0
    if fit.stderr <= TARGET_STDERR and moved <= fit.stderr / 2:
        return "converged"
    if len(past) >= MAX_ROUNDS:
        return f"round cap ({MAX_ROUNDS})"
    return "refine"


def window_probs(p_th: float, nu: float, d: int) -> list[float]:
    """The grid points of one distance's scaling window (p > 0)."""
    nu_w = min(max(nu, scaling_fit.NU_WINDOW[0]), scaling_fit.NU_WINDOW[1])
    half = WINDOW_X / d ** (1 / nu_w)
    for step in (paper.CRG_WINDOW_STEP, FINE_STEP):
        lo, hi = math.ceil((p_th - half) / step - 1e-9), math.floor((p_th + half) / step + 1e-9)
        probs = [round(k * step, 6) for k in range(max(lo, 1), hi + 1)]
        if len(probs) >= MIN_PER_DISTANCE:
            break
    return probs


def plan_point(fit: ScalingFit, rounds: list[dict], key: str) -> dict:
    """One round for one point: its window per distance and their budget scale."""
    probs = {str(d): window_probs(fit.p_th, fit.nu, d) for d in paper.crg_fit_distances()}
    held = max(collected_scale(rounds, key, int(d), p) for d, ps in probs.items() for p in ps)
    factor = min(max((fit.stderr / TARGET_STDERR) ** 2, MIN_FACTOR), MAX_FACTOR)
    fit_record = {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(fit).items()}
    return {"fit": fit_record, "scale": round(held * factor, 3), "probs": probs}


def add_distance(fits: dict[str, ScalingFit], distance: int) -> dict:
    """A round adding `distance` to every point with a valid fit, at the base
    budget, over that distance's scaling window."""
    return {key: {"fit": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in asdict(fit).items()},
                  "scale": 1.0, "probs": {str(distance): window_probs(fit.p_th, fit.nu, distance)}}
            for key, fit in fits.items() if fit.p_th is not None and fit.valid}


def propose(fits: dict[str, ScalingFit], rounds: list[dict]) -> tuple[dict, dict[str, str]]:
    """(the next round's points, verdict per point)."""
    points, verdicts = {}, {}
    for key, fit in fits.items():
        verdicts[key] = verdict(fit, history(rounds, key))
        if verdicts[key] == "refine":
            points[key] = plan_point(fit, rounds, key)
    return points, verdicts


def core_hours(rounds: list[dict], results_dir: Path) -> float:
    """Expected core-hours still to collect for the crg refinement under `rounds`."""
    import cluster
    with paper.refinement_rounds(rounds):
        jobs = cluster.plan_jobs(["crg_*"], split="curve", budget_scale=1.0,
                                 results_dir=str(results_dir))
    costs, progress = cluster.load_costs(), cluster.Progress(results_dir)
    return sum(cluster.job_cost(job, costs, progress).core_hours
               for job in jobs if not job.name.startswith("erasure"))


def _row(key: str, fit: ScalingFit, past: list[dict], verdict_: str) -> str:
    if fit.p_th is None:
        return f"{key:34s} {len(past):2d}  {'--':>7s} {'':8s} {'':8s}  {verdict_}"
    moved = f"{fit.p_th - past[-1]['fit']['p_th']:+.4f}" if past else ""
    return f"{key:34s} {len(past):2d}  {fit.p_th:7.4f} {fit.stderr:8.4f} {moved:>8s}  {verdict_}"


def cmd_status(args) -> None:
    rounds = paper.REFINEMENT_ROUNDS
    fits = current_fits(args.results_dir)
    _, verdicts = propose(fits, rounds)
    print(f"{'point':34s} {'rd':>2s}  {'p_th':>7s} {'stderr':>8s} {'moved':>8s}  verdict")
    for key, fit in fits.items():
        print(_row(key, fit, history(rounds, key), verdicts[key]))
    counts = {}
    for v in verdicts.values():
        counts[v.split(" (")[0]] = counts.get(v.split(" (")[0], 0) + 1
    print(f"\n{len(rounds)} round(s) frozen; target stderr {TARGET_STDERR}; "
          + ", ".join(f"{n} {v}" for v, n in sorted(counts.items())))


def cmd_next(args) -> None:
    rounds = paper.REFINEMENT_ROUNDS
    fits = current_fits(args.results_dir)
    points, verdicts = propose(fits, rounds)
    if args.add_distance:
        points = add_distance(fits, args.add_distance)
    if not points:
        print("Nothing to refine: every point is converged, capped or has no valid fit.")
        return
    proposal = {"date": datetime.date.today().isoformat(), "target_stderr": TARGET_STDERR,
                "points": points}
    print(f"{'point':34s} {'p_th':>7s} {'stderr':>8s} {'scale':>6s}  points per distance")
    for key, pt in points.items():
        per_d = " ".join(f"d{d}:{len(ps)}" for d, ps in pt["probs"].items())
        print(f"{key:34s} {pt['fit']['p_th']:7.4f} {pt['fit']['stderr']:8.4f} {pt['scale']:6.2f}  {per_d}")
    before = core_hours(rounds, args.results_dir)
    after = core_hours(rounds + [proposal], args.results_dir)
    print(f"\nRound {len(rounds) + 1}: {len(points)} of {len(verdicts)} point(s) refined, "
          f"~{after - before:.0f} core-hours expected "
          f"(already outstanding before it: ~{before:.0f}).")
    if args.dry_run:
        return
    paper.REFINEMENT_FILE.write_text(json.dumps({"rounds": rounds + [proposal]}, indent=1) + "\n")
    print(f"Frozen into {paper.REFINEMENT_FILE}; submit with cluster.py submit 'crg_*'.")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("status", cmd_status), ("next", cmd_next)):
        sp = sub.add_parser(name)
        sp.add_argument("--results-dir", type=Path, default=Path(paper.RESULTS_DIR),
                        help="Root of the sinter CSVs (default: %(default)s/).")
        sp.set_defaults(fn=fn)
    sub.choices["next"].add_argument("--dry-run", action="store_true",
                                     help="Show the round and its cost without freezing it.")
    sub.choices["next"].add_argument("--add-distance", type=int,
                                     help="Instead of raising budgets, add this distance at the base "
                                          "budget to every point with a valid fit (converged or not).")
    args = p.parse_args(argv)
    args.results_dir = args.results_dir.resolve()
    args.fn(args)


if __name__ == "__main__":
    main()
