"""Threshold estimates with a scaling-variable window (the finite-size-scaling fit).

The finite-size-scaling (WHP) fit

    LER = A + B x + C x^2,   x = (p - p_th) d^(1/nu)

only holds near the threshold, and how near is set by x, not by p: a larger
distance bends away from the linear regime sooner. So the window is

    |p - p_th| d^(1/nu) <= x_max,

recentred on the fitted (p_th, nu) until it stops moving (starting from the
smallest-pair crossing and nu = 1.5). Scanning the window width on the crg
data found three regimes: too narrow (<= ~6 points)
gives degenerate fits (nu ~ 0.4, chi2/dof ~ 0, error bars of 1e-4 or 0.1),
a plateau where p_th is stable to ~0.001 with nu ~ 1.5-2 and chi2/dof ~ 1, and
too wide (chi2/dof 2-10, p_th drifting). x_max = 0.15 is inside the plateau
(about +-0.03 in p at d=11, +-0.025 at d=15).

A fit is valid only with
  * at least MIN_POINTS (3; target 4) points per distance below the largest,
    at least 2 at the largest, each distance with points on both sides of p_th;
  * 0.8 <= nu <= 2.5.
Its error is the fit's statistical error on p_th (from the binomial errors on
each LER), as is usual for QEC thresholds; there is no chi2 cut.

History (2026-10-28 to 11-04): the crossing drifts down with d (up to ~0.006
per two distances at crg_local p_c 0.5-0.9%, d = 11-19). We tried fitting the
WHP correction D d^(-1/mu) (p_th nearly undetermined at d <= 19), then
finite-size systematics (half the gap of the 11-17 and 13-19 fits; a
leave-one-distance-out jackknife), a window systematic over x_max and
sqrt(chi2/dof) error scaling. All were dropped as not standard in QEC
threshold papers; the drift is reported in the text instead (the shift when
d = 11 is left out).

Every threshold in the paper's figures comes from `scaling_threshold`: the
crg figures' points (`crg_points`) and the pure-erasure threshold
(`erasure_point`), which is both the crg curves' p_l = 0 point and the master
figure's threshold mark.
"""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sinter

from _regime_common import estimate_nonlocal_threshold, flat_csv_path_for, load_stats_by_local_prob

X_MAX = 0.15
NU_START = 1.5
# Valid fitted nu. The lower end was 1 until 2026-10-01: a few circuit-noise
# fits (5-/8-Chain ZZ at p_c 0.7-0.8%, BellZigZag prep XX 5%) settle at
# nu ~ 0.85-0.95 even in a full-width window, with thresholds that move by
# < 0.001 between windows -- an effective exponent at d <= 15, not a failure.
NU_RANGE = (0.8, 2.5)
# nu used to size the window is clipped to this (unchanged): sizing it with a
# fitted nu above 2.5 or below 1 was tried and loses well-behaved points.
NU_WINDOW = (1.0, 2.5)
# Points per distance (below the largest) a window must hold. 4 is the target;
# 3 is accepted (2026-10-01): the paper's sweeps stepped 0.01-0.015 in p, so
# the window held only 3 at d=11. The reproduction windows step 0.005.
MIN_POINTS = 3


@dataclass
class ScalingFit:
    p_th: float | None
    stderr: float | None   # the fit's statistical error on p_th
    nu: float | None
    chi2: float | None
    n: int
    valid: bool
    reason: str = ""


def _aggregate(stats: list[sinter.TaskStats], distances: list[int]) -> dict[tuple[int, float], tuple[int, int]]:
    agg: dict[tuple[int, float], tuple[int, int]] = {}
    seen: set[str] = set()
    for s in stats:
        if s.strong_id in seen:
            continue
        seen.add(s.strong_id)
        md = s.json_metadata or {}
        d, p = int(md["distance"]), float(md["p"])
        shots = s.shots - s.discards
        if d not in distances or shots <= 0:
            continue
        e, n = agg.get((d, p), (0, 0))
        agg[(d, p)] = (e + s.errors, n + shots)
    return agg


def _fit_window(agg, p_c: float, nu_w: float, x_max: float):
    """One WHP fit over the points with |p - p_c| d^(1/nu_w) <= x_max."""
    from scipy.optimize import curve_fit
    keys = [(d, p) for (d, p) in agg if abs(p - p_c) * d ** (1 / nu_w) <= x_max]
    if len(keys) < 6:
        return None, keys
    d = np.array([k[0] for k in keys], float)
    p = np.array([k[1] for k in keys], float)
    e = np.array([agg[k][0] for k in keys], float)
    n = np.array([agg[k][1] for k in keys], float)
    y = e / n
    sigma = np.sqrt(np.maximum(e * (1 - y), 1.0)) / n   # binomial, floored at one error

    def model(X, p_th, nu, a, b, c):
        x = (X[1] - p_th) * X[0] ** (1 / nu)
        return a + b * x + c * x * x

    span = p.max() - p.min()
    try:
        popt, pcov = curve_fit(model, (d, p), y, p0=[p_c, nu_w, float(y.mean()), 1.0, 0.0],
                               sigma=sigma, absolute_sigma=True,
                               bounds=([p.min() - span, 0.3, 0.0, -np.inf, -np.inf],
                                       [p.max() + span, 5.0, 1.0, np.inf, np.inf]),
                               maxfev=20000)
    except (RuntimeError, ValueError):
        return None, keys
    dof = len(y) - 5
    chi2 = float((((model((d, p), *popt) - y) / sigma) ** 2).sum() / dof) if dof > 0 else float("nan")
    stat = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else float("nan")
    return (float(popt[0]), float(popt[1]), stat, chi2), keys


def _iterate(agg, start: float, x_max: float, iterations: int = 8):
    """Recentre the window on the fitted (p_th, nu) until it stops moving."""
    p_c, nu_w = start, NU_START
    result, keys = None, []
    for _ in range(iterations):
        result, keys = _fit_window(agg, p_c, nu_w, x_max)
        if result is None:
            return None, keys
        p_new, nu_new = result[0], result[1]
        nu_new = min(max(nu_new, NU_WINDOW[0]), NU_WINDOW[1])   # window only; validity checks the fit's nu
        if abs(p_new - p_c) < 1e-5 and abs(nu_new - nu_w) < 1e-3:
            break
        p_c, nu_w = p_new, nu_new
    return result, keys


def _check(result, keys, distances: list[int]) -> str:
    if result is None:
        return f"fit failed ({len(keys)} points in window)"
    p_th, nu, _, _ = result
    dmax = max(distances)
    for d in distances:
        ps = sorted(p for dd, p in keys if dd == d)
        need = 2 if d == dmax else MIN_POINTS
        if len(ps) < need:
            return f"d={d}: {len(ps)} point(s) in window (need {need})"
        if not (ps[0] < p_th < ps[-1]):
            return f"d={d}: points don't straddle p_th"
    if not NU_RANGE[0] <= nu <= NU_RANGE[1]:
        return f"nu = {nu:.2f} outside {NU_RANGE}"
    return ""


def scaling_threshold(stats: list[sinter.TaskStats], distances: list[int]) -> ScalingFit:
    distances = sorted(distances)
    agg = _aggregate(stats, distances)
    present = sorted({d for d, _ in agg})
    if len(present) < 2:
        return ScalingFit(None, None, None, None, len(agg), False, "fewer than 2 distances")
    with contextlib.redirect_stdout(io.StringIO()):
        start = estimate_nonlocal_threshold(stats, present[:2])
    if start is None:
        return ScalingFit(None, None, None, None, len(agg), False, "no starting crossing")
    result, keys = _iterate(agg, start, X_MAX)
    reason = _check(result, keys, present)
    if result is None:
        return ScalingFit(None, None, None, None, len(keys), False, reason)
    p_th, nu, stat, chi2 = result
    return ScalingFit(p_th, stat, nu, chi2, len(keys), not reason, reason)


def crg_points(config: dict, *, results_dir: Path,
               distances: list[int] | None = None) -> dict[float, ScalingFit]:
    """{local_prob: fit} over every local_prob a crg config has data for, over
    `distances` (default: the config's own)."""
    distances = distances or config["sweep"]["distances"]
    return {l_prob: scaling_threshold(stats, distances) for l_prob, stats
            in sorted(load_stats_by_local_prob(config, results_dir=results_dir).items())}


def erasure_point(config: dict, *, results_dir: Path) -> ScalingFit | None:
    """The pure-erasure threshold (the local_prob = 0 point), same estimator,
    over the erasure sweep's own distances (d=11-19). Every figure takes the
    value from here, so they all quote one number."""
    path = flat_csv_path_for(config, results_dir=results_dir)
    if not path.exists():
        return None
    return scaling_threshold(list(sinter.read_stats_from_csv_files(path)), config["sweep"]["distances"])


def describe(f: ScalingFit) -> str:
    if f.p_th is None:
        return f"--  ({f.reason})"
    flag = "" if f.valid else f"   FLAGGED: {f.reason}"
    return (f"{f.p_th:.4f} ± {f.stderr:.4f}  "
            f"nu={f.nu:.2f} chi2/dof={f.chi2:.1f} n={f.n}{flag}")
