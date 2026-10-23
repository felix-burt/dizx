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
  * 0.8 <= nu <= 2.5 and chi2/dof <= 5 (relaxed from 3, 2026-10-01: the quadratic
    form is only approximate over the window, and with ~1e6 shots per point
    the slow drift of the crossing with d shows up in chi2; such fits carry the
    sqrt(chi2/dof)-scaled error).
Its error is the statistical one (scaled by sqrt(chi2/dof) when that exceeds
1) in quadrature with a systematic: half the spread of p_th over
x_max in {0.10, 0.15, 0.20} (the valid ones).

Every threshold in the paper's figures comes from `scaling_threshold`: the
crg figures' points (`crg_points`) and the pure-erasure threshold
(`erasure_point`), which is both the crg curves' p_l = 0 point and the master
figure's threshold mark.
"""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sinter

from _regime_common import estimate_nonlocal_threshold, flat_csv_path_for, load_stats_by_local_prob

X_MAX = 0.15
X_MAX_SYSTEMATIC = (0.10, 0.15, 0.20)
NU_START = 1.5
# Valid fitted nu. The lower end was 1 until 2026-10-01: a few circuit-noise
# fits (5-/8-Chain ZZ at p_c 0.7-0.8%, BellZigZag prep XX 5%) settle at
# nu ~ 0.85-0.95 even in a full-width window, with thresholds that move by
# < 0.001 between windows -- an effective exponent at d <= 15, not a failure.
NU_RANGE = (0.8, 2.5)
# nu used to size the window is clipped to this (unchanged): sizing it with a
# fitted nu above 2.5 or below 1 was tried and loses well-behaved points.
NU_WINDOW = (1.0, 2.5)
MAX_CHI2 = 5.0
# Points per distance (below the largest) a window must hold. 4 is the target;
# 3 is accepted (2026-10-01): the paper's sweeps stepped 0.01-0.015 in p, so
# the window held only 3 at d=11. The reproduction windows step 0.005.
MIN_POINTS = 3


@dataclass
class ScalingFit:
    p_th: float | None
    stderr: float | None   # total: statistical (chi2-scaled) + systematic
    stat: float | None
    sys: float | None
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
    p_th, nu, _, chi2 = result
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
    if not chi2 <= MAX_CHI2:
        return f"chi2/dof = {chi2:.1f} > {MAX_CHI2}"
    return ""


def scaling_threshold(stats: list[sinter.TaskStats], distances: list[int]) -> ScalingFit:
    distances = sorted(distances)
    agg = _aggregate(stats, distances)
    present = sorted({d for d, _ in agg})
    if len(present) < 2:
        return ScalingFit(None, None, None, None, None, None, len(agg), False, "fewer than 2 distances")
    with contextlib.redirect_stdout(io.StringIO()):
        start = estimate_nonlocal_threshold(stats, present[:2])
    if start is None:
        return ScalingFit(None, None, None, None, None, None, len(agg), False, "no starting crossing")
    result, keys = _iterate(agg, start, X_MAX)
    reason = _check(result, keys, present)
    if result is None:
        return ScalingFit(None, None, None, None, None, None, len(keys), False, reason)
    p_th, nu, stat, chi2 = result
    stat_scaled = stat * max(1.0, math.sqrt(chi2)) if math.isfinite(chi2) else stat
    others = []
    for x_max in X_MAX_SYSTEMATIC:
        r, k = _iterate(agg, start, x_max)
        if r is not None and not _check(r, k, present):
            others.append(r[0])
    sys_err = (max(others) - min(others)) / 2 if len(others) >= 2 else 0.0
    total = math.hypot(stat_scaled, sys_err)
    return ScalingFit(p_th, total, stat_scaled, sys_err, nu, chi2, len(keys), not reason, reason)


def crg_points(config: dict, *, results_dir: Path) -> dict[float, ScalingFit]:
    """{local_prob: fit} over every local_prob a crg config has data for."""
    distances = config["sweep"]["distances"]
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
    return (f"{f.p_th:.4f} ± {f.stderr:.4f} (stat {f.stat:.4f}, sys {f.sys:.4f})  "
            f"nu={f.nu:.2f} chi2/dof={f.chi2:.1f} n={f.n}{flag}")
