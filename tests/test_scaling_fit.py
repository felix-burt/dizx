"""The finite-size systematic of the crg threshold fit (experiments/figures/scaling_fit.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import sinter

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from figures import scaling_fit as sf  # noqa: E402

SHOTS = 10_000_000


def _stats(distances, drift):
    """Exact (noise-free) LERs of a quadratic scaling form whose crossing moves
    as 0.30 + drift / d."""
    out = []
    for d in distances:
        for k in range(13):
            p = round(0.27 + 0.005 * k, 6)
            x = (p - (0.30 + drift / d)) * d ** (1 / 1.5)
            errors = round((0.25 + 0.6 * x + 0.2 * x * x) * SHOTS)
            out.append(sinter.TaskStats(strong_id=f"{d}_{p}", decoder="d", shots=SHOTS, errors=errors,
                                        json_metadata={"distance": d, "p": p}))
    return out


def test_no_drift_no_finite_size_error():
    fit = sf.scaling_threshold(_stats([11, 13, 15, 17], 0.0), [11, 13, 15, 17])
    assert fit.valid and abs(fit.p_th - 0.30) < 1e-4
    assert fit.fs is not None and fit.fs < 1e-4


def test_drift_is_carried_as_an_error():
    stats = _stats([11, 13, 15, 17], 0.3)
    fit = sf.scaling_threshold(stats, [11, 13, 15, 17])
    lo = sf.scaling_threshold(stats, [11, 13, 15], finite_size=False)
    hi = sf.scaling_threshold(stats, [13, 15, 17], finite_size=False)
    assert abs(hi.p_th - lo.p_th) > 1e-3
    assert abs(fit.fs - abs(hi.p_th - lo.p_th) / 2) < 1e-12
    assert fit.stderr >= fit.fs > fit.stat


def test_three_distances_have_no_finite_size_error():
    assert sf.scaling_threshold(_stats([11, 13, 15], 0.3), [11, 13, 15]).fs is None
