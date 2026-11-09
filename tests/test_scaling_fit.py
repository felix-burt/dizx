"""The crg threshold fit (experiments/figures/scaling_fit.py)."""
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


def test_recovers_the_crossing_with_the_fit_error():
    fit = sf.scaling_threshold(_stats([11, 13, 15, 17], 0.0), [11, 13, 15, 17])
    assert fit.valid and abs(fit.p_th - 0.30) < 1e-4
    assert 0 < fit.stderr < 1e-3


def test_a_poor_fit_stays_valid():
    # A crossing drifting with d makes chi2/dof large (exact LERs, 1e7 shots); no chi2 cut.
    fit = sf.scaling_threshold(_stats([11, 13, 15, 17], 0.3), [11, 13, 15, 17])
    assert fit.chi2 > 5 and fit.valid
