"""Draw the crg figures with the finite-size-corrected ansatz A+Bx+Cx^2+D/d (scratch only)."""
import sys, math, warnings
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit
sys.path[:0] = ["experiments", "scripts"]
import reproduce, paper
from figures import scaling_fit as sf
warnings.filterwarnings("ignore")
_orig = sf.scaling_threshold

def corrected(stats, distances):
    base = _orig(stats, distances)          # window centre and nu from the usual fit
    if base.p_th is None:
        return base
    agg = sf._aggregate(stats, sorted(distances))
    nu_w = min(max(base.nu, sf.NU_WINDOW[0]), sf.NU_WINDOW[1])
    keys = [k for k in agg if abs(k[1] - base.p_th) * k[0] ** (1 / nu_w) <= sf.X_MAX]
    d = np.array([k[0] for k in keys], float); p = np.array([k[1] for k in keys], float)
    e = np.array([agg[k][0] for k in keys], float); n = np.array([agg[k][1] for k in keys], float)
    y = e / n; s = np.sqrt(np.maximum(e * (1 - y), 1)) / n
    def f(X, th, nu, a, b, c, D):
        x = (X[1] - th) * X[0] ** (1 / nu); return a + b * x + c * x * x + D / X[0]
    try:
        P, C = curve_fit(f, (d, p), y, p0=[base.p_th, base.nu, y.mean(), 1, 0, 0], sigma=s,
                         absolute_sigma=True, maxfev=50000,
                         bounds=([p.min() - .05, .3, 0, -np.inf, -np.inf, -np.inf],
                                 [p.max() + .05, 5, 1, np.inf, np.inf, np.inf]))
    except Exception as ex:
        return sf.ScalingFit(None, None, None, None, None, None, len(keys), False, f"corrected fit failed: {ex}")
    chi2 = float((((f((d, p), *P) - y) / s) ** 2).sum() / (len(y) - 6))
    err = math.sqrt(C[0, 0]) * max(1.0, math.sqrt(chi2))
    reason = "" if (chi2 <= sf.MAX_CHI2 and sf.NU_RANGE[0] <= P[1] <= sf.NU_RANGE[1]) else f"chi2 {chi2:.1f} / nu {P[1]:.2f}"
    return sf.ScalingFit(float(P[0]), err, err, 0.0, float(P[1]), chi2, len(keys), not reason, reason)

sf.scaling_threshold = corrected
out = Path(sys.argv[1])
for key in ("crg_local_ZZ", "crg_local_XX", "crg_prep_ZZ", "crg_prep_XX"):
    reproduce.plot(paper.FIGURES[key], out_dir=out, results_dir=Path("results/cluster").resolve(), pdf=True)
