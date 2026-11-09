# Erasure thresholds in the correctable-regime plots

How each point in the correctable-regime (crg) figures is obtained
(`experiments/figures/scaling_fit.py`).

## What a point is

For a fixed local noise strength (circuit noise $p_c$, or resource-state error
$p_r$), we sweep the erasure rate $p_e$ and estimate the logical error rate
$P_L(p_e, d)$ at $d = 11, 13, 15, 17, 19$. Each $(d, p_e)$ point is sampled
until $10^6$ shots or $10^5$ logical errors, whichever comes first, and decoded
with correlated matching. Distances 11–15 are sampled on a grid of spacing
0.005 over a fixed ±0.03 window around a prior estimate (`paper.crg_window`).
Distances 17 and 19 were added later, in two refinement rounds, over the
scaling window of the fit at the time. Those rounds are frozen in
`experiments/crg_refinement.json` and are part of the plan (`paper.Curve.runs`).
The erasure threshold $p_e^*$ at that noise strength is where the curves for
different $d$ cross: below it, larger codes do better, and above it they do
worse. The $p_c = p_r = 0$ point is the pure-erasure sweep, fitted the same
way over its own distances ($d = 11$–19). It is the same number as the
threshold in the pure-erasure figure.

## The fit

Close to threshold, finite-size scaling says $P_L$ depends on $p_e$ and $d$
only through the rescaled variable

$$x = (p_e - p_e^*)\, d^{1/\nu},$$

which we expand to second order (Wang, Harrington & Preskill 2003):

$$P_L = A + B x + C x^2 .$$

The parameters $p_e^*$, $\nu$, $A$, $B$, $C$ are fitted jointly to all
distances by weighted least squares. Each point is weighted by its binomial
standard error, $\sqrt{N_{\rm err}(1 - P_L)}/N_{\rm shots}$, with the error
count floored at one.

**Fit window.** The expansion only holds near threshold. How near is set by
$x$, not $p_e$, because larger distances leave the linear regime sooner. So
the fit uses only points with

$$|p_e - p_e^*|\, d^{1/\nu} \le x_{\max} = 0.15,$$

which is about $\pm 0.03$ in $p_e$ at $d = 11$. The window is first centred on
the crossing of the two smallest distances with $\nu = 1.5$, then re-centred
on the fitted $(p_e^*, \nu)$ until it stops moving. We chose
$x_{\max} = 0.15$ by scanning the window width. Narrower windows hold too few
points and give degenerate fits. Wider windows include points where the
quadratic form fails, and $\chi^2/\mathrm{dof}$ rises to 2–10. In between is a
plateau where $p_e^*$ is stable to about $10^{-3}$ and $\chi^2/\mathrm{dof}
\approx 1$.

**Validity.** A point is plotted only if its fit satisfies all of these:

- at least 3 points per distance inside the window, and at least 2 at the
  largest distance;
- points on both sides of $p_e^*$ at every distance;
- $0.8 \le \nu \le 2.5$.

There is no $\chi^2$ cut. With $\sim 10^6$ shots per point, the slow drift
of the crossing with $d$ shows up in $\chi^2$, and such fits are kept. Points
that fail are left out of the figures, not replaced by a cruder estimate.
At present 154 of the 156 crg points pass. The other two are closed regions
at $p_r = 6\%$, where the curves don't cross.

**Error bar.** The quoted error is the statistical error of $p_e^*$ from the
fit covariance (median about $4 \times 10^{-4}$), as is usual for QEC
thresholds. On 2026-11-04 we dropped a $\sqrt{\chi^2/\mathrm{dof}}$ scaling,
a window systematic and a finite-size (leave-one-distance-out) systematic,
because none of them is standard in QEC threshold papers. The finite-size drift
is reported separately instead (see below).

**Data use.** About 88% of the collected $(d, p_e)$ points and shots fall
inside the final fit windows. The rest are edges of the fixed d = 11–15 grids
that the fit window, which narrows as $d^{-1/\nu}$, leaves out.

## Caveats for the text

- **The error bars describe the estimate at the simulated distances, not the
  infinite-size threshold.** The scaling form is leading order: it has no
  correction-to-scaling term $D\, d^{-1/\mu}$. So the crossing can still drift
  slowly with $d$, and that drift is not in the bar. Adding the correction
  term was tried; at $d \le 19$ it leaves $p_e^*$ nearly undetermined (error
  about 9 times larger).
- **Measured drift.** Refitting without $d = 11$ shifts $p_e^*$ by $0.001$
  (median $|$shift$|$), at most about $0.008$, and downwards at 116 of 154 points.
  The shift exceeds the fit error at 121 of 154 points.
  - **Circuit noise:** the shift is larger for $X\bar X$ than for $Z\bar Z$ in
    every family (mean $|$shift$|$ 0.0026 against 0.0016; $X\bar X$ is larger at 42
    of 50 paired points). In all five families, the $X\bar X$ logical has lower
    distance across the interface than $Z\bar Z$; for BellZigZag the deficit
    is only $d$ against $d + 1$ (erasure).
  - **Resource-state noise:** the drift is mostly below 0.002, with no
    consistent $X\bar X$/$Z\bar Z$ pattern.

  Suggested wording: "Excluding $d = 11$ shifts the thresholds by $0.001$
  (median), at most $0.008$, mostly downward, indicating residual
  finite-size drift."
- **At pure erasure, the drift has a known source.** For Bell,
  BellZigZag and 3d-Chain the two observables are exact duals:
  $P_L^{XX}(p) = \tfrac12 - P_L^{ZZ}(1-p)$, checked numerically at every
  distance. So $p_e^{*,XX} = 1 - p_e^{*,ZZ}$, and the offsets from 50% (up to
  about 0.6%) are a finite-size bias toward one observable. Refitting over
  $d = 11$–15, 13–17 and 15–19 moves the estimates toward 50%. The fitted
  $\nu \approx 1.35$–1.45 is close to the 2D percolation value $4/3$.

## Reference

C. Wang, J. Harrington, J. Preskill, *Confinement-Higgs transition in a
disordered gauge theory and the accuracy threshold for quantum memory*,
Ann. Phys. **303**, 31 (2003).
