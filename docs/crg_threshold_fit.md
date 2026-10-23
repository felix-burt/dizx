# Erasure thresholds in the correctable-regime plots

How each point in the correctable-regime (crg) figures is obtained
(`experiments/figures/scaling_fit.py`).

## What a point is

For a fixed local noise strength (circuit noise $p_c$, or resource-state error
$p_r$), we sweep the erasure rate $p_e$ and estimate the logical error rate
$P_L(p_e, d)$ at several code distances: $d = 11, 13, 15$ for circuit noise,
$d = 11, 13$ for resource-state noise, with $d = 15$ added where the two-distance
fit was not constrained. Each $(d, p_e)$ point is a Monte Carlo estimate from
roughly $10^5$–$10^6$ shots, decoded with correlated matching. The erasure
threshold $p_e^*$ at that noise strength is where the curves for different $d$
cross: below it, larger codes do better, and above it they do worse. The
$p_c = p_r = 0$ point is the pure-erasure sweep, fitted the same way over its
own distances ($d = 11$–19); it is the same number as the threshold in the
pure-erasure figure.

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
- $0.8 \le \nu \le 2.5$ and $\chi^2/\mathrm{dof} \le 5$.

Points that fail are left out of the figures rather than replaced by a
cruder estimate.

**Error bar.** The quoted error combines two parts in quadrature:

1. the statistical error of $p_e^*$ from the fit covariance, scaled up by
   $\sqrt{\chi^2/\mathrm{dof}}$ when that exceeds 1;
2. a window systematic: half the spread of $p_e^*$ over
   $x_{\max} \in \{0.10, 0.15, 0.20\}$, taking only the fits that pass the
   validity checks.

## Caveats for the text

- **The error bars describe the estimate at the simulated distances, not the
  infinite-size threshold.** The scaling form has no correction-to-scaling
  term, so any slow drift of the crossing with $d$ does not show up in the
  bar.
- **At pure erasure the drift is visible and has a known source.** For Bell,
  BellZigZag and 3d-Chain the two observables are exact duals:
  $P_L^{XX}(p) = \tfrac12 - P_L^{ZZ}(1-p)$, checked numerically at every
  distance. So $p_e^{*,XX} = 1 - p_e^{*,ZZ}$, and the offsets from 50% (up to
  about 0.6%) are a finite-size bias toward one observable. Refitting over
  $d = 11$–15, 13–17 and 15–19 moves the estimates toward 50%. The fitted
  $\nu \approx 1.35$–1.45 is close to the 2D percolation value $4/3$.
- **Assume a comparable bias of order 0.5–1% at nonzero local noise**, even
  though there is no exact reference value to expose it there. Differences
  between protocols smaller than this should not be read as significant.

## Reference

C. Wang, J. Harrington, J. Preskill, *Confinement-Higgs transition in a
disordered gauge theory and the accuracy threshold for quantum memory*,
Ann. Phys. **303**, 31 (2003).
