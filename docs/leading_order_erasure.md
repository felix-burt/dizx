# Leading-order logical error rate under pure erasure

This note explains the "leading order" curves (the translucent bands) in the
loss-only row of the Λ figures (`subthreshold_lambda_erasure_grid`): what they
compute, why the computation is exact in structure, when it is accurate, and
how it was checked. Code: `experiments/figures/leading_order.py` (`erasure_edges`,
`min_logical`), drawn by `_leading_order_lambda` in
`experiments/figures/lambda_grid.py`.

Everything here is for the **loss-only** setting, $p_c = p_r = 0$: the only
noise is photon loss, which appears as heralded fusion erasure.

---

## 1. From photon loss to erasure

A fusion boosted at level $n$ fails (is erased) with probability

$$
p_e = 1 - \left(1 - \frac{p_f}{2}\right)(1 - p_\mathrm{loss})^{1/p_f},
\qquad p_f = 2^{-n},
$$

(`boosted_erasure_probability`). Every fusion measurement in the circuit is
erased independently with this probability, and the erasure is **heralded**:
the decoder knows which measurements were lost. So the whole problem is a
function of $p_e$ alone. The boost level $n$ only decides which $p_e$ a given
loss rate produces: $n=1,2,3$ have floors $p_e = 0.25, 0.125, 0.0625$ at zero
loss.

## 2. Pure erasure is a percolation problem

**When the decoder fails.** An erased measurement outcome is replaced by a
uniformly random one. That is equivalent to applying a uniformly random Pauli
on the erased locations. Two cases:

* The erased set **does not contain** a logical operator (an undetectable
  error with odd observable parity). Then the syndrome, together with the
  known erased locations, fixes the error up to stabilisers, and any decoder
  that corrects within the erased set succeeds. PyMatching with zero weight on
  the heralded edges is such a decoder.
* The erased set **does contain** a logical. Then the syndrome cannot tell the
  error from itself times that logical. The two are equally likely, so every
  decoder fails with probability exactly $1/2$.

Hence, exactly,

$$
\mathrm{LER} = \tfrac12 \,\Pr[\text{erased set contains a logical}].
$$

No decoder approximation enters. This is also the maximum-likelihood failure
rate.

**The decoding graph** (`erasure_edges`). stim's
`explain_detector_error_model_errors` lists, for every erasure location
(every `HERALDED_ERASE` target), which detectors and which observable its X
and its Z error flip. An erased location can apply any Pauli, so what it can
do is the span of its X and Z effects.

Each fusion also has a private consistency detector, which any flip of that
single fusion's outcome trips. An undetectable logical must leave that
detector unflipped, so only the span elements avoiding it can take part.
In every one of our circuits, each location has **exactly one** such usable
effect, and it touches **at most two detectors**. The code checks both
conditions and warns otherwise.

So each location is an **edge** between two detectors, or between one
detector and a single **boundary** node. Each edge carries a bit: whether it
flips the observable.

* A **logical** is then a closed walk through the boundary whose edges flip
  the observable an odd number of times. The code also checks, at small $d$,
  that no shorter odd cycle exists away from the boundary.
* Several locations can have identical effects (same detectors, same
  observable bit). They form one edge **class** of multiplicity $m$. The class
  is "open" if any of its $m$ locations is erased:
  $$q_e = 1 - (1-p_e)^{m_e} \approx m_e\, p_e.$$

Pure erasure decoding is therefore exactly **bond percolation** on a small
graph. A failure is an open odd-parity cycle through the boundary.

Graph sizes (ZZ observable; $N$ = erasure locations):

| family | $d=5$: $N$ / edge classes | $d=7$: $N$ / edge classes | multiplicities |
|---|---|---|---|
| Bell | 56 / 56 | 106 / 106 | all 1 |
| BellZigZag | 100 / 100 | 196 / 196 | all 1 |
| 5-Chain | 210 / 96 | 406 / 190 | 1–4 |
| 8-Chain | 160 / 96 | 308 / 190 | 1–3 |
| 3d-Chain | 110 / 96 | 210 / 190 | 1–2 |

## 3. The leading-order estimate

At small $p_e$ the failure is dominated by the **minimum-weight logicals**:
those using the fewest edges, $w$. Summing over them (a union bound, which
becomes exact as $p_e \to 0$) gives

$$
\mathrm{LER}_\mathrm{LO}(p_e) \;=\; \tfrac12 \sum_{L\ \text{minimal}} \;\prod_{e\in L} q_e
\;\approx\; \tfrac12\, \mathcal{N}\, p_e^{\,w},
\qquad \mathcal N = \sum_{L\ \text{minimal}} \prod_{e\in L} m_e .
$$

$\mathcal N$ counts minimal logicals weighted by multiplicity, i.e. the number
of distinct sets of $w$ erasure *locations* that complete a logical.

**How it is computed** (`min_logical`). A breadth-first search from the
boundary over states (node, observable parity). The first time the search
returns to (boundary, odd parity) gives $w$. The number of shortest walks,
each weighted by $\prod q_e$, gives the sum.

> **Each logical is counted once (fixed 2026-09-30).** A closed walk through
> the boundary can be traversed in two directions, so the BFS finds every
> minimal logical of weight $w \ge 2$ twice. `min_logical` halves its sum to
> correct for this. A $w=1$ logical is a self-loop and is a single walk.
>
> Before the fix the absolute $\mathrm{LER}_\mathrm{LO}$ was 2× too high. Λ,
> and hence the figure bands, never changed, because the factor cancels in the
> ratio. The corrected counts match a brute-force enumeration of all
> $w$-subsets exactly where that is feasible: Bell ZZ/XX at $d=3$ (2 and 4
> logicals) and 3d-Chain XX at $d=3$ (24).
> `results/pymatching/erasure_leading_order.json` holds the corrected counts.

Minimum weights and counts:

| family | ZZ: $w_5,\ \mathcal N_5$ | ZZ: $w_7,\ \mathcal N_7$ | XX: $w_5,\ \mathcal N_5$ | XX: $w_7,\ \mathcal N_7$ |
|---|---|---|---|---|
| Bell | 6, 3 | 8, 4 | 3, 6 | 4, 8 |
| BellZigZag | 6, 5 | 8, 7 | 5, 6 | 7, 8 |
| 5-Chain | 11, 433 269 | 15, 1.01×10⁸ | 5, 4176 | 7, 94 496 |
| 8-Chain | 11, 57 056 | 15, 5.9×10⁶ | 5, 1008 | 7, 12 706 |
| 3d-Chain | 11, 1783 | 15, 46 055 | 5, 148 | 7, 808 |

So for ZZ, Bell-type states have erasure distance $w = d+1$ and the chains
$w = 2d+1$. That factor of two is the source of the chains' asymptotic
advantage.

## 4. From the LER to Λ

The figures define Λ through $\mathrm{LER}(d) \propto \Lambda^{-(d+1)/2}$,
fitted between $d=5$ and $d=7$. That makes

$$
\Lambda_\mathrm{LO} = \frac{\mathrm{LER}_\mathrm{LO}(5)}{\mathrm{LER}_\mathrm{LO}(7)}
\;\approx\; \frac{\mathcal N_5}{\mathcal N_7}\; p_e^{-(w_7 - w_5)} .
$$

(The code evaluates the full $\prod q_e$ form, not the small-$p_e$ power law.
For general $d_1 \to d_2$ it raises the ratio to the power $2/(d_2-d_1)$.)

On the log–log axes, the band's slope is the **distance increment**
$w_7 - w_5$:

* **ZZ:** 4 for every chain, 2 for Bell/BellZigZag. So the chains' Λ grows
  as $p_e^{-4}$ and Bell's as $p_e^{-2}$.
* **XX:** 2 for the chains and BellZigZag, 1 for Bell.

The prefactor $\mathcal N_5/\mathcal N_7$ is the **entropy penalty**. The
chains have vastly more minimal logicals, and their number grows faster with
$d$. For example, ZZ 3d-Chain has $1783/46\,055 = 0.039$, while Bell has
$3/4 = 0.75$. The chains therefore start below Bell, and the steeper power law
overtakes it only at small $p_e$.

Worked example, ZZ at $p_e = 0.07$:

* **3d-Chain:** $0.039 \times 0.07^{-4} \approx 1600$. The full estimate
  gives 1612.
* **5-Chain:** $0.0043 \times 0.07^{-4} \approx 180$.
* **Bell:** $0.75 \times 0.07^{-2} \approx 150$, while Bell's measured Λ is
  $106 \pm 23$; see §5 for why Bell falls short of its own leading order.

## 5. When is leading order accurate?

Leading order ignores two things. The first is logicals of weight $w+1,
w+2, \dots$. The second is overlaps between minimal logicals. It is accurate
when the minimal logicals dominate. That depends on **how numerous the
near-minimal logicals are compared with the minimal ones**, not only on how
small $p_e$ is.

Measured LER divided by $\mathrm{LER}_\mathrm{LO}$, ZZ:

| family, $d$ | $p_e \approx 0.1$–$0.16$ | $\approx 0.2$ | $\approx 0.27$ | $\approx 0.32$ | $\approx 0.43$ |
|---|---|---|---|---|---|
| 3d-Chain, 5 | 1.00 | 1.00 | 0.9 | 0.8 | 0.5 |
| 3d-Chain, 7 | 1.00* | 0.98 | 0.94 | 0.75 | 0.41 |
| 8-Chain, 7 | 1.0–1.2 | 1.0 | 0.8 | 0.55 | 0.15 |
| 5-Chain, 7 | 1.1–1.2 | 1.1–1.2 | 0.7–0.8 | 0.4 | 0.07 |
| Bell, 5 | 1.9–2.7 (1.6 at 0.07) | 3.0 | 4.1 | 4.3 | 4.6 |
| Bell, 7 | 2.6–4.3 | 6–7 | 10 | 12 | 15 |

Sinter data (≥100 logical errors) except (*), which is from the rare-event
estimate of §6.

**Chains.** Leading order is accurate to about 5–20% for all
$p_e \lesssim 0.2$. It is still within 2× at 0.3. Far above that it
*over*-estimates, because the union bound double-counts overlapping logicals
once many edges are open.

**Bell / BellZigZag.** Leading order *under*-estimates, by 1.6× at
$p_e = 0.07$ and more at larger $p_e$ and $d$. There are very few minimal
logicals ($\mathcal N = 3$–$7$), so logicals of weight $w+1, w+2$ (paths with
one or two detours) outnumber them heavily. Each extra edge costs a factor
$p_e$ but gains a large combinatorial factor, so these terms are not
negligible at any $p_e$ in our range. Because the undercount grows with $d$,
Bell's measured Λ sits **below** its leading-order band, by ≈0.6× on the
0.08–0.16 plateau.

(This is our interpretation of the trend. The data only show that the
correction is larger at $d=7$ than at $d=5$ and grows with $p_e$.)

**Consequence for the figure.**

* For the chains below ~20% erasure, the leading-order band *is* the
  answer, to within the ≤20% shown above.
* For Bell/BellZigZag the band is an upper bound on Λ in practice, and the
  plotted points must come from sampling.
* The measured 5→7 vs 7→9 check (d=9 runs) found
  $\Lambda(7\to9)/\Lambda(5\to7) \approx 1.0$ within errors. The leading
  order predicts 1.07–1.13, which is consistent.

## 6. How the leading order was validated below sampling reach

Direct sampling cannot test the chains below $p_e \approx 0.13$, because
their $d=7$ LERs there are $10^{-9}$ to $10^{-13}$. Instead, the exact
percolation formula

$$
\mathrm{LER}(p_e) = \tfrac12 \sum_{k=0}^{N} \binom{N}{k} p_e^k (1-p_e)^{N-k}\, f_k,
$$

was evaluated directly. Here $f_k$ is the probability that $k$ uniformly
random erased locations contain a logical. One set of $f_k$ gives the LER at
every $p_e$ at once.

**Estimating $f_k$.**

* At large $k$, $f_k$ is not small and is sampled directly. Checking one
  erasure set is a union-find pass with parity over at most $N$ edges.
* At smaller $k$, $f_k$ is obtained one step at a time from the identity
  $$
  \frac{F_k}{F_{k+1}} = \frac{k+1}{N-k}\; g_{k+1},
  $$
  where $F_k = \binom Nk f_k$ counts failing $k$-sets. $g_{k+1}$ is the mean
  fraction of a failing $(k+1)$-set's elements whose removal leaves it
  failing.
* $g$ is never rare (≈0.1–1). It is estimated with a Markov chain over
  failing sets: swap one erased location for one unerased, and accept if the
  set still fails. This is the splitting idea of Bravyi and Vargo, applied to
  the number of erasures.

**Checks.**

* **Against sinter:** agreement within 1–5% (inside the sinter error bars) at
  every resolved point. That is 13–16 points per chain and distance, over
  LERs from $10^{-7}$ to $10^{-1}$, for all three chains at $d=5$ and $d=7$.
* **At the bottom of the ladder**, $k=w$: $F_w = \mathcal N$ exactly. The
  chain's estimate matches the exact count within 5% (5-Chain, 3d-Chain) and
  15% (8-Chain). This check is what exposed the (now fixed) 2× double count
  in `min_logical`.
* **Error bars:** from the spread over independent seeds (in progress at the
  time of writing).

For the chains at $p_e < 0.2$, these estimates reproduce the leading order to
within the table in §5: Λ/Λ_LO = 0.97–0.99 for 3d-Chain, 0.90–0.98 for
8-Chain, and 0.82–0.92 for 5-Chain. That is the quantitative basis for trusting
the chains' leading-order bands where no direct data exist.

## 7. Limits

* **Loss only.** With circuit or resource-state noise ($p_c, p_r > 0$), Pauli
  errors are not heralded and failure is no longer percolation. A mixed
  leading order would weigh each unheralded edge by roughly $\sqrt{p}$
  against $q$ per erased edge. That extension has been deferred.
* **Graphlike structure is checked, not assumed.** A circuit with a location
  of two usable effects, or an effect on three or more detectors, makes
  `erasure_edges` warn. No current family does.
* **The $d=5 \to 7$ step.** The bands use the same distance pair as the
  data. Leading order at larger $d$ costs nothing to evaluate. It is the
  natural way to show how Λ evolves beyond $d=7$.
