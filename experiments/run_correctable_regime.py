import argparse
import copy
import sys
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_threshold_experiment import bisection_tolerance_at, has_bisection_tolerance, run_threshold, run_threshold_bisection, run_threshold_bracket_sweep, fit_threshold, load_threshold_stats, refine_window_probs, results_dir, snap_center, slide_refine_window
from _regime_common import (crg_data_path_for, crg_stem, read_crg_data, write_crg_data,
                            refine_state_path_for, read_refine_state, write_refine_state,
                            fit_boundary_curve, eval_boundary_curve, boundary_x_max)
def _crossing_x(xs, y_a, y_b):
    diff = [a - b for a, b in zip(y_a, y_b)]
    for i in range(len(xs) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if d0 == 0:
            return xs[i]
        if d0 * d1 < 0:
            x0, x1 = xs[i], xs[i + 1]
            return x0 - d0 * (x1 - x0) / (d1 - d0)
    return None


def _estimate_threshold(stats, distances):
    by_d = defaultdict(dict)
    for s in stats:
        md = s.json_metadata or {}
        d = int(md.get('distance', 0))
        p = float(md.get('p', 0))
        shots = s.shots - s.discards
        if shots > 0:
            by_d[d][p] = s.errors / shots

    sorted_ds = sorted(distances)
    if not all(d in by_d for d in sorted_ds):
        return None

    common_ps = set(by_d[sorted_ds[0]])
    for d in sorted_ds[1:]:
        common_ps &= set(by_d[d])
    xs = sorted(common_ps)
    if len(xs) < 2:
        return None

    # Find first x where highest distance has strictly lower LER than lowest distance
    # (correctable ordering begins), then include one step back to bracket the crossing.
    # Skip any points where LER is 0 for the highest distance, as those are likely noise floor artifacts rather than true crossings.
    d_high, d_low = sorted_ds[-1], sorted_ds[0]
    start_idx = next(
        (i for i, x in enumerate(xs) if by_d[d_high][x] < by_d[d_low][x] and by_d[d_high][x] > 0),
        None,
    )
    if start_idx is None:
        return None
    search_xs = xs[max(0, start_idx - 1):]

    crossings = []
    for d_a, d_b in zip(sorted_ds, sorted_ds[1:]):
        c = _crossing_x(search_xs, [by_d[d_a][x] for x in search_xs], [by_d[d_b][x] for x in search_xs])
        if c is not None:
            crossings.append(c)
    return sum(crossings) / len(crossings) if crossings else None


def estimate_threshold_from_stats(sweep, stats, distances, p_center=None, p_halfwidth=None):
    """Estimates the threshold from collected stats per sweep['estimator'].

    'fit' uses the finite-size scaling fit (falling back to the pairwise
    crossing estimate if the fit fails). When a window is given, the fit
    window is recentred on the data's crossing within it (same halfwidth),
    falling back to p_center when no crossing is found. 'crossing' (the default, and any
    other value) linearly interpolates the crossing of adjacent distance
    curves directly, averaged over the distance pairs.

    Returns (threshold, stderr, method). stderr is None for the crossing
    estimator, which has no associated error model -- so a `target_stderr`
    skip can never trigger for a crossing-only config, and every local_prob
    is re-collected on each run.
    """
    if sweep.get('estimator') == 'fit':
        if p_center is not None and p_halfwidth is not None:
            # The window centre handed in is wherever the search stopped (slide
            # bracket, bisection midpoint, stored threshold), which can sit
            # off-centre from the actual crossing. Recentre the fit window on
            # the crossing of the data inside that window, keeping its width,
            # so the scaling ansatz is fitted symmetrically about the threshold.
            in_window = [s for s in stats
                         if abs(float((s.json_metadata or {}).get('p', 0.0)) - p_center)
                         <= p_halfwidth * (1 + 1e-9)]
            crossing = _estimate_threshold(in_window, distances)
            if crossing is not None:
                print(f"Fit window recentred on data crossing {crossing:.6g} "
                      f"(was {p_center:.6g}, halfwidth {p_halfwidth:.6g})")
                p_center = crossing
        fit = fit_threshold(stats, distances, p_center=p_center, p_halfwidth=p_halfwidth)
        if fit is not None:
            return fit[0], fit[1], 'fit'
        print("Scaling fit failed; falling back to crossing estimate")
    return _estimate_threshold(stats, distances), None, 'crossing'


def _save_correctable_regime_plot(local_probs_done, thresholds, out_path):
    pairs = [(x, t) for x, t in zip(local_probs_done, thresholds) if t is not None]
    xs = [x for x, _ in pairs]
    ys = [t for _, t in pairs]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if xs:
        ax.plot(xs, ys, marker='o', linewidth=2, label='Threshold boundary')
        ax.fill_between(xs, 0, ys, alpha=0.2, label='Correctable region')
    ax.set_xlabel('Local noise probability')
    ax.set_ylabel('Non-local noise threshold')
    ax.set_title('Correctable regime boundary')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _slide_start_center(run_config, distances, lo, hi):
    """Starting centre for the slide search when there is no stored threshold.

    Uses the midpoint of whatever probabilities are already collected for this
    local_prob (so a resync of mis-centred windows starts the search right on
    top of them, making the first edge probe a cache hit), falling back to the
    middle of the configured [min, max] range when nothing has been collected.
    The slide corrects any of these to a bracketing window regardless.
    """
    try:
        stats = load_threshold_stats(run_config)
    except Exception:
        stats = []
    probs = sorted({float((s.json_metadata or {}).get('p'))
                    for s in stats if (s.json_metadata or {}).get('p') is not None})
    if probs:
        return (probs[0] + probs[-1]) / 2.0
    return None


def _record_refine_state(refine_state, path, l_prob, probs, center, threshold,
                         max_shots, max_errors, plot_only, distances):
    """Persist the outcome of one refine-window collection for `l_prob`.

    `found_in_region` is True when the estimated threshold fell inside the
    swept window -- the signal that the refinement has converged and a later
    run may skip re-collecting it. plot_only runs never mutate the state (they
    collect nothing, so there is no new outcome to record). `distances` is the
    set of distances collected this pass, so a later run that adds a new
    distance can tell its budget wasn't actually covered by this freeze (see
    the `refine_done` check in `run_correctable_regime`).
    """
    if plot_only:
        return
    found = threshold is not None and probs[0] <= threshold <= probs[-1]
    refine_state[l_prob] = {
        'probs': list(probs),
        'center': center,
        'threshold': threshold,
        'found_in_region': bool(found),
        'max_shots': max_shots,
        'max_errors': max_errors,
        'distances': list(distances),
    }
    write_refine_state(path, refine_state)
    if not found:
        print(f"  refine local_prob={l_prob}: threshold "
              f"{'not found' if threshold is None else f'{threshold:.6g} landed outside'} "
              f"the window [{probs[0]:.6g}, {probs[-1]:.6g}] -- will re-refine next run")


def _sweep_for_local_prob(sweep, local_prob):
    """Return the effective sweep config for one local_prob.

    Configs may optionally provide `sweep.local_prob_sweeps`, a mapping
    `{local_prob: {<sweep overrides>}}`, to tighten/shift the non-local sweep
    range per local noise value directly from YAML.
    """
    local_overrides = sweep.get('local_prob_sweeps')
    if local_overrides is None:
        return copy.deepcopy(sweep)
    if not isinstance(local_overrides, dict):
        raise ValueError("sweep.local_prob_sweeps must be a mapping of local_prob -> sweep overrides")
    if local_prob is None:
        return copy.deepcopy(sweep)
    match = None
    for k, v in local_overrides.items():
        try:
            key_local_prob = float(k)
        except (TypeError, ValueError) as ex:
            raise ValueError(f"sweep.local_prob_sweeps key {k!r} is not a valid float") from ex
        if key_local_prob == float(local_prob):
            if not isinstance(v, dict):
                raise ValueError(
                    f"sweep.local_prob_sweeps[{k!r}] must be a mapping of sweep keys to override values")
            match = v
            break
    out = copy.deepcopy(sweep)
    if match is None:
        return out
    out.update(copy.deepcopy(match))
    # Keep the probability axis unambiguous when an override specifies one form.
    if 'probs' in match:
        out.pop('min', None)
        out.pop('max', None)
        out.pop('step', None)
    elif {'min', 'max', 'step'} & set(match.keys()):
        out.pop('probs', None)
    return out


def run_correctable_regime(config, plot_only=False):
    sweep = config['sweep']
    distances = sweep['distances']
    nl_thresholds = []
    stderrs = []
    methods = []
    data_path = crg_data_path_for(config)
    existing_data = read_crg_data(data_path)
    # Running tally of every known (threshold, stderr, method), seeded from
    # what's already on disk. The crg file is rewritten from this each pass, so
    # a local_prob that times out (threshold None) or is never reached this run
    # keeps its previously stored estimate instead of being dropped -- which
    # would otherwise force a fresh seed bisection next run.
    resolved = dict(existing_data)
    target_stderr = sweep.get('target_stderr')
    max_shots = int(config['sinter']['max_shots'])
    max_errors = int(config['sinter']['max_errors'])
    refine_state_path = refine_state_path_for(config)
    refine_state = read_refine_state(refine_state_path)
    for l_prob in sweep.get('local_probs', [None]):
        local_sweep = _sweep_for_local_prob(sweep, l_prob)
        distances = local_sweep['distances']
        cached = existing_data.get(l_prob)
        noise_model = config['noise_model']
        mapped_noise_model = []
        for ch in noise_model:
            if ch['probability'] == 'p_l':
                mapped_ch = {'tag' : ch['tag'], 'kind': ch['kind'], 'probability': l_prob}
                mapped_noise_model.append(mapped_ch)
            elif ch['probability'] == 'p_nl':
                mapped_ch = {'tag' : ch['tag'], 'kind': ch['kind'], 'scale': 1.0}
                mapped_noise_model.append(mapped_ch)
            else:
                # Pass through unchanged, e.g. fixed probabilities or 'p_e'
                # channels (which run_threshold resolves against the swept p
                # using their own 'n_boost').
                mapped_noise_model.append(ch)
        run_config = copy.deepcopy(config)
        run_config['noise_model'] = mapped_noise_model
        run_config['sweep'] = copy.deepcopy(local_sweep)
        # target_stderr skips curves already known accurately enough, but refine
        # mode deliberately re-collects a denser window around exactly those
        # known thresholds, so it must ignore the skip.
        skip = (not plot_only and not local_sweep.get('refine_around_threshold')
                and target_stderr is not None and cached is not None
                and cached[1] is not None and cached[1] <= target_stderr)
        # A refine curve is "done" once a previous run's estimate landed inside
        # the window it swept (found_in_region), the budget has not grown, and
        # every currently-configured distance was already part of that frozen
        # collection. Re-deriving the window from the refined threshold would
        # otherwise shift it slightly every run (the crossing estimate jitters
        # across the snap grid) and re-collect near-duplicate points forever.
        # Bumping sinter.max_shots/max_errors -- or adding a new distance -- is
        # the explicit "collect more" signal; a new distance falls through to
        # the reuse_window branch below, which re-sweeps the frozen window at
        # the current (now larger) distances list, and run_threshold's own
        # per-(distance, prob) cache skips everything except the new distance.
        prev_refine = refine_state.get(l_prob)
        # found_in_region alone isn't trustworthy: it may have been recorded
        # under a since-changed estimator -- exactly what happened when d=7
        # was excluded from threshold estimation, so a window centered on the
        # old d=7-biased crossing can silently no longer bracket the real
        # (d=9,11)-only threshold. Before treating the refinement as done,
        # re-derive the threshold from the data already collected in that
        # window using the *current* distances/estimator, and only trust
        # found_in_region if that re-derivation still lands inside the
        # window. If it doesn't (or can't be computed at all), don't skip --
        # fall through so the normal reuse_window/slide/bisection search
        # keeps looking instead of silently treating a stale window as
        # converged.
        window_reverified = False
        if (not plot_only and prev_refine is not None and prev_refine.get('found_in_region')
                and prev_refine.get('probs') and cached is not None):
            verify_frac = (float(local_sweep['refine_frac'])
                           if local_sweep.get('refine_around_threshold') else None)
            verify_center = prev_refine.get('center')
            verify_stats = load_threshold_stats(run_config)
            verify_threshold, _, _ = estimate_threshold_from_stats(
                local_sweep, verify_stats, distances, p_center=verify_center,
                p_halfwidth=(verify_frac * verify_center) if (verify_frac and verify_center) else None)
            lo_p, hi_p = prev_refine['probs'][0], prev_refine['probs'][-1]
            window_reverified = verify_threshold is not None and lo_p <= verify_threshold <= hi_p
            if not window_reverified:
                print(f"local_prob={l_prob}: re-checking with distances={distances} against the "
                      f"already-collected data in window [{lo_p:.6g}, {hi_p:.6g}] -- threshold "
                      f"{'not found' if verify_threshold is None else f'{verify_threshold:.6g} landed outside it'} "
                      f"-- won't treat this as converged, will keep searching")
        refine_done = (not plot_only and local_sweep.get('refine_around_threshold')
                       and prev_refine is not None and prev_refine.get('found_in_region')
                       and max_shots <= prev_refine.get('max_shots', 0)
                       and max_errors <= prev_refine.get('max_errors', 0)
                       and set(distances) <= set(prev_refine.get('distances', []))
                       and cached is not None
                       and window_reverified)
        if refine_done:
            threshold, stderr, method = cached
            print(f"local_prob={l_prob}: refinement already converged in its window "
                  f"(th={threshold:.6g}, budget {prev_refine['max_shots']} shots / "
                  f"{prev_refine['max_errors']} errors not exceeded) -- skipping")
        elif skip:
            threshold, stderr, method = cached
            print(f"local_prob={l_prob}: already known to stderr={stderr:.3g} "
                  f"<= target_stderr={target_stderr:.3g} ({method}) -- skipping sweep")
        else:
            print(f"{'Loading' if plot_only else 'Running'} for local_prob={l_prob} with noise model: {mapped_noise_model}")
            center = existing_data.get(l_prob, (None,))[0]
            # Curve-based estimate from every *other* local_prob already
            # resolved so far in this run (`resolved`, not the static
            # `existing_data` snapshot, so later points benefit from earlier
            # ones just computed) -- same fit used for the CRG plots, see
            # `_regime_common.fit_boundary_curve`. Used both to seed a search
            # when nothing is cached at this exact local_prob yet, and below
            # to sanity-check a frozen refine window.
            fit_params = fit_boundary_curve(resolved, x_max=boundary_x_max(config))
            curve_estimate = eval_boundary_curve(fit_params, l_prob) if fit_params is not None else None
            if curve_estimate is not None and curve_estimate <= 0:
                curve_estimate = None  # past the curve's vanishing point -- not a useful estimate
            if center is None and curve_estimate is not None:
                center = curve_estimate
                print(f"local_prob={l_prob}: no cached threshold -- seeding center="
                      f"{center:.6g} from the boundary curve fit (y0={fit_params[0]:.6g}, "
                      f"x_max={fit_params[1]:.6g}, m={fit_params[2]:.3g})")
            elif center is None and fit_params is None and resolved and l_prob is not None:
                # The curve needs 3 resolved points (4 when x_max is fitted too),
                # all anchored at local_prob=0. Early in a config's life -- or
                # after a run where most local_probs failed to resolve -- there
                # are fewer than that, and the search would otherwise start from
                # a bisection over the whole [min, max] range for every point.
                # The boundary is monotonic, so the nearest already-resolved
                # local_prob is a far better starting centre than the middle of
                # the range: slide_refine_window walks a ~18%-per-step window
                # from there and brackets in a handful of edge probes. As
                # `resolved` grows during the run this is the *previous*
                # local_prob, i.e. the search continues from the threshold it
                # just found.
                #
                # Deliberately gated on there being no curve *at all*, rather
                # than on curve_estimate being None: a fitted curve that
                # evaluates to <=0 is saying this local_prob is past the
                # vanishing point, and sliding down to near-zero from a
                # neighbour's threshold takes ~13 probes where the bisection
                # this falls through to gets there in ~5.
                nearest = min(resolved, key=lambda p: abs(p - l_prob))
                center = resolved[nearest][0]
                print(f"local_prob={l_prob}: no cached threshold and too few points for a "
                      f"boundary-curve fit ({len(resolved)}) -- continuing from the nearest "
                      f"resolved local_prob={nearest:g} (th={center:.6g})")
            # A prior run that converged in-window recorded the exact points it
            # swept. Re-collect *those* (to top up shots, or to plot) -- but
            # first recentre on wherever the curve now says the threshold
            # should be, rather than blindly trusting the window's originally
            # recorded centre. found_in_region is recorded True the moment a
            # threshold lands inside the swept window, but that recording is
            # never invalidated later if the *estimator itself* changes
            # underneath it -- exactly what happened when d=7 was excluded
            # from threshold estimation: the curve estimate can end up near
            # one edge of the old window (or outside it entirely) rather than
            # near its recorded centre, and blindly reusing the frozen centre
            # would keep sweeping a poorly-balanced or outright wrong window
            # forever. Recentring on whichever already-collected point in the
            # old window is *closest* to the curve estimate, then sweeping
            # the union of the old and new windows, means old points are free
            # (already collected -- run_threshold's own per-(distance,p)
            # cache skips them) while new ones extend coverage toward where
            # the curve now points.
            frac = float(local_sweep['refine_frac']) if local_sweep.get('refine_around_threshold') else None
            # Only reuse a frozen window that still *brackets* the threshold under
            # the current distances/estimator (window_reverified, computed above).
            # Reusing one that no longer does just re-sweeps a wrong window and
            # records found_in_region=False, deferring the real search to a later
            # run -- fine when re-runs are cheap, useless when the whole
            # collection has to converge inside one job. Falling through to the
            # slide branch instead lets slide_refine_window walk the window onto
            # the threshold within this run.
            # plot_only never re-verifies (it collects nothing), so it keeps the
            # old behaviour of re-rendering whatever window was last recorded.
            reuse_window = (local_sweep.get('refine_around_threshold') and prev_refine is not None
                            and prev_refine.get('found_in_region') and prev_refine.get('probs')
                            and (window_reverified or plot_only))
            if reuse_window:
                old_probs = list(prev_refine['probs'])
                if curve_estimate is not None:
                    n_pts = int(local_sweep.get('refine_points', 10))
                    lo_bound, hi_bound = float(local_sweep.get('min', 0.0)), float(local_sweep.get('max', 1.0))
                    closest = min(old_probs, key=lambda p: abs(p - curve_estimate))
                    new_probs = refine_window_probs(closest, frac, n_pts, lo_bound, hi_bound)
                    probs = sorted(set(old_probs) | set(new_probs))
                    center = closest
                    if not (old_probs[0] <= curve_estimate <= old_probs[-1]):
                        print(f"local_prob={l_prob}: boundary-curve estimate ({curve_estimate:.6g}) "
                              f"falls outside the previously converged window "
                              f"[{old_probs[0]:.6g}, {old_probs[-1]:.6g}] -- recentring around the "
                              f"nearest already-collected point ({closest:.6g}) instead of reusing "
                              f"the window unchanged")
                else:
                    probs = old_probs
                    center = prev_refine.get('center', probs[len(probs) // 2])
                print(f"Refine local_prob={l_prob}: re-collecting the previously "
                      f"converged window [{probs[0]:.6g}, {probs[-1]:.6g}] ({len(probs)} points) "
                      f"-- budget grew to {max_shots} shots / {max_errors} errors")
                rs = run_config['sweep']
                for key in ('min', 'max', 'step', 'tolerance', 'tolerance_rel', 'tolerance_min', 'bisection',
                            'refine_step', 'refine_pad', 'probs'):
                    rs.pop(key, None)
                rs['probs'] = probs
                stats = (load_threshold_stats(run_config) if plot_only
                         else run_threshold(run_config))
                threshold, stderr, method = estimate_threshold_from_stats(
                    local_sweep, stats, distances, p_center=center, p_halfwidth=frac * center)
                # The window was verified against the data available *before* this
                # sweep; filling in a missing distance can move the threshold back
                # out of it (the partly-collected-d=11 case). Rather than recording
                # the miss and leaving it for a later run, slide onto the threshold
                # and sweep again now -- a single retry, so a pathological curve
                # can't loop.
                if not plot_only and (threshold is None
                                      or not (probs[0] <= threshold <= probs[-1])):
                    print(f"local_prob={l_prob}: threshold "
                          f"{'not found in' if threshold is None else f'{threshold:.6g} left'} "
                          f"the reused window [{probs[0]:.6g}, {probs[-1]:.6g}] once the "
                          f"missing distances were collected -- sliding onto it instead of "
                          f"deferring to a later run")
                    n_pts = int(local_sweep.get('refine_points', 10))
                    lo, hi = float(local_sweep.get('min', 0.0)), float(local_sweep.get('max', 1.0))
                    retry_start = threshold if threshold is not None else center
                    retry_center, retry_probs, _ = slide_refine_window(
                        run_config, retry_start, frac, n_pts, lo, hi, distances)
                    if retry_probs:
                        rs['probs'] = retry_probs
                        stats = run_threshold(run_config)
                        threshold, stderr, method = estimate_threshold_from_stats(
                            local_sweep, stats, distances, p_center=retry_center,
                            p_halfwidth=frac * retry_center)
                        probs, center = retry_probs, retry_center
                    else:
                        print(f"local_prob={l_prob}: slide could not bracket a threshold in "
                              f"[{lo:g}, {hi:g}] either -- leaving the window as swept")
                _record_refine_state(refine_state, refine_state_path, l_prob, probs, center,
                                     threshold, max_shots, max_errors, plot_only, distances)
            elif local_sweep.get('refine_around_threshold') and not plot_only:
                n_pts = int(local_sweep.get('refine_points', 10))
                lo, hi = float(local_sweep.get('min', 0.0)), float(local_sweep.get('max', 1.0))
                # Centre the refine window by sliding it until it brackets the
                # threshold (first point below, last above), probing only the
                # edges -- cheaper than a bisection and it reuses any
                # already-collected edge. Start from the stored threshold if we
                # have one, else from the midpoint of existing data.
                start = center if center else _slide_start_center(run_config, distances, lo, hi)
                probs = []
                if start is not None:
                    center, probs, _slide_stats = slide_refine_window(
                        run_config, start, frac, n_pts, lo, hi, distances)
                # Fall back to bisection when there is nothing to slide from (no
                # stored threshold, no existing data) or the slide never bracketed
                # (e.g. a coarse window stepped over the crossing): bisection
                # locates a centre over [min, max], then the slide validates and
                # produces the exact bracketing window from there.
                if (not probs and local_sweep.get('bisection')
                        and {'min', 'max'} <= local_sweep.keys()
                        and has_bisection_tolerance(local_sweep)):
                    print(f"Refine local_prob={l_prob}: "
                          f"{'no data to slide from' if start is None else 'slide did not bracket'}"
                          f" -- falling back to bisection to locate the centre")
                    bis_center, _ = run_threshold_bisection(run_config, persist=False)
                    center, probs, _slide_stats = slide_refine_window(
                        run_config, bis_center, frac, n_pts, lo, hi, distances)
                if not probs:
                    print(f"Refine local_prob={l_prob}: could not bracket a threshold in "
                          f"[{lo:g}, {hi:g}] -- leaving any prior estimate untouched")
                    threshold, stderr, method = None, None, None
                else:
                    rs = run_config['sweep']
                    for key in ('min', 'max', 'step', 'tolerance', 'tolerance_rel', 'tolerance_min', 'bisection',
                                'refine_step', 'refine_pad', 'probs'):
                        rs.pop(key, None)
                    rs['probs'] = probs
                    stats = run_threshold(run_config)
                    threshold, stderr, method = estimate_threshold_from_stats(
                        local_sweep, stats, distances, p_center=center, p_halfwidth=frac * center)
                    _record_refine_state(refine_state, refine_state_path, l_prob, probs, center,
                                         threshold, max_shots, max_errors, plot_only, distances)
            elif local_sweep.get('refine_around_threshold') and center:
                # plot_only with a known threshold: rebuild the window (snapped,
                # as historically saved) and re-render from cached CSVs without
                # collecting anything.
                n_pts = int(local_sweep.get('refine_points', 10))
                lo, hi = float(local_sweep.get('min', 0.0)), float(local_sweep.get('max', 1.0))
                snap = float(local_sweep['refine_snap']) if local_sweep.get('refine_snap') else None
                snap_rel = float(local_sweep['refine_snap_rel']) if local_sweep.get('refine_snap_rel') else None
                if snap and snap_rel:
                    raise ValueError("sweep sets both refine_snap (absolute grid) and "
                                     "refine_snap_rel (grid relative to the threshold's "
                                     "scale); pick one")
                center = snap_center(center, snap, snap_rel)
                probs = refine_window_probs(center, frac, n_pts, lo, hi)
                rs = run_config['sweep']
                for key in ('min', 'max', 'step', 'tolerance', 'tolerance_rel', 'tolerance_min', 'bisection',
                            'refine_step', 'refine_pad', 'probs'):
                    rs.pop(key, None)
                rs['probs'] = probs
                stats = load_threshold_stats(run_config)
                threshold, stderr, method = estimate_threshold_from_stats(
                    local_sweep, stats, distances, p_center=center, p_halfwidth=frac * center)
            elif local_sweep.get('refine_around_threshold'):
                # plot_only with no stored threshold -- nothing collected to show.
                print(f"No known threshold for local_prob={l_prob} to refine around "
                      f"(plot_only) -- skipping")
                threshold, stderr, method = None, None, None
            elif local_sweep.get('bisection'):
                if local_sweep.get('refine_step'):
                    center, stats = run_threshold_bracket_sweep(run_config)
                    pad = float(local_sweep.get('refine_pad', 2 * bisection_tolerance_at(local_sweep, center)))
                    threshold, stderr, method = estimate_threshold_from_stats(local_sweep, stats, distances,
                                                              p_center=center, p_halfwidth=pad)
                else:
                    threshold, _ = run_threshold_bisection(run_config)
                    stderr, method = None, 'bisection'
            else:
                stats = load_threshold_stats(run_config) if plot_only else run_threshold(run_config)
                threshold, stderr, method = estimate_threshold_from_stats(local_sweep, stats, distances)
            print(f"Estimated threshold local_prob={l_prob}: {threshold} (stderr={stderr}, method={method})")
        nl_thresholds.append(threshold)
        stderrs.append(stderr)
        methods.append(method)
        # Only a resolved (non-None) threshold updates the stored estimate; a
        # None (timeout / no crossing this run) leaves the prior value intact.
        if threshold is not None:
            resolved[l_prob] = (threshold, stderr, method)
        out_path = f'{results_dir(config)}/{crg_stem(config)}.png'
        # Save intermediate results and plot
        local_probs_done = sweep.get('local_probs', [None])[:len(nl_thresholds)]
        _save_correctable_regime_plot(local_probs_done, nl_thresholds, out_path)
        write_crg_data(data_path,
                       [(lp, t, se, m) for lp, (t, se, m) in resolved.items()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--plot-only', action='store_true',
                         help='Skip data collection and only plot from existing results')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_correctable_regime(config, plot_only=args.plot_only)
