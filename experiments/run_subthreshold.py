import argparse
import copy
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import sinter
from run_threshold_experiment import (run_threshold, aggregate_identifier, config_prefix,
                                      local_prob_label, results_dir, sinter_csv_path)
from dizx import result_names


def _read_all_stats(config: dict) -> list[sinter.TaskStats]:
    """Read every stat collected for a config from its result CSV(s).

    Keyed exactly like ``run_threshold`` writes them, so it picks up all
    distances/merge/observable combos collected so far. Deduplicated by
    strong_id in case a CSV holds re-collected shards.
    """
    noise_channels = config['noise_model']
    sweep = config['sweep']
    merge_types = sweep.get('merge_types', [None])
    observables = sweep.get('observables', [None])

    stats: list[sinter.TaskStats] = []
    seen: set[str] = set()
    for merge_type in merge_types:
        for observable in observables:
            path = sinter_csv_path(config, noise_channels, merge_type, observable,
                                   out_dir=results_dir(config))
            if Path(path).exists():
                for s in sinter.read_stats_from_csv_files(path):
                    if s.strong_id not in seen:
                        seen.add(s.strong_id)
                        stats.append(s)
    return stats


def _aggregate_by_p_distance(stats: list[sinter.TaskStats]):
    """Sum shots/errors/discards over rows sharing (p, distance)."""
    shots: dict[float, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    errors: dict[float, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    discards: dict[float, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for s in stats:
        md = s.json_metadata or {}
        p = float(md.get('p', -1))
        d = int(md.get('distance', -1))
        shots[p][d] += s.shots
        errors[p][d] += s.errors
        discards[p][d] += s.discards
    return shots, errors, discards


def _aggregate_by_p_distance_observable(stats):
    """Sum shots/errors/discards over rows sharing (p, distance, observable)."""
    shots: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    errors: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    discards: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for s in stats:
        md = s.json_metadata or {}
        p = float(md.get('p', -1))
        d = int(md.get('distance', -1))
        obs = md.get('observable')
        shots[p][d][obs] += s.shots
        errors[p][d][obs] += s.errors
        discards[p][d][obs] += s.discards
    return shots, errors, discards


def _bell_average_rates(shots_o, errors_o, discards_o):
    """Mean logical error rate over the observables, per (p, distance).

    The merge leaves a logical Bell pair, whose fidelity depends on failures of
    *both* logical observables -- the one matching the merge basis and the
    perpendicular one.  The figure of merit is therefore the mean of the two
    per-observable rates, not the pooled rate over all shots: sinter stops each
    task at ``max_errors``, so the two observables generally get different shot
    counts and pooling would silently weight the slower-failing one more
    heavily.

    Returns ``{p: {d: (mean_ler, sigma, {observable: ler})}}``.  ``sigma`` is
    the binomial error on the mean, ``0.5 * sqrt(sum(sigma_i^2))``.
    """
    out: dict = defaultdict(dict)
    for p in shots_o:
        for d in shots_o[p]:
            per_obs, variances = {}, []
            for obs in sorted(shots_o[p][d], key=lambda o: (o is None, str(o))):
                n = shots_o[p][d][obs] - discards_o[p][d][obs]
                if n <= 0:
                    continue
                ler = errors_o[p][d][obs] / n
                per_obs[obs] = ler
                variances.append(ler * (1.0 - ler) / n)
            if not per_obs:
                continue
            mean = sum(per_obs.values()) / len(per_obs)
            sigma = math.sqrt(sum(variances)) / len(per_obs)
            out[p][d] = (mean, sigma, per_obs)
    return out


def _print_bell_summary(rates) -> None:
    print("\nBell-pair mean logical error rate vs distance "
          "(mean over observables):")
    for p in sorted(rates):
        print(f"  p = {p:g}")
        for d in sorted(rates[p]):
            mean, sigma, per_obs = rates[p][d]
            detail = "  ".join(f"{obs}={ler:.4e}" for obs, ler in per_obs.items())
            print(f"    d={d:>3}: mean LER={mean:.4e} +/- {sigma:.1e}   [{detail}]")


def _save_alpha_plot(shots, errors, discards, *, title: str, out_path: str,
                     rates=None) -> None:
    """LER vs code distance (log y) at each fixed probability, with error bars.

    Points with zero observed errors can't sit on a log axis, so they're drawn
    as rule-of-three upper bounds (3 / shots) with a hollow marker.

    When *rates* is supplied (the mean-over-observables series from
    :func:`_bell_average_rates`) it is plotted instead of the pooled counts --
    pooling across observables is not the Bell-pair figure of merit, so the
    plot would otherwise disagree with the printed summary and the fit.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    source = rates if rates is not None else shots
    for p in sorted(source):
        ds = sorted(source[p])
        xs, ys, yerr = [], [], []
        ub_xs, ub_ys = [], []
        for d in ds:
            if rates is not None:
                ler, sigma, _per_obs = rates[p][d]
                if ler <= 0.0:
                    n = sum(shots[p][d].values()) - sum(discards[p][d].values())
                    if n > 0:
                        ub_xs.append(d)
                        ub_ys.append(3.0 / n)
                else:
                    xs.append(d)
                    ys.append(ler)
                    yerr.append(sigma)
                continue
            n = shots[p][d] - discards[p][d]
            if n <= 0:
                continue
            ler = errors[p][d] / n
            if ler <= 0.0:
                ub_xs.append(d)
                ub_ys.append(3.0 / n)  # rule of three: 95% CI upper bound
            else:
                xs.append(d)
                ys.append(ler)
                yerr.append(math.sqrt(ler * (1.0 - ler) / n))
        line = None
        if xs:
            (line, _, _) = ax.errorbar(xs, ys, yerr=yerr, marker='o', capsize=3,
                                       linewidth=1.5, label=f'p = {p:g}')
        if ub_xs:
            colour = line.get_color() if line is not None else None
            ax.scatter(ub_xs, ub_ys, marker='v', facecolors='none',
                       edgecolors=colour, zorder=3,
                       label=None if line is not None else f'p = {p:g} (upper bound)')

    ax.set_yscale('log')
    ax.set_xlabel('Code distance $d$')
    ax.set_ylabel('Mean logical error rate over observables'
                  if rates is not None else 'Logical error rate')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)


def _print_summary(shots, errors, discards) -> None:
    print("\nSubthreshold LER vs distance:")
    for p in sorted(shots):
        print(f"  p = {p:g}")
        for d in sorted(shots[p]):
            n = shots[p][d] - discards[p][d]
            ler = errors[p][d] / n if n > 0 else float('nan')
            print(f"    d={d:>3}: LER={ler:.4e}  ({errors[p][d]}/{n} shots)")


def _extrapolate_distance(shots, errors, discards, target_ler: float,
                          rates=None) -> None:
    """Least-squares fit of ln(LER) vs d, extrapolated to *target_ler*.

    Below threshold the logical error rate falls exponentially in the code
    distance, so ln(LER) is linear in d and a straight-line fit extrapolates to
    an arbitrary target.  Only distances with at least one *observed* error are
    fitted: a zero-error point carries no LER estimate (the alpha plot draws it
    as a rule-of-three upper bound instead), and including it would bias the
    slope.  At least two such distances are needed for a fit.

    Reported alongside the raw crossing is the next odd distance at or beyond
    it, since the rotated surface code is defined at odd d.  This is an
    extrapolation well past the measured range -- treat it as an estimate whose
    accuracy rests on the fit staying exponential, not as a measurement.
    """
    label = "mean LER over observables" if rates is not None else "LER"
    print(f"\nExtrapolation to target {label} = {target_ler:.1e}:")
    source = rates if rates is not None else shots
    for p in sorted(source):
        pts = []
        if rates is not None:
            for d in sorted(rates[p]):
                mean = rates[p][d][0]
                if mean > 0:
                    pts.append((d, mean))
        else:
            for d in sorted(shots[p]):
                n = shots[p][d] - discards[p][d]
                if n > 0 and errors[p][d] > 0:
                    pts.append((d, errors[p][d] / n))
        if len(pts) < 2:
            print(f"  p = {p:g}: need >=2 distances with observed errors "
                  f"to fit (have {len(pts)})")
            continue
        ds = [float(d) for d, _ in pts]
        lys = [math.log(ler) for _, ler in pts]
        n = len(ds)
        mean_d = sum(ds) / n
        mean_y = sum(lys) / n
        denom = sum((d - mean_d) ** 2 for d in ds)
        if denom == 0:
            print(f"  p = {p:g}: all fitted points share one distance; cannot fit")
            continue
        slope = sum((d - mean_d) * (y - mean_y) for d, y in zip(ds, lys)) / denom
        intercept = mean_y - slope * mean_d
        fitted = ", ".join(f"d={int(d)}" for d in ds)
        if slope >= 0:
            print(f"  p = {p:g}: LER is not decreasing with distance "
                  f"(slope {slope:+.3f}); above threshold or too noisy to fit "
                  f"[fitted {fitted}]")
            continue
        # Suppression per two distance steps (the usual Lambda factor).
        lam = math.exp(-2.0 * slope)
        d_cross = (math.log(target_ler) - intercept) / slope
        d_odd = int(math.ceil(d_cross))
        if d_odd % 2 == 0:
            d_odd += 1
        print(f"  p = {p:g}: fit ln(LER) = {slope:+.4f}*d {intercept:+.4f}   "
              f"Lambda = {lam:.2f} per d+2   [fitted {fitted}]")
        print(f"           crossing at d = {d_cross:.1f}  ->  use d = {d_odd}")


def _config_for_local_prob(config: dict, l_prob: float) -> dict:
    """Resolve the ``p_l`` / ``p_nl`` placeholders for one ``sweep.local_probs`` entry.

    Same convention as ``run_correctable_regime``: ``probability: p_l`` channels
    take *l_prob*, ``probability: p_nl`` channels become ``scale: 1.0`` (driven
    by ``sweep.probs``), anything else passes through. ``local_probs`` is
    dropped from the returned sweep, so the result is an ordinary fixed-noise
    subthreshold config -- and ``extract_local_prob`` then keys its sinter CSVs
    by *l_prob*, giving each value its own file.
    """
    run_config = copy.deepcopy(config)
    run_config['noise_model'] = result_names.resolve_local_prob(config['noise_model'], l_prob)
    del run_config['sweep']['local_probs']
    return run_config


def run_subthreshold(config: dict) -> list[sinter.TaskStats]:
    """Collect subthreshold "Alpha plot" data: LER vs code distance at a fixed
    sub-threshold operating point.

    With ``sweep.local_probs`` set, the whole collection is repeated once per
    value, substituted for every ``probability: p_l`` channel. Each value gets
    its own sinter CSVs (keyed by local_prob, as everywhere else) and its own
    alpha plot, suffixed ``_<pl|pr><local_prob>``. Without it the config's
    noise model is used as-is and output names are unchanged.
    """
    local_probs = config['sweep'].get('local_probs')
    if local_probs is None:
        if any(ch.get('probability') in ('p_l', 'p_nl') for ch in config['noise_model']):
            raise ValueError("noise_model uses a p_l/p_nl placeholder but sweep.local_probs "
                             "is not set")
        return _run_subthreshold_point(config)

    stats: list[sinter.TaskStats] = []
    for l_prob in local_probs:
        run_config = _config_for_local_prob(config, l_prob)
        label = f"{local_prob_label(run_config['noise_model'])}{l_prob}"
        print(f"\n===== local_prob {label} =====")
        stats.extend(_run_subthreshold_point(run_config, label=label))
    return stats


def _run_subthreshold_point(config: dict, label: str = '') -> list[sinter.TaskStats]:
    """Subthreshold collection at one fixed local-noise level.

    Runs ``run_threshold`` one distance at a time (so the config's fixed local
    noise / single non-local point is swept over ``sweep.distances``) and, after
    each distance completes, re-reads the result CSV and refreshes a diagnostic
    PNG (log LER vs distance) at ``<results_dir>/<alpha_plot_stem>.png``
    (legacy configs: ``<results_dir>/alpha_plot_<id>[_<label>].png``)
    -- mirroring the incremental plotting of ``run_correctable_regime`` so a
    long run can be eyeballed as it progresses.
    """
    distances = list(config['sweep']['distances'])
    title = f'{config_prefix(config)} ({label})' if label else config_prefix(config)
    # A non-legacy stem already carries the substituted local_prob (as its
    # pl/pr noise param), so only legacy names need the label appended.
    if result_names.is_legacy(config):
        suffix = f'_{label}' if label else ''
        out_path = f'{results_dir(config)}/alpha_plot_{aggregate_identifier(config)}{suffix}.png'
    else:
        out_path = f'{results_dir(config)}/{result_names.alpha_plot_stem(config)}.png'
    target_ler = config.get('target_ler')
    average_observables = bool(config.get('average_observables', False))

    def _views(stats):
        """(plot/fit inputs) for this config's figure of merit."""
        if average_observables:
            so, eo, do = _aggregate_by_p_distance_observable(stats)
            return so, eo, do, _bell_average_rates(so, eo, do)
        s_, e_, d_ = _aggregate_by_p_distance(stats)
        return s_, e_, d_, None

    for d in distances:
        point_config = copy.deepcopy(config)
        point_config['sweep']['distances'] = [d]
        run_threshold(point_config)

        # Replot from everything collected so far, not just this distance.
        stats = _read_all_stats(config)
        shots, errors, discards, rates = _views(stats)
        _save_alpha_plot(shots, errors, discards, title=title,
                         out_path=out_path, rates=rates)
        print(f"[after d={d}] refreshed alpha plot -> {out_path}")
        # Refresh the extrapolation too, so a long run shows the estimate
        # settling rather than only revealing it at the end.
        if target_ler is not None:
            _extrapolate_distance(shots, errors, discards, float(target_ler), rates=rates)

    stats = _read_all_stats(config)
    shots, errors, discards, rates = _views(stats)
    if rates is not None:
        _print_bell_summary(rates)
    else:
        _print_summary(shots, errors, discards)
    if target_ler is not None:
        _extrapolate_distance(shots, errors, discards, float(target_ler), rates=rates)
    print(f"\nSaved alpha plot to {out_path}")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Collect a subthreshold Alpha-plot point (LER vs distance at a '
                    'fixed sub-threshold noise point) from a subthreshold_* config.')
    parser.add_argument('config', help='Path to a subthreshold_* experiment config (.yaml).')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    run_subthreshold(config)
