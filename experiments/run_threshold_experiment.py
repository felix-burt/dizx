import copy
import math
import os
import shutil
import sys
import tempfile
import time

from pathlib import Path
from sinter._data._task import Task
from stim import Circuit
from typing import Any
import numpy as np
import yaml
import sinter
from dizx.builders import BUILDERS, SEDBuild
from dizx.noise_models import NoiseSpec, apply_tagged_noise_model
from dizx.sinter_csv import compact_sinter_csv
from dizx import result_names
from dizx.sinter_decoders import PyMatchingHeraldedDecoder
import matplotlib.pyplot as plt

def build_circuit(builder_name: str, distance: int, observable: str | None = None,
                  merge_type: str | None = None) -> SEDBuild:
    """Build a config's experiment with dizx.builders.BUILDERS. Memory configs
    have no merge type, so it is forwarded only when set."""
    if builder_name not in BUILDERS:
        raise ValueError(f"Unknown builder name: {builder_name} (known: {sorted(BUILDERS)})")
    kwargs: dict[str, Any] = dict(distance=distance, observable=observable, n_cycles=distance)
    if merge_type is not None:
        kwargs['merge_type'] = merge_type
    return BUILDERS[builder_name](**kwargs)


_circuit_cache: dict[tuple, Circuit] = {}

CIRCUIT_CACHE_DIR = Path('results/circuits')


def circuit_file_name(builder_name: str, distance: int, observable: str | None,
                      merge_type: str | None) -> str:
    """`<builder>_d<d>[_merge<merge>]_obs<obs>.stim`, e.g.
    surgery_bell_prep_fusion_d5_mergeZZ_obsZZ.stim. The builder name fixes
    everything else about the circuit (n_cycles is always d)."""
    merge = f'_merge{merge_type}' if merge_type else ''
    return f'{builder_name}_d{distance}{merge}_obs{observable}.stim'


def cached_circuit(builder_name: str, distance: int, observable: str | None,
                   merge_type: str | None) -> Circuit:
    """Cached noiseless circuit for a build configuration.

    The built circuit depends only on these arguments, not on the noise
    probability, so it is shared across sweep points, bisection midpoints,
    and (via a stim file under CIRCUIT_CACHE_DIR) across processes and
    experiment configs. Stim's text format round-trips exactly, including
    the instruction tags apply_tagged_noise_model keys on, and the noise
    pass returns a fresh circuit, so the cached one is never mutated.

    The file name doesn't track the builder code: after a change that alters
    a builder's circuits, delete its files under CIRCUIT_CACHE_DIR.
    """
    key = (builder_name, distance, observable, merge_type)
    if key not in _circuit_cache:
        cache_file = CIRCUIT_CACHE_DIR / circuit_file_name(*key)
        if not cache_file.exists():
            circuit = build_circuit(builder_name, distance, observable, merge_type).circuit
            CIRCUIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            # Atomic write so concurrent experiment runs racing on the same
            # circuit never observe a partially written cache file.
            tmp_file = cache_file.with_suffix(f'.tmp{os.getpid()}')
            circuit.to_file(tmp_file)
            tmp_file.replace(cache_file)
        # Use the round-tripped circuit, not the in-memory build: stim's text
        # format truncates float args (e.g. DETECTOR coordinate -10/9 ->
        # -1.11111), so the two differ, give different DEMs and strong_ids,
        # and data collected by the run that built the circuit would never be
        # recognised by later runs that load the file.
        _circuit_cache[key] = Circuit.from_file(cache_file)
    return _circuit_cache[key]


def snap_grid(center: float, snap: float | None, snap_rel: float | None) -> float | None:
    """Grid spacing to normalise a threshold estimate onto.

    `snap` is an absolute spacing. `snap_rel` instead asks for a spacing that
    is a fraction of the threshold's own *scale* -- literally snap_rel times
    the decade the centre lives in, so a 0.5 threshold and a 0.005 one get
    proportionally sized grids (0.05*0.1 = 0.005 and 0.05*0.001 = 5e-5).
    Anchoring on the decade rather than on the centre itself keeps the grid
    fixed across a run of nearby thresholds, which is the whole point of
    snapping; anchoring on the centre would make every point its own grid.
    """
    if snap_rel:
        return snap_rel * 10.0 ** math.floor(math.log10(center))
    return snap or None


def snap_center(center: float | None, snap: float | None = None,
                snap_rel: float | None = None) -> float | None:
    """Round a threshold estimate onto the grid given by snap/snap_rel.

    The estimate a previous run persisted is a fitted value with arbitrary
    digits (e.g. 0.13418...), so re-running the refine sweep would probe a
    slightly different window every time and never reuse cached sinter rows.
    Normalising the centre first makes the window reproducible across runs.

    Prefer `snap_rel` over an absolute `snap`: the refine window is itself
    relative (centre * (1 +/- refine_frac)), so a fixed absolute grid is too
    coarse for small thresholds -- snapping can then move the centre further
    than the window's own half-width and leave the true crossing outside the
    swept range entirely. A relative grid keeps the displacement a fixed
    fraction of the centre: at snap_rel <= refine_frac the centre moves by at
    most half the half-width, so the crossing always stays inside.

    Snapping is skipped -- centre returned unchanged -- if no grid is given,
    or if it would collapse the centre to zero.
    """
    if center is None or center <= 0:
        return center
    grid = snap_grid(center, snap, snap_rel)
    if not grid:
        return center
    snapped = round(center / grid) * grid
    if snapped <= 0:
        print(f"Refine: centre {center:.6g} is below half of snap={grid:g}; "
              f"using the unsnapped value")
        return center
    return round(snapped, 12)


def refine_window_probs(center: float | None, frac: float, n_points: int,
                        lo: float = 0.0, hi: float = 1.0,
                        snap: float | None = None,
                        snap_rel: float | None = None) -> list[float]:
    """n_points evenly spaced probs across center*(1 +/- frac), clipped to [lo, hi].

    Used by run_correctable_regime's refine-around-known-threshold mode to build
    a tight per-curve sweep window centred on a previously computed threshold.
    `snap`/`snap_rel` first normalise that centre onto a grid (see snap_center).
    Endpoints inclusive; values rounded to kill float dust and de-duplicated so
    a window clipped at a bound doesn't emit repeated probabilities. Returns []
    if center is None/<=0 (no known threshold to refine around).
    """
    if not center or center <= 0:
        return []
    center = snap_center(center, snap, snap_rel)
    half = frac * center
    a, b = max(lo, center - half), min(hi, center + half)
    if n_points == 1 or b <= a:
        return [round(center, 9)]
    step = (b - a) / (n_points - 1)
    return sorted({round(a + i * step, 9) for i in range(n_points)})


def canonical_prob(prob: float, *, digits: int = 10) -> float:
    """Stable probability value for task metadata / cache-key reuse."""
    p = round(float(prob), digits)
    return 0.0 if p == 0.0 else p


def sweep_probs(sweep: dict) -> list[float]:
    """Sweep probabilities with float dust removed for stable strong_ids."""
    if 'min' not in sweep:
        return [canonical_prob(p) for p in sweep['probs']]
    lo = float(sweep['min'])
    hi = float(sweep['max'])
    step = float(sweep['step'])
    if step <= 0:
        raise ValueError(f"sweep.step must be > 0, got {step!r}")
    raw = np.arange(lo, hi + step, step)
    eps = abs(step) * 1e-9
    out: list[float] = []
    for p in raw:
        q = canonical_prob(p)
        if q < lo - eps or q > hi + eps:
            continue
        if not out or q != out[-1]:
            out.append(q)
    if not out:
        return [canonical_prob(lo)]
    return out


def boosted_failure_probability(n: int) -> float:
    """Failure probability of a fusion boosted with 2**n - 2 ancillas (Grice 2011)."""
    return 2.0 ** (-n)


def grice_erasure_probability(p_loss: float, p_fail: float) -> float:
    """Effective erasure probability of a boosted fusion (Bartolucci et al. 2023)."""
    return 1.0 - (1.0 - p_fail / 2.0) * (1.0 - p_loss) ** (1.0 / p_fail)


def boosted_erasure_probability(p_loss: float, n: int) -> float:
    """Effective erasure probability at boosting level n, given photon loss p_loss."""
    return grice_erasure_probability(p_loss, boosted_failure_probability(n))


def noise_model_from_config(noise_channels: dict, p : float) -> dict[tuple[str, str], list[NoiseSpec]]:
    noise_model: dict[Any, list[NoiseSpec]] = {}
    for channel in noise_channels:
        tags: tuple[Any, ...] = tuple[Any, ...](channel['tag'])
        kind: str = channel['kind']
        scale: float = channel.get('scale', None)
        probability: float = channel.get('probability', None)
        if scale is not None:
            probability: float = scale * p if p is not None else None
        elif probability == 'p_e':
            probability = boosted_erasure_probability(p_loss=p, n=channel['n_boost'])
        # A tag may appear on multiple channels (e.g. DEPOLARIZE2 + HERALDED_ERASE
        # both on 'bell_prep_two_qubit'); accumulate rather than overwrite so
        # apply_tagged_noise_model applies all of them, in config order.
        noise_model.setdefault(tags, []).append(NoiseSpec(instruction=kind, probability=probability))
    return noise_model

def task_strong_id(task: Task) -> str:
    """Mirrors sinter's internal DEM fill-in so strong_id matches collected CSV rows."""
    if task.detector_error_model is None:
        circuit = task.circuit
        try:
            dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        except ValueError:
            try:
                dem = circuit.detector_error_model(approximate_disjoint_errors=True)
            except ValueError:
                dem = circuit.detector_error_model(approximate_disjoint_errors=True, flatten_loops=True)
        task.detector_error_model = dem
    return task.strong_id()


_csv_stats_cache: dict[str, dict[str, sinter.TaskStats]] = {}


def csv_stats_index(csv_path: str) -> dict[str, sinter.TaskStats]:
    """Cached strong_id -> aggregated TaskStats index for a results CSV.

    Result CSVs accumulate rows for a whole sweep, so re-parsing them on
    every (distance, prob) iteration is quadratic in sweep size. Each file
    is parsed from disk at most once per process; after a sinter.collect
    call the index is refreshed in memory from collect's return value
    instead of re-reading the file.
    """
    if csv_path not in _csv_stats_cache:
        index: dict[str, sinter.TaskStats] = {}
        if Path(csv_path).exists():
            for stat in sinter.read_stats_from_csv_files(csv_path):
                index[stat.strong_id] = stat
        _csv_stats_cache[csv_path] = index
    return _csv_stats_cache[csv_path]


def extract_local_prob(noise_channels: list[dict]) -> float:
    """Fixed local-noise level used to key per-run result files.

    A channel is swept by the probs loop (see noise_model_from_config) if it
    has a 'scale' key or an unresolved 'p_e' placeholder; anything else
    already holds a concrete, prob-independent probability regardless of
    which tag it's on (e.g. 'graph' for crg_prep_* configs, not just
    'circuit'). Defaults to 0.0 if every channel is swept, e.g. single-channel
    pure-erasure configs.
    """
    for channel in noise_channels:
        if channel.get('scale') is not None:
            continue
        prob = channel.get('probability', None)
        if isinstance(prob, str):
            continue
        if prob is not None:
            return prob
    return 0.0


def local_prob_label(noise_channels: list[dict]) -> str:
    """Filename prefix for the fixed local-noise value.

    'pr' for prep-style configs whose first noise channel is tagged 'graph'
    (e.g. crg_prep_*), 'pl' otherwise (circuit-tagged local noise).
    """
    return 'pr' if noise_channels[0]['tag'][0] == 'graph' else 'pl'


# Sweep keys that used to select a circuit variant; each variant is now its own
# builder (see dizx.builders), so a config still setting one is stale.
REMOVED_SWEEP_KEYS = ('max_ancillas_per_resource', 'unfuse_time_edges', 'interface')


def check_builder_options(config: dict) -> None:
    stale = [k for k in REMOVED_SWEEP_KEYS if k in config.get('sweep', {})]
    if stale:
        raise ValueError(f"sweep keys {stale} are gone: pick the builder variant instead "
                         f"(e.g. surgery_hybrid_linear_5chain, surgery_bell_prep_fusion_zigzag)")


def config_prefix(config: dict) -> str:
    """Short identity of a config: legacy resource_state + noise_model_name, or
    family + run type (see dizx.result_names)."""
    if not result_names.is_legacy(config):
        return f"{result_names.family(config)}_{result_names.run_type(config)}"
    return f"{config['resource_state']}_{config['noise_model_name']}"


DEFAULT_RESULTS_DIR = 'results'


def results_dir(config: dict) -> str:
    """Root directory (relative to the repo root) for this config's output.

    Sinter CSVs go in <results_dir>/sinter/, threshold plots in
    <results_dir>/plots/, and the aggregate correctable-regime PNG/CSV at the
    top of it. Configs may override the top-level `results_dir` key to keep a
    re-run separate from the original tree -- e.g. `results/scale` for the
    higher-distance refine sweeps, which deliberately re-collect the same
    (resource_state, noise model, builder) identities as `results/`.
    """
    return str(config.get('results_dir', DEFAULT_RESULTS_DIR)).rstrip('/')


def decoder_variant_suffix(sinter_args: dict) -> str:
    """Filename suffix for decoder settings that change the predictions.

    `decoder` alone does not identify what the decoder actually did:
    PyMatchingHeraldedDecoder run with correlated matching is a different
    decoder from the uncorrelated one -- lower LER, different predictions --
    but both report `decoder: pymatching_heralded`. Without this suffix the two
    write into the same sinter CSV and the same aggregate, and a threshold fit
    would be taken over a mixture of the two.

    Empty for every mwpf config and for the pymatching defaults, so existing
    result paths are unchanged.

    Kept in sync with `_regime_common.decoder_variant_suffix`, which repeats the
    convention for the plotting/backfill scripts.
    """
    parts = []
    if sinter_args.get('pm_decompose'):
        parts.append('dec')
    if sinter_args.get('pm_enable_correlations'):
        parts.append('corr')
    return ('_' + '_'.join(parts)) if parts else ''


LEGACY_MERGE_TYPES = ['XX']
LEGACY_OBSERVABLES = ['XX']


def sweep_variant_suffix(config: dict) -> str:
    """Filename suffix for the merge/observable axes of a config's sweep.

    The per-(local_prob, merge, observable) sinter CSVs have always carried
    merge_type and observable in their names (see `csv_path_for`), but the
    aggregates built from them did not -- so two configs differing only in
    `sweep.merge_types` / `sweep.observables` wrote the *same*
    correctable_regime_<id>.csv/.png and _refine_state.json, and the second run
    silently overwrote the first's thresholds even though its shots lived in
    separate files. Folding the two axes into the identifier keeps them apart.

    Empty for the legacy XX/XX sweep every already-collected aggregate under
    results/ was produced with, so those paths are unchanged -- same reasoning
    as `decoder_variant_suffix`. Multi-valued axes are joined with '-', since
    such an aggregate really is taken over that whole set.

    Kept in sync with `_regime_common.sweep_variant_suffix`, which repeats the
    convention for the plotting/backfill scripts.
    """
    sweep = config.get('sweep', {})
    merge_types = sweep.get('merge_types', [None])
    observables = sweep.get('observables', [None])
    if merge_types == LEGACY_MERGE_TYPES and observables == LEGACY_OBSERVABLES:
        return ''
    merges = '-'.join(str(m) for m in merge_types)
    obs = '-'.join(str(o) for o in observables)
    return f'_merge{merges}_observable{obs}'


def mwpf_naming(sinter_args: dict) -> tuple:
    """The mwpf decoder settings as they appear in per-run sinter CSV names.

    Both default to '' so a non-mwpf config (e.g. any pymatching one, which
    never sets either key) names its files the same way whether they are being
    written by `_run_threshold` or looked up again by `load_threshold_stats`.
    The two used to carry different defaults, so every pymatching resume read
    back zero stats from a file it had just written.
    """
    return (sinter_args.get('mwpf_decoder_type', ''),
            sinter_args.get('mwpf_cluster_node_limit', ''))


def aggregate_identifier(config: dict) -> str:
    """Prefix for aggregate plots (correctable-regime / photon-loss-threshold).

    config_prefix alone (resource_state + noise_model_name) can collide across
    configs that intentionally share cached per-run data but use a different
    builder or decoder (e.g. crg_prep_6ring vs crg_prep_fusion both being
    6ring_prep_and_erasure), or only in the merge type / observable being
    swept. Folding those in keeps aggregate plots distinct.

    A config that sets neither mwpf key isn't using mwpf at all (every
    pymatching config), so the mwpf segment is dropped rather than filled in
    with a meaningless solver name. A config that sets *either* key keeps the
    legacy defaults for the other, so every mwpf name already on disk stays
    byte-identical -- only the pymatching aggregates change, and
    scripts/rename_aggregate_mwpf_suffix.py migrated the existing ones.

    Non-legacy configs use `result_names.aggregate_stem` instead.
    """
    if not result_names.is_legacy(config):
        return result_names.aggregate_stem(config)
    sinter_args = config['sinter']
    decoder_name = sinter_args['decoder']
    suffix = f"{decoder_variant_suffix(sinter_args)}{sweep_variant_suffix(config)}"
    stem = f"{config_prefix(config)}_{config['builder']['name']}_{decoder_name}"
    if 'mwpf_decoder_type' not in sinter_args and 'mwpf_cluster_node_limit' not in sinter_args:
        return f"{stem}{suffix}"
    mwpf_decoder_type = sinter_args.get('mwpf_decoder_type', 'SolverSerialJointSingleHair')
    mwpf_cluster_node_limit = sinter_args.get('mwpf_cluster_node_limit', 50)
    return f"{stem}_{mwpf_decoder_type}_{mwpf_cluster_node_limit}{suffix}"


def csv_path_for(name: str, local_prob: float | None, builder_name: str, merge_type: str | None,
                  observable: str | None, decoder_name: str, mwpf_decoder_type: str,
                  mwpf_cluster_node_limit: int, prefix: str = 'pl',
                  *, out_dir: str = DEFAULT_RESULTS_DIR, variant: str = '') -> str:
    """Per-(local_prob, merge, observable) sinter resume CSV.

    `variant` comes from `decoder_variant_suffix`, so a correlated run never
    shares a file with the uncorrelated one.
    """
    return (f'{out_dir}/sinter/{name}_{prefix}{local_prob}_{builder_name}_merge{merge_type}'
            f'_observable{observable}_{decoder_name}_{mwpf_decoder_type}'
            f'_{mwpf_cluster_node_limit}{variant}.csv')


def run_stem(config: dict, noise_channels: list[dict], merge_types: list, observables: list) -> str:
    """Stem shared by a run's sinter CSV (<results_dir>/sinter/<stem>.csv) and its
    threshold plot (<results_dir>/plots/<stem>.png).

    `noise_channels` are the channels actually run, i.e. with any local_prob
    already substituted. Several merge types / observables only occur for the
    combined plot of `plot_threshold_curves`; a CSV always has one of each.
    Legacy configs name the combined plot merge{None}_observable{None}.
    """
    if not result_names.is_legacy(config):
        return result_names.sinter_stem(config, noise_channels, merge_types, observables)
    sinter_args = config['sinter']
    mwpf_decoder_type, mwpf_cluster_node_limit = mwpf_naming(sinter_args)
    merge_type = merge_types[0] if len(merge_types) == 1 else None
    observable = observables[0] if len(observables) == 1 else None
    path = csv_path_for(config_prefix(config), extract_local_prob(noise_channels),
                        config['builder']['name'], merge_type, observable, sinter_args['decoder'],
                        mwpf_decoder_type, mwpf_cluster_node_limit,
                        local_prob_label(noise_channels), out_dir='',
                        variant=decoder_variant_suffix(sinter_args))
    return Path(path).stem


def sinter_csv_path(config: dict, noise_channels: list[dict], merge_type, observable,
                    *, out_dir: str) -> str:
    return f'{out_dir}/sinter/{run_stem(config, noise_channels, [merge_type], [observable])}.csv'


def plot_threshold_curves(results: list[sinter.TaskStats], stem: str, builder_name: str,
                           decoder_name: str, local_prob: float | None, prefix: str = 'pl',
                           *, out_dir: str = DEFAULT_RESULTS_DIR) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharey=True)
    sinter.plot_error_rate(
        ax=ax,
        stats=results,
        x_func=lambda stat: stat.json_metadata['p'],
        group_func=lambda stat: f"d={stat.json_metadata['distance']}, merge={stat.json_metadata['merge_type']}, observable={stat.json_metadata['observable']}, {prefix}={local_prob}",
        failure_units_per_shot_func=lambda stat: 1,
    )

    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('Logical error rate')
    ax.set_title(decoder_name)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    fig.suptitle(f'Rotated surface code - {builder_name}', fontsize=13)
    fig.tight_layout()
    Path(f'{out_dir}/plots/').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{out_dir}/plots/{stem}.png')
    plt.close(fig)


def save_threshold_plot(stats: list[sinter.TaskStats], stem: str, prefix: str, local_prob: float | None,
                        builder_name: str, decoder_name: str,
                        *, out_dir: str = DEFAULT_RESULTS_DIR) -> None:
    """Regenerate the per-(merge, observable) threshold plot from a full stats set.

    Called with everything in the run's CSV (via csv_stats_index), not just
    the points visited by the current call, so plots always show all
    distances/probs collected so far — bisection calls in particular only
    visit a single prob each.
    """
    if not stats:
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 5), sharey=True)
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata['p'],
        group_func=lambda stat: f"d={stat.json_metadata['distance']}, merge={stat.json_metadata['merge_type']}, observable={stat.json_metadata['observable']}, {prefix}={local_prob}",
        failure_units_per_shot_func=lambda stat: 1,
    )
    ax.set_xlabel('Physical error rate')
    ax.set_ylabel('Logical error rate')
    ax.set_title(decoder_name)
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)
    fig.suptitle(f'Rotated surface code - {builder_name}', fontsize=13)
    fig.tight_layout()
    Path(f'{out_dir}/plots/').mkdir(parents=True, exist_ok=True)
    plt.savefig(f'{out_dir}/plots/{stem}.png')
    plt.close(fig)


def load_threshold_stats(config: dict) -> list[sinter.TaskStats]:
    """Read previously collected stats from CSV without running any new collection."""
    builder_name = config['builder']['name']
    noise_channels = config['noise_model']
    sweep = config['sweep']
    merge_types = sweep.get('merge_types', [None])
    observables = sweep.get('observables', [None])
    decoder_name = config['sinter']['decoder']
    local_prob = extract_local_prob(noise_channels)
    prefix = local_prob_label(noise_channels)

    out_dir = results_dir(config)

    csv_paths = {
        sinter_csv_path(config, noise_channels, merge_type, observable, out_dir=out_dir)
        for merge_type in merge_types
        for observable in observables
    }

    results: list[sinter.TaskStats] = []
    for csv_path in csv_paths:
        if Path(csv_path).exists():
            results.extend(sinter.read_stats_from_csv_files(csv_path))

    # plot_threshold_curves combines every (merge_type, observable) pair into
    # one plot -- with only one pair swept (the common case) that's identical
    # to the single save_threshold_plot output already written during
    # collection, just under a misleading merge{None}_observable{None} name.
    # Only worth the extra file when there's actually more than one pair to
    # combine.
    if results and len(merge_types) * len(observables) > 1:
        plot_threshold_curves(results, run_stem(config, noise_channels, merge_types, observables),
                              builder_name, decoder_name, local_prob, prefix, out_dir=out_dir)
    return results


def run_threshold(config, *, persist: bool = True):
    """Collects threshold stats for every (distance, merge_type, observable, p).

    With persist=False the shots are treated as disposable: sinter resumes
    from (and appends to) a throwaway CSV in a temporary directory that is
    deleted on return, and no threshold plot is written. Used by the
    seed-bisection pass in run_correctable_regime, whose only output is a
    coarse threshold estimate -- its points would otherwise land in the same
    sinter CSV and plot as the refine window that follows it.
    """
    if persist:
        Path(results_dir(config), 'sinter').mkdir(parents=True, exist_ok=True)
        return _run_threshold(config, out_dir=results_dir(config), plot=True)
    with tempfile.TemporaryDirectory(prefix='threshold_scratch_') as tmp_dir:
        Path(tmp_dir, 'sinter').mkdir(parents=True, exist_ok=True)
        return _run_threshold(config, out_dir=tmp_dir, plot=False)


def _run_threshold(config, *, out_dir: str, plot: bool):
    name = config_prefix(config)
    builder = config['builder']
    builder_name = builder['name']
    noise_channels = config['noise_model']
    sweep = config['sweep']
    distances = sweep['distances']
    merge_types = sweep.get('merge_types', [None])
    observables = sweep.get('observables', [None])
    check_builder_options(config)
    probs: list[float] = sweep_probs(sweep)
    sinter_args = config['sinter']
    decoder_name = sinter_args['decoder']
    max_shots = sinter_args['max_shots']
    max_errors: int = sinter_args['max_errors']
    _nw = sinter_args.get('num_workers', 1)
    num_workers = os.cpu_count() if _nw == 'max' else int(_nw)
    if decoder_name not in ('pymatching', 'pymatching_heralded'):
        raise ValueError(f"decoder {decoder_name!r}: dizx runs pymatching / pymatching_heralded only")
    # PyMatchingHeraldedDecoder options. `pm_enable_correlations` needs a
    # pymatching whose decode_batch accepts `edge_reweights` (see
    # dizx.sinter_decoders.SUPPORTS_EDGE_REWEIGHTS); without it correlations
    # fall back to a per-shot DEM rebuild that is ~40x slower.
    pm_decompose = bool(sinter_args.get('pm_decompose', False))
    pm_enable_correlations = bool(sinter_args.get('pm_enable_correlations', False))
    if pm_enable_correlations and not pm_decompose:
        raise ValueError(
            "pm_enable_correlations requires pm_decompose: pymatching rejects "
            "undecomposed hyperedges when correlations are enabled."
        )
    results : list[sinter.TaskStats] = []
    local_prob = extract_local_prob(noise_channels)
    prefix = local_prob_label(noise_channels)
    print("name:", name)
    print("builder_name:", builder_name)
    if result_names.is_legacy(config):
        print("noise model name:", config['noise_model_name'])
    print("resource state:", config['resource_state'])
    print("local_prob:", local_prob)

    for distance in distances:
        for merge_type in merge_types:
            for observable in observables:
                circuit: Circuit = cached_circuit(builder_name, distance, observable, merge_type)
                print(f"Running distance={distance}, merge_type={merge_type}, observable={observable}")
                # Correlated / decomposed runs are a different decoder: the
                # name keeps their CSVs and plots separate so a threshold fit
                # is never taken over a mixture.
                stem = run_stem(config, noise_channels, [merge_type], [observable])
                csv_path = f'{out_dir}/sinter/{stem}.csv'

                for prob in probs:
                    prob = canonical_prob(prob)
                    print(f"Running with noise probability p={prob}")
                    noise_model = noise_model_from_config(noise_channels=noise_channels, p=prob)
                    noisy_circuit: Circuit = apply_tagged_noise_model(circuit, noise_by_type=noise_model)
                    print("applied noise")

                    task_metadata = {
                        'builder_name': builder_name,
                        'distance': distance,
                        'p': prob,
                        'merge_type': merge_type,
                        'observable': observable,
                        'noise_model': noise_channels,
                        'pl': local_prob,
                    }
                    # Decoder settings that change the predictions must be part
                    # of the task identity, or a correlated run would resume
                    # from (and be merged with) an uncorrelated one's rows.
                    # Only recorded when non-default, so existing collected
                    # results keep their strong_id.
                    if pm_decompose:
                        task_metadata['pm_decompose'] = True
                    if pm_enable_correlations:
                        task_metadata['pm_corr'] = True

                    task: Task = Task(
                        circuit=noisy_circuit,
                        decoder = decoder_name,
                        json_metadata=task_metadata,
                    )
                    
                    strong_id = task_strong_id(task)
                    index = csv_stats_index(csv_path)
                    existing = index.get(strong_id)
                    existing_shots = existing.shots if existing is not None else 0
                    existing_errors = existing.errors if existing is not None else 0
                    if existing_shots or existing_errors:
                        print(f"Found existing data: {existing_shots}/{max_shots} shots, {existing_errors}/{max_errors} errors")
                    if existing_shots >= max_shots or existing_errors >= max_errors:
                        print("Target already reached, skipping collection")
                        results.append(existing)
                        continue

                    print("Target not yet reached, running collection")
                    if decoder_name == 'pymatching_heralded':
                        custom_decoders = {
                            'pymatching_heralded' : PyMatchingHeraldedDecoder(
                                decompose=pm_decompose,
                                enable_correlations=pm_enable_correlations,
                            ).with_circuit(circuit=noisy_circuit)
                        }
                    else:
                        custom_decoders = {}
                    # collect returns aggregated stats for every task in the
                    # resume CSV, not just the one passed in: refresh the whole
                    # index from it, then pick out this task's stats.
                    collect_start = time.monotonic()
                    collected: list[sinter.TaskStats] = sinter.collect(tasks=[task],
                                                                    max_shots=max_shots,
                                                                    max_errors=max_errors,
                                                                    num_workers=num_workers,
                                                                    custom_decoders=custom_decoders,
                                                                    print_progress=False,
                                                                    save_resume_filepath=csv_path)
                    collect_seconds = time.monotonic() - collect_start
                    # collect appends one row per batch; fold them so the CSV
                    # stays one row per task. Never fatal: the rows are valid
                    # either way, just bulkier.
                    try:
                        compact_sinter_csv(csv_path)
                    except Exception as e:
                        print(f"Warning: could not compact {csv_path}: {e}")
                    for stat in collected:
                        index[stat.strong_id] = stat
                    task_stats = index[strong_id]
                    results.append(task_stats)
                    new_shots = task_stats.shots - existing_shots
                    if new_shots > 0:
                        print(f"Collection took {collect_seconds:.1f}s for {new_shots} shots "
                              f"({collect_seconds / new_shots * 1e3:.3f} ms/shot, "
                              f"{num_workers} workers)")
                    else:
                        print(f"Collection took {collect_seconds:.1f}s for 0 new shots")
                    ler = task_stats.errors / task_stats.shots if task_stats.shots > 0 else 0.0
                    print(f"Logical error rate for distance={distance}, merge_type={merge_type}, observable={observable}, p={prob}: {ler:.4e}")
                    if plot:
                        save_threshold_plot(list(index.values()), stem, prefix, local_prob, builder_name,
                                            decoder_name, out_dir=out_dir)

                # Regenerate once per pass so the plot reflects the full CSV
                # even when every point above was skipped as already collected.
                if plot:
                    save_threshold_plot(list(csv_stats_index(csv_path).values()), stem, prefix, local_prob,
                                        builder_name, decoder_name, out_dir=out_dir)

    return results


def _is_above_threshold(ler_by_distance: dict[int, float], distances: list[int]) -> bool | None:
    """Compares the smallest and largest swept distances at a single point.

    Below threshold, larger distances suppress errors more (lower LER); above
    threshold they amplify them. Returns None if data for either extremal
    distance is missing.
    """
    sorted_ds = sorted(distances)
    d_low, d_high = sorted_ds[0], sorted_ds[-1]
    if d_low not in ler_by_distance or d_high not in ler_by_distance:
        return None
    return ler_by_distance[d_high] > ler_by_distance[d_low]


_DEFAULT_TOLERANCE_MIN = 1e-4


def bisection_tolerance_at(sweep: dict, center: float) -> float:
    """Absolute bracket tolerance for a threshold sitting near `center`.

    `tolerance` is a fixed absolute width. `tolerance_rel` instead asks for a
    width that is a fraction of the threshold's own scale, so a threshold near
    0.3 and one near 0.05 are located to the same *relative* precision, and
    takes precedence when both are given.

    Relative is usually what you want here, for the reason `snap_center` gives
    for preferring snap_rel: the estimate is consumed as the centre of a refine
    window that is itself relative (centre * (1 +/- refine_frac)). A fixed
    absolute tolerance is therefore far tighter than needed at high thresholds
    and too loose at low ones -- bisection returns a bracket midpoint, so its
    error is tolerance/2 regardless of scale, and at threshold 0.05 with
    tolerance 0.01 that error equals the window's entire half-width, letting
    the crossing fall outside the swept range entirely.

    `tolerance_min` floors the relative width. A purely relative width shrinks
    just as fast as the bracket does, so a bracket collapsing toward zero would
    never meet it and the loop would not terminate.
    """
    tol_rel = sweep.get('tolerance_rel')
    if tol_rel:
        return max(float(tol_rel) * float(center),
                   float(sweep.get('tolerance_min', _DEFAULT_TOLERANCE_MIN)))
    return float(sweep['tolerance'])


def has_bisection_tolerance(sweep: dict) -> bool:
    """Whether `sweep` specifies a bisection stopping width at all."""
    return sweep.get('tolerance') is not None or sweep.get('tolerance_rel') is not None


def run_threshold_bisection(config: dict, *, persist: bool = True) -> tuple[float, list[sinter.TaskStats]]:
    """Locates a threshold via recursive bisection instead of a dense sweep.

    Expects sweep['min'], sweep['max'] and a stopping width -- either
    sweep['tolerance'] (absolute) or sweep['tolerance_rel'] (a fraction of the
    bracket midpoint, see bisection_tolerance_at) -- in place of
    sweep['probs']/'step'. At each candidate point, runs (and caches to CSV
    exactly as run_threshold does) all distances/merge_types/observables,
    then narrows [lo, hi] based on whether the largest distance's LER is
    above or below the smallest distance's LER.

    persist=False discards each point's shots after reading its LER (see
    run_threshold), so bisection contributes only its threshold estimate --
    no rows in the sinter CSV, no points on the plot.
    """
    sweep = config['sweep']
    distances = sweep['distances']
    lo, hi = float(sweep['min']), float(sweep['max'])
    # Re-evaluated each pass: with tolerance_rel the target width tracks the
    # bracket midpoint, which moves as the bracket narrows onto the threshold.
    tol = bisection_tolerance_at(sweep, (lo + hi) / 2)

    all_stats: list[sinter.TaskStats] = []
    while hi - lo > tol:
        mid = (lo + hi) / 2
        point_config = copy.deepcopy(config)
        point_sweep = point_config['sweep']
        for key in ('min', 'max', 'step', 'tolerance', 'tolerance_rel',
                    'tolerance_min', 'bisection', 'probs'):
            point_sweep.pop(key, None)
        point_sweep['probs'] = [mid]

        stats = run_threshold(point_config, persist=persist)
        all_stats.extend(stats)

        ler_by_distance: dict[int, float] = {}
        for s in stats:
            md = s.json_metadata or {}
            d = int(md.get('distance', -1))
            shots = s.shots - s.discards
            if shots > 0:
                ler_by_distance[d] = s.errors / shots

        above = _is_above_threshold(ler_by_distance, distances)
        if above is None:
            raise RuntimeError(
                f"Insufficient data at p={mid:g} to determine threshold direction "
                f"(need shots for distances {min(distances)} and {max(distances)})"
            )
        print(f"Bisection: p={mid:.6g} is {'above' if above else 'below'} threshold (bracket [{lo:.6g}, {hi:.6g}])")

        if above:
            hi = mid
        else:
            lo = mid
        tol = bisection_tolerance_at(sweep, (lo + hi) / 2)

    threshold = (lo + hi) / 2
    tol_rel = sweep.get('tolerance_rel')
    how = f"tolerance_rel={tol_rel:g} -> width {tol:.3g}" if tol_rel else f"tolerance={tol:.3g}"
    print(f"Bisection threshold estimate: {threshold:.6g} ({how})")
    return threshold, all_stats


def _probe_point(config: dict, prob: float, distances: list[int],
                 *, persist: bool = True) -> tuple[bool | None, list[sinter.TaskStats]]:
    """Collect a single probability at all distances and report above/below.

    Returns (above, stats): above is True when the largest distance's LER
    exceeds the smallest's (above threshold), False when below, None when
    either extremal distance has no shots. persist=True saves the point to the
    sinter CSV exactly like a normal window point, so a probe that later lands
    inside the swept window (or a re-probed abutting edge) is reused rather than
    recollected.
    """
    point_config = copy.deepcopy(config)
    ps = point_config['sweep']
    for key in ('min', 'max', 'step', 'tolerance', 'bisection',
                'refine_step', 'refine_pad', 'probs'):
        ps.pop(key, None)
    ps['probs'] = [prob]
    stats = run_threshold(point_config, persist=persist)
    ler_by_distance: dict[int, float] = {}
    for s in stats:
        md = s.json_metadata or {}
        d = int(md.get('distance', -1))
        shots = s.shots - s.discards
        if shots > 0:
            ler_by_distance[d] = s.errors / shots
    return _is_above_threshold(ler_by_distance, distances), stats


def slide_refine_window(config: dict, center: float, frac: float, n_points: int,
                        lo: float, hi: float, distances: list[int],
                        *, max_slides: int = 20) -> tuple[float | None, list[float], list[sinter.TaskStats]]:
    """Slide a fixed-relative-width window until it brackets the threshold.

    Cheaper alternative to a seed bisection for centring the refine window.
    Each step probes only the window's edges: if the first (lowest) point is
    above threshold the whole window is too high, so that edge becomes the next
    window's upper bound (slide down); if the first is below but the last is
    still below, the last edge becomes the next lower bound (slide up). When the
    first point is below and the last above, the threshold lies inside and the
    window is returned to sweep. Successive windows abut at the shared edge, so
    that edge -- already collected and persisted -- is reused, not recollected.

    Returns (center, probs, stats) for the bracketing window, or (None, [],
    stats) if the search leaves [lo, hi] or exhausts max_slides without
    bracketing (threshold likely outside the configured range). The returned
    window is deliberately not snapped: the threshold then always lies within
    +/-frac of centre, i.e. inside the swept window, so the refinement can be
    marked converged (found_in_region) instead of looping.
    """
    all_stats: list[sinter.TaskStats] = []
    for _ in range(max_slides):
        probs = refine_window_probs(center, frac, n_points, lo, hi)
        if len(probs) < 2:
            return None, [], all_stats
        first_above, s = _probe_point(config, probs[0], distances)
        all_stats += s
        if first_above is None:
            print(f"Slide: indeterminate direction at p={probs[0]:.6g}; aborting")
            return None, [], all_stats
        if first_above:
            new_top = probs[0]
            center = new_top / (1.0 + frac)
            print(f"Slide: window [{probs[0]:.6g}, {probs[-1]:.6g}] entirely above "
                  f"threshold -> shift down (new top {new_top:.6g})")
            if center * (1.0 - frac) < lo:
                print(f"Slide: shifted below min={lo:g}; threshold likely < {lo:g}")
                return None, [], all_stats
            continue
        last_above, s = _probe_point(config, probs[-1], distances)
        all_stats += s
        if last_above is None:
            print(f"Slide: indeterminate direction at p={probs[-1]:.6g}; aborting")
            return None, [], all_stats
        if not last_above:
            new_bot = probs[-1]
            center = new_bot / (1.0 - frac)
            print(f"Slide: window [{probs[0]:.6g}, {probs[-1]:.6g}] entirely below "
                  f"threshold -> shift up (new bottom {new_bot:.6g})")
            if center * (1.0 + frac) > hi:
                print(f"Slide: shifted above max={hi:g}; threshold likely > {hi:g}")
                return None, [], all_stats
            continue
        print(f"Slide: threshold bracketed in [{probs[0]:.6g}, {probs[-1]:.6g}] "
              f"(first below, last above) -> sweeping")
        return center, probs, all_stats
    print(f"Slide: no bracket after {max_slides} slides; aborting")
    return None, [], all_stats


def run_threshold_bracket_sweep(config: dict) -> tuple[float, list[sinter.TaskStats]]:
    """Coarse bisection to localize the crossing, then a dense sweep across it.

    Expects the bisection keys (sweep['min'], sweep['max'], sweep['tolerance'])
    plus sweep['refine_step']. The (coarse) tolerance only needs to bracket the
    crossing; the dense sweep then covers bracket centre +/- refine_pad
    (default 2*tolerance, clipped to [min, max]) in steps of refine_step.
    Returns the bisection centre and the combined bisection + sweep stats;
    estimating the threshold from the stats is left to the caller.
    """
    sweep = config['sweep']
    step = float(sweep['refine_step'])
    sweep_min = float(sweep['min'])
    sweep_max = float(sweep['max'])

    bisection_config = copy.deepcopy(config)
    for key in ('refine_step', 'refine_pad'):
        bisection_config['sweep'].pop(key, None)
    center, stats = run_threshold_bisection(bisection_config)

    # Derived after bisection, not before: with tolerance_rel the tolerance is
    # a fraction of the threshold's own scale, so it is only meaningful once
    # there is a centre to evaluate it at.
    pad = float(sweep.get('refine_pad', 2 * bisection_tolerance_at(sweep, center)))

    # Bisection's centre is stochastic (it depends on sampled shot outcomes),
    # so re-running would otherwise probe a slightly different window each
    # time and defeat the CSV resume cache, which is keyed on exact
    # probability. Snap to a fixed grid anchored at sweep['min'] with spacing
    # refine_step: runs whose centre lands in the same grid cell then probe
    # bit-identical absolute probabilities and reuse prior results instead of
    # duplicating them.
    n_max = round((sweep_max - sweep_min) / step)
    n_center = min(max(round((center - sweep_min) / step), 0), n_max)
    n_pad = round(pad / step)
    n_lo = max(0, n_center - n_pad)
    n_hi = min(n_max, n_center + n_pad)
    probs = [canonical_prob(sweep_min + step * n) for n in range(n_lo, n_hi + 1)]
    grid_center = canonical_prob(sweep_min + step * n_center)
    print(f"Bracket sweep: {len(probs)} points in [{probs[0]:.6g}, {probs[-1]:.6g}] "
          f"(step={step:g}) around {center:.6g} (grid centre {grid_center:.6g})")

    refine_config = copy.deepcopy(config)
    refine_sweep = refine_config['sweep']
    for key in ('min', 'max', 'step', 'tolerance', 'tolerance_rel', 'tolerance_min',
                'bisection', 'refine_step', 'refine_pad', 'probs'):
        refine_sweep.pop(key, None)
    refine_sweep['probs'] = probs
    stats = stats + run_threshold(refine_config)
    return center, stats


def fit_threshold(stats: list[sinter.TaskStats], distances: list[int],
                  p_center: float | None = None,
                  p_halfwidth: float | None = None) -> tuple[float, float] | None:
    """Finite-size scaling fit of the threshold (Wang-Harrington-Preskill).

    Fits LER = A + B*x + C*x^2 with x = (p - p_th) * d**(1/nu) jointly over
    every (distance, p) point, weighted by binomial error bars, so all shots
    at all distances constrain p_th together. When p_center/p_halfwidth are
    given, only points inside that window enter the fit (the scaling ansatz
    is only valid near the crossing). Returns (p_th, stderr), or None when
    there is not enough data or the fit fails to converge.
    """
    from scipy.optimize import curve_fit

    # Aggregate shots/errors per (distance, p); duplicate strong_ids (e.g.
    # a bisection midpoint re-loaded by the refine sweep) count once.
    agg: dict[tuple[int, float], tuple[int, int]] = {}
    seen: set[str] = set()
    for s in stats:
        if s.strong_id in seen:
            continue
        seen.add(s.strong_id)
        md = s.json_metadata or {}
        d, p = int(md.get('distance', 0)), float(md.get('p', 0.0))
        if d not in distances:
            continue
        if p_center is not None and p_halfwidth is not None and abs(p - p_center) > p_halfwidth * (1 + 1e-9):
            continue
        shots = s.shots - s.discards
        if shots <= 0:
            continue
        errors, prev_shots = agg.get((d, p), (0, 0))
        agg[(d, p)] = (errors + s.errors, prev_shots + shots)

    if len(agg) < 6 or len({d for d, _ in agg}) < 2:
        return None

    ds = np.array([d for d, _ in agg], dtype=float)
    ps = np.array([p for _, p in agg], dtype=float)
    errors = np.array([agg[k][0] for k in agg], dtype=float)
    shots = np.array([agg[k][1] for k in agg], dtype=float)
    ler = errors / shots
    # Binomial error bars; floor at 1 observed error so zero-error points get
    # finite weight instead of dominating the fit.
    sigma = np.sqrt(np.maximum(errors, 1.0)) / shots

    def model(X, p_th, nu, a, b, c):
        d, p = X
        x = (p - p_th) * d ** (1.0 / nu)
        return a + b * x + c * x * x

    span = ps.max() - ps.min()
    p0 = [p_center if p_center is not None else float(np.median(ps)),
          1.0, float(np.mean(ler)), 1.0, 0.0]
    bounds = ([ps.min() - span, 0.3, 0.0, -np.inf, -np.inf],
              [ps.max() + span, 5.0, 1.0, np.inf, np.inf])
    try:
        popt, pcov = curve_fit(model, (ds, ps), ler, p0=p0, sigma=sigma,
                               absolute_sigma=True, bounds=bounds, maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    p_th, nu = float(popt[0]), float(popt[1])
    # A fit pinned at (or essentially at) a bound didn't converge to an
    # interior optimum -- it ran off to the edge of the search space instead,
    # typically because the (d, p) points span far more than the region the
    # WHP scaling ansatz is valid in (e.g. stale wide-sweep points cached
    # alongside a narrow bracket-sweep in the same CSV). Treat that the same
    # as a failed fit rather than reporting a physically meaningless
    # threshold (seen in practice as p_th landing exactly on ps.max()+span,
    # or nu pinned at its 5.0 ceiling).
    (p_lo, nu_lo, *_), (p_hi, nu_hi, *_) = bounds
    eps = 1e-6
    if not (p_lo + eps * (p_hi - p_lo) < p_th < p_hi - eps * (p_hi - p_lo)):
        print(f"Scaling fit rejected: p_th = {p_th:.6g} is pinned at its search bound "
              f"[{p_lo:.6g}, {p_hi:.6g}]")
        return None
    if not (nu_lo + eps * (nu_hi - nu_lo) < nu < nu_hi - eps * (nu_hi - nu_lo)):
        print(f"Scaling fit rejected: nu = {nu:.3g} is pinned at its search bound "
              f"[{nu_lo:.6g}, {nu_hi:.6g}]")
        return None
    if not (0.0 <= p_th <= 1.0):
        print(f"Scaling fit rejected: p_th = {p_th:.6g} is not a valid probability")
        return None
    stderr = float(np.sqrt(pcov[0, 0])) if np.isfinite(pcov[0, 0]) else float('nan')
    print(f"Scaling fit: p_th = {p_th:.6g} +/- {stderr:.2g} (nu = {nu:.3g}, "
          f"{len(agg)} points, {len({d for d, _ in agg})} distances)")
    return p_th, stderr


if __name__ == "__main__":
    with open(file=sys.argv[1]) as f:
        config = yaml.safe_load(stream=f)
    sweep = config['sweep']
    if sweep.get('bisection'):
        if sweep.get('refine_step'):
            threshold, result = run_threshold_bracket_sweep(config=config)
            if sweep.get('estimator') == 'fit':
                pad = float(sweep.get('refine_pad', 2 * bisection_tolerance_at(sweep, threshold)))
                fit_threshold(result, sweep['distances'], p_center=threshold, p_halfwidth=pad)
        else:
            threshold, result = run_threshold_bisection(config=config)
    else:
        result = run_threshold(config=config)
        if sweep.get('estimator') == 'fit':
            fit_threshold(result, sweep['distances'])
    


                


