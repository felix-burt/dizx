"""Shared helpers for the correctable-regime runner and figures.

They read the per-`local_prob` sinter `.csv` files written by
`experiments/run_threshold_experiment.py` / `run_correctable_regime.py`. This
module centralises the path convention, CSV loading, and the
distance-crossing threshold estimate (mirrors
`experiments/run_correctable_regime.py::_estimate_threshold`).
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sinter

from dizx import result_names

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = _REPO_ROOT / "results" / "sinter"
DEFAULT_CRG_DATA_DIR = _REPO_ROOT / "results"


def results_root(config: dict[str, Any]) -> Path:
    """Repo-root-relative output tree for a config's results.

    Mirrors `run_threshold_experiment.results_dir`: configs may set a
    top-level `results_dir` (e.g. `results/scale`) to keep a re-run's sinter
    CSVs, plots and crg data separate from the default `results/` tree.
    """
    return _REPO_ROOT / str(config.get("results_dir", "results")).rstrip("/")


def local_prob_label(noise_channels: list[dict]) -> str:
    """Filename prefix for the fixed local-noise value.

    'pr' for prep-style configs whose first noise channel is tagged 'graph'
    (e.g. crg_prep_*, erasure_fusion), 'pl' otherwise (circuit-tagged local
    noise). Mirrors `run_threshold_experiment.local_prob_label`.
    """
    return "pr" if noise_channels[0]["tag"][0] == "graph" else "pl"


def decoder_variant_suffix(sinter_args: dict[str, Any]) -> str:
    """Filename suffix for decoder settings that change the predictions.

    Mirrors `run_threshold_experiment.decoder_variant_suffix` -- see that
    docstring for why it exists. Empty for every mwpf config and the pymatching
    defaults, so existing paths are unchanged.
    """
    parts = []
    if sinter_args.get("pm_decompose"):
        parts.append("dec")
    if sinter_args.get("pm_enable_correlations"):
        parts.append("corr")
    return ("_" + "_".join(parts)) if parts else ""


def csv_path_for(config: dict[str, Any], local_prob: float, *, results_dir: Path) -> Path:
    """Path convention used by `run_threshold_experiment.run_stem`, for the
    config's first merge type / observable at `local_prob`."""
    sweep = config["sweep"]
    merge_type = sweep.get("merge_types", [None])[0]
    observable = sweep.get("observables", [None])[0]
    if not result_names.is_legacy(config):
        channels = result_names.resolve_local_prob(config["noise_model"], local_prob)
        return results_dir / (
            f"{result_names.sinter_stem(config, channels, [merge_type], [observable])}.csv")
    name = config_prefix(config)
    builder_name = config["builder"]["name"]
    sinter_args = config["sinter"]
    decoder_name = sinter_args["decoder"]
    # Defaults match what `run_threshold_experiment._run_threshold` passes when
    # WRITING these files -- empty, not the SolverSerialJointSingleHair/50 that
    # `aggregate_identifier` uses for aggregate names. A non-mwpf config (which
    # sets neither key) is written as `..._pymatching_heralded__.csv`; defaulting
    # to the aggregate's values here would look for a name nothing ever wrote.
    mwpf_decoder_type = sinter_args.get("mwpf_decoder_type", "")
    mwpf_cluster_node_limit = sinter_args.get("mwpf_cluster_node_limit", "")
    prefix = local_prob_label(config["noise_model"])
    return results_dir / (
        f"{name}_{prefix}{local_prob}_{builder_name}_merge{merge_type}"
        f"_observable{observable}_{decoder_name}_{mwpf_decoder_type}"
        f"_{mwpf_cluster_node_limit}{decoder_variant_suffix(sinter_args)}.csv"
    )


def flat_csv_path_for(config: dict[str, Any], *, results_dir: Path) -> Path:
    """CSV path for a "flat reference" config (no `sweep.local_probs`).

    Honours an explicit `csv_path` override (relative to the repo root) if
    present -- needed when the on-disk file was produced by a different
    collection script and so doesn't follow the plain `csv_path_for`
    convention.
    Otherwise falls back to `csv_path_for` at local_prob=0.0.
    """
    if "csv_path" in config:
        return _REPO_ROOT / config["csv_path"]
    return csv_path_for(config, 0.0, results_dir=results_dir)


def config_prefix(config: dict[str, Any]) -> str:
    """The `name` run_threshold.py prints and keys its result CSVs by.

    Mirrors `run_threshold_experiment.config_prefix`.
    """
    if not result_names.is_legacy(config):
        return f"{result_names.family(config)}_{result_names.run_type(config)}"
    return f"{config['resource_state']}_{config['noise_model_name']}"


LEGACY_MERGE_TYPES = ["XX"]
LEGACY_OBSERVABLES = ["XX"]


def sweep_variant_suffix(config: dict[str, Any]) -> str:
    """Filename suffix for the merge/observable axes of a config's sweep.

    Without it, configs differing only in `sweep.merge_types` /
    `sweep.observables` share one aggregate name and overwrite each other's
    thresholds. Empty for the legacy XX/XX sweep, so existing paths are
    unchanged. Mirrors `run_threshold_experiment.sweep_variant_suffix` -- see
    that docstring.
    """
    sweep = config.get("sweep", {})
    merge_types = sweep.get("merge_types", [None])
    observables = sweep.get("observables", [None])
    if merge_types == LEGACY_MERGE_TYPES and observables == LEGACY_OBSERVABLES:
        return ""
    merges = "-".join(str(m) for m in merge_types)
    obs = "-".join(str(o) for o in observables)
    return f"_merge{merges}_observable{obs}"


def aggregate_identifier(config: dict[str, Any]) -> str:
    """Filename identifier shared by a config's PNG and crg data file.

    Mirrors `run_threshold_experiment.aggregate_identifier`.
    """
    if not result_names.is_legacy(config):
        return result_names.aggregate_stem(config)
    name = config_prefix(config)
    builder_name = config["builder"]["name"]
    sinter_args = config["sinter"]
    decoder_name = sinter_args["decoder"]
    suffix = f"{decoder_variant_suffix(sinter_args)}{sweep_variant_suffix(config)}"
    stem = f"{name}_{builder_name}_{decoder_name}"
    if "mwpf_decoder_type" not in sinter_args and "mwpf_cluster_node_limit" not in sinter_args:
        return f"{stem}{suffix}"
    mwpf_decoder_type = sinter_args.get("mwpf_decoder_type", "SolverSerialJointSingleHair")
    mwpf_cluster_node_limit = sinter_args.get("mwpf_cluster_node_limit", 50)
    return f"{stem}_{mwpf_decoder_type}_{mwpf_cluster_node_limit}{suffix}"


def crg_stem(config: dict[str, Any]) -> str:
    """Stem of a config's correctable-regime aggregates (PNG, crg data CSV,
    refine-state JSON). Non-legacy aggregate stems already start with the run
    type (crg_local, crg_prep, ...), so only legacy ones get a prefix."""
    if not result_names.is_legacy(config):
        return aggregate_identifier(config)
    return f"correctable_regime_{aggregate_identifier(config)}"


def crg_data_path_for(config: dict[str, Any], *, data_dir: Path | None = None) -> Path:
    """Path to the persisted (local_prob, threshold, stderr, method) summary CSV.

    Written incrementally by `run_correctable_regime.py` alongside its PNG
    (`correctable_regime_<id>.png`) -- holds the accurate threshold estimate
    (fit_threshold's finite-size scaling fit, when `sweep.estimator: fit`) per
    local_prob, computed from the same points the bracket-sweep/bisection
    search actually collected, so plotting doesn't have to re-derive a cruder
    estimate from raw shots after the fact.

    Defaults to the config's own `results_dir` (see `results_root`) so a
    scale-up re-run reads and writes its thresholds in its own tree.
    """
    if data_dir is None:
        data_dir = results_root(config)
    return data_dir / f"{crg_stem(config)}.csv"


def refine_state_path_for(config: dict[str, Any], *, data_dir: Path | None = None) -> Path:
    """Path to the per-local_prob refinement-progress sidecar (JSON).

    Records, for each local_prob, the window `run_correctable_regime.py`'s
    refine-around-threshold mode actually collected, whether the resulting
    threshold estimate landed *inside* that window, and the shot/error budget
    used. Lets a re-run tell "already refined and converged here" from "still
    searching", so a converged curve is not re-collected on a slightly shifted
    window every time (the crossing estimate jitters across the snap grid).
    Lives beside the crg data CSV, keyed the same way.
    """
    if data_dir is None:
        data_dir = results_root(config)
    return data_dir / f"{crg_stem(config)}_refine_state.json"


def _refine_key(local_prob: float | None) -> str:
    return "null" if local_prob is None else repr(float(local_prob))


def read_refine_state(path: Path) -> dict[float | None, dict[str, Any]]:
    """Reads a refinement-state sidecar into {local_prob: entry}."""
    if not path.exists():
        return {}
    with open(path) as f:
        raw = json.load(f)
    out: dict[float | None, dict[str, Any]] = {}
    for k, v in raw.items():
        out[None if k == "null" else float(k)] = v
    return out


def write_refine_state(path: Path, state: dict[float | None, dict[str, Any]]) -> None:
    """Writes the {local_prob: entry} refinement-state map as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump({_refine_key(k): v for k, v in state.items()}, f, indent=2, sort_keys=True)
    tmp.replace(path)


def read_crg_data(path: Path) -> dict[float, tuple[float, float | None, str]]:
    """Reads a crg data file into {local_prob: (threshold, stderr, method)}.

    Rows with method == 'ported' are skipped -- these are migrated/legacy
    values of unverified provenance (some are corrupted, e.g. a stray extra
    '0.' in the threshold column that isn't even parseable as a float) and
    are treated as if that local_prob had never been collected at all.
    """
    out: dict[float, tuple[float, float | None, str]] = {}
    if not path.exists():
        return out
    skipped = 0
    # Some imported files have a stray space after every comma (not the
    # format write_crg_data ever produces) -- strip whitespace on every field
    # so that doesn't make the 'ported' check silently miss a row and fall
    # through to parsing " " as a float.
    with open(path, newline="") as f:
        for raw_row in csv.DictReader(f):
            row = {k: v.strip() for k, v in raw_row.items()}
            if row["method"] == "ported":
                skipped += 1
                continue
            stderr = float(row["stderr"]) if row["stderr"] else None
            out[float(row["local_prob"])] = (float(row["threshold"]), stderr, row["method"])
    if skipped:
        print(f"{path}: ignoring {skipped} 'ported' row(s)")
    return out


def write_crg_data(
    path: Path, rows: list[tuple[float, float | None, float | None, str]],
) -> None:
    """Writes rows of (local_prob, threshold, stderr, method); unresolved
    (threshold is None) entries are skipped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["local_prob", "threshold", "stderr", "method"])
        for local_prob, threshold, stderr, method in rows:
            if threshold is None:
                continue
            writer.writerow([local_prob, threshold, stderr if stderr is not None else "", method])


# Rough local-noise-only threshold: the p_l at which pure circuit noise
# alone (no non-local channel) already exceeds threshold, shared across the
# circuit-and-erasure/circuit-and-loss variants. Pinning this as a fixed
# vanishing point (rather than fitting it) removes a free parameter from
# `fit_boundary_curve` -- important now that several variants only have 3-4
# real points to fit against.
LOCAL_NOISE_THRESHOLD = 0.0065


def boundary_x_max(config: dict[str, Any]) -> float | None:
    """The vanishing point to pin :func:`fit_boundary_curve` to, or None to fit it.

    LOCAL_NOISE_THRESHOLD is the *circuit*-noise-only threshold, so it is only a
    valid pin when the axis being swept is that circuit noise -- the
    `circuit_and_*` noise models. The `prep_and_erasure` sweeps move a different
    channel over a different range: `crg_prep_bell` reaches p=0.02, three times
    LOCAL_NOISE_THRESHOLD, so pinning made `fit_boundary_curve` bail out through
    its `x_known_max >= x_max` guard and that config could never get a curve
    seed or a fitted boundary at all.

    Keyed on `noise_model_name` rather than :func:`local_prob_label` on purpose.
    That label picks the *filename* prefix from the first channel's tag and has
    to keep returning 'pl' for crg_prep_bell, whose collected CSVs are all named
    `bell_prep_and_erasure_pl*` -- the two questions look alike but only one of
    them may change without orphaning data on disk.
    """
    if not result_names.is_legacy(config):
        circuit_axis = result_names.run_type(config) in ("crg_local", "crg_local_loss")
        return LOCAL_NOISE_THRESHOLD if circuit_axis else None
    name = str(config.get("noise_model_name", ""))
    return LOCAL_NOISE_THRESHOLD if name.startswith("circuit_and_") else None


def fit_boundary_curve(
    known: dict[float, tuple[float, float | None, str]],
    *, min_points: int | None = None, x_max: float | None = None,
) -> tuple[float, float, float] | None:
    """Fits the correctable-regime boundary (local_prob -> nonlocal
    threshold) to a symmetric superellipse, pinned at the known local_prob=0
    intercept y0 (real data -- always the first collected point) and
    vanishing at x_max, the local-noise-only threshold beyond which no
    nonlocal error is correctable at all:

        (p_l / x_max) ** m + (p_nl / y0) ** m = 1
        p_nl(p_l) = y0 * (1 - (p_l / x_max) ** m) ** (1 / m)

    A single shared exponent m governs curvature at *both* ends
    symmetrically. x_max and y0 are each the pure-channel threshold of an
    independent noise mechanism -- circuit noise alone, or non-local noise
    alone -- so there's no physical reason for one end to land tangentially
    smooth while the other cusps; an earlier version of this fit pinned the
    p_l-axis exponent to 1 (always smooth there) and only fit the p_nl-axis
    exponent, which was an arbitrary asymmetry, not a physical one. m < 1
    gives a vertical-tangent cusp at both ends, m = 1 is the straight
    budget-sharing line p_l/x_max + p_nl/y0 = 1, m > 1 bows outward at both
    ends.

    If `x_max` is given, it's pinned to that value and only m is fit --
    appropriate when the vanishing point is known independently (e.g.
    LOCAL_NOISE_THRESHOLD for the crg_local_* variants, whose local_prob axis
    is circuit noise). If `x_max` is None (default), it's fit adaptively
    alongside m, bounded to lie beyond the largest known local_prob --
    appropriate when there's no independently known vanishing point (e.g. the
    crg_prep_* variants, whose local_prob axis is a different noise channel
    -- prep/graph noise, swept over a very different range -- with no shared
    known threshold).

    Returns (y0, x_max, m), or None if there's nothing to fit: fewer than
    `min_points` real points (default 3 with x_max pinned, 4 when fitting
    x_max too -- one more free parameter needs one more point), local_prob=0
    hasn't been collected (no y0 to pin to), real data already reaches/
    exceeds a pinned x_max (no longer consistent with the data), or the fit
    doesn't converge. Evaluate with :func:`eval_boundary_curve`.
    """
    from scipy.optimize import curve_fit

    if min_points is None:
        min_points = 3 if x_max is not None else 4
    if len(known) < min_points or 0.0 not in known:
        return None
    y0 = known[0.0][0]
    xs = np.array(sorted(known))
    ys = np.array([known[x][0] for x in xs])
    x_known_max = xs.max()

    if x_max is not None:
        if x_known_max >= x_max:
            return None

        def model(p_l, m):
            ratio = np.clip(p_l / x_max, 0.0, 1.0)
            base = np.clip(1.0 - ratio ** m, 0.0, None)
            return y0 * base ** (1.0 / m)

        try:
            popt, _ = curve_fit(model, xs, ys, p0=[1.0], bounds=([0.3], [5.0]), maxfev=10000)
        except (RuntimeError, ValueError):
            return None
        return y0, float(x_max), float(popt[0])

    def model(p_l, x_max_fit, m):
        ratio = np.clip(p_l / x_max_fit, 0.0, 1.0)
        base = np.clip(1.0 - ratio ** m, 0.0, None)
        return y0 * base ** (1.0 / m)

    p0 = [x_known_max * 1.2, 1.0]
    bounds = ([x_known_max * (1 + 1e-6), 0.3], [x_known_max * 5.0, 5.0])
    try:
        popt, _ = curve_fit(model, xs, ys, p0=p0, bounds=bounds, maxfev=10000)
    except (RuntimeError, ValueError):
        return None
    x_max_fit, m = popt
    return y0, float(x_max_fit), float(m)


def eval_boundary_curve(params: tuple[float, float, float], p_l: float) -> float:
    """Evaluates the :func:`fit_boundary_curve` model at a given local_prob."""
    y0, x_max, m = params
    ratio = min(max(p_l / x_max, 0.0), 1.0)
    base = max(1.0 - ratio ** m, 0.0)
    return float(y0 * base ** (1.0 / m))


def ler(stat: sinter.TaskStats, *, include_discards: bool = False) -> float | None:
    shots = stat.shots - (0 if include_discards else stat.discards)
    hits = stat.errors + (stat.discards if include_discards else 0)
    if shots <= 0:
        return None
    return hits / shots


def load_stats_by_local_prob(
    config: dict[str, Any], *, results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict[float, list[sinter.TaskStats]]:
    """Load one CSV per `sweep.local_probs` entry, skipping missing files."""
    out: dict[float, list[sinter.TaskStats]] = {}
    for local_prob in config["sweep"]["local_probs"]:
        path = csv_path_for(config, local_prob, results_dir=results_dir)
        if path.exists():
            out[local_prob] = list(sinter.read_stats_from_csv_files(path))
        else:
            print(f"Warning: missing {path}")
    return out


def crossing_x(xs: list[float], y_a: list[float], y_b: list[float]) -> float | None:
    diff = [a - b for a, b in zip(y_a, y_b)]
    for i in range(len(xs) - 1):
        d0, d1 = diff[i], diff[i + 1]
        # An exact tie where both curves read zero isn't a real crossing --
        # it's the below-threshold noise floor where no errors have been
        # sampled yet at either distance, so it carries no information about
        # where the curves actually cross.
        if d0 == 0 and not (y_a[i] == 0 and y_b[i] == 0):
            return xs[i]
        if d0 * d1 < 0:
            x0, x1 = xs[i], xs[i + 1]
            return x0 - d0 * (x1 - x0) / (d1 - d0)
    return None


def estimate_nonlocal_threshold(
    stats: list[sinter.TaskStats], distances: list[int], *, include_discards: bool = False,
) -> float | None:
    """Estimate the non-local threshold p_nl* from distance-curve crossings.

    Mirrors `experiments/run_correctable_regime.py::_estimate_threshold`.
    """
    # Sum shots/hits per (d, p) first: the same point can sit under several
    # strong_ids (e.g. circuits written out differently on two machines, or
    # before/after a cache rebuild), and keeping one row's LER per point would
    # silently drop the others' shots.
    totals: dict[tuple[int, float], list[int]] = defaultdict(lambda: [0, 0])
    for s in stats:
        md = s.json_metadata or {}
        key = (int(md.get("distance", 0)), float(md.get("p", 0)))
        totals[key][0] += s.shots - (0 if include_discards else s.discards)
        totals[key][1] += s.errors + (s.discards if include_discards else 0)
    by_d: dict[int, dict[float, float]] = defaultdict(dict)
    for (d, p), (shots, hits) in totals.items():
        if shots > 0:
            by_d[d][p] = hits / shots

    # Degrade gracefully to whichever requested distances actually have data
    # (e.g. d=7 not yet collected for some variants) rather than refusing to
    # estimate at all -- a 2-distance crossing is noisier but still usable.
    sorted_ds = sorted(d for d in distances if d in by_d)
    if len(sorted_ds) < 2:
        return None

    common_ps = set(by_d[sorted_ds[0]])
    for d in sorted_ds[1:]:
        common_ps &= set(by_d[d])
    xs = sorted(common_ps)
    if len(xs) < 2:
        return None

    # Find first x where highest distance has strictly lower LER than lowest
    # distance (correctable ordering begins), then include one step back to
    # bracket the crossing. Skip points where the highest-distance LER is 0,
    # which are likely noise-floor artifacts rather than true crossings.
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
        c = crossing_x(search_xs, [by_d[d_a][x] for x in search_xs], [by_d[d_b][x] for x in search_xs])
        if c is not None:
            crossings.append(c)
    return sum(crossings) / len(crossings) if crossings else None
