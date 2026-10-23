"""Aggregate result filenames must distinguish every axis a config sweeps.

The per-(local_prob, merge, observable) sinter CSVs always carried merge_type
and observable in their names, but `aggregate_identifier` did not -- so two
configs differing only in `sweep.merge_types` / `sweep.observables` wrote the
same correctable_regime_<id>.csv/.png/_refine_state.json and the second run
overwrote the first's thresholds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_threshold_experiment as rte  # noqa: E402
import _regime_common as rc  # noqa: E402

def _config(merge_types: list[str], observables: list[str]) -> dict:
    return {
        "builder": {"name": "build_surgery_bell_prep"},
        "resource_state": "bell",
        "noise_model_name": "circuit_and_erasure",
        "sweep": {"merge_types": list(merge_types), "observables": list(observables)},
        "sinter": {"decoder": "mwpf", "mwpf_cluster_node_limit": 1},
    }


def test_legacy_xx_sweep_keeps_its_identifier() -> None:
    """Everything already collected under results/ was an XX/XX sweep."""
    assert rte.sweep_variant_suffix(_config(["XX"], ["XX"])) == ""
    assert rte.aggregate_identifier(_config(["XX"], ["XX"])) == (
        "bell_circuit_and_erasure_build_surgery_bell_prep_mwpf"
        "_SolverSerialJointSingleHair_1"
    )


@pytest.mark.parametrize(
    "merge_types, observables",
    [(["ZZ"], ["ZZ"]), (["ZZ"], ["XX"]), (["XX"], ["ZZ"]), (["XX", "ZZ"], ["XX"])],
)
def test_other_sweeps_get_distinct_identifiers(
    merge_types: list[str], observables: list[str],
) -> None:
    legacy = rte.aggregate_identifier(_config(["XX"], ["XX"]))
    assert rte.aggregate_identifier(_config(merge_types, observables)) != legacy


def test_merge_and_observable_reach_every_aggregate_path() -> None:
    xx, zz = _config(["XX"], ["XX"]), _config(["ZZ"], ["ZZ"])
    assert rc.crg_data_path_for(xx) != rc.crg_data_path_for(zz)
    assert rc.refine_state_path_for(xx) != rc.refine_state_path_for(zz)


@pytest.mark.parametrize(
    "merge_types, observables",
    [(["XX"], ["XX"]), (["ZZ"], ["ZZ"]), (["X", "Z"], ["joint", "perp"])],
)
def test_both_implementations_agree(
    merge_types: list[str], observables: list[str],
) -> None:
    """`_regime_common` mirrors the collection script; they must not drift."""
    config = _config(merge_types, observables)
    assert rc.aggregate_identifier(config) == rte.aggregate_identifier(config)


def test_min_max_step_sweep_probs_are_canonicalized() -> None:
    sweep = {"min": 0.0, "max": 0.44, "step": 0.02}
    probs = rte.sweep_probs(sweep)
    assert probs[-1] == 0.44
    assert all(p == round(p, 12) for p in probs)


def test_explicit_sweep_probs_are_canonicalized() -> None:
    sweep = {"probs": [0.1, 0.440000000003, 0.5000000000004]}
    assert rte.sweep_probs(sweep) == [0.1, 0.44, 0.5]


# --- Current convention (dizx.result_names) -------------------------------

from dizx import result_names as rn  # noqa: E402

_CIRCUIT_PL = [
    {"tag": ["circuit", "two_qubit_gate"], "kind": "DEPOLARIZE2", "probability": "p_l"},
    {"tag": ["circuit-graph", "fusion_gate"], "kind": "DEPOLARIZE2", "probability": "p_l"},
]
_ERASURE = {"tag": ["graph", "fusion_measurement"], "kind": "HERALDED_ERASE", "probability": "p_nl"}
_LOSS_N2 = {"tag": ["graph", "fusion_measurement"], "kind": "HERALDED_ERASE",
            "probability": "p_e", "n_boost": 2}
_CORR = {"decoder": "pymatching_heralded", "pm_decompose": True, "pm_enable_correlations": True}


def _new_config(name: str, noise_model: list[dict], *, resource_state: str = "linear_k1",
                sinter_args: dict = _CORR, merge_types=("ZZ",), observables=("XX",)) -> dict:
    return {
        "name": name,
        "builder": {"name": "surgery_hybrid_linear_5chain"},
        "resource_state": resource_state,
        "noise_model": noise_model,
        "results_dir": "results/pymatching",
        "sweep": {"merge_types": list(merge_types), "observables": list(observables)},
        "sinter": dict(sinter_args),
    }


@pytest.mark.parametrize(
    "name, noise_model, local_prob, sinter_args, sinter_stem, aggregate_stem",
    [
        ("crg_local_linear_k1_corr_obsXX", _CIRCUIT_PL + [_ERASURE], 0.002, _CORR,
         "linear_k1_crg_local_pl0.002_mtZZ_obsXX_pm_h_corr",
         "crg_local_linear_k1_mtZZ_obsXX_pm_h_corr"),
        ("crg_local_loss_linear_k1_corr_obsXX_n2", _CIRCUIT_PL + [_LOSS_N2], 0.002, _CORR,
         "linear_k1_crg_local_loss_pl0.002_n2_mtZZ_obsXX_pm_h_corr",
         "crg_local_loss_linear_k1_n2_mtZZ_obsXX_pm_h_corr"),
        ("crg_prep_linear_k1_obsXX",
         [{"tag": ["graph", "state_prep", "matter"], "kind": "Z_ERROR", "probability": "p_l"},
          _ERASURE], 0.03, {"decoder": "pymatching_heralded"},
         "linear_k1_crg_prep_pr0.03_mtZZ_obsXX_pm_h",
         "crg_prep_linear_k1_mtZZ_obsXX_pm_h"),
        ("subthreshold_loss_target_linear_k1_corr_n3",
         [{"tag": ["circuit", "reset"], "kind": "RESET_FLIP", "probability": 0.001},
          {"tag": ["graph", "reset", "matter"], "kind": "RESET_FLIP", "probability": 0.002},
          {**_LOSS_N2, "n_boost": 3}], None, _CORR,
         "linear_k1_subthreshold_loss_pl0.001_pr0.002_n3_mtZZ_obsXX_pm_h_corr",
         "subthreshold_loss_linear_k1_pl0.001_pr0.002_n3_mtZZ_obsXX_pm_h_corr"),
        ("subthreshold_lambda_linear_k1_corr_obsXX_n2", [_LOSS_N2], None, _CORR,
         "linear_k1_subthreshold_lambda_n2_mtZZ_obsXX_pm_h_corr",
         "subthreshold_lambda_linear_k1_n2_mtZZ_obsXX_pm_h_corr"),
        ("erasure_linear_k1_corr_obsXX",
         [{"tag": ["graph", "fusion_measurement"], "kind": "HERALDED_ERASE", "scale": 1.0}],
         None, _CORR,
         "linear_k1_erasure_mtZZ_obsXX_pm_h_corr", "erasure_linear_k1_mtZZ_obsXX_pm_h_corr"),
        ("circuit_linear_k1_corr_obsXX",
         [{"tag": ["circuit", "reset"], "kind": "RESET_FLIP", "scale": 1.0}], None,
         {"decoder": "pymatching", "pm_decompose": True, "pm_enable_correlations": True},
         "linear_k1_circuit_mtZZ_obsXX_pm_corr", "circuit_linear_k1_mtZZ_obsXX_pm_corr"),
    ],
)
def test_current_convention_names(name, noise_model, local_prob, sinter_args,
                                  sinter_stem, aggregate_stem) -> None:
    config = _new_config(name, noise_model, sinter_args=sinter_args)
    channels = rn.resolve_local_prob(noise_model, local_prob)
    out = Path("results/pymatching")
    assert rte.sinter_csv_path(config, channels, "ZZ", "XX", out_dir=str(out)) == (
        f"{out}/sinter/{sinter_stem}.csv")
    # The scripts' lookup must land on the file the runner writes.
    assert rc.csv_path_for(config, local_prob, results_dir=out / "sinter") == (
        out / "sinter" / f"{sinter_stem}.csv")
    assert rte.aggregate_identifier(config) == rc.aggregate_identifier(config) == aggregate_stem
    root = rc.results_root(config)
    assert rc.crg_data_path_for(config) == root / f"{aggregate_stem}.csv"
    assert rc.refine_state_path_for(config) == root / f"{aggregate_stem}_refine_state.json"


def test_alpha_plot_is_named_subthresh() -> None:
    config = _new_config("subthreshold_loss_linear_k1_corr_obsXX_n2",
                         [{"tag": ["circuit", "reset"], "kind": "RESET_FLIP", "probability": 0.001},
                          _LOSS_N2])
    assert rn.alpha_plot_stem(config) == "subthresh_loss_linear_k1_pl0.001_n2_mtZZ_obsXX_pm_h_corr"
    config["name"] = "subthreshold_linear_k1_zz_corr"
    assert rn.alpha_plot_stem(config) == "subthresh_linear_k1_pl0.001_n2_mtZZ_obsXX_pm_h_corr"


def test_legacy_config_keeps_its_names() -> None:
    config = _config(["ZZ"], ["ZZ"])
    config["noise_model"] = [{"tag": ["circuit", "reset"], "kind": "RESET_FLIP", "probability": 0.001}]
    stem = rte.run_stem(config, config["noise_model"], ["ZZ"], ["ZZ"])
    assert stem == ("bell_circuit_and_erasure_pl0.001_build_surgery_bell_prep_mergeZZ_observableZZ"
                    "_mwpf__1")
    assert rc.crg_stem(config) == f"correctable_regime_{rte.aggregate_identifier(config)}"


def test_boundary_pin_follows_run_type() -> None:
    assert rc.boundary_x_max(_new_config("crg_local_bell_obsZZ", _CIRCUIT_PL)) is not None
    assert rc.boundary_x_max(_new_config("crg_local_loss_bell_obsZZ", _CIRCUIT_PL)) is not None
    assert rc.boundary_x_max(_new_config("crg_prep_bell_obsZZ", _CIRCUIT_PL)) is None


@pytest.mark.parametrize("key", rte.REMOVED_SWEEP_KEYS)
def test_variant_sweep_keys_are_refused(key: str) -> None:
    config = _new_config("crg_local_bell_obsZZ", _CIRCUIT_PL, resource_state="bell")
    config["sweep"][key] = 1
    with pytest.raises(ValueError, match="builder variant"):
        rte.check_builder_options(config)


# Keys a config may change and still write into another's CSV: which points
# it runs and how hard (e.g. a d=9 or low-noise companion of a base config).
_POINT_KEYS = {"name", "notes", "sweep.distances", "sweep.probs", "sweep.local_probs",
               "sinter.max_errors", "sinter.max_shots", "sinter.num_workers"}


def _experiment(config: dict, prefix: str = "") -> dict:
    """``config`` flattened, without the keys in ``_POINT_KEYS``."""
    out = {}
    for k, v in config.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_experiment(v, key + "."))
        elif key not in _POINT_KEYS:
            out[key] = v
    return out


def test_paper_configs_have_distinct_names() -> None:
    """Every paper config resolves, and two configs share an aggregate only
    when they are the same experiment at different points."""
    import paper
    configs = [paper.erasure_config(f, o) for f in paper.FAMILIES for o in paper.OBSERVABLES]
    configs += [paper.crg_config(k, f, o) for k in paper.CRG_LOCAL_PROBS for f in paper.FAMILIES
                for o in paper.OBSERVABLES]
    configs += [paper.lambda_config(f, o, n, [lp]) for (f, o, lp), by_n in paper.LAMBDA_POINTS.items()
                for n in by_n]
    stems: dict[str, tuple[str, dict]] = {}
    for config in configs:
        assert not rn.is_legacy(config)
        stem = rn.aggregate_stem(config)
        if stem in stems:
            other, experiment = stems[stem]
            assert _experiment(config) == experiment, f"{config['name']} and {other} share {stem}"
        else:
            stems[stem] = (config["name"], _experiment(config))
    assert len(stems) == 10 + 20 + 20
