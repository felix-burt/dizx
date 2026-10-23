"""Refine-window search behaviour in `run_correctable_regime`.

These drive the real branch logic with `run_threshold` stubbed out, so they
exercise which window gets swept without collecting any shots. The synthetic
LER model puts the crossing of every distance pair at exactly one probability,
so the estimated threshold is whatever the test says it is.

Covers the three ways a refine window can be wrong (see conversation
2026-08-07, when d=7 was dropped from `distances` and every stored window's
`found_in_region` became untrustworthy):

* a local_prob with no converged window at all -> slide search
* a stored window that no longer brackets before collection -> slide, not reuse
* a stored window that stops bracketing *because of* what this run collected
  -> reuse sweep, then slide onto the threshold within the same run
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import sinter

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import run_correctable_regime as rcr  # noqa: E402
from _regime_common import (  # noqa: E402
    crg_data_path_for, read_refine_state, refine_state_path_for, write_crg_data,
    write_refine_state,
)

DISTANCES = [9, 11]
TRUE_THRESHOLD = 0.100


def _config(tmp_path: Path, local_probs: list[float]) -> dict:
    """A minimal refine-mode config writing everything under tmp_path.

    `results_root`/`results_dir` join this onto the repo root; an absolute path
    wins that join, so nothing lands in the real results/ tree.
    """
    return {
        "name": "test_crg",
        "builder": {"name": "build_surgery_bell_prep"},
        "resource_state": "bell",
        "noise_model_name": "circuit_and_erasure",
        "results_dir": str(tmp_path),
        "sweep": {
            "distances": list(DISTANCES),
            "merge_types": ["XX"],
            "observables": ["XX"],
            "estimator": "crossing",
            "local_probs": list(local_probs),
            "bisection": True,
            "min": 0.0,
            "max": 0.5,
            "tolerance": 0.01,
            "refine_around_threshold": True,
            "refine_frac": 0.1,
            "refine_points": 10,
        },
        "noise_model": [
            {"tag": ["circuit", "reset"], "kind": "RESET_FLIP", "probability": "p_l"},
            {"tag": ["circuit", "bell_prep_two_qubit"], "kind": "HERALDED_ERASE",
             "probability": "p_nl"},
        ],
        "sinter": {"decoder": "mwpf", "max_shots": 1000, "max_errors": 100,
                   "mwpf_decoder_type": "SolverSerialUnionFind",
                   "mwpf_cluster_node_limit": 0},
    }


class _Collector:
    """Stands in for run_threshold: synthesises stats and records the sweeps.

    LER(d, p) = 0.5 + (p - threshold) * d, so every distance pair crosses at
    `threshold` exactly and the higher distance is lower below it.
    """

    def __init__(self, threshold: float = TRUE_THRESHOLD):
        self.threshold = threshold
        self.collected: dict[tuple[int, float], sinter.TaskStats] = {}
        self.sweeps: list[list[float]] = []

    def run_threshold(self, config, persist=True):
        probs = list(config["sweep"]["probs"])
        self.sweeps.append(probs)
        for p in probs:
            for d in DISTANCES:
                ler = min(max(0.5 + (p - self.threshold) * d, 1e-3), 1 - 1e-3)
                self.collected[(d, p)] = sinter.TaskStats(
                    strong_id=f"{d}_{p}", decoder="mwpf",
                    json_metadata={"distance": d, "p": p},
                    shots=10_000, errors=int(ler * 10_000), discards=0, seconds=1.0,
                )
        return self.load(config)

    def load(self, config):
        return list(self.collected.values())


@pytest.fixture
def collector(monkeypatch):
    c = _Collector()
    monkeypatch.setattr(rcr, "run_threshold", c.run_threshold)
    monkeypatch.setattr(rcr, "load_threshold_stats", c.load)
    monkeypatch.setattr(rcr, "run_threshold_bisection",
                        lambda config, persist=True: (c.threshold, []))
    # slide_refine_window probes points through run_threshold_experiment's own
    # module-level reference, which the monkeypatch above does not reach.
    import run_threshold_experiment as rte
    monkeypatch.setattr(rte, "run_threshold", c.run_threshold)
    return c


def _seed_state(config, local_prob, probs, center, threshold):
    """Record a converged window, as a previous run would have left it."""
    write_refine_state(refine_state_path_for(config), {
        local_prob: {
            "probs": list(probs), "center": center, "threshold": threshold,
            "found_in_region": True,
            "max_shots": config["sinter"]["max_shots"],
            "max_errors": config["sinter"]["max_errors"],
            "distances": list(DISTANCES),
        },
    })
    write_crg_data(crg_data_path_for(config), [(local_prob, threshold, None, "crossing")])


def test_slide_branch_records_state_without_crashing(tmp_path, collector):
    """A local_prob with no prior window goes through the slide branch.

    Regression test: that branch's `_record_refine_state` call was missing the
    `distances` argument, so it raised TypeError before writing anything --
    which is the path every brand-new local_prob takes.
    """
    config = _config(tmp_path, [0.0])
    rcr.run_correctable_regime(config)

    state = read_refine_state(refine_state_path_for(config))
    assert state[0.0]["found_in_region"] is True
    assert state[0.0]["distances"] == DISTANCES
    assert state[0.0]["probs"][0] <= TRUE_THRESHOLD <= state[0.0]["probs"][-1]


def test_stale_window_is_not_reused(tmp_path, collector):
    """A stored window that no longer brackets must not be swept as-is.

    Reusing it would re-collect a window known to be wrong and record another
    miss, deferring the real search to a later run.
    """
    config = _config(tmp_path, [0.0])
    stale = [round(0.180 + 0.004 * i, 6) for i in range(10)]  # far above 0.100
    _seed_state(config, 0.0, stale, center=0.20, threshold=0.20)

    rcr.run_correctable_regime(config)

    assert collector.sweeps, "nothing was collected"
    assert stale not in collector.sweeps, "the stale window was swept unchanged"
    state = read_refine_state(refine_state_path_for(config))
    assert state[0.0]["probs"][0] <= TRUE_THRESHOLD <= state[0.0]["probs"][-1]
    assert state[0.0]["found_in_region"] is True


def test_window_that_breaks_during_collection_is_retried(tmp_path, collector):
    """A window valid before collection but not after still converges in-run.

    This is the partly-collected-distance case: the stored window brackets the
    threshold implied by the data already on disk, so it is reused -- and only
    once the sweep fills in the rest does the threshold move out of it. The
    reuse branch must slide onto the threshold rather than record the miss.
    """
    config = _config(tmp_path, [0.0])
    window = [round(0.095 + 0.001 * i, 6) for i in range(10)]  # 0.095..0.104
    _seed_state(config, 0.0, window, center=0.0995, threshold=0.0995)
    # Pre-load exactly the stored window so window_reverified passes, then move
    # the true threshold outside it -- what a newly-collected distance does.
    collector.run_threshold({"sweep": {"probs": window}})
    collector.sweeps.clear()
    collector.threshold = 0.130
    collector.collected.clear()

    rcr.run_correctable_regime(config)

    state = read_refine_state(refine_state_path_for(config))
    assert state[0.0]["found_in_region"] is True, "did not converge within the run"
    assert state[0.0]["probs"][0] <= 0.130 <= state[0.0]["probs"][-1]
    assert len(collector.sweeps) >= 2, "expected a retry sweep after the reused window"


def test_converged_window_is_skipped(tmp_path, collector):
    """A window that still brackets under the current distances is not re-swept."""
    config = _config(tmp_path, [0.0])
    window = [round(0.095 + 0.001 * i, 6) for i in range(10)]
    _seed_state(config, 0.0, window, center=0.0995, threshold=TRUE_THRESHOLD)
    collector.run_threshold({"sweep": {"probs": window}})
    collector.sweeps.clear()

    rcr.run_correctable_regime(config)

    assert collector.sweeps == [], "a converged window was re-collected"


def test_local_prob_sweep_overrides_bisection_range(tmp_path, monkeypatch):
    """Per-local sweep overrides are forwarded into the bisection config."""
    config = _config(tmp_path, [0.0, 0.001])
    config["sweep"]["refine_around_threshold"] = False
    config["sweep"]["bisection"] = True
    config["sweep"]["local_prob_sweeps"] = {
        "0.0": {"min": 0.01, "max": 0.11, "tolerance": 0.005},
        "0.001": {"min": 0.02, "max": 0.12, "tolerance": 0.006},
    }

    seen: list[tuple[float, float, float]] = []

    def _run_threshold_bisection(run_config, persist=True):
        sw = run_config["sweep"]
        seen.append((sw["min"], sw["max"], sw["tolerance"]))
        return TRUE_THRESHOLD, []

    monkeypatch.setattr(rcr, "run_threshold_bisection", _run_threshold_bisection)
    monkeypatch.setattr(rcr, "load_threshold_stats", lambda cfg: [])

    rcr.run_correctable_regime(config)

    assert seen == [(0.01, 0.11, 0.005), (0.02, 0.12, 0.006)]


def test_local_prob_sweep_overrides_support_explicit_probs(tmp_path, monkeypatch):
    """A per-local override can swap between explicit probs and min/max/step."""
    config = _config(tmp_path, [0.0, 0.001])
    config["sweep"]["refine_around_threshold"] = False
    config["sweep"]["bisection"] = False
    config["sweep"]["step"] = 0.01
    config["sweep"]["local_prob_sweeps"] = {
        "0.0": {"probs": [0.05, 0.06]},
        "0.001": {"min": 0.07, "max": 0.09, "step": 0.01},
    }

    seen: list[dict] = []

    def _run_threshold(run_config, persist=True):
        seen.append(dict(run_config["sweep"]))
        return []

    monkeypatch.setattr(rcr, "run_threshold", _run_threshold)
    monkeypatch.setattr(rcr, "estimate_threshold_from_stats", lambda *args, **kwargs: (0.1, None, "crossing"))

    rcr.run_correctable_regime(config)

    assert seen[0]["probs"] == [0.05, 0.06]
    assert "min" not in seen[0] and "max" not in seen[0] and "step" not in seen[0]
    assert seen[1]["min"] == 0.07 and seen[1]["max"] == 0.09 and seen[1]["step"] == 0.01
    assert "probs" not in seen[1]
