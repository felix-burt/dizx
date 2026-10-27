"""crg refinement rounds (experiments/refine_crg.py) and their place in the plan."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import cluster  # noqa: E402
import paper  # noqa: E402
import refine_crg as rc  # noqa: E402
from figures.scaling_fit import ScalingFit  # noqa: E402


@pytest.fixture(autouse=True)
def _base_plan():
    """These test the paper's base plan, not the frozen crg refinement rounds."""
    with paper.refinement_rounds([]):
        yield

KEY = "crg_local/bell/ZZ/0.004"


def _fit(p_th=0.42, stderr=0.001, nu=1.5, valid=True):
    return ScalingFit(p_th, stderr, stderr, 0.0, nu, 1.0, 30, valid)


def _round(scale=4.0, p_th=0.42, probs=None):
    probs = probs or {"11": [0.41, 0.415, 0.42], "15": [0.42]}
    return {"points": {KEY: {"fit": {"p_th": p_th}, "scale": scale, "probs": probs}}}


def test_window_is_the_scaling_window_on_the_grid():
    probs = rc.window_probs(0.4213, 1.5, 11)
    half = rc.WINDOW_X / 11 ** (1 / 1.5)
    assert probs and all(abs(p - 0.4213) <= half for p in probs)
    assert all(abs(p / paper.CRG_WINDOW_STEP - round(p / paper.CRG_WINDOW_STEP)) < 1e-9 for p in probs)
    assert len(rc.window_probs(0.4213, 1.5, 15)) < len(probs)


def test_narrow_window_is_densified():
    # nu = 1 at d = 30: +-0.005, three points on the coarse grid.
    probs = rc.window_probs(0.3, 1.0, 30)
    assert len(probs) >= rc.MIN_PER_DISTANCE
    assert min(b - a for a, b in zip(probs, probs[1:])) < paper.CRG_WINDOW_STEP


def test_window_stays_positive():
    assert all(p > 0 for p in rc.window_probs(0.004, 1.5, 11))


def test_verdict():
    assert rc.verdict(_fit(stderr=0.0004), []) == "converged"
    assert rc.verdict(_fit(stderr=0.001), []) == "refine"
    # On target but moved more than half its error since the last round.
    assert rc.verdict(_fit(stderr=0.0004), [{"fit": {"p_th": 0.4210}}]) == "refine"
    assert rc.verdict(_fit(stderr=0.0004), [{"fit": {"p_th": 0.4199}}]) == "converged"
    assert rc.verdict(_fit(), [{"fit": {"p_th": 0.42}}] * rc.MAX_ROUNDS).startswith("round cap")
    assert rc.verdict(_fit(valid=False), []).startswith("no valid fit")


def test_budget_scale_is_cumulative_and_clamped():
    assert rc.plan_point(_fit(stderr=0.0006), [], KEY)["scale"] == rc.MIN_FACTOR
    assert rc.plan_point(_fit(stderr=0.003), [], KEY)["scale"] == rc.MAX_FACTOR
    # (0.001 / 0.0005)^2 = 4, clamped, on top of the 4x already planned at 0.42.
    assert rc.plan_point(_fit(stderr=0.001), [_round(4.0)], KEY)["scale"] == 4.0 * min(4.0, rc.MAX_FACTOR)


def test_rounds_become_runs_of_the_curve():
    curve = paper.CURVES["crg_local/bell/ZZ"]
    base = curve.runs()
    with paper.refinement_rounds([_round(4.0)]):
        runs = curve.runs()
    extra = [r for r in runs if r not in base]
    assert [(r["sweep"]["distances"], r["sweep"]["probs"]) for r in extra] == \
        [([11], [0.41, 0.415, 0.42]), ([15], [0.42])]
    shots, errors = paper.THRESHOLD_BUDGET
    assert all((r["sinter"]["max_shots"], r["sinter"]["max_errors"]) == (4 * shots, 4 * errors)
               for r in extra)
    # Same CSV as the base run at that p_l.
    assert {cluster.run_stem(r) for r in extra} == {cluster.run_stem(base[3])}


def test_refined_budget_prices_the_top_up(tmp_path):
    with paper.refinement_rounds([_round(4.0, probs={"11": [0.42]})]):
        [job] = [j for j in cluster.plan_jobs(["crg_local/bell/ZZ"], split="curve", budget_scale=1.0,
                                              results_dir=str(tmp_path)) if j.name == "crg_local-bell-ZZ"]
    [run] = [r for r in job.runs if r["sinter"]["max_errors"] > paper.THRESHOLD_BUDGET[1]]
    [task] = cluster.run_tasks(run)
    costs = cluster.load_costs()
    # Collected to the base budget: three more base budgets to go.
    shots, errors = paper.THRESHOLD_BUDGET
    path = tmp_path / "sinter" / f"{task.stem}.csv"
    path.parent.mkdir(parents=True)
    import sinter
    path.write_text(sinter.CSV_HEADER + "\n" + sinter.TaskStats(
        strong_id="x", decoder="pymatching_heralded", json_metadata={"distance": 11, "p": 0.42},
        shots=shots, errors=errors // 10).to_csv_line() + "\n")
    single = cluster.Job("x", runs=[run])
    cost = cluster.job_cost(single, costs, cluster.Progress(tmp_path))
    assert abs(cost.core_hours - 3 * cluster.task_cost(costs, task) / 3600) < 1e-9


def test_split_keeps_a_points_runs_in_one_job(tmp_path):
    with paper.refinement_rounds([_round(4.0)]):
        jobs = cluster.plan_jobs(["crg_local/bell/ZZ"], split="curve", budget_scale=1.0,
                                 results_dir=str(tmp_path))
        split = cluster.split_oversized(jobs, cluster.load_costs(), cluster.Progress(tmp_path),
                                        cluster.Limits(target_hours=1e-9))
    names = [job.name for job in split]
    assert len(names) == len(set(names))
    owners = {}
    for job in split:
        for stem in job.stems:
            assert owners.setdefault(stem, job.name) == job.name
    [point] = [job for job in split if job.name == "crg_local-bell-ZZ-pl0p004"]
    assert len(point.runs) == 3


def test_added_distance_joins_the_fits():
    assert paper.crg_fit_distances() == paper.CRG_DISTANCES
    fits = {KEY: _fit(), "crg_local/bell/ZZ/0.005": _fit(valid=False)}
    points = rc.add_distance(fits, 17)
    assert list(points) == [KEY] and list(points[KEY]["probs"]) == ["17"]
    assert points[KEY]["scale"] == 1.0
    with paper.refinement_rounds([{"points": points}]):
        assert paper.crg_fit_distances() == paper.CRG_DISTANCES + [17]
        [extra] = [r for r in paper.CURVES["crg_local/bell/ZZ"].runs() if r["sweep"]["distances"] == [17]]
    assert extra["sinter"]["max_errors"] == paper.THRESHOLD_BUDGET[1]
