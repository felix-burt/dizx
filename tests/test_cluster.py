"""PBS submission of the paper plan (experiments/cluster.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import sinter

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import cluster  # noqa: E402
import paper  # noqa: E402
import reproduce  # noqa: E402


def _plan(targets, split="curve", results_dir="unused"):
    return cluster.plan_jobs(targets, split=split, budget_scale=1.0, results_dir=results_dir)


def test_every_run_in_exactly_one_job():
    curves, _ = reproduce.resolve_targets(["all"])
    wanted = {cluster.run_stem(run) for c in curves for run in c.runs()}
    for split in ("curve", "run"):
        jobs = _plan(["all"], split)
        stems = [s for job in jobs for s in job.stems]
        assert sorted(stems) == sorted(wanted), split
    assert len(_plan(["all"], "curve")) == len(paper.CURVES)


def test_crg_erasure_run_goes_to_the_erasure_job():
    jobs = {job.name: job for job in _plan(["crg_prep/bell/ZZ"])}
    assert set(jobs) == {"crg_prep-bell-ZZ", "erasure-bell-ZZ"}
    assert len(jobs["crg_prep-bell-ZZ"].runs) == len(paper.CRG_LOCAL_PROBS["crg_prep"]["bell"])
    run_names = {job.name for job in _plan(["crg_prep/bell/ZZ"], "run")}
    assert "crg_prep-bell-ZZ-pr0p01" in run_names and "erasure-bell-ZZ-p0" in run_names


def test_cost_table_covers_the_plan():
    costs = cluster.load_costs()
    for job in _plan(["all"], "run"):
        for run in job.runs:
            for task in cluster.run_tasks(run):
                assert repr(task.p) in costs[task.stem][str(task.distance)], task


def _write(path: Path, rows: list[tuple[int, float, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [sinter.CSV_HEADER] + [
        sinter.TaskStats(strong_id=f"{d}_{p}", decoder="pymatching_heralded",
                         json_metadata={"distance": d, "p": p}, shots=shots, errors=errors,
                         seconds=1.0).to_csv_line()
        for d, p, shots, errors in rows]
    path.write_text("\n".join(lines) + "\n")


def test_collected_tasks_cost_nothing(tmp_path):
    [job] = _plan(["erasure/bell/ZZ"], results_dir=str(tmp_path))
    [run] = job.runs
    tasks = cluster.run_tasks(run)
    max_shots, max_errors = run["sinter"]["max_shots"], run["sinter"]["max_errors"]
    csv = tmp_path / "sinter" / f"{tasks[0].stem}.csv"
    # One task at its shot budget, one at its error budget, one half way.
    _write(csv, [(tasks[0].distance, tasks[0].p, max_shots, 10),
                 (tasks[1].distance, tasks[1].p, 2 * max_errors, max_errors),
                 (tasks[2].distance, tasks[2].p, max_shots // 2, 10)])
    costs = cluster.load_costs()
    fresh = cluster.job_cost(job, costs, cluster.Progress(tmp_path / "empty"))
    cost = cluster.job_cost(job, costs, cluster.Progress(tmp_path))
    assert (fresh.tasks, cost.tasks) == (len(tasks), len(tasks) - 2)
    expected = fresh.core_hours - sum(cluster.task_cost(costs, t) for t in tasks[:2]) / 3600 \
        - cluster.task_cost(costs, tasks[2]) / 2 / 3600
    assert abs(cost.core_hours - expected) < 1e-9


def test_resources_respect_limits():
    [job] = _plan(["crg_local/linear_k1/ZZ"])[:1]
    limits = cluster.Limits(max_cpus=64, max_mem_gb=1000, max_hours=48, target_hours=10)
    big = cluster.Cost(tasks=1, total_tasks=1, core_hours=10_000, overhead_hours=0)
    res = cluster.resources(job, big, limits)
    assert res.ncpus == 64 and res.walltime_h == 48 and res.truncated
    # Memory caps the workers.
    tight = cluster.resources(job, big, cluster.Limits(max_mem_gb=40))
    assert tight.mem_gb <= 40 and tight.ncpus < 64
    small = cluster.resources(job, cluster.Cost(1, 1, 0.1, 0.0), limits)
    assert (small.ncpus, small.walltime_h) == (limits.min_cpus, 1)


def test_auto_split_breaks_up_only_long_curves(tmp_path):
    jobs = _plan(["crg_local/linear_k1/ZZ", "crg_prep/bell/ZZ"], results_dir=str(tmp_path))
    split = cluster.split_oversized(jobs, cluster.load_costs(), cluster.Progress(tmp_path),
                                    cluster.Limits())
    names = {job.name for job in split}
    assert "crg_prep-bell-ZZ" in names and "crg_local-linear_k1-ZZ" not in names
    assert "crg_local-linear_k1-ZZ-pl0p005" in names


def test_pbs_script():
    res = cluster.Resources(ncpus=16, mem_gb=40, walltime_h=7, needs_hours=6.5)
    script = cluster.pbs_script("crg_prep-bell-ZZ", res=res, command="python x.py",
                                log=Path("/l/x.log"), setup="source .venv/bin/activate",
                                queue=None, account="proj", extra=["-m ae"])
    for line in ["#PBS -N crg_prep-bell-ZZ", "#PBS -l select=1:ncpus=16:mem=40gb",
                 "#PBS -l walltime=07:00:00", "#PBS -o /l/x.log", "#PBS -P proj", "#PBS -m ae",
                 "source .venv/bin/activate", "python x.py"]:
        assert line in script.splitlines()
    assert "#PBS -q" not in script
