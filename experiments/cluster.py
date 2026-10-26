#!/usr/bin/env python
"""Collect the paper's data on a PBS cluster, one job per curve.

    python experiments/cluster.py plan all                              # jobs, resources, cost
    python experiments/cluster.py submit all --results-dir results/cluster
    python experiments/cluster.py submit 'crg_prep_*' --dry-run         # write the scripts only
    python experiments/cluster.py status                                # progress of the last batch
    python experiments/cluster.py calibrate ../DISQEC/results/pymatching

Targets are those of reproduce.py (figure keys, curve ids, globs, `all`).
Every run they need goes into exactly one job. With `--split curve` a job is
one curve; a crg curve's p_l = 0 run is the pure-erasure sweep and goes to that
erasure curve's job instead. With `--split run` every run (one sinter CSV) is
its own job. `--split auto` (the default) splits only the curves that would
outlast `--target-hours` on `--max-cpus` (the crg_local ones, at full budget).
Either way no two jobs write the same CSV: concurrent appends and in-place
compaction from two nodes could corrupt it.

Resources come from a cost model. `cluster_costs.json` holds the expected
core-seconds of every task of the plan, measured from the DISQEC collection
that produced the paper (`calibrate` rebuilds it), less whatever the results
directory already holds. A job's ncpus is sized so that it finishes in about
`--target-hours`, its memory from the circuit size at its largest distance,
and its walltime from both, times `--safety`.

Collection resumes by strong_id, so a job that runs out of walltime loses at
most its current sinter batch: resubmit the same targets and only the missing
shots are collected (finished jobs are skipped, jobs still queued or running
from an earlier batch are left alone). When everything is in, draw the
figures with

    python experiments/reproduce.py all --plot-only --results-dir results/cluster

or pass `--plot` to `submit` for a job that does that once the batch ends.

Each batch writes its scripts, job specs and manifest to
`<jobs-dir>/<timestamp>/`, and PBS logs to `<jobs-dir>/<timestamp>/logs/`.
The job environment is set up by `--setup` (default: activate the in-project
.venv), e.g.

    --setup 'module load Python/3.12.3-GCCcore-13.3.0; source .venv/bin/activate'

`DIZX_PBS_SETUP`, `DIZX_PBS_QUEUE` and `DIZX_PBS_ACCOUNT` set the defaults.
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

_EXPERIMENTS = Path(__file__).resolve().parent
_REPO = _EXPERIMENTS.parent
sys.path.insert(0, str(_EXPERIMENTS))
sys.path.insert(0, str(_REPO / "scripts"))

import paper  # noqa: E402
import reproduce  # noqa: E402
from paper import Curve  # noqa: E402

COSTS_FILE = _EXPERIMENTS / "cluster_costs.json"
JOBS_DIR = _REPO / "results" / "pbs"

# Per-worker peak memory (GB) of the correlated heralded decoder for a
# surgery_bell_prep_fusion circuit at d = 19 (measured 1.6), scaled ~d^3 by
# distance and by circuit size across families (stim file size at d = 7).
_WORKER_GB_D19 = 1.7
_WORKER_GB_FLOOR = 0.2
_PARENT_GB = 2.0
_FAMILY_SIZE = {"bell": 1.0, "bell_zigzag": 1.1, "linear_kinf": 1.15, "linear_k2": 1.3,
                "linear_k1": 1.45}
# Wall seconds per task spent off the shot budget (noise model, DEM and
# decoder compilation in every worker): ~10 s measured at d = 19, doubled.
_TASK_OVERHEAD_S_D19 = 20.0
_TASK_OVERHEAD_S_FLOOR = 2.0
# Core-seconds per task when the calibration has nothing near it: a d = 15
# threshold point at its full shot budget.
_FALLBACK_TASK_S = 2000.0


# ---------------------------------------------------------------------------
# Tasks and jobs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    stem: str           # the sinter CSV it lands in, <results_dir>/sinter/<stem>.csv
    distance: int
    p: float


def run_stem(run: dict) -> str:
    from run_threshold_experiment import run_stem as stem
    sweep = run["sweep"]
    return stem(run, run["noise_model"], sweep["merge_types"], sweep["observables"])


def run_tasks(run: dict) -> list[Task]:
    from run_threshold_experiment import sweep_probs
    stem = run_stem(run)
    return [Task(stem, d, p) for d in run["sweep"]["distances"] for p in sweep_probs(run["sweep"])]


def _group(curve: Curve, run: dict) -> str:
    """The curve whose job collects `run`: a crg curve's pure-erasure run
    belongs to the erasure curve, which other crg curves share too."""
    if run["name"].startswith("erasure_"):
        return Curve("erasure", curve.family, curve.observable).id
    return curve.id


def job_name(group: str) -> str:
    """A PBS-safe job name (letters, digits, - and _; starting with a letter)."""
    return re.sub(r"[^A-Za-z0-9_-]", "-", group.replace(".", "p").replace("/", "-"))


@dataclass
class Job:
    name: str
    budget_scale: float = 1.0
    runs: list[dict] = field(default_factory=list)

    @property
    def stems(self) -> set[str]:
        return {run_stem(run) for run in self.runs}

    @property
    def family(self) -> str:
        return self.runs[0]["resource_state"]

    @property
    def max_distance(self) -> int:
        return max(max(run["sweep"]["distances"]) for run in self.runs)


def _run_name(group: str, run: dict) -> str:
    """`<curve>-<fixed noise levels>`, e.g. crg_local-bell-ZZ-pl0p005."""
    from dizx import result_names
    return job_name(f"{group}-{result_names.noise_param(run['noise_model']) or 'p0'}")


def plan_jobs(targets: list[str], *, split: str, budget_scale: float, results_dir: str) -> list[Job]:
    """The jobs collecting every run `targets` need: one per curve (`split`
    "curve") or one per run ("run")."""
    curves, _ = reproduce.resolve_targets(targets)
    jobs: dict[str, Job] = {}
    seen: set[str] = set()
    for curve in curves:
        for run in curve.runs():
            key = repr(sorted(run.items(), key=str))
            if key in seen:
                continue
            seen.add(key)
            # Workers are the job's to choose (`run --workers`).
            run = reproduce._scaled(run, workers=1, budget_scale=budget_scale, results_dir=results_dir)
            name = job_name(_group(curve, run))
            if split == "run":
                name = _run_name(name, run)
            jobs.setdefault(name, Job(name, budget_scale)).runs.append(run)
    owners: dict[str, str] = {}
    for job in jobs.values():
        for stem in job.stems:
            if stem in owners:
                raise RuntimeError(f"{stem}.csv would be written by jobs {owners[stem]} and {job.name}")
            owners[stem] = job.name
    return list(jobs.values())


# ---------------------------------------------------------------------------
# Progress and cost
# ---------------------------------------------------------------------------


def csv_progress(path: Path) -> dict[tuple[int, float], tuple[int, int, float]]:
    """{(distance, p): (shots, errors, core-seconds)} of one sinter CSV.

    Keyed by metadata rather than strong_id: computing a strong_id needs the
    noisy circuit, and within one run's CSV (distance, p) already identifies
    the task."""
    from run_threshold_experiment import canonical_prob
    out: dict[tuple[int, float], list] = defaultdict(lambda: [0, 0, 0.0])
    if not path.exists():
        return {}
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = [h.strip() for h in next(reader, [])]
        if not {"shots", "errors", "seconds", "json_metadata"} <= set(header):
            return {}
        idx = {h: i for i, h in enumerate(header)}
        for row in reader:
            if len(row) < len(header):
                continue
            meta = json.loads(row[idx["json_metadata"]])
            entry = out[(int(meta["distance"]), canonical_prob(meta["p"]))]
            entry[0] += int(row[idx["shots"]])
            entry[1] += int(row[idx["errors"]])
            entry[2] += float(row[idx["seconds"]])
    return {k: tuple(v) for k, v in out.items()}


class Progress:
    """Collected shots per task under one results directory (CSVs read once)."""

    def __init__(self, results_dir: str | Path):
        self.sinter_dir = Path(results_dir) / "sinter"
        self._cache: dict[str, dict] = {}

    def csv(self, stem: str) -> dict[tuple[int, float], tuple[int, int, float]]:
        if stem not in self._cache:
            self._cache[stem] = csv_progress(self.sinter_dir / f"{stem}.csv")
        return self._cache[stem]

    def of(self, task: Task) -> tuple[int, int, float]:
        return self.csv(task.stem).get((task.distance, task.p), (0, 0, 0.0))


def remaining_fraction(shots: int, errors: int, max_shots: int, max_errors: int) -> float:
    return max(0.0, 1.0 - max(shots / max_shots, errors / max_errors))


def expected_shots(shots: int, errors: int, max_shots: int, max_errors: int) -> float:
    """Shots a task needs to reach its budget, from a sample of it."""
    if errors <= 0:
        return max_shots
    return min(max_shots, max_errors * shots / errors)


def load_costs(path: Path = COSTS_FILE) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def task_cost(costs: dict, task: Task) -> float:
    """Expected core-seconds of a task at the plan's full budget.

    Falls back to the mean over the same CSV and distance, then to the
    nearest distance of the same CSV scaled as d^3, then to a flat guess."""
    by_d = costs.get(task.stem, {})
    at_d = by_d.get(str(task.distance), {})
    if (key := repr(task.p)) in at_d:
        return at_d[key]
    if at_d:
        return sum(at_d.values()) / len(at_d)
    if by_d:
        d0 = min(by_d, key=lambda d: abs(int(d) - task.distance))
        return sum(by_d[d0].values()) / len(by_d[d0]) * (task.distance / int(d0)) ** 3
    return _FALLBACK_TASK_S * (task.distance / 15) ** 3


@dataclass
class Cost:
    tasks: int              # tasks still short of their budget
    total_tasks: int
    core_hours: float       # expected, before the safety factor
    overhead_hours: float   # wall time off the shot budget




def job_cost(job: Job, costs: dict, progress: Progress) -> Cost:
    todo = tasks = 0
    core_s = overhead_s = 0.0
    for run in job.runs:
        si = run["sinter"]
        for task in run_tasks(run):
            tasks += 1
            frac = remaining_fraction(*progress.of(task)[:2], si["max_shots"], si["max_errors"])
            if frac <= 0:
                continue
            todo += 1
            # Calibrated at the full plan budget; a scaled budget scales the cost.
            core_s += task_cost(costs, task) * job.budget_scale * frac
            overhead_s += max(_TASK_OVERHEAD_S_FLOOR, _TASK_OVERHEAD_S_D19 * (task.distance / 19) ** 3)
    return Cost(todo, tasks, core_s / 3600, overhead_s / 3600)


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@dataclass
class Limits:
    safety: float = 2.0         # multiplies the expected time
    target_hours: float = 12.0  # aim for jobs about this long
    min_cpus: int = 4
    max_cpus: int = 64
    max_mem_gb: int = 250
    max_hours: int = 48


@dataclass
class Resources:
    ncpus: int
    mem_gb: int
    walltime_h: int
    needs_hours: float          # the walltime the estimate asks for, uncapped

    @property
    def truncated(self) -> bool:
        return self.needs_hours > self.walltime_h


def split_oversized(jobs: list[Job], costs: dict, progress: Progress, limits: Limits) -> list[Job]:
    """Curve jobs too long for `target_hours` even at `max_cpus`, one job per run."""
    out = []
    for job in jobs:
        res = resources(job, job_cost(job, costs, progress), limits)
        if res.needs_hours <= limits.target_hours or len(job.runs) == 1:
            out.append(job)
            continue
        out += [Job(_run_name(job.name, run), job.budget_scale, [run]) for run in job.runs]
    return out


def worker_gb(job: Job) -> float:
    d = job.max_distance
    return _FAMILY_SIZE.get(job.family, 1.5) * (_WORKER_GB_D19 * (d / 19) ** 3 + _WORKER_GB_FLOOR)


def resources(job: Job, cost: Cost, limits: Limits) -> Resources:
    """ncpus to finish in about `target_hours` (a multiple of 4 above
    `min_cpus`, capped by `max_cpus` and by memory), memory for that many
    workers with 25% headroom, walltime in whole hours."""
    core_h = cost.core_hours * limits.safety
    ncpus = max(limits.min_cpus, 4 * math.ceil(core_h / limits.target_hours / 4))
    ncpus = min(ncpus, limits.max_cpus)
    per_worker = 1.25 * worker_gb(job)
    ncpus = max(1, min(ncpus, int((limits.max_mem_gb - _PARENT_GB) / per_worker)))
    mem_gb = math.ceil(_PARENT_GB + ncpus * per_worker)
    needs = core_h / ncpus + cost.overhead_hours * limits.safety + 0.25
    return Resources(ncpus, mem_gb, min(limits.max_hours, max(1, math.ceil(needs))), needs)


# ---------------------------------------------------------------------------
# PBS
# ---------------------------------------------------------------------------


def pbs_script(name: str, *, res: Resources, command: str, log: Path, setup: str,
               queue: str | None, account: str | None, extra: list[str],
               depend: str | None = None) -> str:
    lines = ["#!/bin/bash",
             f"#PBS -N {name}",
             f"#PBS -l select=1:ncpus={res.ncpus}:mem={res.mem_gb}gb",
             f"#PBS -l walltime={res.walltime_h:02d}:00:00",
             "#PBS -j oe",
             f"#PBS -o {log}"]
    if queue:
        lines.append(f"#PBS -q {queue}")
    if account:
        lines.append(f"#PBS -P {account}")
    if depend:
        lines.append(f"#PBS -W depend={depend}")
    lines += [f"#PBS {line}" for line in extra]
    lines += ["",
              f"cd {shlex.quote(str(_REPO))}",
              setup,
              "set -eo pipefail",
              # sinter's workers are the parallelism: keep numpy single-threaded.
              "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
              'echo "job $PBS_JOBID on $(hostname), $(date), commit '
              '$(git rev-parse --short HEAD 2>/dev/null)"',
              command, ""]
    return "\n".join(lines)


def qsub(script: Path) -> str:
    out = subprocess.run(["qsub", str(script)], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"qsub {script.name}: {out.stderr.strip()}")
    return out.stdout.strip()


def qstat_states(pbs_ids: list[str]) -> dict[str, str]:
    """{pbs id: state letter} for the jobs PBS still lists (queued, running,
    held, ...); finished jobs are absent. Empty without qstat."""
    if not pbs_ids or shutil.which("qstat") is None:
        return {}
    out = subprocess.run(["qstat", *pbs_ids], capture_output=True, text=True).stdout
    number = {i.split(".")[0]: i for i in pbs_ids}
    states = {}
    for line in out.splitlines():
        tokens = line.split()
        if len(tokens) >= 5 and (pbs_id := number.get(tokens[0].split(".")[0])):
            states[pbs_id] = tokens[4]
    return states


_DONE_STATES = {"C", "F", "X"}


def manifests(jobs_dir: Path) -> list[Path]:
    return sorted(jobs_dir.glob("*/manifest.json"))


def live_stems(jobs_dir: Path) -> dict[str, str]:
    """{CSV stem: pbs id} of every job of an earlier batch PBS still lists."""
    entries = []
    for path in manifests(jobs_dir):
        manifest = json.loads(path.read_text())
        entries += [e for e in manifest["jobs"].values() if e.get("pbs_id")]
    states = qstat_states([e["pbs_id"] for e in entries])
    return {stem: e["pbs_id"] for e in entries
            if states.get(e["pbs_id"], "F") not in _DONE_STATES for stem in e["stems"]}


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _limits(args) -> Limits:
    return Limits(safety=args.safety, target_hours=args.target_hours, min_cpus=args.min_cpus,
                  max_cpus=args.max_cpus, max_mem_gb=args.max_mem_gb, max_hours=args.max_hours)


def _planned(args) -> tuple[list[Job], dict, Progress, str]:
    results_dir = str(Path(args.results_dir).resolve())
    jobs = plan_jobs(args.targets, split="run" if args.split == "run" else "curve",
                     budget_scale=args.budget_scale, results_dir=results_dir)
    costs = load_costs()
    if not costs:
        print(f"warning: no {COSTS_FILE.name}; every task gets a flat cost guess", file=sys.stderr)
    progress = Progress(results_dir)
    if args.split == "auto":
        jobs = split_oversized(jobs, costs, progress, _limits(args))
    return jobs, costs, progress, results_dir


def _row(name: str, cost: Cost, res: Resources | None, note: str = "") -> str:
    if res is None:
        return f"{name:34s} {cost.tasks:4d}/{cost.total_tasks:<4d} {'':40s} {note}"
    flag = " (walltime capped: resubmit to finish)" if res.truncated else ""
    return (f"{name:34s} {cost.tasks:4d}/{cost.total_tasks:<4d} {cost.core_hours:9.1f} "
            f"{res.ncpus:5d} {res.mem_gb:6d} {res.walltime_h:5d}h {res.needs_hours:7.1f}h"
            f"  {note}{flag}")


_HEADER = (f"{'job':34s} {'todo/tasks':10s} {'core-h':>8s} {'ncpus':>5s} {'mem GB':>6s} "
           f"{'wall':>6s} {'est.':>8s}")


def cmd_plan(args) -> None:
    jobs, costs, progress, _ = _planned(args)
    limits = _limits(args)
    print(_HEADER)
    core_h = 0.0
    for job in jobs:
        cost = job_cost(job, costs, progress)
        if cost.tasks == 0:
            print(_row(job.name, cost, None, "complete"))
            continue
        core_h += cost.core_hours
        print(_row(job.name, cost, resources(job, cost, limits)))
    print(f"\n{len(jobs)} job(s), ~{core_h:.0f} core-hours expected "
          f"(~{core_h * limits.safety:.0f} with the x{limits.safety:g} safety factor)")


def _check_decoder(allow_slow: bool) -> None:
    from dizx.sinter_decoders import SUPPORTS_EDGE_REWEIGHTS
    if SUPPORTS_EDGE_REWEIGHTS or allow_slow:
        return
    raise SystemExit(
        "This environment has stock pymatching: correlated heralded decoding would run ~40x slower "
        "than the cost model assumes. Install the edge-reweighting fork (see README) into the "
        "environment the jobs use, or pass --allow-slow-decoder.")


def cmd_submit(args) -> None:
    _check_decoder(args.allow_slow_decoder)
    if not args.dry_run and shutil.which("qsub") is None:
        raise SystemExit("qsub not found: run this on the cluster's login node, or pass --dry-run")
    jobs, costs, progress, results_dir = _planned(args)
    limits = _limits(args)
    jobs_dir = Path(args.jobs_dir).resolve()
    live = {} if args.dry_run else live_stems(jobs_dir)
    batch = jobs_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    (batch / "logs").mkdir(parents=True)
    commit = subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    manifest = {"created": datetime.datetime.now().isoformat(timespec="seconds"),
                "commit": commit, "targets": args.targets, "split": args.split,
                "budget_scale": args.budget_scale, "results_dir": results_dir, "jobs": {}}
    pbs = {"setup": args.setup, "queue": args.queue, "account": args.account, "extra": args.pbs}

    print(_HEADER)
    submitted = []
    for job in jobs:
        cost = job_cost(job, costs, progress)
        if cost.tasks == 0:
            print(_row(job.name, cost, None, "complete, skipped"))
            continue
        if busy := sorted({live[s] for s in job.stems if s in live}):
            print(_row(job.name, cost, None, f"still queued/running as {', '.join(busy)}, skipped"))
            continue
        res = resources(job, cost, limits)
        spec = batch / f"{job.name}.json"
        spec.write_text(json.dumps({"name": job.name, "budget_scale": job.budget_scale,
                                    "results_dir": results_dir, "runs": job.runs}, indent=1))
        script = batch / f"{job.name}.pbs"
        command = (f"python experiments/cluster.py run {shlex.quote(str(spec))} "
                   f"--workers {res.ncpus}")
        script.write_text(pbs_script(job.name, res=res, command=command,
                                     log=batch / "logs" / f"{job.name}.log", **pbs))
        pbs_id = None if args.dry_run else qsub(script)
        if pbs_id:
            submitted.append(pbs_id)
        manifest["jobs"][job.name] = {"pbs_id": pbs_id, "spec": str(spec), "stems": sorted(job.stems),
                                      "ncpus": res.ncpus, "mem_gb": res.mem_gb,
                                      "walltime_h": res.walltime_h,
                                      "core_hours": round(cost.core_hours, 2)}
        print(_row(job.name, cost, res, pbs_id or ""))

    if args.plot and (submitted or args.dry_run):
        command = (f"python experiments/reproduce.py {' '.join(map(shlex.quote, args.targets))} "
                   f"--plot-only --results-dir {shlex.quote(results_dir)} "
                   f"--out-dir {shlex.quote(str(Path(results_dir) / 'figures'))}")
        script = batch / "plot.pbs"
        depend = f"afterany:{':'.join(submitted)}" if submitted else None
        script.write_text(pbs_script("dizx-plot", res=Resources(1, 8, 1, 1), command=command,
                                     log=batch / "logs" / "plot.log", depend=depend, **pbs))
        plot_id = None if args.dry_run else qsub(script)
        manifest["plot"] = plot_id
        print(f"plot job {plot_id or script} after the batch")

    (batch / "manifest.json").write_text(json.dumps(manifest, indent=1))
    n = len(manifest["jobs"])
    print(f"\n{n} job(s) {'written' if args.dry_run else 'submitted'} under {batch}")
    if not args.dry_run and n:
        print("Progress: python experiments/cluster.py status")


def cmd_status(args) -> None:
    paths = manifests(Path(args.jobs_dir).resolve())
    if args.batch:
        paths = [p for p in paths if p.parent.name == args.batch]
    if not paths:
        raise SystemExit(f"no batch under {args.jobs_dir}")
    path = paths[-1]
    manifest = json.loads(path.read_text())
    entries = manifest["jobs"]
    states = qstat_states([e["pbs_id"] for e in entries.values() if e.get("pbs_id")])
    costs = load_costs()
    progress = Progress(manifest["results_dir"])
    print(f"batch {path.parent.name} (commit {manifest['commit'][:8]}), "
          f"results in {manifest['results_dir']}\n")
    print(f"{'job':34s} {'pbs id':16s} {'state':5s} {'todo/tasks':10s} {'core-h left':>11s}")
    left = 0.0
    todo = 0
    for name, entry in entries.items():
        spec = json.loads(Path(entry["spec"]).read_text())
        cost = job_cost(Job(name, spec["budget_scale"], spec["runs"]), costs, progress)
        state = states.get(entry["pbs_id"], "done" if entry.get("pbs_id") else "-")
        left += cost.core_hours
        todo += cost.tasks
        print(f"{name:34s} {entry.get('pbs_id') or '-':16s} {state:5s} "
              f"{cost.tasks:4d}/{cost.total_tasks:<5d} {cost.core_hours:11.1f}")
    unfinished = [n for n, e in entries.items() if states.get(e["pbs_id"]) is None]
    print(f"\n{todo} task(s) short of budget, ~{left:.0f} core-hours left")
    if todo and unfinished:
        print("Jobs that ended with work left ran out of walltime (or failed: see "
              f"{path.parent / 'logs'}). Resubmit the same targets to finish them.")


def cmd_run(args) -> None:
    """A job's body: collect each of its runs in turn."""
    from dizx.sinter_decoders import SUPPORTS_EDGE_REWEIGHTS
    from run_threshold_experiment import run_threshold
    spec = json.loads(Path(args.spec).read_text())
    if not SUPPORTS_EDGE_REWEIGHTS:
        print("warning: stock pymatching, correlated decoding will be slow", flush=True)
    os.chdir(_REPO)   # the circuit cache is relative to the repo root
    workers = args.workers or (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                               else os.cpu_count())
    for run in spec["runs"]:
        run = copy.deepcopy(run)
        run["sinter"]["num_workers"] = workers
        print(f"\n===== {run_stem(run)} ({workers} workers) =====", flush=True)
        run_threshold(run)


def _seconds_per_shot(progress: Progress, task: Task) -> float | None:
    """Mean worker seconds per shot over a CSV's tasks at the task's distance."""
    rows = [(s, sec) for (d, _), (s, _, sec) in progress.csv(task.stem).items() if d == task.distance]
    shots = sum(s for s, _ in rows)
    return sum(sec for _, sec in rows) / shots if shots else None


def cmd_calibrate(args) -> None:
    """Expected core-seconds of every task of the full plan, from a results
    directory that has collected it (shots, errors and worker seconds per task)."""
    source = Path(args.source)
    progress = Progress(source)
    out: dict[str, dict[str, dict[str, float]]] = {}
    found = estimated = total = 0
    for job in plan_jobs(["all"], split="run", budget_scale=1.0, results_dir=str(source)):
        for run in job.runs:
            si = run["sinter"]
            for task in run_tasks(run):
                total += 1
                shots, errors, seconds = progress.of(task)
                if shots:
                    found += 1
                    cost = expected_shots(shots, errors, si["max_shots"], si["max_errors"]) * seconds / shots
                elif rate := _seconds_per_shot(progress, task):
                    # Not collected there (e.g. outside its refine window):
                    # the full shot budget at that CSV's rate at this distance.
                    estimated += 1
                    cost = si["max_shots"] * rate
                else:
                    continue
                out.setdefault(task.stem, {}).setdefault(str(task.distance), {})[repr(task.p)] = \
                    float(f"{cost:.3g}")
    COSTS_FILE.write_text(json.dumps(out, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"{total} tasks: {found} measured, {estimated} priced at full budget from their CSV's "
          f"rate, {total - found - estimated} left to the fallback; {source} -> {COSTS_FILE}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    planning = argparse.ArgumentParser(add_help=False)
    planning.add_argument("targets", nargs="+", help="`all`, figure keys, curve ids or globs.")
    planning.add_argument("--split", choices=("auto", "curve", "run"), default="auto",
                          help="One job per curve, per run (one sinter CSV), or per curve unless "
                               "that exceeds --target-hours at --max-cpus (default: %(default)s).")
    planning.add_argument("--budget-scale", type=float, default=1.0,
                          help="Multiply every point's max_shots / max_errors.")
    planning.add_argument("--results-dir", default=paper.RESULTS_DIR,
                          help="Root of the sinter CSVs (default: %(default)s/).")
    defaults = Limits()
    planning.add_argument("--safety", type=float, default=defaults.safety,
                          help="Multiply the expected time (default: %(default)s).")
    planning.add_argument("--target-hours", type=float, default=defaults.target_hours,
                          help="Size ncpus so a job takes about this long (default: %(default)s).")
    planning.add_argument("--min-cpus", type=int, default=defaults.min_cpus)
    planning.add_argument("--max-cpus", type=int, default=defaults.max_cpus,
                          help="Per job, at most one node (default: %(default)s).")
    planning.add_argument("--max-mem-gb", type=int, default=defaults.max_mem_gb,
                          help="Per job (default: %(default)s).")
    planning.add_argument("--max-hours", type=int, default=defaults.max_hours,
                          help="Walltime cap (default: %(default)s).")

    sub.add_parser("plan", parents=[planning], help="Print the jobs and their resources.")
    s = sub.add_parser("submit", parents=[planning], help="Write the job scripts and qsub them.")
    s.add_argument("--jobs-dir", default=str(JOBS_DIR), help="Default: %(default)s.")
    s.add_argument("--setup", default=os.environ.get("DIZX_PBS_SETUP", "source .venv/bin/activate"),
                   help="Shell lines that put the dizx environment's python on PATH "
                        "(default: %(default)r).")
    s.add_argument("--queue", default=os.environ.get("DIZX_PBS_QUEUE"))
    s.add_argument("--account", default=os.environ.get("DIZX_PBS_ACCOUNT"),
                   help="Project to charge (#PBS -P).")
    s.add_argument("--pbs", action="append", default=[], metavar="LINE",
                   help="Extra '#PBS LINE' directive, e.g. --pbs '-m ae'. Repeatable.")
    s.add_argument("--plot", action="store_true",
                   help="Also submit a job that draws the figures once every job has ended.")
    s.add_argument("--dry-run", action="store_true", help="Write the scripts without submitting.")
    s.add_argument("--allow-slow-decoder", action="store_true",
                   help="Submit even without the edge-reweighting pymatching fork.")

    st = sub.add_parser("status", help="Progress of the latest (or --batch) batch.")
    st.add_argument("--jobs-dir", default=str(JOBS_DIR))
    st.add_argument("--batch", help="Batch directory name (a timestamp).")

    r = sub.add_parser("run", help="Job body (called from the PBS scripts).")
    r.add_argument("spec", help="The job's .json spec.")
    r.add_argument("--workers", type=int, help="Default: the CPUs this process may use.")

    c = sub.add_parser("calibrate", help=f"Rebuild {COSTS_FILE.name} from collected results.")
    c.add_argument("source", help="A results directory holding sinter/*.csv for the paper plan.")

    args = p.parse_args(argv)
    {"plan": cmd_plan, "submit": cmd_submit, "status": cmd_status, "run": cmd_run,
     "calibrate": cmd_calibrate}[args.command](args)


if __name__ == "__main__":
    main()
