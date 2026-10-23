"""Compare dizx's circuits with the previous pipeline's (DISQEC).

For each builder, at each distance and observable, this compares:
- measurement, detector and observable counts;
- operation counts per tag (type, operation, gate);
- graphlike distance under circuit noise, and under circuit noise plus
  resource-state Pauli noise.

The old circuits come from DISQEC's last commit, built in DISQEC's own
environment: ``git archive HEAD src`` goes to a temporary directory that is put
first on ``PYTHONPATH``, so uncommitted work in DISQEC isn't used. They are
dumped as .stim files, which are kept for later runs.

    poetry run python scripts/compare_with_old.py                 # d = 3, 5
    poetry run python scripts/compare_with_old.py -d 3 -c bell_meas_fusion
    poetry run python scripts/compare_with_old.py --old-repo ../DISQEC --dump-dir /tmp/old

Runs one process at a time, under ``nice``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

# case -> (new builder, old builder, old builder's kwargs, observables)
SURGERY_OBS = ("ZZ", "XX")
MEMORY_OBS = ("X", "Z")
LINEAR = dict(unfuse_time_edges=True)
CASES = {
    "circuit": ("surgery_circuit", "build_surgery_circuit", {}, SURGERY_OBS),
    "bell": ("surgery_bell_prep_fusion", "build_surgery_bell_prep_fusion", {}, SURGERY_OBS),
    "bell_zigzag": ("surgery_bell_prep_fusion_zigzag", "build_surgery_bell_prep_fusion",
                    dict(interface="zigzag"), SURGERY_OBS),
    "bell_meas_fusion": ("surgery_bell_meas_fusion", "build_surgery_fusion_resource_split", {},
                         SURGERY_OBS),
    "bell_meas_fusion_zigzag": ("surgery_bell_meas_fusion_zigzag",
                                "build_surgery_fusion_resource_split", dict(interface="zigzag"),
                                SURGERY_OBS),
    "linear_k1": ("surgery_hybrid_linear_k1", "build_surgery_hybrid_linear_long",
                  dict(max_ancillas_per_resource=1, **LINEAR), SURGERY_OBS),
    "linear_k2": ("surgery_hybrid_linear_k2", "build_surgery_hybrid_linear_long",
                  dict(max_ancillas_per_resource=2, **LINEAR), SURGERY_OBS),
    "linear_kinf": ("surgery_hybrid_linear", "build_surgery_hybrid_linear_long",
                    dict(max_ancillas_per_resource=100, **LINEAR), SURGERY_OBS),
    "memory_circuit": ("memory_circuit", "build_memory_circuit", {}, MEMORY_OBS),
    "memory_cluster": ("memory_cluster", "build_memory_cluster", {}, MEMORY_OBS),
    "memory_fusion": ("memory_fusion", "build_memory_fusion", {}, MEMORY_OBS),
}
MERGE = "ZZ"
P = 1e-3
# Tags the previous pipeline named differently. No noise config uses the old name.
OLD_TAGS = {("graph", "boundary_measurement"): ("graph", "single_qubit_measurement")}
# Count differences that are expected, as long as the distances agree.
KNOWN = {"memory_fusion": "one extra readout photon per split output spider (memory.line_pattern)"}


def _kwargs(case: str, d: int, obs: str, old: bool = False) -> dict:
    """Builder arguments; the old builders also take the variant's options."""
    _, _, extra, observables = CASES[case]
    kwargs = dict(distance=d, n_cycles=d, observable=obs, **(extra if old else {}))
    if observables is SURGERY_OBS:
        kwargs["merge_type"] = MERGE
    return kwargs


def _stem(case: str, d: int, obs: str) -> str:
    return f"{case}_d{d}_obs{obs}"


# ---------------------------------------------------------------------------
# Old side (runs in DISQEC's environment; imports only disqec)
# ---------------------------------------------------------------------------

def dump(cases: list[str], distances: list[int], out: Path) -> None:
    from disqec import experiments

    out.mkdir(parents=True, exist_ok=True)
    for case in cases:
        for d in distances:
            for obs in CASES[case][3]:
                path = out / f"{_stem(case, d, obs)}.stim"
                if path.exists():
                    continue
                build = getattr(experiments, CASES[case][1])(**_kwargs(case, d, obs, old=True))
                path.write_text(str(build.circuit))
                print(f"dumped {path.name}", flush=True)


def ensure_dumped(cases: list[str], distances: list[int], out: Path, old_repo: Path) -> None:
    missing = sorted({c for c in cases for d in distances for o in CASES[c][3]
                      if not (out / f"{_stem(c, d, o)}.stim").exists()})
    if not missing:
        return
    with tempfile.TemporaryDirectory() as tmp:
        archive = subprocess.run(["git", "-C", str(old_repo), "archive", "HEAD", "src"],
                                 check=True, capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        env = dict(os.environ, PYTHONPATH=str(Path(tmp, "src")), PYTHONHASHSEED="0")
        env.pop("VIRTUAL_ENV", None)
        cmd = ["nice", "-n", "10", "poetry", "run", "python", str(Path(__file__).resolve()), "_dump",
               "--dump-dir", str(out.resolve()), "-d", *map(str, distances),
               "-c", *missing]
        subprocess.run(cmd, cwd=old_repo, env=env, check=True)


# ---------------------------------------------------------------------------
# Comparison (dizx)
# ---------------------------------------------------------------------------

def _load(path: Path):
    import stim
    return stim.Circuit(path.read_text())


def tag_counts(circuit) -> Counter:
    from dizx.noise_models import parse_op_tag

    counts = Counter()
    for inst in circuit.flattened():
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "TICK", "SHIFT_COORDS"):
            continue
        tag = parse_op_tag(inst.tag) if inst.tag else {}
        kind = OLD_TAGS.get((tag.get("type"), tag.get("operation")), (tag.get("type"), tag.get("operation")))
        n = len(inst.targets_copy()) // (2 if inst.name in ("CX", "CZ") else 1)
        counts[(*kind, inst.name)] += n
    return counts


def noise_models() -> dict[str, list[dict]]:
    sys.path.insert(0, str(_REPO / "experiments"))
    import paper

    circuit = paper._circuit_noise()
    graph = [{"tag": ["graph", "reset"], "kind": "RESET_FLIP", "probability": "p_l"},
             {"tag": ["graph", "fusion_measurement"], "kind": "MEASUREMENT_FLIP", "probability": "p_l"}]
    return {"circuit": circuit, "circuit+graph": circuit + graph}


def distance(circuit, channels: list[dict]) -> int | None:
    """Graphlike distance; None when the noise touches nothing in the circuit."""
    from dizx.noise_models import NoiseSpec, apply_tagged_noise_model

    model: dict = {}
    for ch in channels:
        model.setdefault(tuple(ch["tag"]), []).append(NoiseSpec(instruction=ch["kind"], probability=P))
    noisy = apply_tagged_noise_model(circuit, noise_by_type=model)
    dem = noisy.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True,
                                     ignore_decomposition_failures=True, allow_gauge_detectors=True)
    if dem.num_errors == 0:
        return None
    return len(dem.shortest_graphlike_error())


def compare(cases: list[str], distances: list[int], old_dir: Path) -> bool:
    from dizx.builders import BUILDERS

    models = noise_models()
    ok = True
    for case in cases:
        for d in distances:
            for obs in CASES[case][3]:
                old = _load(old_dir / f"{_stem(case, d, obs)}.stim")
                new = BUILDERS[CASES[case][0]](**_kwargs(case, d, obs)).circuit
                rows = []
                counts = lambda c: (c.num_measurements, c.num_detectors, c.num_observables)
                if counts(old) != counts(new):
                    rows.append(f"  meas/det/obs  old {counts(old)}  new {counts(new)}")
                to, tn = tag_counts(old), tag_counts(new)
                for key in sorted(set(to) | set(tn), key=str):
                    if to[key] != tn[key] and key[1] != "idle":
                        rows.append(f"  {'/'.join(map(str, key)):45s} old {to[key]:6d}  new {tn[key]:6d}")
                dists = {name: (distance(old, ch), distance(new, ch)) for name, ch in models.items()}
                for name, (a, b) in dists.items():
                    if a != b:
                        rows.append(f"  distance ({name})  old {a}  new {b}")
                known = (rows and case in KNOWN
                         and all(a == b for a, b in dists.values()))
                status = "same" if not rows else "known" if known else "DIFFERS"
                ok &= not rows or known
                if known:
                    rows.insert(0, f"  ({KNOWN[case]})")
                summary = ", ".join(f"{n} {b}" for n, (_, b) in dists.items())
                print(f"{_stem(case, d, obs):32s} {status:8s} distance: {summary}", flush=True)
                for row in rows:
                    print(row)
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="compare", choices=["compare", "_dump"])
    ap.add_argument("-d", "--distances", type=int, nargs="+", default=[3, 5])
    ap.add_argument("-c", "--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--old-repo", type=Path, default=_REPO.parent / "DISQEC")
    ap.add_argument("--dump-dir", type=Path, default=_REPO / "results" / "old_circuits")
    args = ap.parse_args(argv)
    if args.mode == "_dump":
        dump(args.cases, args.distances, args.dump_dir)
        return 0
    try:
        os.nice(10)
    except OSError:
        pass
    ensure_dumped(args.cases, args.distances, args.dump_dir, args.old_repo)
    return 0 if compare(args.cases, args.distances, args.dump_dir) else 1


if __name__ == "__main__":
    sys.exit(main())
