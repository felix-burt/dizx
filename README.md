# dizx

Distributed quantum error correction from syndrome extraction ZX diagrams (SEDs).

dizx builds fault-tolerant experiments as ZX diagrams, rewrites them with web-preserving transformations into distributed realisations (circuits, resource states and fusions), and extracts tagged Stim circuits that are sampled with sinter and decoded with (heralded, correlated) PyMatching.

## Install

```bash
poetry install          # Python 3.12
poetry run pytest -q
```

`poetry install` takes `pymatching` from the edge-reweighting fork ([felix-burt/PyMatching](https://github.com/felix-burt/PyMatching), prebuilt wheels from release `v2.4.0-reweight1` for Linux x86_64 and macOS arm64), which makes the heralded-erasure decoder much faster; `poetry install -E stock-pymatching` takes PyPI's instead, which also works. `dizx.sinter_decoders.SUPPORTS_EDGE_REWEIGHTS` reports whether the fork is present.

## Pipeline

| Module | Role |
|---|---|
| `code` | stabiliser codes (`rotated_surface_code`, …) |
| `zx_diagram` | `SEDiagram`: SEDs and their Pauli webs (detectors, stabilisers, logicals) |
| `transformations` | web-preserving rewrites; `SEDEditor` applies them in place |
| `realisation` | `RealisedSED` (circuit-based, measurement-based or hybrid) and `QECExperiment` |
| `schedules` | gate schedules for circuit-based regions |
| `circuit_extractor` | `to_stim`: tagged Stim circuits |
| `surgery`, `memory`, `builders` | lattice-surgery and memory experiments, registered in `builders.BUILDERS` |
| `noise_models` | tag-based noise (`apply_tagged_noise_model`) |
| `erasure_matching`, `sinter_decoders` | heralded-erasure decoding for sinter |

The maths is in `docs/`:
- `definitions.md` and `sed_definitions.md` cover SEDs and webs;
- `transformations.md` covers the rewrite rules and the editor;
- `realisations.md` covers realisations;
- `pauli_webs_f2.md` covers how webs are found with F2 linear algebra;
- `crg_threshold_fit.md` and `leading_order_erasure.md` cover the analysis behind the figures.

`examples/` has notebooks for each stage.

## Builders

Every builder takes `(distance, n_cycles, merge_type, observable)` and returns a build whose `.circuit` is a tagged, noiseless Stim circuit. Each variant (seam shape, chain truncation) is its own builder, so the name alone identifies the circuit. `BUILDERS` keys each one by its function name without `build_`.

| Name | Experiment |
|---|---|
| `surgery_circuit` | circuit-based two-patch ZZ/XX surgery |
| `surgery_unrotated_circuit` | the same with unrotated surface code patches |
| `surgery_bell_prep_fusion`, `_zigzag` | patches joined by Bell pairs from matter–photon emission, across a straight or zigzag seam |
| `surgery_unrotated_bell_prep_fusion` | Bell-pair surgery with unrotated patches, across a straight seam |
| `surgery_bell_meas_fusion`, `_zigzag` | patches joined by pairs of Bell-pair resource states and fusions |
| `surgery_hybrid_linear_3dchain` | hybrid surgery with untruncated linear resource states |
| `surgery_hybrid_linear_5chain`, `_8chain` | the same with every first or second time edge cut |
| `surgery_hybrid_linear_looped` | untruncated, with a looped clusterised region |
| `memory_circuit` / `_cluster` / `_fusion` | single-patch memory, circuit-based, one cluster state, or fused line states |
| `memory_unrotated_circuit` | circuit-based memory on an unrotated surface code patch |

```python
from dizx.builders import BUILDERS
c = BUILDERS["surgery_bell_prep_fusion_zigzag"](5, 5, "ZZ", "ZZ").circuit
```

## Reproducing the paper

`experiments/paper.py` holds the paper's whole sampling plan:
- the five resource-state families;
- the noise models;
- every curve of every figure, with its distances, probability windows and shot budgets.

The crg curves also include two refinement rounds, which added d = 17 and 19 over each point's fitted scaling window. They are frozen in `experiments/crg_refinement.json`, which `paper.py` loads, so they are part of the fixed plan. `experiments/refine_crg.py` planned them (`status` shows each point's fit; `next` freezes a new round). Reproducing the paper doesn't need it. The threshold fit and its error bars are described in `docs/crg_threshold_fit.md`.

`experiments/reproduce.py` collects data and draws the figures.

```bash
poetry run python experiments/reproduce.py list                     # figures and curve ids
poetry run python experiments/reproduce.py crg_prep/bell/ZZ         # one curve
poetry run python experiments/reproduce.py crg_prep_ZZ              # all curves of one figure
poetry run python experiments/reproduce.py all --workers 8          # everything
poetry run python experiments/reproduce.py all --budget-scale 0.01  # quick low-statistics preview
poetry run python experiments/reproduce.py all --plot-only --pdf    # redraw from existing data
```

| Figure key | Paper figure |
|---|---|
| `master_threshold` | pure-erasure thresholds, all families, ZZ and XX |
| `crg_local_ZZ`, `crg_local_XX` | correctable regime: circuit noise vs erasure |
| `crg_prep_ZZ`, `crg_prep_XX` | correctable regime: resource-state noise vs erasure |
| `lambda_grid` | sub-threshold Λ vs photon loss |

Data goes to `results/sinter/*.csv`. Figures go to `figures/paper/<paper file name>.tex`, each with a `_data.csv`.

Collection resumes. Every task is keyed by sinter's `strong_id` (circuit, noise and decoder), so an interrupted run, a repeated run or a curve shared by two figures never samples a shot twice. A low-budget preview is topped up by a later full run. Circuits are cached in `results/circuits/` as `<builder>_d<d>_merge<merge>_obs<obs>.stim`, e.g. `surgery_bell_prep_fusion_d5_mergeZZ_obsZZ.stim` (memory circuits have no merge segment). The name doesn't track the code, so delete a builder's files after changing what it builds.

Defaults are deliberately light: 2 workers, under `nice`. Pass `--workers` and `--no-nice` on a machine to spare.

### On a PBS cluster

`experiments/cluster.py` submits the same collection as separate PBS jobs. It takes the same targets as `reproduce.py`.

```bash
python experiments/cluster.py plan all --results-dir results/cluster      # jobs, ncpus, memory, walltime
python experiments/cluster.py submit all --results-dir results/cluster --plot \
    --setup 'module load tools/prod Python/3.12.3-GCCcore-13.3.0; source .venv/bin/activate'
python experiments/cluster.py status                                      # progress, PBS state
```

How jobs are split:
- By default each job is one curve. A curve that would run past `--target-hours` (12) on `--max-cpus` (64) is split into one job per sinter CSV. At full budget this applies to the crg_local curves.
- The full plan, including the crg refinement rounds, comes to 164 jobs and about 14,000 expected core-hours. The planner overestimated the refinement rounds by about 1.5×.
- No two jobs write the same CSV. A crg point's refinement runs share its CSV, so they stay in one job.

How resources are sized:
- `experiments/cluster_costs.json` holds the expected core-seconds of every task, measured from the DISQEC collection behind the paper. Rebuild it with `cluster.py calibrate <results dir>`.
- Work already in `--results-dir` is subtracted.
- `ncpus` aims for `--target-hours`. Memory comes from the circuit size at the job's largest distance. Walltime is scaled by `--safety` (×2).
- `--max-mem-gb`, `--max-hours`, `--queue`, `--account` (`#PBS -P`) and `--pbs '<directive>'` adapt the jobs to your cluster. `DIZX_PBS_SETUP`, `DIZX_PBS_QUEUE` and `DIZX_PBS_ACCOUNT` set defaults.

Resubmitting:
- Collection resumes, so a job that hits its walltime only loses its current batch. Resubmit the same targets to finish.
- On resubmission, complete jobs are skipped, and jobs still queued or running from an earlier batch are left alone.
- Each batch keeps its scripts, job specs, manifest and logs under `results/pbs/<timestamp>/`.

Before submitting:
- `submit` refuses to run without the edge-reweighting pymatching fork, which the cost model assumes.
- Jobs run the code checked out when they start, not when they were submitted, so don't change the working tree while a batch is queued.
- `--plot` adds a job that runs after the batch and draws the figures into `<results-dir>/figures/`. To draw them yourself instead: `reproduce.py all --plot-only --results-dir results/cluster`.

## Running other experiments

The runners in `experiments/` take a YAML config with the same structure as the dicts in `paper.py` (`paper.erasure_config(...)` and friends return complete examples):

```yaml
name: erasure_bell_corr_obsZZ
builder: {name: surgery_bell_prep_fusion}
resource_state: bell
results_dir: results
sweep: {distances: [5, 7], merge_types: [ZZ], observables: [ZZ], min: 0.2, max: 0.4, step: 0.02}
noise_model:
  - {tag: [graph, fusion_measurement], kind: HERALDED_ERASE, scale: 1.0}
sinter: {decoder: pymatching_heralded, max_shots: 100000, max_errors: 1000, num_workers: 2,
         pm_decompose: true, pm_enable_correlations: true}
```

```bash
poetry run python experiments/run_threshold_experiment.py my_config.yaml   # threshold sweep
poetry run python experiments/run_correctable_regime.py my_config.yaml     # erasure threshold vs local noise
poetry run python experiments/run_subthreshold.py my_config.yaml           # Λ vs loss
```

## Parity with the previous pipeline

The circuits reproduce those of the previous codebase (DISQEC), checked with:

```bash
poetry run python scripts/compare_with_old.py    # d = 3, 5, every builder
```

The script builds the old circuits from DISQEC's last commit in DISQEC's own environment. It then compares measurement, detector and observable counts, operation counts per tag, and graphlike distance under circuit and resource-state noise.

Two differences are expected:
- the old graph `boundary_measurement` tag is now `single_qubit_measurement`;
- `memory_fusion` has one extra readout photon per split output spider.

Neither changes a distance.
