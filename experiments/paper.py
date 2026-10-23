"""Every experiment behind the paper's figures, as data.

The paper (DLEFBQC, section 7) has four figures, all for the same five
resource-state families, a ZZ lattice-surgery merge and correlated
pymatching:

  master_threshold   LER vs fusion erasure rate, d = 11-19, one cell per family
                     (both observables).
  crg_local_{ZZ,XX}  erasure threshold vs circuit noise p_c (correctable regime).
  crg_prep_{ZZ,XX}   erasure threshold vs resource-state error p_r.
  lambda_grid        suppression factor Lambda (d = 5 -> 7) vs fusion erasure
                     rate, one row per local-noise level, ZZ | XX.

Each figure is a list of curves; each curve is a list of sinter runs (ordinary
runner configs with every noise level fixed). Collection goes through
`run_threshold`, whose CSVs resume by strong_id (circuit + noise + decoder),
so a curve shared by two figures, or a re-run, never collects a shot twice.

The sampling plan is fixed, not adaptive:

  * erasure sweeps: the paper's grids, step 0.01;
  * crg points: a window of `CRG_WINDOW_POINTS` erasure rates at
    `CRG_WINDOW_STEP`, centred on the threshold the paper reports
    (`CRG_THRESHOLDS`, from its _data.csv files), at d = 11, 13, 15. A local
    noise level the paper lists with no threshold (the correctable region has
    closed) gets only `CRG_CLOSING_PROBE`, which the prep figure uses to stop
    its dashed extrapolation;
  * Lambda points: exactly the (boost level, loss rate) points the paper
    plots (`LAMBDA_POINTS`), at d = 5, 7, each to `LAMBDA_BUDGET`.

`reproduce.py` is the command-line front end.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from dizx import result_names

# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Family:
    key: str            # resource_state: names the result files
    label: str          # legend label (LaTeX)
    color: str          # xcolor (dvipsnames)
    mark: str           # pgfplots mark
    builder: str        # dizx.builders.BUILDERS key


FAMILIES: dict[str, Family] = {f.key: f for f in [
    Family("bell", r"\textsc{Bell}", "blue", "diamond*", "surgery_bell_prep_fusion"),
    Family("bell_zigzag", r"\textsc{BellZigZag}", "red", "diamond", "surgery_bell_prep_fusion_zigzag"),
    Family("linear_k1", r"\textsc{$5$-Chain}", "OliveGreen", "*", "surgery_hybrid_linear_5chain"),
    Family("linear_k2", r"\textsc{$8$-Chain}", "orange", "square*", "surgery_hybrid_linear_8chain"),
    Family("linear_kinf", r"$3d$-\textsc{Chain}", "Purple", "triangle*", "surgery_hybrid_linear_3dchain"),
]}
OBSERVABLES = ("ZZ", "XX")
MERGE = "ZZ"

# ---------------------------------------------------------------------------
# Noise models. Placeholders: p_l (local noise, fixed per run), p_nl (the
# swept erasure rate), p_e (the swept photon loss rate, mapped to a fusion
# erasure rate at boost level n_boost).
# ---------------------------------------------------------------------------


def _ch(tag: tuple[str, ...], kind: str, **prob) -> dict:
    return {"tag": list(tag), "kind": kind, **prob}


def _circuit_noise() -> list[dict]:
    """Circuit-level depolarising noise at p_l on the code patch and its
    fusion interface."""
    return [
        _ch(("circuit", "reset"), "RESET_FLIP", probability="p_l"),
        _ch(("circuit", "single_qubit_gate"), "DEPOLARIZE1", probability="p_l"),
        _ch(("circuit", "two_qubit_gate"), "DEPOLARIZE2", probability="p_l"),
        _ch(("circuit", "single_qubit_measurement"), "MEASUREMENT_FLIP", probability="p_l"),
        _ch(("circuit-graph", "fusion_gate"), "DEPOLARIZE2", probability="p_l"),
        _ch(("circuit-graph", "fusion_measurement"), "MEASUREMENT_FLIP", probability="p_l"),
    ]


def erasure_noise() -> list[dict]:
    """Heralded fusion erasure only, at the swept rate."""
    return [_ch(("graph", "fusion_measurement"), "HERALDED_ERASE", scale=1.0)]


def crg_local_noise() -> list[dict]:
    return _circuit_noise() + [_ch(("graph", "fusion_measurement"), "HERALDED_ERASE", probability="p_nl")]


def crg_prep_noise() -> list[dict]:
    """Resource-state error p_r on the matter qubits plus fusion outcome flips.

    The paper's XX prep curves were collected with the earlier model, a Z_ERROR
    on each prepared matter qubit ((graph, state_prep, matter)) rather than a
    RESET_FLIP on each matter reset; the two are effectively equivalent."""
    return [_ch(("graph", "reset", "matter"), "RESET_FLIP", probability="p_l"),
            _ch(("graph", "fusion_measurement"), "MEASUREMENT_FLIP", probability="p_l"),
            _ch(("graph", "fusion_measurement"), "HERALDED_ERASE", probability="p_nl")]


def lambda_noise(n_boost: int) -> list[dict]:
    """Circuit noise and resource-state error, both p_l, plus photon loss."""
    return _circuit_noise() + [
        _ch(("graph", "reset", "matter"), "RESET_FLIP", probability="p_l"),
        _ch(("graph", "fusion_measurement"), "MEASUREMENT_FLIP", probability="p_l"),
        _ch(("graph", "fusion_measurement"), "HERALDED_ERASE", probability="p_e", n_boost=n_boost),
    ]


# ---------------------------------------------------------------------------
# Sampling plan
# ---------------------------------------------------------------------------

THRESHOLD_DISTANCES = [11, 13, 15, 17, 19]
CRG_DISTANCES = [11, 13, 15]
LAMBDA_DISTANCES = [5, 7]

# Per-point sinter budgets (max_shots, max_errors).
THRESHOLD_BUDGET = (1_000_000, 100_000)
LAMBDA_BUDGET = (10_000_000_000, 2_000)

# Erasure sweeps (min, max) per family, step 0.01.
ERASURE_RANGES = {"bell": (0.45, 0.55), "bell_zigzag": (0.45, 0.55), "linear_k1": (0.25, 0.35),
                  "linear_k2": (0.35, 0.45), "linear_kinf": (0.45, 0.55)}
ERASURE_STEP = 0.01

# Local noise levels each crg curve sweeps.
_PC = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]
CRG_LOCAL_PROBS = {
    "crg_local": {fam: _PC for fam in FAMILIES},
    "crg_prep": {"bell": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                 "bell_zigzag": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
                 "linear_k1": [0.01, 0.02, 0.03, 0.04],
                 "linear_k2": [0.01, 0.02, 0.03, 0.04, 0.05],
                 "linear_kinf": [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]},
}
CRG_WINDOW_STEP = 0.005
CRG_WINDOW_POINTS = 13          # +-0.03 about the centre: the scaling fit's window at d = 11
CRG_CLOSING_PROBE = [0.005]

# The thresholds the paper reports (3 d.p.): the crg window centres.
CRG_THRESHOLDS = {
    ('crg_local', 'bell', 'ZZ'): {0.001: 0.490, 0.002: 0.470, 0.003: 0.445, 0.004: 0.423, 0.005: 0.404, 0.006: 0.391, 0.007: 0.377, 0.008: 0.366, 0.009: 0.343, 0.01: 0.284},
    ('crg_local', 'bell_zigzag', 'ZZ'): {0.001: 0.472, 0.002: 0.450, 0.003: 0.424, 0.004: 0.395, 0.005: 0.369, 0.006: 0.345, 0.007: 0.324, 0.008: 0.307, 0.009: 0.277, 0.01: 0.214},
    ('crg_local', 'linear_k1', 'ZZ'): {0.001: 0.304, 0.002: 0.291, 0.003: 0.277, 0.004: 0.263, 0.005: 0.249, 0.006: 0.237, 0.007: 0.228, 0.008: 0.221, 0.009: 0.203, 0.01: 0.161},
    ('crg_local', 'linear_k2', 'ZZ'): {0.001: 0.367, 0.002: 0.353, 0.003: 0.336, 0.004: 0.321, 0.005: 0.305, 0.006: 0.290, 0.007: 0.280, 0.008: 0.269, 0.009: 0.253, 0.01: 0.204},
    ('crg_local', 'linear_kinf', 'ZZ'): {0.001: 0.484, 0.002: 0.469, 0.003: 0.450, 0.004: 0.430, 0.005: 0.410, 0.006: 0.394, 0.007: 0.380, 0.008: 0.368, 0.009: 0.342, 0.01: 0.285},
    ('crg_local', 'bell', 'XX'): {0.001: 0.487, 0.002: 0.476, 0.003: 0.463, 0.004: 0.450, 0.005: 0.439, 0.006: 0.424, 0.007: 0.410, 0.008: 0.397, 0.009: 0.367, 0.01: 0.294},
    ('crg_local', 'bell_zigzag', 'XX'): {0.001: 0.492, 0.002: 0.476, 0.003: 0.458, 0.004: 0.434, 0.005: 0.413, 0.006: 0.392, 0.007: 0.369, 0.008: 0.356, 0.009: 0.327, 0.01: 0.256},
    ('crg_local', 'linear_k1', 'XX'): {0.001: 0.307, 0.002: 0.297, 0.003: 0.286, 0.004: 0.276, 0.005: 0.263, 0.006: 0.248, 0.007: 0.234, 0.008: 0.221, 0.009: 0.199, 0.01: 0.152},
    ('crg_local', 'linear_k2', 'XX'): {0.001: 0.372, 0.002: 0.362, 0.003: 0.348, 0.004: 0.336, 0.005: 0.320, 0.006: 0.307, 0.007: 0.292, 0.008: 0.273, 0.009: 0.242, 0.01: 0.196},
    ('crg_local', 'linear_kinf', 'XX'): {0.001: 0.491, 0.002: 0.479, 0.003: 0.464, 0.004: 0.451, 0.005: 0.433, 0.006: 0.415, 0.007: 0.395, 0.008: 0.374, 0.009: 0.341, 0.01: 0.284},
    ('crg_prep', 'bell', 'ZZ'): {0.01: 0.436, 0.02: 0.355, 0.03: 0.282, 0.04: 0.211, 0.05: 0.135, 0.06: 0.048},
    ('crg_prep', 'bell_zigzag', 'ZZ'): {0.01: 0.419, 0.02: 0.335, 0.03: 0.257, 0.04: 0.182, 0.05: 0.099},
    ('crg_prep', 'linear_k1', 'ZZ'): {0.01: 0.237, 0.02: 0.164, 0.03: 0.093, 0.04: 0.021},
    ('crg_prep', 'linear_k2', 'ZZ'): {0.01: 0.302, 0.02: 0.227, 0.03: 0.158, 0.04: 0.084, 0.05: 0.006},
    ('crg_prep', 'linear_kinf', 'ZZ'): {0.005: 0.462, 0.01: 0.427, 0.02: 0.349, 0.03: 0.274, 0.04: 0.197, 0.05: 0.118, 0.06: 0.033},
    ('crg_prep', 'bell', 'XX'): {0.01: 0.430, 0.02: 0.347, 0.03: 0.262, 0.04: 0.173, 0.05: 0.081},
    ('crg_prep', 'bell_zigzag', 'XX'): {0.01: 0.443, 0.02: 0.361, 0.03: 0.280, 0.04: 0.205, 0.05: 0.119, 0.06: 0.034},
    ('crg_prep', 'linear_k1', 'XX'): {0.01: 0.249, 0.02: 0.164, 0.03: 0.091, 0.04: 0.018},
    ('crg_prep', 'linear_k2', 'XX'): {0.01: 0.318, 0.02: 0.235, 0.03: 0.156, 0.04: 0.085, 0.05: 0.012},
    ('crg_prep', 'linear_kinf', 'XX'): {0.005: 0.473, 0.01: 0.444, 0.02: 0.360, 0.03: 0.277, 0.04: 0.207, 0.05: 0.126, 0.06: 0.042},
}

# Rows of the Lambda grid: p_c = p_r, falling to loss only.
LAMBDA_LOCAL_PROBS = [0.001, 0.0001, 0.0]
# The paper's Lambda points: {(family, observable, p_l): {n_boost: [loss rates]}}.
# n = 3 supplies the erasure rates below n = 2's floor (p_e = 0.125). Points the
# paper leaves out (too few errors at d = 7, or LER saturated) are not collected;
# the leading-order band stands in for the low-p_e loss-only ones.
_N2 = [0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
_N3 = [0.001, 0.0025, 0.005]
LAMBDA_POINTS = {
    ('bell', 'XX', 0.0): {3: _N3, 2: _N2[:-1]},
    ('bell', 'XX', 0.0001): {3: _N3, 2: _N2[:-1]},
    ('bell', 'XX', 0.001): {3: _N3, 2: _N2[:-1]},
    ('bell', 'ZZ', 0.0): {3: _N3, 2: _N2},
    ('bell', 'ZZ', 0.0001): {3: _N3, 2: _N2},
    ('bell', 'ZZ', 0.001): {3: _N3, 2: _N2},
    ('bell_zigzag', 'XX', 0.0): {3: _N3, 2: _N2},
    ('bell_zigzag', 'XX', 0.0001): {3: _N3, 2: _N2},
    ('bell_zigzag', 'XX', 0.001): {3: _N3, 2: _N2},
    ('bell_zigzag', 'ZZ', 0.0): {3: _N3, 2: _N2},
    ('bell_zigzag', 'ZZ', 0.0001): {3: _N3, 2: _N2},
    ('bell_zigzag', 'ZZ', 0.001): {3: _N3, 2: _N2},
    ('linear_k1', 'XX', 0.0): {3: _N3, 2: _N2[:5]},
    ('linear_k1', 'XX', 0.0001): {3: _N3, 2: _N2[:5]},
    ('linear_k1', 'XX', 0.001): {3: _N3, 2: _N2[:5]},
    ('linear_k1', 'ZZ', 0.0): {3: [0.005, 0.0075], 2: _N2},
    ('linear_k1', 'ZZ', 0.0001): {3: _N3, 2: _N2},
    ('linear_k1', 'ZZ', 0.001): {3: _N3, 2: _N2},
    ('linear_k2', 'XX', 0.0): {3: _N3, 2: _N2[:7]},
    ('linear_k2', 'XX', 0.0001): {3: _N3, 2: _N2[:7]},
    ('linear_k2', 'XX', 0.001): {3: _N3, 2: _N2[:7]},
    ('linear_k2', 'ZZ', 0.0): {2: _N2},
    ('linear_k2', 'ZZ', 0.0001): {3: _N3, 2: _N2},
    ('linear_k2', 'ZZ', 0.001): {3: _N3, 2: _N2},
    ('linear_kinf', 'XX', 0.0): {3: _N3, 2: _N2[:-1]},
    ('linear_kinf', 'XX', 0.0001): {3: _N3, 2: _N2[:-1]},
    ('linear_kinf', 'XX', 0.001): {3: _N3, 2: _N2[:-1]},
    ('linear_kinf', 'ZZ', 0.0): {2: _N2[3:]},
    ('linear_kinf', 'ZZ', 0.0001): {3: [0.005], 2: _N2},
    ('linear_kinf', 'ZZ', 0.001): {3: _N3, 2: _N2},
}

# ---------------------------------------------------------------------------
# Runner configs
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"


def _sinter(budget: tuple[int, int]) -> dict:
    max_shots, max_errors = budget
    return {"decoder": "pymatching_heralded", "max_shots": max_shots, "max_errors": max_errors,
            "num_workers": 2, "pm_decompose": True, "pm_enable_correlations": True}


def _config(name: str, family: str, observable: str, sweep: dict, noise: list[dict],
            budget: tuple[int, int]) -> dict:
    fam = FAMILIES[family]
    return {
        "name": name,
        "builder": {"name": fam.builder},
        "sweep": {"merge_types": [MERGE], "observables": [observable],
                  **sweep},
        "noise_model": noise,
        "resource_state": family,
        "results_dir": RESULTS_DIR,
        "sinter": _sinter(budget),
    }


def erasure_config(family: str, observable: str) -> dict:
    lo, hi = ERASURE_RANGES[family]
    return _config(f"erasure_{family}_corr_obs{observable}", family, observable,
                   {"distances": THRESHOLD_DISTANCES, "min": lo, "max": hi, "step": ERASURE_STEP},
                   erasure_noise(), THRESHOLD_BUDGET)


def crg_window(center: float | None) -> list[float]:
    """The erasure rates collected at one local noise level: CRG_WINDOW_POINTS
    on the CRG_WINDOW_STEP grid about `center` (rates >= 0), or the closing
    probe when there is no threshold to centre on."""
    if center is None:
        return list(CRG_CLOSING_PROBE)
    mid = round(center / CRG_WINDOW_STEP)
    half = CRG_WINDOW_POINTS // 2
    return [round(k * CRG_WINDOW_STEP, 6) for k in range(mid - half, mid + half + 1) if k >= 0]


def crg_config(kind: str, family: str, observable: str) -> dict:
    """A correctable-regime config: one fixed window per local noise level
    (`sweep.local_prob_sweeps`), in the schema run_correctable_regime reads."""
    local_probs = CRG_LOCAL_PROBS[kind][family]
    thresholds = CRG_THRESHOLDS[(kind, family, observable)]
    noise = crg_local_noise() if kind == "crg_local" else crg_prep_noise()
    sweep = {"distances": CRG_DISTANCES, "local_probs": local_probs,
             "local_prob_sweeps": {str(lp): {"probs": crg_window(thresholds.get(lp))}
                                   for lp in local_probs}}
    return _config(f"{kind}_{family}_corr_obs{observable}", family, observable, sweep, noise,
                   THRESHOLD_BUDGET)


def lambda_config(family: str, observable: str, n_boost: int,
                  local_probs: list[float] | None = None) -> dict:
    """The Lambda config for one boost level: every loss rate any of its local
    noise levels needs (`local_probs`, default all three rows)."""
    local_probs = LAMBDA_LOCAL_PROBS if local_probs is None else local_probs
    probs = sorted({p for lp in local_probs
                    for p in LAMBDA_POINTS[(family, observable, lp)].get(n_boost, [])})
    sweep = {"distances": LAMBDA_DISTANCES, "probs": probs, "local_probs": list(local_probs)}
    return _config(f"subthreshold_lambda_{family}_corr_obs{observable}_n{n_boost}", family,
                   observable, sweep, lambda_noise(n_boost), LAMBDA_BUDGET)


def resolve(config: dict, local_prob: float, probs: list[float]) -> dict:
    """A fixed-noise run of `config` at one local noise level over `probs`:
    p_l channels take `local_prob`, p_nl channels follow the swept rate
    (`scale: 1.0`), anything else (fixed rates, p_e) passes through -- the
    convention of run_correctable_regime and run_subthreshold, so the sinter
    CSV is the one the figures read for (config, local_prob)."""
    run = copy.deepcopy(config)
    run["noise_model"] = result_names.resolve_local_prob(config["noise_model"], local_prob)
    for key in ("local_probs", "local_prob_sweeps", "min", "max", "step"):
        run["sweep"].pop(key, None)
    run["sweep"]["probs"] = list(probs)
    return run


# ---------------------------------------------------------------------------
# Curves and figures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Curve:
    """One curve of one figure: `id` is `<kind>/<family>/<observable>` (crg,
    erasure) or `lambda/<family>/<observable>/<p_l>`."""
    kind: str               # "erasure" | "crg_local" | "crg_prep" | "lambda"
    family: str
    observable: str
    local_prob: float | None = None   # lambda only: the grid row

    @property
    def id(self) -> str:
        base = f"{self.kind}/{self.family}/{self.observable}"
        return base if self.local_prob is None else f"{base}/{self.local_prob:g}"

    def runs(self) -> list[dict]:
        """The fixed-noise runner configs that collect this curve."""
        if self.kind == "erasure":
            return [erasure_config(self.family, self.observable)]
        if self.kind in CRG_LOCAL_PROBS:
            config = crg_config(self.kind, self.family, self.observable)
            windows = config["sweep"]["local_prob_sweeps"]
            # The p_l = 0 point is the pure-erasure sweep.
            return ([resolve(config, lp, windows[str(lp)]["probs"]) for lp in config["sweep"]["local_probs"]]
                    + Curve("erasure", self.family, self.observable).runs())
        points = LAMBDA_POINTS[(self.family, self.observable, self.local_prob)]
        return [resolve(lambda_config(self.family, self.observable, n, [self.local_prob]),
                        self.local_prob, losses)
                for n, losses in sorted(points.items())]


@dataclass(frozen=True)
class Figure:
    key: str
    stem: str               # the paper's file name: <stem>.tex and <stem>_data.csv
    curves: tuple[Curve, ...]


def _figures() -> dict[str, Figure]:
    figs = [Figure("master_threshold", "master_threshold_pm_corr_mergeZZ_obsZZ-XX",
                   tuple(Curve("erasure", f, o) for f in FAMILIES for o in OBSERVABLES))]
    for kind, stem in (("crg_local", "circuit_and_erasure_correctable_regime"),
                       ("crg_prep", "prep_and_erasure_correctable_regime")):
        for obs in OBSERVABLES:
            figs.append(Figure(f"{kind}_{obs}", f"{stem}_pm_corr_mergeZZ_obs{obs}",
                               tuple(Curve(kind, f, obs) for f in FAMILIES)))
    figs.append(Figure("lambda_grid", "subthreshold_lambda_erasure_grid_pm_corr_mergeZZ_obsZZ-XX",
                       tuple(Curve("lambda", f, o, lp) for lp in LAMBDA_LOCAL_PROBS
                             for o in OBSERVABLES for f in FAMILIES)))
    return {f.key: f for f in figs}


FIGURES = _figures()
CURVES: dict[str, Curve] = {c.id: c for fig in FIGURES.values() for c in fig.curves}


def boosted_erasure_floor(n_boost: int) -> float:
    """Fusion erasure rate at zero loss, boost level n: p_fail / 2 = 2^-(n+1)."""
    return 2.0 ** -(n_boost + 1)


def p_l_tex(local_prob: float) -> str:
    exponent = round(math.log10(local_prob))
    mantissa = local_prob / 10 ** exponent
    return (f"10^{{{exponent}}}" if math.isclose(mantissa, 1.0)
            else f"{mantissa:g}\\times 10^{{{exponent}}}")
