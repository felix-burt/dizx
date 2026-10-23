"""
Pure-PyMatching heralded-erasure decoding.

stim's ``HERALDED_ERASE`` injects a depolarising erasure on a qubit *and*
records a herald bit flagging whether the erasure fired.  stim can sample the
heralds and syndromes but cannot decode them.  This module decodes such
circuits with plain MWPM (pymatching), without any mwpf-specific heralded-DEM
machinery, using a two-circuit strategy:

* **Sampling circuit** (the input) keeps every ``HERALDED_ERASE`` and its herald
  ``DETECTOR``.  We sample herald bits + data detection events + observables
  from it.
* **Matching circuit** replaces every ``HERALDED_ERASE`` with a plain
  ``DEPOLARIZE1`` at the same qubit/time and deletes the herald measurement and
  its detector (adjusting all later ``rec[-k]`` offsets, since the heralded
  erase also functioned as a measurement).  Its detector error model is an
  ordinary graphlike matching graph — no ``approximate_disjoint_errors`` needed.

Each replacement ``DEPOLARIZE1`` is tagged ``erase_<k>``; stim propagates the
tag onto the resulting DEM error instructions, giving a direct
erasure -> matching-edge map.  Per shot the decoder reads which heralds fired,
gives each fired erasure edge weight 0 (``p = 0.5``) and turns the others off,
then decodes the herald-stripped syndrome (see :mod:`dizx.sinter_decoders`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import stim

from dizx.noise_models import (
    ALL_MEASUREMENT_GATES, HERALDING_INSTRUCTIONS, _n_meas_inst as _meas_count,
)

# Probability used to turn an un-heralded erasure edge "off" (weight ~ +inf).
_OFF_EDGE_PROB = 1e-15


def _herald_meas_indices(flat: list[stim.CircuitInstruction]) -> set[int]:
    """Absolute measurement indices produced by heralding instructions."""
    herald: set[int] = set()
    offset = 0
    for inst in flat:
        n = _meas_count(inst)
        if inst.name in HERALDING_INSTRUCTIONS:
            herald.update(range(offset, offset + n))
        offset += n
    return herald


def _detector_meas_map(
    flat: list[stim.CircuitInstruction],
) -> dict[int, list[int]]:
    """detector_id -> sorted absolute measurement indices feeding it
    (heralding instructions count as measurements)."""
    det_to_meas: dict[int, list[int]] = {}
    offset = 0
    det_id = 0
    for inst in flat:
        n = _meas_count(inst)
        if n:
            offset += n
        elif inst.name == "DETECTOR":
            idxs = [
                offset + t.value
                for t in inst.targets_copy()
                if t.is_measurement_record_target
            ]
            det_to_meas[det_id] = sorted(idxs)
            det_id += 1
    return det_to_meas


# ---------------------------------------------------------------------------
# Circuit rewriting via text.
#
# ``stim.Circuit.append`` costs ~80 us per call (stim 1.16), so rebuilding a
# 40k-instruction circuit instruction-by-instruction takes seconds.  Emitting
# the circuit as text and parsing it once moves that work into C++ and makes
# the rewrite ~17x faster end to end.  The one subtlety is that
# ``str(instruction)`` prints gate arguments with 6 significant digits, which
# would silently round noise probabilities; ``_inst_text`` re-emits those
# arguments at full precision whenever the printed form is not lossless.
# ---------------------------------------------------------------------------

_REC_TEXT_RE = re.compile(r"rec\[-?\d+\]")


def _args_round_trip(args: list[float]) -> bool:
    """Whether ``str(instruction)``'s 6-significant-digit args are lossless."""
    return all(float(f"{v:.6g}") == v for v in args)


def _split_inst_text(
    inst: stim.CircuitInstruction, text: str
) -> tuple[str, str, str]:
    """Split ``text`` (``str(inst)``) into ``(name+tag, args, targets)``.

    The tag is bracket-delimited and stim escapes ``]`` inside it as ``\\C``
    (verified over the whole ASCII range), so a literal ``]`` cannot occur
    within the printed tag and the first one terminates it.
    """
    i = len(inst.name)
    if inst.tag:
        i = text.index("]", i + 1) + 1
    if i < len(text) and text[i] == "(":
        j = text.index(")", i)
        return text[:i], text[i : j + 1], text[j + 1 :]
    return text[:i], "", text[i:]


def _fmt_args(args: list[float]) -> str:
    return "(" + ", ".join(repr(float(v)) for v in args) + ")"


def _inst_text(inst: stim.CircuitInstruction) -> str:
    """Circuit text for ``inst``, preserving gate arguments exactly."""
    text = str(inst)
    args = inst.gate_args_copy()
    if not args or _args_round_trip(args):
        return text
    head, _, targets = _split_inst_text(inst, text)
    return f"{head}{_fmt_args(args)}{targets}"


def _rec_inst_text(
    inst: stim.CircuitInstruction, new_offsets: list[int]
) -> str:
    """Circuit text for ``inst`` with its ``rec[-k]`` targets renumbered.

    ``new_offsets`` supplies one replacement offset per ``rec`` target, in
    order.  Substitution is confined to the targets portion so that a tag
    containing the literal text ``rec[-1]`` cannot be corrupted.
    """
    head, args, targets = _split_inst_text(inst, str(inst))
    if args:
        gate_args = inst.gate_args_copy()
        if not _args_round_trip(gate_args):
            args = _fmt_args(gate_args)
    it = iter(new_offsets)
    text = head + args + _REC_TEXT_RE.sub(lambda _m: f"rec[{next(it)}]", targets)
    assert next(it, None) is None, f"unconsumed rec offsets in {inst}"
    return text


def make_matching_circuit(
    circuit: stim.Circuit,
    *,
    keep_herald_detectors: bool = False,
) -> tuple[stim.Circuit, int, dict[int, int]]:
    """Build the matching circuit (circuit 2) from the sampling circuit.

    Every ``HERALDED_ERASE(p) q...`` is replaced by one tagged
    ``DEPOLARIZE1(p) q`` per qubit (``tag="erase_<k>"``); its herald measurement
    bits and herald detectors are removed, and all later ``rec[-k]`` offsets in
    ``DETECTOR`` / ``OBSERVABLE_INCLUDE`` are adjusted.

    With ``keep_herald_detectors`` each herald detector instead stays in place
    as a target-less ``DETECTOR``: it never fires and no error touches it, but
    it keeps its index, so the matching circuit's detectors are numbered exactly
    like the sampling circuit's.

    Returns
    -------
    matching_circuit : stim.Circuit
    n_erasures : int
        Number of erasure locations (tags ``erase_0 .. erase_{n-1}``).
    erase_to_herald_meas : dict[int, int]
        erasure id ``k`` -> absolute herald-bit measurement index in the
        original (sampling) circuit.  Used to link an erasure to its herald
        detector.
    """
    flat = list(circuit.flattened())
    herald_meas = _herald_meas_indices(flat)

    # Pass 1: original abs meas index -> new abs meas index (heralds dropped).
    orig_to_new: dict[int, int] = {}
    orig_off = 0
    new_off = 0
    for inst in flat:
        n = _meas_count(inst)
        if n == 0:
            continue
        if inst.name in HERALDING_INSTRUCTIONS:
            orig_off += n  # herald bits removed; new index unchanged
            continue
        for k in range(n):
            orig_to_new[orig_off + k] = new_off + k
        orig_off += n
        new_off += n

    # Pass 2: rebuild as text (see the note above ``_inst_text``).
    # ``orig_running`` counts original measurements (incl. herald bits);
    # ``new_running`` counts surviving measurements.
    lines: list[str] = []
    orig_running = 0
    new_running = 0
    erase_to_herald_meas: dict[int, int] = {}
    k = 0
    for inst in flat:
        if inst.name in HERALDING_INSTRUCTIONS:
            args = inst.gate_args_copy()
            p = repr(float(args[0])) if args else "0"
            for t in inst.targets_copy():
                lines.append(
                    f"DEPOLARIZE1[erase_{k}]({p}) {t.qubit_value}"
                )
                erase_to_herald_meas[k] = orig_running
                orig_running += 1
                k += 1
        elif inst.name in ALL_MEASUREMENT_GATES:
            lines.append(_inst_text(inst))
            n = _meas_count(inst)
            orig_running += n
            new_running += n
        elif inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            abs_idxs = [
                orig_running + t.value
                for t in inst.targets_copy()
                if t.is_measurement_record_target
            ]
            # Drop herald detectors (all rec targets are herald bits).
            if (
                inst.name == "DETECTOR"
                and abs_idxs
                and all(a in herald_meas for a in abs_idxs)
            ):
                if keep_herald_detectors:
                    lines.append("DETECTOR")
                continue
            # rec offsets are relative to the *new* running count.
            lines.append(
                _rec_inst_text(
                    inst, [orig_to_new[a] - new_running for a in abs_idxs]
                )
            )
        else:
            lines.append(_inst_text(inst))

    return stim.Circuit("\n".join(lines)), k, erase_to_herald_meas


@dataclass
class ErasureMatchingData:
    """Precomputed structures for heralded-erasure MWPM decoding.

    The matching circuit keeps each herald detector as a target-less
    ``DETECTOR``, so ``base_dem`` numbers detectors like the sampling circuit.
    """

    base_dem: stim.DetectorErrorModel
    n_erasures: int
    # erasure id k -> herald detector id in the sampling circuit (or None).
    herald_det_for_erase: list[int | None]
    # Every herald detector id in the sampling circuit, sorted (including
    # multi-bit herald detectors that no single erasure maps to).
    herald_det_ids: list[int]


def build_erasure_matching_data(circuit: stim.Circuit, *, decompose: bool = True
                                ) -> ErasureMatchingData:
    """Precompute everything needed to decode ``circuit`` (a sampling circuit
    containing ``HERALDED_ERASE`` + herald detectors) with heralded MWPM.
    ``decompose`` is passed to ``detector_error_model(decompose_errors=...)``
    when building the matching graph."""
    matching_circuit, n_erasures, erase_to_herald_meas = make_matching_circuit(
        circuit, keep_herald_detectors=True
    )
    base_dem = matching_circuit.detector_error_model(decompose_errors=decompose)

    flat = list(circuit.flattened())
    herald_meas = _herald_meas_indices(flat)
    det_to_meas = _detector_meas_map(flat)
    # herald bit measurement index -> herald detector id.
    herald_dets: set[int] = set()
    herald_meas_to_det: dict[int, int] = {}
    for d, meas in det_to_meas.items():
        if not meas or not set(meas) <= herald_meas:
            continue
        herald_dets.add(d)
        if len(meas) == 1:
            herald_meas_to_det[meas[0]] = d
    herald_det_for_erase = [
        herald_meas_to_det.get(erase_to_herald_meas[k]) for k in range(n_erasures)
    ]
    return ErasureMatchingData(
        base_dem=base_dem,
        n_erasures=n_erasures,
        herald_det_for_erase=herald_det_for_erase,
        herald_det_ids=sorted(herald_dets),
    )


def _reweighted_dem(
    base_dem: stim.DetectorErrorModel,
    fired: set[int],
    erased_edge_prob: float,
) -> stim.DetectorErrorModel:
    """One-shot DEM: fired erasure edges at ``erased_edge_prob``, un-fired ones off."""
    out = stim.DetectorErrorModel()
    for inst in base_dem:
        if inst.type == "error" and inst.tag.startswith("erase_"):
            k = int(inst.tag[len("erase_"):])
            p = erased_edge_prob if k in fired else _OFF_EDGE_PROB
            out.append("error", p, inst.targets_copy(), tag=inst.tag)
        else:
            out.append(inst)
    return out
