from __future__ import annotations

import json
from bisect import bisect_right
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from typing import Any, Mapping

import stim

SINGLE_QUBIT_GATES = frozenset({'H', 'I', 'R', 'RX'})
TWO_QUBIT_GATES = frozenset({'CNOT', 'CX', 'CZ'})
MEASUREMENT_GATES = frozenset({'M', 'MX', 'MPP', 'MY', 'MZ', 'MR', 'MRX', 'MRY', 'MRZ'})
# Two-qubit correlated measurement gates: each pair of targets produces 1 measurement.
PAIRED_MEASUREMENT_GATES = frozenset({'MXX', 'MYY', 'MZZ', 'MXZ', 'MZX'})

ALL_MEASUREMENT_GATES = MEASUREMENT_GATES | PAIRED_MEASUREMENT_GATES
ALL_GATE_INSTRUCTIONS = SINGLE_QUBIT_GATES | TWO_QUBIT_GATES | ALL_MEASUREMENT_GATES

# Instructions whose target list is a flat sequence of qubit *pairs*: one gate
# per pair, so a single instruction line may carry many gates
# (e.g. ``CX 0 1 2 3 4 5`` is three CNOTs). Selecting "the first qubit of the
# gate" therefore means every even-indexed target, not just target 0.
PAIRED_GATE_INSTRUCTIONS = TWO_QUBIT_GATES | PAIRED_MEASUREMENT_GATES

# Allowed values of ``NoiseSpec.qubits``.
QUBIT_SELECTORS = frozenset({'all', 'first', 'second'})

# Noise instructions that emit one herald measurement per targeted qubit and
# therefore shift every subsequent rec(-) index. These require the remap pass.
HERALDING_INSTRUCTIONS = frozenset({'HERALDED_ERASE', 'HERALDED_PAULI_CHANNEL_1'})


@dataclass(frozen=True)
class NoiseSpec:
    """Instruction and probability/-ies used to inject one noise operation.

    ``probability`` is either a single float (the usual case) or an iterable of
    floats for multi-argument channels such as ``HERALDED_PAULI_CHANNEL_1``,
    which carries one probability per Pauli component in ``(p_I, p_X, p_Y, p_Z)``
    order.

    ``position`` overrides where the noise is inserted relative to the matched
    instruction: ``'before'`` or ``'after'``. When ``None`` (the default) the
    insertion point follows the usual convention — noise goes *before*
    measurement instructions and *after* all other gate instructions.

    ``qubits`` restricts *which* targets of the matched instruction the noise
    acts on:

    - ``'all'`` (default): every target, with the instruction's own pairing
      preserved (so ``DEPOLARIZE2`` on ``CX 0 1 2 3`` correlates 0-1 and 2-3).
    - ``'first'``: only the first qubit of each two-qubit gate.
    - ``'second'``: only the second qubit of each two-qubit gate.

    ``'first'``/``'second'`` are only meaningful for instructions whose targets
    are a flat list of qubit pairs (:data:`PAIRED_GATE_INSTRUCTIONS`), and they
    select *per gate*, not per instruction line: on ``CX 0 1 2 3 4 5``,
    ``qubits='first'`` targets qubits 0, 2 and 4. The paired instruction
    ``instruction`` should be a single-qubit channel (``X_ERROR``,
    ``DEPOLARIZE1``, ``HERALDED_ERASE``, ...), since only one qubit per gate is
    handed to it.
    """

    instruction: str
    probability: float | Collection[float] = 0.0
    position: str | None = None
    qubits: str = 'all'

    def __post_init__(self) -> None:
        if self.position not in (None, 'before', 'after'):
            raise ValueError(
                f"position must be 'before', 'after', or None, got {self.position!r}"
            )
        if self.qubits not in QUBIT_SELECTORS:
            raise ValueError(
                f"qubits must be one of {sorted(QUBIT_SELECTORS)}, got {self.qubits!r}"
            )

    @property
    def gate_args(self) -> list[float]:
        """Probability arguments to pass to ``stim.Circuit.append``."""
        if isinstance(self.probability, (int, float)):
            return [float(self.probability)]
        return [float(p) for p in self.probability]

    @property
    def total_probability(self) -> float:
        """Combined probability mass; used for activity / pruning checks."""
        return float(sum(self.gate_args))

    @property
    def is_active(self) -> bool:
        """Whether this spec injects anything (skipped entirely if not)."""
        return self.total_probability > 0.0


# A tag key may map to a single spec, or a list of specs applied in order
# (e.g. both DEPOLARIZE2 and HERALDED_ERASE on the same tagged gate).
NoiseSpecOrList = NoiseSpec | list[NoiseSpec]
NoiseKey = str | tuple[str, str] | tuple[str, str, str]


def _candidate_keys(
    op_type: Any,
    op_operation: Any,
    op_qubit_type: Any,
) -> list[NoiseKey]:
    """Noise-map lookup keys for one tagged instruction, most specific first.

    An instruction tagged ``{"type": "graph", "operation": "reset",
    "qubit_type": "matter"}`` is matched by ``('graph', 'reset', 'matter')``,
    then ``('graph', 'reset')``, then ``'graph'`` — the first key present in the
    map wins, so a matter-only entry overrides the generic reset entry for those
    qubits while leaving every other reset on the generic one.
    """
    if not isinstance(op_type, str):
        return []
    keys: list[NoiseKey] = []
    if isinstance(op_operation, str):
        if isinstance(op_qubit_type, str):
            keys.append((op_type, op_operation, op_qubit_type))
        keys.append((op_type, op_operation))
    keys.append(op_type)
    return keys


def _to_noise_specs(value: NoiseSpecOrList) -> list[NoiseSpec]:
    """Normalize a ``noise_by_type`` value to a list of specs, applied in order."""
    return list(value) if isinstance(value, list) else [value]


def parse_op_tag(tag: str) -> dict[str, Any]:
    """Parse a JSON instruction tag; anything else (e.g. ``erase_k``) is ``{}``."""
    try:
        parsed = json.loads(tag) if tag else {}
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _escape_tag(tag: str) -> str:
    """Escape an instruction tag the way stim's text format does."""
    return (tag.replace('\\', '\\B').replace('\r', '\\r')
            .replace('\n', '\\n').replace(']', '\\C'))


def _target_text(target: int | stim.GateTarget) -> str:
    if isinstance(target, int):
        return str(target)
    if target.is_qubit_target and not target.is_inverted_result_target:
        return str(target.value)
    if target.is_measurement_record_target:
        return f'rec[{target.value}]'
    if target.is_sweep_bit_target:
        return f'sweep[{target.value}]'
    raise ValueError(f'unsupported target for an inserted operation: {target!r}')


class _CircuitText:
    """Builds a circuit as stim text lines, parsed into a circuit once at the end.

    ``stim.Circuit.append`` costs tens of microseconds per instruction and
    milliseconds per inserted noise operation, which made noise application
    take seconds on large circuits; stim parses the equivalent text ~100x
    faster. Parsing fuses adjacent compatible instructions exactly as
    ``append`` does, and probabilities are written with ``repr`` so they
    round-trip bit-for-bit.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def append_instruction(self, inst: stim.CircuitInstruction) -> None:
        self.lines.append(str(inst))

    def append_op(
        self,
        name: str,
        targets: Iterable[int | stim.GateTarget],
        gate_args: Iterable[float] = (),
        tag: str = '',
    ) -> None:
        head = f'{name}[{_escape_tag(tag)}]' if tag else name
        args = ', '.join(repr(float(a)) for a in gate_args)
        if args:
            head = f'{head}({args})'
        self.lines.append(' '.join([head, *map(_target_text, targets)]))

    def to_circuit(self) -> stim.Circuit:
        return stim.Circuit('\n'.join(self.lines))


def _append_noise(
    noisy: _CircuitText,
    spec: NoiseSpec,
    targets: list[stim.GateTarget],
) -> None:
    if not spec.is_active:
        return
    noisy.append_op(spec.instruction, targets, spec.gate_args)


def _combine_flip_prob(existing: float | None, extra: float) -> float:
    if existing is None:
        return extra
    return 1.0 - (1.0 - existing) * (1.0 - extra)


def _append_instruction_with_measurement_flip(
    noisy: _CircuitText,
    inst: stim.CircuitInstruction,
    *,
    flip_probability: float | None = None,
) -> None:
    if flip_probability is None:
        noisy.append_instruction(inst)
        return
    if inst.name not in ALL_MEASUREMENT_GATES:
        raise ValueError(
            "MEASUREMENT_FLIP can only be applied to measurement instructions."
        )
    args = inst.gate_args_copy()
    if len(args) > 1:
        raise ValueError(
            f"{inst.name} has unsupported measurement args {args!r}; "
            "cannot apply MEASUREMENT_FLIP."
        )
    existing = float(args[0]) if args else None
    combined = _combine_flip_prob(existing, flip_probability)
    # Measurement targets can be inverted (!q) or Pauli products (MPP), so reuse
    # stim's own target text: ``NAME[tag](args) targets``, where the escaped tag
    # holds no raw ']'.
    text = str(inst)
    end = text.index(']') + 1 if inst.tag else len(inst.name)
    if text.startswith('(', end):
        end = text.index(')', end) + 1
    head = f'{inst.name}[{_escape_tag(inst.tag)}]' if inst.tag else inst.name
    noisy.lines.append(f'{head}({combined!r}){text[end:]}')


def _reset_flip_instruction(inst: stim.CircuitInstruction) -> str:
    """Concrete Pauli error instruction for a ``RESET_FLIP``, by reset basis.

    A reset prepares a basis eigenstate; the "flip" is the Pauli error that maps
    it to the orthogonal eigenstate: ``Z_ERROR`` for X-basis resets
    (``RX``/``MRX``) and ``X_ERROR`` for Z- and Y-basis resets
    (``R``/``RZ``/``RY``/``MR``/``MRZ``/``MRY``).
    """
    name = inst.name
    if name in ('RX', 'MRX'):
        return 'Z_ERROR'
    if name in ('R', 'RZ', 'RY', 'MR', 'MRZ', 'MRY'):
        return 'X_ERROR'
    raise ValueError(
        f"RESET_FLIP can only be applied to reset instructions, got {name!r}."
    )


def _append_heralded_with_detectors(
    noisy: _CircuitText,
    *,
    instruction: str,
    qubits: list[int],
    gate_args: list[float],
    tag: str,
) -> None:
    """Append a heralding noise op and immediate detector(s) for its herald bits.

    Heralding instructions (``HERALDED_ERASE``, ``HERALDED_PAULI_CHANNEL_1``)
    produce one measurement bit per targeted qubit. We append one DETECTOR per
    herald bit immediately after the operation.
    """
    if not qubits:
        return
    noisy.append_op(instruction, qubits, gate_args, tag=tag)
    n = len(qubits)
    for k in range(n, 0, -1):
        noisy.lines.append(f'DETECTOR rec[{-k}]')


def _n_meas_inst(inst: stim.CircuitInstruction) -> int:
    """Number of measurements produced by one instruction."""
    if inst.name in HERALDING_INSTRUCTIONS:
        # Produces 1 measurement (herald bit) per targeted qubit.
        return len(inst.targets_copy())
    if inst.name in PAIRED_MEASUREMENT_GATES:
        # Each pair of qubit targets produces 1 measurement (e.g. MXX q0 q1 q2 q3 → 2).
        return len(inst.targets_copy()) // 2
    if inst.name not in MEASUREMENT_GATES:
        return 0
    targets = inst.targets_copy()
    if inst.name == 'MPP':
        n_non = sum(1 for t in targets if not t.is_combiner)
        n_com = sum(1 for t in targets if t.is_combiner)
        return n_non - n_com
    return len(targets)


def _meas_qubits(inst: stim.CircuitInstruction) -> list[int]:
    """Measurement qubits touched by an instruction, deduplicated in order."""
    targets = inst.targets_copy()
    if inst.name == 'MPP':
        seen: dict[int, None] = {}
        for t in targets:
            if t.is_combiner:
                continue
            seen[t.qubit_value] = None
        return list(seen)
    return [t.qubit_value for t in targets]


def _op_qubits(inst: stim.CircuitInstruction) -> list[int]:
    """Qubits touched by a non-measurement operation, deduplicated in order."""
    seen: dict[int, None] = {}
    for t in inst.targets_copy():
        if t.is_combiner:
            continue
        try:
            seen[t.qubit_value] = None
        except AttributeError:
            continue
    return list(seen)


def _gate_pair_targets(
    inst: stim.CircuitInstruction,
    which: str,
) -> list[stim.GateTarget]:
    """Select one qubit from *every* two-qubit gate on an instruction line.

    A paired instruction packs one gate per consecutive target pair, so a line
    such as ``CX 0 1 2 3 4 5`` holds three gates. ``which='first'`` returns the
    controls ``[0, 2, 4]`` and ``which='second'`` the targets ``[1, 3, 5]``.

    Non-qubit targets (``rec[-1]`` controls, sweep bits) and any pair they
    belong to are skipped: there is no qubit there to apply a Pauli channel to.
    """
    if inst.name not in PAIRED_GATE_INSTRUCTIONS:
        raise ValueError(
            f"qubits={which!r} requires a two-qubit gate instruction "
            f"(one of {sorted(PAIRED_GATE_INSTRUCTIONS)}), got {inst.name!r}."
        )
    targets = inst.targets_copy()
    if len(targets) % 2 != 0:
        raise ValueError(
            f"{inst.name} has an odd number of targets ({len(targets)}); "
            'cannot split it into two-qubit gates.'
        )
    offset = 0 if which == 'first' else 1
    selected: list[stim.GateTarget] = []
    for i in range(0, len(targets), 2):
        chosen = targets[i + offset]
        if not chosen.is_qubit_target:
            continue
        selected.append(chosen)
    return selected


def _spec_targets(
    inst: stim.CircuitInstruction,
    spec: NoiseSpec,
) -> list[stim.GateTarget]:
    """Targets an individual spec should be applied to on ``inst``."""
    if spec.qubits == 'all':
        return inst.targets_copy()
    return _gate_pair_targets(inst, spec.qubits)


def _spec_herald_qubits(
    inst: stim.CircuitInstruction,
    spec: NoiseSpec,
    default_qubits: list[int],
) -> list[int]:
    """Qubit indices a heralding spec should herald on, deduplicated in order."""
    if spec.qubits == 'all':
        return default_qubits
    seen: dict[int, None] = {}
    for t in _gate_pair_targets(inst, spec.qubits):
        seen[t.qubit_value] = None
    return list(seen)


def apply_tagged_noise_model(
    circuit: stim.Circuit,
    *,
    noise_by_type: Mapping[NoiseKey, NoiseSpecOrList] | None = None,
) -> stim.Circuit:
    """
    Apply a tag-driven noise model over the circuit.

    The tag ``type`` field alone determines the noise applied to each instruction —
    gate category (single-qubit vs two-qubit) is ignored.  The insertion point is
    determined by instruction kind by default: measurement instructions receive
    noise immediately *before* them; all other gate instructions receive it
    *after*. A :class:`NoiseSpec` may override this per entry via its ``position``
    field (``'before'``/``'after'``).

    ``noise_by_type`` maps any of:

    - ``type`` strings,
    - ``(type, operation)`` tuples, or
    - ``(type, operation, qubit_type)`` tuples,

    to :class:`NoiseSpec`. Lookup is most-specific-first: the 3-tuple wins over the 2-tuple,
    which wins over the type-only key, and only the first match applies.
    Instructions whose tag key is not in the map receive no noise.

    The ``qubit_type`` field distinguishes qubit roles that share an operation.
    Resource-state preparation tags it ``"matter"`` for qubits sitting on the
    circuit/graph boundary (consumed by a circuit–graph fusion, or CNOT-entangled
    with a circuit ancilla in the bell-prep-fusion gadget) and ``"flying"`` for
    the rest of the state, so a matter-only reset error is just::

        {('graph', 'reset', 'matter'): NoiseSpec('RESET_FLIP', p_matter)}

    Circuit-mode operations use the same field for ``"data"``/``"ancilla"``.

    A mapping value may also be a **list** of specs, applied in list order —
    e.g. ``[NoiseSpec('DEPOLARIZE2', p2), NoiseSpec('HERALDED_ERASE', pl)]``
    attaches both a depolarizing channel and a heralded-erasure channel to the
    same tagged gate. Each spec in the list resolves its ``'before'``/``'after'``
    position independently (via its own ``position`` field, defaulting the same
    way a single spec would), and is applied in the given order within its
    position group.

    ``MEASUREMENT_FLIP`` is handled specially: instead of appending a separate
    noise instruction, the resolved probability is applied directly to the
    matched measurement instruction gate-arg. Multiple ``MEASUREMENT_FLIP``
    entries for the same instruction are combined in order (independent-flip
    composition), then applied once.

    A spec may also restrict itself to *one qubit of each two-qubit gate* via
    ``NoiseSpec.qubits='first'`` (or ``'second'``). This is resolved per
    gate rather than per instruction line, so a packed line such as
    ``CX 0 1 2 3 4 5`` yields noise on qubits 0, 2 and 4 — for example
    ``NoiseSpec('X_ERROR', p, qubits='first')`` puts an X error on every CNOT
    control. Applying such a spec to an instruction that is not a two-qubit
    gate raises :class:`ValueError`, so scope the key with a
    ``(type, operation)`` tuple when a ``type`` covers mixed gate arities.

    ``RESET_FLIP`` is also handled specially: it expands to an ``X_ERROR`` or
    ``Z_ERROR`` (chosen by the matched reset's basis) inserted immediately
    *after* the reset, so the flip acts on the freshly prepared eigenstate.

    If any resolved spec uses ``HERALDED_ERASE``, detector/observable
    ``rec(-)`` targets are remapped in a single pass by tracking how many
    herald measurements have been inserted at each original measurement offset.
    Each inserted ``HERALDED_ERASE`` is also followed immediately by
    detector(s) on its herald measurement bit(s).
    """
    flat = list(circuit.flattened())
    _noise_map: dict[NoiseKey, list[NoiseSpec]] = {
        k: _to_noise_specs(v) for k, v in (noise_by_type or {}).items()
    }

    def _resolve(inst: stim.CircuitInstruction) -> list[tuple[NoiseSpec, str]]:
        """Return [(spec, 'before'|'after'|'inline'), ...] in application order."""
        if inst.name not in ALL_GATE_INSTRUCTIONS:
            return []

        tag_data = parse_op_tag(inst.tag)
        op_type = tag_data.get('type')
        op_operation = tag_data.get('operation')
        default_position = 'before' if inst.name in ALL_MEASUREMENT_GATES else 'after'

        specs: list[NoiseSpec] = []
        for key in _candidate_keys(op_type, op_operation, tag_data.get('qubit_type')):
            if key in _noise_map:
                specs = _noise_map[key]
                break

        resolved: list[tuple[NoiseSpec, str]] = []
        for spec in specs:
            if not spec.is_active:
                continue
            if spec.instruction == 'MEASUREMENT_FLIP':
                resolved.append((spec, 'inline'))
                continue
            if spec.instruction == 'RESET_FLIP':
                # Resolve to the concrete Pauli error for this reset's basis. Reset
                # flips default to *after* the reset (overriding the measurement-gate
                # 'before' default for MR* gates); an explicit position still wins.
                concrete = _reset_flip_instruction(inst)
                concrete_spec = NoiseSpec(concrete, spec.probability, position=spec.position)
                resolved.append((concrete_spec, spec.position or 'after'))
                continue
            resolved.append((spec, spec.position or default_position))
        return resolved

    resolved = [_resolve(inst) for inst in flat]

    # Heralding instructions insert extra measurements, shifting all rec(-)
    # indices. Detect this up front so we can choose the right rebuild strategy.
    needs_remap = any(
        spec.instruction in HERALDING_INSTRUCTIONS
        for entries in resolved
        for spec, _ in entries
    )

    def _combined_flip_probability(entries: list[tuple[NoiseSpec, str]]) -> float | None:
        combined: float | None = None
        for spec, position in entries:
            if position != 'inline':
                continue
            combined = _combine_flip_prob(combined, spec.total_probability)
        return combined

    if not needs_remap:
        # Fast path: no measurement index shifts, single pass is safe.
        noisy = _CircuitText()
        for inst, entries in zip(flat, resolved):
            for spec, position in entries:
                if position == 'before':
                    _append_noise(noisy, spec, _spec_targets(inst, spec))
            flip_probability = _combined_flip_probability(entries)
            if flip_probability is not None:
                _append_instruction_with_measurement_flip(
                    noisy,
                    inst,
                    flip_probability=flip_probability,
                )
            else:
                noisy.append_instruction(inst)
            for spec, position in entries:
                if position == 'after':
                    _append_noise(noisy, spec, _spec_targets(inst, spec))
        return noisy.to_circuit()

    # Single-pass remap path: keep insertion events indexed by original
    # measurement offset. For an original absolute measurement index `x`, its
    # new absolute index is `x + inserted_before(x)`.
    noisy = _CircuitText()
    insert_positions: list[int] = []
    insert_prefix: list[int] = []

    def _record_insert(pos: int, count: int) -> None:
        if count <= 0:
            return
        insert_positions.append(pos)
        insert_prefix.append((insert_prefix[-1] if insert_prefix else 0) + count)

    def _inserted_before(orig_abs_index: int) -> int:
        j = bisect_right(insert_positions, orig_abs_index)
        return insert_prefix[j - 1] if j else 0

    def _append_noise_or_herald(
        spec: NoiseSpec,
        herald_qubits: list[int],
        targets: list[stim.GateTarget],
    ) -> None:
        nonlocal new_running
        if spec.instruction in HERALDING_INSTRUCTIONS:
            if not herald_qubits:
                return
            qn = len(herald_qubits)
            _append_heralded_with_detectors(
                noisy,
                instruction=spec.instruction,
                qubits=herald_qubits,
                gate_args=spec.gate_args,
                tag=inst.tag,
            )
            _record_insert(orig_running, qn)
            new_running += qn
        else:
            # Non-heralding noise (e.g. DEPOLARIZE2) must see the raw,
            # non-deduplicated target list so multi-qubit pairing is preserved.
            _append_noise(noisy, spec, targets)

    orig_running = 0
    new_running = 0
    for inst, entries in zip(flat, resolved):
        # Noise / herald before measurement
        for spec, position in entries:
            if position == 'before':
                _append_noise_or_herald(
                    spec,
                    _spec_herald_qubits(inst, spec, _meas_qubits(inst)),
                    _spec_targets(inst, spec),
                )

        # Remap rec(-) targets in detectors and observables
        if inst.name in ('DETECTOR', 'OBSERVABLE_INCLUDE'):
            adjusted: list[int | stim.GateTarget] = []
            for t in inst.targets_copy():
                if t.is_measurement_record_target:
                    orig_abs = orig_running + t.value
                    new_abs = orig_abs + _inserted_before(orig_abs)
                    adjusted.append(stim.target_rec(new_abs - new_running))
                else:
                    adjusted.append(t)
            noisy.append_op(inst.name, adjusted, inst.gate_args_copy(), tag=inst.tag)
        else:
            flip_probability = _combined_flip_probability(entries)
            if flip_probability is not None:
                _append_instruction_with_measurement_flip(
                    noisy,
                    inst,
                    flip_probability=flip_probability,
                )
            else:
                noisy.append_instruction(inst)

        n = _n_meas_inst(inst)
        if n > 0:
            orig_running += n
            new_running += n

        # Noise / herald after gate
        for spec, position in entries:
            if position == 'after':
                _append_noise_or_herald(
                    spec,
                    _spec_herald_qubits(inst, spec, _op_qubits(inst)),
                    _spec_targets(inst, spec),
                )

    return noisy.to_circuit()
