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


NoiseSpecLike = (
    NoiseSpec
    | tuple[str, float | Collection[float]]
    | tuple[str, float | Collection[float], str | None]
    | tuple[str, float | Collection[float], str | None, str]
    | float
)
# A tag key may map to a single spec, or a list of specs applied in order
# (e.g. both DEPOLARIZE2 and HERALDED_ERASE on the same tagged gate).
NoiseSpecOrList = NoiseSpecLike | list[NoiseSpecLike]
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


def _to_noise_spec(value: NoiseSpecLike, default_instruction: str) -> NoiseSpec:
    if isinstance(value, NoiseSpec):
        return value
    if isinstance(value, tuple):
        position = value[2] if len(value) > 2 else None
        qubits = value[3] if len(value) > 3 else 'all'
        return NoiseSpec(
            instruction=value[0],
            probability=value[1],
            position=position,
            qubits=qubits,
        )
    return NoiseSpec(instruction=default_instruction, probability=value)


def _to_noise_specs(value: NoiseSpecOrList, default_instruction: str) -> list[NoiseSpec]:
    """Normalize a ``noise_by_type`` value to a list of specs, applied in order."""
    if isinstance(value, list):
        return [_to_noise_spec(v, default_instruction) for v in value]
    return [_to_noise_spec(value, default_instruction)]


def _coerce_scalar(raw: str) -> Any:
    text = raw.strip()
    if text == 'true':
        return True
    if text == 'false':
        return False
    if text.lstrip('-').isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    try:
        return float(text)
    except ValueError:
        return text


def parse_op_tag(tag: str) -> dict[str, Any]:
    """Parse modern JSON tags and legacy key=value tag strings."""
    if not tag:
        return {}

    try:
        parsed = json.loads(tag)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    parsed: dict[str, Any] = {}
    for token in tag.split():
        if '=' in token:
            key, value = token.split('=', maxsplit=1)
            parsed[key] = _coerce_scalar(value)
        else:
            parsed[token] = True
    return parsed


def _tag_time(tag_data: Mapping[str, Any]) -> int | None:
    t = tag_data.get('t')
    if isinstance(t, int):
        return t
    if isinstance(t, float) and float(t).is_integer():
        return int(t)
    if isinstance(t, str) and t.lstrip('-').isdigit():
        return int(t)
    return None


def find_max_time(circuit: stim.Circuit) -> int:
    """Find the maximum time step in a circuit based on instruction tags."""
    max_t = 0
    for op in circuit.flattened():
        t = _tag_time(parse_op_tag(op.tag))
        if t is not None:
            max_t = max(max_t, t)
    return max_t


def _is_boundary_round(
    tag_data: Mapping[str, Any],
    *,
    max_t: int,
    leading_rounds_to_skip: int,
    trailing_rounds_to_skip: int,
) -> bool:
    # Data-qubit operations (init, final readout) are never skipped: their
    # measurement errors directly affect the logical observable (analogous to
    # Stim's before_measure_flip_probability on data qubits in a
    # phenomenological model).
    if tag_data.get('qubit_type') == 'data':
        return False
    t = _tag_time(tag_data)
    if t is None:
        return False
    if t < leading_rounds_to_skip:
        return True
    return t >= (max_t - trailing_rounds_to_skip)


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
    skip_boundary_rounds: bool = False,
    leading_rounds_to_skip: int = 2,
    trailing_rounds_to_skip: int = 2,
    excluded_tag_types: set[str] | None = None,
) -> stim.Circuit:
    """
    Apply a tag-driven noise model over the circuit.

    The tag ``type`` field alone determines the noise applied to each instruction —
    gate category (single-qubit vs two-qubit) is ignored.  The insertion point is
    determined by instruction kind by default: measurement instructions receive
    noise immediately *before* them; all other gate instructions receive it
    *after*. A :class:`NoiseSpec` may override this per entry via its ``position``
    field (``'before'``/``'after'``), or by passing a 3-tuple
    ``(instruction, probability, position)`` as the mapping value.

    ``noise_by_type`` maps any of:

    - ``type`` strings,
    - ``(type, operation)`` tuples, or
    - ``(type, operation, qubit_type)`` tuples,

    to :class:`NoiseSpec` (or a ``(instruction, probability)`` tuple / bare
    float). Lookup is most-specific-first: the 3-tuple wins over the 2-tuple,
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
    ``NoiseSpec.qubits='first'`` (or ``'second'``), or by passing a 4-tuple
    ``(instruction, probability, position, qubits)``. This is resolved per
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
    max_t = find_max_time(circuit)
    excluded_types = excluded_tag_types or set()
    _noise_map: dict[NoiseKey, list[NoiseSpec]] = {
        k: _to_noise_specs(v, 'DEPOLARIZE1') for k, v in (noise_by_type or {}).items()
    }

    def _resolve(inst: stim.CircuitInstruction) -> list[tuple[NoiseSpec, str]]:
        """Return [(spec, 'before'|'after'|'inline'), ...] in application order."""
        if inst.name not in ALL_GATE_INSTRUCTIONS:
            return []

        tag_data = parse_op_tag(inst.tag)
        op_type = tag_data.get('type')
        op_operation = tag_data.get('operation')

        should_skip = (
            (skip_boundary_rounds and _is_boundary_round(
                tag_data,
                max_t=max_t,
                leading_rounds_to_skip=leading_rounds_to_skip,
                trailing_rounds_to_skip=trailing_rounds_to_skip,
            ))
            or (isinstance(op_type, str) and op_type in excluded_types)
        )
        if should_skip:
            return []

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

def count_noise_instructions(circuit: stim.Circuit) -> dict[str, int]:
    """Count the number of noise instructions in a circuit by instruction name."""
    counts: dict[str, int] = {}
    for inst in circuit.flattened():
        if inst.name in ALL_GATE_INSTRUCTIONS:
            continue
        if inst.name not in counts:
            counts[inst.name] = 0
        targets = inst.targets_copy()
        counts[inst.name] += len(targets)
    return counts


# Reset instructions that leave their targets *alive* (ready to idle): plain
# resets and the measure-and-reset family (which end in a reset).
RESET_GATES = frozenset({'R', 'RX', 'RY', 'RZ'})
RESET_MEASURE_GATES = frozenset({'MR', 'MRX', 'MRY', 'MRZ'})
# Pure measurements leave their targets *dead* until the next reset.
PURE_MEASUREMENT_GATES = (MEASUREMENT_GATES | PAIRED_MEASUREMENT_GATES) - RESET_MEASURE_GATES


def _graph_qubits(circuit: stim.Circuit) -> set[int]:
    """Qubit indices that ever appear in a ``type="graph"`` tagged instruction.

    These are the resource/graph-state qubits: every such qubit is prepared by a
    ``type="graph"`` reset in the synthesis layer, so this captures all of them.
    """
    graph_qs: set[int] = set()
    for inst in circuit.flattened():
        if inst.name not in ALL_GATE_INSTRUCTIONS:
            continue
        if parse_op_tag(inst.tag).get('type') == 'graph':
            graph_qs.update(_op_qubits(inst))
    return graph_qs


def _is_round_idle_marker(inst: stim.CircuitInstruction) -> bool:
    """Whether *inst* is a per-round ``"idle"`` identity from the synthesis layer.

    Only these are dropped by ``strip_identities``; other tagged identities —
    notably the ``"state_prep"`` resource-state markers — carry their own noise
    channel and must survive the pass.
    """
    return inst.name == 'I' and parse_op_tag(inst.tag).get('operation') == 'idle'


def add_idle_step_markers(
    circuit: stim.Circuit,
    *,
    operation: str = "idle_step",
    op_type: str = "circuit",
    graph_op_type: str = "graph",
    strip_identities: bool = False,
) -> stim.Circuit:
    """Insert per-timestep idle identity markers into a circuit.

    Walks the circuit moment-by-moment (segments delimited by ``TICK``) and, for
    every moment that contains at least one gate, appends a tagged ``I`` marker on
    each qubit that is *alive* (reset and not yet measured) but *not* touched by a
    gate in that moment. The markers carry the JSON tag
    ``{"type": <type>, "operation": operation, "t": <round>}`` so that
    :func:`apply_tagged_noise_model` can attach idle noise via the
    ``(<type>, operation)`` key — independently of the per-round ``"idle"``
    channel emitted by the synthesis layer.

    Resource/graph-state qubits are tagged with ``graph_op_type`` ("graph");
    all other (circuit data/ancilla) qubits with ``op_type`` ("circuit"). This
    lets resource-state idling carry a different rate — or none at all, by simply
    not registering the ``(graph_op_type, operation)`` key.

    A qubit is considered alive once it is reset (``R``/``RX``/``RY``/``RZ`` or a
    measure-and-reset ``MR*``) and dead again after a pure measurement. The ``t``
    field is inherited from the timed tags present in the moment (the maximum seen)
    so ``skip_boundary_rounds`` keeps working; it is omitted for moments with no
    timed tags.

    If ``strip_identities`` is True, the synthesis layer's per-round ``"idle"``
    identity markers are dropped, and any moment that becomes empty as a result
    has its trailing ``TICK`` collapsed. Use this to *replace* the coarse
    per-round idle channel with this per-timestep one. Identities carrying a
    different ``operation`` (notably the ``"state_prep"`` resource-state markers)
    are kept, as is anything this pass itself inserts.

    This pass must be run **once, on a clean circuit, before**
    :func:`apply_tagged_noise_model`; it is not idempotent.
    """
    out = stim.Circuit()
    alive: set[int] = set()
    graph_qs = _graph_qubits(circuit)

    moment: list[stim.CircuitInstruction] = []

    def _emit_markers(qubits: list[int], qtype: str, max_t: int | None) -> None:
        if not qubits:
            return
        tag_data: dict[str, Any] = {"type": qtype, "operation": operation}
        if max_t is not None:
            tag_data["t"] = max_t
        out.append("I", qubits, tag=json.dumps(tag_data))

    def _flush_moment(closed_by_tick: bool) -> None:
        nonlocal alive
        kept = [
            inst for inst in moment
            if not (strip_identities and _is_round_idle_marker(inst))
        ]
        touched: set[int] = set()
        has_gate = False
        max_t: int | None = None
        for inst in kept:
            out.append(inst)
            if inst.name not in ALL_GATE_INSTRUCTIONS:
                continue
            has_gate = True
            touched.update(_op_qubits(inst))
            t = _tag_time(parse_op_tag(inst.tag))
            if t is not None:
                max_t = t if max_t is None else max(max_t, t)

        # Idle markers only for genuine, TICK-terminated gate moments. Resource
        # qubits are tagged separately so their idling can be priced (or zeroed)
        # independently of circuit qubits.
        if closed_by_tick and has_gate:
            idle_qubits = sorted(alive - touched)
            _emit_markers([q for q in idle_qubits if q not in graph_qs], op_type, max_t)
            _emit_markers([q for q in idle_qubits if q in graph_qs], graph_op_type, max_t)

        # Update liveness after computing idleness for this moment. (Stripped
        # ``I`` gates never affect liveness, so ``kept`` and ``moment`` agree here.)
        for inst in kept:
            if inst.name in RESET_GATES or inst.name in RESET_MEASURE_GATES:
                alive.update(_op_qubits(inst))
            elif inst.name in PURE_MEASUREMENT_GATES:
                alive.difference_update(_op_qubits(inst))

        # Collapse a moment that became empty after stripping identities.
        if closed_by_tick and not (strip_identities and not kept):
            out.append("TICK")
        moment.clear()

    for inst in circuit.flattened():
        if inst.name == 'TICK':
            _flush_moment(closed_by_tick=True)
        else:
            moment.append(inst)
    if moment:
        _flush_moment(closed_by_tick=False)

    return out


# ---------------------------------------------------------------------------
# Convenience wrappers with preset tag maps
# ---------------------------------------------------------------------------

def add_circuit_depolarizing_noise(
    circuit: stim.Circuit,
    p1: float,
    p2: float,
    *,
    single_qubit_types: Collection[str] = (),
    two_qubit_types: Collection[str] = (),
    p_idle: float = 0.0,
    idle_types: Collection[str] = ('circuit',),
    p_idle_step: float = 0.0,
    skip_boundary_rounds: bool = False,
    leading_rounds_to_skip: int = 2,
    trailing_rounds_to_skip: int = 2,
    excluded_tag_types: set[str] | None = None,
) -> stim.Circuit:
    """
    Add depolarizing noise to circuit gate operations by tag type.

    ``single_qubit_types`` maps to ``DEPOLARIZE1(p1)``;
    ``two_qubit_types`` maps to ``DEPOLARIZE2(p2)``.

    ``p_idle`` (if > 0) attaches ``DEPOLARIZE1(p_idle)`` to per-round idle
    identity markers emitted by the synthesis layer. ``idle_types`` controls
    which tag ``type`` strings carry the idle marker (defaults to
    ``('circuit',)``); the matching ``(type, 'idle')`` key is registered.

    ``p_idle_step`` (if > 0) attaches ``DEPOLARIZE1(p_idle_step)`` to per-timestep
    idle markers via the ``('circuit', 'idle_step')`` key. The caller must have run
    :func:`add_idle_step_markers` on the circuit beforehand for this to take effect.
    """
    noise_by_type: dict[NoiseKey, NoiseSpec] = {t: NoiseSpec('DEPOLARIZE1', p1) for t in single_qubit_types}
    noise_by_type.update({t: NoiseSpec('DEPOLARIZE2', p2) for t in two_qubit_types})
    if p_idle > 0.0:
        for t in idle_types:
            noise_by_type[(t, 'idle')] = NoiseSpec('DEPOLARIZE1', p_idle)
    if p_idle_step > 0.0:
        noise_by_type[('circuit', 'idle_step')] = NoiseSpec('DEPOLARIZE1', p_idle_step)
    return apply_tagged_noise_model(
        circuit,
        noise_by_type=noise_by_type,
        skip_boundary_rounds=skip_boundary_rounds,
        leading_rounds_to_skip=leading_rounds_to_skip,
        trailing_rounds_to_skip=trailing_rounds_to_skip,
        excluded_tag_types=excluded_tag_types,
    )


def add_depolarizing_to_measurements(
    circuit: stim.Circuit,
    p: float,
    *,
    measurement_types: Collection[str] = (),
    skip_boundary_rounds: bool = False,
    leading_rounds_to_skip: int = 2,
    trailing_rounds_to_skip: int = 2,
    excluded_tag_types: set[str] | None = None,
) -> stim.Circuit:
    """
    Add ``DEPOLARIZE1(p)`` immediately before measurement instructions by tag type.

    ``measurement_types`` lists the tag-type strings that should receive noise.
    """
    noise_by_type: dict[str, NoiseSpec] = {t: NoiseSpec('DEPOLARIZE1', p) for t in measurement_types}
    return apply_tagged_noise_model(
        circuit,
        noise_by_type=noise_by_type,
        skip_boundary_rounds=skip_boundary_rounds,
        leading_rounds_to_skip=leading_rounds_to_skip,
        trailing_rounds_to_skip=trailing_rounds_to_skip,
        excluded_tag_types=excluded_tag_types,
    )


def add_selective_heralded_erasure(
    circuit: stim.Circuit,
    *,
    p_fusion_meas: float,
    p_graph_prep_step: float,
    fusion_measurement_types: Collection[str] = ('fusion_meas',),
    graph_prep_types: Collection[str] = ('graph_prep', 'graph_entangle'),
    skip_boundary_rounds: bool = False,
    leading_rounds_to_skip: int = 2,
    trailing_rounds_to_skip: int = 2,
) -> stim.Circuit:
    """
    Add ``HERALDED_ERASE`` noise by tag type.

    - Measurement instructions whose type is in ``fusion_measurement_types``
      receive ``HERALDED_ERASE(p_fusion_meas)`` inserted immediately before them.
    - Gate instructions whose type is in ``graph_prep_types`` receive
      ``HERALDED_ERASE(p_graph_prep_step)`` inserted immediately after them.

    All ``rec(-)`` targets in DETECTOR and OBSERVABLE_INCLUDE instructions are
    remapped automatically.
    """
    noise_by_type: dict[str, NoiseSpec] = {
        t: NoiseSpec('HERALDED_ERASE', p_fusion_meas) for t in fusion_measurement_types
    }
    noise_by_type.update({
        t: NoiseSpec('HERALDED_ERASE', p_graph_prep_step) for t in graph_prep_types
    })
    return apply_tagged_noise_model(
        circuit,
        noise_by_type=noise_by_type,
        skip_boundary_rounds=skip_boundary_rounds,
        leading_rounds_to_skip=leading_rounds_to_skip,
        trailing_rounds_to_skip=trailing_rounds_to_skip,
    )


def add_heralded_pauli_channel(
    circuit: stim.Circuit,
    *,
    p_x: float,
    p_y: float,
    p_z: float,
    p_i: float = 0.0,
    measurement_types: Collection[str] = (),
    gate_types: Collection[str] = (),
    skip_boundary_rounds: bool = False,
    leading_rounds_to_skip: int = 2,
    trailing_rounds_to_skip: int = 2,
    excluded_tag_types: set[str] | None = None,
) -> stim.Circuit:
    """
    Add ``HERALDED_PAULI_CHANNEL_1`` noise by tag type.

    Unlike :func:`add_selective_heralded_erasure` — where a herald implies a
    uniform depolarizing (I/X/Y/Z) error — this channel carries an independent
    probability for each Pauli component. On each targeted qubit, exactly one of
    the following happens:

    - with probability ``p_i``: heralded, no error (identity),
    - with probability ``p_x`` / ``p_y`` / ``p_z``: heralded X / Y / Z error,
    - otherwise: nothing, and no herald.

    A herald measurement bit (1 if any of the four cases fired) is produced per
    qubit, exactly like ``HERALDED_ERASE``. ``HERALDED_ERASE(p)`` is the special
    case ``p_i = p_x = p_y = p_z = p / 4``.

    - Measurement instructions whose type is in ``measurement_types`` receive the
      channel inserted immediately *before* them.
    - Gate instructions whose type is in ``gate_types`` receive it immediately
      *after* them.

    All ``rec(-)`` targets in DETECTOR and OBSERVABLE_INCLUDE instructions are
    remapped automatically.
    """
    # stim gate-arg order is (p_I, p_X, p_Y, p_Z).
    spec = NoiseSpec(
        'HERALDED_PAULI_CHANNEL_1',
        probability=(float(p_i), float(p_x), float(p_y), float(p_z)),
    )
    noise_by_type: dict[NoiseKey, NoiseSpec] = {t: spec for t in measurement_types}
    noise_by_type.update({t: spec for t in gate_types})
    return apply_tagged_noise_model(
        circuit,
        noise_by_type=noise_by_type,
        skip_boundary_rounds=skip_boundary_rounds,
        leading_rounds_to_skip=leading_rounds_to_skip,
        trailing_rounds_to_skip=trailing_rounds_to_skip,
        excluded_tag_types=excluded_tag_types,
    )


def add_depolarizing_to_all(
    circuit: stim.Circuit,
    *,
    p1: float,
    p2: float,
    pm: float,
    p_idle: float = 0.0,
    p_idle_step: float = 0.0,
    p_state_prep: float = 0.0,
    skip_boundary_rounds: bool = False,
    leading_rounds_to_skip: int = 2,
    trailing_rounds_to_skip: int = 2,
) -> stim.Circuit:
    """
    Apply depolarizing noise to all known graph/circuit tagged operations.

    Mapping used:

    - ``(type, reset)`` and ``(type, single_qubit_gate)`` -> ``DEPOLARIZE1(p1)``
    - ``(type, two_qubit_gate)`` -> ``DEPOLARIZE2(p2)``
    - ``(type, single_qubit_measurement)`` and
      ``(type, fusion_measurement)`` -> ``DEPOLARIZE1(pm)``
    - ``(circuit, idle)`` -> ``DEPOLARIZE1(p_idle)`` (only when ``p_idle > 0``)
    - ``(circuit, idle_step)`` -> ``DEPOLARIZE1(p_idle_step)`` (only when
      ``p_idle_step > 0``; requires :func:`add_idle_step_markers` to have run)
    - ``(type, state_prep)`` -> ``DEPOLARIZE1(p_state_prep)`` (only when
      ``p_state_prep > 0``): one error per *qubit* of a finished resource
      state, applied after the preparing entangling gates

    for both ``type in {circuit, graph}``.
    """
    noise_by_type: dict[NoiseKey, NoiseSpec] = {}
    for op_type in ('circuit', 'graph'):
        noise_by_type[(op_type, 'reset')] = NoiseSpec('DEPOLARIZE1', p1)
        noise_by_type[(op_type, 'single_qubit_gate')] = NoiseSpec('DEPOLARIZE1', p1)
        noise_by_type[(op_type, 'two_qubit_gate')] = NoiseSpec('DEPOLARIZE2', p2)
        noise_by_type[(op_type, 'single_qubit_measurement')] = NoiseSpec('DEPOLARIZE1', pm)
        noise_by_type[(op_type, 'fusion_measurement')] = NoiseSpec('DEPOLARIZE1', pm)
        if p_state_prep > 0.0:
            noise_by_type[(op_type, 'state_prep')] = NoiseSpec('DEPOLARIZE1', p_state_prep)
    if p_idle > 0.0:
        noise_by_type[('circuit', 'idle')] = NoiseSpec('DEPOLARIZE1', p_idle)
    if p_idle_step > 0.0:
        noise_by_type[('circuit', 'idle_step')] = NoiseSpec('DEPOLARIZE1', p_idle_step)

    return apply_tagged_noise_model(
        circuit,
        noise_by_type=noise_by_type,
        skip_boundary_rounds=skip_boundary_rounds,
        leading_rounds_to_skip=leading_rounds_to_skip,
        trailing_rounds_to_skip=trailing_rounds_to_skip,
    )


def add_hybrid_depolarizing_erasure(
    circuit: stim.Circuit,
    *,
    p1: float,
    p2: float,
    pm: float,
    pl: float,
    pf: float,
    p_idle: float = 0.0,
    p_idle_step: float = 0.0,
    p_state_prep: float = 0.0,
    skip_boundary_rounds: bool = False,
    leading_rounds_to_skip: int = 2,
    trailing_rounds_to_skip: int = 2,
) -> stim.Circuit:
    """
    Mixed model:

    - Circuit operations receive depolarizing noise:
      ``reset``/``single_qubit_gate`` -> ``DEPOLARIZE1(p1)``
      ``two_qubit_gate`` -> ``DEPOLARIZE2(p2)``
      ``single_qubit_measurement`` -> ``DEPOLARIZE1(pm)``
      ``idle`` -> ``DEPOLARIZE1(p_idle)`` (only when ``p_idle > 0``)
      ``idle_step`` -> ``DEPOLARIZE1(p_idle_step)`` (only when ``p_idle_step > 0``;
      requires :func:`add_idle_step_markers` to have run)
    - Graph prep gates receive heralded erasure:
      ``reset``/``single_qubit_gate``/``two_qubit_gate`` -> ``HERALDED_ERASE(pl)``
    - Graph fusion measurements receive heralded erasure:
      ``fusion_measurement`` -> ``HERALDED_ERASE(pf)``
    - Finished resource states receive one heralded erasure *per qubit*
      (only when ``p_state_prep > 0``):
      ``(graph, state_prep)`` and ``(circuit, state_prep)`` ->
      ``HERALDED_ERASE(p_state_prep)``. Prefer this over the per-gate
      ``two_qubit_gate`` channel when the loss is a property of the prepared
      state rather than of each entangling gate — a degree-``d`` qubit gets
      one draw here instead of ``d``.
    """
    noise_by_type: dict[NoiseKey, NoiseSpec] = {
        ('circuit', 'reset'): NoiseSpec('DEPOLARIZE1', p1),
        ('circuit', 'single_qubit_gate'): NoiseSpec('DEPOLARIZE1', p1),
        ('circuit', 'two_qubit_gate'): NoiseSpec('DEPOLARIZE2', p2),
        ('circuit', 'single_qubit_measurement'): NoiseSpec('DEPOLARIZE1', pm),
        ('graph', 'reset'): NoiseSpec('HERALDED_ERASE', pl),
        ('graph', 'single_qubit_gate'): NoiseSpec('HERALDED_ERASE', pl),
        ('graph', 'two_qubit_gate'): NoiseSpec('HERALDED_ERASE', pl),
        ('graph', 'fusion_measurement'): NoiseSpec('HERALDED_ERASE', pf),
    }
    if p_state_prep > 0.0:
        for op_type in ('graph', 'circuit'):
            noise_by_type[(op_type, 'state_prep')] = NoiseSpec('HERALDED_ERASE', p_state_prep)
    if p_idle > 0.0:
        noise_by_type[('circuit', 'idle')] = NoiseSpec('DEPOLARIZE1', p_idle)
    if p_idle_step > 0.0:
        noise_by_type[('circuit', 'idle_step')] = NoiseSpec('DEPOLARIZE1', p_idle_step)

    return apply_tagged_noise_model(
        circuit,
        noise_by_type=noise_by_type,
        skip_boundary_rounds=skip_boundary_rounds,
        leading_rounds_to_skip=leading_rounds_to_skip,
        trailing_rounds_to_skip=trailing_rounds_to_skip,
    )
