"""
Sinter integration for the pure-pymatching heralded-erasure decoder.

`PyMatchingHeraldedDecoder` is a `sinter.Decoder` that decodes
``HERALDED_ERASE`` circuits with plain MWPM, reusing
:func:`dizx.erasure_matching.build_erasure_matching_data` (matching circuit +
tag-driven erasure->edge map) instead of mwpf's heralded-DEM construction.

The decoder is bound to one (noisy) circuit, so use one instance per circuit::

    dec = PyMatchingHeraldedDecoder().with_circuit(circuit=noisy_circuit)
    sinter.collect(
        tasks=[sinter.Task(circuit=noisy_circuit, decoder="pymatching_heralded")],
        custom_decoders={"pymatching_heralded": dec},
        ...,
    )

Per shot the herald bits select which erasure edges fired; those edges are set
to weight 0 (``p = 0.5``, a random-Pauli erasure), the syndrome is decoded, and
the edges are restored.  ``p ~ 1`` is *not* used: the resulting large negative
weight breaks pymatching (see :mod:`dizx.erasure_matching`).

The matching circuit keeps each herald detector as a target-less ``DETECTOR``
(``keep_herald_detectors``), so the matching graph is numbered exactly like
the sampled detectors, with every herald an edgeless node.  The bit-packed
sinter syndrome then goes to pymatching with only the herald bits cleared: no
unpacking, and no slicing the heralds out.

How the weights are applied depends on the installed pymatching and the
circuit.  Where ``decode_batch`` accepts ``edge_reweights`` the whole batch
goes over in one call and the weights are changed in place; otherwise, or when
erasing some edge would change which observables it flips (reweighting changes
weights only), each shot falls back to ``_decode_legacy``, which invalidates
the compiled graph and costs ~50x more.  See ``SUPPORTS_EDGE_REWEIGHTS``.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pymatching
import sinter
import stim

from dizx.erasure_matching import build_erasure_matching_data, _reweighted_dem

# Whether the installed pymatching accepts per-shot ``edge_reweights``.
#
# Stock pymatching (<=2.4.0) has no such argument, so every per-shot weight
# change must go through ``add_edge``/``add_boundary_edge``, which invalidates
# the compiled matching graph and makes the next ``decode`` rebuild it in full
# -- measured at 30-60x the cost of the decode itself.  Builds carrying the
# reweighting work (github.com/felix-burt/PyMatching @ edge-reweighting) apply
# the new weights in place and restore them after the shot, which also makes
# correlated matching affordable: no per-shot DEM rebuild is needed.
_DECODE_BATCH_PARAMS = inspect.signature(pymatching.Matching.decode_batch).parameters
SUPPORTS_EDGE_REWEIGHTS = "edge_reweights" in _DECODE_BATCH_PARAMS
# Whether ``decode_batch`` also takes the table form (``edge_reweight_table`` +
# CSR ``indptr``/``indices``; github.com/felix-burt/PyMatching @
# edge-reweight-table).  The table is parsed once per batch instead of one
# array per shot, and building the CSR selection is pure numpy.
SUPPORTS_REWEIGHT_TABLE = "edge_reweight_table" in _DECODE_BATCH_PARAMS

# Cap on the (shots x erasure-rows) boolean mask built while assembling
# per-shot reweights, so a large sinter batch cannot blow up memory.
_REWEIGHT_MASK_BUDGET = 4_000_000


def _fault_ids_set(observable_ids) -> set[int]:
    return {int(i) for i in observable_ids}


@dataclass
class _EdgeSpec:
    """One graphlike edge controlled by a herald, with its baseline snapshot."""

    endpoints: tuple[int, ...]          # 1 (boundary) or 2 detector nodes
    on_fault_ids: set[int]              # observables flipped when erased
    base_fault_ids: set[int]            # baseline (erasure-off) fault ids
    base_weight: float                  # baseline (erasure-off) weight


def _parse_components(inst: stim.DemInstruction) -> list[tuple[list[int], set[int]]]:
    """Split a DEM error instruction into graphlike components.

    Returns a list of ``(detector_ids, observable_ids)`` — one per ``^``-
    separated component.
    """
    components: list[tuple[list[int], set[int]]] = []
    dets: list[int] = []
    obs: set[int] = set()
    for t in inst.targets_copy():
        if t.is_separator():
            components.append((dets, obs))
            dets, obs = [], set()
        elif t.is_relative_detector_id():
            dets.append(t.val)
        elif t.is_logical_observable_id():
            obs.add(t.val)
    components.append((dets, obs))
    return components


@dataclass
class PyMatchingHeraldedCompiledDecoder(sinter.CompiledDecoder):
    """Compiled heralded-erasure MWPM decoder (one circuit).

    Every detector id here is in the sampling circuit's numbering, which the
    matching graph shares (see the module docstring).
    """

    matching: Any                       # pymatching.Matching (erasure edges off)
    herald_edges: tuple[tuple[_EdgeSpec, ...], ...]  # per erasure id k
    herald_mask: np.ndarray             # bool, len = num_dets
    det_to_herald_id: np.ndarray        # intp, len = num_dets
    num_dets: int
    num_obs: int
    erased_weight: float
    base_dem: stim.DetectorErrorModel   # matching-circuit DEM (for rebuild path)
    erased_edge_prob: float
    # Two-pass correlated matching.  On the legacy path, edge weights set by
    # ``add_edge`` are not reflected in the correlation bookkeeping (fixed at
    # graph-build time), so correlations force a per-shot matching rebuild
    # instead of the in-place edge swap.  The ``edge_reweights`` path updates
    # both graphs, so neither rebuild is needed.
    enable_correlations: bool
    # ``edge_reweights`` rows [node1, node2 (-1 for a boundary edge), weight]
    # for every heralded erasure edge, and for each row the column (into
    # ``herald_byte``/``herald_shift``) of the herald that switches it on.
    # None when the batch path is unavailable (see ``_use_batch_reweights``).
    reweight_rows: np.ndarray | None = None
    row_herald_col: np.ndarray | None = None
    # Where each herald's bit lives in a bit-packed shot: byte and bit offset.
    herald_byte: np.ndarray | None = None
    herald_shift: np.ndarray | None = None
    # Per-byte AND mask clearing every herald bit: a herald is an edgeless node
    # in the matching graph, so a fired one would be unmatchable.
    data_byte_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, np.uint8))

    @property
    def _use_batch_reweights(self) -> bool:
        return SUPPORTS_EDGE_REWEIGHTS and self.reweight_rows is not None

    def decode_shots_bit_packed(
        self, *, bit_packed_detection_event_data: np.ndarray
    ) -> np.ndarray:
        if self._use_batch_reweights:
            return self._decode_batch_reweights(bit_packed_detection_event_data)
        return self._decode_legacy(bit_packed_detection_event_data)

    def _shot_reweights(self, data: np.ndarray) -> dict[str, Any]:
        """``decode_batch`` reweight kwargs for a bit-packed batch.

        The herald bits are read straight out of the packed bytes.  Where the
        table form is supported the result is a CSR selection of rows from
        ``reweight_rows``; otherwise one ``edge_reweights`` array per shot, each
        a view into one gathered array rather than a fancy-indexed copy.
        Chunked so the (shots x rows) mask stays bounded however large a batch
        sinter hands over.
        """
        n_rows = self.reweight_rows.shape[0]
        if n_rows == 0:
            return {}
        chunk = max(1, _REWEIGHT_MASK_BUDGET // n_rows)
        indices: list[np.ndarray] = []
        counts: list[np.ndarray] = []
        for lo in range(0, data.shape[0], chunk):
            block = data[lo : lo + chunk]
            hits = (block[:, self.herald_byte] >> self.herald_shift) & 1
            row_mask = hits[:, self.row_herald_col].view(bool)
            indices.append(np.nonzero(row_mask)[1])
            counts.append(row_mask.sum(axis=1))
        flat = np.concatenate(indices)
        indptr = np.zeros(data.shape[0] + 1, dtype=np.int64)
        np.cumsum(np.concatenate(counts), out=indptr[1:])
        if SUPPORTS_REWEIGHT_TABLE:
            return dict(
                edge_reweight_table=self.reweight_rows,
                edge_reweight_indptr=indptr,
                edge_reweight_indices=flat.astype(np.int64, copy=False),
            )
        return dict(edge_reweights=np.split(self.reweight_rows[flat], indptr[1:-1]))

    def _decode_batch_reweights(self, data: np.ndarray) -> np.ndarray:
        """One ``decode_batch`` call, with per-shot erasure weights.

        Equivalent to :meth:`_decode_legacy` but hands the whole batch to
        pymatching in a single call: the erased edges are reweighted in place
        for the shot and restored afterwards, so the compiled matching graph is
        never invalidated.  Only edge *weights* change, which is why the caller
        must have verified that every erasure edge's on/off fault ids agree
        (see ``compile_decoder_for_dem``).
        """
        num_obs_bytes = (self.num_obs + 7) // 8
        packed = np.asarray(
            self.matching.decode_batch(
                data & self.data_byte_mask,
                bit_packed_shots=True,
                **self._shot_reweights(data),
                enable_correlations=self.enable_correlations,
                bit_packed_predictions=True,
            ),
            dtype=np.uint8,
        )
        if packed.shape[1] == num_obs_bytes:
            return packed
        predictions = np.zeros((data.shape[0], num_obs_bytes), dtype=np.uint8)
        n = min(num_obs_bytes, packed.shape[1])
        predictions[:, :n] = packed[:, :n]
        return predictions

    def _decode_legacy(self, bit_packed_detection_event_data: np.ndarray) -> np.ndarray:
        """Per-shot decoding for pymatching builds without ``edge_reweights``."""
        num_shots = bit_packed_detection_event_data.shape[0]
        num_obs_bytes = (self.num_obs + 7) // 8
        predictions = np.zeros((num_shots, num_obs_bytes), dtype=np.uint8)
        ec = self.enable_correlations

        for shot in range(num_shots):
            row = bit_packed_detection_event_data[shot]
            active = np.flatnonzero(
                np.unpackbits(row, count=self.num_dets, bitorder="little")
            )
            fired = self.det_to_herald_id[active[self.herald_mask[active]]]
            syndrome = np.unpackbits(
                row & self.data_byte_mask, count=self.num_dets, bitorder="little"
            )

            if ec:
                # Correct under correlated matching: rebuild the shot's matching.
                if len(fired):
                    shot_dem = _reweighted_dem(
                        self.base_dem, set(int(k) for k in fired), self.erased_edge_prob
                    )
                    matching = pymatching.Matching.from_detector_error_model(
                        shot_dem, enable_correlations=True
                    )
                else:
                    matching = self.matching
                pred = matching.decode(syndrome, enable_correlations=True)
            else:
                # Fast in-place edge swap (valid only without correlations).
                touched: list[_EdgeSpec] = []
                for k in fired:
                    for spec in self.herald_edges[k]:
                        self._set_edge(spec.endpoints, spec.on_fault_ids, self.erased_weight)
                        touched.append(spec)
                try:
                    pred = self.matching.decode(syndrome)
                finally:
                    for spec in touched:
                        self._set_edge(spec.endpoints, spec.base_fault_ids, spec.base_weight)

            packed = np.packbits(np.asarray(pred, dtype=np.uint8), bitorder="little")
            predictions[shot, : packed.size] = packed
        return predictions

    def _set_edge(self, endpoints, fault_ids, weight) -> None:
        if len(endpoints) == 1:
            self.matching.add_boundary_edge(
                endpoints[0], fault_ids=set(fault_ids), weight=weight,
                merge_strategy="replace",
            )
        else:
            self.matching.add_edge(
                endpoints[0], endpoints[1], fault_ids=set(fault_ids), weight=weight,
                merge_strategy="replace",
            )


@dataclass
class PyMatchingHeraldedDecoder(sinter.Decoder):
    """Sinter `Decoder` for heralded erasure, pure pymatching.

    Bind a circuit with :meth:`with_circuit` before passing to ``sinter.collect``.
    """

    circuit: stim.Circuit | None = None
    erased_edge_prob: float = 0.5
    decompose: bool = False
    enable_correlations: bool = False

    def with_circuit(self, circuit: stim.Circuit | None) -> "PyMatchingHeraldedDecoder":
        self.circuit = None if circuit is None else circuit.copy()
        return self

    def compile_decoder_for_dem(
        self, *, dem: stim.DetectorErrorModel
    ) -> PyMatchingHeraldedCompiledDecoder:
        assert self.circuit is not None, (
            "PyMatchingHeraldedDecoder requires a circuit; call .with_circuit(...)"
        )
        data = build_erasure_matching_data(self.circuit, decompose=self.decompose)
        num_dets = self.circuit.num_detectors
        # Erasure edges off; detectors numbered as sampled (module docstring).
        matching = pymatching.Matching.from_detector_error_model(
            _reweighted_dem(data.base_dem, fired=set(), erased_edge_prob=self.erased_edge_prob),
            enable_correlations=self.enable_correlations,
        )
        erased_weight = math.log(
            (1.0 - self.erased_edge_prob) / self.erased_edge_prob
        )

        # Group erasure edges by erasure id k, snapshotting each baseline edge.
        per_k: list[list[_EdgeSpec]] = [[] for _ in range(data.n_erasures)]
        for inst in data.base_dem.flattened():
            if inst.type != "error" or not inst.tag.startswith("erase_"):
                continue
            k = int(inst.tag[len("erase_"):])
            for det_ids, obs_ids in _parse_components(inst):
                if not 1 <= len(det_ids) <= 2:
                    # 0-detector (observable-only) or hyperedge components cannot
                    # be represented as a single matching edge; skip.
                    continue
                endpoints = tuple(det_ids)
                if len(endpoints) == 1:
                    base = matching.get_boundary_edge_data(endpoints[0])
                else:
                    base = matching.get_edge_data(endpoints[0], endpoints[1])
                per_k[k].append(
                    _EdgeSpec(
                        endpoints=endpoints,
                        on_fault_ids=_fault_ids_set(obs_ids),
                        base_fault_ids=set(base["fault_ids"]),
                        base_weight=float(base["weight"]),
                    )
                )

        herald_mask = np.zeros(num_dets, dtype=bool)
        det_to_herald_id = np.zeros(num_dets, dtype=np.intp)
        for k, hd in enumerate(data.herald_det_for_erase):
            if hd is not None:
                herald_mask[hd] = True
                det_to_herald_id[hd] = k
        # Clear every herald, including multi-bit ones no erasure maps to.
        any_herald = np.zeros(num_dets, dtype=bool)
        any_herald[data.herald_det_ids] = True
        data_byte_mask = ~np.packbits(any_herald, bitorder="little")

        # ``edge_reweights`` changes weights only, so the batch path is valid
        # exactly when erasing an edge never changes which observables it
        # flips.  That holds for every circuit measured so far, but a mismatch
        # would silently mispredict, so fall back rather than assume it.
        fault_ids_stable = all(
            spec.on_fault_ids == spec.base_fault_ids
            for specs in per_k
            for spec in specs
        )
        reweight_rows = row_herald_col = herald_byte = herald_shift = None
        if fault_ids_stable and SUPPORTS_EDGE_REWEIGHTS:
            # One column per herald detector; an erasure without a herald
            # detector can never fire, so its edges get no rows.
            herald_dets = [hd for hd in data.herald_det_for_erase if hd is not None]
            col_of_det = {hd: c for c, hd in enumerate(herald_dets)}
            rows: list[tuple[float, float, float]] = []
            cols: list[int] = []
            for k, specs in enumerate(per_k):
                hd = data.herald_det_for_erase[k]
                if hd is None:
                    continue
                for spec in specs:
                    node2 = spec.endpoints[1] if len(spec.endpoints) == 2 else -1
                    rows.append((spec.endpoints[0], node2, erased_weight))
                    cols.append(col_of_det[hd])
            reweight_rows = np.asarray(rows, dtype=np.float64).reshape(-1, 3)
            row_herald_col = np.asarray(cols, dtype=np.intp)
            herald_arr = np.asarray(herald_dets, dtype=np.intp)
            herald_byte = herald_arr >> 3
            herald_shift = (herald_arr & 7).astype(np.uint8)

        return PyMatchingHeraldedCompiledDecoder(
            matching=matching,
            herald_edges=tuple(tuple(specs) for specs in per_k),
            herald_mask=herald_mask,
            det_to_herald_id=det_to_herald_id,
            num_dets=num_dets,
            num_obs=self.circuit.num_observables,
            erased_weight=erased_weight,
            base_dem=data.base_dem,
            erased_edge_prob=self.erased_edge_prob,
            enable_correlations=self.enable_correlations,
            reweight_rows=reweight_rows,
            row_herald_col=row_herald_col,
            herald_byte=herald_byte,
            herald_shift=herald_shift,
            data_byte_mask=data_byte_mask,
        )
