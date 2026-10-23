#!/usr/bin/env python
"""Leading-order logical error rate of a surgery under pure (heralded) erasure.

With erasure the only noise, the decoder fails exactly when the erased set
supports an undetectable logical, and then with probability 1/2 (the Pauli on
the erased set is uniformly random). So

    LER = 1/2 * Pr[erased set contains a logical]
        ~ 1/2 * sum over minimum-weight logicals L of prod_{e in L} q_e      (p -> 0)

The logicals are read off the circuit, not assumed. stim's
explain_detector_error_model_errors gives, for every HERALDED_ERASE target (an
independent erasure location), the detectors and observables its X and its Z
flip. Each such effect must touch at most two detectors (graphlike; checked),
so the effects are the edges of a decoding graph with one boundary node, and a
logical is a closed walk with odd observable parity. Edges with identical
(detectors, observable) signature are one class of multiplicity m (the
locations that produce it), erased with q = 1 - (1 - p_e)^m.

The minimum weight w and the weighted count C = sum over minimal logicals of
prod m_e come from a BFS over (node, observable parity) from the boundary, so
at small p_e, LER ~ 1/2 * C * p_e^w. The Lambda figure draws
LER_lo(5) / LER_lo(7) -- Lambda for one d -> d+2 step -- as its
leading-order band.
"""
from __future__ import annotations

from collections import defaultdict, deque

from run_threshold_experiment import apply_tagged_noise_model, cached_circuit, noise_model_from_config

BOUNDARY = -1


def erasure_edges(config: dict, distance: int, observable: str
                  ) -> tuple[dict[tuple[frozenset, int], int], list[str]]:
    """{(detectors, observable parity): multiplicity} over the X and Z effects
    of every erasure location, plus structural warnings."""
    sweep = config["sweep"]
    circuit = cached_circuit(config["builder"]["name"], distance, observable, sweep["merge_types"][0])
    channels = [ch for ch in config["noise_model"] if ch.get("probability") == "p_e"]
    # Any small rate: only which locations exist matters, not their probability.
    noisy = apply_tagged_noise_model(circuit, noise_by_type=noise_model_from_config(channels, 1e-3))
    explained = noisy.explain_detector_error_model_errors(reduce_to_one_representative_error=False)

    # location -> {pauli: (detectors, obs parity)}
    effects: dict[tuple, dict[str, tuple[frozenset, int]]] = defaultdict(dict)
    warnings = []
    for err in explained:
        dets = frozenset(t.dem_target.val for t in err.dem_error_terms
                         if t.dem_target.is_relative_detector_id())
        obs = sum(1 for t in err.dem_error_terms if t.dem_target.is_logical_observable_id()) % 2
        for loc in err.circuit_error_locations:
            if loc.instruction_targets.gate != "HERALDED_ERASE":
                continue
            paulis = loc.flipped_pauli_product
            if not paulis:
                continue  # the herald flag itself: a record flip, no Pauli
            if len(paulis) != 1:
                warnings.append(f"multi-qubit erasure effect at tick {loc.tick_offset}")
                continue
            g = paulis[0].gate_target
            pauli = "X" if g.is_x_target else "Z" if g.is_z_target else "Y"
            key = (loc.tick_offset,
                   tuple((f.instruction_offset, f.iteration_index) for f in loc.stack_frames),
                   g.value)
            effects[key][pauli] = (dets, obs)

    # An erased location applies any Pauli, so what it can do is the span of
    # its X and Z effects. A detector flipped only by one location's effects
    # (the fusion's own consistency check: every single outcome flip trips it)
    # must be left unflipped by any logical, so only span elements avoiding
    # the location's private detectors can take part. Each location must leave
    # exactly one such element, touching at most two detectors: that is its
    # edge, and the decoding problem is ordinary bond percolation.
    owners: dict[int, set] = defaultdict(set)
    for key, eff in effects.items():
        for dets, _ in eff.values():
            for x in dets:
                owners[x].add(key)
    classes: dict[tuple[frozenset, int], int] = defaultdict(int)
    for key, eff in effects.items():
        ex = eff.get("X", (frozenset(), 0))
        ez = eff.get("Z", (frozenset(), 0))
        span = {ex, ez, (ex[0] ^ ez[0], ex[1] ^ ez[1])}
        private = {x for dets, _ in span for x in dets if owners[x] == {key}}
        usable = [(dets, obs) for dets, obs in span
                  if (dets or obs) and not (dets & private)]
        if len(usable) != 1:
            warnings.append(f"location with {len(usable)} usable effects")
            continue
        dets, obs = usable[0]
        if len(dets) > 2:
            warnings.append(f"non-graphlike usable effect ({len(dets)} detectors)")
            continue
        classes[(dets, obs)] += 1
    return dict(classes), warnings


def _graph(classes: dict[tuple[frozenset, int], int], q_of) -> dict[int, list[tuple[int, int, float]]]:
    adj: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for (dets, obs), m in classes.items():
        nodes = sorted(dets)
        a, b = (nodes[0], nodes[1]) if len(nodes) == 2 else (nodes[0] if nodes else BOUNDARY, BOUNDARY)
        w = q_of(m)
        adj[a].append((b, obs, w))
        if a != b:
            adj[b].append((a, obs, w))
    return adj


def min_logical(classes, q_of, source=BOUNDARY) -> tuple[int | None, float]:
    """(w, sum over minimal logicals through `source` of prod q_of(m)).

    The BFS counts closed walks, and a minimal logical of weight w >= 2 is
    one cycle walked in either direction: two walks, halved here. (A w = 1
    logical, a self-loop, is a single walk.)"""
    adj = _graph(classes, q_of)
    dist = {(source, 0): 0}
    weight = {(source, 0): 1.0}
    queue = deque([(source, 0)])
    while queue:
        node, par = queue.popleft()
        for nb, obs, w in adj[node]:
            state = (nb, par ^ obs)
            nd = dist[(node, par)] + 1
            if state not in dist:
                dist[state] = nd
                weight[state] = 0.0
                queue.append(state)
            if dist[state] == nd:
                weight[state] += weight[(node, par)] * w
    target = (source, 1)
    if target not in dist:
        return None, 0.0
    w = dist[target]
    return w, weight[target] / 2 if w >= 2 else weight[target]


