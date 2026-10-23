"""Helpers shared by several test modules."""
import stim

from dizx.transformations import cut_edges


def has_matching_boundary(sed, S):
    """Whether dE(S) is a matching: no two cut edges share an endpoint."""
    ends = [x for u, v, _ in cut_edges(sed, S) for x in (u, v)]
    return len(ends) == len(set(ends))


def noisy(c: stim.Circuit, p: float = 1e-3, gates=("CX", "CZ")) -> stim.Circuit:
    """Depolarising noise after each of ``gates`` and flips on every measurement."""
    out = stim.Circuit()
    for inst in c.flattened():
        if inst.name in gates:
            out.append(inst)
            out.append("DEPOLARIZE2", inst.targets_copy(), p)
        elif inst.name in ("M", "MX", "MPP"):
            out.append(inst.name, inst.targets_copy(), p)
        else:
            out.append(inst)
    return out


def distance(c: stim.Circuit, gates=("CX", "CZ")) -> int:
    """Circuit distance under ``noisy(c, gates=gates)``."""
    return len(noisy(c, gates=gates).search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False))


def check_deterministic(c: stim.Circuit) -> None:
    """Noiseless detectors and observables never fire; the DEM builds."""
    dets, obs = c.compile_detector_sampler().sample(200, separate_observables=True)
    assert not dets.any() and not obs.any()
    c.detector_error_model()
