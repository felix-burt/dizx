"""Helpers shared by several test modules."""
from dizx.transformations import cut_edges


def has_matching_boundary(sed, S):
    """Whether dE(S) is a matching: no two cut edges share an endpoint."""
    ends = [x for u, v, _ in cut_edges(sed, S) for x in (u, v)]
    return len(ends) == len(set(ends))
