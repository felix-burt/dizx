import numpy as np
import pytest
import scipy.sparse as sp

from dizx._f2 import (
    SparseEchelon, mask_indices, rank, solve, sp_canonical, sp_matmul, sp_rank, sp_row,
    sp_rows, sp_solver,
)


def random_sparse(rng, m, n, density):
    return (rng.random((m, n)) < density).astype(np.uint8)


@pytest.mark.parametrize("seed", range(2))
def test_sparse_rank_and_product_match_dense(seed):
    rng = np.random.default_rng(seed)
    A = random_sparse(rng, 40, 60, 0.08)
    B = random_sparse(rng, 60, 30, 0.1)
    assert sp_rank(sp_canonical(A)) == rank(A)
    assert np.array_equal(sp_matmul(A, B).toarray(), A.astype(int) @ B % 2)


@pytest.mark.parametrize("seed", range(2))
def test_sparse_solver_matches_dense(seed):
    rng = np.random.default_rng(seed)
    A = random_sparse(rng, 50, 80, 0.06)
    keep = SparseEchelon()
    rows = [r for r in range(len(A)) if keep.add(np.flatnonzero(A[r]))]
    M = sp_canonical(A[rows])
    solve_sp = sp_solver(M)
    for _ in range(10):
        x = (rng.random(len(rows)) < 0.3).astype(np.uint8)
        t = x.astype(int) @ A[rows] % 2
        assert np.array_equal(solve_sp(np.flatnonzero(t)), np.flatnonzero(x))
        assert np.array_equal(solve(A[rows], t), x)
    outside = np.zeros(80, np.uint8)
    outside[np.flatnonzero(~A[rows].any(axis=0))[:1]] = 1
    if outside.any():
        assert solve_sp(np.flatnonzero(outside)) is None


def test_sp_rows_round_trip():
    rows = [np.array([0, 3]), np.array([], dtype=np.int64), np.array([5])]
    M = sp_rows(rows, 6)
    assert M.shape == (3, 6)
    assert [sp_row(M, r).tolist() for r in range(3)] == [[0, 3], [], [5]]
    assert sp.issparse(M)


def test_mask_indices():
    assert mask_indices(0).tolist() == []
    assert mask_indices((1 << 0) | (1 << 9) | (1 << 300)).tolist() == [0, 9, 300]
