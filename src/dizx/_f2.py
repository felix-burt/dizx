"""Minimal linear algebra over F2 on numpy uint8 arrays, and on sparse rows
(sorted arrays of the columns holding a 1, or scipy CSR matrices)."""
import numpy as np
import scipy.sparse as sp


def rref(A: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form of A over F2 and its pivot columns.

    Rows are bit-packed for the eliminations."""
    A = np.asarray(A, dtype=np.uint8) % 2
    m, n = A.shape
    P = np.packbits(A, axis=1, bitorder="little")
    pivots: list[int] = []
    row = 0
    for col in range(n):
        if row == m:
            break
        byte, bit = col >> 3, np.uint8(1 << (col & 7))
        column = (P[:, byte] & bit).astype(bool)
        hits = np.flatnonzero(column[row:])
        if hits.size == 0:
            continue
        p = row + hits[0]
        if p != row:
            P[[row, p]] = P[[p, row]]
            column[[row, p]] = column[[p, row]]
        column[row] = False
        P[column] ^= P[row]
        pivots.append(col)
        row += 1
    return np.unpackbits(P[:row], axis=1, count=n, bitorder="little"), pivots


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A B over F2, as uint8. Goes through float BLAS (integer matmul in numpy
    does not); the sums stay exact for inner dimensions below 2^24."""
    A, B = np.asarray(A), np.asarray(B)
    dtype = np.float32 if A.shape[-1] < 1 << 24 else np.float64
    return (A.astype(dtype) @ B.astype(dtype) % 2).astype(np.uint8)


def rank(A: np.ndarray) -> int:
    return len(rref(A)[1])


class Echelon:
    """Incremental F2 row echelon form. Each row carries a tag (e.g. the
    combination it came from) that is XORed along with it."""

    def __init__(self) -> None:
        self.rows: list[tuple[int, np.ndarray, np.ndarray | None]] = []

    def reduce(self, v: np.ndarray, tag: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        v = np.array(v, dtype=np.uint8)
        tag = None if tag is None else np.array(tag, dtype=np.uint8)
        for p, r, t in self.rows:
            if v[p]:
                v ^= r
                if tag is not None:
                    tag ^= t
        return v, tag

    def add(self, v: np.ndarray, tag: np.ndarray | None = None) -> bool:
        """Add v unless it is in the span; returns whether it was added."""
        v, tag = self.reduce(v, tag)
        nz = np.flatnonzero(v)
        if nz.size == 0:
            return False
        self.rows.append((int(nz[0]), v, tag))
        return True


def solve(A: np.ndarray, t: np.ndarray) -> np.ndarray | None:
    """Some x with x A = t over F2 (A has one vector per row), or None."""
    A = np.array(A, dtype=np.uint8).reshape(-1, len(t))
    for y in nullspace(np.vstack([A, t]).T):
        if y[-1]:
            return y[:-1]
    return None


def nullspace(A: np.ndarray) -> np.ndarray:
    """Basis of {v : A v = 0} over F2, one basis vector per row."""
    n = A.shape[1]
    R, pivots = rref(A)
    pivot_set = set(pivots)
    free = [c for c in range(n) if c not in pivot_set]
    N = np.zeros((len(free), n), dtype=np.uint8)
    N[np.arange(len(free)), free] = 1
    if pivots and free:
        N[:, pivots] = R[:, free].T
    return N


# ---------------------------------------------------------------------------
# Sparse
# ---------------------------------------------------------------------------

def sp_rows(rows: list[np.ndarray], ncols: int) -> sp.csr_matrix:
    """CSR matrix (uint8) with a 1 at each column listed in ``rows[r]``
    (sorted, distinct)."""
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    if rows:
        np.cumsum([len(r) for r in rows], out=indptr[1:])
    indices = np.concatenate(rows).astype(np.int64) if rows else np.zeros(0, np.int64)
    return sp.csr_matrix((np.ones(len(indices), dtype=np.uint8), indices, indptr),
                         shape=(len(rows), ncols))


def sp_canonical(M) -> sp.csr_matrix:
    """``M`` reduced mod 2 as a CSR uint8 matrix with sorted indices and no
    explicit zeros."""
    M = sp.csr_matrix(M)
    M.sum_duplicates()
    M.data %= 2
    M.eliminate_zeros()
    M.sort_indices()
    return M.astype(np.uint8)


def sp_matmul(A, B) -> sp.csr_matrix:
    """A B over F2 for sparse (or dense) A and B, as canonical CSR."""
    A = sp.csr_matrix(A).astype(np.int32)
    B = sp.csr_matrix(B).astype(np.int32)
    return sp_canonical(A @ B)


def sp_row(M: sp.csr_matrix, r: int) -> np.ndarray:
    """The columns of row r of a canonical CSR matrix."""
    return M.indices[M.indptr[r]:M.indptr[r + 1]].astype(np.int64)


def mask_indices(mask: int) -> np.ndarray:
    """Positions of the set bits of a Python int, ascending."""
    if not mask:
        return np.zeros(0, dtype=np.int64)
    b = np.frombuffer(mask.to_bytes((mask.bit_length() + 7) // 8, "little"), dtype=np.uint8)
    return np.flatnonzero(np.unpackbits(b, bitorder="little")).astype(np.int64)


class SparseEchelon:
    """Incremental F2 echelon form of sparse rows (sorted column arrays),
    pivoting on each row's lowest column. Each row carries a tag (a Python
    int used as a bitmask, e.g. the combination it came from) that is XORed
    along with it.

    Stored rows have distinct lowest columns, so a nonzero combination of
    them has one of those as its lowest column: a vector whose lowest column
    is not a pivot is outside the span."""

    def __init__(self) -> None:
        self.rows: dict[int, tuple[np.ndarray, int]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def reduce(self, v: np.ndarray, tag: int = 0) -> tuple[np.ndarray, int]:
        """Reduce v until its lowest column is not a pivot (or it is zero)."""
        v = np.asarray(v, dtype=np.int64)
        while v.size:
            hit = self.rows.get(int(v[0]))
            if hit is None:
                break
            r, t = hit
            v = np.setxor1d(v, r, assume_unique=True)
            tag ^= t
        return v, tag

    def add(self, v: np.ndarray, tag: int = 0) -> bool:
        """Add v unless it is in the span; returns whether it was added."""
        v, tag = self.reduce(v, tag)
        if not v.size:
            return False
        self.rows[int(v[0])] = (v, tag)
        return True


def sp_rank(M: sp.csr_matrix) -> int:
    """Rank over F2 of a canonical CSR matrix."""
    ech = SparseEchelon()
    for r in range(M.shape[0]):
        ech.add(sp_row(M, r))
    return len(ech)


def sp_solver(M: sp.csr_matrix):
    """``solve(t)``: the x with x M = t (t as sorted columns), as sorted row
    indices, or None. M's rows must be independent (so x is unique)."""
    ech = SparseEchelon()
    for r in range(M.shape[0]):
        if not ech.add(sp_row(M, r), 1 << r):
            raise ValueError("rows are not independent")

    def solve(t: np.ndarray) -> np.ndarray | None:
        rest, tag = ech.reduce(t)
        return None if rest.size else mask_indices(tag)
    return solve
