"""Sparse amp-vector utilities: coverage-based top-K and CSR reconstruction.

We store the per-token output distribution sparsely in CSR-like layout:

    indptr : int32[T+1]      cumulative count of nonzero entries per token
    idx    : int32[total]    vocabulary indices (concatenated, token-major)
    val    : float32[total]  probability mass at each index

For each token t, the (idx, val) pair for position t lives in the slice
[indptr[t] : indptr[t+1]]. This lets us keep a *variable* number of tokens
per position — exactly enough to cover, say, 99.9% of the probability mass
— rather than a fixed top-K that wastes space on narrow distributions and
truncates wide ones.

Amplitude form: we always store prob-mass (not sqrt). Callers convert to
amplitudes (a_v = sqrt(p_v)) as needed; amps live on the unit hypersphere
S^{V-1} (since Σ_v p_v = 1 ⇔ Σ_v a_v² = 1), which is the correct geometry
for Bhattacharyya coefficient and SLERP.
"""
from __future__ import annotations

import numpy as np


def coverage_topk(P: np.ndarray, coverage: float = 0.999,
                   probe_m: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Return the indices + prob masses covering at least ``coverage`` of
    the total mass, sorted in descending probability order.

    Fast path: use ``argpartition`` to find an over-estimate of the top-M
    candidates (M ≫ expected k), then full-sort only those. This avoids an
    O(V log V) full-vocab argsort on every token, which is the dominant
    cost of per-step lens capture when V ≳ 100k. For V=152k and probe_m=2048
    this is ~50× faster than a full argsort.

    If the chosen probe_m proves insufficient (the M-th partition doesn't
    yet cover the threshold), we fall back to the full argsort.

    Args:
        P:         shape (V,) probability distribution (must sum to ≈ 1).
        coverage:  minimum cumulative mass to retain (default 0.999).
        probe_m:   over-estimate of how many candidates we'll need; 2048
                   is plenty for essentially all distributions at 99.9%
                   coverage that we've seen on DeepSeek-R1-Distill-Qwen.

    Returns:
        idx: int32 array of selected vocab indices, descending by prob.
        val: float32 array of the corresponding masses.
    """
    if not (0.0 < coverage <= 1.0):
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    V = P.shape[0]
    M = min(V, probe_m)

    # Partial partition: after this call the first M positions of idx_part
    # hold the M largest-prob vocab indices (unordered among themselves).
    idx_part = np.argpartition(-P, M - 1)[:M]
    # Sort just those M.
    order_local = np.argsort(-P[idx_part])
    sorted_idx = idx_part[order_local]
    sorted_p = P[sorted_idx]
    cum = np.cumsum(sorted_p)
    total_in_M = float(cum[-1]) if cum.size else 0.0
    # We need Σ cum ≥ coverage · total_mass. Use Σ P as total_mass; if the
    # top-M already covers ≥ coverage of that, we're done. Otherwise expand.
    total_mass = float(P.sum())
    target = coverage * total_mass
    if total_in_M >= target:
        k = int(np.searchsorted(cum, target, side="left")) + 1
        k = max(1, min(k, M))
        return sorted_idx[:k].astype(np.int32), sorted_p[:k].astype(np.float32)

    # Fallback: argpartition over-estimate wasn't enough. Full sort.
    order = np.argsort(-P)
    sorted_p = P[order]
    cum = np.cumsum(sorted_p)
    k = int(np.searchsorted(cum, target, side="left")) + 1
    k = max(1, min(k, V))
    return order[:k].astype(np.int32), sorted_p[:k].astype(np.float32)


def csr_append(indptr_list: list[int], idx_list: list[np.ndarray],
                val_list: list[np.ndarray], idx: np.ndarray, val: np.ndarray) -> None:
    """Append one step's (idx, val) to running CSR-style buffers."""
    indptr_list.append(indptr_list[-1] + int(idx.shape[0]))
    idx_list.append(idx.astype(np.int32, copy=False))
    val_list.append(val.astype(np.float32, copy=False))


def csr_finalize(indptr_list: list[int], idx_list: list[np.ndarray],
                 val_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate running CSR-style buffers into packed arrays."""
    indptr = np.asarray(indptr_list, dtype=np.int32)
    idx = np.concatenate(idx_list) if idx_list else np.zeros((0,), dtype=np.int32)
    val = np.concatenate(val_list) if val_list else np.zeros((0,), dtype=np.float32)
    return indptr, idx, val


def reconstruct_amp(indptr: np.ndarray, idx: np.ndarray, val: np.ndarray,
                     pos: int, V: int, buf: np.ndarray | None = None) -> np.ndarray:
    """Reconstruct the dense (V,) unit-amplitude vector at position ``pos``.

    The stored ``val`` is probability mass; amplitude is ``sqrt(p)``. We also
    re-normalize so the reconstructed amp has unit L2 norm (the missing-tail
    mass is at most (1 - coverage), so this is ≈ 1.0).
    """
    if buf is None:
        buf = np.zeros(V, dtype=np.float32)
    else:
        buf.fill(0.0)
    s, e = int(indptr[pos]), int(indptr[pos + 1])
    if e > s:
        vs = val[s:e]
        amp_slice = np.sqrt(np.maximum(vs, 0.0))
        n = float(np.sqrt((amp_slice * amp_slice).sum()))
        if n > 0:
            amp_slice = amp_slice / n
        buf[idx[s:e]] = amp_slice.astype(np.float32, copy=False)
    return buf


def csr_slice_amp(indptr: np.ndarray, idx: np.ndarray, val: np.ndarray,
                  t0: int, t1: int, V: int):
    """Build a scipy sparse (T, V) unit-amplitude matrix for a token range.

    Used by the all-pairs BC computation: each row is sqrt(p_t) at token t,
    with rows normalized so that <row, row> = 1.
    """
    from scipy.sparse import csr_matrix
    s = int(indptr[t0])
    e = int(indptr[t1])
    offsets = indptr[t0:t1 + 1] - s                       # shape (t1-t0+1,)
    vals_raw = val[s:e]
    cols = idx[s:e].astype(np.int64, copy=False)
    data = np.sqrt(np.maximum(vals_raw, 0.0)).astype(np.float64, copy=False)
    # Renormalize each row to unit amp-norm
    for r in range(t1 - t0):
        rs, re_ = offsets[r], offsets[r + 1]
        if re_ > rs:
            row = data[rs:re_]
            n = float(np.sqrt((row * row).sum()))
            if n > 0:
                data[rs:re_] = row / n
    return csr_matrix((data, cols, offsets), shape=(t1 - t0, V))
