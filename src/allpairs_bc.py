"""Pairwise Bhattacharyya-coefficient matrix over phrase windows.

For each unordered pair of trajectories (i < j) we compute the W-window
Bhattacharyya coefficient at every (phrase-center t, offset k) as the
average of PER-TOKEN Bhattacharyya coefficients aligned along the window:

    BC_W(i, j; t, k)  =  (1/W) Σ_{s=0}^{W-1} BC(p_i[t+s], p_j[t+k+s])
                      =  (1/W) Σ_{s=0}^{W-1} ⟨ a_i[t+s], a_j[t+k+s] ⟩

where a_X[s] = sqrt(p_X[s]) is the unit-norm amplitude at position s in
trajectory X. Each per-token inner product ⟨a_X[s], a_Y[s']⟩ IS the
proper Bhattacharyya coefficient between the two distributions at those
positions.

We then store the raw angular distance

    d(t, k) = arccos(BC_W)            (one scalar per (t, k), radians)

in shape (|k_range|, T_used) per pair. This is the geodesic distance on
the unit amplitude hypersphere S^(V-1) and is linear in the angle, so it
does NOT saturate the way sin θ = sqrt(1 - BC²) does near θ = π/2.
Downstream (step 3) feeds θ directly into a Gaussian affinity kernel
exp(-θ²/(2σ²)).

Fast path
---------
One sparse matmul per pair gives all per-token BCs at once; the W-window
average is a length-W rolling mean along each k-diagonal.

1.  G = Φ_A @ Φ_B.T         (dense (T_A, T_B) float32)

    G[a, b] = ⟨a_A[a], a_B[b]⟩ = BC(p_A[a], p_B[b])   (per-token BC)

2.  For each k, extract the k-diagonal:
       diag_k[i] = G[i, i+k]    for valid i.

    Apply a length-W rolling mean via a cumulative-sum trick:
       bc_W(t, k) = mean( diag_k[t : t+W] ).

This is O(T_A · T_B) per pair for the matmul and O(T · |k_range|) for
the diagonal extraction + rolling sums — cheap in practice (for T~600
and |k_range|~200, one pair is well under a second).

Why the per-token average rather than a "BC of window-means"?
------------------------------------------------------------
For two probability distributions p and q, BC is literally
⟨sqrt(p), sqrt(q)⟩. Extending that to a W-token window has (at least)
three distinct interpretations:

  (A) (1/W) Σ_s ⟨a_A[t+s], a_B[t+k+s]⟩            — PER-TOKEN AVERAGE  ← what we use
  (B) cos( mean(a_A-over-window), mean(a_B-over-window) )
  (C) ⟨ sqrt(mean(p_A-over-window)), sqrt(mean(p_B-over-window)) ⟩

(A) is the straightforward "what is the average Bhattacharyya similarity
between aligned tokens across the window". (B) is a cosine of mean-amp
directions, which washes out per-token temporal alignment. (C) is BC of
mean-distributions and requires materializing dense (W, V) tensors per
window. (A) is faithful to the per-position cognitive-state similarity
and cheapest to compute via the matmul-then-rolling-mean trick.
"""
import argparse
import pickle
import time
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from sparse_utils import csr_slice_amp


def load_lens_csr(path):
    """Load a schema-v2 lens pickle (CSR sparse, coverage-based top-K)."""
    lens = pickle.load(open(path, "rb"))
    if lens.get("schema_version") != "v2":
        raise RuntimeError(
            f"{path}: expected schema_version=v2, got "
            f"{lens.get('schema_version')}. Regenerate with generate_lens.py."
        )
    return lens


def pair_dist_fast(Phi_A: csr_matrix, Phi_B: csr_matrix, W: int,
                    k_range: list[int]) -> np.ndarray:
    """Compute d[ki, t] = arccos(BC_W) for all (t, k) using per-token
    BC averaged over a W-token window (option A from the module docstring).

    Steps:
      G = Phi_A @ Phi_B.T          (T_A, T_B) dense float32, per-token BC
      For each k, extract k-diagonal diag_k[i] = G[i, i+k]
      Rolling mean of length W via cumsum trick → BC_W(t, k)
      d(t, k) = arccos(BC_W)       angular distance in radians, [0, π/2]

    Phi_A and Phi_B are per-token amplitude matrices with UNIT-NORM rows
    (produced by csr_slice_amp), so G is directly cosine/BC with no
    renormalization step. arccos is used instead of sqrt(1 - BC²) = sin θ
    to avoid the saturation of sin near θ = π/2 — linear-in-angle metric
    makes the Gaussian-affinity kernel in step 3 discriminate between
    "mostly unrelated" and "fully orthogonal" pairs.
    """
    T_A = Phi_A.shape[0]
    T_B = Phi_B.shape[0]
    # Per-token BC matrix: G[a, b] = <a_A[a], a_B[b]> = BC(p_A[a], p_B[b]).
    G = (Phi_A @ Phi_B.T).astype(np.float32).toarray()
    # Float-noise guard; true BCs are in [0, 1].
    G = np.clip(G, 0.0, 1.0)

    d = np.full((len(k_range), T_A), np.nan, dtype=np.float32)
    for ki, k in enumerate(k_range):
        # Valid start-indices i where both G[i, i+k] and the full W-window
        # fit within both trajectories.
        i_lo = max(0, -k)
        i_hi = min(T_A, T_B - k)
        length = i_hi - i_lo
        if length < W:
            continue

        # Extract the k-diagonal: diag_k[j] = G[i_lo + j, i_lo + j + k].
        rows = np.arange(i_lo, i_hi)
        cols = rows + k
        diag_k = G[rows, cols]

        # Rolling mean of length W via cumulative sum.
        # bc_per_window[j] = mean(diag_k[j : j+W])  corresponds to phrase
        # center t = i_lo + j.
        cs = np.concatenate([[0.0], np.cumsum(diag_k, dtype=np.float64)])
        rolling_sum = cs[W:] - cs[:-W]
        bc_per_window = (rolling_sum / W).astype(np.float32)
        bc_per_window = np.clip(bc_per_window, 0.0, 1.0)

        n = bc_per_window.shape[0]       # = length - W + 1
        out_end = min(i_lo + n, T_A)
        out_n = out_end - i_lo
        # Raw angular distance: θ = arccos(BC_W), radians, in [0, π/2].
        # Linear in the angle, no saturation near π/2.
        d[ki, i_lo:out_end] = np.arccos(bc_per_window[:out_n]).astype(np.float32)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lens", nargs="+", required=True,
                    help="paths to schema-v2 lens pickles (generate_lens.py output)")
    ap.add_argument("--W", type=int, default=50,
                    help="Window width for phrase-mean amps (default 50)")
    ap.add_argument("--skip-prompt", type=int, default=48,
                    help="Skip this many leading generated tokens")
    ap.add_argument("--k-range", type=int, nargs=2, default=[-100, 101],
                    help="Offsets to scan, as [lo, hi)")
    ap.add_argument("--out-summary", required=True)
    ap.add_argument("--out-per-pair-dir", required=True)
    args = ap.parse_args()

    Path(args.out_per_pair_dir).mkdir(parents=True, exist_ok=True)

    names = [Path(p).stem for p in args.lens]
    print(f"{len(args.lens)} lens files", flush=True)

    # Peek one file to get V and layer_idx for the summary.
    _peek = load_lens_csr(args.lens[0])
    V = _peek["n_vocab"]
    layer_idx = list(_peek.get("layer_idx", [27]))
    del _peek
    print(f"  n_vocab={V}  layer(s)={layer_idx}", flush=True)

    # Collect each trajectory's post-skip generated length so step 3 can
    # enumerate phrase windows in its OWN length rather than inferring
    # (incorrectly) from per-pair d.shape[1].
    trajectory_lengths: list[int] = []

    # Load per-trajectory amplitude matrices one at a time, FREEING the raw
    # lens pickle between iterations. Each Phi is (T_gen - skip, V) with
    # unit-normed rows; stored as a scipy CSR sparse matrix.
    print("loading per-token amplitude matrices ...", flush=True)
    t0 = time.time()
    phis: list[csr_matrix] = []
    import gc
    for idx_lens, p in enumerate(args.lens):
        lens = load_lens_csr(p)
        indptr = lens["indptr"]
        idx = lens["idx"]
        val = lens["val"]
        T_gen = indptr.shape[0] - 1
        t_start = min(args.skip_prompt, T_gen)
        # Record per-traj post-skip length for step 3's phrase enumeration.
        trajectory_lengths.append(max(0, T_gen - t_start))
        Phi = csr_slice_amp(indptr, idx, val, t_start, T_gen, V)
        phis.append(Phi)
        if (idx_lens + 1) % 10 == 0 or (idx_lens + 1) == len(args.lens):
            elapsed = time.time() - t0
            print(f"  [{idx_lens+1}/{len(args.lens)}] {Path(p).stem}: "
                  f"T={T_gen - t_start} nnz={Phi.nnz}  "
                  f"({elapsed:.1f}s elapsed)", flush=True)
        del lens, indptr, idx, val
        gc.collect()
    print(f"amp matrices loaded in {time.time() - t0:.1f}s", flush=True)

    k_range = list(range(args.k_range[0], args.k_range[1]))
    summary = dict(names=names, W=args.W, skip_prompt=args.skip_prompt,
                   n_vocab=V, layer_idx=layer_idx,
                   trajectory_lengths=trajectory_lengths,
                   distance_kind="theta_rad")

    # Pairwise matmul.
    N = len(args.lens)
    total_pairs = N * (N - 1) // 2
    # Aim for ~50 progress lines regardless of N; always print first 10.
    stride = max(1, total_pairs // 50)
    t0 = time.time()
    last_print_t = t0
    n_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            d = pair_dist_fast(phis[i], phis[j], args.W, k_range)
            pair_path = Path(args.out_per_pair_dir) / f"pair_{i}_{j}.pkl"
            with open(pair_path, "wb") as f:
                pickle.dump(dict(d_total=d, k_range=np.array(k_range)), f,
                             protocol=4)
            n_pairs += 1
            # Print if (a) first 10 pairs, (b) hit the stride, (c) last pair,
            # or (d) 10+ seconds have elapsed since last print.
            now = time.time()
            if (n_pairs <= 10 or n_pairs % stride == 0
                    or n_pairs == total_pairs or (now - last_print_t) > 10):
                d_finite = d[np.isfinite(d)]
                dmin = float(d_finite.min()) if d_finite.size else float("nan")
                dmean = float(d_finite.mean()) if d_finite.size else float("nan")
                elapsed = now - t0
                rate = n_pairs / elapsed if elapsed > 0 else 0
                eta_s = (total_pairs - n_pairs) / rate if rate > 0 else 0
                print(f"  pair {n_pairs:>4}/{total_pairs} ({i},{j}) "
                      f"{names[i]} × {names[j]}:  "
                      f"d min={dmin:.3f} mean={dmean:.3f}  "
                      f"[{rate:.1f} pairs/s, ETA {eta_s:.0f}s]", flush=True)
                last_print_t = now
    print(f"{n_pairs} pairs computed in {time.time() - t0:.1f}s", flush=True)

    with open(args.out_summary, "wb") as f:
        pickle.dump(summary, f, protocol=4)
    print(f"wrote summary {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
