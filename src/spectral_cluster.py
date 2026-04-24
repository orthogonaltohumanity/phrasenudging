"""Spectral clustering over phrase-level angular distances, with cluster labeling.

Consumes the per-pair distance files produced by ``allpairs_bc.py`` (one
``pair_i_j.pkl`` per pair, each with ``d_total`` of shape (|k_range|, T_used),
holding θ = arccos(BC_W) in radians) and the ``summary.pkl``, and:

1. Enumerates phrase windows ``(traj_idx, center)`` at stride S across each
   trajectory's usable range.
2. Assembles the N×N phrase angular-distance matrix THETA from the per-pair
   d-files, respecting the offset k = center_j − center_i.
3. Converts θ → Gaussian affinity ``A = exp(-θ²/(2σ²))``.
4. Runs ``sklearn.cluster.SpectralClustering(n_clusters=k, affinity="precomputed")``.
5. Optionally decodes and prints sample phrases per cluster so the user can
   assign human labels ("RECALL-DOMINANT", "WAIT-REFLECT", ...).

Math notes
----------
Spectral clustering on a precomputed affinity matrix ``A`` is equivalent to
k-means on the eigenvectors of the normalized Laplacian
    L = I − D^{-1/2} A D^{-1/2},  where D = diag(A·1).
The Gaussian affinity kernel width σ is in the SAME units as θ (radians
on the amp sphere, [0, π/2]) and is a manifold-scale knob:
    σ ≪ median(θ)  ⇒ nearly disconnected graph, mega-cluster collapse
    σ ≫ median(θ)  ⇒ over-smoothed, uniform-size noise clusters
    σ ≈ median(θ)  ⇒ clusters reflect the underlying manifold structure
Default σ = 0.5 rad is a starting point for phrase-amp angular distances
on DeepSeek-R1-Distill-Qwen-1.5B; sweep as needed.

Note: prior pipeline versions stored d = sin θ = sqrt(1 − BC_W²) instead
of θ = arccos(BC_W); that saturated near θ = π/2. If you have old pair
files, regenerate with the current allpairs_bc.py before clustering.
"""
import argparse
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import SpectralClustering


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--summary", required=True,
                    help="summary.pkl from allpairs_bc.py")
    ap.add_argument("--pair-dir", required=True,
                    help="Directory holding pair_i_j.pkl files")
    ap.add_argument("--lens", nargs="+", required=True,
                    help="Lens pickles, SAME ORDER as allpairs_bc.py --lens")
    ap.add_argument("--W", type=int, default=50,
                    help="Phrase window width (must match --summary)")
    ap.add_argument("--stride", type=int, default=25,
                    help="Step between phrase centers within a trajectory")
    ap.add_argument("--sigma", type=float, default=0.5,
                    help="Gaussian affinity width in radians (default 0.5, "
                         "matches units of θ stored in pair files)")
    ap.add_argument("--k", type=int, default=20,
                    help="Number of spectral clusters (default 20)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-samples", type=int, default=3,
                    help="Print this many sample phrases per cluster")
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
                    help="Tokenizer GGUF path for decoding sample phrases "
                         "(vocab-only load). Override with LENS_MODEL env "
                         "or leave blank (\"\") to skip decoding.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    summ = pickle.load(open(args.summary, "rb"))
    W = summ["W"]
    skip = summ["skip_prompt"]
    names = summ["names"]
    assert W == args.W, f"W mismatch: summary={W} cli={args.W}"
    dk = summ.get("distance_kind", "sin_theta")
    if dk != "theta_rad":
        raise RuntimeError(
            f"summary.distance_kind={dk!r}, expected 'theta_rad'. Pair "
            "files were produced by an older allpairs_bc.py that stored "
            "d = sin θ; regenerate step 2 with the current allpairs_bc.py."
        )

    # Load all pair-distance matrices.
    pair_d = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p = Path(args.pair_dir) / f"pair_{i}_{j}.pkl"
            pair_d[(i, j)] = pickle.load(open(p, "rb"))

    # Resolve each trajectory's post-skip usable length. Prefer the list
    # stored in summary["trajectory_lengths"] (newer pipeline); fall back
    # to reading the lens files directly for backward-compatible summaries
    # produced before that field was added.
    if "trajectory_lengths" in summ and len(summ["trajectory_lengths"]) == len(names):
        lengths_post_skip: list[int] = list(summ["trajectory_lengths"])
    else:
        print("  (summary lacks trajectory_lengths; reading lengths from lens files)",
              flush=True)
        lengths_post_skip = []
        for p in args.lens:
            try:
                lens = pickle.load(open(p, "rb"))
                T_gen = lens["indptr"].shape[0] - 1
                lengths_post_skip.append(max(0, T_gen - skip))
            except Exception as e:
                print(f"    [warn] could not read {p}: {e}", flush=True)
                lengths_post_skip.append(0)

    # Enumerate phrase windows per trajectory, using each trajectory's OWN
    # post-skip length — NOT the min across pairs. Earlier versions used
    # min(d.shape[1]) as a proxy, which silently clipped every trajectory
    # to the length of the shortest one (the axis was traj-A's length, so
    # pairs where tr was traj-B would report a completely unrelated length).
    # That bug removed terminal `\boxed{}`-containing regions from the pool.
    phrase_list: list[tuple[int, int]] = []
    for tr in range(len(names)):
        Tu = lengths_post_skip[tr]
        if Tu < W:
            continue
        for c in range(W // 2, Tu - W // 2, args.stride):
            phrase_list.append((tr, c))
    N = len(phrase_list)
    by_traj: dict[int, list[tuple[int, int]]] = {}
    for pi, (tr, c) in enumerate(phrase_list):
        by_traj.setdefault(tr, []).append((pi, c))
    print(f"phrases: {N}  trajectories: {len(names)}", flush=True)

    # Identify trajectories with zero phrases (T_used < W) and warn, so the
    # user knows when early-EOS samples are being dropped.
    empty_trajs = [tr for tr in range(len(names)) if tr not in by_traj]
    if empty_trajs:
        names_short = [names[tr] for tr in empty_trajs]
        print(f"  skipping {len(empty_trajs)} trajectories with no phrases "
              f"(T_used < W={W}): {names_short[:5]}"
              + ("..." if len(empty_trajs) > 5 else ""), flush=True)

    # Assemble N×N angular-distance matrix THETA directly from the per-pair
    # d-files (which hold θ = arccos(BC_W) in radians, [0, π/2]).
    THETA = np.zeros((N, N), dtype=np.float32)
    for (i, j), pd in pair_d.items():
        # Skip pairs where either trajectory produced no phrases.
        if i not in by_traj or j not in by_traj:
            continue
        d = pd["d_total"]  # θ in radians
        k_to_i = {int(kk): ki for ki, kk in enumerate(pd["k_range"])}
        for pi, ci in by_traj[i]:
            for pj, cj in by_traj[j]:
                kk = cj - ci
                if kk not in k_to_i:
                    continue
                ki = k_to_i[kk]
                if ci >= d.shape[1]:
                    continue
                v = d[ki, ci]
                if not np.isnan(v):
                    THETA[pi, pj] = v
                    THETA[pj, pi] = v

    A = np.exp(-THETA ** 2 / (2 * args.sigma ** 2)).astype(np.float64)
    np.fill_diagonal(A, 1.0)
    print(f"θ stats: min={THETA.min():.3f} mean={THETA.mean():.3f} "
          f"max={THETA.max():.3f}  (radians)", flush=True)
    print(f"A stats (σ={args.sigma}): min={A.min():.3f} mean={A.mean():.3f} "
          f"max={A.max():.3f}", flush=True)

    sc = SpectralClustering(n_clusters=args.k, affinity="precomputed",
                            random_state=args.seed, assign_labels="kmeans")
    labels = sc.fit_predict(A)
    sizes = Counter(labels.tolist())
    print(f"\nσ={args.sigma} k={args.k}  cluster sizes (sorted desc):")
    for lab in sorted(sizes, key=lambda x: -sizes[x]):
        print(f"  c{lab}: {sizes[lab]}", flush=True)

    # Sample-phrase labeling. Resolve the tokenizer model path in this
    # precedence: LENS_MODEL env > --model CLI. Set either to "" (empty)
    # to skip decoding and only print token counts.
    if args.show_samples > 0:
        lenses = [pickle.load(open(p, "rb")) for p in args.lens]
        import os
        model_path = os.environ.get("LENS_MODEL", args.model)
        llm = None
        if model_path and Path(model_path).exists():
            try:
                from llama_cpp import Llama
                llm = Llama(model_path=model_path, n_ctx=256, n_gpu_layers=0,
                            vocab_only=True, verbose=False)
            except Exception as e:
                print(f"  (could not load tokenizer from {model_path}: {e})",
                      flush=True)
        elif model_path:
            print(f"  (tokenizer not found at {model_path}; sample phrases "
                  "shown as token counts)", flush=True)
        half = W // 2
        for lab in sorted(sizes, key=lambda x: -sizes[x]):
            members = [pi for pi in range(N) if labels[pi] == lab]
            if not members:
                continue
            print(f"\n=== c{lab}  n={len(members)} sample phrases ===")
            for pi in members[:args.show_samples]:
                tr, c = phrase_list[pi]
                abs_c = c + skip
                toks = lenses[tr]["tokens"]
                s = max(0, abs_c - half)
                e = min(len(toks), abs_c + (W - half))
                segment = toks[s:e]
                if llm is not None:
                    try:
                        txt = llm.detokenize([int(t) for t in segment]).decode(errors="replace")
                    except Exception:
                        txt = f"<{len(segment)} tokens>"
                else:
                    txt = f"<{len(segment)} tokens; set LENS_MODEL env to decode>"
                print(f"  [{names[tr]}@{abs_c}] {txt[:160]!r}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(dict(phrase_list=phrase_list, labels=labels, names=names,
                          sigma=args.sigma, k=args.k, W=W, skip_prompt=skip),
                     f, protocol=4)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
