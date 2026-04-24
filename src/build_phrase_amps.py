"""Build per-position phrase amp-seqs (variable-K sparse) from lens pickles
and cluster labels.

For each phrase window ``(traj, center)`` in a chosen cluster, extract the
W consecutive per-token distributions around ``center`` from the lens file
and repack them as a CSR-style amp-sequence:

    phrase_k = {
        "traj":    str,
        "center":  int,           # absolute token index in the lens file
        "indptr":  int32[W+1],    # per-position nnz offsets
        "idx":     int32[nnz],    # concatenated vocab indices
        "val":     float16[nnz],  # amplitudes (sqrt(prob), unit-normed per pos)
    }

Because each lens-file position already stores top-K at variable coverage,
the phrase-amp-seq inherits the same variable K per position. No fixed-K
padding.

The controller consumes this format directly: at inference step t it picks
``pos = (t % nudge_window) * W / nudge_window`` and reconstructs the dense
(V,) target amp from phrase[pos]'s CSR slice.

Streaming write: because we never hold more than one phrase's (W, ~1K) nnz
in memory, total memory is ~1 MB regardless of how many phrases we pack.
"""
import argparse
import io
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np


def build_phrase_seq(lens, abs_center, W):
    """Extract W positions centered on abs_center from a v2 lens.

    Padding convention: if the phrase window extends past either end of the
    lens, the out-of-range positions get empty entries (no-op at nudge time).
    """
    indptr = lens["indptr"]
    idx    = lens["idx"]
    val    = lens["val"]
    T_gen  = indptr.shape[0] - 1

    half = W // 2
    start_abs = abs_center - half
    end_abs   = abs_center + (W - half)

    # Translate absolute-token coords into generated-row coords.
    n_prompt = int(lens.get("n_prompt", 0))
    start_gen = max(0, start_abs - n_prompt)
    end_gen   = min(T_gen, end_abs - n_prompt)

    out_indptr = [0]
    out_idx_chunks: list[np.ndarray] = []
    out_val_chunks: list[np.ndarray] = []

    # For each of the W phrase positions, figure out which generated row
    # (if any) backs it and copy the CSR slice. Positions outside the lens
    # become empty entries (indptr[pos+1] == indptr[pos]).
    for pos_in_seq in range(W):
        t_gen = start_gen + (pos_in_seq - (start_abs - (start_abs)))  # = pos_in_seq + (start_gen - (start_abs - n_prompt))? simpler:
        # pos_in_seq 0 corresponds to start_abs; so t_gen = start_abs - n_prompt + pos_in_seq
        t_gen = start_abs - n_prompt + pos_in_seq
        if t_gen < 0 or t_gen >= T_gen:
            out_indptr.append(out_indptr[-1])
            continue
        s, e = int(indptr[t_gen]), int(indptr[t_gen + 1])
        if e == s:
            out_indptr.append(out_indptr[-1])
            continue
        p = val[s:e].astype(np.float64)
        a = np.sqrt(np.maximum(p, 0.0))
        n = float(np.sqrt((a * a).sum()))
        if n > 0:
            a = a / n
        out_idx_chunks.append(idx[s:e].astype(np.int32, copy=False))
        out_val_chunks.append(a.astype(np.float16, copy=False))
        out_indptr.append(out_indptr[-1] + (e - s))

    return (np.asarray(out_indptr, dtype=np.int32),
            np.concatenate(out_idx_chunks) if out_idx_chunks else np.zeros((0,), dtype=np.int32),
            np.concatenate(out_val_chunks) if out_val_chunks else np.zeros((0,), dtype=np.float16))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", required=True,
                    help="labels.pkl from spectral_cluster.py")
    ap.add_argument("--lens", nargs="+", required=True,
                    help="lens pickles in the same order used for clustering")
    ap.add_argument("--clusters", nargs="+", type=int, required=True,
                    help="cluster ids to pack (include every cluster your "
                         "schedule + commit ever references)")
    ap.add_argument("--out", required=True,
                    help="Base path; writes <out>_meta.pkl and <out>_amps.npz")
    args = ap.parse_args()

    lab = pickle.load(open(args.labels, "rb"))
    phrase_list = lab["phrase_list"]
    labels      = lab["labels"]
    names       = lab["names"]
    W           = lab["W"]
    skip        = lab["skip_prompt"]

    lenses = [pickle.load(open(p, "rb")) for p in args.lens]
    if lenses[0].get("schema_version") != "v2":
        raise SystemExit("lens files must be schema v2 — regenerate via generate_lens.py")
    V = lenses[0]["n_vocab"]

    out_stem  = str(Path(args.out).with_suffix(""))
    meta_path = out_stem + "_meta.pkl"
    npz_path  = out_stem + "_amps.npz"
    Path(meta_path).parent.mkdir(parents=True, exist_ok=True)

    # Stream each phrase directly into an npz (which is a zipfile of .npy).
    zf = zipfile.ZipFile(npz_path, "w", zipfile.ZIP_STORED, allowZip64=True)
    cluster_metas = {cid: [] for cid in args.clusters}

    for cid in args.clusters:
        member_pis = [pi for pi in range(len(phrase_list)) if labels[pi] == cid]
        for pi in member_pis:
            tr_idx, center = phrase_list[pi]
            # phrase_list centers are in post-skip-prompt coords; reconstruct abs.
            abs_center = center + skip
            out_indptr, out_idx, out_val = build_phrase_seq(
                lenses[tr_idx], abs_center, W)
            idx_in_cluster = len(cluster_metas[cid])
            arr_key = f"c{cid}_{idx_in_cluster}"
            for name, arr in [("indptr", out_indptr),
                              ("idx", out_idx),
                              ("val", out_val)]:
                bio = io.BytesIO()
                np.save(bio, arr, allow_pickle=False)
                zf.writestr(f"{arr_key}_{name}.npy", bio.getvalue())
            cluster_metas[cid].append(dict(
                traj=names[tr_idx], center=int(abs_center), arr_key=arr_key))
        print(f"  c{cid}: packed {len(cluster_metas[cid])} phrase amp-seqs", flush=True)
    zf.close()

    meta = dict(
        schema_version="v2",
        schedule=list(args.clusters),
        W=W, n_vocab=V,
        clusters={cid: dict(phrases=cluster_metas[cid])
                  for cid in args.clusters},
    )
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=4)
    size_meta = Path(meta_path).stat().st_size / 1e3
    size_npz  = Path(npz_path).stat().st_size / 1e6
    print(f"wrote meta ({size_meta:.1f} KB) + amps ({size_npz:.1f} MB)")


if __name__ == "__main__":
    main()
