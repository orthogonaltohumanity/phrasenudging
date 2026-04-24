"""Step 4 — build per-position sparse phrase amp-seqs (CSR, variable-K).

Produces:
    data/cd_demo_phrase_amps_meta.pkl    metadata (small)
    data/cd_demo_phrase_amps_amps.npz    per-phrase CSR amp-seqs (bulk)

For the demo we pack every cluster 0..(k-1) so any schedule can be tried
later. For full-scale runs you may want to pack only the clusters your
schedule actually references to save disk.

Usage (from repo root, inside your venv):
    python3 examples/step4_build_phrase_amps.py
Optional:
    python3 examples/step4_build_phrase_amps.py --clusters 0 1 2 3 4 5 6 7 8 9
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default="data/cd_demo_labels.pkl")
    ap.add_argument("--lens-dir", default="data/lens_demo")
    ap.add_argument("--out", default="data/cd_demo_phrase_amps.pkl",
                    help="Base; writes <out>_meta.pkl + <out>_amps.npz")
    ap.add_argument("--clusters", nargs="+", type=int, default=None,
                    help="Cluster ids to pack (default: all clusters found)")
    args = ap.parse_args()

    # Default: pack every cluster that exists.
    if args.clusters is None:
        lab = pickle.load(open(args.labels, "rb"))
        args.clusters = sorted(set(int(x) for x in lab["labels"]))
        print(f"packing all {len(args.clusters)} clusters: {args.clusters}")

    lens_files = sorted(Path(args.lens_dir).glob("*.pkl"))
    if not lens_files:
        sys.exit(f"no lens files in {args.lens_dir} — run step1 first")

    script = Path(__file__).resolve().parent.parent / "src" / "build_phrase_amps.py"
    cmd = [sys.executable, str(script),
           "--labels", args.labels,
           "--lens", *[str(p) for p in lens_files],
           "--clusters", *[str(c) for c in args.clusters],
           "--out", args.out]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
