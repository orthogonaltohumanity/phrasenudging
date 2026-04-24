"""Step 3 — spectral cluster the phrase-BC matrix and print sample phrases.

For the 10-trajectory demo we use k=10 clusters; for the full 100-trajectory
pipeline try k=20 with σ=0.3.

Prints a few sample phrases per cluster so you can hand-label them
("RECALL-DOMINANT", "WAIT-REFLECT", etc.). The tokenizer is loaded from
the DeepSeek-R1 GGUF by default. To skip decoding (just print token counts),
pass --model "" or set LENS_MODEL="".

Usage (from repo root, inside your venv):
    python3 examples/step3_spectral_cluster.py
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lens-dir", default="data/lens_demo")
    ap.add_argument("--summary", default="data/cd_demo_summary.pkl")
    ap.add_argument("--pair-dir", default="data/pairs_demo")
    ap.add_argument("--out", default="data/cd_demo_labels.pkl")
    ap.add_argument("--W", type=int, default=50)
    ap.add_argument("--stride", type=int, default=25)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--show-samples", type=int, default=3)
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
                    help="Tokenizer GGUF path for decoding sample phrases "
                         '(pass "" to disable)')
    args = ap.parse_args()

    lens_files = sorted(Path(args.lens_dir).glob("*.pkl"))
    if not lens_files:
        sys.exit(f"no lens files in {args.lens_dir} — run step1 first")

    script = Path(__file__).resolve().parent.parent / "src" / "spectral_cluster.py"
    cmd = [sys.executable, str(script),
           "--summary", args.summary,
           "--pair-dir", args.pair_dir,
           "--lens", *[str(p) for p in lens_files],
           "--W", str(args.W),
           "--stride", str(args.stride),
           "--sigma", str(args.sigma),
           "--k", str(args.k),
           "--show-samples", str(args.show_samples),
           "--model", args.model,
           "--out", args.out]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
