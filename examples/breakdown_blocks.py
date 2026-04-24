"""Step 6 — annotated per-block decode trace for one problem in the sweep.

This is the tool to see where each nudge fires and what the free-window
continuation looks like. Useful for diagnosing why a particular problem
succeeded or failed.

Usage (from repo root, inside your venv):
    python3 examples/step6_breakdown.py --pid L4_0016 \\
        --schedule 0 3 5 7 3 --commit 9
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", default="data/sweep_demo.jsonl")
    ap.add_argument("--pid", required=True)
    ap.add_argument("--schedule", nargs="+", type=int, required=True)
    ap.add_argument("--commit", type=int, required=True)
    ap.add_argument("--nudge", type=int, default=50)
    ap.add_argument("--free", type=int, default=35)
    ap.add_argument("--commit-free", type=int, default=150)
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
    ap.add_argument("--n-ctx", type=int, default=6144)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    args = ap.parse_args()

    script = Path(__file__).resolve().parent.parent / "src" / "breakdown_one.py"
    cmd = [sys.executable, str(script),
           "--jsonl", args.jsonl,
           "--pid", args.pid,
           "--schedule", *[str(c) for c in args.schedule],
           "--commit", str(args.commit),
           "--nudge", str(args.nudge),
           "--free", str(args.free),
           "--commit-free", str(args.commit_free),
           "--model", args.model,
           "--n-ctx", str(args.n_ctx),
           "--n-gpu-layers", str(args.n_gpu_layers)]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
