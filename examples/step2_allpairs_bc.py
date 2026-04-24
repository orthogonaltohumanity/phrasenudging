"""Step 2 — pairwise Bhattacharyya-coefficient matrix over phrase windows.

Runs src/allpairs_bc.py on all lens files in data/lens_demo/ and writes

    data/pairs_demo/pair_i_j.pkl      (one per pair)
    data/cd_demo_summary.pkl          (trajectory index + BC metadata)

Usage (from repo root, inside your venv):
    python3 examples/step2_allpairs_bc.py
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lens-dir", default="data/lens_demo")
    ap.add_argument("--out-summary", default="data/cd_demo_summary.pkl")
    ap.add_argument("--out-pair-dir", default="data/pairs_demo")
    ap.add_argument("--W", type=int, default=50)
    ap.add_argument("--skip-prompt", type=int, default=48)
    ap.add_argument("--k-lo", type=int, default=-60)
    ap.add_argument("--k-hi", type=int, default=61)
    args = ap.parse_args()

    lens_files = sorted(Path(args.lens_dir).glob("*.pkl"))
    if not lens_files:
        sys.exit(f"no lens files in {args.lens_dir} — run step1 first")
    print(f"[{len(lens_files)} lens files]")

    script = Path(__file__).resolve().parent.parent / "src" / "allpairs_bc.py"
    cmd = [sys.executable, str(script),
           "--lens", *[str(p) for p in lens_files],
           "--W", str(args.W),
           "--skip-prompt", str(args.skip_prompt),
           "--k-range", str(args.k_lo), str(args.k_hi),
           "--out-summary", args.out_summary,
           "--out-per-pair-dir", args.out_pair_dir]
    # Run the subprocess WITHOUT capturing stdout/stderr so the user sees
    # allpairs_bc.py's progress and any traceback directly. check_call will
    # still raise on non-zero exit, but the traceback will already be on
    # screen.
    rc = subprocess.call(cmd)
    if rc != 0:
        # Common non-zero exits:
        #   -9 / 137: OOM kill (kernel killed the subprocess)
        #    1      : Python exception
        oom_hint = ""
        if rc in (-9, 137):
            oom_hint = ("\n\nThis looks like an OOM kill — the subprocess "
                        "was terminated by the kernel. Causes: too many lens "
                        "files loaded at once, or very large lens files (check "
                        f"{args.lens_dir}/ for >100 MB files). Reduce the "
                        "number of trajectories or use a lower --coverage "
                        "in step 1.")
        sys.exit(f"\nstep 2 subprocess failed with return code {rc}{oom_hint}")
    print(f"\nsummary:   {args.out_summary}")
    print(f"pair dir:  {args.out_pair_dir}")


if __name__ == "__main__":
    main()
