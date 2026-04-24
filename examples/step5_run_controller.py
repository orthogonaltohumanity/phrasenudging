"""Step 5 — run the SLERP-nudged controller on a problem set.

You must pick a schedule (an ordered list of cluster ids) and a commit
cluster id based on what step3 printed.

Problem-set presets:
    --preset demo    →  data/problems_sample.jsonl + data/problems_demo_pids.txt (6 problems, ~15 min)
    --preset full    →  data/problems.jsonl + data/test_500_hard_first.txt (500 MATH problems, ~3-4 hrs)
    --preset custom  →  use --problems and --problem-ids-file directly

Use --n N to cap the run at the first N problems of whichever PID list is
selected (e.g. --preset full --n 50 runs the 50 hardest L5 problems).

Schedule grammar: --schedule "c<id>:<nudge>+<free>[@<alpha>],c<id>:<nudge>+<free>[@<alpha>],..."

Each block: <nudge> SLERP-nudged tokens toward that cluster, then <free> free
tokens. '+<free>' may be omitted (0 default). '@<alpha>' may be omitted (falls
back to --alpha). To terminate with a boxed answer, use --force-commit.

Usage:
    python3 examples/step5_run_controller.py \\
        --schedule "c4:50+35,c1:50+35,c11:50+35,c7:50+35"   # demo, 6 problems
    python3 examples/step5_run_controller.py --preset full \\
        --schedule "c4:50+35,c1:50+35,c11:50+35,c7:50+35,c5:50+35" \\
        --force-commit                                       # all 500, forced box
    python3 examples/step5_run_controller.py --preset full --n 100 \\
        --schedule "c4:50+35@0.005,c1:50+35@0.01,c11:50+35@0.02" \\
        --force-commit --force-at post-schedule
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


PRESETS = {
    "demo":   {"problems": "data/problems_sample.jsonl",
               "pids":     "data/problems_demo_pids.txt",
               "out":      "data/sweep_demo.jsonl"},
    "full":   {"problems": "data/problems.jsonl",
               "pids":     "data/test_500_hard_first.txt",
               "out":      "data/sweep_full500.jsonl"},
    "custom": {"problems": None, "pids": None, "out": "data/sweep_custom.jsonl"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default="demo",
                    help="Problem-set preset (default: demo, 6 problems)")
    ap.add_argument("--problems", default=None,
                    help="Override the preset's --problems path (JSONL of "
                         "{problem_id, problem, answer})")
    ap.add_argument("--problem-ids-file", default=None,
                    help="Override the preset's --problem-ids-file (one pid/line)")
    ap.add_argument("--n", type=int, default=None,
                    help="Cap the run at the first N problem ids from the "
                         "selected PID list (default: run them all)")
    ap.add_argument("--phrase-amps",
                    default="data/cd_demo_phrase_amps_meta.pkl")
    ap.add_argument("--schedule", required=True,
                    help='Schedule string with optional per-block @alpha, e.g. '
                         '"c4:50+35,c1:50+35@0.02,c11:50+35,c7:50+35@0.05"')
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Default SLERP α for schedule blocks without @α (default 0.01)")
    ap.add_argument("--baseline-temp", type=float, default=0.0,
                    help="Baseline temperature (0 = argmax greedy; default)")
    ap.add_argument("--free-temp", type=float, default=0.0,
                    help="Controller free-window temperature (0 = greedy)")
    ap.add_argument("--nudge-temp", type=float, default=1.0,
                    help="Controller nudge-step temperature pre-SLERP (1.0 = standard)")
    ap.add_argument("--force-commit", action="store_true",
                    help="Force \\boxed{} emission if absent at --force-at cutoff")
    ap.add_argument("--force-at", default="end",
                    help="Cutoff position: 'end', 'post-schedule', <int>, or <pct>%%")
    ap.add_argument("--force-prefix", default="\n\nFinal answer: \\boxed{",
                    help="Forcing prefix injected when no box present")
    ap.add_argument("--force-budget", type=int, default=50,
                    help="Max tokens inside forced \\boxed{ (default 50)")
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: preset-appropriate)")
    ap.add_argument("--mode", choices=["both", "baseline", "controller"],
                    default="both")
    args = ap.parse_args()

    # Resolve preset defaults (explicit flags override).
    preset = PRESETS[args.preset]
    problems_path = args.problems or preset["problems"]
    pids_path     = args.problem_ids_file or preset["pids"]
    out_path      = args.out or preset["out"]
    if problems_path is None or pids_path is None:
        sys.exit("preset=custom requires --problems and --problem-ids-file")
    if not Path(problems_path).exists():
        sys.exit(f"problems file not found: {problems_path}")
    if not Path(pids_path).exists():
        sys.exit(f"problem-ids file not found: {pids_path}")

    # Apply --n cap (write a temp pid file with the prefix, so we don't
    # mutate the user's input file).
    tmp_pid_file = None
    if args.n is not None:
        with open(pids_path) as f:
            all_pids = [l.strip() for l in f if l.strip()]
        if args.n > len(all_pids):
            print(f"WARNING: --n {args.n} exceeds PID list length "
                  f"{len(all_pids)}; using all {len(all_pids)}.")
            args.n = len(all_pids)
        capped = all_pids[:args.n]
        tmp_pid_file = tempfile.NamedTemporaryFile(
            mode="w", suffix="_capped_pids.txt", delete=False)
        for pid in capped:
            tmp_pid_file.write(pid + "\n")
        tmp_pid_file.close()
        pids_path = tmp_pid_file.name
        print(f"[using first {args.n} pids from {args.problem_ids_file or preset['pids']}]")

    print(f"preset:    {args.preset}")
    print(f"problems:  {problems_path}")
    print(f"pids file: {pids_path}")
    print(f"output:    {out_path}")

    script = Path(__file__).resolve().parent.parent / "src" / "run_controller.py"
    cmd = [sys.executable, str(script),
           "--model", args.model,
           "--problems", problems_path,
           "--problem-ids-file", pids_path,
           "--phrase-amps", args.phrase_amps,
           "--schedule", args.schedule,
           "--alpha", str(args.alpha),
           "--baseline-temp", str(args.baseline_temp),
           "--free-temp", str(args.free_temp),
           "--nudge-temp", str(args.nudge_temp),
           "--force-at", args.force_at,
           "--force-prefix", args.force_prefix,
           "--force-budget", str(args.force_budget),
           "--max-new", str(args.max_new),
           "--mode", args.mode,
           "--out", out_path]
    if args.force_commit:
        cmd.append("--force-commit")
    try:
        subprocess.check_call(cmd)
    finally:
        if tmp_pid_file is not None:
            try: Path(tmp_pid_file.name).unlink()
            except Exception: pass

    # Also emit a human-readable report alongside the JSONL.
    report_script = Path(__file__).resolve().parent / "show_sweep.py"
    if report_script.exists() and Path(out_path).exists():
        report_path = Path(out_path).with_suffix(".report.md")
        print(f"\n[writing human-readable report → {report_path}]")
        try:
            subprocess.call([sys.executable, str(report_script),
                             "--jsonl", out_path,
                             "--out", str(report_path)],
                            stdout=subprocess.DEVNULL)
        except Exception as e:
            print(f"  (report generation failed: {e})")


if __name__ == "__main__":
    main()
