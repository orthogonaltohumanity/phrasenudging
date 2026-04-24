"""Step 6 — seed-variance sweep on a single problem.

Runs baseline and controller N times each on the same problem with
different seeds, to measure how stable each approach is under sampling
noise. The model is loaded ONCE and reused; typical per-seed time is
20-60 s depending on T_gen.

Both decoders are stochastic:
  - Baseline: temperature-sampled (default temp = 1.0) with a seeded RNG.
              At temp = 0 the sampler degenerates to argmax and all seeds
              give the identical result (one-seed-is-enough mode).
  - Controller: nudged sampling as in step 5. The seed affects (a) which
              phrase is picked from each cluster per nudge block and
              (b) which token wins inverse-CDF after the SLERP nudge.

A run with N_seeds ≥ 10 gives you two matched-size distributions of
(pred, ok, T). The useful comparison is controller-correct-rate vs
baseline-correct-rate at the same temperature — that isolates the
scaffold's contribution from raw sampling-noise.

Usage:
    python3 examples/step6_seed_sweep.py --pid L4_0016 \\
        --schedule "c4:50+35,c1:50+35,c11:50+35,c7:50+35,c5:50+35" \\
        --n-seeds 10 --baseline-temp 1.0 --force-commit

Output:
    data/seed_sweep_<pid>.jsonl      one record per seed
    (plus a summary printed to stdout)
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# Make sibling `src/` importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_controller import (
    PROMPT_HEAD, PROMPT_TAIL, INSTRUCTION,
    gen_baseline, gen_controlled,
    load_phrase_amps, is_correct, extract_boxed,
    parse_schedule, build_plan,
    parse_force_at, truncate_plan, force_box_emission,
)


def tokenize_prompt(llm, problem_text: str):
    """Build the prompt_tokens list the decoders expect."""
    head = llm.tokenize(PROMPT_HEAD.encode(), add_bos=True, special=True)
    body = llm.tokenize(f"{problem_text}\n\n{INSTRUCTION}".encode(),
                         add_bos=False, special=False)
    tail = llm.tokenize(PROMPT_TAIL.encode(), add_bos=False, special=True)
    return head + body + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pid", required=True,
                    help="Problem id to sweep (must exist in --problems)")
    ap.add_argument("--schedule", required=True,
                    help='Schedule string with optional per-block @alpha, e.g. '
                         '"c4:50+35,c1:50+35@0.02,c11:50+35,c7:50+35"')

    ap.add_argument("--problems", default="data/problems.jsonl",
                    help="Falls back to data/problems_sample.jsonl if the "
                         "default doesn't contain the pid.")
    ap.add_argument("--phrase-amps",
                    default="data/cd_demo_phrase_amps_meta.pkl")
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")

    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-base", type=int, default=2024,
                    help="First seed; subsequent seeds are seed_base + i")
    ap.add_argument("--baseline-temp", type=float, default=1.0,
                    help="Sampling temperature for the (now stochastic) "
                         "baseline. Default 1.0 to match typical chat "
                         "sampling. Set to 0 for a single deterministic "
                         "greedy baseline.")

    # Controller params (same meaning as step 5).
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Default SLERP α for schedule blocks without @α")
    ap.add_argument("--free-temp", type=float, default=0.0,
                    help="Controller free-window temperature (0 = greedy)")
    ap.add_argument("--nudge-temp", type=float, default=1.0,
                    help="Controller nudge-step temperature pre-SLERP")
    ap.add_argument("--force-commit", action="store_true",
                    help="Force \\boxed{} emission if absent at --force-at cutoff")
    ap.add_argument("--force-at", default="end",
                    help="Cutoff position: 'end', 'post-schedule', <int>, or <pct>%%")
    ap.add_argument("--force-prefix", default="\n\nFinal answer: \\boxed{",
                    help="Forcing prefix injected when no box present")
    ap.add_argument("--force-budget", type=int, default=50,
                    help="Max tokens inside forced \\boxed{ (default 50)")
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--n-ctx", type=int, default=6144)

    ap.add_argument("--out", default=None,
                    help="Output JSONL path (default: data/seed_sweep_<pid>.jsonl)")
    args = ap.parse_args()

    # Locate the problem (search --problems first, then sample fallback).
    problem = None
    for candidate_path in [args.problems, "data/problems_sample.jsonl"]:
        if not Path(candidate_path).exists(): continue
        for line in open(candidate_path):
            r = json.loads(line)
            if r.get("problem_id") == args.pid:
                problem = r
                print(f"found {args.pid} in {candidate_path}")
                break
        if problem is not None: break
    if problem is None:
        sys.exit(f"problem_id {args.pid} not found in --problems or sample")

    out_path = Path(args.out) if args.out \
               else Path("data") / f"seed_sweep_{args.pid}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load the model and phrase amps ONCE.
    print(f"loading model {args.model} ...", flush=True)
    t0 = time.time()
    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=-1,
                logits_all=False, verbose=False)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    phrase_seqs_by_cid, W_seq, V_seq = load_phrase_amps(args.phrase_amps)

    # Parse schedule grammar (shared with step5 and step7). No commit stage.
    schedule_blocks = parse_schedule(args.schedule)
    schedule_total  = sum(n + f for _, n, f, _ in schedule_blocks)
    needed = {cid for cid, _, _, _ in schedule_blocks}
    missing = [c for c in sorted(needed) if c not in phrase_seqs_by_cid]
    if missing:
        sys.exit(f"clusters {missing} not in phrase-amps file")

    # Resolve force-at cutoff; truncate plan and cap baseline length if needed.
    force_at_pos = parse_force_at(args.force_at, args.max_new, schedule_total)
    baseline_cap = (force_at_pos if (args.force_commit
                                      and force_at_pos < args.max_new)
                    else args.max_new)
    plan = build_plan(schedule_blocks, None, args.max_new)
    if args.force_commit and force_at_pos < sum(s[2] for s in plan):
        plan = truncate_plan(plan, force_at_pos)
        print(f"[force-at {args.force_at}] plan truncated to {force_at_pos} tk "
              f"(baseline capped at {baseline_cap} tk)")

    # Build prompt and tokenize once — identical across seeds.
    prompt_tokens = tokenize_prompt(llm, problem["problem"])
    gold = problem.get("answer")

    print(f"\n[{args.pid}] gold={gold!r}")
    print(f"  schedule={args.schedule}  α={args.alpha}\n")

    # If baseline_temp <= 0, baseline is deterministic — run it just once
    # and copy the result into every record, saving ~N × T_baseline seconds.
    deterministic_baseline = args.baseline_temp <= 0.0
    cached_baseline = None
    if deterministic_baseline:
        print("running baseline once (greedy, deterministic) ...", flush=True)
        t0 = time.time()
        b = gen_baseline(llm, prompt_tokens, baseline_cap)
        forced_b = False
        if args.force_commit and extract_boxed(b["text"]) is None:
            extra = force_box_emission(llm, args.force_prefix,
                                        args.force_budget, rng=None, temp=0.0)
            b["tokens"] = list(b["tokens"]) + extra
            b["text"] = llm.detokenize(b["tokens"]).decode(
                "utf-8", errors="replace")
            b["pred"] = extract_boxed(b["text"])
            forced_b = True
        dt = time.time() - t0
        b_ok = is_correct(b["pred"], gold)
        cached_baseline = dict(pred=b["pred"], ok=b_ok, T=len(b["tokens"]),
                                text=b["text"], forced=forced_b)
        print(f"  baseline T={len(b['tokens']):>4}  pred={b['pred']!r}  "
              f"{'OK' if b_ok else '--'}{' [FORCED]' if forced_b else ''}  "
              f"({dt:.1f}s)", flush=True)

    # Per-seed loop: run baseline (if stochastic) AND controller at each seed.
    print(f"\nrunning {args.n_seeds} seeds (baseline temp={args.baseline_temp}) ...",
          flush=True)
    records = []
    for i in range(args.n_seeds):
        seed_i = args.seed_base + i

        if deterministic_baseline:
            b_rec = cached_baseline
        else:
            t0 = time.time()
            b_rng = random.Random(seed_i)
            b = gen_baseline(llm, prompt_tokens, baseline_cap,
                              temp=args.baseline_temp, rng=b_rng)
            forced_b = False
            if args.force_commit and extract_boxed(b["text"]) is None:
                extra = force_box_emission(llm, args.force_prefix,
                                            args.force_budget,
                                            rng=None, temp=0.0)
                b["tokens"] = list(b["tokens"]) + extra
                b["text"] = llm.detokenize(b["tokens"]).decode(
                    "utf-8", errors="replace")
                b["pred"] = extract_boxed(b["text"])
                forced_b = True
            dt = time.time() - t0
            b_ok = is_correct(b["pred"], gold)
            b_rec = dict(pred=b["pred"], ok=b_ok, T=len(b["tokens"]),
                          text=b["text"], forced=forced_b)
            print(f"  seed={seed_i} base T={len(b['tokens']):>4}  "
                  f"pred={b['pred']!r}  "
                  f"{'OK' if b_ok else '--'}{' [FORCED]' if forced_b else ''}  "
                  f"({dt:.1f}s)", flush=True)

        c_rng = random.Random(seed_i)
        t0 = time.time()
        c = gen_controlled(llm, prompt_tokens, plan, phrase_seqs_by_cid,
                            c_rng, W_seq,
                            args.alpha, None, None,
                            free_temp=args.free_temp,
                            nudge_temp=args.nudge_temp)
        forced_c = False
        if args.force_commit and extract_boxed(c["text"]) is None:
            extra = force_box_emission(llm, args.force_prefix,
                                        args.force_budget, rng=None, temp=0.0)
            c["tokens"] = list(c["tokens"]) + extra
            c["text"] = llm.detokenize(c["tokens"]).decode(
                "utf-8", errors="replace")
            c["pred"] = extract_boxed(c["text"])
            forced_c = True
        dt = time.time() - t0
        c_ok = is_correct(c["pred"], gold)
        records.append(dict(
            problem_id=args.pid, gold=gold, seed=seed_i,
            baseline=b_rec,
            controller=dict(pred=c["pred"], ok=c_ok, T=len(c["tokens"]),
                             cluster_usage=c["cluster_usage"], text=c["text"],
                             forced=forced_c),
        ))
        print(f"  seed={seed_i} ctrl T={len(c['tokens']):>4}  "
              f"pred={c['pred']!r}  "
              f"{'OK' if c_ok else '--'}{' [FORCED]' if forced_c else ''}  "
              f"({dt:.1f}s)", flush=True)

    # Write all records — baseline fields are duplicated in each line for
    # uniformity (the JSONL is seed-indexed).
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nwrote {out_path}  ({len(records)} records)")

    # 3) Summary.
    from collections import Counter
    print("\n=== SUMMARY ===")
    print(f"problem: {args.pid}  gold={gold!r}")
    print(f"N_seeds={len(records)}   baseline_temp={args.baseline_temp}")

    # Baseline stats
    base_preds = [r["baseline"]["pred"] for r in records]
    base_Ts    = [r["baseline"]["T"]    for r in records]
    base_correct = sum(1 for r in records if r["baseline"]["ok"])
    print(f"\nbaseline: {base_correct}/{len(records)} seeds correct")
    print(f"  T distribution (min/med/mean/max): "
          f"{min(base_Ts)} / {sorted(base_Ts)[len(base_Ts)//2]} / "
          f"{sum(base_Ts)/len(base_Ts):.0f} / {max(base_Ts)}")
    b_hist = Counter(str(p) for p in base_preds)
    print(f"  pred histogram (top 5 across {len(records)} seeds):")
    for pred, n in b_hist.most_common(5):
        pmark = "  ←correct" if is_correct(pred, gold) else ""
        print(f"    {n:>3}× {pred!r}{pmark}")

    # Controller stats
    ctrl_preds = [r["controller"]["pred"] for r in records]
    ctrl_Ts    = [r["controller"]["T"]    for r in records]
    ctrl_correct = sum(1 for r in records if r["controller"]["ok"])
    print(f"\ncontroller: {ctrl_correct}/{len(records)} seeds correct")
    print(f"  T distribution (min/med/mean/max): "
          f"{min(ctrl_Ts)} / {sorted(ctrl_Ts)[len(ctrl_Ts)//2]} / "
          f"{sum(ctrl_Ts)/len(ctrl_Ts):.0f} / {max(ctrl_Ts)}")
    c_hist = Counter(str(p) for p in ctrl_preds)
    print(f"  pred histogram (top 5 across {len(records)} seeds):")
    for pred, n in c_hist.most_common(5):
        pmark = "  ←correct" if is_correct(pred, gold) else ""
        print(f"    {n:>3}× {pred!r}{pmark}")

    # Head-to-head per seed
    both = sum(1 for r in records if r["baseline"]["ok"] and r["controller"]["ok"])
    ups  = sum(1 for r in records if r["controller"]["ok"] and not r["baseline"]["ok"])
    downs= sum(1 for r in records if r["baseline"]["ok"] and not r["controller"]["ok"])
    neither = sum(1 for r in records if not r["baseline"]["ok"] and not r["controller"]["ok"])
    print(f"\nhead-to-head per seed: UP={ups}  DOWN={downs}  BOTH_OK={both}  BOTH_BAD={neither}")
    delta = ctrl_correct - base_correct
    print(f"controller − baseline:  {delta:+d}/{len(records)}")


if __name__ == "__main__":
    main()
