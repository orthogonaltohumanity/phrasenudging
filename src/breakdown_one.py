"""Per-block decode trace for one controller record.

For the given problem_id (from a sweep's JSONL output), re-tokenizes the
controller's decoded text and prints the decoded-text segment for each
schedule block + each commit block, annotated with the cluster label.

This is the tool to see *where* each nudge fires and *what effect* the
nudge has on the subsequent free-window continuation.

Usage example:
  python3 src/breakdown_one.py \
      --jsonl data/sweep_demo.jsonl \
      --pid L4_0016 \
      --schedule 16 15 18 0 6 2 0 \
      --commit 10 \
      --nudge 50 --free 35 --commit-free 150
"""
import argparse
import json
import sys


# Human-readable labels users assigned to clusters (edit to match your clusters).
CLUSTER_LABELS = {
    0:  "KNOWLEDGE-CLAIMS",
    2:  "WAIT-REFLECT",
    6:  "RECALL-MID",
    10: "THINK-AND-CONCLUDE",
    15: "KNOW-WITH-HEDGE",
    16: "PROBLEM-ENTRY",
    18: "RECALL-DOMINANT",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", required=True,
                    help="sweep output from run_controller.py")
    ap.add_argument("--pid", required=True, help="problem_id to break down")
    ap.add_argument("--schedule", nargs="+", type=int, required=True,
                    help="Schedule cluster ids, in order")
    ap.add_argument("--commit", type=int, required=True,
                    help="Commit cluster id")
    ap.add_argument("--nudge", type=int, default=50)
    ap.add_argument("--free", type=int, default=35)
    ap.add_argument("--commit-free", type=int, default=150)
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf",
                    help="Full GGUF model path (loaded with GPU offload)")
    ap.add_argument("--n-ctx", type=int, default=6144)
    ap.add_argument("--n-gpu-layers", type=int, default=-1,
                    help="-1 = all layers on GPU (default); 0 = CPU only")
    args = ap.parse_args()

    # Find the record.
    rec = None
    for line in open(args.jsonl):
        r = json.loads(line)
        if r["problem_id"] == args.pid:
            rec = r
            break
    if rec is None:
        sys.exit(f"pid {args.pid} not in {args.jsonl}")

    c = rec["controller"]
    text = c["text"]
    T_recorded = c["T"]
    print(f"=== {args.pid}  gold={rec['gold']!r}  pred={c['pred']!r}  "
          f"ok={c['ok']}  T={T_recorded} ===\n")

    # Load the full GGUF (GPU-offloaded, not vocab-only) so the tokenizer
    # matches what the controller used exactly AND downstream callers have a
    # real model available if they want to inspect logits/layers.
    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers,
                logits_all=False, verbose=False)
    toks = llm.tokenize(text.encode(), add_bos=False, special=True)
    use_T = min(len(toks), T_recorded)
    if len(toks) != T_recorded:
        print(f"(re-tokenize produced {len(toks)} tokens; recorded T={T_recorded}; "
              f"using T={use_T} for alignment)\n")

    def detok(ts):
        try: return llm.detokenize(ts).decode("utf-8", errors="replace")
        except Exception: return "?"

    sched_bs  = args.nudge + args.free
    commit_bs = args.nudge + args.commit_free
    sched_end = len(args.schedule) * sched_bs

    # Schedule-phase blocks.
    for b, cid in enumerate(args.schedule):
        lbl = CLUSTER_LABELS.get(cid, f"c{cid}")
        n_start = b * sched_bs
        n_end   = n_start + args.nudge
        f_end   = n_start + sched_bs
        if n_start >= use_T:
            break
        print(f"─── Block {b}  c{cid} {lbl}  (tokens {n_start}..{min(f_end, use_T) - 1}) ───")
        nudge_text = detok([int(t) for t in toks[n_start:min(n_end, use_T)]])
        free_text  = detok([int(t) for t in toks[n_end:min(f_end, use_T)]])
        print(f"  NUDGE [{n_start}..{min(n_end, use_T) - 1}] c{cid}:")
        print(f"    {nudge_text!r}")
        print(f"  FREE  [{n_end}..{min(f_end, use_T) - 1}] greedy:")
        print(f"    {free_text!r}")
        print()

    # Commit-phase blocks.
    commit_idx = 0
    step = sched_end
    while step < use_T:
        n_start = step
        n_end   = n_start + args.nudge
        f_end   = n_start + commit_bs
        lbl = CLUSTER_LABELS.get(args.commit, f"c{args.commit}")
        print(f"─── Commit block {commit_idx}  c{args.commit} {lbl}  "
              f"(tokens {n_start}..{min(f_end, use_T) - 1}) ───")
        nudge_text = detok([int(t) for t in toks[n_start:min(n_end, use_T)]])
        free_text  = detok([int(t) for t in toks[n_end:min(f_end, use_T)]])
        print(f"  NUDGE [{n_start}..{min(n_end, use_T) - 1}] c{args.commit}:")
        print(f"    {nudge_text!r}")
        print(f"  FREE  [{n_end}..{min(f_end, use_T) - 1}] greedy:")
        print(f"    {free_text!r}")
        print()
        step = f_end
        commit_idx += 1


if __name__ == "__main__":
    main()
