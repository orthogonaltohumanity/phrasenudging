"""Step 8 — Markov-process nudge probe.

Define N clusters plus an (N+1)×(N+1) transition matrix over them (the
extra state is "free generation"). At each token we:

  1. Sample next_state ~ T[current_state]  (row of the transition matrix)
  2. If next_state ∈ [0, N):  SLERP-nudge the logits toward clusters[state]
                              at a fixed --alpha
  3. If next_state == N:      free-sample (argmax if --free-temp=0, else T)

The process updates every token — no fixed block structure, just a
Markov chain over nudge targets. This is a strictly more general control
surface than step 7's explicit span plan: any plan corresponds to a
transition matrix with appropriate cycle structure.

Run modes (choose one):

  A. Single probe:        --pid <id> (optionally --n-seeds 1)    or --prompt <text>
  B. Seed sweep (1 pid):  --pid <id> --n-seeds <N>               JSONL out
  C. Preset sweep:        --preset {demo|full|custom}            JSONL out
     (multi-pid × optional multi-seed)

Single + single-seed → pretty-printed to stdout.
Any other combination → JSONL at --out (default: data/markov_<preset>.jsonl),
resume-friendly on the (problem_id, seed) pair.

Usage:
    # A. single probe
    python3 examples/step8_markov_nudge.py --pid L4_0020 --n-tokens 600 \\
        --clusters "c4,c1,c11,c7" \\
        --transition "0.8,0.05,0.05,0.05,0.05;0.1,0.7,0.1,0.05,0.05;..." \\
        --alpha 0.02 --compare-greedy --force-commit

    # B. seed sweep on one pid
    python3 examples/step8_markov_nudge.py --pid L4_0020 --n-seeds 10 \\
        --clusters "c4,c1,c11,c7" --transition "..." --alpha 0.02 \\
        --baseline-temp 1.0 --force-commit

    # C. full 500-problem sweep
    python3 examples/step8_markov_nudge.py --preset full \\
        --clusters "c4,c1,c11,c7" --transition "..." --alpha 0.02 \\
        --force-commit

Transition file format (JSON): {"transition": [[...], ...]} OR a bare
2D array. Or .npy.  Rows must be non-negative and each row is
renormalized to sum to 1.
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_controller import (
    PROMPT_HEAD, PROMPT_TAIL, INSTRUCTION,
    load_phrase_amps, slerp, softmax, sample_from_logits, extract_boxed,
    is_correct, force_box_emission,
)
from sparse_utils import reconstruct_amp


PRESETS = {
    "demo":   {"problems": "data/problems_sample.jsonl",
               "pids":     "data/problems_demo_pids.txt",
               "out":      "data/markov_demo.jsonl"},
    "full":   {"problems": "data/problems.jsonl",
               "pids":     "data/test_500_hard_first.txt",
               "out":      "data/markov_full500.jsonl"},
    "custom": {"problems": None, "pids": None, "out": "data/markov_custom.jsonl"},
}


# -------------------------------------------------------- config parsing

def parse_clusters(s: str) -> list[int]:
    """Parse 'c4,c1,c11,c7' → [4, 1, 11, 7]."""
    cids: list[int] = []
    for chunk in s.split(","):
        chunk = chunk.strip().lower()
        if not chunk:
            continue
        if chunk.startswith("c"):
            chunk = chunk[1:]
        try:
            cids.append(int(chunk))
        except ValueError:
            raise ValueError(f"bad cluster id {chunk!r} in --clusters")
    if not cids:
        raise ValueError("--clusters must list at least one cluster id")
    return cids


def parse_transition(s: str, expected_dim: int) -> np.ndarray:
    """Parse inline 'r00,r01;r10,r11;...' OR load a .json / .npy file.

    Rows are validated (non-negative, non-zero sum) and renormalized so
    each row sums exactly to 1. The result has shape (expected_dim, expected_dim).
    """
    if os.path.isfile(s):
        if s.endswith(".npy"):
            T = np.load(s).astype(np.float64)
        elif s.endswith(".json"):
            data = json.load(open(s))
            if isinstance(data, dict) and "transition" in data:
                data = data["transition"]
            T = np.asarray(data, dtype=np.float64)
        else:
            raise ValueError(f"unknown file extension for --transition {s!r}")
    else:
        rows = []
        for row_str in s.split(";"):
            row_str = row_str.strip()
            if not row_str:
                continue
            try:
                cols = [float(x.strip()) for x in row_str.split(",") if x.strip()]
            except ValueError:
                raise ValueError(f"bad float in transition row {row_str!r}")
            rows.append(cols)
        if not rows:
            raise ValueError("--transition parsed as empty")
        T = np.array(rows, dtype=np.float64)

    if T.ndim != 2 or T.shape[0] != expected_dim or T.shape[1] != expected_dim:
        raise ValueError(
            f"transition matrix shape {T.shape} != ({expected_dim},{expected_dim}); "
            f"you gave --clusters with N={expected_dim-1} → need {expected_dim} rows "
            f"and {expected_dim} cols")
    if (T < 0).any():
        raise ValueError("transition matrix has negative entries")
    row_sums = T.sum(axis=1)
    if (row_sums <= 0).any():
        bad = [i for i, r in enumerate(row_sums) if r <= 0]
        raise ValueError(f"transition matrix rows {bad} have zero sum")
    return T / row_sums[:, None]


def state_label(s: int, clusters: list[int]) -> str:
    return "free" if s >= len(clusters) else f"c{clusters[s]}"


def find_problem(pid: str, candidates: list[str]) -> dict | None:
    for path in candidates:
        if not Path(path).exists():
            continue
        for line in open(path):
            r = json.loads(line)
            if r.get("problem_id") == pid:
                return r
    return None


def resolve_problems(args):
    """Return (problems_list, out_path, multi_mode)."""
    # Preset path (multi-problem)
    if args.preset:
        preset = PRESETS[args.preset]
        problems_path = args.problems or preset["problems"]
        pids_path = args.problem_ids_file or preset["pids"]
        if problems_path is None or pids_path is None:
            sys.exit("--preset custom requires --problems and --problem-ids-file")
        if not Path(problems_path).exists():
            sys.exit(f"problems file not found: {problems_path}")
        if not Path(pids_path).exists():
            sys.exit(f"problem-ids file not found: {pids_path}")
        pmap = {}
        for line in open(problems_path):
            r = json.loads(line); pmap[r["problem_id"]] = r
        pids = [l.strip() for l in open(pids_path) if l.strip()]
        if args.n is not None:
            if args.n > len(pids):
                print(f"WARNING: --n {args.n} exceeds PID list length "
                      f"{len(pids)}; using all {len(pids)}.")
            pids = pids[:args.n]
        missing = [p for p in pids if p not in pmap]
        if missing:
            print(f"WARNING: {len(missing)} pids in ids-file not found in "
                  f"problems jsonl (first 5: {missing[:5]})")
        problems = [pmap[pid] for pid in pids if pid in pmap]
        out_path = args.out or preset["out"]
        return problems, out_path, True

    # Single-pid path
    if args.pid:
        search = ["data/problems.jsonl",
                  "data/problems_sample.jsonl",
                  "data/raw/fever_sample.jsonl",
                  "data/raw/strategyqa_sample.jsonl",
                  "data/raw/fever_200.jsonl",
                  "data/raw/strategyqa_50.jsonl"]
        if args.problems:
            search = [args.problems] + search
        problem = find_problem(args.pid, search)
        if problem is None:
            sys.exit(f"--pid {args.pid} not found in known dataset files")
        # Multi-mode is triggered by n_seeds > 1 or explicit --out.
        multi = (args.n_seeds > 1) or (args.out is not None)
        return [problem], args.out, multi

    # Raw prompt path
    if args.prompt:
        if args.n_seeds > 1:
            sys.exit("--prompt is single-seed only; use --pid for seed sweeps")
        return ([{"problem_id": "raw_prompt",
                  "problem": args.prompt, "answer": None}],
                args.out, False)

    sys.exit("specify one of --preset, --pid, or --prompt")


def load_done_keys(path):
    """Read JSONL at path; return set of (pid, seed) keys already recorded."""
    done: set[tuple[str, int]] = set()
    if path and Path(path).exists():
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["problem_id"], int(r.get("seed", 0))))
            except Exception:
                pass
    return done


# -------------------------------------------------------- decoders

def generate_markov(llm, prompt_tokens, n_tokens, clusters, T, start_state,
                     phrase_seqs_by_cid, alpha, W, V, rng,
                     free_temp: float = 0.0, nudge_temp: float = 1.0):
    """Decode `n_tokens` under a Markov process over N clusters + a free state.

    Returns (tokens, state_trace) with state_trace[i] = state used to
    produce token i. Phrase sampled fresh on state entry (from a
    different state); resampled every W steps within the same state.
    """
    from llama_cpp import llama_get_logits_ith
    V_llm = llm.n_vocab(); eos = llm.token_eos(); ctx = llm._ctx.ctx
    llm.reset(); llm.eval(prompt_tokens)

    N = len(clusters)
    tokens: list[int] = []
    state_trace: list[int] = []
    buf = np.zeros(V, dtype=np.float32)

    state = start_state
    cur_phrase = None
    steps_in_state = 0

    for _ in range(n_tokens):
        row = T[state]
        cdf_states = np.cumsum(row); cdf_states[-1] = 1.0
        u = rng.random()
        new_state = int(np.searchsorted(cdf_states, u, side="right"))
        if new_state > N:
            new_state = N

        if new_state != state:
            cur_phrase = None
            steps_in_state = 0
        state = new_state
        state_trace.append(state)

        ptr = llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(V,)).astype(np.float64)

        if state < N:
            if cur_phrase is None or (steps_in_state > 0 and steps_in_state % W == 0):
                cid = clusters[state]
                seqs = phrase_seqs_by_cid[cid]
                cur_phrase = seqs[rng.randrange(len(seqs))]
            indptr, idx_arr, val_arr = cur_phrase
            pos = steps_in_state % W
            target = reconstruct_amp(indptr, idx_arr, val_arr, pos, V, buf)
            P = softmax(logits, temp=nudge_temp)
            a = np.sqrt(P).astype(np.float32)
            a_new = slerp(a, target, alpha)
            P_new = (a_new * a_new).astype(np.float64)
            tot = P_new.sum()
            if tot > 0:
                P_new /= tot
            cdf_tok = np.cumsum(P_new); cdf_tok[-1] = 1.0
            u2 = rng.random()
            tok = int(np.searchsorted(cdf_tok, u2, side="right"))
            if tok >= V_llm:
                tok = V_llm - 1
        else:
            tok = sample_from_logits(logits, free_temp, rng, V_llm)

        if tok == eos:
            break
        tokens.append(tok)
        try:
            llm.eval([tok])
        except Exception:
            break
        steps_in_state += 1

    return tokens, state_trace


def generate_baseline(llm, prompt_tokens, n_tokens, temp=0.0, rng=None):
    from llama_cpp import llama_get_logits_ith
    V = llm.n_vocab(); eos = llm.token_eos(); ctx = llm._ctx.ctx
    llm.reset(); llm.eval(prompt_tokens)
    tokens: list[int] = []
    for _ in range(n_tokens):
        ptr = llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(V,)).astype(np.float64)
        tok = sample_from_logits(logits, temp, rng, V)
        if tok == eos:
            break
        tokens.append(tok)
        try:
            llm.eval([tok])
        except Exception:
            break
    return tokens


def format_markov_run(llm, tokens, state_trace, clusters) -> str:
    """Render text with a boundary marker at every state transition."""
    def det(ts): return llm.detokenize(ts).decode("utf-8", errors="replace")
    if not tokens:
        return ""
    parts: list[str] = []
    start = 0
    cur_state = state_trace[0]
    abs_pos = 0
    for i in range(1, len(tokens)):
        if state_trace[i] != cur_state:
            seg = tokens[start:i]
            parts.append(f"\n⟨⟨ {state_label(cur_state, clusters)}  "
                          f"len={len(seg)}  @step {abs_pos} ⟩⟩\n")
            parts.append(det(seg))
            start = i
            abs_pos += len(seg)
            cur_state = state_trace[i]
    seg = tokens[start:]
    parts.append(f"\n⟨⟨ {state_label(cur_state, clusters)}  "
                  f"len={len(seg)}  @step {abs_pos} ⟩⟩\n")
    parts.append(det(seg))
    return "".join(parts)


# -------------------------------------------------------- per-run helpers

def tokenize_prompt(llm, problem_text: str):
    head = llm.tokenize(PROMPT_HEAD.encode(), add_bos=True, special=True)
    body = llm.tokenize(f"{problem_text}\n\n{INSTRUCTION}".encode(),
                         add_bos=False, special=False)
    tail = llm.tokenize(PROMPT_TAIL.encode(), add_bos=False, special=True)
    return head + body + tail


def run_one_markov(llm, prompt_tokens, clusters, T, start_state,
                    phrase_seqs_by_cid, W, V, seed, args):
    rng = random.Random(seed)
    toks, state_trace = generate_markov(
        llm, prompt_tokens, args.n_tokens, clusters, T, start_state,
        phrase_seqs_by_cid, args.alpha, W, V, rng,
        free_temp=args.free_temp, nudge_temp=args.nudge_temp)
    forced_tokens: list[int] = []
    natural_text = llm.detokenize(toks).decode("utf-8", errors="replace")
    forced = False
    if args.force_commit and extract_boxed(natural_text) is None:
        forced_tokens = force_box_emission(
            llm, args.force_prefix, args.force_budget, rng=None, temp=0.0)
        toks = toks + forced_tokens
        forced = True
    text = llm.detokenize(toks).decode("utf-8", errors="replace")
    N = len(clusters)
    occ = Counter(state_trace)
    occupancy = {state_label(s, clusters): occ.get(s, 0) for s in range(N + 1)}
    return dict(tokens=list(toks), text=text, pred=extract_boxed(text),
                state_trace=state_trace, state_occupancy=occupancy,
                T=len(toks), forced=forced)


def run_one_baseline(llm, prompt_tokens, seed, args):
    rng = random.Random(seed) if args.baseline_temp > 0 else None
    toks = generate_baseline(llm, prompt_tokens, args.n_tokens,
                              temp=args.baseline_temp, rng=rng)
    forced = False
    natural_text = llm.detokenize(toks).decode("utf-8", errors="replace")
    if args.force_commit and extract_boxed(natural_text) is None:
        extra = force_box_emission(
            llm, args.force_prefix, args.force_budget, rng=None, temp=0.0)
        toks = toks + extra
        forced = True
    text = llm.detokenize(toks).decode("utf-8", errors="replace")
    return dict(tokens=list(toks), text=text, pred=extract_boxed(text),
                T=len(toks), forced=forced)


# -------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])

    # Problem source (one of the three mutually exclusive paths).
    ap.add_argument("--preset", choices=list(PRESETS.keys()), default=None,
                    help="Problem-set preset — same semantics as step5")
    ap.add_argument("--problems", default=None,
                    help="Override preset's problems JSONL (or direct file "
                         "to search for --pid).")
    ap.add_argument("--problem-ids-file", default=None,
                    help="Override preset's pids list")
    ap.add_argument("--n", type=int, default=None,
                    help="Cap preset pids to first N")
    ap.add_argument("--pid", default=None,
                    help="Single problem id (looked up in --problems / known datasets)")
    ap.add_argument("--prompt", default=None,
                    help="Raw problem text (single-seed only)")

    ap.add_argument("--clusters", required=True,
                    help='Cluster list: "c4,c1,c11,c7". State i ∈ [0,N) means '
                         "nudge toward this cluster; state N is free generation.")
    ap.add_argument("--transition", required=True,
                    help='(N+1)×(N+1) transition matrix. Inline: '
                         '"r0c0,r0c1,...;r1c0,...;..." or .json/.npy path. '
                         "Rows renormalized to 1.")
    ap.add_argument("--start-state", type=int, default=None,
                    help="Initial state index (default: N = free)")
    ap.add_argument("--alpha", type=float, default=0.02,
                    help="Fixed SLERP α applied on every nudge step")

    ap.add_argument("--n-seeds", type=int, default=1,
                    help="Number of seeds per problem (default 1). Seeds are "
                         "--seed + i.")
    ap.add_argument("--seed", type=int, default=2024,
                    help="Seed base; subsequent seeds are base+1, base+2, ...")
    ap.add_argument("--n-tokens", type=int, default=600)
    ap.add_argument("--phrase-amps", default="data/cd_demo_phrase_amps_meta.pkl")
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")

    ap.add_argument("--baseline-temp", type=float, default=0.0)
    ap.add_argument("--free-temp", type=float, default=0.0)
    ap.add_argument("--nudge-temp", type=float, default=1.0)

    ap.add_argument("--compare-greedy", action="store_true",
                    help="Also run a baseline decode for each (pid, seed)")
    ap.add_argument("--force-commit", action="store_true")
    ap.add_argument("--force-prefix", default="\n\nFinal answer: \\boxed{")
    ap.add_argument("--force-budget", type=int, default=50)

    ap.add_argument("--n-ctx", type=int, default=6144)
    ap.add_argument("--out", default=None,
                    help="JSONL output path (always JSONL when multi-mode).")
    args = ap.parse_args()

    # Parse + validate config.
    clusters = parse_clusters(args.clusters)
    N = len(clusters)
    T = parse_transition(args.transition, N + 1)
    start_state = N if args.start_state is None else args.start_state
    if not (0 <= start_state <= N):
        sys.exit(f"--start-state must be in [0, {N}], got {start_state}")

    # Resolve problem list + output path.
    problems, out_path, multi = resolve_problems(args)
    if multi and out_path is None:
        out_path = "data/markov_custom.jsonl"

    # Show config.
    print(f"clusters: {clusters}  (N={N})  alpha={args.alpha}  "
          f"start_state={state_label(start_state, clusters)}")
    labels = [state_label(s, clusters) for s in range(N + 1)]
    print("transition matrix (rows renormalized to 1):")
    print("  " + "      " + " ".join(f"{l:>7}" for l in labels))
    for i, row in enumerate(T):
        print(f"  {labels[i]:>5}: " + " ".join(f"{v:>7.3f}" for v in row))
    print(f"problems: {len(problems)}   n_seeds: {args.n_seeds}   "
          f"total runs: {len(problems) * args.n_seeds}   "
          f"multi-mode: {multi}  out: {out_path}")

    # Phrase amps + cluster-availability check.
    phrase_seqs_by_cid, W, V = load_phrase_amps(args.phrase_amps)
    missing = [c for c in clusters if c not in phrase_seqs_by_cid]
    if missing:
        sys.exit(f"clusters {missing} not in phrase-amps "
                 f"(available: {sorted(phrase_seqs_by_cid.keys())})")
    for c in clusters:
        print(f"  c{c}: {len(phrase_seqs_by_cid[c])} phrases")

    # Load model once.
    print(f"\nloading model {args.model} ...", flush=True)
    t0 = time.time()
    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=-1,
                logits_all=False, verbose=False)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    # Resume-friendly JSONL append.
    done_keys = load_done_keys(out_path) if multi else set()
    out_f = None
    if multi:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(out_path, "a")
        if done_keys:
            print(f"[resume] skipping {len(done_keys)} (pid,seed) pairs "
                  f"already in {out_path}")

    total = len(problems) * args.n_seeds
    n_done = 0
    n_ok_markov = 0
    n_ok_base = 0
    n_attempts = 0
    seed_base = args.seed

    for prob in problems:
        pid = prob["problem_id"]
        gold = prob.get("answer")
        prompt_tokens = tokenize_prompt(llm, prob["problem"])

        # Baseline caching — deterministic baseline = 1 run per problem, reused.
        cached_baseline = None
        deterministic_baseline = args.baseline_temp <= 0.0

        for i in range(args.n_seeds):
            seed_i = seed_base + i
            if (pid, seed_i) in done_keys:
                n_done += 1
                continue

            t0 = time.time()
            markov_rec = run_one_markov(
                llm, prompt_tokens, clusters, T, start_state,
                phrase_seqs_by_cid, W, V, seed_i, args)
            m_ok = is_correct(markov_rec["pred"], gold) if gold is not None else None
            dt_m = time.time() - t0

            base_rec = None
            b_ok = None
            if args.compare_greedy:
                if deterministic_baseline and cached_baseline is not None:
                    base_rec = cached_baseline
                else:
                    t1 = time.time()
                    base_rec = run_one_baseline(llm, prompt_tokens, seed_i, args)
                    base_rec["_time_s"] = time.time() - t1
                    if deterministic_baseline:
                        cached_baseline = base_rec
                b_ok = is_correct(base_rec["pred"], gold) if gold is not None else None

            markov_rec["ok"] = m_ok
            if base_rec is not None:
                base_rec["ok"] = b_ok

            record = dict(problem_id=pid, gold=gold, seed=seed_i,
                           markov=markov_rec)
            if base_rec is not None:
                record["baseline"] = base_rec

            n_attempts += 1
            n_done += 1
            n_ok_markov += int(bool(m_ok))
            n_ok_base   += int(bool(b_ok))

            m_tag = ("OK" if m_ok else "--") if gold is not None else "??"
            b_tag = ("OK" if b_ok else "--") if (gold is not None and b_ok is not None) else "  "
            print(f"  [{n_done}/{total}] {pid} seed={seed_i}  "
                  f"markov T={markov_rec['T']} pred={markov_rec['pred']!r} "
                  f"{m_tag}{' [F]' if markov_rec['forced'] else ''}"
                  + (f"  | base T={base_rec['T']} pred={base_rec['pred']!r} "
                     f"{b_tag}{' [F]' if base_rec and base_rec['forced'] else ''}"
                     if base_rec else "")
                  + f"  ({dt_m:.1f}s)", flush=True)

            if multi:
                out_f.write(json.dumps(record, default=str) + "\n")
                out_f.flush()
            else:
                # Single-problem + single-seed pretty print.
                print_single(llm, record, clusters, args)

    if out_f:
        out_f.close()

    if multi:
        print(f"\n=== SUMMARY ===")
        print(f"  runs:     {n_attempts}/{total}   (resume skipped: {total - n_attempts})")
        if args.compare_greedy:
            print(f"  baseline: {n_ok_base}/{n_attempts}")
        print(f"  markov:   {n_ok_markov}/{n_attempts}")
        print(f"  wrote {out_path}")


def print_single(llm, record, clusters, args):
    """Pretty-print a single markov+baseline record for interactive runs."""
    markov = record["markov"]
    base = record.get("baseline")

    # Reconstruct the text with state markers.
    toks_n = markov["tokens"]
    state_trace = markov["state_trace"]
    forced = markov["forced"]
    n_forced = 0
    if forced:
        # Forced tokens weren't tagged into state_trace; they sit at the tail.
        n_forced = len(toks_n) - len(state_trace)
    body_tokens = toks_n[:len(toks_n) - n_forced]
    text_body = format_markov_run(llm, body_tokens, state_trace, clusters)
    if n_forced:
        forced_text = llm.detokenize(toks_n[-n_forced:]).decode(
            "utf-8", errors="replace")
        text_body += f"\n⟨⟨ FORCED COMMIT (+{n_forced} tk) ⟩⟩" + forced_text
    print("\n── MARKOV RUN ──────────────────────────────────────────")
    print(text_body)
    print("────────────────────────────────────────────────────────")

    # State occupancy.
    print("state occupancy:")
    for name, n in markov["state_occupancy"].items():
        total = len(state_trace)
        pct = 100 * n / max(total, 1)
        print(f"  {name:>5}: {n:>5} ({pct:.1f}%)")

    if base is not None:
        print("\n── GREEDY BASELINE ─────────────────────────────────────")
        print(base["text"])
        print("────────────────────────────────────────────────────────")

    print("\n── VERDICT ─────────────────────────────────────────────")
    tag = "FORCED" if forced else "natural"
    print(f"  markov  [{tag}]   pred={markov['pred']!r}   "
          f"ok={markov['ok']}  gold={record['gold']!r}")
    if base is not None:
        btag = "FORCED" if base["forced"] else "natural"
        print(f"  greedy  [{btag}]   pred={base['pred']!r}   "
              f"ok={base['ok']}  gold={record['gold']!r}")
    print("────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
