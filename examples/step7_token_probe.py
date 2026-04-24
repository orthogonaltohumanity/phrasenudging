"""Step 7 — single-trajectory cluster nudge probe (single or scheduled).

Two modes on one prompt:

(1) SINGLE-CLUSTER mode: --cluster C + --portion SPEC
    Nudge one cluster during a chosen portion of the run; rest is greedy.

(2) SCHEDULED mode: --schedule "c6:50+35,c9:50+35,c2:50+35,c3:50+35"
    Run through an ordered list of (cluster, nudge-tokens, free-tokens)
    blocks. Optional --commit "c0:50+150" cycles after the schedule
    runs out, continuing until --n-tokens is reached.

Usage:
    # (1) Nudge toward cluster 1 during the LAST 50 tokens of a 200-token run
    python3 examples/step7_token_probe.py \\
        --pid L5_0002 --cluster 1 --n-tokens 200 --portion last:50

    # (2) Schedule 4 clusters at 50 nudge + 35 free each, then commit c0
    python3 examples/step7_token_probe.py \\
        --pid L5_0002 --n-tokens 600 \\
        --schedule "c6:50+35,c9:50+35,c2:50+35,c3:50+35" \\
        --commit "c0:50+150" --alpha 0.01 --commit-alpha 0.02 \\
        --compare-greedy

Portion format (mode 1):  <first|last|all>:<count> OR <first|last|all>:<pct>%
  first:30   → nudge the first 30 tokens
  first:30%  → nudge the first 30% of N tokens
  last:100   → nudge the last 100 tokens
  last:50%   → nudge the last 50% of N
  all        → alias for first:100%
  first:0    → no nudging (identical to greedy)

Schedule format (mode 2):  "c<id>:<nudge>+<free>,c<id>:<nudge>+<free>,..."
  Each block: <nudge> tokens SLERP toward that cluster, then <free>
  tokens of greedy. `+<free>` may be omitted (same as +0).
  Commit block uses same grammar; it is cycled until n_tokens ends.

Output: decoded text with boundary markers for every nudge span, and
optionally a side-by-side pure-greedy trace on the same prompt.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_controller import (
    PROMPT_HEAD, PROMPT_TAIL, INSTRUCTION,
    load_phrase_amps, slerp, softmax, sample_from_logits, extract_boxed,
    parse_schedule, parse_commit, build_plan, is_correct,
    parse_force_at, truncate_plan, force_box_emission,
)
from sparse_utils import reconstruct_amp


def parse_portion(s: str) -> tuple[str, float | int, bool]:
    """Parse '<first|last|all>[:<amount>[%]]' into
    (position, amount, is_percent).

    Bare integer amount → token COUNT (int, is_percent=False).
    Integer with trailing '%' → percentage (float in [0,1], is_percent=True).
    'all' without amount → (all, 1.0, True) — nudge everything.
    """
    s = s.strip().lower()
    # "all" alone is shorthand for "all:100%".
    if s == "all":
        return "all", 1.0, True
    if ":" not in s:
        raise ValueError(
            f"bad --portion {s!r}; expected e.g. 'last:100' or 'last:50%'")
    pos, amt = s.split(":", 1)
    pos = pos.strip()
    amt = amt.strip()
    if pos not in ("first", "last", "all"):
        raise ValueError(f"--portion position must be first|last|all, got {pos!r}")
    is_pct = amt.endswith("%")
    body = amt[:-1] if is_pct else amt
    try:
        val = float(body)
    except ValueError:
        raise ValueError(f"--portion amount must be a number, got {amt!r}")
    if is_pct:
        if not (0 <= val <= 100):
            raise ValueError(f"--portion percent must be in [0, 100], got {val}")
        return pos, val / 100.0, True
    # Bare integer = token count.
    if val < 0 or val != int(val):
        raise ValueError(f"--portion token count must be a non-neg int, got {amt!r}")
    return pos, int(val), False


def resolve_portion(position: str, amount, is_percent: bool,
                     n_tokens: int) -> tuple[int, int]:
    """Return (nudge_start, nudge_end) token indices for the chosen portion.
    Clamps the nudge count to n_tokens and warns via print if clipped.
    """
    if is_percent:
        n_nudge = int(round(n_tokens * amount))
    else:
        n_nudge = int(amount)
    if n_nudge > n_tokens:
        print(f"  [warn] --portion requested {n_nudge} tokens but "
              f"--n-tokens={n_tokens}; clamping to {n_tokens}")
        n_nudge = n_tokens
    n_nudge = max(0, n_nudge)
    if position == "first" or position == "all":
        return 0, n_nudge
    # position == "last"
    return n_tokens - n_nudge, n_tokens


def find_problem(pid: str, candidates: list[str]) -> dict | None:
    for path in candidates:
        if not Path(path).exists(): continue
        for line in open(path):
            r = json.loads(line)
            if r.get("problem_id") == pid:
                return r
    return None


def generate_scheduled(llm, prompt_tokens, plan, phrase_seqs_by_cid,
                        alpha, alpha_commit, commit_cid, W, V, rng,
                        free_temp: float = 0.0, nudge_temp: float = 1.0):
    """Decode under a span plan = list of (kind, cid_or_None, length).

    Nudge spans SLERP each token toward a phrase from the cluster's pool
    (a fresh phrase is sampled at each nudge-span boundary). Free spans
    are greedy argmax. `alpha_commit` (optional) overrides `alpha` during
    commit-cluster nudges — pass commit_cid=None to always use `alpha`.
    Returns (tokens, span_tags) where span_tags[i] = (kind, cid) for step i.
    """
    from llama_cpp import llama_get_logits_ith
    eos = llm.token_eos(); ctx = llm._ctx.ctx
    llm.reset(); llm.eval(prompt_tokens)

    tokens: list[int] = []
    span_tags: list[tuple[str, int | None]] = []
    buf = np.zeros(V, dtype=np.float32)

    for span in plan:
        # Accept 4-tuple (kind, cid, length, alpha) spans; 3-tuples fall
        # back to alpha=None for forward compat.
        if len(span) == 4:
            kind, cid, length, span_alpha = span
        else:
            kind, cid, length = span; span_alpha = None
        cur_phrase = None  # fresh phrase at each span boundary
        for s_in_span in range(length):
            ptr = llama_get_logits_ith(ctx, -1)
            logits = np.ctypeslib.as_array(ptr, shape=(V,)).astype(np.float64)

            if kind in ("nudge_sched", "nudge_commit", "nudge"):
                if cur_phrase is None:
                    seqs = phrase_seqs_by_cid[cid]
                    cur_phrase = seqs[rng.randrange(len(seqs))]
                indptr, idx_arr, val_arr = cur_phrase
                pos = int(s_in_span * W / max(1, length))
                if pos >= W:
                    pos = W - 1
                target = reconstruct_amp(indptr, idx_arr, val_arr, pos, V, buf)
                P = softmax(logits, temp=nudge_temp)
                a = np.sqrt(P).astype(np.float32)
                if span_alpha is not None:
                    al = span_alpha
                elif kind == "nudge_commit" or (commit_cid is not None
                                                 and cid == commit_cid):
                    al = alpha_commit if alpha_commit is not None else alpha
                else:
                    al = alpha
                a_new = slerp(a, target, al)
                P_new = (a_new * a_new).astype(np.float64)
                tot = P_new.sum()
                if tot > 0: P_new /= tot
                cdf = np.cumsum(P_new); cdf[-1] = 1.0
                u = rng.random()
                tok = int(np.searchsorted(cdf, u, side="right"))
                if tok >= V: tok = V - 1
                span_tags.append(("nudge", cid))
            else:  # free
                tok = sample_from_logits(logits, free_temp, rng, V)
                span_tags.append(("free", None))

            if tok == eos:
                return tokens, span_tags
            tokens.append(tok)
            try:
                llm.eval([tok])
            except Exception:
                return tokens, span_tags

    return tokens, span_tags


def generate_baseline(llm, prompt_tokens, n_tokens, temp=0.0, rng=None):
    """Baseline decode. temp <= 0 → argmax (deterministic); temp > 0 → sampled."""
    from llama_cpp import llama_get_logits_ith
    V = llm.n_vocab(); eos = llm.token_eos(); ctx = llm._ctx.ctx
    llm.reset(); llm.eval(prompt_tokens)
    tokens: list[int] = []
    for _ in range(n_tokens):
        ptr = llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(V,)).astype(np.float64)
        tok = sample_from_logits(logits, temp, rng, V)
        if tok == eos: break
        tokens.append(tok)
        try: llm.eval([tok])
        except Exception: break
    return tokens


def format_scheduled_run(llm, tokens, plan) -> str:
    """Walk the plan, slice tokens accordingly, emit boundary markers
    around each nudge span. Free spans render as plain greedy text."""
    def det(ts): return llm.detokenize(ts).decode("utf-8", errors="replace")
    parts: list[str] = []
    idx = 0
    abs_pos = 0
    for span in plan:
        if len(span) == 4:
            kind, cid, length, span_alpha = span
        else:
            kind, cid, length = span; span_alpha = None
        if idx >= len(tokens):
            break
        seg_end = min(idx + length, len(tokens))
        seg = tokens[idx:seg_end]
        if kind in ("nudge_sched", "nudge_commit", "nudge"):
            a_tag = f"  α={span_alpha}" if span_alpha is not None else ""
            parts.append(
                f"\n⟨⟨ c{cid} nudge START @ step {abs_pos} "
                f"(len {len(seg)}){a_tag} ⟩⟩\n")
            parts.append(det(seg))
            parts.append(
                f"\n⟨⟨ c{cid} nudge END   @ step {abs_pos + len(seg)} ⟩⟩\n")
        else:
            parts.append(det(seg))
        idx = seg_end
        abs_pos += len(seg)
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # One of --cluster or --schedule is required (enforced manually below).
    ap.add_argument("--cluster", type=int, default=None,
                    help="(single-cluster mode) cluster id to nudge toward; "
                         "use --portion to locate the nudge span.")
    ap.add_argument("--portion", default="last:50",
                    help="(single-cluster mode) <first|last|all>:<count> "
                         "or :<pct>%%, e.g. 'last:50' or 'last:50%%'")
    ap.add_argument("--schedule", default=None,
                    help="(scheduled mode) comma-list of 'c<id>:<nudge>+<free>' "
                         'blocks, e.g. "c6:50+35,c9:50+35,c2:50+35,c3:50+35".')
    ap.add_argument("--commit", default=None,
                    help="(scheduled mode) commit block cycled after the "
                         'schedule runs out, e.g. "c0:50+150".')
    ap.add_argument("--commit-alpha", type=float, default=None,
                    help="SLERP alpha for commit-cluster nudges (defaults "
                         "to --alpha if unset).")

    ap.add_argument("--n-tokens", type=int, default=150,
                    help="Total tokens to generate")

    ap.add_argument("--pid", default=None,
                    help="Problem id (looked up in data/problems.jsonl or "
                         "data/problems_sample.jsonl). If not set, use --prompt.")
    ap.add_argument("--prompt", default=None,
                    help="Raw problem text (used if --pid not set). No "
                         "chat-template wrapping — pass the full body.")

    ap.add_argument("--phrase-amps",
                    default="data/cd_demo_phrase_amps_meta.pkl")
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
    ap.add_argument("--alpha", type=float, default=0.02,
                    help="SLERP alpha (default 0.02; try 0.01-0.05)")
    ap.add_argument("--baseline-temp", type=float, default=0.0,
                    help="Baseline (--compare-greedy) temperature. 0 "
                         "(default) = argmax greedy; >0 = sampled.")
    ap.add_argument("--free-temp", type=float, default=0.0,
                    help="Controller free-window (non-nudge) temperature. "
                         "0 (default) = argmax greedy; >0 = sampled.")
    ap.add_argument("--nudge-temp", type=float, default=1.0,
                    help="Controller nudge-step temperature applied to "
                         "logits before softmax → SLERP. Default 1.0 "
                         "reproduces prior behavior.")
    ap.add_argument("--seed", type=int, default=2024)

    ap.add_argument("--n-ctx", type=int, default=6144)
    ap.add_argument("--compare-greedy", action="store_true",
                    help="Also run pure greedy for the same prompt/length "
                         "and print side-by-side")
    ap.add_argument("--force-commit", action="store_true",
                    help="If the natural decode reaches --force-at without a "
                         "\\boxed{...} answer, inject --force-prefix and decode "
                         "greedily until '}' closes. Normalizes the pred-rate "
                         "so both truncation and hedge-loop failures get scored.")
    ap.add_argument("--force-at", default="end",
                    help="Where to cut off natural decode and force-commit. "
                         "One of: 'end' (default: after full --n-tokens / plan), "
                         "'post-schedule' (scheduled mode only: right after the "
                         "last schedule block, before the commit cycle), "
                         "<int> (absolute token position), <pct>%% (percent of "
                         "--n-tokens). If a \\boxed{} is emitted naturally BEFORE "
                         "the cutoff, the forced commit is skipped.")
    ap.add_argument("--force-prefix",
                    default="\n\nFinal answer: \\boxed{",
                    help='Forcing prefix inserted when --force-commit is on '
                         'and no box is present (default: "\\n\\nFinal answer: \\\\boxed{")')
    ap.add_argument("--force-budget", type=int, default=50,
                    help="Max tokens to generate inside the forced \\boxed{ (default 50)")
    ap.add_argument("--out", default=None,
                    help="Optional: write the decoded text to this file")
    args = ap.parse_args()

    use_single = args.cluster is not None
    use_sched = args.schedule is not None
    if use_single and use_sched:
        sys.exit("--cluster and --schedule are mutually exclusive")
    if not use_single and not use_sched:
        sys.exit("specify one of --cluster (single-cluster) or --schedule (scheduled)")

    # Build the span plan.
    commit_block = parse_commit(args.commit) if args.commit else None
    commit_cid = commit_block[0] if commit_block else None

    schedule_total = 0  # sum of schedule-phase lengths (for 'post-schedule')
    if use_single:
        pos, amt, is_pct = parse_portion(args.portion)
        nudge_start, nudge_end = resolve_portion(pos, amt, is_pct, args.n_tokens)
        print(f"portion={args.portion}  →  nudge span = tokens "
              f"[{nudge_start} .. {nudge_end})   of 0..{args.n_tokens}  "
              f"({nudge_end - nudge_start} nudged, "
              f"{args.n_tokens - (nudge_end - nudge_start)} greedy)")
        # Single-cluster plan uses --alpha (no per-block override) so span
        # alpha is None → generator falls back to args.alpha.
        plan: list[tuple[str, int | None, int, float | None]] = []
        if nudge_start > 0:
            plan.append(("free", None, nudge_start, None))
        if nudge_end > nudge_start:
            plan.append(("nudge_sched", args.cluster,
                          nudge_end - nudge_start, None))
        if args.n_tokens > nudge_end:
            plan.append(("free", None, args.n_tokens - nudge_end, None))
    else:
        sched_blocks = parse_schedule(args.schedule)
        schedule_total = sum(n + f for _, n, f, _ in sched_blocks)
        plan = build_plan(sched_blocks, commit_block, args.n_tokens)

    # Resolve --force-at; truncate the plan if cutoff < plan end.
    force_at_pos = parse_force_at(args.force_at, args.n_tokens, schedule_total)
    plan_total = sum(s[2] for s in plan)
    if args.force_commit and force_at_pos < plan_total:
        print(f"\n[force-at] cutoff at token {force_at_pos} "
              f"(was plan total {plan_total}); truncating plan")
        plan = truncate_plan(plan, force_at_pos)

    # Show the resolved plan.
    print("\n[plan]")
    ppos = 0
    for span in plan:
        kind, cid, length, span_alpha = span
        is_nudge = kind in ("nudge_sched", "nudge_commit", "nudge")
        label = f"c{cid} nudge" if is_nudge else "free (greedy)"
        marker = " (commit)" if kind == "nudge_commit" else ""
        # Resolve effective alpha for display.
        if is_nudge:
            if span_alpha is not None:
                eff_a = span_alpha
            elif kind == "nudge_commit":
                eff_a = args.commit_alpha if args.commit_alpha is not None else args.alpha
            else:
                eff_a = args.alpha
            a_tag = f"  α={eff_a}" + ("*" if span_alpha is not None else "")
        else:
            a_tag = ""
        print(f"  [{ppos:>4}..{ppos+length:>4})  ({length:>3} tk)  "
              f"{label}{marker}{a_tag}")
        ppos += length
    total = sum(s[2] for s in plan)
    print(f"  total: {total} tk" + ("" if total == args.n_tokens
                                      else f"  (note: != --n-tokens={args.n_tokens})"))
    print("  (α* marks per-block override; others use --alpha / --commit-alpha)")

    # Resolve problem / prompt.
    if args.pid:
        problem = find_problem(args.pid, ["data/problems.jsonl",
                                          "data/problems_sample.jsonl",
                                          "data/raw/fever_sample.jsonl",
                                          "data/raw/strategyqa_sample.jsonl",
                                          "data/raw/fever_200.jsonl",
                                          "data/raw/strategyqa_50.jsonl"])
        if problem is None:
            sys.exit(f"--pid {args.pid} not found in known dataset files")
        problem_text = problem["problem"]
        gold = problem.get("answer")
        print(f"pid={args.pid}  gold={gold!r}")
    elif args.prompt:
        problem_text = args.prompt
        gold = None
        print("using raw --prompt (no pid)")
    else:
        sys.exit("specify either --pid or --prompt")

    # Phrase amps — check all clusters referenced by the plan are present.
    phrase_seqs_by_cid, W, V = load_phrase_amps(args.phrase_amps)
    required_cids = sorted({s[1] for s in plan
                             if s[0] in ("nudge_sched", "nudge_commit", "nudge")})
    missing = [c for c in required_cids if c not in phrase_seqs_by_cid]
    if missing:
        sys.exit(f"clusters {missing} not in phrase-amps "
                 f"(available: {sorted(phrase_seqs_by_cid.keys())})")
    for c in required_cids:
        print(f"  c{c}: {len(phrase_seqs_by_cid[c])} phrases")

    # Model.
    print(f"loading model {args.model} ...", flush=True)
    t0 = time.time()
    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=-1,
                logits_all=False, verbose=False)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    # Prompt tokens (matches the chat template used by run_controller.py).
    head = llm.tokenize(PROMPT_HEAD.encode(), add_bos=True, special=True)
    body = llm.tokenize(f"{problem_text}\n\n{INSTRUCTION}".encode(),
                         add_bos=False, special=False)
    tail = llm.tokenize(PROMPT_TAIL.encode(), add_bos=False, special=True)
    prompt_tokens = head + body + tail

    # --- Nudged run ---
    rng = random.Random(args.seed)
    alpha_tag = (f"α={args.alpha}"
                 + (f" α_commit={args.commit_alpha}"
                    if args.commit_alpha is not None else ""))
    mode_tag = (f"cluster=c{args.cluster} portion={args.portion}"
                if use_single else f"schedule={args.schedule}"
                + (f" commit={args.commit}" if args.commit else ""))
    print(f"\n[nudged run] {alpha_tag}  {mode_tag}", flush=True)
    t0 = time.time()
    toks_n, span_tags = generate_scheduled(
        llm, prompt_tokens, plan, phrase_seqs_by_cid,
        args.alpha, args.commit_alpha, commit_cid, W, V, rng,
        free_temp=args.free_temp, nudge_temp=args.nudge_temp)
    dt = time.time() - t0
    n_nudged = sum(1 for k, _ in span_tags if k == "nudge")
    print(f"  produced {len(toks_n)} tokens ({n_nudged} nudged)  "
          f"({dt:.1f}s, {len(toks_n)/max(dt,1e-9):.0f} tk/s)", flush=True)

    # --- Force commit (nudged) ---
    forced_n_tokens: list[int] = []
    natural_text_n = llm.detokenize(toks_n).decode("utf-8", errors="replace")
    if args.force_commit and extract_boxed(natural_text_n) is None:
        print(f"  [force-commit] no \\boxed{{}} in natural output; injecting prefix",
              flush=True)
        forced_n_tokens = force_box_emission(
            llm, args.force_prefix, args.force_budget, rng=None, temp=0.0)
        toks_n = toks_n + forced_n_tokens

    nudged_text = format_scheduled_run(llm, toks_n[:len(toks_n) - len(forced_n_tokens)], plan)
    if forced_n_tokens:
        forced_text = llm.detokenize(forced_n_tokens).decode("utf-8", errors="replace")
        nudged_text += (f"\n⟨⟨ FORCED COMMIT "
                         f"(+{len(forced_n_tokens)} tk) ⟩⟩"
                         + forced_text)
    print("\n── NUDGED RUN ──────────────────────────────────────────")
    print(nudged_text)
    print("────────────────────────────────────────────────────────")

    # --- Greedy comparison ---
    greedy_text = None
    toks_g: list[int] = []
    forced_g_tokens: list[int] = []
    if args.compare_greedy:
        btemp_tag = "greedy" if args.baseline_temp <= 0 else f"T={args.baseline_temp}"
        # Cap the greedy run at the same cutoff point the controller uses.
        greedy_cap = (force_at_pos if (args.force_commit
                                         and force_at_pos < args.n_tokens)
                      else args.n_tokens)
        print(f"\n[baseline {btemp_tag}]  n_tokens={greedy_cap}", flush=True)
        t0 = time.time()
        base_rng = random.Random(args.seed)
        toks_g = generate_baseline(llm, prompt_tokens, greedy_cap,
                                    temp=args.baseline_temp, rng=base_rng)
        dt = time.time() - t0
        print(f"  produced {len(toks_g)} tokens  ({dt:.1f}s)", flush=True)

        natural_text_g = llm.detokenize(toks_g).decode("utf-8", errors="replace")
        if args.force_commit and extract_boxed(natural_text_g) is None:
            print(f"  [force-commit] no \\boxed{{}} in greedy output; injecting prefix",
                  flush=True)
            forced_g_tokens = force_box_emission(
                llm, args.force_prefix, args.force_budget, rng=None, temp=0.0)
            toks_g = toks_g + forced_g_tokens

        greedy_text = llm.detokenize(toks_g).decode("utf-8", errors="replace")
        print("\n── GREEDY BASELINE ─────────────────────────────────────")
        print(greedy_text)
        print("────────────────────────────────────────────────────────")

        # Per-step divergence + which span it fell inside (if any).
        first_div = next((i for i in range(min(len(toks_n), len(toks_g)))
                          if toks_n[i] != toks_g[i]), None)
        if first_div is None:
            print(f"\nNO DIVERGENCE in the first "
                  f"{min(len(toks_n), len(toks_g))} tokens.")
        else:
            tag = span_tags[first_div] if first_div < len(span_tags) else None
            where = (f"inside c{tag[1]} nudge span" if tag and tag[0] == "nudge"
                     else "inside a free span" if tag else "past plan end")
            print(f"\nFIRST DIVERGENCE at token {first_div}  ({where})")

    # --- Verdict (if we know gold) ---
    full_nudged = llm.detokenize(toks_n).decode("utf-8", errors="replace")
    pred_n = extract_boxed(full_nudged)
    ok_n = is_correct(pred_n, gold) if gold is not None else None
    tag_n = ("FORCED" if forced_n_tokens else "natural")
    print(f"\n── VERDICT ─────────────────────────────────────────────")
    print(f"  nudged  [{tag_n}]   pred={pred_n!r}   "
          f"ok={ok_n}  gold={gold!r}")
    if args.compare_greedy:
        pred_g = extract_boxed(greedy_text)
        ok_g = is_correct(pred_g, gold) if gold is not None else None
        tag_g = ("FORCED" if forced_g_tokens else "natural")
        print(f"  greedy  [{tag_g}]   pred={pred_g!r}   "
              f"ok={ok_g}  gold={gold!r}")
    print(f"────────────────────────────────────────────────────────")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            f.write(f"# step 7 probe — {mode_tag}  {alpha_tag}  "
                    f"seed={args.seed}\n")
            f.write(f"# pid={args.pid}  gold={gold!r}  n_tokens={args.n_tokens}\n\n")
            f.write("## PLAN\n\n")
            ppos2 = 0
            for span in plan:
                kind, cid, length, span_alpha = span
                is_n = kind in ("nudge_sched", "nudge_commit", "nudge")
                label = f"c{cid} nudge" if is_n else "free (greedy)"
                marker = " (commit)" if kind == "nudge_commit" else ""
                a_tag = f"  α={span_alpha}*" if span_alpha is not None else ""
                f.write(f"  [{ppos2:>4}..{ppos2+length:>4})  {label}{marker}{a_tag}\n")
                ppos2 += length
            f.write("\n## NUDGED RUN\n\n")
            f.write(nudged_text + "\n")
            if greedy_text is not None:
                f.write("\n## GREEDY BASELINE\n\n")
                f.write(greedy_text + "\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
