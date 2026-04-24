"""SLERP-nudged controller for scheduled cognitive-mode steering.

Model of generation (token positions 0 .. max_new-1):

  ── Schedule phase (steps 0 .. sched_end-1) ──────────────────────────
    block b ∈ [0, len(schedule)):
       cluster_b = schedule[b]
       tokens [b·sched_bs,         b·sched_bs + nudge_window):  NUDGE
         at decode step t:
           step_in_nudge = t - b·sched_bs
           pos = floor(step_in_nudge * W / nudge_window)
           amp_target = phrase_k[pos]                       # reconstruct dense (V,)
           a = sqrt(P_t)                                     # current amp
           a' = SLERP(a, amp_target, α = args.alpha)
           token ~ a'² / Σ a'²                               # inverse-CDF sample
       tokens [b·sched_bs + nudge_window, (b+1)·sched_bs): FREE (greedy)

  ── Commit phase (steps ≥ sched_end) ─────────────────────────────────
    Same structure but cluster=commit_cluster, α=args.commit_alpha,
    free length = args.commit_free_window.

Each nudge-block samples a fresh phrase uniformly from the target cluster.
At decode step t within that block, the controller walks through the
phrase's W positions monotonically (1:1 if nudge_window == W, or stretched
if nudge_window > W). This preserves the phrase's own time-ordering
(problem-entry → mid → commit-gesture) rather than treating it as a single
averaged target.

SLERP geometry
--------------
Both `a` and `amp_target` are unit-norm amps on S^{V-1}. SLERP with α∈[0,1]
traces the great-circle arc:
        θ  = arccos(⟨a, amp_target⟩)
        a' = sin((1-α)·θ)/sin θ · a + sin(α·θ)/sin θ · amp_target.
For small α this is ≈ a + α·θ·(amp_target - projection of amp_target onto a),
i.e. we move α·θ radians along the shortest-amp-distance path. We use
α = 0.01 schedule / 0.02 commit: a typical θ is ~0.5-1.0 rad, so the
effective per-step nudge is 0.005-0.02 rad — small enough to preserve most
of the model's own decision while reliably biasing it toward the target.
"""
import argparse
import json
import pickle
import random
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sparse_utils import reconstruct_amp


# DeepSeek-R1 chat-template tokens. We inline these so the repo is
# self-contained; see tokenizer_config.json if you switch models.
PROMPT_HEAD = "<｜begin▁of▁sentence｜><｜User｜>"
PROMPT_TAIL = "<｜Assistant｜><think>\n"
INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
BOXED_RE    = re.compile(r"\\boxed\{")


# ---------------------------------------------------------------------- utils

def extract_boxed(text: str) -> str | None:
    """Return the content of the *last* `\\boxed{...}` in ``text``, or None.

    Handles nested braces: walks forward counting brace depth.
    """
    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    for i, c in enumerate(text[start:]):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:start + i].strip()
    return None


def is_correct(pred, gold) -> bool:
    """Normalize then compare. Handles latex frac/dfrac, numeric coercion."""
    if pred is None or gold is None:
        return False

    def norm(s):
        s = str(s).strip().strip("$")
        s = re.sub(r"\s+", "", s).rstrip(".,;")
        s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        # regularize \frac braces
        s = re.sub(r"\\frac\{([^}]*)\}([0-9a-zA-Z])", r"\\frac{\1}{\2}", s)
        s = re.sub(r"\\frac([0-9a-zA-Z])\{([^}]*)\}", r"\\frac{\1}{\2}", s)
        s = re.sub(r"\\frac([0-9a-zA-Z])([0-9a-zA-Z])", r"\\frac{\1}{\2}", s)
        return s

    if norm(pred) == norm(gold):
        return True

    # Numeric fallback.
    def _num(s):
        if s is None: return None
        s = s.strip().strip("$").replace(",", "").replace(" ", "")
        try: return int(s)
        except Exception:
            try: return float(s)
            except Exception: return None
    p, g = _num(pred), _num(gold)
    if p is not None and g is not None:
        try: return abs(float(p) - float(g)) < 1e-6
        except Exception: return False
    return False


def softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax. temp=1.0 is the usual distribution."""
    x = np.asarray(logits, dtype=np.float64)
    if temp != 1.0:
        x = x / max(temp, 1e-12)
    x -= x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def sample_from_logits(logits: np.ndarray, temp: float, rng,
                        V: int) -> int:
    """Argmax when temp <= 0, else temperature-softmax inverse-CDF sample."""
    if temp <= 0.0:
        return int(np.argmax(logits))
    P = softmax(logits, temp=temp).astype(np.float64)
    cdf = np.cumsum(P); cdf[-1] = 1.0
    u = rng.random()
    tok = int(np.searchsorted(cdf, u, side="right"))
    if tok >= V: tok = V - 1
    return tok


def slerp(a: np.ndarray, a_target: np.ndarray, alpha: float) -> np.ndarray:
    """Spherical linear interpolation between unit amps ``a`` and ``a_target``.

    Returns ``sin((1-α)θ)/sinθ · a + sin(αθ)/sinθ · a_target`` which stays on
    the unit hypersphere (so its element-wise square is a valid probability
    distribution). Degenerates to ``a`` when θ is tiny.
    """
    dot = float(np.clip(np.dot(a, a_target), -1.0, 1.0))
    theta = float(np.arccos(dot))
    if theta < 1e-6:
        return a.astype(np.float32)
    sin_t = np.sin(theta)
    c1 = np.sin((1 - alpha) * theta) / sin_t
    c2 = np.sin(alpha * theta) / sin_t
    return (c1 * a + c2 * a_target).astype(np.float32)


# ----------------------------------------------- schedule / plan parsers

def parse_schedule(s: str) -> list[tuple[int, int, int, float | None]]:
    """Parse 'c6:50+35,c9:50@0.03,c2:50+35@0.02' into
    list of (cid, nudge, free, alpha).

    Grammar per block: c<id>:<nudge>[+<free>][@<alpha>]
      - '+<free>' may be omitted (defaults to 0)
      - '@<alpha>' may be omitted (None → caller falls back to --alpha /
        --commit-alpha depending on block phase)
    """
    blocks: list[tuple[int, int, int, float | None]] = []
    if not s or not s.strip():
        return blocks
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Split off @alpha suffix first, if present.
        alpha: float | None = None
        if "@" in chunk:
            chunk, alpha_str = chunk.rsplit("@", 1)
            chunk = chunk.strip()
            alpha_str = alpha_str.strip()
            try:
                alpha = float(alpha_str)
            except ValueError:
                raise ValueError(f"bad @alpha value in block {chunk!r}@{alpha_str!r}")
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(
                    f"@alpha must be in [0, 1], got {alpha} in {chunk!r}")
        if ":" not in chunk:
            raise ValueError(
                f"bad schedule entry {chunk!r}, expected 'c<id>:<N>[+<F>][@<α>]'")
        cid_str, tail = chunk.split(":", 1)
        cid_str = cid_str.strip().lower()
        if cid_str.startswith("c"):
            cid_str = cid_str[1:]
        try:
            cid = int(cid_str)
        except ValueError:
            raise ValueError(f"bad cluster id in {chunk!r}")
        if "+" in tail:
            n_str, f_str = tail.split("+", 1)
        else:
            n_str, f_str = tail, "0"
        try:
            n_tok = int(n_str); f_tok = int(f_str)
        except ValueError:
            raise ValueError(f"bad token counts in {chunk!r}")
        if n_tok < 0 or f_tok < 0:
            raise ValueError(f"negative token count in {chunk!r}")
        blocks.append((cid, n_tok, f_tok, alpha))
    return blocks


def parse_commit(s):
    """--commit accepts ONE block, same grammar as --schedule, or None."""
    if s is None or not str(s).strip():
        return None
    blocks = parse_schedule(s)
    if len(blocks) != 1:
        raise ValueError(
            f"--commit must be exactly one block, got {len(blocks)}: {s!r}")
    return blocks[0]


def parse_force_at(s, n_tokens: int, schedule_total: int = 0) -> int:
    """Resolve a --force-at string into an absolute token position.
    'end'           → n_tokens
    'post-schedule' → schedule_total (requires schedule_total > 0)
    '<int>'         → that many tokens (clamped to n_tokens)
    '<pct>%'        → percent of n_tokens
    """
    if s is None:
        return n_tokens
    s = str(s).strip().lower()
    if s == "end":
        return n_tokens
    if s == "post-schedule":
        if schedule_total <= 0:
            raise ValueError(
                "--force-at post-schedule requires a non-empty schedule")
        return min(schedule_total, n_tokens)
    if s.endswith("%"):
        try: pct = float(s[:-1])
        except ValueError: raise ValueError(f"bad --force-at pct {s!r}")
        if not (0 <= pct <= 100):
            raise ValueError(f"--force-at pct must be in [0, 100], got {pct}")
        return int(round(n_tokens * pct / 100))
    try: v = int(s)
    except ValueError:
        raise ValueError(f"bad --force-at {s!r}: expected 'end', "
                          "'post-schedule', <int>, or '<pct>%'")
    if v < 0:
        raise ValueError(f"--force-at must be non-negative, got {v}")
    return min(v, n_tokens)


def force_box_emission(llm, prefix: str, max_answer_tokens: int,
                        rng=None, temp: float = 0.0) -> list[int]:
    """Append a forcing `prefix` then decode until '}' or budget.

    Caller must ensure llm state is positioned at the end of the just-generated
    sequence. Returns the list of tokens newly appended (prefix + answer incl.
    closing '}' if reached).
    """
    from llama_cpp import llama_get_logits_ith
    V = llm.n_vocab(); eos = llm.token_eos(); ctx = llm._ctx.ctx
    prefix_tokens = list(llm.tokenize(prefix.encode(), add_bos=False, special=False))
    try: llm.eval(prefix_tokens)
    except Exception: return prefix_tokens
    appended: list[int] = list(prefix_tokens)
    for _ in range(max_answer_tokens):
        ptr = llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(V,)).astype(np.float64)
        tok = sample_from_logits(logits, temp, rng, V)
        if tok == eos: break
        appended.append(tok)
        try: llm.eval([tok])
        except Exception: break
        piece = llm.detokenize([tok]).decode("utf-8", errors="replace")
        if "}" in piece: break
    return appended


def build_plan(schedule_blocks, commit_block, n_tokens: int):
    """Expand schedule + optional commit cycle into contiguous spans
    summing to `n_tokens`.

    Input blocks are 4-tuples (cid, nudge, free, alpha_or_None).
    Output spans are 4-tuples (kind, cid, length, alpha_or_None) with
    kind ∈ {'nudge_sched', 'nudge_commit', 'free'}. The nudge_sched vs
    nudge_commit split lets the generator pick the right default alpha
    (--alpha vs --commit-alpha) when a block has no per-block @alpha.
    """
    spans = []
    used = 0
    def add(kind, cid, length, alpha):
        nonlocal used
        if length <= 0 or used >= n_tokens:
            return
        length = min(length, n_tokens - used)
        spans.append((kind, cid, length, alpha))
        used += length
    for cid, n_tok, f_tok, alpha in schedule_blocks:
        add("nudge_sched", cid, n_tok, alpha)
        if used >= n_tokens: break
        add("free", None, f_tok, None)
        if used >= n_tokens: break
    if commit_block is not None and used < n_tokens:
        ccid, cn, cf, calpha = commit_block
        if not (cn == 0 and cf == 0):
            while used < n_tokens:
                add("nudge_commit", ccid, cn, calpha)
                if used >= n_tokens: break
                add("free", None, cf, None)
    if used < n_tokens:
        add("free", None, n_tokens - used, None)
    return spans


def truncate_plan(plan, at_pos: int):
    """Return a new plan summing to at most at_pos tokens (preserves alpha)."""
    out = []
    used = 0
    for span in plan:
        kind, cid, L, alpha = span
        remain = at_pos - used
        if remain <= 0: break
        take = min(L, remain)
        out.append((kind, cid, take, alpha))
        used += take
    return out


# ---------------------------------------------------------- phrase-amps loader

def load_phrase_amps(meta_path: str) -> Tuple[Dict[int, List[tuple]], int, int]:
    """Load the meta pickle + npz produced by build_phrase_amps.py.

    Returns:
        phrase_seqs_by_cid: dict mapping cluster id → list of
                            (indptr, idx, val) tuples, one per phrase.
        W:       phrase window width
        n_vocab: vocabulary size
    """
    npz_path = (meta_path.replace("_meta.pkl", "_amps.npz")
                if meta_path.endswith("_meta.pkl")
                else str(Path(meta_path).with_suffix("")) + "_amps.npz")
    meta = pickle.load(open(meta_path, "rb"))
    if meta.get("schema_version") != "v2":
        raise SystemExit(
            f"{meta_path}: expected schema_version=v2, got "
            f"{meta.get('schema_version')}. Rebuild via build_phrase_amps.py.")
    W = meta["W"]
    V = meta["n_vocab"]
    print(f"  loading amp-seqs npz: {npz_path} ...", flush=True)
    npz = np.load(npz_path)
    phrase_seqs_by_cid: Dict[int, List[tuple]] = {}
    for cid in meta["schedule"]:
        lst = []
        for ph in meta["clusters"][cid]["phrases"]:
            key = ph["arr_key"]
            lst.append((npz[key + "_indptr"],
                         npz[key + "_idx"],
                         npz[key + "_val"]))
        phrase_seqs_by_cid[cid] = lst
    return phrase_seqs_by_cid, W, V


# ---------------------------------------------------------- decoder functions

def gen_baseline(llm, prompt_tokens, max_new, temp: float = 0.0, rng=None):
    """Baseline decode. temp <= 0 → greedy argmax (deterministic).
    temp > 0 → temperature-sampled using `rng` (reproducible if seeded)."""
    from llama_cpp import llama_get_logits_ith
    V = llm.n_vocab(); eos = llm.token_eos(); ctx = llm._ctx.ctx
    llm.reset()
    llm.eval(prompt_tokens)
    tokens: list[int] = []
    for _ in range(max_new):
        ptr = llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(V,)).astype(np.float64)
        tok = sample_from_logits(logits, temp, rng, V)
        if tok == eos:
            break
        tokens.append(tok)
        try: llm.eval([tok])
        except Exception: break
    text = llm.detokenize(tokens).decode("utf-8", errors="replace")
    return dict(tokens=tokens, text=text, pred=extract_boxed(text))


def gen_controlled(llm, prompt_tokens, plan, phrase_seqs_by_cid, rng, W,
                    alpha, commit_alpha, commit_cid,
                    free_temp: float = 0.0, nudge_temp: float = 1.0):
    """SLERP-nudged controlled decode following an explicit span plan.

    `plan` = list of (kind, cid, length, span_alpha). Nudge spans SLERP
    toward a fresh phrase (sampled at each span boundary) from cluster
    `cid`'s pool. Free spans are greedy (free_temp=0) or sampled.

    Alpha resolution per nudge span:
       span_alpha is not None  → use span_alpha       (per-block override)
       kind == 'nudge_commit'  → use commit_alpha
       else                    → use alpha
    `commit_cid` is kept for backward compatibility (falls back to
    commit_alpha when a schedule-phase span happens to reuse the commit
    cluster id), but the kind tag is the primary phase signal.
    """
    from llama_cpp import llama_get_logits_ith
    V = llm.n_vocab(); eos = llm.token_eos(); ctx = llm._ctx.ctx
    llm.reset()
    llm.eval(prompt_tokens)

    tokens: list[int] = []
    cluster_per_step: list[int | None] = []
    target_buf = np.zeros(V, dtype=np.float32)

    for span in plan:
        # Spans are 4-tuples (kind, cid, length, alpha). Accept 3-tuples as
        # a forward-compat fallback: treat alpha as None.
        if len(span) == 4:
            kind, cid, length, span_alpha = span
        else:
            kind, cid, length = span
            span_alpha = None
        cur_phrase = None  # fresh phrase at every span boundary
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
                target = reconstruct_amp(indptr, idx_arr, val_arr, pos, V, target_buf)
                P = softmax(logits, temp=nudge_temp)
                a = np.sqrt(P).astype(np.float32)
                if span_alpha is not None:
                    al = span_alpha
                elif kind == "nudge_commit" or cid == commit_cid:
                    al = commit_alpha
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
                cluster_per_step.append(cid)
            else:  # free
                tok = sample_from_logits(logits, free_temp, rng, V)
                cluster_per_step.append(None)

            if tok == eos:
                break
            tokens.append(tok)
            try: llm.eval([tok])
            except Exception:
                break
        else:
            continue
        break   # EOS or eval failure inside span → stop iterating spans too

    from collections import Counter
    usage = Counter(c for c in cluster_per_step if c is not None)
    text = llm.detokenize(tokens).decode("utf-8", errors="replace")
    return dict(
        tokens=tokens, text=text, pred=extract_boxed(text),
        cluster_usage={str(k): int(v) for k, v in usage.items()},
    )


# ---------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="GGUF model path")
    ap.add_argument("--problems", required=True,
                    help="JSONL with fields problem_id, problem, answer")
    ap.add_argument("--problem-ids-file", required=True,
                    help="Text file with one problem_id per line")
    ap.add_argument("--phrase-amps", required=True,
                    help="meta.pkl produced by build_phrase_amps.py")
    ap.add_argument("--schedule", required=True,
                    help='Comma-list of "c<id>:<nudge>+<free>[@<alpha>]" blocks, '
                         'e.g. "c4:50+35,c1:50+35@0.02,c11:50+35@0.03,c7:50+35". '
                         "<nudge> tokens SLERP toward that cluster, then <free> "
                         "tokens of greedy/free-sampled. '+<free>' may be omitted "
                         "(0 default). '@<alpha>' may be omitted (falls back to --alpha).")
    ap.add_argument("--commit", default=None,
                    help='(Legacy, optional.) One-block commit cycled after '
                         'schedule, e.g. "c6:50+150@0.03". Superseded by '
                         '--force-commit for terminating decoded runs; prefer '
                         'adding blocks to --schedule + using --force-commit.')
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="SLERP α default for schedule blocks w/o @α (default 0.01)")
    ap.add_argument("--commit-alpha", type=float, default=None,
                    help="SLERP α default for the commit block w/o @α "
                         "(only meaningful if --commit is set; default = --alpha)")
    ap.add_argument("--baseline-temp", type=float, default=0.0,
                    help="Baseline sampling temperature. 0 (default) = argmax "
                         "greedy; >0 = temperature-sampled.")
    ap.add_argument("--free-temp", type=float, default=0.0,
                    help="Controller free-window temperature. 0 (default) = "
                         "argmax greedy (matches prior behavior); >0 = sampled.")
    ap.add_argument("--nudge-temp", type=float, default=1.0,
                    help="Controller nudge-step temperature applied to logits "
                         "before softmax → SLERP. 1.0 (default) reproduces "
                         "prior behavior; <1 sharpens, >1 flattens.")
    ap.add_argument("--max-new", type=int, default=3000)
    ap.add_argument("--n-ctx", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--mode", choices=["both", "baseline", "controller"],
                    default="both")
    ap.add_argument("--force-commit", action="store_true",
                    help="After natural decode, if no \\boxed{} present inject "
                         "--force-prefix and decode until '}' closes. Normalizes "
                         "pred-rate across truncation / hedge-loop failures.")
    ap.add_argument("--force-at", default="end",
                    help="Where to cut off natural decode before force-commit. "
                         "One of: 'end' (default), 'post-schedule', <int>, <pct>%%")
    ap.add_argument("--force-prefix", default="\n\nFinal answer: \\boxed{",
                    help='Forcing prefix (default: "\\n\\nFinal answer: \\\\boxed{")')
    ap.add_argument("--force-budget", type=int, default=50,
                    help="Max tokens inside the forced \\boxed{ (default 50)")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    args = ap.parse_args()

    if args.commit_alpha is None:
        args.commit_alpha = args.alpha

    schedule_blocks = parse_schedule(args.schedule)
    commit_block    = parse_commit(args.commit)
    commit_cid      = commit_block[0] if commit_block else None
    schedule_total  = sum(n + f for _, n, f, _ in schedule_blocks)
    force_at_pos    = parse_force_at(args.force_at, args.max_new, schedule_total)
    baseline_cap    = (force_at_pos if (args.force_commit
                                          and force_at_pos < args.max_new)
                       else args.max_new)

    phrase_seqs_by_cid, W_seq, V_seq = load_phrase_amps(args.phrase_amps)
    needed = {cid for cid, _, _, _ in schedule_blocks}
    if commit_cid is not None:
        needed.add(commit_cid)
    missing = [c for c in sorted(needed) if c not in phrase_seqs_by_cid]
    if missing:
        raise SystemExit(f"clusters {missing} missing from phrase-amps")
    print(f"schedule={args.schedule}  commit={args.commit}  W={W_seq} "
          f"alpha={args.alpha} commit_alpha={args.commit_alpha}  "
          f"baseline_temp={args.baseline_temp} free_temp={args.free_temp} "
          f"nudge_temp={args.nudge_temp}", flush=True)

    p_map = {}
    for line in open(args.problems):
        r = json.loads(line); p_map[r["problem_id"]] = r
    pids = [l.strip() for l in open(args.problem_ids_file) if l.strip()]
    problems = [p_map[pid] for pid in pids if pid in p_map]
    print(f"n_problems: {len(problems)}  mode: {args.mode}", flush=True)

    from llama_cpp import Llama
    llm = Llama(model_path=args.model, n_ctx=args.n_ctx, n_gpu_layers=-1,
                logits_all=False, verbose=False)
    head = llm.tokenize(PROMPT_HEAD.encode(), add_bos=True, special=True)
    tail = llm.tokenize(PROMPT_TAIL.encode(), add_bos=False, special=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if Path(args.out).exists():
        # Resume-friendly: skip problem ids already in the output file.
        for line in open(args.out):
            try: done.add(json.loads(line)["problem_id"])
            except Exception: pass
    out_f = open(args.out, "a")

    ctrl_rng = random.Random(args.seed)
    base_rng = random.Random(args.seed)
    n_base = n_ctrl = n_done = 0

    for p in problems:
        pid = p["problem_id"]
        if pid in done:
            continue
        gold = p.get("answer")
        body = llm.tokenize(f"{p['problem']}\n\n{INSTRUCTION}".encode(),
                             add_bos=False, special=False)
        prompt_tokens = head + body + tail
        record = dict(problem_id=pid, gold=gold)

        if args.mode in ("both", "baseline"):
            t0 = time.time()
            b = gen_baseline(llm, prompt_tokens, baseline_cap,
                              temp=args.baseline_temp, rng=base_rng)
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
            ok = is_correct(b["pred"], gold)
            print(f"  {pid} base T={len(b['tokens']):>4} "
                  f"pred={b['pred']!r} "
                  f"{'OK' if ok else '--'}{' [FORCED]' if forced_b else ''} "
                  f"({dt:.1f}s)", flush=True)
            record["baseline"] = dict(pred=b["pred"], ok=ok,
                                       T=len(b["tokens"]), text=b["text"],
                                       forced=forced_b)
            n_base += int(ok)

        if args.mode in ("both", "controller"):
            t0 = time.time()
            plan = build_plan(schedule_blocks, commit_block, args.max_new)
            if args.force_commit and force_at_pos < sum(s[2] for s in plan):
                plan = truncate_plan(plan, force_at_pos)
            c = gen_controlled(llm, prompt_tokens, plan, phrase_seqs_by_cid,
                                ctrl_rng, W_seq,
                                args.alpha, args.commit_alpha, commit_cid,
                                free_temp=args.free_temp,
                                nudge_temp=args.nudge_temp)
            forced_c = False
            if args.force_commit and extract_boxed(c["text"]) is None:
                extra = force_box_emission(llm, args.force_prefix,
                                            args.force_budget,
                                            rng=None, temp=0.0)
                c["tokens"] = list(c["tokens"]) + extra
                c["text"] = llm.detokenize(c["tokens"]).decode(
                    "utf-8", errors="replace")
                c["pred"] = extract_boxed(c["text"])
                forced_c = True
            dt = time.time() - t0
            ok = is_correct(c["pred"], gold)
            print(f"  {pid} ctrl T={len(c['tokens']):>4} "
                  f"usage={c['cluster_usage']} pred={c['pred']!r} "
                  f"{'OK' if ok else '--'}{' [FORCED]' if forced_c else ''} "
                  f"({dt:.1f}s)", flush=True)
            record["controller"] = dict(pred=c["pred"], ok=ok,
                                         T=len(c["tokens"]),
                                         cluster_usage=c["cluster_usage"],
                                         text=c["text"], forced=forced_c)
            n_ctrl += int(ok)

        out_f.write(json.dumps(record, default=str) + "\n"); out_f.flush()
        n_done += 1

    out_f.close()
    print("\n=== SUMMARY ===")
    if args.mode in ("both", "baseline"):
        print(f"  baseline:   {n_base}/{n_done}")
    if args.mode in ("both", "controller"):
        print(f"  controller: {n_ctrl}/{n_done}")


if __name__ == "__main__":
    main()
