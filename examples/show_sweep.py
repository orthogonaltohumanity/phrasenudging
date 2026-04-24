"""Render a sweep's JSONL into a human-readable report.

Consumes the output of step 5 (baseline + controller per problem) and writes
a Markdown file that's browsable at a glance. By default, prints to stdout
AND writes `data/sweep_demo.report.md` alongside the JSONL.

Usage (from repo root):
    python3 examples/show_sweep.py                                  # default paths
    python3 examples/show_sweep.py --jsonl data/sweep_demo.jsonl
    python3 examples/show_sweep.py --jsonl data/sweep_demo.jsonl --out report.md
    python3 examples/show_sweep.py --pid L3_0001                    # single-problem drill-down
"""
import argparse
import json
import sys
from pathlib import Path


def classify(b_ok: bool, c_ok: bool) -> str:
    if c_ok and b_ok:       return "BOTH_OK"
    if c_ok and not b_ok:   return "UP"
    if not c_ok and b_ok:   return "DOWN"
    return "BOTH_BAD"


def first_diff_char(a: str, b: str) -> int:
    i = 0
    m = min(len(a), len(b))
    while i < m and a[i] == b[i]:
        i += 1
    return i


def summarize_header(recs: list[dict]) -> str:
    cats = {"UP": 0, "DOWN": 0, "BOTH_OK": 0, "BOTH_BAD": 0}
    for r in recs:
        cats[classify(r["baseline"]["ok"], r["controller"]["ok"])] += 1
    ctrl_ok = sum(1 for r in recs if r["controller"]["ok"])
    base_ok = sum(1 for r in recs if r["baseline"]["ok"])
    ctrl_T = [r["controller"]["T"] for r in recs]
    base_T = [r["baseline"]["T"] for r in recs]

    lines = [
        f"# Sweep report — {len(recs)} records",
        "",
        f"- baseline correct: **{base_ok} / {len(recs)}**",
        f"- controller correct: **{ctrl_ok} / {len(recs)}**",
        f"- flips: UP **{cats['UP']}**, DOWN **{cats['DOWN']}**, "
        f"BOTH_OK **{cats['BOTH_OK']}**, BOTH_BAD **{cats['BOTH_BAD']}**",
        f"- net Δ: {cats['UP'] - cats['DOWN']:+d}",
        f"- avg T (baseline / controller): {sum(base_T)//max(1,len(base_T))}"
        f" / {sum(ctrl_T)//max(1,len(ctrl_T))}",
        "",
        "## Per-record summary",
        "",
        "| pid | gold | verdict | b.T | c.T | b.pred | c.pred |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        v = classify(r["baseline"]["ok"], r["controller"]["ok"])
        vmark = {"UP": "✅ UP", "DOWN": "❌ DOWN",
                 "BOTH_OK": "🟰 OK", "BOTH_BAD": "⚫ BAD"}[v]
        gold_short = str(r["gold"])
        if len(gold_short) > 42: gold_short = gold_short[:40] + "…"
        b_pred = str(r["baseline"].get("pred"))
        c_pred = str(r["controller"].get("pred"))
        if len(b_pred) > 30: b_pred = b_pred[:28] + "…"
        if len(c_pred) > 30: c_pred = c_pred[:28] + "…"
        # Escape pipes in table cells.
        for s in (gold_short, b_pred, c_pred):
            s = s.replace("|", "\\|")
        lines.append(
            f"| `{r['problem_id']}` | `{gold_short}` | {vmark} | "
            f"{r['baseline']['T']} | {r['controller']['T']} | "
            f"`{b_pred}` | `{c_pred}` |"
        )
    lines.append("")
    return "\n".join(lines)


def per_record_detail(r: dict, max_text_chars: int = 800) -> str:
    pid = r["problem_id"]
    gold = r["gold"]
    b = r["baseline"]; c = r["controller"]
    v = classify(b["ok"], c["ok"])
    div_char = first_diff_char(b["text"], c["text"])

    lines = [
        f"## {pid}  ·  {v}",
        "",
        f"**gold**: `{gold}`  ",
        f"**baseline** — T={b['T']}  ok={b['ok']}  pred=`{b['pred']}`",
        f"**controller** — T={c['T']}  ok={c['ok']}  pred=`{c['pred']}`  ",
        f"**diverge at char**: {div_char}",
        "",
    ]
    if "cluster_usage" in c:
        cu = c["cluster_usage"]
        total = sum(cu.values())
        lines.append("**cluster usage**: " +
                     ", ".join(f"c{k}:{v}" for k, v in cu.items()) +
                     f"  (total nudge tokens: {total})")
        lines.append("")

    # Common prefix + divergence window.
    prefix = b["text"][max(0, div_char - 60):div_char]
    b_win = b["text"][div_char:div_char + 200]
    c_win = c["text"][div_char:div_char + 200]
    lines += [
        "### Divergence",
        "",
        "**Common prefix** (last 60 chars):",
        "```",
        prefix,
        "```",
        "**Baseline continuation** (200 chars):",
        "```",
        b_win,
        "```",
        "**Controller continuation** (200 chars):",
        "```",
        c_win,
        "```",
        "",
    ]

    # Tails (to see how they end).
    b_tail = b["text"][-max_text_chars:]
    c_tail = c["text"][-max_text_chars:]
    lines += [
        "### Tails",
        "",
        f"**Baseline last {max_text_chars} chars**:",
        "```",
        b_tail,
        "```",
        f"**Controller last {max_text_chars} chars**:",
        "```",
        c_tail,
        "```",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jsonl", default="data/sweep_demo.jsonl")
    ap.add_argument("--out", default=None,
                    help="Markdown output path (default: alongside JSONL with "
                         ".report.md suffix). Pass '-' to print to stdout only.")
    ap.add_argument("--pid", default=None,
                    help="If set, show only this problem (drill-down mode).")
    ap.add_argument("--max-text-chars", type=int, default=800)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.jsonl)]
    if args.pid:
        recs = [r for r in recs if r["problem_id"] == args.pid]
        if not recs:
            sys.exit(f"no record matches --pid {args.pid}")

    parts = [summarize_header(recs), ""]
    for r in recs:
        parts.append(per_record_detail(r, max_text_chars=args.max_text_chars))
    output = "\n".join(parts)

    print(output)

    if args.out != "-":
        out_path = (Path(args.out) if args.out
                    else Path(args.jsonl).with_suffix(".report.md"))
        out_path.write_text(output)
        print(f"\n[wrote {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
