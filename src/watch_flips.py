"""Poll both baseline and controller sweep files, emit one line per NEW flip.

Emits UP/DOWN events as they appear. Designed for use with Monitor.
"""
import json
import re
import sys
import time
from pathlib import Path

BOXED_RE = re.compile(r"\\boxed\{")


def extract_boxed(text):
    ms = list(BOXED_RE.finditer(text))
    if not ms:
        return None
    start = ms[-1].end(); depth = 1
    for i, c in enumerate(text[start:]):
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:start + i].strip()
    return None


def is_correct(pred, gold):
    if pred is None or gold is None:
        return False

    def norm(s):
        s = str(s).strip().strip("$")
        s = re.sub(r"\s+", "", s).rstrip(".,;")
        s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
        s = re.sub(r"\\frac\{([^}]*)\}([0-9a-zA-Z])", r"\\frac{\1}{\2}", s)
        s = re.sub(r"\\frac([0-9a-zA-Z])\{([^}]*)\}", r"\\frac{\1}{\2}", s)
        s = re.sub(r"\\frac([0-9a-zA-Z])([0-9a-zA-Z])", r"\\frac{\1}{\2}", s)
        return s
    if norm(pred) == norm(gold):
        return True

    def _num(s):
        if s is None: return None
        s = s.strip().strip("$").replace(",", "").replace(" ", "")
        try: return int(s)
        except: pass
        try: return float(s)
        except: return None
    pn = _num(pred); gn = _num(gold)
    if pn is not None and gn is not None:
        try:
            return abs(float(pn) - float(gn)) < 1e-6
        except:
            return False
    return False


def load_jsonl(path, up_to=None):
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rows.append(json.loads(line))
    except FileNotFoundError:
        pass
    return rows


def main():
    base_path = sys.argv[1]
    ctrl_path = sys.argv[2]
    target_n = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    interval = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    # Load golds
    pmap = {}
    with open("data/raw/problems.jsonl") as f:
        for line in f:
            r = json.loads(line)
            pmap[r["problem_id"]] = r

    seen_flips = set()
    while True:
        base_rows = load_jsonl(base_path)
        ctrl_rows = load_jsonl(ctrl_path)

        bm = {r["problem_id"]: (r.get("baseline") or {}).get("ok") for r in base_rows if "baseline" in r}
        # Controller can be in two forms: {controller: {ok}} or raw text
        cm = {}
        for r in ctrl_rows:
            pid = r["problem_id"]
            if "controller" in r and r["controller"] is not None and "ok" in r["controller"]:
                cm[pid] = r["controller"]["ok"]
            elif "text" in r:
                gold = pmap.get(pid, {}).get("answer")
                cm[pid] = is_correct(extract_boxed(r["text"]), gold)

        common = set(bm) & set(cm)
        for pid in sorted(common):
            b = bm[pid]; c = cm[pid]
            if b != c and pid not in seen_flips:
                seen_flips.add(pid)
                direction = "UP  " if (not b) and c else "DOWN"
                print(f"[{len(bm):>3}b/{len(cm):>3}c] {direction}  {pid}  gold={pmap.get(pid,{}).get('answer')!r}")
                sys.stdout.flush()
        # Stop when both reach target
        if len(bm) >= target_n and len(cm) >= target_n:
            # final summary
            both = sorted(set(bm) & set(cm))
            nb = sum(bm[p] for p in both); nc = sum(cm[p] for p in both)
            ups = sum(1 for p in both if not bm[p] and cm[p])
            dns = sum(1 for p in both if bm[p] and not cm[p])
            print(f"DONE  base={nb}/{len(both)}  ctrl={nc}/{len(both)}  UPs={ups}  DOWNs={dns}")
            sys.stdout.flush()
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
