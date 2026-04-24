"""Step 1 — generate output-only lens pickles for a configurable set of
trajectories.

Loads the model ONCE, then iterates over the requested (problem_id, traj_idx)
pairs in a single process. The CUDA-built llama-cpp-python decodes at ~100-150
tk/s, so ~500-token trajectories take ~5s each after the ~5s one-time model
load.

Input modes (pick exactly one; listed in order of convenience):

1. Auto-pick N problems from the FEVER+STRAT pool in --raw-dir:
       python3 examples/step1_generate_lens.py --n 100
   Mix is proportional to each dataset's pool size (e.g. with 200 FEVER +
   50 STRAT available, N=100 yields ~80 FEVER + ~20 STRAT), deterministic
   given --seed. Combine with --n-trajs-per-id to produce multiple
   trajectories per problem.

2. Uniform CLI mode with explicit ids:
       --ids PID [PID ...]  --n-trajs-per-id N
   Example: 3 trajectories for each of 5 problems:
       python3 examples/step1_generate_lens.py \\
           --ids FEVER_004 FEVER_013 STRAT_022 STRAT_033 STRAT_041 \\
           --n-trajs-per-id 3 --temp 0.8

3. Manifest mode (per-problem overrides for temp/seed/n_predict):
       --manifest manifest.jsonl
   where each line is one entry, e.g.:
       {"problem_id": "FEVER_004", "n_trajs": 3, "temp": 0.8}
       {"problem_id": "FEVER_013", "n_trajs": 1}
       {"problem_id": "STRAT_022", "n_trajs": 5, "seed": 999}

4. Dataset-JSONL mode (every problem in a JSONL × N trajectories):
       --manifest-from-jsonl data/raw/fever_sample.jsonl --n-trajs-per-id 3

If none is specified, falls back to the 10-ID demo set (1 trajectory each).

Output naming: ``cd_{problem_id}_t{traj_idx}_lens.pkl``. Files are skipped if
they already exist unless you pass ``--force``.

Seeding
-------
For reproducibility, each (problem_id, traj_idx) pair gets a deterministic
seed = base_seed + 1000 * pid_ordinal + traj_idx, unless an entry explicitly
provides its own ``seed``. At temp=0 the seed has no effect (greedy); for
temp > 0 different seeds produce different samples.
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# Make sibling `src/` importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from generate_lens import generate_one, load_model


DEMO_IDS = [
    # 5 FEVER claims
    "FEVER_004", "FEVER_013", "FEVER_022", "FEVER_033", "FEVER_072",
    # 5 StrategyQA questions
    "STRAT_004", "STRAT_011", "STRAT_022", "STRAT_033", "STRAT_041",
]

HEAD = "<｜begin▁of▁sentence｜><｜User｜>"
TAIL = "<｜Assistant｜><think>\n"


def infer_raw_file(pid: str, raw_dir: Path) -> Path:
    """Pick the raw-dataset JSONL that should contain this problem_id."""
    prefix = pid.split("_")[0]
    candidates = {
        "FEVER": ["fever_sample.jsonl", "fever.jsonl", "fever_200.jsonl"],
        "STRAT": ["strategyqa_sample.jsonl", "strategyqa.jsonl", "strategyqa_50.jsonl"],
        "L1":    ["problems_sample.jsonl", "problems.jsonl"],
        "L2":    ["problems_sample.jsonl", "problems.jsonl"],
        "L3":    ["problems_sample.jsonl", "problems.jsonl"],
        "L4":    ["problems_sample.jsonl", "problems.jsonl"],
        "L5":    ["problems_sample.jsonl", "problems.jsonl"],
    }
    for name in candidates.get(prefix, []):
        p = raw_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"no raw dataset file for pid={pid!r} in {raw_dir} "
        f"(looked for {candidates.get(prefix, [])})")


def load_problem(pid: str, raw_dir: Path) -> dict:
    src_file = infer_raw_file(pid, raw_dir)
    for line in open(src_file):
        r = json.loads(line)
        if r["problem_id"] == pid:
            return r
    # Also check a sibling `problems.jsonl`-style file if present.
    for alt in raw_dir.glob("*.jsonl"):
        if alt == src_file:
            continue
        for line in open(alt):
            r = json.loads(line)
            if r["problem_id"] == pid:
                return r
    raise KeyError(f"{pid} not found in {src_file}")


def discover_pool(raw_dir: Path) -> dict[str, list[str]]:
    """Scan raw_dir for FEVER* / STRATEGYQA* / strategyqa* JSONL files and
    return ``{"FEVER": [pid, ...], "STRAT": [pid, ...]}`` from the *largest*
    file per dataset that we find."""
    def _pids_from_jsonl(path: Path) -> list[str]:
        out = []
        for line in open(path):
            try:
                r = json.loads(line)
                pid = r.get("problem_id")
                if pid: out.append(pid)
            except Exception:
                continue
        return out

    fever_files = sorted(list(raw_dir.glob("fever*.jsonl")) +
                         list(raw_dir.glob("FEVER*.jsonl")))
    strat_files = sorted(list(raw_dir.glob("strategyqa*.jsonl")) +
                         list(raw_dir.glob("STRAT*.jsonl")))
    pool: dict[str, list[str]] = {"FEVER": [], "STRAT": []}
    if fever_files:
        fever_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        pool["FEVER"] = _pids_from_jsonl(fever_files[0])
    if strat_files:
        strat_files.sort(key=lambda p: p.stat().st_size, reverse=True)
        pool["STRAT"] = _pids_from_jsonl(strat_files[0])
    return pool


def select_n(pool: dict[str, list[str]], n: int, seed: int) -> list[str]:
    """Deterministically choose N problem_ids from a FEVER+STRAT pool.

    Split N between the two datasets proportional to their pool sizes (so
    with 200 FEVER + 50 STRAT and N=100, you get ~80 FEVER + ~20 STRAT).
    Within each dataset the ids are sampled without replacement using a
    seeded RNG. The overall order interleaves the two samples so that
    downstream all-pairs BC sees a balanced mix as it iterates.
    """
    import random
    rng = random.Random(seed)
    fever = list(pool.get("FEVER", []))
    strat = list(pool.get("STRAT", []))
    total_pool = len(fever) + len(strat)
    if total_pool == 0:
        sys.exit("no FEVER or STRAT problems found in --raw-dir")
    if n > total_pool:
        print(f"  WARNING: requested --n {n} but pool has only "
              f"{total_pool} problems ({len(fever)} FEVER + {len(strat)} "
              f"STRAT). Capping at {total_pool}. To run more, put a "
              f"larger fever*.jsonl / strategyqa*.jsonl in --raw-dir.",
              flush=True)
        n = total_pool

    n_fever_target = int(round(n * (len(fever) / total_pool)))
    # Edge cases: honor availability of each dataset.
    n_fever = min(n_fever_target, len(fever))
    n_strat = min(n - n_fever, len(strat))
    # If one dataset ran out, redistribute.
    if n_fever + n_strat < n:
        n_fever = min(len(fever), n - n_strat)

    fever_pick = sorted(rng.sample(fever, n_fever)) if n_fever > 0 else []
    strat_pick = sorted(rng.sample(strat, n_strat)) if n_strat > 0 else []

    # Interleave deterministically.
    merged: list[str] = []
    i = j = 0
    while i < len(fever_pick) or j < len(strat_pick):
        if i < len(fever_pick):
            merged.append(fever_pick[i]); i += 1
        if j < len(strat_pick):
            merged.append(strat_pick[j]); j += 1
    print(f"  selected {len(fever_pick)} FEVER + {len(strat_pick)} STRAT = "
          f"{len(merged)} problems (pool: {len(fever)} FEVER, {len(strat)} STRAT)",
          flush=True)
    return merged


def build_manifest(args) -> list[dict]:
    """Return a list of entry dicts with per-entry (pid, n_trajs, optional
    temp/seed/n_predict overrides). Exactly one of the input modes must be
    used; if none is, fall back to DEMO_IDS with n_trajs=1."""
    n_sources = sum(bool(x) for x in
                    (args.manifest, args.manifest_from_jsonl, args.ids,
                     args.n is not None))
    if n_sources > 1:
        sys.exit("--manifest, --manifest-from-jsonl, --ids, and --n are "
                 "mutually exclusive; pick one")

    if args.manifest:
        entries = []
        for line in open(args.manifest):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            r = json.loads(line)
            if "problem_id" not in r:
                sys.exit(f"manifest line missing 'problem_id': {line!r}")
            r.setdefault("n_trajs", 1)
            entries.append(r)
        return entries

    if args.manifest_from_jsonl:
        entries = []
        for line in open(args.manifest_from_jsonl):
            r = json.loads(line)
            if "problem_id" in r:
                entries.append({"problem_id": r["problem_id"],
                                "n_trajs": args.n_trajs_per_id})
        return entries

    if args.n is not None:
        if args.n < 1:
            sys.exit("--n must be ≥ 1")
        pool = discover_pool(Path(args.raw_dir))
        ids = select_n(pool, args.n, seed=args.seed)
        return [{"problem_id": pid, "n_trajs": args.n_trajs_per_id}
                for pid in ids]

    ids = args.ids if args.ids else DEMO_IDS
    return [{"problem_id": pid, "n_trajs": args.n_trajs_per_id} for pid in ids]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model",
                    default="models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
    ap.add_argument("--raw-dir", default="data/raw",
                    help="Directory holding problem-source JSONLs.")
    ap.add_argument("--out-dir", default="data/lens_demo",
                    help="Where to write lens pickles.")

    # Input modes (mutually exclusive). Listed in order of precedence.
    ap.add_argument("--manifest", default=None,
                    help="JSONL with one entry per problem. Each entry may "
                         "include n_trajs, temp, seed, n_predict overrides.")
    ap.add_argument("--manifest-from-jsonl", default=None,
                    help="Build a manifest from a raw-problem JSONL using "
                         "--n-trajs-per-id for every entry.")
    ap.add_argument("--ids", nargs="+", default=None,
                    help="Explicit problem_id list (fallback: DEMO_IDS).")
    ap.add_argument("--n", type=int, default=None,
                    help="Auto-select N problems from the FEVER+STRAT pool "
                         "found in --raw-dir. Mix is proportional to each "
                         "dataset's size. Deterministic given --seed.")
    ap.add_argument("--n-trajs-per-id", type=int, default=1,
                    help="Trajectories per problem (applies to --ids, --n, "
                         "and --manifest-from-jsonl modes).")

    # Global defaults (manifest entries may override temp/seed/n_predict).
    ap.add_argument("--coverage", type=float, default=0.999)
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="Sampling temperature (0 = greedy). For n_trajs > 1 "
                         "you almost certainly want temp > 0 — otherwise all "
                         "trajectories are identical.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Base seed; per-trajectory seed = base + 1000*pid_ordinal + traj_idx")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the output file already exists.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(args.raw_dir)

    entries = build_manifest(args)
    total_trajs = sum(e.get("n_trajs", 1) for e in entries)
    print(f"manifest: {len(entries)} problems, {total_trajs} total trajectories",
          flush=True)

    # Warn if the user is asking for multiple trajectories per problem at temp=0.
    global_temp = args.temp
    if global_temp <= 0.0 and any(e.get("n_trajs", 1) > 1 and
                                   e.get("temp", global_temp) <= 0.0
                                   for e in entries):
        print("WARNING: n_trajs > 1 with temp <= 0 produces identical "
              "trajectories (greedy). Set --temp > 0 or specify per-entry "
              "temp in the manifest to get diverse samples.", flush=True)

    # Build the full plan so we can skip-existing / count work.
    plan: list[dict] = []
    for pid_ordinal, entry in enumerate(entries):
        pid = entry["problem_id"]
        n_trajs = int(entry.get("n_trajs", 1))
        for traj_idx in range(n_trajs):
            out_path = out_dir / f"cd_{pid}_t{traj_idx}_lens.pkl"
            if out_path.exists() and not args.force:
                plan.append({"pid": pid, "traj_idx": traj_idx,
                             "out_path": out_path, "skip": True})
                continue
            plan.append({
                "pid":       pid,
                "traj_idx":  traj_idx,
                "out_path":  out_path,
                "temp":      float(entry.get("temp", global_temp)),
                "n_predict": int(entry.get("n_predict", args.n_predict)),
                "coverage":  float(entry.get("coverage", args.coverage)),
                "seed":      int(entry.get("seed",
                                           args.seed + 1000 * pid_ordinal + traj_idx)),
                "skip":      False,
            })

    to_run = [p for p in plan if not p["skip"]]
    n_skipped = len(plan) - len(to_run)
    if n_skipped:
        print(f"skipping {n_skipped} trajectories (outputs exist; use --force to regenerate)")
    if not to_run:
        print("nothing to do")
        return

    print(f"loading model {args.model} ...", flush=True)
    t0 = time.time()
    llm = load_model(args.model, args.n_ctx, args.n_gpu_layers, args.seed)
    print(f"  model loaded in {time.time() - t0:.1f}s", flush=True)

    # Small cache so we only parse the same raw JSONL once per trajectory-
    # producing run.
    record_cache: dict[str, dict] = {}

    total_tokens = 0
    total_wall = 0.0
    for i, p in enumerate(to_run):
        pid = p["pid"]
        if pid not in record_cache:
            record_cache[pid] = load_problem(pid, raw_dir)
        prompt = HEAD + record_cache[pid]["problem"] + TAIL

        t0 = time.time()
        out = generate_one(llm, prompt,
                           n_predict=p["n_predict"],
                           coverage=p["coverage"],
                           temp=p["temp"],
                           seed=p["seed"])
        dt = time.time() - t0
        T_gen = out["indptr"].shape[0] - 1
        avg_k = (out["indptr"][-1] / T_gen) if T_gen > 0 else 0.0
        with open(p["out_path"], "wb") as f:
            pickle.dump(out, f, protocol=4)
        size_kb = p["out_path"].stat().st_size / 1e3
        total_tokens += T_gen
        total_wall += dt
        rate = T_gen / max(dt, 1e-9)
        print(f"[{i+1}/{len(to_run)}] {pid}  t={p['traj_idx']}  "
              f"seed={p['seed']}  temp={p['temp']}  "
              f"T_gen={T_gen}  avg_k={avg_k:.0f}  "
              f"{dt:.1f}s ({rate:.0f} tk/s)  "
              f"size={size_kb:.1f} KB", flush=True)

    overall_rate = total_tokens / max(total_wall, 1e-9)
    print(f"\ngenerated {len(to_run)} trajectories, "
          f"{total_tokens} tokens, {total_wall:.1f}s  "
          f"({overall_rate:.0f} tk/s average)")
    print(f"{len(sorted(out_dir.glob('cd_*_lens.pkl')))} total lens files in {out_dir}")


if __name__ == "__main__":
    main()
