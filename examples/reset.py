"""Reset the pipeline's generated data so you can re-run steps 1-6 from scratch.

Wipes everything produced by the pipeline (lens pickles, pair-BC files,
cluster labels, phrase-amp-seqs, sweep outputs, breakdown logs) while
preserving:

* source code (src/, examples/, docs/)
* shipped config / problems (data/raw/, data/problems_*.{jsonl,txt})
* README / LICENSE / requirements.txt / .venv / .gitignore
* downloaded model weights (models/) — unless --include-models is passed,
  so you don't re-download 1.8 GB for a routine reset

Usage:
    python3 examples/reset.py              # dry-run-ish: prints the plan + confirms
    python3 examples/reset.py --yes        # no prompt
    python3 examples/reset.py --dry-run    # print plan only, delete nothing
    python3 examples/reset.py --include-models   # also wipe models/ (rare)

"""
import argparse
import shutil
import sys
from pathlib import Path


# Paths resolved relative to the repo root (one level above examples/).
REPO_ROOT = Path(__file__).resolve().parent.parent

# Everything in DATA_DIR except these names is eligible for deletion.
DATA_PRESERVE = {
    "raw",                          # shipped FEVER/STRAT/problem samples
    "problems_sample.jsonl",        # shipped MATH demo problem set (6 problems)
    "problems_demo_pids.txt",       # shipped demo pid list
    "problems.jsonl",               # shipped full MATH dataset (4331 problems)
    "test_500_hard_first.txt",      # shipped 500-pid hard-first sweep list
    "README.md",                    # data dir's own README
}

# Directories under the repo root that are entirely derived/disposable.
DERIVED_TOPLEVEL_DIRS = {
    "logs",
}


def enumerate_targets(include_models: bool) -> list[Path]:
    """Return a list of file/directory paths that will be removed."""
    targets: list[Path] = []

    # 1. Anything inside `data/` that isn't in DATA_PRESERVE.
    data_dir = REPO_ROOT / "data"
    if data_dir.is_dir():
        for entry in sorted(data_dir.iterdir()):
            if entry.name in DATA_PRESERVE:
                continue
            targets.append(entry)

    # 2. Top-level derived directories.
    for name in DERIVED_TOPLEVEL_DIRS:
        p = REPO_ROOT / name
        if p.exists():
            targets.append(p)

    # 3. Optional: models (usually skipped — we don't want to re-download).
    if include_models:
        models_dir = REPO_ROOT / "models"
        if models_dir.is_dir():
            targets.append(models_dir)

    # 4. Pycache cruft anywhere under src/ and examples/.
    for base in [REPO_ROOT / "src", REPO_ROOT / "examples"]:
        for pc in base.rglob("__pycache__"):
            targets.append(pc)

    return targets


def humansize(path: Path) -> str:
    """Report the on-disk size of a file or directory in a human-friendly form."""
    try:
        if path.is_file():
            b = path.stat().st_size
        else:
            b = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except Exception:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--yes", "-y", action="store_true",
                    help="skip the confirmation prompt")
    ap.add_argument("--dry-run", "-n", action="store_true",
                    help="print the plan but delete nothing")
    ap.add_argument("--include-models", action="store_true",
                    help="also wipe the models/ directory (forces 1.8 GB "
                         "re-download next time you run step 0)")
    args = ap.parse_args()

    targets = enumerate_targets(include_models=args.include_models)
    if not targets:
        print("nothing to clean — repo already at defaults")
        return

    print(f"would remove the following from {REPO_ROOT}:")
    total = 0
    for p in targets:
        size_str = humansize(p)
        kind = "dir " if p.is_dir() else "file"
        rel = p.relative_to(REPO_ROOT)
        print(f"  [{kind}] {rel}  ({size_str})")
        try:
            if p.is_file():
                total += p.stat().st_size
            else:
                total += sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
        except Exception:
            pass

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if total < 1024:
            total_str = f"{total:.1f} {unit}"; break
        total /= 1024
    else:
        total_str = f"{total:.1f} PB"

    print(f"\ntotal: {len(targets)} entries, {total_str}")

    if args.dry_run:
        print("(dry-run: nothing deleted)")
        return

    if not args.yes:
        reply = input("proceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("aborted; nothing deleted.")
            sys.exit(1)

    for p in targets:
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        except Exception as e:
            print(f"  [warn] could not remove {p}: {e}")
    print(f"removed {len(targets)} entries.")


if __name__ == "__main__":
    main()
