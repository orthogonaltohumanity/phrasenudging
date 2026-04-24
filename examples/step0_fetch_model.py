"""Step 0 — download DeepSeek-R1-Distill-Qwen-1.5B Q8_0 GGUF to ./models/.

Pure Python (urllib). About 1.8 GB; takes a few minutes on a normal link.

Usage (from repo root, inside your venv):
    python3 examples/step0_fetch_model.py
"""
import argparse
import os
import sys
import urllib.request
from pathlib import Path


URL  = ("https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/"
        "resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")
DEST = Path("models/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf")


def _progress(count, block_size, total_size):
    done = count * block_size
    if total_size > 0:
        pct = min(100.0, 100.0 * done / total_size)
        sys.stderr.write(f"\r  {done/1e9:.2f}/{total_size/1e9:.2f} GB  ({pct:5.1f}%)")
    else:
        sys.stderr.write(f"\r  {done/1e9:.2f} GB")
    sys.stderr.flush()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default=URL)
    ap.add_argument("--dest", default=str(DEST))
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already have {dest} ({dest.stat().st_size / 1e9:.2f} GB)")
        return
    print(f"downloading {args.url} -> {dest}")
    urllib.request.urlretrieve(args.url, dest, _progress)
    sys.stderr.write("\n")
    print(f"done ({dest.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
