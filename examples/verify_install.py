"""Quick sanity check that imports, model weights, and paths are set up.

Run with:  python3 examples/verify_install.py
"""
import importlib
import sys
from pathlib import Path

errors = []
def check(ok, msg):
    print(f"  [{'OK' if ok else 'FAIL'}]  {msg}")
    if not ok: errors.append(msg)

print("== python + deps ==")
check(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} >= 3.10")
for mod in ["numpy", "scipy", "sklearn", "llama_cpp"]:
    try:
        importlib.import_module(mod)
        check(True, f"import {mod}")
    except Exception as e:
        check(False, f"import {mod}  ({e})")

print("== repo files ==")
root = Path(__file__).resolve().parent.parent
for rel in ["src/generate_lens.py", "src/allpairs_bc.py",
            "src/spectral_cluster.py", "src/build_phrase_amps.py",
            "src/run_controller.py", "src/watch_flips.py",
            "src/breakdown_one.py",
            "data/problems_sample.jsonl",
            "data/raw/fever_sample.jsonl",
            "data/raw/strategyqa_sample.jsonl"]:
    p = root / rel
    check(p.exists(), f"{rel}  ({p.stat().st_size if p.exists() else '—'} bytes)")

print("== model ==")
model = root / "models" / "DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
check(model.exists(), f"model weights at {model.relative_to(root)}"
      + ("" if model.exists() else "  (run examples/step0_fetch_model.py)"))

print("== llama.cpp GPU ==")
# Sanity-check: the CPU-only build of llama-cpp-python will silently ignore
# n_gpu_layers and decode at ~10 tk/s; that's the single most common cause
# of "why is it so slow" on this pipeline.
try:
    import llama_cpp
    lib = getattr(llama_cpp, "llama_cpp", None) or llama_cpp
    has_cuda_fn = hasattr(lib, "llama_supports_gpu_offload")
    if has_cuda_fn:
        try:
            supported = bool(lib.llama_supports_gpu_offload())
        except Exception:
            supported = None
        if supported is True:
            check(True, "llama-cpp-python reports GPU offload is supported")
        elif supported is False:
            check(False,
                  "llama-cpp-python does NOT support GPU offload — this is a "
                  "CPU-only build. Reinstall with:\n"
                  '        CMAKE_ARGS="-DGGML_CUDA=on" pip install --upgrade '
                  "--force-reinstall --no-cache-dir llama-cpp-python")
        else:
            check(True, "llama-cpp-python loaded (could not probe GPU flag)")
    else:
        check(True, "llama-cpp-python loaded (older version; cannot probe GPU)")
except Exception as e:
    check(False, f"failed to import llama_cpp ({e})")

if errors:
    print(f"\n{len(errors)} issue(s) to fix before running the demo.")
    sys.exit(1)
print("\nall good → proceed with examples/step1_generate_lens.py")
