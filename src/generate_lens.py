"""Output-only lens generator (pure Python, no custom C++ binary).

Records the layer-27 output distribution at every decoded step, sparsified
via **coverage-based top-K**: for each token, keep the smallest prefix of
the descending-probability ordering whose cumulative mass reaches
``--coverage`` (default 99.9%). Narrow (low-entropy) positions keep ~1-3
entries; mid-reasoning positions keep several hundred. The result is an
$\\ell_1$-faithful sparse approximation with bounded truncation error.

This module exposes two entry points:

* `generate_one(llm, prompt_text, ...)`  — reusable function: takes an
  already-loaded `llama_cpp.Llama` and returns the lens pickle dict.
  Callers (e.g. `examples/step1_generate_lens.py`) can load the model
  ONCE and iterate over many prompts without reload overhead.

* `main()` — CLI for single-trajectory use; loads the model and calls
  `generate_one` once.

Output pickle format (schema v2):

    {
      "schema_version": "v2",
      "tokens":         int32[T_total],   # prompt + generated tokens
      "n_prompt":       int,               # how many leading tokens are prompt
      "n_vocab":        int,
      "coverage":       float,
      "indptr":         int32[T_gen + 1],  # CSR row pointers into idx/val
      "idx":            int32[nnz_total],  # per-step top-K vocab indices
      "val":            float32[nnz_total],# per-step masses
      "layer_idx":      int32[1] = [27],
    }
"""
import argparse
import pickle
from pathlib import Path

import numpy as np

from sparse_utils import coverage_topk, csr_append, csr_finalize


def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax. Input/output are (V,)."""
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def generate_one(llm, prompt_text: str, n_predict: int = 1024,
                 coverage: float = 0.999, temp: float = 0.0,
                 seed: int = 42) -> dict:
    """Generate one trajectory from an already-loaded Llama instance.

    Args:
        llm:         a ``llama_cpp.Llama`` object (loaded once by the caller).
        prompt_text: the full formatted prompt (including any <BOS>/<User>/<Assistant> tokens).
        n_predict:   max tokens to generate.
        coverage:    fraction of per-token probability mass to retain (0, 1].
        temp:        sampling temperature (0 = greedy/argmax).
        seed:        RNG seed used only when temp > 0.

    Returns:
        the lens pickle dict (not written to disk — caller persists).
    """
    from llama_cpp import llama_get_logits_ith

    V = llm.n_vocab()
    eos = llm.token_eos()
    ctx = llm._ctx.ctx

    # Tokenize the prompt. special=True treats control tokens like
    # <｜begin▁of▁sentence｜> as single tokens rather than characters.
    prompt_tokens = llm.tokenize(prompt_text.encode(), add_bos=False, special=True)
    n_prompt = len(prompt_tokens)

    llm.reset()
    llm.eval(prompt_tokens)

    # CSR-style growing buffers for the generated segment.
    indptr_list: list[int] = [0]
    idx_list:    list[np.ndarray] = []
    val_list:    list[np.ndarray] = []
    generated_tokens: list[int] = []

    rng = np.random.default_rng(seed)

    for _ in range(n_predict):
        # Read last-position logits.
        ptr = llama_get_logits_ith(ctx, -1)
        logits = np.ctypeslib.as_array(ptr, shape=(V,))
        P = softmax_stable(logits)

        # Coverage-based sparse top-K.
        idx, val = coverage_topk(P, coverage=coverage)
        csr_append(indptr_list, idx_list, val_list, idx, val)

        # Pick the next token.
        if temp <= 0.0:
            tok = int(np.argmax(P))
        else:
            logits_t = (logits.astype(np.float64) / temp)
            logits_t -= logits_t.max()
            P_t = np.exp(logits_t); P_t /= P_t.sum()
            tok = int(rng.choice(V, p=P_t))

        generated_tokens.append(tok)
        if tok == eos:
            break
        try:
            llm.eval([tok])
        except Exception:
            # e.g., context overflow; bail gracefully.
            break

    T_gen = len(generated_tokens)
    indptr, idx_arr, val_arr = csr_finalize(indptr_list, idx_list, val_list)

    all_tokens = np.asarray(list(prompt_tokens) + generated_tokens, dtype=np.int32)
    return dict(
        schema_version="v2",
        tokens=all_tokens,
        n_prompt=n_prompt,
        n_vocab=V,
        coverage=float(coverage),
        indptr=indptr,
        idx=idx_arr,
        val=val_arr,
        layer_idx=np.array([27], dtype=np.int32),
    )


def load_model(model_path: str, n_ctx: int, n_gpu_layers: int, seed: int):
    """Construct a Llama instance once. Returns the object."""
    from llama_cpp import Llama
    return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                 logits_all=False, verbose=False, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help="GGUF model path")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out", required=True, help="Output .pkl path")
    ap.add_argument("--n-predict", type=int, default=1024)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-gpu-layers", type=int, default=-1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coverage", type=float, default=0.999)
    args = ap.parse_args()

    llm = load_model(args.model, args.n_ctx, args.n_gpu_layers, args.seed)
    prompt = open(args.prompt_file).read()
    out = generate_one(llm, prompt, n_predict=args.n_predict,
                       coverage=args.coverage, temp=args.temp, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f, protocol=4)
    T_gen = out["indptr"].shape[0] - 1
    avg_k = (out["indptr"][-1] / T_gen) if T_gen > 0 else 0.0
    size_kb = Path(args.out).stat().st_size / 1e3
    print(f"wrote {args.out}  T_gen={T_gen}  avg_k={avg_k:.1f}  "
          f"size={size_kb:.1f} KB  (coverage={args.coverage})")


if __name__ == "__main__":
    main()
