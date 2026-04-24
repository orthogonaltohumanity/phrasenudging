# Phrase Nudging

Phrase-level inference-time steering for LLMs. Mine W-token "phrases"
from recorded reasoning trajectories, cluster them on the amplitude
hypersphere using per-token Bhattacharyya similarity, then SLERP the
model's output distribution toward scheduled cluster targets at decode
time.

Runs end-to-end on a single GPU against a GGUF model via
`llama-cpp-python`.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the math + code
walkthrough.

## Pipeline

| step | script                                  | produces                                  |
|------|-----------------------------------------|-------------------------------------------|
| 0    | `examples/step0_download_model.py`      | `models/*.gguf`                           |
| 1    | `examples/step1_generate_lens.py`       | `data/lens_*/*.pkl` (per-traj CSR lens)   |
| 2    | `examples/step2_allpairs_bc.py`         | `pair_i_j.pkl` (θ per `(t,k)`), summary   |
| 3    | `examples/step3_spectral_cluster.py`    | phrase labels + sample-phrase print       |
| 4    | `examples/step4_build_phrase_amps.py`   | per-cluster phrase-amp libraries          |
| 5    | `examples/step5_run_controller.py`      | problem-set sweep JSONL + markdown report |
| 6    | `examples/step6_seed_sweep.py`          | N-seed sweep on one problem               |
| 7    | `examples/step7_token_probe.py`         | single-trajectory probe (single / plan)   |
| 8    | `examples/step8_markov_nudge.py`        | Markov-process probe + sweep              |

## Quick start

```bash
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall llama-cpp-python  # for CUDA

python3 examples/step0_download_model.py
python3 examples/step1_generate_lens.py --preset demo --n 100
python3 examples/step2_allpairs_bc.py   --preset demo
python3 examples/step3_spectral_cluster.py --sigma 0.7 --k 15
python3 examples/step4_build_phrase_amps.py
```

Then either:

```bash
# Plan-based controller over a problem set
python3 examples/step5_run_controller.py --preset full \
    --schedule "c4:50+35,c1:50+35,c11:50+35,c7:50+35" \
    --alpha 0.01 --force-commit

# Markov-process controller (strictly more general)
python3 examples/step8_markov_nudge.py --pid L4_0020 --n-tokens 800 \
    --clusters "c4,c1,c11,c7" \
    --transition "0.5,0.25,0.05,0.05,0.15; \
                  0.05,0.5,0.25,0.05,0.15; \
                  0.05,0.05,0.5,0.25,0.15; \
                  0.05,0.05,0.1,0.5,0.3; \
                  0.1,0.1,0.1,0.05,0.65" \
    --alpha 0.02 --compare-greedy --force-commit
```

## Schedule grammar (steps 5 / 6 / 7)

```
c<cid>:<nudge>+<free>[@<alpha>]
```

e.g. `"c4:50+35,c1:50+35@0.02,c11:50+35,c7:50+35@0.05"`. Each block
is `<nudge>` tokens of SLERP-nudged sampling toward that cluster, then
`<free>` greedy/free-sampled tokens. `+<free>` and `@<alpha>` are
optional.

## Force-commit (all steps)

On benchmarks that require `\boxed{answer}` but where the model may
derive-but-not-emit, `--force-commit` injects a forcing prefix after the
natural decode and greedy-decodes to the closing `}`. `--force-at`
picks the cutoff position (`end` | `post-schedule` | `<int>` | `<pct>%`),
and records carry a `forced: bool` field so natural-vs-forced rates can
be analyzed separately.

## Temperature knobs

- `--baseline-temp` — baseline decode (default 0 = argmax greedy)
- `--nudge-temp` — logits temperature *before* softmax → SLERP
- `--free-temp` — free-window decoding (default 0 = argmax greedy)

## Datasets Used
- **FEVER** — Thorne et al., NAACL 2018. [arXiv:1803.05355](https://arxiv.org/abs/1803.05355) · [HF](https://huggingface.co/datasets/fever/fever)            
- **StrategyQA** — Geva et al., TACL 2021. [arXiv:2101.02235](https://arxiv.org/abs/2101.02235) · [project](https://allenai.github.io/strategyqa/)
- **MATH** — Hendrycks et al., NeurIPS 2021. [arXiv:2103.03874](https://arxiv.org/abs/2103.03874) ·[HF](https://huggingface.co/datasets/hendrycks/competition_math)                                                                                           
- **ZebraLogic** — Lin et al., ICML 2025. [arXiv:2502.01100](https://arxiv.org/abs/2502.01100) · [HF](https://huggingface.co/datasets/WildEval/ZebraLogic)

## AI Use Disclosure

I use AI as a research tool — for finding papers, implementing my
designs in code, and as a thinking aid (I like to think out loud, and
there's usually no one to talk to at 3 AM when I'm in the middle of a
work session). I take full responsibility for the code as shared on GitHub.
