# Data directory

## Included

- `raw/fever_sample.jsonl` — 5 FEVER claim-verification problems (for demo clustering)
- `raw/strategyqa_sample.jsonl` — 5 StrategyQA problems (for demo clustering)
- `problems_sample.jsonl` — 6 MATH problems (including L4_0016, the headline example)
- `problems_demo_pids.txt` — problem IDs to run on, one per line

## Not included (fetch separately)

The full FEVER and StrategyQA datasets, plus the full MATH benchmark, are
available from their original sources:

- **FEVER**: https://fever.ai/dataset/fever.html
- **StrategyQA**: https://github.com/eladsegal/strategyqa
- **MATH**: https://github.com/hendrycks/math

Our paper uses:
- 100 trajectories (60 FEVER + 40 StrategyQA) for clustering
- 500 MATH problems (100 per level L1-L5, hardest-first ordering) for evaluation

## Why we ship only 10 trajectories for the demo

Full 100-trajectory pipeline takes ~1 hour to build (~15 min generation,
~12 min all-pairs BC on 5995 pairs, ~5 min clustering + amps).

Ten-trajectory demo takes ~8 minutes end-to-end and demonstrates the full
pipeline, though with fewer clusters and noisier cluster boundaries.

## Reproducing the full 100-trajectory experiment

See `examples/full/README.md`.
