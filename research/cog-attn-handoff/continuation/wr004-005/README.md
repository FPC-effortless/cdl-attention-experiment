# WR-004 and WR-005 raw-text grounding and bounded proposals

The exhaustive raw-text model passed all three compositional accuracy gates in all six seed/size runs. The uniform and destination-conditioned bounded learners passed at 8 entities and failed in all three seeds at 32. The training protocol and the post-diagnostic ablation are described in [protocol.md](protocol.md).

## Results

Accuracy pools examples across the three seeds. Gate results are evaluated per seed. Timings are means for the training loop and exclude data generation, evaluation and checkpoint writes. Executor checks count evaluated candidates. Shortlist scoring, sampling, and tensor operations also contribute to timing.

| Entities | Training | Familiar | Held pairs | Recombined | Checks/example | Seconds/seed | Seeds passing applicable gates |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | exhaustive | 100.00% | 100.00% | 100.00% | 128.00 | 0.52 | 3/3 |
| 8 | top8 | 60.68% | 57.78% | 58.59% | 8.00 | 0.35 | 0/3 |
| 8 | exploration | 100.00% | 100.00% | 100.00% | 8.70 | 0.39 | 3/3 |
| 8 | conditioned | 100.00% | 100.00% | 100.00% | 8.27 | 0.37 | 3/3 |
| 32 | exhaustive | 98.44% | 94.99% | 98.47% | 2048.00 | 4.96 | 3/3 |
| 32 | top8 | 14.91% | 15.01% | 13.18% | 8.00 | 0.64 | 0/3 |
| 32 | exploration | 45.48% | 42.29% | 45.70% | 13.65 | 0.88 | 0/3 |
| 32 | conditioned | 62.43% | 57.91% | 62.57% | 10.34 | 0.74 | 0/3 |

| Entities | Training | Pre-fallback support | Post-fallback support | Rescued among initial misses | Novel verb state accuracy | Novel alias state accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 8 | exhaustive | 100.00% | 100.00% | — | 62.14% | 7.78% |
| 8 | top8 | 69.74% | 69.74% | 0.00% | 42.90% | 12.04% |
| 8 | exploration | 97.83% | 99.35% | 70.14% | 61.52% | 8.24% |
| 8 | conditioned | 98.24% | 100.00% | 100.00% | 62.01% | 8.27% |
| 32 | exhaustive | 100.00% | 100.00% | — | 61.69% | 8.01% |
| 32 | top8 | 29.97% | 29.97% | 0.00% | 15.56% | 10.19% |
| 32 | exploration | 82.36% | 86.93% | 25.90% | 36.88% | 7.81% |
| 32 | conditioned | 92.69% | 100.00% | 99.98% | 47.88% | 7.29% |

At 32 entities, destination-conditioned training found compatible support on 76,799 of 76,800 training draws. Its changed-transition top-8 support recall was 93.19% on familiar commands, 91.63% on held pairs, and 94.35% on cue/verb recombinations, below the 95% gate in each pooled cohort. Exact action accuracies were 51.79%, 44.73%, and 50.46% respectively. Exhaustive training reached 97.92%, 93.55%, and 97.88% exact action accuracy on the same splits.

The intervention recovered nearly all missing training support without recovering generalization. The remaining gap is consistent with bias from discarding compatible actions outside the shortlist. This pilot does not isolate that cause from optimization effects. A direct follow-up is to sample residual actions even when the shortlist contains a compatible candidate, compare with an importance-weighted estimate of residual mass, and measure retained probability mass against exhaustive scoring. A separate changed/no-op breakdown would help locate the bias. These are synthetic learned lexical mappings within a fixed executor and entity set. The held-out verb and alias controls expose poor transfer to unseen lexical items. TAC/LRC integration remains open.

## Reproduction

Run with Python 3.12.14 and NumPy 2.3.5. Timings were collected on Linux x86-64 with an AMD EPYC 9V74 CPU and single-thread OpenBLAS. Run in a copy to retain the original timing and result files.

```bash
python run.py > evidence.log
python audit.py > audit.log
python summarize.py
```

The checkpoint audit regenerates evaluation data from the recorded seed, aliases, and vocabulary, and checks every reported evaluation count with explicit state mutation and dense top-k ranking. It reuses the tokenizer, data generator, and model forward pass. It checks training counter invariants and complement sampling constraints. The audit does not independently retrain the models. The JSON retains seed-level results and training support counts. Each NPZ stores source, destination and operation weights in that order. The manifest records source and output hashes.
