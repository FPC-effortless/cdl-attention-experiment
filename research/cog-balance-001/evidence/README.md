# COG Adaptive · whole-model pilot

All 12 architectures trained with three paired seeds at about 1.08M parameters. There were 96 updates, 49,152 real-text targets and 384 questions per fit. With this budget all 36 models scored zero both-correct on the 64 symbolic retrieval pairs, 64 generated English revision pairs and 64 held cancellation pairs. The same evaluation worlds were reused for each fit. An input-only grammar rule solved all 576 pairs across nine panels. A constant answer solved zero across the eight panels requiring a changed answer.

| Model | Dev bits/byte | Median cached decode step (ms) |
| --- | ---: | ---: |
| A · Native HH | 4.1026 | 1.775 |
| B · Unified | 3.9973 | 2.154 |
| C · Slow state | 4.0126 | 1.842 |
| D · Contextual memory | 4.0001 | 2.353 |
| E · Event permissions | 3.9959 | 2.304 |
| F · Competition | 4.0076 | 2.044 |
| G · Reliability | 4.0229 | 2.114 |
| H · Geometry | 4.0550 | 2.156 |
| GQA | 4.2265 | 0.856 |
| Gated DeltaNet reference | 3.6380 | 1.675 |
| GRU | 3.5891 | 0.591 |
| Mamba-2 reference | 3.4959 | 1.577 |

Table values are arithmetic means across three model seeds. Timing uses the sequential replay of saved checkpoints on a common 32-token prefill and 16 carried decode tokens, one CPU thread, three repeated decodes per checkpoint. Replay order was randomized. Local timing from training jobs is also preserved in the per-fit records. Package energy and DRAM traffic were unavailable. No hardware performance claim follows from comparisons between these CPU kernels and production serving kernels.

## Why the optional modules were idle

On nine held diagnosis panels, the chosen routing penalty of 0.01 drove the optional computation almost entirely to zero. Each model family had 201,312 layer-token decisions across seeds. F and G chose null everywhere. H selected only four geometric operations. Removing the selection penalty, with matched initialization, data and 96 updates, restored the following activity. Neither condition produced a correct pair on the main symbolic retrieval or English revision panel. These nine unpriced fits were follow-ups chosen after observing the initial collapse.

| Model | Selections at 0.01 | Selections at zero |
| --- | ---: | ---: |
| F | 0 | 124,085 |
| G | 0 | 99,137 |
| H | 4 | 191,591 |

E made 107,935 writes in its additional bank and also failed paired revision at this budget. On the labeled conflicting-claim tokens, the mean Brier loss of the G and H reliability estimates was 0.253 and 0.266. A constant 0.5 probability gets 0.250. The model has a reliable mechanism to skip writes, but learning to use a fact remains unresolved.

## Engineering evidence

All 24 architecture/scale builds passed causality, gradient and streaming checks. A 4,096-byte random-input stream produced finite state and logits on the first trained seed for every arm. All 11 recurrent arms had constant carried backing storage from 2,048 to 4,096 bytes. GQA grew by 4 MiB across the two streamed rows. The long-state files report both logical and backing bytes.

The per-fit records preserve data hashes, parameters, weights hashes, both budgets, held panels, and raw profiling. The files describe CPU measurements and the generated English questions. The 384-update, two-scale campaign has its own protocol and artifacts. [Larger campaign](https://github.com/FPC-effortless/cdl-attention-experiment/actions/runs/46180812123)
