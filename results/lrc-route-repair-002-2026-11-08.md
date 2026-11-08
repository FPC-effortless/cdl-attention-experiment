# LRC-ROUTE-REPAIR-002: repaired identity keys and content-aware collisions

Completed 8 November 2026. Frozen fitting/evaluation source commit `6c3976e`.

The new sentence-level LRC-derived model learns identity-preserving fine keys,
quantizes read/write keys through one shared codebook, and resolves collisions
by scoring bounded candidate contents. It replaces the unobservable page/rank
policy and changes the training objective. Original LRC is preserved as a control.

All 12 fits closed before five fresh 1,024-example datasets were generated.
Across seeds 901–903, repair and supervised MemNet achieved **100% answer accuracy,
100% both-support retrieval, and 100% perfect quartets on every split**, including
partner switches. Every repaired story uses 16 distinct addresses, with four
person keys per era/relation; every payload/bridge quartet preserves the edited
record's address. Reversals preserve answers and mapped supports. Forced paired
collisions remain 100% correct with content resolution; a blind ordinal selector
falls to 34.28%, 40.04%, and 30.47%.

| Model | Mean withheld answers | Mean withheld both supports | Complete call | Complete fit |
| --- | ---: | ---: | ---: | ---: |
| Repair | 100.0000% | 100.0000% | 0.580 ms | 9.076 s |
| Original LRC, retrained | 40.3809% | 8.1380% | 0.922 ms | 38.711 s |
| Repair, shuffled support | 34.3424% | 1.9206% | — | — |
| Supervised MemNet | 100.0000% | 100.0000% | 0.211 ms | 9.128 s |

Withheld means cover two fresh cohorts in each seed. Timing means summarize
per-seed medians of 30 interleaved 32-call blocks, including encoding and index
construction. Fit times are sequential one-thread CPU runs. Repair's mean
12:1 inference-weighted cost score is +12.150 against MemNet's zero. Capability
is repaired; CPU cost still favors MemNet in every seed.

Repair has 12,900 parameters versus 18,879 for original LRC and 19,129 for MemNet.
It allows two code probes × two candidates, at most four scored records per hop,
versus the old cap of two. Normal test reads score two records per hop; all have
zero overflow and ties. Training uses dense support losses and gold-first/gold-final
teacher values, never gold inference inputs. The fixed 16-code table is not an
unbounded-memory solution. These sentence-task results do not establish a new
Tiny Shakespeare result.

All 29 structural tests pass. Independent audit reconciles 12 fits, 24,000 update
rows, all 90 cells and 92,160 predictions, and 1,440 checkpoint replays; no discrete
replay mismatch, max loss error 1.1920929e-7. All declared gates pass. The review
records a prefit stale-gradient clipping correction and the independent results.

- [Full report](https://chatgpt.com/library?file_id=libfile_fde3_f-1ae02ecdb72c8191aba53b5fabdca613_d6ad6c4f86c88191a55fed20a976a353&account_id=personal)
- [Checkpoints and audited evidence](https://chatgpt.com/library?file_id=libfile_fde3_f-d331dba61a2c8191826612e47a687d74_d6ad6c4f86c88191a55fed20a976a353&account_id=personal)
- [Implementation and protocol](../experiments/lrc_route_repair/PROTOCOL.md)
- [Independent review](../experiments/lrc_route_repair/REVIEW.md)

Evidence ZIP SHA-256: `06ad6c3d1e6647fa4912ec8631ca7367e5e4c775420f3573c1a024ea5f50daf6`.
The code and result record are prepared on a local review branch; no remote
publication or merge is claimed.
