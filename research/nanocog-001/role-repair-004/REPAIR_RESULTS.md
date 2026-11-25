# ROLE-REPAIR-004: more second-clause answers, but no qualified gate repair

The 552-parameter tap-content adapter and its literal-only control were
trained on the same known-weak frozen backbone. Neither met the unchanged
qualification screen on any of three adapter seeds. Extra tap content did
not consistently beat the matched literal arm on both-query correctness.

At 1,280 updates (native row: frozen, no adapter):

| Arm | Adapter seed | Worlds / 960 | All-eight / 120 | Both queries / 480 | First / 480 | Second / 480 |
|---|---:|---:|---:|---:|---:|---:|
| Native | n/a | 871 | 76 | 391 | 475 | 396 |
| Literal | 1471 | 895 | 79 | 416 | 440 | 455 |
| Literal | 1571 | 893 | 79 | 414 | 437 | 456 |
| Literal | 1671 | 893 | 80 | 414 | 436 | 457 |
| Tap content | 1471 | 895 | 83 | 416 | 437 | 458 |
| Tap content | 1571 | 893 | 83 | 413 | 433 | 460 |
| Tap content | 1671 | 896 | 83 | 416 | 434 | 462 |

The content arm gained **62 to 66** second-clause hits but lost
**38 to 42** first-clause hits compared with native. Relative to the
literal-only arm, the paired-query deltas were 0, -1, +2. Both missed
the frozen ≥912 worlds and ≥108 families per seed. Both learned somewhat
better average selection on this panel, but neither improved binding
reliably across every variant. The late two-color and full-vocabulary
world totals agree, so the remaining failure is not merely emitting
a token outside the offered value pair.

Separate retrospective diagnostic (not a learned policy): choose
native for every first-clause question and the corresponding content
adapter for second-clause questions using the known rendered role.
This gives **933–937/960** worlds but just **97–100/120** complete
families, still short of 108. Preserving existing first-clause behavior
alone is insufficient with these retained predictions. This does not
rule out a mechanism that improves both roles. No extra model run.

Both arms use the same optimizer schedule/initialization paired by seed
and the same 552 nominal scalars and rank-51 TRAIN feature design. Rank
does not imply the same hypothesis class. All original weights remained
frozen, and earlier logits stayed invariant in explicit parity checks.
Three fresh adapter seeds probe repeatability on one known backbone,
not robustness across unseen backbones. One seven-token synthetic
grammar with 120 fresh joint combinations disjoint from 712 prior
families; no language-model quality, retention, or speed claim.
Do not weaken the primary gate or promote the architecture.

Independent audit: 6 exact successful fits, 7,680 updates and 12
adapter-only checkpoints; 13 sealed native predictions (12,480 rows,
all 287 IDs) independently reconciled across all 429 score counts.
Native fit subprocess totals are execution records, not serving latency.
No MAIN, protected TEST, paid compute or full-model training.

- FIT stage ledger: `1bfd2de5842ced47984603cf8d6be24277808ca37bae1b1e6150b42475c86b2c`
- FIT seal: `2b231b0e918e561f04adf7e0375c37bc3caa72090d9fd954a5ad4702600ecfed`
- Prediction ledger: `317019d1cd792b6c62625deef81566a8297d0b90b0998160749797322ae98d37`
- Prediction seal: `0971d030ad7b74f7456e5eabdfa39b87dcd3641d90b6c2dfbf9e31e51366f1da`
- Score aggregate: `32954364574988fb6fbe21b11e490177d40f375014102eb0377e85401a544873`
- Independent recount: `b069fc5de4ee7aaf2ee46ae9864ad7c3b032afba6ece688ef946499230f5ed07`
- Retrospective role-oracle counts: `811414cdedf79dc83167a96dacf580dfad3deb829fb136bfce76d5fd8c35ce2c`
