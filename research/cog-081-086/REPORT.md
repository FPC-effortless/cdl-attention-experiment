# COG-ATTN-081 local puzzle pilot — measured controls

**Status:** puzzle benchmark/environment validation passed; COG-ATTN model not evaluated.

## Summary

- Instances: **396**, unique inputs: **326** (70 intentional repeated canonical Hanoi states).
- Valid reference paths: **396/396**.
- Simple domain-specific heuristic solutions: **258/396**.
- Repeated exact-task cache hits: **396/396**; the novel-cache hit metric is zero by deliberate per-task isolation and is not a learning result.
- 36/36 maze shifts rejected stale paths and 36/36 were recovered by a fresh search.

## Per-domain controls

| Condition | n | Reference valid | Shallow heuristic success | Median oracle moves | Median oracle expansions | Median exact search ms | Median verified replay ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| chess_Q | 36 | 100.0% | 0.0% | 1.0 | 27.0 | 0.0566 | 0.0310 |
| chess_R | 36 | 100.0% | 0.0% | 1.0 | 19.0 | 0.0308 | 0.0248 |
| hanoi_3 | 36 | 100.0% | 100.0% | 7.0 | 25.0 | 0.0513 | 0.0131 |
| hanoi_5 | 36 | 100.0% | 100.0% | 31.0 | 233.0 | 0.4730 | 0.0480 |
| maze_11 | 36 | 100.0% | 58.3% | 16.0 | 53.0 | 0.0447 | 0.0301 |
| maze_7 | 36 | 100.0% | 94.4% | 8.0 | 16.0 | 0.0175 | 0.0149 |
| maze_shift | 36 | 100.0% | 83.3% | 16.0 | 55.0 | 0.0435 | 0.0296 |
| slide_14 | 36 | 100.0% | 50.0% | 14.0 | 55.5 | 0.2302 | 0.0174 |
| slide_8 | 36 | 100.0% | 72.2% | 8.0 | 11.0 | 0.0560 | 0.0115 |
| sudoku_4 | 36 | 100.0% | 100.0% | 9.0 | 10.0 | 0.0793 | 0.0275 |
| sudoku_9 | 36 | 100.0% | 58.3% | 49.0 | 50.0 | 2.6918 | 0.1581 |

**Interpretation:** Reference valid means the coded solver and validator agreed, not that the learning architecture solved the task. Shallow controls differ by domain; their aggregate rate is not a rigorous common baseline. Search and replay timing cover only the corresponding local operations, not model end-to-end inference or memory training. No performance promotion is justified.

## Split integrity

Seeds: primary (8101, 8107, 8113); replication (9101, 9107, 9113). Non-Hanoi puzzles were deduplicated across both splits. Exactly two Hanoi input states intentionally repeat across both splits (36 trials each). Their 70 redundant instances must be excluded from independent generalization statistical tests.

## Prior architecture connection

Existing COG-ATTN-071–080 issue-workflow planner, provisional belief, signed ledger, reusable plan store and causal probe policy were examined on GitHub PR #74. They were **not** loaded into this puzzle runner because no environment-to-action adapter exists. Direct application of the issue classifier to a chess grid would falsely advertise model competence. The next falsifiable test is learning puzzle actions and verified procedures from observation/action/outcome data.

## Provenance and limitations

Only standard library Python puzzle controls ran. Source code, complete per-instance results, non-oracle observations, protocol and regression tests are packaged in the archive. Full chess, sparse visual perception, external benchmarks, learning, and complete COG cost accounting remain untested.
