# Four-record development result

All three controllers produced the exact target state and cleared scratch on every one of the 256 goals. Total weighted costs were 4,205 for primitive only, 4,661 for SWAP, and 4,317 for ROTATE3. Choosing the cheapest arm after observing its costs gives 4,119, an 86-unit (2.05%) ceiling on savings over the best fixed controller on this panel. This bounds the opportunity on four records. The development run used a hindsight oracle and did not fit or test the proposed learned selector.

The [protocol](README.md) specifies the proposed held-out evaluation. The [development bundle](development.zip) includes the TrainLab receipt, per-goal scores, inferred programs, Veritas evidence record and Research Lab OS database. Use the Python environment documented in ../pipeline/README.md and run the following from this directory with fresh output paths.

```sh
python lab.py /tmp/skill006-fresh
python -m zipfile -e development.zip /tmp/skill006-recorded
diff /tmp/skill006-recorded/runs/*/output/panel.csv /tmp/skill006-fresh/runs/*/output/panel.csv
```

A distinct reviewer must assess the model, training budget, dataset and comparator rationale before the claim-bearing TrainLab run. The protocol uses supplied candidate bindings and exhaustive binding search, which limits conclusions about retrieval cost. The four-record ceiling makes the magnitude of any held-out benefit an important review question.

## Reviewed six-record evaluation

The scientific and pipeline reviews, fitted tree and TrainLab admission criteria were frozen at ec4a0e3 before holdout execution. TrainLab reported EXECUTED_CRITERIA_MET for the declared conditions. All three controllers executed each six-record assignment exactly, including scratch cleanup. The learned selector passed the strict action-cost gate by 205 units (0.017%) relative to the best fixed controller. This small gain concerns weighted action cost alone.

| Assignment family | Goals | Primitive | SWAP | ROTATE3 | Learned | Longest cycle | Hindsight |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Permutations | 720 | 26,815 | 27,310 | 26,815 | 26,610 | 27,070 | 26,610 |
| Repeated values | 45,936 | 1,192,950 | 1,287,750 | 1,234,950 | 1,192,950 | 1,329,150 | 1,181,310 |
| All | 46,656 | 1,219,765 | 1,315,060 | 1,261,765 | 1,219,560 | 1,356,220 | 1,207,920 |

All gains came from permutations. The selector used SWAP 15 times and ROTATE3 40 times, saving 45 and 160 units respectively. It used the primitive controller on all other goals. On repeated-value goals there remained 11,640 units of hindsight headroom. The selected route incurred 119,027 tree comparisons and 1,250 binding candidates, including 75 candidate executions. The primary score excludes those costs and feature extraction. No latency benefit was measured. A next experiment can distinguish which repeated-value goals benefit from a macro, using new goal sizes as the holdout.

The [research run](research-run.zip) retains the complete per-goal score table, both adequacy receipts, Veritas evidence and the local Research Lab OS database. The scientific claim is restricted to this constructed roster and metric. Use the pinned pipeline environment documented in ../pipeline/README.md. From this directory, run:

```sh
python -m zipfile -e research-run.zip /tmp/skill006-saved
python verify_research.py /tmp/skill006-saved
python campaign.py run /tmp/skill006-replay
python verify_research.py /tmp/skill006-replay
diff /tmp/skill006-saved/runs/*/output/scores.csv /tmp/skill006-replay/runs/*/output/scores.csv
```
