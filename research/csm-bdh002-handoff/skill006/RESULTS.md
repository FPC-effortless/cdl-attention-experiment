# Four-record development result

All three controllers produced the exact target state and cleared scratch on every one of the 256 goals. Total weighted costs were 4,205 for primitive only, 4,661 for SWAP, and 4,317 for ROTATE3. Choosing the cheapest arm after observing its costs gives 4,119, an 86-unit (2.05%) ceiling on savings over the best fixed controller on this panel. This bounds the opportunity on four records. The development run used a hindsight oracle and did not fit or test the proposed learned selector.

The [protocol](README.md) specifies the proposed held-out evaluation. The [development bundle](development.zip) includes the TrainLab receipt, per-goal scores, inferred programs, Veritas evidence record and Research Lab OS database. Use the Python environment documented in ../pipeline/README.md and run the following from this directory with fresh output paths.

```sh
python lab.py /tmp/skill006-fresh
python -m zipfile -e development.zip /tmp/skill006-recorded
diff /tmp/skill006-recorded/runs/*/output/panel.csv /tmp/skill006-fresh/runs/*/output/panel.csv
```

A distinct reviewer must assess the model, training budget, dataset and comparator rationale before the claim-bearing TrainLab run. The protocol uses supplied candidate bindings and exhaustive binding search, which limits conclusions about retrieval cost. The four-record ceiling makes the magnitude of any held-out benefit an important review question.
