# COG-ATTN-091 · Verified maze experience and local-control transfer

This branch **tests** learning from interaction across the Novice→Frontier maze ladder, but it is **not a full COG-ATTN neural agent**. The experiment imports the original COG-066 trace-fragment acquirer and trains a separate, transparent source-root-admitted preference policy. The existing restricted chess, Sudoku, sliding and Hanoi benchmark environments are unchanged. The local controller sees only current cell, goal and immediate legal actions at inference time.

[Protocol](PROTOCOL.md) · [Report](REPORT.md) · [Raw original results](results/maze-091.json) · [Fresh replication](results/maze-091-fresh.json) · [Budget sensitivity](results/maze-091-budget.json)

## Reproduce

Use Python 3.12 (standard library) from this directory. The puzzle cases are bundled in the preceding 087–090 evidence ZIP and not fed to the policy as labels. The generator/replay run from `extreme/` can also rebuild them.

```bash
cd research/cog-081-086/extreme
unzip -o artifacts/puzzle-087-090-cases-evidence.zip -d .
sha256sum -c CASE_SHA256SUMS
cd learning
python maze_transfer_091.py
python replicate_091.py
python budget_stress_091.py
python -m unittest test_maze_transfer_091 -v
python verify_stable_091.py
```

The original reserved panel uses two seeds, while the fresh replication uses four previously unevaluated seeds. Each changed maze is a second evaluation of an existing root (correlated). All reported timings are observed CPU policy wall time, and host-dependent. The stable digest audit excludes timings, preserving hard-failure vs success decisions and physical work.

**No model promotion:** training cost and inference latency have not been amortized away and the learned policy does not beat the strongest untrained control consistently. Read the limitations before claiming attention replacement or independently acquired reasoning.
