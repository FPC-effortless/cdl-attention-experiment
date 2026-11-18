# ASSOC-004 — Redundancy restored recall, with substantial index work

The fixed factorial panel used three existing development seeds and loads of 29, 64 and 256. All approximate arms retained at most 32 keys for a final 64-dimensional comparison.

| Retrieval at 256 associations | Correct packets | Mean index visits per cue | Mean Hamming comparisons per cue | Persistent numeric bytes |
| --- | ---: | ---: | ---: | ---: |
| One table, 32-bit ranking | 29.95% | 64.46 | 64.46 | 150,600 |
| Two tables, 32-bit ranking | 77.60% | 128.86 | 112.37 | 152,720 |
| Four tables, 16-bit ranking | 84.05% | 257.63 | 175.25 | 156,448 |
| Four tables, 32-bit ranking | 97.85% | 257.63 | 175.25 | 156,960 |
| Global 32-bit signature scan | 98.76% | 256.00 | 256.00 | 148,480 |
| Full exact search | 100% | 256.00 | 0 | 147,456 |

All entries summarize 1,536 packets. Four-table 32-bit accuracy was 97.66%, 97.27% and 98.63% across seeds. Its pre-cap routing ceiling was 1,513/1,536 packets (98.50%); 1,503 packets survived ranking and were recovered exactly. Widening signatures recovered many targets displaced by distractors in the larger union. Global 32-bit ranking recovered 1,517 packets with 32 exact dots per cue. Its lack of an index reduces persistent state, while it scans more signatures. The exact full search uses 256 exact dots.

Four tables reduced Hamming evaluations to 175.25 distinct keys per cue, but walked 257.63 memberships before deduplication and performed eight bucket probes. The implementation sorts the union by ID and sorts Hamming scores on overflow, and allocates transient deduplication state. These operations and query signature extraction require timing before selecting an inference path. Numeric storage excludes Python object headers and temporary buffers, and includes an allowance for a flat index offset representation. The tested implementation is Python/NumPy.

The per-cue audit verifies every aggregate, exact target-based scoring, shared targets, the 32-dot cap, and nested union coverage. It also reports paired improvements and regressions relative to one-table 16-bit ranking. The nine arms share development memories and packets; their rows are dependent. The observed 95% aspiration was exceeded by four-table 32-bit retrieval on these seeds. Fresh-store, varied-noise evaluation and full latency accounting remain open. The global 32-bit signature scan is the simpler reference for that comparison.

## Replay

Use the Python environment in ../pipeline/README.md. Run from this directory with new destinations.

```sh
OPENBLAS_NUM_THREADS=1 python lab.py /tmp/assoc004-replay
python audit.py /tmp/assoc004-replay/runs/*/output > /tmp/assoc004-paired.json
diff scores.csv /tmp/assoc004-replay/runs/*/output/scores.csv
diff paired.json /tmp/assoc004-paired.json
```

[Per-seed scores](scores.csv) · [Paired audit](paired.json) · [TrainLab receipts, per-cue observations, Veritas envelope and Research Lab OS database](development.zip)
