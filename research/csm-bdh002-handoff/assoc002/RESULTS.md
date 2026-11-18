# ASSOC-002 development results

Mean four-cue packet accuracy across seeds 11, 23 and 37, with 512 packets per seed and load:

| Memory | 29 associations | 64 associations | 256 associations |
| --- | ---: | ---: | ---: |
| Hebbian | 3.5% | 0.0% | 0.0% |
| Delta | 6.2% | 0.1% | 0.0% |
| Full nearest neighbor | 100% | 100% | 100% |
| Two-bank index, capped at 32 candidates | 31.0% | 39.4% | 4.3% |
| Delta with selective bank fallback | 24.7% | 7.7% | 0.4% |
| Random candidates with matched count | 0.4% | 0.8% | 0.1% |
| Incorrect binding control | 0.0% | 0.0% | 0.0% |

The two-bank index beat count-matched random retrieval but lost most packets at the highest load. A post-run candidate audit at 256 associations located the true key in the two selected banks for 4,591 of 6,144 cues (74.7%). Only 2,967 (48.3%) survived the first-32 cap. Both the bank choice and candidate truncation discarded useful keys. The hybrid requested fallback on 51.3% of evaluation cues and used a mean of 16.4 candidate dot products per cue versus 256 for full search. It stored 153,600 numeric bytes versus 147,456 for full search. Fewer comparisons gave no measured throughput result.

The supplied UNIFIED-001 numbers used a different, unavailable construction. These results apply to this panel's exact eight-bit payload decoder and noise level. A next comparison can replace insertion-order truncation with a second cheap similarity score under the same storage and query budget, measuring true-key survival before packet accuracy.

The [development archive](development.zip) contains per-seed data, the TrainLab receipt, Veritas evidence and Research Lab OS database. Use the environment in ../pipeline/README.md. Replay with a fresh directory:

```sh
OPENBLAS_NUM_THREADS=1 python lab.py /tmp/assoc002-new
python -m zipfile -e development.zip /tmp/assoc002-saved
diff /tmp/assoc002-new/runs/*/output/panel.csv /tmp/assoc002-saved/runs/*/output/panel.csv
OPENBLAS_NUM_THREADS=1 python coverage.py | diff coverage.csv -
```
