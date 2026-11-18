# ASSOC-003 development result

At load 256 the same two-bank route contained the target on 4,591 of 6,144 cues. Insertion-order truncation kept 2,967. Hamming ranking kept 4,578, recovering all but 13 routed keys. Every surviving true key won the exact comparison. Direct packet counts on 1,536 matched packets were 66 correct for insertion order, 456 for Hamming, and 460 with all four true keys inside the probed banks. At loads 29 and 64 both arms returned identical aggregate counts, with zero overflow-related Hamming comparisons. The insertion arm matched ASSOC-002 in every cell.

Both arms used 32 full 64-dimensional dot products per cue at load 256. The ranked arm also used 64.46 packed Hamming evaluations per cue on average and 512 extra persistent signature bytes per store. This Python implementation also extracts query signatures and sorts candidates on overflow. No elapsed-time advantage was tested. The three development stores have fixed insertion orders and a single noise level.

The next routing experiment can compare disjoint hash tables under a common full-dot cap. Measure union reachability and four-cue packet intersections directly; include duplicate elimination, candidate scans, storage and latency. The 95% packet target requires empirical validation.

Use the environment in ../pipeline/README.md. Run from this directory with a fresh output directory:

```sh
OPENBLAS_NUM_THREADS=1 python lab.py /tmp/assoc003-new
diff scores.csv /tmp/assoc003-new/runs/*/output/panel.csv
```

[Per-seed counts](scores.csv) · [TrainLab, Research Lab OS and Veritas evidence bundle](development.zip)
