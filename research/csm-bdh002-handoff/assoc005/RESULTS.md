# ASSOC-005 — Recall errors can become valid incorrect writes

Three fresh stores each executed 256 balanced SET, COPY, SWAP and ROTATE3 commands at each noise level. States were independently reset. Every field was validated before a command could commit.

| Noise per coordinate | Method | Exact final states / 768 | Rejected packets | Incorrect committed states |
| --- | --- | ---: | ---: | ---: |
| 0 | Full search | 768 | 0 | 0 |
| 0 | Global 32-bit filter | 768 | 0 | 0 |
| 0.09 | Full search | 768 | 0 | 0 |
| 0.09 | Global 32-bit filter | 755 | 11 | 2 |
| 0.18 | Full search | 730 | 37 | 1 |
| 0.18 | Global 32-bit filter | 480 | 263 | 25 |

At noise 0.09, filtering reached 98.31% state accuracy; types detected eleven wrong packets while two passed the grammar and committed a wrong state. At noise 0.18, filtered state accuracy fell to 62.5%, and full search reached 95.05%. Exact packet and final-state counts happened to agree in this sample. Known original command IDs and independently calculated effects define correctness.

The paired median filtered/full latency ratio was 4.75, ranging from 3.29 to 5.65 across blocks. At noise 0.09 median per-command times were 36.95 µs filtered and 7.72 µs full. Both measured paths include retrieval, validation and execution, with store construction excluded. The full path uses a vectorized four-cue matrix multiplication; filtering iterates four candidate sorts and small dot products in Python. This is an implementation comparison at 256 keys and cannot establish intrinsic complexity or performance at larger stores. Single-threaded OpenBLAS was requested. The four-token grammar, semantic roles, binding and executor are supplied. SWAP and ROTATE3 bodies are inferred from prior demonstrations.

These observations establish two follow-up requirements for integration: measure the residual rate of valid incorrect commands across corruption levels, and benchmark any filtered implementation against optimized full search at the intended storage scale. Typed validation alone did not reject every incorrect packet. No adaptive rejection threshold was fitted on these outcomes.

## Reproduction and evidence

The retained archive includes the TrainLab engineering-diagnostic receipt, Veritas observation envelope, Research Lab OS database and all scored packets. A separate model subagent reviewed the design before execution. Scientific qualification remains open. The audit recomputes aggregates, per-operation errors and semantic-class confusions from the retained command IDs.

From this directory using the environment in ../pipeline/README.md:

```sh
OPENBLAS_NUM_THREADS=1 python lab.py /tmp/assoc005-new
diff scores.csv /tmp/assoc005-new/runs/*/output/scores.csv
python audit.py /tmp/assoc005-new/runs/*/output > /tmp/assoc005-new-audit.json
```

Timing samples vary between executions. [Recorded packet outcomes](scores.csv) · [Timing samples](timing.csv) · [Audit](audit.json) · [Pipeline bundle](development.zip)
