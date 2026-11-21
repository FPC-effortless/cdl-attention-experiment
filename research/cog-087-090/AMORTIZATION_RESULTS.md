# COG-ATTN-087–090: qualification closeout and Hanoi index follow-up
2026-11-21 UTC. This is state-conditioned puzzle skill selection, not the similarly numbered COG-LIFE experiments.

## Main qualification: completed and failed on cost
Replayed exact upstream c5285d386d6ec98b3d2ff7dc21f12e428e18a8f9 using Python 3.12 (`sha256sum -c SHA256SUMS && python replay.py`): 26 artifacts matched; frozen decisions, work counters, shortest costs, environment-shift records, and supplementary Hanoi decisions reproduced. Replay deliberately ignores archived times; below are the original host measurements.

| Method | Optimal | Primitive evaluations | Recorded summed time |
| --- | ---: | ---: | ---: |
| Exact A* | 144/144 | 17,587 | 62.77 ms |
| Unconditional options | 144/144 | 161,972 | 343.81 ms |
| Shuffled options | 144/144 | 161,326 | 338.97 ms |
| Domain-gated options | 144/144 | 17,587 | 64.61 ms |
| Learned state-conditioned selector | 144/144 | 17,579 | 81.01 ms |
| Efficient primitive A* | 144/144 | 17,587 | 24.71 ms |

The learned selector incurred 20,719 charged units versus 19,282 for Exact A*. The version checks were separate from selector timing: 36/36 stale plans rejected and optimally replanned, 524 stale records refused, 4,446 unaffected records retained. E did not invoke the maze mechanism after DEV gating, and masked tests used authoritative observations, not asynchronous race safety.

## R1 measured follow-up: narrow reuse gain, broad gate failed
432 distinct six-disc problems across 36 fixed 12-query sessions, six seed blocks, three initial graph families, two subgoal regimes. All 15,120 submitted paths (5 arms × 432 tasks × 7 timings) matched independent shortest lengths and replayed legally. Two native TrainLab stages passed receipt verification; Forge froze constructed fixture identities and Research Lab indexed both receipts. No model fit, private final test, or paid compute. Retained first Forge preflight failure: release rejected its permission declaration; no measurements ran in that attempt. The source constructs fiction-only fixture identities. Measurement froze the corrected recipe before the first measured run.

| 12-query regime/arm | Time: sum of session medians | Search and build transitions | Cache hits |
| --- | ---: | ---: | ---: |
| Repeated projection, cheap primitive | 653.22 ms | 539,014 | 0 |
| Repeated projection, cold three discs | 653.18 ms | 544,579 | 0 |
| Repeated projection, cached three discs | 623.00 ms | 509,539 | 180/216 |
| Repeated projection, cached five discs | 704.31 ms | 819,199 | 29/216 |
| Mixed projection, cheap primitive | 682.27 ms | 574,921 | 0 |
| Mixed projection, cold three discs | 682.00 ms | 574,631 | 0 |
| Mixed projection, cached three discs | 688.79 ms | 571,579 | 17/216 |
| Mixed projection, cached five discs | 757.61 ms | 908,015 | 0/216 |

Under engineered reuse, the three-disc table cut time by 4.6% and transitions by 5.5%. Paired six-block time ratio cheap/cached 95% interval was 1.011–1.076×. Under mixed goals, elapsed time was 1.0% higher; interval 0.969–1.011×. The broad criterion required a positive time interval and lower recorded work in both regimes, so it failed. The tested predecessor-style tower bound also did not improve wall time over the cheap control (683.98 ms shared, 718.74 ms mixed). Shared projected goals are deliberately supplied as a workload property; these timings do not estimate real reuse prevalence. A solution returning to a prior topology was not tested. At the halfway topology change all arms started rebuilding when applicable; this cache clears on observed topology change. Independent memory tracing (full sessions, not each prefix) measured Python allocations, not RSS.

Interpretation: exact reusable indexing is useful when queries predictably share structure, but it did not rescue the learned selector or demonstrate a general inference speedup. Do not promote it as default. Time includes table construction, lookup/invalidation, search and legal execution replay on preloaded synthetic queries. It excludes sensor acquisition, networking, dataset generation and offline evaluator work. The confidence intervals describe paired measurements on this host and six seed blocks.

## Resume
Use Python 3.12 and Veritas eec439492407b0aa26922c23b5ae40d95289f4eb. From the repository root:
```bash
export VERITAS=/path/to/veritas
export PYTHONPATH="$VERITAS/integrations/forge/src:$VERITAS/integrations/trainlab/src:$VERITAS/integrations/research_lab/src"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python research/cog-087-090/amortization_lab.py /absolute/new/run-directory
```
Run `python research/cog-087-090/amortization_audit.py /absolute/run-directory` to repeat the independent transition audit. The raw measurements are in the owner's Library evidence package; do not expect wall-clock parity on rerun. The archived receipt and aggregate are public-safe. The upstream five-arm run remains unchanged. No successor PR opened while #74 lacks review and #75 is still a draft dependent on it.
