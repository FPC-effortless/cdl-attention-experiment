# Stage B — CDL distillation result

Run: GitHub Actions `33231945184`  
Artifact: `stage-b-results` (`9708841751`)  
Model: `HuggingFaceTB/SmolLM2-135M`  
Train cases: 800  
Held-out cases: 240  
Candidate memories per case: 6  
Student dimension: 64  
Epochs: 12

## Primary result

| Method | Top-1 | Top-3 | MRR | Teacher-choice agreement | Selected-answer NLL |
|---|---:|---:|---:|---:|---:|
| gzip conditional | 65.0% | 94.17% | 0.7949 | 62.08% | 6.1154 |
| SmolLM2 CDL teacher | 83.33% | 100% | 0.9160 | 100% | 5.2425 |
| direct-label Q/K student | **77.92%** | 100% | **0.8854** | 67.08% | **5.5113** |
| CDL-distilled Q/K student | 73.75% | 99.58% | 0.8635 | **72.92%** | 5.7686 |
| oracle relevant memory | 100% | 100% | 1.0 | — | 4.4473 |
| query only | — | — | — | — | 10.1531 |

The tested CDL-distillation recipe did **not** outperform direct relevance supervision. The direct-label student was +4.17 percentage points higher in Top-1 accuracy and produced lower answer NLL (5.5113 vs 5.7686).

The paired correctness table is:

- direct correct / distilled wrong: 39 cases
- distilled correct / direct wrong: 29 cases
- both correct: 148 cases
- both wrong: 24 cases

An exact paired McNemar/binomial test on the 68 discordant cases gives `p ≈ 0.275`, so the observed direct-vs-distilled Top-1 difference is **not statistically significant** at conventional thresholds. The defensible conclusion is therefore **no demonstrated benefit from this distillation formulation**, not that CDL supervision is conclusively worse.

## What did transfer

The distilled student agreed with the SmolLM2 CDL teacher's top choice more often than the direct-label student:

- direct-label Q/K: 67.08%
- CDL-distilled Q/K: **72.92%**

So distillation did transfer some of the teacher's ranking geometry. However, greater teacher agreement did not improve the ground-truth retrieval objective in this run.

## Relation-level Top-1

| Relation | CDL teacher | Direct Q/K | CDL-distilled Q/K | gzip |
|---|---:|---:|---:|---:|
| capital | 79.6% | 64.8% | **66.7%** | 51.9% |
| currency | 84.1% | 73.0% | **76.2%** | 52.4% |
| language | **91.7%** | 86.7% | 83.3% | 71.7% |
| leader | 77.8% | **85.7%** | 68.3% | 82.5% |

The largest regression for the distilled student is the `leader` relation. This is consistent with a teacher-imitation objective inheriting teacher preferences even where direct supervision learns a better task-specific decision boundary.

## Training behavior

The direct student's cross-entropy fell from ~1.77 to ~0.33 over 12 epochs. The distilled objective fell from ~1.78 to ~1.46 and plateaued much higher. This is not directly comparable as an absolute loss value because the objectives differ, but it indicates the 64-dimensional mean-pooled bilinear student did not closely fit the normalized soft CDL target distribution.

The implemented distillation target normalizes the six teacher scores within each candidate set and applies a unit-temperature softmax. That design choice is a plausible source of lost signal: it discards absolute score scale and may produce a target distribution too soft for the simple student.

## Current conclusion

Stage A remains positive evidence that conditional description length is a strong **runtime relevance score** on this controlled benchmark. Stage B does **not** yet establish that this signal can be distilled into a cheap Q/K router better than ordinary supervised routing.

Next falsification priorities are:

1. temperature sweep for the CDL target;
2. mixed objective `CE(label) + λ KL(CDL)` rather than pure teacher imitation;
3. rank/margin distillation instead of normalized softmax imitation;
4. stronger pair scorer or richer pooling to test whether student capacity is the bottleneck;
5. larger independent test set before treating small student differences as real.
