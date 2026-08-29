# CASM v0.1 controlled result

Primary GitHub Actions run: `33234911110`  
Artifact: `casm-v01-results` (`9709681650`)  
Head: `43232ed47f373bfc6351db9e9a1c99bb7b69eb19`  
Training: 800 steps per variant, 8-byte-chunks worth of packed sequence context (`seq_len=193`, `chunk_size=24`), answer bytes weighted 8x, hard-example probability ramped to 0.5.

## Stream-level held-out metrics

| Variant | Easy LM NLL | Easy answer NLL | Hard LM NLL | Hard answer NLL | Hard token accuracy |
|---|---:|---:|---:|---:|---:|
| local-only | 0.6433 | 1.9264 | 0.8656 | 2.1209 | 68.55% |
| Q/K compressed memory | 0.6356 | 1.8873 | 0.8643 | 2.0846 | 68.97% |
| **compression-attention memory** | **0.6353** | **1.8556** | **0.8478** | **2.0563** | **69.37%** |

Unlike CASM v0, both memory variants no longer generalize worse than the local-only control. Compression attention has the best hard stream loss and answer NLL.

## Heterogeneous hard task evaluation

480 examples: 80 per task, seed `20260902 + 100000`.

| Task | local-only answer NLL | Q/K memory | compression attention |
|---|---:|---:|---:|
| associative recall | 2.3287 | **2.3219** | 2.3250 |
| state tracking | 0.8046 | 0.7993 | **0.7853** |
| arithmetic | **2.2260** | 2.2434 | 2.2388 |
| rule induction | 2.0960 | 2.0847 | **2.0751** |
| graph reachability | 0.5252 | 0.4949 | **0.2348** |
| reverse/copy | 2.1933 | **2.1368** | 2.1495 |
| **overall** | 1.6956 | 1.6802 | **1.6348** |

Compression attention's largest gain is graph reachability, but its answer NLL is also lower than local-only on state tracking, rule induction, reverse/copy and associative recall. Arithmetic is a regression.

## Paired analysis

The same 480 hard examples were rescored from all three checkpoints.

### Compression vs local-only

- mean paired answer-NLL delta: **-0.06084 nats**
- compression lower-NLL fraction: **61.88%**
- bootstrap 95% interval for mean delta: **[-0.07417, -0.04780]**
- mean byte-accuracy delta: **+0.41 percentage points**

### Compression vs Q/K memory

- mean paired answer-NLL delta: **-0.04541 nats**
- compression lower-NLL fraction: **58.75%**
- bootstrap 95% interval: **[-0.05796, -0.03329]**
- mean byte-accuracy delta: **+0.19 percentage points**

### Q/K memory vs local-only

- mean paired answer-NLL delta: **-0.01544 nats**
- bootstrap 95% interval: **[-0.02249, -0.00791]**

These intervals are bootstrap uncertainty over the synthetic held-out examples; they do not replace replication across independent training seeds.

## Causal inference-time ablations

Using the trained compression checkpoint on the same 480 hard examples:

### Disable learned compression-score correction only

Keep the same trained model and memory values, but route using normalized Q/K only (`use_compression_score=False`).

- full compression minus score-disabled answer NLL: **-0.00857 nats**
- full compression has lower NLL on 52.92% of examples
- bootstrap 95% interval: **[-0.01296, -0.00408]**
- byte-accuracy delta: **-0.55 percentage points**

Thus the learned compression score has a small causal NLL benefit, but does not improve top-1 byte accuracy in this run.

### Zero retrieved memory vector

Keep the trained compression checkpoint and its post-routing compute, but replace the retrieved memory vector with zero.

- full compression minus retrieval-zeroed answer NLL: **-1.0614 nats**
- full compression lower NLL on **100%** of the evaluated examples
- bootstrap 95% interval: **[-1.1024, -1.0199]**
- byte-accuracy delta: **+5.64 percentage points**

This establishes that the trained v0.1 model is causally using retrieved memory rather than merely benefiting from parameterization or auxiliary-loss regularization.

### Compression-trained model with score disabled vs separately trained Q/K model

- answer-NLL delta: **-0.03684 nats**
- bootstrap 95% interval: **[-0.05041, -0.02397]**

Therefore most of the compression-vs-Q/K advantage in this run comes from compression-shaped training of the shared representations/memory pathway; the runtime compression-score correction contributes a smaller additional effect.

## Router diagnostics at step 800

Compression model:

- router entropy: 1.1865
- mean learned compression gain: 0.0729
- cross-candidate gain standard deviation: 0.00226
- memory residual gate mean: 0.3027

Q/K model:

- router entropy: 1.7458
- memory residual gate mean: 0.2837

The learned gain no longer collapses exactly to zero as in v0, but candidate separation is still small. This remains a target for improvement rather than evidence that the predictor is already well calibrated.

## Current conclusion

CASM v0.1 passes the first architectural falsifier that v0 failed:

1. compressed memory causally improves held-out prediction over an otherwise fair local-only control;
2. compression-shaped training improves the memory model over direct Q/K routing on paired answer NLL;
3. the learned compression correction itself has a smaller but measurable causal NLL effect.

The evidence is still **single-seed synthetic evidence**. The next gate is multi-seed replication before increasing model size or adding new memory mechanisms.
