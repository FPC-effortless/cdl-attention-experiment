# Post-WR staged writes and causal Transformer comparison

On the original WR-010 operation streams, the 0.80 confidence gate committed four incorrect actions and Veritas scored 126 downstream reads below full reward. At 0.995 the runtime proposed each mutation privately, waited for a local verification service, discarded the proposal on disagreement and applied the verified write in order. Both model families obtained 11,470/11,470 full-reward reads. This required 6,962 service calls for 6,962 writes, versus 4,034 at the original gate. The strict gate discarded 1,021 incorrect proposals. Staging and a stricter verification rule were tested together; a blocking confirm-all WR control obtained the same accuracy (implemented using the same staged path with an unreachable confidence threshold).

The verifier service ran in a separate local process using real Unix socket request/reply messages with no injected sleep. On WR the mean observed round trip was 20.9 microseconds at the original gate and 20.8 at the strict gate. Across the twelve workload cells, summed median WR runtime increased from 280 to 433 milliseconds. Measured Transformer service round trips were 52.4 and 60.1 microseconds respectively. Process scheduling and socket overhead are included. Replies are truthful exact bindings looked up in a constructed table. Waiting is serial. These measurements cannot estimate human verification latency, WAN latency, verifier error or concurrent reader availability.

## Matched attention benchmark

The CPU comparator is a causal Transformer decoder block with one attention head, 80-wide embeddings, a 160-wide feedforward layer, residuals and a final summary token. It reuses actual prefix keys and values for the first two tokens, even across distinct commands. Both systems also use identical confidence-gated whole-command memoization, lazy read caching, executor and responder. Cached and full-sequence Transformer logits agreed within 4.8e-6 on the checked cases. One PyTorch CPU thread was used.

Two training runs from random initialization failed the validation criterion. The logit-MSE student collapsed, and the temperature-2 KL student achieved 55.50% held-pair action accuracy against its teacher's 93.33%. The successful latency control initialized the Transformer value path from the WR teacher, then fine-tuned all parameters for 300 steps. On a fresh 12,288-command validation slice teacher and student achieved identical pooled accuracies in all four cohorts, including 93.26% on held pairs and 48.54% on novel verbs. Both also matched downstream read accuracy and clarification count at each shared gate. Distillation used the teacher's original examples and frozen representations. This controls competence for implementation timing and leaves independently learned architecture parity unresolved.

| Write probability | Median Transformer time divided by WR time, 0.80 | Same ratio, 0.995 |
| ---: | ---: | ---: |
| 1% | 3.26 | 3.47 |
| 10% | 4.10 | 3.96 |
| 50% | 5.13 | 5.46 |
| 90% | 5.36 | 5.57 |

Timing starts with cold caches in a warm process. It includes tokenization, inference, service round trips, writes and reads. It excludes fixture generation, checkpoint loading and offline scoring. The comparison is specific to NumPy versus eager PyTorch at batch one and short command prefixes. Within a job the three repetitions randomize gate order. Frameworks run in separate processes and timing does not isolate architecture from implementation overhead. Timings for strict versus original gates also contain the cost of lost high-confidence semantic cache hits.

## Evidence and replay

Python 3.12, NumPy 2.3.5, torch 2.15.0+cpu, Veritas and its Forge, TrainLab and Research Lab OS components at eec439492407b0aa26922c23b5ae40d95289f4eb. Training happened locally; registered Lab OS jobs used frozen model files for evaluation. The original MSE and repaired KL validation slices are shared. The function-matched control used a fresh validation slice, while all behavioral workloads deliberately reuse WR-010 streams. The audit checks TrainLab receipts and independently replays predicted versus gold writes before scoring saved submissions in CompanyWorld. It reuses the trace generator. Same-host replay and receipt checks do not confer scientific qualification.

For reproduction, copy the scripts and original WR checkpoint/vocabulary files into a fresh directory with the shipped layout. Activate the environment and run:

    veritas-ecosystem run "$PWD/ecosystem"
    python train.py
    python launch.py
    python train_kl.py
    python train_matched.py
    python launch_matched.py
    python launch_confirmed.py
    python audit.py
    python summarize.py
    veritas-ecosystem verify "$PWD/ecosystem"

Each registered recipe records its precise input snapshot. The KL validation run is retained as a failed competence comparison. The staged private proposal is a single-cell write intent. Production transactions, durable undo and changing meanings remain outside this experiment.

## Confirmed-command reuse

In the post-hoc fixed-epoch follow-up, both model families verified each distinct exact command on first use, then reused that confirmation. Each made 1,166 requests across 6,962 writes, discarded 190 initial proposals, and retained all 11,470 full-reward reads. Summed median times were 101 ms for WR and 415 ms for the matched Transformer. The workload deliberately repeats commands, and the responder never changes its answer. This result measures reuse under stable meaning rather than robustness to semantic drift. Across the original, matched and follow-up jobs the final audit rescored 137,640 saved reads and recorded its extended results in a new immutable Lab OS evidence node.
