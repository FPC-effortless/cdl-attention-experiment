# Weakness repair cycle

This cycle tested improvements against the failure modes identified across COG-ATTN and WR. Research Lab OS registered 30 TrainLab jobs across five protocols. Veritas ecosystem reference verification independently passed. The new suites use their own exact state and data oracles. The audit checked all run receipts and 46,080 language predictions, 3,456 historical queries, and 4,800 durable journal entries. All three protocol seeds (321, 337, 353) are included. Memory probes ran separately on the previous WR-010 workload.

## 1 and 6. Grounding and credit assignment

Both systems start from random weights. The GRU pointer and Transformer pointer use the same candidate actions, state transitions and optimizer schedule. Soft compatibility marginalization sends gradients through the encoder and all three scoring factors. Entity names come from a known registry. The action family contains four supplied operators, and no gold action labels are used during training. This credits the learned grounding path. Discrete runtime routing and verification remain outside model training.

| Action accuracy across 1,152 examples per cohort | GRU pointer | Transformer pointer |
| --- | ---: | ---: |
| Familiar, before adaptation | 100.0% | 100.0% |
| Held syntax, before adaptation | 10.9% | 13.5% |
| Novel verb in support format, before adaptation | 0.0% | 20.5% |
| Novel verb in support format, after 96 support transitions | 99.0% | 95.8% |
| Novel verb and held syntax together, after support | 0.0% | 7.6% |

The adapted pointer's held-syntax effect accuracy was 29.3% versus 23.0% for the Transformer, outside the five-point matching gate. Exact accuracy was worse than effect accuracy because multiple wrong bindings can coincidentally yield the correct update. At confidence 0.80 the adapted pointer accepted 168 held-syntax examples and only 25 accepted actions matched the intended actions. Rejecting all unknown words removed every tested unsupported example and every held-syntax example. Thus the preregistered coverage and precision gate failed. The base models fell within the effect-parity tolerance on held syntax while achieving only 26.6% and 27.3% effect accuracy respectively.

The follow-up changed both the training template diversity and the composition of the support set, preserving the pointer architecture, training counts and held-out templates. With this curriculum, held-syntax action accuracy after adaptation rose from 14.8% to 33.3% and effect accuracy rose from 29.3% to 44.2%. Joint novel-verb and held-syntax action accuracy remained zero. The single-format comparison gives stronger adaptation to that same lexical format. The follow-up uses the already examined test split, and the template inventory is hand-authored. Unrestricted language and open-ended generation remain open. CUDA was unavailable, so the independent baseline has CPU measurements only. There is no high-competence accelerator comparison in this cycle.

## 2 and 5. Silent shifts and operator discovery

Of 192 shifts per channel, event-only checking recorded none. Under shared bias, periodic passive checks left the same aggregate number of incorrect predictions as event-only checking (101,226 of 196,608). One apparent detection under that shared bias was an accidental majority of agreeing noise. With independent anchors, auditing every 32 commands detected all 192 shifts, with maximum observed delay of 31 commands. It reduced wrong predictions to 2,858 using 18,432 anchor readings, or 9.375 readings per 100 commands. Three readings are taken per audit. The active intervention restores independence by construction. An unchanged or jointly biased evidence channel still cannot identify silent drift.

A finite program search composes addition and multiplication with two inputs and two constants over a 25-point grid. It promotes only when surviving programs agree on unobserved cells. The search admitted all 120 in-grammar operators correctly and declined all 120 random tables. Under shared bias, passive held-out validation admitted 47 wrong rules out of 120 trials. Independent validation rejected all 47 and produced no accepted rule in the biased condition. This implements a check against shared bias and combines primitive operations into candidate programs. Learning under a misspecified grammar, recovering from biased training labels, and discovering unseen primitives remain open.

## 3, 4 and 7. Concurrent commits, history and memory

Four actual client processes contend on a local SQLite WAL database. Every client reads authority before each write. An authority change is placed between lookup and the first commit in each client. Split check/commit produced 12 stale commits in 1,200 writes across three seeds. Validation and state updates inside one transaction produced zero and retried all 12. Journal IDs deduplicate repeated deliveries. A hard exit inside an open write transaction left committed counters unchanged. This repairs the local multi-process race. Cross-host replicas and failing network consensus remain outside this test.

For temporal memory, a checkpoint captures each object's latest visible row at the boundary, and retention keeps any event newer on either timeline. An independent full scan matched all 3,456 tested query answers. At 65,536 writes a bounded indexed snapshot occupied 593,920 bytes versus 2,269,184 bytes for the full index, a 73.8% reduction. The supported domain begins at both time cutoffs. For 192 queries, median warm query time was 16.65 ms on the bounded snapshot and 16.17 ms on the full index. Preparation plus build took about 23.8 ms and 60.7 ms respectively. These files are coherent static snapshots. The supplied query wrapper rejects an earlier valid or transaction cutoff. Its focused check covers a late correction at an old valid timestamp. Sustained online archival, concurrent compaction and adversarial multi-dimensional query performance remain open.

On three separately instrumented WR-010 50%-write runs, median main-process peak RSS was 269.1 MiB for WR and 284.1 MiB for the teacher-initialized matched decoder. End-of-run sampled PSS summed across runtime and verifier was 238.8 and 250.4 MiB respectively. Both processes used the original benchmark's common loaded Python stack. PSS is a snapshot rather than a simultaneous process-tree peak. Runtime/allocator overhead dominates these small parameter sets. The matched decoder's cached prefix tensor payload was 6,400 bytes. WR's retained command and answer caches spanned 19,009–21,154 and 26,105–28,557 bytes under Python object size accounting. In the separate drift run, 191 traced bindings occupied about 100 KB, including an estimated 57 KB for their dependency maps. These estimates can overlap through shared references. Full parameter payloads and process probes are in the raw results.

## Replay

Use Python 3.12, NumPy 2.3.5, PyTorch 2.15 CPU and the four Veritas ecosystem components at eec439492407b0aa26922c23b5ae40d95289f4eb. In a fresh directory with the supplied source and inputs run:

    veritas-ecosystem run "$PWD/ecosystem"
    python launch.py
    python launch_refresh.py
    python launch_curriculum.py
    python audit.py > audit.log
    PYTHONPATH=recipe python history_boundary_check.py
    bash memory/run.sh
    veritas-ecosystem verify "$PWD/ecosystem"

The 30 original frozen jobs, checkpoints, model predictions, operator outcomes, concurrency journals, receipts, protocol and audit are retained. The audits reuse the declared fixture generators but use separate execution and history comparison logic. Performance and memory readings vary with host load and package versions. The independent Transformer parity result is limited to its measured accuracy gates, and independent scientific review remains open.
