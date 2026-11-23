# GEN-004: A verified record changes the model’s answer after restart

The verified, versioned registry supplied one current fact and the original question to the unchanged 993,574-parameter model. The model generated freely from all 65 native characters. No model was trained or given an answer selector. The registry, alias check, and typed write rules are engineered, not learned.

**Three of four frozen checkpoints passed all preregistered gates.** Every checkpoint passed the transaction and cold-restart checks. GEN-002 seed 1 missed both the usefulness and mechanism thresholds because 86/96 (89.6%) opposing-current pairs were exact, below the 90% minimum. That checkpoint still passed the transaction checks and exceeded 95% exact on ordinary reads.

| Exact outcome | GEN-001 1 | GEN-001 2 | GEN-002 1 | GEN-002 2 |
|---|---:|---:|---:|---:|
| Ordinary authorized reads | 3,244/3,264 | 3,264/3,264 | 3,200/3,264 | 3,164/3,264 |
| Fresh sidecar, both different current answers after restart at fixed old state | 96/96 | 96/96 | 86/96 | 92/96 |
| Stale state, question only: both outputs match the old fact | 95/96 | 96/96 | 94/96 | 94/96 |
| Fresh state, question only: exact current answer | 0/192 | 0/192 | 0/192 | 0/192 |
| Stale state plus current evidence: exact current answer | 91/192 | 106/192 | 79/192 | 100/192 |
| All authorized reads exact for the whole session | 188/192 | 192/192 | 176/192 | 160/192 |
| Corrected owner, exact old answer then exact new answer after cold | 190/192 | 192/192 | 180/192 | 184/192 |
| Other owner sharing the same old word, exact unchanged answer | 190/192 | 192/192 | 192/192 | 186/192 |
| Corrected owner, exact unchanged answer across later update to other owner | 192/192 | 192/192 | 182/192 | 188/192 |
| Other scope, exact unchanged answer for first owner | 190/192 | 192/192 | 188/192 | 188/192 |
| Other scope, exact unchanged answer for second owner | 190/192 | 192/192 | 192/192 | 186/192 |

Every checkpoint executed 384 authorized commits. Each of three distinct worker processes per checkpoint handled one chronological phase; the two real restart boundaries reopened 192 session records each, for 384 verified per-session cold reopens, not 384 process launches. Canceled, stale, unauthorized, unknown, and ambiguous writes left both scoped boards unchanged. Each checkpoint also refused 384 unsupported reads without decoding; that is a policy result, not a model-generated “none.”. Each checkpoint was tested on the same 48 independent source families, crossed into 192 sessions.

**The answer follows accepted durable evidence when it is read in a fresh sidecar.** The viable stale control emitted the old fact; the same question alone did not produce either correct new answer. Feeding the current fact into the already stale neural state did not reliably override it (79–106 of 192 exact current answers across checkpoints). The successful path uses isolation and an explicit verified record, not learned correction of stale hidden state. The cold children reopened SQLite without old transcript or hidden state. They still read a typed record, which is rendered into a 25-character prompt. This does not establish whole-history extraction, arbitrary aliases, production authorization, resilience to concurrent edits or alias remaps, or a serving-speed advantage.
