# GEN-003: Does giving the 1M model the right evidence solve the binding failure?

Four exact frozen checkpoints generated freely from the original 65-character head. No new training. Compare two facts, a supplied correct fact at the same 39-character width and slot, an auxiliary prior-known “saw” filler, and a narrower one-fact prompt. The correct evidence was selected by the test, not by the model.

| Held exact responses | GEN-001 1 | GEN-001 2 | GEN-002 1 | GEN-002 2 |
|---|---:|---:|---:|---:|
| base | 608/2048 | 678/2048 | 784/2048 | 770/2048 |
| selected | 546/2048 | 586/2048 | 1366/2048 | 1340/2048 |
| selected-saw | 12/2048 | 24/2048 | 308/2048 | 324/2048 |
| narrow | 508/512 | 507/512 | 508/512 | 487/512 |
| decoy | 1438/2048 | 1440/2048 | 816/2048 | 770/2048 |
| decoy-saw | 2038/2048 | 2034/2048 | 1796/2048 | 1806/2048 |
| no-fact | 1964/2048 | 2000/2048 | 2044/2048 | 2036/2048 |
| no-fact-saw | 1952/2048 | 1860/2048 | 2048/2048 | 2048/2048 |
| unsupported | 420/1024 | 344/1024 | 392/1024 | 367/1024 |

| Exact on the entire root family | GEN-001 1 | GEN-001 2 | GEN-002 1 | GEN-002 2 |
|---|---:|---:|---:|---:|
| base | 0/128 | 0/128 | 0/128 | 0/128 |
| selected | 0/128 | 0/128 | 13/128 | 15/128 |
| selected-saw | 0/128 | 0/128 | 0/128 | 0/128 |
| narrow | 124/128 | 123/128 | 124/128 | 104/128 |

**Giving the right evidence helped the newer checkpoints, but did not solve the task.** For both GEN-002 checkpoints, same-width natural replacement improved exact answers by about 28 percentage points. The row-aligned nets were +332/+250 with the answer fact first/last for one checkpoint, and +354/+216 for the other (1,024 aligned rows per position). Both words were exact on only 526/1,024 and 480/1,024 matched target-value pairs, still below the 85% gate. GEN-001 did not improve overall. All four passed the narrow one-fact gate, none passed same-width selected competence.

**The surrounding words still mattered.** Keeping the same correct clause in the same physical slot but using the old “saw” negation-training cue in the other clause sharply reduced correct answers in all four checkpoints. And all four failed the natural decoy-only and unsupported-subject gates. For GEN-002, with only the wrong subject’s fact left, 1,232/2,048 and 1,278/2,048 answers incorrectly gave an item. Removing all evidence produced the correct “none.” in almost every case. This is a gap in using evidence for the right subject, not just emitting the response vocabulary. The matched-width effect isolates the tested replacements, not pure competition, width, or position alone.

The nine panels share 128 independent source families, not thousands of independent prompts. Exact outputs include the full four-character answer and period; the no-support target is the literal “none.”. No general epistemic abstention, natural-language retrieval, persistence, or architecture ranking follows.

The latest line gained +28.4 and +27.8 percentage points with the natural, same-width selected prompt; the older line lost 3.0 and 4.5 points. The narrow prompt achieved 95.1–99.2% exact across all four checkpoints. The next test should feed the actual verified registry result into a canonical short prompt and use the same native head across cold restarts. A stale-state control and an unrelated update that shares the old value can check whether answers follow the record identity rather than a familiar word or old hidden state. The present probe does not prove that mechanism or justify training a larger model as the next move.
