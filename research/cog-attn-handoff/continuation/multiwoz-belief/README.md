# Verified writes and provisional reads on MultiWOZ

A separate belief state cut wrong carry-forward reads from **1,267 to 665** on the same 7,372 official MultiWOZ 2.2 test turns used in PR #72. The two hotel slots were jointly correct on 6,560 turns. The original SQLite store made exactly the same 83 slot writes and returned the same verified state on every turn. This supports using provisional reads for dialogue context. It leaves consequential actions subject to the original admission gate. [Protocol](PROTOCOL.md) · [Milestone](https://github.com/FPC-effortless/cdl-attention-experiment/tree/cog-attn-001-060-milestone)

| Read policy | Joint state correct / 7,372 | Correct on 422 change turns | Wrong on 6,950 unchanged turns | Mean slot Brier score ↓ |
| --- | ---: | ---: | ---: | ---: |
| Verified state only | 5,740 | 57 | 1,267 | .3002 |
| Unrestricted argmax carry | 6,500 | 247 | 697 | .1419 |
| Development-selected hard provisional overlay | 6,500 | 247 | 697 | .1419 |
| Unmodified probabilistic filter | 6,537 | 250 | 663 | .1086 |
| Development-selected probabilistic filter | **6,560** | **275** | **665** | **.1081** |

Across 1,000 dialogues, the selected filter avoided 602 wrong unchanged turns against verified-only carry. The 95% paired dialogue-bootstrap interval was **469 to 740 fewer**. Against unrestricted carry, the estimated reduction was 32 turns with an interval from **36 more to 99 fewer**. The evidence for the improvement over strict carry is considerably stronger. The hard overlay selected cutoff zero on development and therefore reproduced unrestricted argmax reads. Development selection optimized total joint accuracy, explaining why the unmodified soft filter has two fewer stale turns. [Development sweep](selection.json) · [Per-dialogue results](results.json)

**Cost of acting on a belief.** The selected read disagreed with the verified joint state on 1,429 turns. It corrected 947 wrong verified reads, spoiled 127 correct ones, and left both readings wrong on the remaining 355. The overlay changed its displayed slot value 386 times. Of the 83 verified slot writes, 78 matched annotated changes. Verified update precision was 93.98% and recall was 16.28% (78/479). Area remained at 55/59 correct writes and 55/219 recall. Price remained at 23/24 and 23/260. There were no human replies or external evidence arrivals. The term *verified* denotes the milestone's high-confidence commit policy. It does not imply a human attested to those values. [Turn traces](traces.json) · [Independent write-stream audit](audit.py)

## How reads and writes differ

The frozen direct model predicts a distribution over hold, clear and named values for each slot. On a high-confidence update the runtime commits to the original SQLite store and resets the provisional distribution for that slot. On other turns it carries the previous distribution forward in proportion to the probability of hold, then adds the mass for each newly proposed value. The chosen development settings square the model probabilities and multiply the hold weight by 0.25 before normalizing. A differing estimate is flagged for the caller. Only the high-confidence path can submit a patch. The runtime resets between dialogues and rejects repeated or out-of-order turns. It has no mechanism to promote repeated provisional guesses into verified evidence. [Runtime](belief.py) · [Isolation regression](test_belief.py)

The protocol was committed before scoring. The policy settings were committed after development and before this experiment scored test. The underlying model and test cohort had already appeared in PR #72. This is a paired follow-up on two hotel slots, with original source-shard digests, a fixed model checkpoint and an unchanged language encoder. The outcomes concern read accuracy. Cross-domain transfer, live clarification, cross-host state visibility and the 99%/70% prospective action gate remain open. [Frozen protocol revision](https://github.com/FPC-effortless/cdl-attention-experiment/commit/4e9827d) · [Development lock](https://github.com/FPC-effortless/cdl-attention-experiment/commit/52ce165)

## Reproduce

Run from this directory in a checkout containing the sibling milestone code. Python 3.12, NumPy 2.3.5, SciPy 1.17.0, scikit-learn 1.8.0 and joblib 1.5.3 were used. The numeric audit only needs the standard library. The hosted script downloads the pinned public development/test shards, verifies their hashes, recomputes selection and compares the per-turn decisions of all five policies. The original annotation source and license are linked in the milestone. [MultiWOZ](https://github.com/budzianowski/multiwoz/tree/fe0c8e65cfcd8462bd33c86e35f21addc84ca82b) · [License](../cog-051-060/MULTIWOZ-LICENSE.txt)

```bash
sha256sum --quiet -c SHA256SUMS
python test_belief.py
python audit.py
python hosted.py
```

For offline replay, set MULTIWOZ_DIR to the local MultiWOZ_2.2 directory. Run study.py select into a separate directory and compare selection.json with the lock. Run study.py score with the committed lock. For the recorded Veritas path, set BELIEF_LAB_DIR to a fresh scratch directory, add the pinned Veritas research_lab and trainlab src directories to PYTHONPATH, and run lab.py. It registers the study in Research Lab OS, runs primary and clean-directory TrainLab scoring jobs and verifies that their result and trace hashes agree. [Lab receipts](lab-receipt.json) · [Veritas dependency](https://github.com/FPC-effortless/veritas/tree/eec439492407b0aa26922c23b5ae40d95289f4eb)
