# Research decision: COG-ATTN-097d / 097d2

**No promotion. Both training experiments failed their transfer gates.**

- **097d:** The actual network was trained on 297 raw bodies with 104 updates/seed. Training loss decreased, but top-1 held-out score was **24/192** across three seeds reusing 64 distinct author queries, compared to **69/192** for the best lexical control. One exact duplicate was removed between source-module splits after an audit; the corrected results were rerun. No self-supervised retrieval advantage.
- **097d2:** Allowed 591 training retrieval attempts with one-bit consequence feedback, achieving 196 successes. Updated 152 times per seed before freezing. On **161 completely new authored document queries**, outcomes **35/483** after feedback, **44/483** before, **36/483** with shuffled feedback, **216/483** for char TF-IDF. Same 161 cases repeated across seeds; don't count 483 as independent test subjects. No outcome-learning transfer advantage.
- **Safety/integrity:** Missing-evidence case remains unimplemented: no abstain action. No real issue was modified; no authoritative COG execution tool was connected. No synthetic success is portrayed as authentic dialogue. The binary feedback world is an experimental proxy created from docstring source association, not a real user.
- **Cost:** Mean 097d2 outcome training **9.36 s/seed**, plus ~**0.50 s** acquisition-policy work, 591 attempts, plus all indexing and inference. Neural test scorer is faster than char TF-IDF but dramatically worse in accuracy. Never compare raw latency as a speed win at unmatched competence.

## Scientific interpretation

The training objective rewards predicting another nearby portion of a document, which is not equivalent to knowing which evidence answers a user's different phrasing. The few positive online feedback events are concentrated in examples already discoverable through lexical retrieval, so optimizing them does not force abstract transfer. The proposed fix is a new grounded, *agent-generated* prediction/update objective with causal environment consequences; it must be tested on genuinely unseen behavior without directly supplying test answers or prebuilt semantic slots. That proposal is a hypothesis, not a measured capability.

## Reproducibility limits

The live environment's CPython 3.13.5 standard-library sources serve as actual data. The bundle retains hashes, models, optimizer update counts, per-case ranks, reward-event hashes and exact code. External replication should verify hashes and perform independent reruns; the current audit is same-machine replay, not independent-investigator replication. A full learned COG controller remains blocked because existing schema-bound components have no raw perception/action training interface.
