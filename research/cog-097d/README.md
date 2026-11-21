# COG-ATTN-097d — Train, freeze, then evaluate without supplying answers

**Completed, negative scientific finding (21 Nov 2026).** This package executes two *actual trainable, attention-free retrieval-subsystem* experiments and publishes their losses, binary feedback traces, model checkpoints, frozen test predictions, exact source hashes and independent replay. It does **not** execute the complete COG-ATTN controller: the existing 097b `Archive` expects supplied `alias` and `field` keys and cannot ingest unrestricted prose. Never describe this study as successful full-attention replacement, human-level learning, or a real-dialogue experiment.

## Why this differs from 097c

097c used *untrained* BM25/TF-IDF/LSA controls. Here a PyTorch encoder with an embedding-bag word/character-subword layer and MLP **really trains** and is frozen before the held-out queries are scored. No gold query-to-document pairs, entity IDs, relation labels, candidate list containing the correct response, pretrained semantic network, hidden test association or reference solver is delivered to its parameters.

### 097d — Raw-sequence self-supervision

- **297** naturally authored Python documentation bodies from a fixed collection of source modules for training. The model sees only *two disjoint consecutive windows from each body*. The prediction signal comes from raw contiguity, not the later query-answer linkage.
- 8 training epochs, **104 optimizer updates** per seed. Train loss fell from ~2.9 to ~0.5 across seeds. Freeze weights before evaluation.
- The original train/test source-module split included one exactly duplicated docstring (`inspect.getdoc` / `pydoc._getdoc`); an explicit cross-split body/query deduplication was added and all seeds rerun. **64** unique held-out test passages remained. This repair occurred after the first pilot results, so 097d alone is exploratory rather than a fully sealed prospective evaluation.
- Top-one results over three seeds (the same 64 distinct questions reused across three fits; **not 192 independent questions**): Word TF-IDF **69/192**, character TF-IDF **60/192**, untrained neural model **26/192**, true-contiguity-trained neural model **24/192**, mismatched-pair training **19/192**. No success gate passed.

### 097d2 — Consequences without supplying the answer

- A separate protocol (`PROTOCOL_FEEDBACK.md`) fixed **34 fresh documentation source modules** before evaluating them. The fresh test contained **161 distinct queries and passages**; none appeared during the earlier evaluation or training.
- On the 297 training requests, an **explicitly disclosed character TF-IDF acquisition policy** nominated up to three passages by searching all raw passages. Each attempted action received only a binary successful/unsuccessful signal from a simulated evaluator. **591 attempted actions**, **196 success outcomes**, and **101 requests unresolved** per seed. The evaluator never returned the correct answer after a failure.
- Eight epochs and **152 reward-based updates** per seed; test then ran with frozen weights and **no feedback**. The lexical acquisition policy is an important engineered scaffold and its computation/feedback count must be charged. The feedback signal was synthesized from the author's original docstring association, **not a human interaction**.
- On the same 161 fresh queries, repeated across three model seeds: pre-feedback neural **44/483 (9.1%)**; outcome-trained neural **35/483 (7.2%)**; shuffled-reward control **36/483 (7.5%)**; word TF-IDF **171/483 (35.4%)**; character TF-IDF **216/483 (44.7%)**. Feedback learning **did not transfer**.
- The current readout has **no unknown/abstain action**. It must select a stored passage even when none is relevant; the 23 absent-query controls expose this structural failure. The missing-question count is not evidence of trained uncertainty calibration.

## Inference and economic limits

Outcome-based training alone cost **6.8–10.7 seconds per seed** in the recorded Python CPU runs, plus ~0.3–0.7 seconds of lexical acquisition policy building/actions, and 591 feedback interactions. On 161 test queries, average neural scoring inference cost was approximately **12.5 ms total**, versus **17.9 ms** for character TF-IDF, with model index costs ~77.5 ms and ~170.7 ms respectively. These are different-quality comparisons and **do not establish a valid inference speed victory**; the neural method performed far worse.

This is a single-CPU local diagnostic. Performance and source hashes depend on Python and installed versions. The raw data are publicly authored Python standard-library docstrings; they were not written as a retrieval benchmark, and retrieving the passage associated with a summary is a narrow proxy for grounding.

## Reproduce

Python 3.13.5; see `requirements.txt`. This package includes the vendored 097c corpus extraction source to make the source code self-contained. The authors' underlying Python stdlib sources are loaded from the local Python installation; match module SHA-256 hashes in `result-*.json` and `feedback-*.json` for a same-input replay. No binary corpus snapshots are included; exact data reproduction requires the pinned interpreter sources. The GitHub draft publishes code, protocols and compact per-query evidence, **but not the 48 MB of frozen weights or full raw feedback traces**; those are included in the separately downloadable complete archive. The GitHub `reproduce.sh` script regenerates them. Public source-hash checking is provided by `audit_compact.py`.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python experiment.py --seed 1701 --epochs 8 --out result-1701.json --checkpoint model-1701.pt
python feedback.py --seed 1701 --out feedback-1701.json --weights-dir .
python audit.py --replay
python -m pytest -q test_experiment.py test_feedback.py
sha256sum -c SHA256SUMS
```

The one-command source checkout route is `bash reproduce.sh`: it checks exact source hashes, trains and scores all three seeds, and replays checkpoints. Repeat for seeds 1707 and 1711 when running individual commands. `audit.py --replay` reloads the **actual saved frozen model weights** and independently rescored fresh queries; the audit was completed successfully. After regeneration, expect model-file bytes and timing receipts to vary by host even if fixed-seed predictions match. The manifest applies to original published snapshots, not overwritten regenerations. `SUMMARY.json` contains independently recomputed totals.

## Research conclusion

The present training signal is inadequate. Raw adjacency trains topical co-occurrence; neither that objective nor 196 binary success outcomes from a lexically biased proposal policy taught reliable summary-to-evidence grounding on unfamiliar modules. The next study must improve the learning objective and inference representation, not feed the model gold actions or treat a falling training loss as acquired competence. Preserve verified state/authorization layers separately; no authoritative commits occurred here.
