# COG-ATTN-097e protocol — outcome-trained retrieval and abstention

Status: executable local research diagnostic. This is not a full COG-ATTN agent, artificial-general-intelligence demonstration, semantic competence qualification, or live authorized execution.

## No training wheels — operational contract

- The model observes only raw text requests, an archive of raw text memories, generic retrieval scores, and the **one-bit** outcome of actions it itself selected during training. No entity ID, relationship schema, gold answer/candidate list, task demonstration or true answer after failure reaches the learner.
- `World` privately keeps query/body source associations to return a single success/failure bit for the chosen document. This is a controlled simulator of retrieval success, not a natural user's feedback.
- The action set is necessary generic interface `SELECT(document_id)` or `ABSTAIN`; the corpus provides no correct document ID to the learned policy.
- Candidate generation by full-archive word/character TF-IDF is *inside the system*, unsupervised and included in all total cost comparisons. Top candidates are not oracle preselected. Oracle labels are only accessible inside the evaluator.
- No feedback or optimizer update on test modules. Raw docstrings were naturally written by CPython authors and are not synthetic grammar, but the task uses their first sentence as a query and remaining docstring as an archive record. This is a documentation retrieval proxy rather than spontaneous conversation.

## Freeze and splits

- Three preselected seeds: 1701, 1707, 1711, reused from the prior study for comparability; data split module lists are frozen in `experiment.py`.
- Disjoint train/dev/test by Python module; remove exact duplicate body or query hashes in train→dev→test order.
- Training: 297 naturally authored documentation records. Three independently incomplete archives; 891 training requests per seed. Outcomes come only after each chosen retrieval action (up to 3 attempts).
- Development: 48 positive documents and 13 missing-evidence requests from separate modules; learn an `ABSTAIN` threshold using binary outcomes of chosen actions. Threshold objective: `correct - 2*wrong - 0.05*abstain` with ≥15% dev coverage.
- Frozen test: 161 answerable queries and 23 absent-document queries from entirely different modules. Models and thresholds frozen. Same 184 underlying queries evaluated on each seed; do **not** count 552 as independently authored cases.

## Arms and falsification criteria

1. Character TF-IDF retrieval, including independently calibrated abstention.
2. Word TF-IDF retrieval, same calibration.
3. Outcome-trained raw char/word-hash EmbeddingBag+MLP decision scorer, with gradients to the input embedding.
4. Neural-free decision scorer trained on identical lexical features/feedback.
5. Reward-shuffled neural scorer as a negative control.

Compare both *forced-answer accuracy* and *selective precision/coverage*, missing-document false actions, total wall time including index construction, feedback acquisition, model fitting, calibration, and query decisions. A learned model cannot be promoted for improved precision if it achieves that only by not answering. Admission requires at least equal answerable top-1 accuracy and no worse missing-evidence harm than the strongest lexical control, plus matched-quality online efficiency and a 15%-coverage floor. A clear shortfall on any criterion fails qualification.

No evaluation after viewing test labels may change thresholds, seeds, model sizes, epochs, candidate budget, split composition or metrics and still count as this registered comparison.
