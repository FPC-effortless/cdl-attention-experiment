# COG-ATTN-097d preregistered experimental rules (before scoring)

## Question
Can an attention-free trainable encoder learn useful representations from unlabelled sequential raw-text experience and improve retrieval on unseen author-written documentation, without being given query-to-evidence supervision, entity names, role labels or solver hints?

## Learner-visible data
Only the raw docstring **body** text in TRAIN_MODULES. For each text, two disjoint consecutive windows are drawn from the same natural observation. Predictive/contrastive training uses these contiguous observations. Their adjacency is a property of the original observed stream; no evaluator IDs or task answers are provided. All author summaries (including training-module summaries), module labels, and document indices are excluded from training input. Disjoint module split (train/dev/test/unknown) is declared in `experiment.py` and may not be revised after test.

## Evaluation
Freeze all model parameters before any TEST_MODULES summary query is read. The scorer privately uses the author's summary->body linkage for test accuracy. All test document bodies are available only for indexing, never gradient updates. UNKNOWN_MODULES are entirely missing from the indexed corpus. Scores: top-1, recall@5, MRR, missing-query false answers and total training/index/query costs.

## Controls
Untrained same encoder, mismatched adjacent-window pair training with same number of updates, word TF-IDF and character TF-IDF. No transformer comparison yet. Models have differing representation sizes and the lexical controls do not train a network.

## Decision rule
Pass a *bounded representation learning* gate only when trained encoder is better than the untrained and shuffled-pair ablation on heldout summary retrieval, and competitive with the best lexical control; a mere training-loss decline does not qualify. Missing-query false answers count as a separate failure gate. There is no right to claim integrated COG-ATTN, language learning, reasoning or attention parity.

## Limitations and evidence discipline
This uses real Python standard-library docstrings but engineered segment boundaries and an evaluator linkage, not live natural user dialogue. Original 097b COG Archive needs alias/field keys and is not inserted in the learner path. Seeded training may vary with PyTorch CPU. Report all runs including failures, archive exact source and input module hashes, and never choose the best seed post hoc. The test is a proxy and does not establish autonomous skill acquisition.
