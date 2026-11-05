# LRR-003 preregistration — raw causal token addressing (2026-11-05)

Status: frozen before confirmatory GPU/CPU scientific run. Local seeds 0-2 are exploratory only; not evidence.

## Hypothesis
A trainable causal local-convolution parser can predict entity-string and relation-code addresses at arbitrary word positions in raw sentences, write positions to a hash index, and retrieve prior answer tokens without runtime QK attention or full-history scan.

## Inputs and model
Restrict v0.3 to ordinary unmarked text of the form 'mira owns value42 . ... what does mira possess ?'. No <F>/<Q> tags, no entity IDs or value boundaries supplied to the runtime API. Twelve named entities; four relation families with four aliases each. Value strings are new, outside the model vocabulary. All relation aliases are seen in training. The four-code capacity is deliberately supplied and must be counted as a strong prior.
Model: 3-layer causal dilated convolution, learned local entity/predicate offset (one of preceding four tokens), single-word four-way learned relation code, and trained token write gate. Hard inferred (entity_surface, relation_code) keys address a hash posting list at runtime. Only the newest eligible prior value token is returned. Encoder and memory ingest cost O(sequence length); expected direct lookup O(1)+O(k); no total O(1) claim.
Training-only gradients use gold answer-token positions derived from answer strings, a soft four-offset relaxation, and token-ranking loss. This is answer-position supervision, not purely reward-only training; no canonical relation labels, fact boundaries or full synonym-pair matrix are passed to the network.

## Split
Training: H=5..8 facts with hard distractors; all aliases are present, but ordered fact/query alias pairs (0,1) and (2,3) are excluded in each relation family. Developer evaluation uses pair (0,1). Confirmatory TEST uses pair (2,3), untouched by developer runs. Confirmation uses training seeds 21,22,23 and 32 independent new test documents per seed for H=6,32,128 facts. Evaluate only exact earlier gold token indices; all failures are retained.

## Comparators
(1) Grammar-aware symbolic clause parser plus union-find, with pair links extracted from the identical answer-labeled training stories, not from hidden relation-class IDs.
(2) Causal Transformer pointer model trained on same examples and updates, with matched parameter budget within 5% if feasible; report mismatch.
(3) Entity-key ablation: relation code only.
(4) Learned-relation ablation: exact predicate word.
(5) Oracle evaluator position (never supplied at runtime).

## Metrics and decision
Primary: mean and per-seed TEST recall@1 at H=32. Bounded learned-feasibility gate requires >=0.70 on at least 2/3 seeds; this is NOT a superiority gate. Also record H=6/128, abstentions, wrong-index errors, actual posting writes, hard-code occupancy, CPU timing including encoder/index ingest, source hash, configuration and test count. Challenge-only controls: contradictory late update, nonce-name substitution, unknown predicate and out-of-grammar phrasing, separated from primary.
Report all controls. A neural novelty claim requires improvement over the strongest symbolic baseline under ambiguity/unseen semantics, not merely high synthetic recall. Do not merge into a language model or claim attention replacement without trained matched LM loss and throughput.

## Non-claims
Fact grammar remains restricted, statement/query punctuation provides structural cues, training uses answer-derived pointer labels, entity identity is exact literal equality rather than learned coreference, relation ontology count is supplied, and no open-domain text or LM perplexity is measured.
