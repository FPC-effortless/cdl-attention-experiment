# LRR-002 — Pairwise answerability credit without canonical relation labels

**Preregistration for GitHub Actions confirmatory run (2026-11-05).**

## Question

Can a discrete attention-free relation address be learned using only binary
answerability outcomes between a fact predicate and a query predicate, with
no canonical relation-class labels passed to the learner?

LRR-001 reached perfect accuracy under *supervised canonical lexical labels*.
LRR-002 removes that kind of training supervision while retaining explicit
fact/entity segmentation, the same finite lexeme vocabulary, the same
causal index, and the same held-out story generator.

A balanced-code-usage objective is justified by an **exploratory local
probe on seeds 0–5**, where plain pair agreement collapsed multiple
semantic families onto a few codes and a marginal-usage penalty recovered
separation. Those exploratory seeds are not confirmatory evidence.

## Fixed training protocol

- Lexemes: 24 predicate synonyms from six synthetic relation families.
- Training supervision: 276 binary fact/query answerability pairs (36
  positives; 240 negatives); these are pair outcomes, **not canonical
  class IDs** supplied to the trainable encoder.
- Model: embedding + 24-wide Tanh MLP with six output codes; exact parameter
  count recorded by runner.
- Training: AdamW, 800 updates, learning rate 0.04, weight decay 0.001.
- Loss: mean positive -log P(same address) plus mean negative
  -log P(different address) plus 0.02 token-code entropy.
- Treatment: add 0.5 × KL(mean code usage || uniform).
- Control: identical training with KL coefficient zero.
- Training-only differentiable code agreement:
  P(same) = sum_c P(c|fact_predicate)P(c|query_predicate).
- Runtime: each token receives **one hard address** from independent
  encoder; lookup uses hash posting lists. No query-key token attention.
- Fixed number of codes (six) is an oracle *capacity assumption*, not a
  learned ontology size. Record it as a limitation.

## Confirmatory test (not used for hyperparameter selection)

Seeds **101,102,103,104,105,106**, 32 trials/seed/arm/context size,
H in {8,64,256,1024}, k=1. Each trial has new opaque entity and value
symbols, forced predicate alias mismatch, same-entity/wrong-relation
distractors and wrong-entity/same-relation distractors. Neither runtime
index sees hidden correct relation family or gold token index.

Primary endpoint: exact gold token recall@1 at H=256, across six seeds.
Support requires **6/6** balanced seeds with:
- positive pair agreement 1.0, negative pair separation 1.0,
  six uniquely occupied hard codes;
- H=256 recall@1 at least 0.99 per seed; and
- mean H=256 recall@1 at least 0.10 higher than unbalanced control.

All observations, including failed seeds and any invalid runs, must
be preserved. This gate is deliberately ambitious; failure is admissible.

Secondary: H=8/64/1024 recall@1, seed spread, code occupancy.
Index query probes and posting visits are instrumentation, not a standalone
end-to-end runtime benchmark.

## Non-claims

No unseen lexical relations, free-text parsing, learned entity identities,
compositional pronoun resolution, end-to-end LM training, long-context
language-model quality, or GPU efficiency. Successful pairwise code learning
would be **outcome-based alias clustering on a closed synthetic vocabulary**,
not a general semantic attention replacement.

Future LRR-003 should remove supplied fact/entity segmentation and test
natural-language phenomena and autoregressive LM quality under matched
parameter/data/training budgets.
