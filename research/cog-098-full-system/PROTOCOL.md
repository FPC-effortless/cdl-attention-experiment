# COG-ATTN-098: full-system qualification

Scientific claim is **NOT QUALIFIED** until a single trained, persistent, executable agent completes a genuine interactive and unassisted train-test cycle. The purpose of this protocol is to prevent component demonstrations from being misreported as full-model success.

## Required operation

The same agent must implement raw observation -> learned perception -> persistent memory -> hypothesis revision -> planning -> verified action -> real outcome -> online update. The evaluator may define legal actions and score outcomes but must not send correct entity IDs, reference plans, semantic gold labels, or the intended action to the learner. The complete agent freezes before independent test.

## Authentic tasks

Software: SWE-bench Verified or comparable externally-authored executable bugs and acceptance tests. Dialogue: MultiWOZ 2.2 or a stronger independent multi-turn corpus with withheld goals and corrections. Interactive action: an independently released environment such as MiniGrid and a separate puzzle-control family with unseen tasks. Gold patches, slot annotations and optimal paths are evaluator-only. No public account mutations; use isolated workspaces.

## Strong controls and measurements

Compare a similarly resourced attention model, existing COG controller, semantic/typed/lexical baselines, and the integrated learner on identical tasks and budgets. Retain all failures. Measure complete-task success, unsupported/incorrect actions, illegitimate writes, generalization, recovery, and actual end-to-end wall time including sensing, routing, memory, verification, tool actions, repair, and online updates. Publish model SHA, exact data splits, per-episode traces, resource usage, and independent evaluator receipts. No claimed speed advantage without matched competence.

CI source-level gates never constitute scientific qualification. A scientific PASS requires complete trained model, authentic sealed test receipts, and grounded outcomes across all three domains.