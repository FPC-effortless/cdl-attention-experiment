# COG adaptive repair 002

The main question is whether a language model can use both the relevant record and the person named in a question. On an identical complete history, switch the requested person to another whose answer is different. On a separate paired panel, change the authoritative value and keep the person fixed. A pair is correct only if both unrestricted top-1 answers are correct.

## Main whole-model screen

There are 39 fits from the 96-update BALANCE pilot checkpoints across seeds 9731, 9743 and 9769. Per seed, C receives final, trace, pair or combined supervision on balanced examples plus final and combined supervision on the old generator. R receives final or combined supervision on balanced examples. H receives combined supervision. GQA and GRU receive final or combined supervision. Each model starts from its own original checkpoint. R starts from exactly the same C checkpoint, then attaches a zero-output copy head. All runs reset the optimizer. H receives no optional-work penalty during this continuation.

Each fit uses 256 updates, eight question streams per update, and WikiText replay on 64 updates (four 128-byte windows per replay). English and symbolic batches alternate. All arms in a seed and data condition receive byte-identical streams. The price of changing an objective is paid out of a fixed total cognitive loss weight. Trace supervision uses labels computed from the visible prefix at the supervised position. For a changed answer, the paired loss increases the change in relative log odds. For a stable answer it penalizes distribution divergence. These labels are training-only.

The balanced English training generator gives every subject one official record and one rumor, randomizes claim order, and includes both relevant revisions and changes to the named person in the question. The original generator always repeats the target subject in the rumor, admitting a question-blind strategy. The old held symbolic cancellation pairs are solvable by copying the retrieved value. Report the old panel for continuity, alongside genuinely query-dependent and shifted-form tests. Independent generated confirmation uses 64 worlds per panel with a fixed dev seed shared across models. The same worlds reused across seeds give 192 model–world evaluations and 64 unique worlds per panel. Distinguish these from natural-text evaluations.

## R model

The contextual-copy model appends a finite 12-slot byte distribution to C. Learned causal 24-byte features and the C hidden state produce write keys, queries and write rates. A trainable scalar projects the retrieved byte mass into the ordinary 288-way logits. Initial predictions equal the C parent. Report the comparison against the same checkpoint with the read contribution disabled after training.

## Reproduction

Use the GitHub Actions replay or the local run_repair.py CLI. The workflow validates the prior 96-update training and test data hashes before reusing the parent. The terminal 39-fit collector verifies checkpoint and result hashes. Report language BPB, new and old paired answers, and remaining generalization and deployment limits. The optional specialist experiments have access to training annotations beyond the whole-model final-answer condition and are reported separately.
