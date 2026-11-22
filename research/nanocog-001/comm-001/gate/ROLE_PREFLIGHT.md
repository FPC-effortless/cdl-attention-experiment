# COG-COMM-ROLE-001: local role-binding continuation

**Status:** constructed design fixtures and initialized decoder geometry checked.
No learned fit or result. No original or protected data was opened. This does
not reopen COG-COMM-001.

## Why this is the next test

If the prompt says "ada is red ben is blue ada", the answer must follow Ada
when the colors are exchanged, the clauses are reordered, or the question
asks about Ben instead. Cross all three factors in each family. With strict
complete-family scoring, individual success cannot conceal systematic
inversions or fixed-position selection.

Fresh deterministic split: 320 FIT joint name-pair/color-pair families and
120 disjoint public synthetic CONTROL families. Both splits contain the same
individual vocabulary. Nine CONTROL families also have a name pair not in
FIT; report these separately if comparing pair familiarity. Fixed canonical
name choice, fixed color and fixed rendered position all scored exactly
480/960 worlds and 0/120 families as deterministic fixtures; the correct
reference scored 960/960 and 120/120. Reverse-role scored 0/960 and 0/120.
These are code fixtures, not learned, and not authentic-language evidence.
The fixture scorer has a reduced vocabulary. The future model scoring path
must include all native vocabulary IDs, including reserved IDs, with ties
failing the strict correct-answer rank.

## Verified zero-update readout geometry

Base source `fbe0fcda0e2dc294c8e0435c3b70a3e15ac6deaf`.
Veritas/TrainLab source `eec439492407b0aa26922c23b5ae40d95289f4eb`.
Python 3.12.14, PyTorch 2.10.0+cpu, one thread, seed 1171.
The new renderer uses seven word IDs, with old BOS and ANSWER wrapper IDs
removed to put the whole two-clause prompt in the unconditional seven-token
field. This is a fresh prompt/tokenizer geometry, not the old benchmark.
For one FIT family the chosen last-position logit's embedding derivative was
nonzero at all 56 token positions across eight inputs. Replacing an eighth,
earlier token left the entire output vector exactly unchanged for all eight
inputs at initialization. This shows reach, not semantic use or trainability.
TrainLab `20261122T154328-10f1969bde8b` succeeded, zero optimizer updates,
and `trainlab verify` confirmed record/file hashes. Qualification:
`NOT_QUALIFIED`. Model 37,414 parameters, vocabulary 287; nonzero gradients
could be entirely unrelated to correct choices.

Replay from this research path: put `comm-001/cog_comm` and `nanocog` in a
common scratch directory (or put both package parents on PYTHONPATH), with
CPU Torch installed. Run `python -m cog_comm.role_control` and
`python -m cog_comm.role_control --fixture-report`. For the geometry
software fixture, set TRAINLAB_SEED=1171 and TRAINLAB_OUTPUT_DIR to an empty
user-created directory, then run `python -m cog_comm.role_runtime`. Native
receipt used this exact module and seed, `engineering_diagnostic`, 25-second
timeout, three forward calls, and geometry.json as required artifact.

## Next research admission, not yet run

Freeze exact data bytes and independently verify family split, vocabulary and
edge invariants. Fit the same Arm A local architecture on training-only
factorial data, compare fixed 128-update reference to a fixed longer endpoint
of the same trajectory (recommended 1,280 updates, 8 worlds per step,
three new seeds 1171/1271/1371, fixed AdamW and paired loss). Retain both
checkpoints and all failed receipts, and partition actor inference from
scorer labels. This estimates learnability under declared exposure, not
architectural incapacity, and is not apples-to-apples with the old EASY
dataset. A trained positional-only hypothesis and simple matched learned
word-interaction baseline are useful controls. Freeze numeric acceptance
criteria before evaluation; use complete family and strict full-vocabulary
world counts, query-pair reversal and clause-order controls, and raw logit
vectors. Setup requires claim-specific native adequacy with independently
reviewed contracts, then exact-bound checkpoint prediction and independent
scoring. Do not turn this public familiar grammar into sealed evidence.
