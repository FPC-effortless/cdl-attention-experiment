# Draft native TRAIN worker

Stage role_fit.py into the pinned checkout's cog_comm namespace along with
the reviewed role_control.py and role_mixers.py. An independent native
admission must pin the complete source, runtime and exact new TRAIN artifact.
Do not stage CONTROL actor/truth or historical splits.

The worker requires the native seed and output directory (TRAINLAB_SEED and
TRAINLAB_OUTPUT_DIR), a fresh output directory and exact TRAIN SHA-256.
It validates the complete eight-world factorial but sends the model only seven
ordinary visible word IDs per world. Four arms share the unchanged paired
objective, with contiguous s-flip pairs, and the same one-family-per-update
schedule on the same seed. Old BOS/ANSWER tokens are absent; last-position
answer CE is not counted as ordinary prefix CE.

One continuous AdamW trajectory uses linear eight-update warmup then constant
learning rate. It saves model-only endpoints at 128 and 1280 updates; there is
no refit or selection between them. Private resume sidecars bind each model
checkpoint hash to optimizer and Torch CPU RNG state. The saved schedule and
vocabulary, per-commit losses and gradient norms, optimizer steps and parameter
hashes support replay and independent receipt checks.

Each arm/seed presents exactly 1,024 then 10,240 worlds; 7,168 then 71,680 input
tokens; 6,144 then 61,440 prefix targets; 1,024 then 10,240 answer targets. The
width-48 arm is exposure-matched, not compute-matched. The worker produces no
held predictions or gate outcome. No fitting was run while constructing this
draft.
