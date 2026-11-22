# ROLE-REPAIR-004 frozen adapter: engineering preflight passed

Both candidate and literal-only control add 552 nominal coefficients. The
literal query is captured directly from the original embedding before any
smear/context, and both branches use a second-layer, final-token-only
residual on top of the intact native gate. Only the candidate receives the
actual frozen tap vectors; the control sees distinct fixed signed
permutations of the literal query, keyed only by lag.

On each of two exact frozen ROLE-002 backbones (1171, 1371), a zero-initialized
adapter reproduced every native output across 64 fixture prompts.
A deterministic nonzero fixture changed final logits for all 64 but
none of the preceding-token logits. The Frozen backbone did not change.
The separate native worker captured no answer logits or labels and
executed no optimizer. Its data were the existing TRAIN prompt projection,
not the fresh evaluation panel.

Both arms and both backbones showed full numerical design rank 51, common
literal+lag rank 19, and 32 extra source directions at relative singular
thresholds 1e-6 and 1e-4; independently verified. The singular spectra
and conditioning differ, so this is not a proof of effective hypothesis
equivalence. It addresses a basic control-rank mismatch before research fits.

No repair was learned. This is engineering evidence establishing which
component the next controlled fit would change, not performance. Next is
a TRAIN-only controlled comparison on known-weak backbone 1371, with
three fresh adapter seeds and an independent unseen joint-combination
panel. That comparison, even if it passes, cannot qualify unseen backbones.
Do not open protected TEST or main experiments.

- Rank admission ledger SHA256:
  `d6d1b4fd5328f408659786699c45de860bf41b329ba82c9a3d76e1c6a5ddd729`
- Native receipt/feature seal SHA256:
  `3878393d89de5f85491f05f06416534741f400bd2fd32971fbbf78ca5f3b61e0`
- Rank results SHA256:
  `fcb2c502733ae3e5aadaa159d003bf06b7185c0d6cfb5bef9bc9532064b37348`
- Independent rank audit SHA256:
  `e26ece9ce000621235138f338dd29adcdbed26ab5018142e7065b2698c9966c4`
