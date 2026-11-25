# ROLE-REPAIR-004 train-only residual fit candidate

Scientific question: can contextual local tap features improve a frozen
weak ROLE-002 checkpoint relative to a lag/literal-query control of the
same nominal size? This stage records optimizer trajectories only.
No held data, prediction, task qualification, continuation of MAIN
or retention, scoring, or model promotion is permitted by this packet.

- Backbone: only exact weak seed1371 update1280 gated32 ROLE-002 clip1 /
  quarter-prefix checkpoint SHA 517c990e5834a1648524b59bd287f97f472583fa642ce0e5a8dc95270536e890.
- FIT: 320 exact synthetic TRAIN families × 8 fixed-order worlds
  (SHA 02607c19e760e164989865a54422eb7cd78fed7d117a7fa2ff6c4df9d700f9f8).
  No actor receives an answer, role ID, pair field, or grid choice; sampled
  labels enter only full-vocabulary answer CE and same-assignment-flip
  softplus foil; coefficient 1 each, no prefix CE.
- Arms: tap_compare and literal_control, same nominal 552 new FP32
  weights and matched exact initialization in each adapter seed
  1471/1571/1671. Frozen base unchanged (37,414 original scalars).
  Natural no-fit TRAIN feature geometry: common rank 19 and joint rank
  51 both arms and both checkpoint witnesses at 1e-6/1e-4; these ranks
  do not equate the feature spaces or effective expressivity.
- One family per update, four complete permutations (320 each) seeded
  by adapter seed xor 0xC060001, identical order for paired arms.
  Single uninterrupted 1,280 updates; keeps both updates 128 and 1,280
  without checkpoint selection. AdamW peak .001, linear warmup 8,
  betas (.9,.999), epsilon 1e-8, weight decay .01, adapter clip1.
- Each native attempt one CPU thread, ≤180s, no paid compute or retries.
  Actual optimizer traffic 1,280 forwards/backwards, 10,240 labeled
  worlds, 71,680 input tokens, 5,120 foil pairs, zero prefix targets.
  Three additional forward calls on the first TRAIN family only check
  native no-op and endpoint earlier-logit parity without labels.
- Data validation and schedule helper source is intact original ROLE-002.
  Actor adapter explicitly acts only on L2 at the last of seven tokens,
  initially exact no-op. Isolate adapter gradients; verify frozen weights
  against source and at the two endpoints. Save adapter weights only
  with exact base, vocabulary, seed, data, schedule and state digests.
  No original model duplication in outputs or optimizer sidecar.
- Interpretation solely repeatability across **adapter RNG/data-order
  seeds on the known weak backbone**. This is not a fresh backbone-seed
  comparison or a language, new architecture, or serving-cost claim.
  A separately locked fresh synthetic panel and scorer would be needed
  to test competence. No protected or previous holdout inputs enter FIT.
