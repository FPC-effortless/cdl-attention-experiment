# CSM-COLLAPSE-004 prospective diagnostic

Parent:21b0d41cebae4762577d4c34f60cc07bdceac6c7. Investigate the observed
space-prediction collapse, not establish generalization or architectural superiority.
Authentic Tiny Shakespeare SHA256 remains
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed.

Twelve fresh fits: seeds11/23/37 × hard, soft_routes, soft_admission, residual_front.
All retain the identical 34,479 learned parameters and seeded initial arrays.
Regulator fixed to multi, original record width32/8slots. Training uses the first
129 bytes of TRAIN:128 next-byte targets repeated256 times, batch1, AdamW lr.003,
WD.01, clip1. Reset all state at each presentation. Full128-token gradients.
All fits see identical text, supervision, optimizer and update budgets. Nominal
parameter counts match; operations differ and hardware FLOPs are not matched.
These are mechanistic ablations, not a claim of compute-matched superiority.

hard:unchanged archived operations and biased straight-through categorical choices.
soft_routes:softmax addresses/operator/link traversal; keep value/link edit choices
hard. Destination becomes distributed, so original single-slot frame guarantees
and hard-address semantics do not apply to this diagnostic variant.
soft_admission:only value-edit categorical choice becomes soft probability.
residual_front:readout input becomes (written_record + current_frontend)/sqrt(2).
No new parameters. Other operations remain unchanged; this tests an alternative
information/gradient path, not proof of improved memory.

Primary: final training NLL, accuracy, top-1 space share and prediction diversity;
fixed fit criterion NLL<0.7 AND accuracy>=80%. No early stopping or best checkpoint
selection. Curves at0/32/64/128/256; save every final model/optimizer/RNG checkpoint.
Use adjacent TRAIN bytes[129:258] as a distinct diagnostic transfer excerpt,
never as a selection signal. Exposed DEV first1025 bytes scored only at final;
protected TEST never scored. These tiny-training fits are debugging evidence,
not evidence of broad real-world performance.
Record activation saturation, readout feature variance and blockwise gradient
norms; additionally audit frozen longbench immediate and multi_tb64 checkpoints
on the same real128-token training excerpt without optimizer updates.
Verify baseline exact forward/gradient parity; variants' causal prefixes,
parameter/initialization equality, finite gradients and state independence.
Independent cold replay checks hashes and NLL at absolute tolerance1e-6.
Retain every result. No paid compute or GPU, no merge/deployment.
