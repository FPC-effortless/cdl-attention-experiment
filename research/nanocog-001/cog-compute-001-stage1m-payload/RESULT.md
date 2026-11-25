# COG-COMPUTE-001 Stage 1m: fixed-route payload swap

**The frozen model uses some subject-value content, but it does not bind subjects reliably.** On the weakest original ×4 checkpoint (seed 173), substituting the correct span's counterpart payload after routing increased the exact swapped answer count from 192/384 to 237/384 and both-orientations-correct from 10/192 to 48/192. The margin favoring the swapped value increased on 377/384 subject rows (mean +1.323 logits), yet the top choice changed on only 46/384. Of 190 originally correct subject answers, only 43 exactly followed the new value. Fixed-route coherent swapping of *all* payloads solved just 9/192 subject pairs but 188/192 relation pairs. This is a subject-specific failure in this checkpoint, not a generic inability of the model to act on its payloads.

## Matched causal intervention

One frozen scalar ×4 checkpoint; no optimization, new memory, or new held data. Use all 1,152 exposed prompts for actor capture, score both orientations of the 384 verified assignment-twin pairs (192 per change type). Only the values in the two intact facts change in a twin. The actor constructs twins from the raw synthetic prompt grammar, without supplied answer labels or target span, enumerates every span intervention, and passes full-vocabulary winners to its unchanged native continuation. Detached evaluation uses the native first answer marker and native EOS after the candidate token. The intervention is applied at the answer decision, not replayed at EOS. The "actual twin marker" control uses twin marker logits with the original continuation and marker eligibility; separate native twin performance is also reported.

| Intervention | Subject swapped exact /384 | Subject both /192 | Subject all 8 /48 | Relation both /192 |
|---|---:|---:|---:|---:|
| Native, scored against the opposite answer | 192 | 10 | 0 | 0 |
| Full rival weight to target, original payloads | 140 | 3 | 0 | 0 |
| Swap only the target span, native weights | 237 | 48 | 3 | 0 |
| Swap only the rival span, native weights | 143 | 1 | 0 | 0 |
| Swap target span, full rival weight to target | 240 | 50 | 5 | 0 |
| Swap all aligned payloads, native weights | 191 | 9 | 0 | 188 |
| All swapped payloads and twin residual, native weights | 191 | 9 | 0 | 188 |
| Actual twin marker logits, original suffix | 190 | 9 | 0 | 190 |

Genuine native free-generation correctness on both unmodified orientations: 9/192 subject, 190/192 relation. Native emitted the same value token or other top choice on 171/192 subject pairs, which is why pooled opposite-label accuracy around 50% is not a success. One-span replacement puts the rival value in the target span without removing the original rival; it is a diagnostic and can duplicate value evidence. It is not proof of coherent binding. Swapping all payloads keeps the two-fact assignment coherent and exposes the failure even with the original routing fixed. Neither this nor an oracle routing intervention isolates the residual uniquely to routing, pooling, payload encoding, or downstream decoding.

Four checks: all native marker logits/weights match the prior endpoint within the declared tolerance; every full-transfer candidate's argmax matches the prior 100% tape; supplying the twin's marker residual additionally produces identical logits because the changed tokens lie outside that residual's local window; all 4,608 replacement windows that contain neither swapped token were *bitwise* no-ops at the marker.

**Next test:** retain two messages separately through the readout in a same-budget comparison against pooled summation, then test coherent role/value reversals with no label-supplied addresses. That would test whether pooling discards subject identity without adding persistent memory.

## Reproducibility and scope

One CPU TrainLab engineering diagnostic, 3.822 s, zero optimizer updates; not a benchmark or promotion. Frozen checkpoint, raw actor rows and terminal artifacts reconciled. Separate scalar recount agreed 24 result counters; first recount stopped at JSON serialization of a NumPy scalar and its partial output was preserved. Serialization-only recovery did not rerun the actor or change arithmetic. The additional no-op and conditional-choice audit is explicitly post-run. Private replay includes the actor/tapes, scorer/recount, verification inputs and integrity manifest.
- LOCK.json SHA-256: c3fe0c2faace011fdc72cab74ef9562b48b231ec288231947ca2d29a77003ed5
- RUN.json SHA-256: b5877f42f40620117bffed827716cbcad599d15ac4fab53a5f54563d63b9cee8
- SCORE.json SHA-256: 03bf49ce177f45f20056f778e9d1a46d824fadbccba40ebb40e3c22bc5adc94c
- RECOUNT.json SHA-256: 6be905ff5b92b0bea26e7ed6ab58cdebba2042f4b177da7d151abbdfd7719d0f
- ATTRIBUTION.json SHA-256: 0850f352a8bec0bd9e6d6dd79c1b79fad36f65403b17fd6514b8e8e5477fc784
