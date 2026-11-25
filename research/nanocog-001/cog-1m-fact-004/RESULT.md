# COG-1M-FACT-004: the frozen router scales to eight known records, but cannot safely recognize absence

We reused all three exact FACT-003 checkpoints without training, updates or new parameters. The same learned dot-product score matched names against 2, 3, 4 or 8 explicitly declared records. This is a 2,252-parameter mechanism with a finite raw-text parser, not the roughly 1M language model and not a replacement for full Transformer attention.

| Held known-record test | Strict complete families across the three seeds (out of 64 each) |
|---|---|
| Two records | 64, 63, 63 |
| Three records | 64, 63, 63 |
| Four records | 63, 63, 63 |
| Eight records | 63, 63, 62 |
| Record permutation | 64, 63, 63 |
| Preserving an untouched record | 63, 63, 62 |

Each family crosses two factual worlds and two queries. All registered known-record criteria passed in every seed at every roster size. Reversing candidates gave bit-identical aligned scores and zero rank changes or ties. Expanding a roster preserved old pairwise scores within the preregistered FP32 tolerance: maximum absolute drift was 0.00000668. K=2 bitwise reproduced the actual original DEV output on all 768 archived rows. This establishes transfer of the existing scorer to more distinct known names; it is not a speed or long-context benchmark.

**The absence test failed.** In each seed the closed-set model wrongly answered every absent query (128/128) and assigned every absent active update to a real address (128/128). These were not just ranking errors: 112–126 of 128 absent active updates changed a slot immediately, and 106–118 still corrupted state after both updates. An absent canceled update did not mutate state, but the router still selected a real address.

DEV-only thresholds reduced but did not solve the errors. The max-score cutoff still emitted 108, 85 and 30 false answers on 128 absent queries per seed; the margin cutoff emitted 118, 117 and 85. The max-score cutoff still caused 112, 80 and 20 immediate state changes on 128 absent active updates per seed; the margin cutoff caused 120, 112 and 70. Some thresholds also degraded known-case performance. None met the registered absence gate.

A scorer-only exact-spelling no-match guard, while keeping the learned disposition and value decisions unchanged, produced the exact answer and complete state on all 7,680 held rows with no false accepts. This is an engineered reference for distinct literal names, not proof of aliases or open-world identity. It isolates the failure to addressing and admission within this fixture: useful selection among matches is not permission to return a fact or mutate a record when the target is missing.

**Next:** use an explicit authoritative existence gate for updates to literal keys. For learned or fuzzy requests, separately test unknown and ambiguous aliases with refusal and confirmed non-mutation before deployment; do not substitute either tested confidence cutoff for this requirement.

The first score process failed before decoding any data because an error-check helper required a diagnostic argument. It emitted no result or calibration. A separately reviewed compatibility wrapper supplied only that diagnostic default and invoked the unchanged scorer over unchanged pinned wires. No model reruns were performed. Independent raw recount matched the complete score and calibration.

Official score SHA-256: c9a1f6b64070fbc0b35a1839f1b06c70c57a6f4be037c7b0f258b6df04a897fc. Independent recount: c930508b87acbd13746a79d10188b3d78ee02c23530d98fd2aa6bb141b236f46.
