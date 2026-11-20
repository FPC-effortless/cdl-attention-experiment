# Role and operation separation, with bias recovery

## Frozen grounding comparison

Three fresh seeds 419, 443 and 467. Compare a consequence-trained modular local role model and transition-induced exact verb lexicon with independently trained joint GRU and Transformer pointers. Give each model the same 4,096 familiar transitions and 96 support transitions, 24 per novel verb. No gold actions enter training. Randomly permute the four extra word meanings per seed. The candidate reads a supplied exact vocabulary of support spellings and masks verb identity from role inputs. The baseline may use the same spellings. Candidate support estimates verb identity by marginalizing role uncertainty and is frozen before evaluation. Baselines update on the same support transitions with familiar replay.

Use 512 examples per seed and cohort. Freeze the familiar template, extra verb in support layout, unfamiliar layout with familiar verbs, joint extra verb and unfamiliar layout, implicit unfamiliar clauses, and ambiguous/unsupported cohorts. The held layout recombines shared role phrases in fresh orders with new filler words. The implicit set uses different role wording. Compare the candidate's locality repair with the same modular role learner using full-sequence context. Never use unsupported examples as executable gold actions.

Joint compositional gate: at least 90% exact action accuracy and at least 90% exact precision among actions with confidence at least .80, covering at least 70% of joint examples. Report the same figures for implicit and ambiguous cohorts. Record action and state-effect scores separately. Timing comparisons require both independently trained neural peers to hit 90% exact accuracy and fall within 3 points in the relevant cohort. This is a constructed familiar operator family.

## Frozen training-bias recovery

On the 25-cell finite program task, use eight biased training observations, nine potential interventions, and eight sealed test cells. Simulate an unknown constant output offset (including clean), a misspecified input-dependent offset, and random tables. Fit program × offset hypotheses jointly. Compare no correction, a one-anchor correction, and disagreement-selected interventions at budgets 0, 1, 2, 4 and 8 with fixed random matched-budget anchors. Also test a compromised anchor channel. Admit only if surviving hypotheses agree on all unobserved cells. Report coverage and accepted accuracy, and require zero wrong admissions and at least 95% true constant-offset recovery on a clean anchor channel with at most four independent queries. The independent channel is supplied by the fixture. This tests bias recovery within a declared observation model and probes behavior when that model breaks.

## Follow-up after the constant-offset run

Use disjoint seeds 521, 547 and 569. Extend the nuisance family to a single unknown change point on either input axis with independently unknown offsets on its two sides. Compare constant-only recovery at the same 8 training labels and intervention budget. Hold out eight cells for scoring. Use at most four independent query labels and reserve two random blind anchors for a qualification ablation. Include a checkerboard offset outside both nuisance families. Gate the extended family at 95% correct recovery with zero wrong accepted programs on the piecewise cohort; report feasibility when the gate fails. The supplied grid and primitives remain fixed.

## Follow-up on role wording

Use new seeds 523, 541 and 563. Freeze an implicit-wording suite after the earlier implicit failure. Compare the same local model with equal training counts, randomly replacing half of familiar training commands with a separate authored bank of six plain-language clauses. Keep role and verb evidence separated and keep novel-word support at 96 examples in the original layout. Score the original implicit examples and two fresh implicit cohorts. The first reuses role vocabulary from the enlarged training bank in entirely new clauses. The second uses sparse or different role cues. Freeze the competence gate at 90% exact accuracy on both fresh implicit cohorts. Report confidence-only ambiguity acceptance.
