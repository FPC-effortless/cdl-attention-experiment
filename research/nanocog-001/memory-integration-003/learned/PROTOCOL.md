# MEMORY-INTEGRATION-003: prospective bounded factual adapter

## Claim and limits

The candidate is a newly trained categorical factual mode attached to the actual
frozen **COG_CONNECTED_1M** language model. It is not a repair of unrestricted
language generation, a demonstration of lifetime retention, or a replacement for
the separate historical PULSE/ANT studies. Those historical TEST panels are not
used for selection or qualification.

The model makes three learned decisions: an event operation, a question route,
and an answer label. Atomic persistence, the schema, and zero/one/two-record typed
traversal are engineered infrastructure and are reported separately. Invalid
learned record/value combinations are rejected and counted wrong on required
writes, not projected onto the correct operation. There are no oracle routes,
text parsing callbacks, gold-ledger inputs, or query-label inputs at inference.

## Admitted native model and exact size

The caller loads `COG_CONNECTED_1M.pt`, SHA256
`d8bab3d1d99a43dfb6cca2257f2caae4d3a7282d62b079c92874ce736a44ec3c`,
strictly into the supplied Connected COG v2 source. This is seed 33, update 4096,
a 65-character TinyShakespeare continuation model. All 993,574 native parameters
remain present and frozen. Features pass through all six native local blocks,
the connected native record board, and final normalization. Features start from
fresh native transient state for each bounded event or query. No diagnostic
route or commit override is used.

| Component | Parameters | Trainable |
| --- | ---: | ---: |
| Complete native language model | 993,574 | 0 |
| GRU, 122 native features to 32 hidden units | 14,976 | 14,976 |
| Joint NOOP/record/value head, 32 to 97 | 3,201 | 3,201 |
| Query route head, 32 to 8 | 264 | 264 |
| Answer label decoder, 13 to 9 | 126 | 126 |
| Total | **1,012,141** | **18,567** |

The joint 97-way operation comprises NOOP and eight records times twelve value
labels. Only the schema-valid subset is accepted. Owner values are Amara, Bela,
Cato, Dina, Evan, Fara, Gus, Hana. Boxes are chest, crate, basket, trunk. Items are
coin, ring, watch, key. Four owner-of-box records and four container-of-item
records persist. Unknown is a distinct read value and answer. The learned answer
head receives a thirteen-category one-hot typed read result; it does not receive
the target label. Native character features are cached in float32 without
quantization; caching changes training cost, not the deployed computation.

## Prospective data lock

`dataset.py` defines the complete grammar and generator. `DATA_MANIFEST.json`
records corpus, split, lineage, and source hashes before the first fit. Private
full generated data, native features, and checkpoint weights remain outside git.

Semantic family roots use seed 823197 and distinct split/index hash lineages:
512 TRAIN, 128 DEV, 256 familiar TEST, and 256 fresh TEST. Each family contains
two histories with lengths 8, 24, or 64. The two worlds differ in one required
transfer versus a cancelled transfer; their final answers must differ. All four
box and all four item question routes are represented. Four initial owner
records and one item link establish the state; other events include overwrites,
cancelled actions, distractor roles, and unrelated later writes. A correct
initial fact can be damaged by any subsequent erroneous prediction.

The control receives only the canonical wording of each TRAIN lineage. The
candidate receives three matched views of the same lineages: canonical,
recipient-first, and completion/cancellation descriptions. Noncanonical views
also apply a fixed globally consistent within-type name/box/item permutation.
All views of a lineage remain in its split. DEV and familiar TEST use canonical
wording. Fresh TEST contains four completely held-out grammar families: passive
delivery, status-front, retention contrast, and destination-first. Each has
64 pairs and each includes all eight question routes. These templates do not
enter feature-based fitting, calibration, DEV, hyperparameter choice, or model
selection. Familiarity here refers to this adapter's grammar exposure; the
native language pretraining is not claimed to exclude English words.

## Fixed training and compute policy

Two recipes times three head seeds **17, 29, 41**, with the same initial head
weights in each matched pair. Adam, learning rate 0.003, betas (0.9, 0.999),
epsilon 1e-8, no weight decay, gradient norm cap 1.0. Each update has 96 sampled
event positions and 32 sampled queries; total batch 128. Each loss is the mean
cross entropy for its task, with equal coefficients for event, query, and the
13-value answer dictionary. Sample family/world/event positions uniformly using
the same seeded stream in both arms; only the candidate wording/view differs.
No checkpoint selection: evaluate the final prescribed update in every run.

The initial cap is **2048 updates per run**. Before any scored fit, one
discarded TRAIN-only calibration uses 32 warm-up and 64 timed updates of the
candidate recipe. Extract only the required TRAIN native features for that
calibration. Fix the actual shared update count to the largest multiple of 128
at most 2048 such that `6 * updates * timed_CPU_seconds_per_update <= 540`.
If 128 updates cannot fit, do not run scored training. Record this decision in
`CALIBRATION.json` before any scored fit. This is a compute adjustment only;
no TRAIN accuracy is used to adjust it. All six scored fits must complete the
same count or the study is incomplete. The combined scored-fitting cap is
600 CPU seconds, extraction cap 900 CPU seconds. Record calibration, extraction,
training, evaluation, persistence/I/O and total wall/CPU time separately. No
paid or remote training is authorized or used.

## Qualification gates fixed before fitting

1. **Admission and frozen controls:** exact checkpoint/source identity, strict
   loading, counted parameters, native feature-to-logit equivalence, and unchanged
   native weights/language outputs after fitting. No claim of language NLL gain.
2. **Nontrivial competence:** at least 95% of familiar DEV pairs per seed have
   every predicted operation and every post-event ledger correct across both
   histories, plus both answers correct. Report TRAIN and DEV components too.
   Failure is failed qualification for this bounded fit, not proof the entire
   COG architecture cannot learn memory.
3. **Complete learned path:** the strict metric above is primary. Report all
   writes, complete final ledger, question routes, both answers, invalid writes,
   first wrong operation, damage after initially correct state, and length/grammar
   subgroups. A correct target answer alone is insufficient.
4. **Familiar preservation:** candidate must lose zero control-correct familiar
   TEST pairs in each matched seed. Report gains and losses and paired aggregate
   differences; a net improvement cannot hide lost previously correct histories.
5. **Fresh generalization:** at least 95% strict pair correctness in each seed,
   with no fresh grammar subgroup below 90%. Report Wilson intervals rather than
   interpreting finite-sample success as a lifetime guarantee.
6. **Causal use:** direct accepted-fact edits on the actual learned query path:
   relevant owner and item-link changes produce the expected changed answer;
   irrelevant facts leave answers intact. Also report shuffled/zero native
   features, empty archive and shuffled archive values as negative controls.
   Exact typed routing/traversal is disclosed even if these controls pass.
7. **Durability and concurrent changes:** fresh-process reload in a new session,
   source dependencies, revision-zero absence, deleted/recreated facts, retry
   identities/authority, invalid predictions, and mutation during inference.
   Revalidate actual read revisions and live sources atomically before saving
   or returning a new reply. Existing exact-request replay is a historical
   receipt; a fresh request observes later corrections.
8. **Cost:** record full native feature extraction and identity checks as well
   as typed/archive/head costs. Short/long histories use the same trained model
   and accuracy panel. Do not call this faster than another model without matched
   accuracy and complete cost accounting. Infrastructure microbenchmarks remain
   separate from semantic qualification.

If competence or fresh/familiar gates fail, keep the artifact experimental and
report the failure. No new objective, grammar, extra fit, selected seed, or tuned
threshold may be substituted after seeing DEV/TEST outcomes in this study.
