# LRC-ROUTE-REPAIR-002: prospective repair confirmation

## Claim and falsifiers

This experiment repairs the sentence-adapter's identity-erasing hard addresses
and content-blind page/rank selection. It tests a new LRC-derived routing recipe,
not a source-compatible patch or a general language-model advance. The recipe
must distinguish people in learned write keys, retrieve via actual bounded
posting lists, and follow a changed first-hop partner to the appropriate second
record. Accuracy without both-support and bridge evidence is insufficient.

The existing 16-record, four-person, two-era VB-001 sentence task and its training
set are unchanged. No semantic parser, canonical identity, era/person metadata,
support index, or answer enters normal inference. Sentence boundaries and the
finite vocabulary remain supplied to every arm. Fresh pairs are combinations of
known people, not new entities or language templates.

## Development and selection

Development data only: seeds 84001 (IID), 84002 (withheld pairs), 84003 (bridge),
64 quartets each. The independent retrieval probe used initialization seeds
84101–84103. All three completed the same 1600+400 schedule and achieved complete
support retrieval; no architecture or outcome was discarded. A full decoder
integration used seed 84111. Review then removed stale frozen gradients from
Stage-B clipping and made a missing first bridge abort the entire chain. Seed
84112 qualifies those corrections with the same DEV data. These are exploratory
development results, excluded from confirmation statistics. All logs are kept.

## Fixed repair architecture

Fresh SentenceEncoder, width 32, identical design to the original adapter. Writer
and first reader are separate 32→32→32 GELU networks. Second reader is a
64→64→32 GELU network applied to the query encoding concatenated with the first
record's sentence value. Writer/reader outputs are L2-normalized continuous fine
keys. A shared 16×32 learned codebook maps both into the same discrete addresses.

At memory construction, encode each record, compute its address, and build one
posting index per example. Each read compares its query key to 16 codewords,
probes the top two addresses, and scores only returned candidate fine keys.
Each bucket has capacity two. A bucket above capacity returns no candidates and
sets an overflow flag; no record prefix, full-memory fallback, or label-assisted
fallback is allowed. Deduplicate codes and candidates. Equal maximum content
scores produce a null read. A null first read aborts the two-hop chain. Codeword
ties use fixed code-index order, independent of occurrence order.

The maximum is four candidate record comparisons per hop, twice the archived
adapter's two-candidate budget. Memory construction is O(N); codeword comparison
cost and candidate scoring must both be reported. A 16-code table with cap two
cannot maintain full accessibility for arbitrarily large memories. No claim of
long-memory scalability follows this fixed 16-record experiment.

The answer head is 32→32→4 with GELU. Its sole input is the actual final selected
record value, or zero after a null chain. It cannot answer from the query alone.
Total allocated trainable parameters: 12,900, plus 512 nontrainable codebook
values. Original LRC has 18,879 parameters; MemNet has 19,129. We report this
difference; unused weights are not added to manufacture matching counts.

## Fixed learning schedule

Every arm sees exactly the original 8,192 TRAIN examples generated at seed71001,
2000 minibatches of32, and the same per-initialization minibatch stream from
generator100000+seed. AdamW has zero weight decay and norm cap1. Original LRC
uses the archived agreement trainer, two forwards per minibatch, lr.002 and its
original counterfactual RNG formula. MemNet uses the archived support objective,
one forward, lr.002. Its default runtime is soft attention; a hard-read diagnostic
also scores all16 records and is not called indexed inference.

Repair Stage A: 1600 minibatches, AdamW lr.003. Use both supplied support indices
to train cosine-score cross entropy (temperature .08) plus 1 minus the positive
cosine. Gold first value is used only in this training objective. Dense all-record
support comparisons exist in training. This replaces the old stochastic
writer/page/rank credit assignment rather than silently retaining its objective.

At the boundary, fit spherical K-means16 on learned writer keys for every unique
TRAIN token string. Five restarts, at most100 iterations, generatorseed+20000;
select minimum TRAIN inertia. No TEST strings, semantic IDs or metadata. Include
deduplication and K-means in fit wall time. Freeze encoder/writer, clear their
gradients, and preserve the Stage-A optimizer state.

Repair Stage B: 400 minibatches, new reader AdamW lr.002. Optimize codeword CE
against the gold supporting writer's code plus .25 times the support-score CE.
Clip only active reader parameters. No early stopping or checkpoint selection.

The answer decoder trains on the detached gold-final sentence value throughout
all2000 minibatches, with a separate AdamW optimizer, separate clipping, and the
same .003/.002 learning-rate phases. Thus answer gradients cannot change key
formation. The model has two optimizer step calls per minibatch. Teacher support
use is stronger than the old recipe's conditional loss; this is declared rather
than described as identical optimization.

The shuffled arm applies one random joint permutation of the two support indices
within each minibatch, with dedicated RNG200000+seed. Use that same pair for the
first teacher value, every key/code target, and the decoder's final teacher value.
Answers stay correct. This prevents a correct-support bypass in the control.

## Confirmation barrier and comparisons

Arms: repair, repair_shuffled, original_lrc, memnet. Initialization seeds901–903,
all fresh. Twelve fixed fits, sequential CPU execution with one Torch/BLAS thread.
Rotate arm order by seed. No concurrent fit workers. Save every checkpoint,
optimizer state, RNG state, per-update log, minibatch digest and receipt. Freeze
source hashes and a local review commit before prepare/fit. Check hashes before
every fit and before scoring. All fits close before any new TEST is generated.

After closure, construct256 quartets (1024 examples) per fresh split:
IID A85002, withheld A85003, IID B85005, withheld B85006, bridge85004. All five
datasets and every per-example prediction/support/code/inspection counter are
saved. Independent audit recomputes labels, quartets, input overlap, metrics,
checkpoint replay, key separation/payload stability and bounded access.

Primary screens, every seed and every fresh split: accuracy≥95%, perfect answer
quartets≥90%, and both intended supports≥95%. Require no overflow in this task,
no more than four inspected candidates per hop, and at least10 percentage points
mean withheld gain over shuffled supervision. These are engineering screens,
not a three-seed significance or equivalence claim. Do not claim superiority for
matching a 100% conventional baseline.

On the full fresh bridge set also perform fixed diagnostic modes: one address
probe; paired collisions formed by integer-dividing every learned code by2;
and paired collisions with a deliberately content-blind smallest-occurrence
selector. Candidate cap stays two per bucket; up to two probed buckets remain.
Reversing records within each era must preserve answers and mapped support
occurrences in both normal and content-resolved paired-collision modes. These
tests isolate the selector's access to contents. They do not assign all training
gains causally to one module or authorize selecting a different runtime mode.

## Cost and interpretation

After fits and scoring finish, use complete batch1 calls including encoding and
index construction on the same32-example TRAIN panel. Warm32 calls;30 interleaved
blocks of32 calls per model and seed. Report median block means and raw samples.
Prepared-memory query timings, if reported, are separate and explicitly exclude
construction. The main cost preference remains
J = 12 ln(I/I_memnet) + ln(T/T_memnet), using the same-seed isolated complete-fit
and complete-call times. It is not a requirement to be twelve times faster.
Quality screens precede cost selection. No paid compute or remote publication.

Success would establish that this learned-key/content-selector revision repairs
the known sentence-task failure under this supervision and budget. It would not
establish new-entity generalization, byte-language-model gains, absence of all
future collision failures, or a deployment-speed advantage unless measured.
