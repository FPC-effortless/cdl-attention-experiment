# LRC-REFINEMENT-001: frozen-weight address-refinement diagnostic

## 1. Status, question, and allowed conclusion

This is a prospective diagnostic of the published LRC-ROUTE-REPAIR-002
checkpoints. It changes the physical partition and probe policy around frozen
learned keys. It does not retrain a model, change the original codebook, increase
the number of logical facts, or replace the published inference default.

The question is whether a correct coarse address remains accessible when its
continuous key is partitioned more finely, especially when valid answer payloads
change while the actual reader information stays fixed. We distinguish:

1. The unrestricted ranking induced by the learned fine-key score.
2. The maximum fixed-prefix coverage permitted by the counterfactual targets.
3. The candidates actually captured by each bounded probe policy.
4. Selection after capture and the resulting dependent two-hop answers.

The primary falsifier is a loss of bounded candidate capture under matched
refinement descriptors. Such a loss must be attributed to coarse-code choice,
fine-prefix coverage/alignment, overflow, selection, or an earlier failed hop.
Prefix diversity alone is not a failure: several required prefixes may still fit
within the available probes. Results apply to the existing four-name,
two-era, 16-record sentence task. They do not establish new-identity capacity,
long-memory scaling, natural-language readiness, or a speed advantage.

## 2. Frozen starting point and environment

Use repaired confirmation checkpoints `repair_901.pt`, `repair_902.pt`, and
`repair_903.pt`, from the original completed evidence bundle. The published
repair HEAD is `3f71af6eb53827eb7eb3c74e10964f70e8e84460`; its frozen fitting
source reference is `6c3976efc2382e28731009f9637ecf31872e5c13`. Verify the
original source hashes and checkpoint hashes against the archived manifests and
receipts before constructing this diagnostic. Preserve original files and
archived results unchanged.

Use CPython 3.12.14, PyTorch 2.5.1+cpu, NumPy 2.3.5, and one CPU Torch/BLAS
thread. Record observed versions, platform, source hashes, checkpoint hashes,
and thread settings. Load every model in evaluation mode with gradients disabled.
The encoder, writer, both readers, answer decoder, and 16-by-32 codebook remain
unchanged. No optimizer, gradient update, checkpoint selection, or weight
adaptation belongs in this experiment. Check the frozen state again after the
run.

All new implementation, datasets, traces, manifests, and reports belong to this
experiment. Software qualification may use declared TRAIN/DEV or synthetic
fixtures, but the fresh cohorts below must not be scored to choose projections,
depths, thresholds, policies, or reported seeds.

## 3. Fixed datasets and unchanged split rules

Generate exactly 128 quartets, or 512 separate worlds, per cohort after the
pre-scoring freeze. Use the original generators without changes:

| Cohort | Generator | Seed | Held-out setting |
| --- | --- | ---: | --- |
| Fresh IID A | `vb.dataset` | 94201 | `False` |
| Fresh withheld A | `vb.dataset` | 94202 | `True` |
| Fresh IID B | `vb.dataset` | 94203 | `False` |
| Fresh withheld B | `vb.dataset` | 94204 | `True` |
| Fresh bridge | `bridge_dataset` | 94205 | Original bridge rule |

The same 2,560 worlds are shared across model seeds, hash seeds, descriptor
families, depths, and policies. Store the exact generated tensors and digests.
Independently validate templates, quartet edits, labels, unique facts, support
positions, and split membership; check exact input overlap with TRAIN and retain
the overlap result rather than changing the generator after seeing it.
Any nonzero cross-panel overlap stops qualification before scoring. Preserve
that failed attempt without silently reseeding. Report within-panel duplicates
separately; they are not grounds for selecting replacement worlds.

Every world contains one partner fact and one location fact for each of four
people in each of two eras. The task supplies sentence boundaries and the finite
vocabulary to all paths. The IID generator excludes the cyclic successor pair
`person == (entity + 1) % 4`; the withheld generator requires that pair. These are
held-out combinations of familiar names, not new entities or new templates.

### Location-payload quartets

For each IID or withheld quartet, only the relevant location record's value token
changes. The query, other facts, first support/value, and both gold support
occurrences are fixed. The four worlds have the four different valid location
answers. This is the primary hop-2 payload-invariance intervention.

### Bridge-payload quartets

Only the queried partner record's person value changes. The query and other
facts stay fixed, while the correct second support and final answer change. The
queried-era locations are distinct. This tests hop-1 payload invariance and
legitimate changes to the second reader after the first value changes.

Keep all four variants in separate 16-record worlds. Inserting all four versions
into one memory would introduce a different, potentially ambiguous versioning
task and is prohibited here.

## 4. Fixed refinement descriptors

Hash seeds are exactly `94101, 94102, 94103, 94104, 94105`. Depths are exactly
`0, 1, 2, 4, 8, 16, 32`. For each hash seed, sample 32 Gaussian float64 rows of
width 32 and normalize each row to unit length. Save the exact matrix and its
hash. The same matrix is shared across coarse codes and descriptor families;
depth h always uses its first h rows. The projection sequence is not selected
using model success or any evaluation data.

Let `c(k)` be the original learned coarse-code assignment and `a_l` the l-th
projection row. Refined addresses are

`address_h(k) = (c(k), (1[a_l dot k >= t[c(k), l]]) for l < h)`.

The coarse code remains part of the address. Equal fine prefixes in different
coarse codes do not share a bucket. Hash projections and threshold calculations
use float64 views of the frozen float32 keys; this is index arithmetic, not a
change to the neural model precision. Projection equality belongs to the
nonnegative side of the split.

### Descriptor family A: fixed origin

All thresholds are zero. The complete split description is independent of TRAIN
and evaluation contents, apart from the frozen representation being tested.

### Descriptor family B: fixed TRAIN median

Use the original archived TRAIN data. Deduplicate by the complete raw record
token tuple, encode those unique strings with the frozen writer, and assign each
to its original coarse code. For every coarse code and projection row, take the
median of those unique-string projections. For an even count, use the midpoint
of the two central sorted values. A coarse code with no unique TRAIN strings
uses threshold zero, and every such fallback is recorded.

These are per-coarse-code medians, fitted once from TRAIN, not recursively fitted
child medians. They are not recomputed from a quartet, test memory, occupied
leaf, evaluation label, or insertion order. Store the unique TRAIN token-string
digest, coarse-code counts, threshold arrays, fallbacks, and descriptor hashes.
Both families use the same projection matrix and nested depths.

Within any comparison, exact normals, thresholds, coarse-code assignments, and
prefix labels must be comparable. Rebuilding different adaptive trees for the
four payloads and comparing arbitrary leaf IDs would not test this hypothesis.

## 5. Bounded index and fixed probe policies

Build each world's posting index from its observed record keys without semantic
labels. Retain every occurrence. A leaf has capacity B=2: if it contains more
than two occurrences, that lookup returns no candidates and records overflow
and the withheld bucket size. It never returns a positional prefix or silently
falls back to a scan. Deduplicate probes and candidates.

### Primary policy: `two_coarse`

Use the unchanged stable ranking of the original 16 codebook scores. Take the top
two coarse codes. Within each code, probe the fine prefix computed from the
actual read key and that code's fixed thresholds. Both requested leaves are
determined before either lookup returns information. This gives at most two
dictionary probes and four candidate content scores per hop.

At depth zero, each fine prefix is empty, so this policy is the original
two-coarse-code router. Both descriptor families and all hash seeds must share
this same depth-zero behavior.

### Budget-matched alternative: `sibling`

At positive depth, take the highest-ranked original coarse code. Probe its
matching query prefix and a second prefix obtained by flipping the bit with the
smallest absolute query-to-threshold projection margin. Unit-normalized rows
make those margins comparable. Break equal-margin ties by the lowest bit index.
Choose both probes before observing occupancy or candidates. The allowance is
again two probes, capacity two per leaf, and at most four candidate scores.

At depth zero, `sibling` is explicitly a one-coarse-code, one-probe diagnostic;
there is no bit to flip. It is not the headline depth-zero baseline. No extra
coarse-code fallback or additional sibling probes are permitted.

The primary policy spends only one refined leaf on a stable gold coarse code.
The sibling policy reallocates both probes to one code. Report this distinction;
two nominal probes do not mean both policies cover two prefixes in every
potential gold code.

## 6. Selection, dense reference, and dependent execution

Use one common float32 componentwise dot-product reduction for fine-key scores
in both dense and candidate selection. This prevents subset-dependent matrix
kernel reductions from silently changing score comparisons. The original
codebook routing policy is unchanged. An exact tie at the highest fine-key
score produces a null read; no occurrence-order winner is allowed.

The dense reference scores all 16 fine keys using that same score. It uses the
same frozen reader functions, selected sentence values, decoder, tie rule, and
null-chain rule. This is an exact reference for the continuous score, not an
assumed upper bound on task accuracy: coarse-code gating can exclude a globally
higher-scoring wrong record.

Every actual two-hop execution is closed loop. The first reader sees only the
query. The second reader receives the query encoding and the value of the record
actually selected on hop 1. If hop 1 is null, abort the chain, issue no hop-2
probes/scores, and retain the zero final value. A null final read also uses zero.
The frozen answer head sees only the actual final value. No gold support,
canonical identity, relation/era metadata, answer, or oracle first value enters
this inference path.

### Mandatory depth-zero parity gate

On every fresh example and all three model seeds, `two_coarse` at depth zero
must match the native frozen repaired model's requested codes, selected records,
answer predictions, and route counters, including returned/inspected candidates,
probes, bucket sizes, overflows, and ties. Logits must match with absolute
tolerance `1e-6` and relative tolerance zero. Failure is an implementation/parity
bug and stops this run's interpretation; it must not be presented as a research
result or repaired by selecting a favorable hash seed.

## 7. Fixed-reader diagnostics and information-limited coverage

In addition to closed-loop outcomes, define reference readers from the native
coarse baseline's actual execution. Hop 1 uses its query state/key. Hop 2 uses
its actual first selected record/value and resulting reader state/key, provided
that read exists. Never substitute a gold first value. These are explicitly
fixed-reader diagnostics for localization, not alternative end-to-end scores.

For a location quartet's hop 2 and a bridge quartet's hop 1, verify equality of
the actual reference pre-routing information across the four worlds. This
includes the query, relevant first value where applicable, read key, and exact
split descriptors. Require an available reference pre-routing state at the
analyzed hop; hop 2 therefore requires a nonnull preceding first read. Do not
condition eligibility on whether retrieval at the analyzed hop succeeds. Record
eligibility and its denominator; retain ineligible worlds in actual end-to-end
metrics. If the coarse baseline makes first-selection errors, report them and
do not silently correct them with labels.

For each eligible quartet, let the four intended support records have required
refined addresses with sorted frequencies `n_1 >= n_2 >= ...`. Report

`U_K = sum of the K largest n_j / 4`, for `K = 1, 2, 4`.

This is the empirical fixed-probe coverage ceiling over the four uniformly
enumerated valid payloads. It is not actual model capture or a measured learned
policy. The audit uses gold supports only to identify the required addresses.
The bound assumes identical reader information and a fixed set of K prefix
requests chosen without memory-dependent feedback. It does not apply unchanged
to occupancy-aware routing, adaptive lookup feedback, or independently rebuilt
payload-dependent directories. Leaf overflow can lower achievable candidate
capture below this prefix-only ceiling.

Measure actual capture under these fixed reference readers separately from
closed-loop capture. Report both U_1 and U_2 so that a policy's coarse-code probe
allocation is not mistaken for a fundamental two-probe limitation.

Bridge hop 2 is a positive change control: a different retrieved partner value
may legitimately change its reader and intended second record. Do not claim an
identical-information ceiling merely because the original question is the same.

## 8. Metrics, denominators, and positive controls

Keep results identifiable by checkpoint seed, hash seed, cohort, descriptor
family, depth, and policy. Keep dense and native baseline results separately
identifiable even when reused across refinement configurations. Retain all seeds
and all cases; no favorable-hash, successful-run, or nonoverflow-only selection
is allowed.

Report, where applicable:

- Answer accuracy, both intended supports, hop-specific support retrieval, and
  perfect answer quartets for actual closed-loop execution.
- Gold-support candidate capture, dense-target candidate capture, agreement with
  exact dense selections, and correct selection conditional on gold capture.
  Keep each dense-target comparison tied to its stated reader; a dense result
  from another trajectory is not a same-reader target.
- Original coarse-code stability, required-prefix diversity, and U_1/U_2/U_4
  with explicit eligibility counts and fixed-reader capture.
- Fine-key changes under payload edits, query/target fine-key scores or ranks,
  and separation of distinct person/relation/era addresses.
- Empty probes, overflow probes and withheld occurrences, null reads, tied
  maxima, aborted chains, actual dictionary probes, returned candidates, and
  actual content scores. Keep routing and dense audit work separate.

Every real example remains in the denominator for final answers, both-support
success, and unconditional hop retrieval/capture. A hop-1 abort counts as no
hop-2 support capture or retrieval, while its actual hop-2 work is zero.
Conditional metrics must show their own denominator; an empty denominator is
undefined, not zero or perfect. A correct answer produced after a null or wrong
chain does not count as correct support retrieval.

Use an additional key-space audit of all 64 valid record templates: two eras,
four subjects, two relations, and four valid relation-specific payloads. Labels
are auditor-only. Check payload variation within logical addresses and
separation when subject, relation, or era changes. These are 64 sentence strings
representing 16 logical addresses; they are not 64 simultaneous distinct
logical records or a larger-memory benchmark. Do not infer support for inverse
queries or new query templates from this audit.

Preserve quartet structure in summaries. Three model seeds and five projection
seeds are not fifteen independent training runs. Descriptive aggregate rows
must not conceal collapsed seeds or hashes, and this experiment makes no
equivalence or general significance claim from a pooled world count.

### Fixed record-permutation check

Generate one arbitrary within-era record permutation with seed `94501`, save it
and its inverse, and verify that it is a valid permutation of each eight-record
era block and is not self-inverse. A reversal alone would not detect confusion
between a permutation and its inverse.

For every model seed and every backend/configuration, repeat inference on the
first 16 examples, or four complete quartets, of each fresh cohort using
`records_permuted = records[:, p]`. This includes the native coarse baseline,
dense reference, both descriptor families, all hash seeds/depths, and both
bounded policies. Transform gold support positions with `inverse_p[support]`.
Map nonnull selected positions back with `p[selected]`, preserving nulls as -1.
Do not use p in both directions unless its equivalence to the inverse is
actually established; this check deliberately excludes that special case.

Require unchanged answer predictions, correctly mapped selected occurrences,
and unchanged per-hop routing counters. Compare mapped candidate sets where
they are retained; occurrence enumeration order is not a semantic difference.
Keep the first-16 selection fixed regardless of outcomes. A failure is an
implementation/order-dependence defect to report and resolve under a new source
freeze, not evidence for or against the refinement learning hypothesis.

## 9. Offline reuse and work accounting

For a feasible offline diagnostic, encodings and writer/read keys may be reused
by exact raw token tuples. A first-reader cache key includes the raw query; a
second-reader cache key includes the raw query and the actually selected first
record/value. Gold identities, support labels, or answer metadata may not choose
cached inference values. Preserve the same frozen computation and validate the
depth-zero behavior against native execution.

This cache-assisted diagnostic is not an end-to-end deployment benchmark.
Report actual dictionary probes and candidate score evaluations performed in
the diagnostic separately from declared unamortized projection and codeword
work for an independently issued read. The latter describes the mathematical
query policy before offline caching; it is not a count of every operation
physically repeated by this cache-assisted sweep. Fixed-origin projections may
be shared across coarse codes; the median family uses the same projections with
code-specific thresholds. Record prefix/margin work for the sibling policy
rather than presenting four candidate scores as the entire acquisition cost.

Full-memory dense scores are computed separately for the dense reference and
auditing. Extra dense comparisons, semantic parsing, target-prefix enumeration,
and auditing are measurement work and must be labeled separately from the
bounded query path; they cannot supply an undeclared scan fallback. Reusing
projections across the depth sweep must not be presented as the deployment cost
of each independently requested depth. Record preparation and memory
construction are outside any isolated read counters. Do not claim constant
total work, latency or FLOP superiority, a latency crossover, CPU/GPU speedup,
or a new value of the historical 12:1 inference/training objective from this
experiment.

## 10. Freeze, validation, evidence, and decision rules

Before generating fresh cohorts or scoring them, save an immutable pre-scoring
manifest containing this protocol's hash, all new source hashes, original frozen
source/checkpoint hashes, observed environment/thread settings, exact generator
and hash seeds, cohort sizes, depth grid, policies, budgets, and descriptor
hashes. Descriptor preparation uses only archived TRAIN strings and fixed
random seeds. Save the descriptor values so their identity can be checked
without trusting a later RNG implementation.

Then generate and validate all five cohorts, save their tensors/digests, execute
the native parity gate and fixed evaluation matrix, retain per-example or
otherwise exactly reconstructible traces, and produce all-cell summaries.
Verify source, descriptors, and frozen checkpoint state again at completion.
An independent review must recompute denominators/counters, check trace-derived
metrics and bounds, and replay a fixed bounded sample using the saved evidence.
Preserve any failed attempt and state its reason; fixes require a new source
freeze and an explicit record that those inputs are no longer unseen.

Interpret findings as follows:

1. Dense success with poor actual capture but U_2 equal to one points first to
   query/write alignment or probe allocation. It does not establish a need for
   writer-invariance training or a two-probe impossibility.
2. Dense success with U_2 below one under verified identical information shows
   that these partitions cannot be covered perfectly by a fixed two-prefix
   policy over those payload quartets. Different partitions, writer invariance,
   or explicitly wider search become motivated follow-ups, not proven repairs.
3. Capture success with wrong selection localizes a continuous-score/resolver
   problem. Correct selected supports with wrong answers localizes a value or
   decoder problem. Earlier-hop failures must be distinguished from local
   second-hop routing failures.
4. A failure of unrestricted dense ranking is a limitation of that score; it is
   not automatically failure of the original codebook-gated addressing policy.
5. Stable coarse codes with failed fine refinement establish a compatibility
   limit for the proposed refinement, not a regression in the unchanged coarse
   router. Stability obtained by collapsing different logical addresses is not
   an acceptable repair.

This run ends with a diagnosis and preserved evidence. Any retraining, new
representation, dynamic/adaptive tree, memory-size expansion, or performance
campaign is a subsequent experiment requiring its own prospective specification.
