# LRC version inventory and source provenance

Inventory date: 8 November 2026. LRC means **Learned Relational Composition**;
the separate LRR family is not included. This document describes source coverage
and the planned comparison. It contains no new training result.

## Scope of “all versions”

The experiment covers every recovered general next-byte LRC implementation,
its original relation/reference ablations, the later lexical-credit training
recipes, and the frozen-backbone utility gate. Some historical studies are
diagnostics or repeated evaluations of the same model rather than additional
architectures. The distinction is recorded below so that study identifiers are
not presented as independent model inventions.

The comparison has fourteen fresh backbone/training-recipe arms, followed by
utility, shuffled-label and random-label gate post-training on each new V4-p8
backbone. Three seeds, 2,048 backbone updates, batch four and 128 next-byte
targets per sequence are the intended common fitting budget; the executable
protocol and runner define the final frozen settings. Gate training adds work
and must have a separate resource receipt.

## Public implementation lineage

| Stage | Source | Immutable commit | Meaning |
| --- | --- | --- | --- |
| LRC-001 | [PR #49](https://github.com/FPC-effortless/cdl-attention-experiment/pull/49), `research/relational-composition-v01` | `d73e7f903fe35b629f090825178c7de42e02a6b8` | Raw-byte indexed-retrieval LM; coupled, independent and no-reference variants. |
| LRC-002 | [PR #50](https://github.com/FPC-effortless/cdl-attention-experiment/pull/50), `research/lrc-information-diagnostics-v02` | `474fba0059deb6f2f054401231d8317e5a896da5` | Supplied-endpoint and loss-weighting diagnostics; ordinary nonoracle forward unchanged. |
| V3/V4 integration | [PR #51](https://github.com/FPC-effortless/cdl-attention-experiment/pull/51), `research/lrc-v4-integration-review` | `b50e1615e16410b06af34081e83762460c0f9de3` | Dependent two-hop queries, historical paging and exact incremental execution paths. |

All three were open drafts when inventoried. This experiment starts from the
last listed commit on `research/lrc-tinyshakespeare-all-20261108`; it does not
silently modify historical results or promote the architecture.

## Fresh backbone and training-recipe coverage

All LRC rows use two layers, local kernel three, vocabulary 257 (byte+1 with
zero reserved for padding), four relation namespaces, eight addresses and a
two-candidate cap. Widths are chosen to approximately match **allocated**
parameters near 43,000. The table's labels are inventory labels; the runner's
configuration is authoritative for output identifiers.

| Inventory arm | Implementation | Width | Page bits | Allocated parameters | Distinction |
| --- | --- | ---: | ---: | ---: | --- |
| V1 coupled | `RelationalLM` | 30 | — | 43,939 | Shared read/transform relation decision; ordered operands and reference continuation. |
| V1 independent | `RelationalLM` | 30 | — | 43,939 | Separate read and transform relation decisions. |
| V1 no-reference | `RelationalLM` | 30 | — | 43,939 | Removes the carried occurrence-reference path. |
| V3 | `SerialMultiHopLM` | 26 | — | 42,139 | Second query depends on the first retrieved value; distinct first/second/transform roles. |
| V4-p16 | `PagedMultiHopLM` | 22 | 16 | 42,967 | Original page-head capacity, with width reduced for this matched comparison. |
| V4-p8 | `PagedMultiHopLM` | 24 | 8 | 42,877 | Later prose/backbone page configuration. |
| Local LRC | `Model("lrc_local")` | 24 | 8 | 42,882 | V4 memory disabled during both training and evaluation. |
| Local LRC + lexical cache | `Model("lrc_cache")` | 24 | 8 | 42,882 | Same local core plus exact suffix cache and learned mixture admission. |
| V4 lexical-address | V4-p8 plus `lexical_address` auxiliary | 24 | 8 | 42,877 | Training-only lexical write/read address supervision. |
| V4 lexical-address+slot | V4-p8 plus `lexical_address_slot` auxiliary | 24 | 8 | 42,877 | Adds causal page/slot selection supervision. |
| V4 shuffled lexical teacher | V4-p8 plus `shuffled_lexical_address_slot` | 24 | 8 | 42,877 | Matched combined auxiliary with row-rotated targets/masks. |
| GRU reference | `Model("gru")` | embedding 24, hidden 68 | — | 43,082 | One recurrent layer. |
| GRU + lexical cache | `Model("gru_cache")` | embedding 24, hidden 68 | — | 43,082 | Same GRU and identical lexical-cache mechanism. |
| Transformer reference | `Model("transformer")` | 36 | — | 43,133 | Two layers, three heads, RoPE, RMSNorm, SwiGLU and causal SDPA window 128. |

Parameter counts were checked by fresh constructor instantiation and
`sum(p.numel())`, without fitting. Inactive allocation is still counted: local
LRC retains unused historical-memory parameters, and noncache matched pairs
retain a five-parameter admission head. Allocation matching therefore does
not imply equal active parameters, equal computation, or equal latency.

This is a new size-matched comparison, not a replay of historical checkpoint
sizes. In particular, V4-p16 at width 24 would allocate 48,317 parameters;
using width 22 changes both width and page capacity relative to V4-p8. Their
comparison cannot isolate the causal effect of the page-bit count alone.

At cap two, a B-bit page head can select at most `2 * 2**B` most-recent
occurrences in a single relation/address bucket: 512 for eight bits and
131,072 for sixteen. These are bucket-addressability limits, not total
sequence-length limits or evidence of learned long-range retrieval.

The lexical cache uses exact suffix lengths 2/4/8, bounded LRU namespaces and
two retained observed continuations. Its addressing is engineered lexical
matching. A benefit from it would not establish learned LRC semantic routing.

## Utility-gate post-training coverage

`vendor/utility/gated_model.py` supplies `GatedMultiHopLM`,
`StreamingGatedMultiHop` and `LocalOnlyStream`. The full gated model allocates
42,977 parameters: the 42,877-parameter V4-p8 backbone plus two 50-parameter
gate heads. The first layer is fixed open; only the final 50-parameter head
is fitted while the new Shakespeare backbone remains frozen.

Three historical recipes are covered: signed utility, shuffled utility and
bootstrap-random utility. Training utility is the clipped difference between
next-byte losses with final-layer reads off and on. Runtime inputs contain
only causal hidden features. Same-backbone full-memory, final-layer-off and
all-memory-off evaluations distinguish useful admission from constant
abstention. The gate stage receives extra teacher passes and optimizer work;
it must not be represented as an equal-total-compute fresh backbone fit.

The historical Sherlock-trained gate/checkpoints are provenance, not new
Shakespeare results. Reusing their source while fitting a new gate on the
new training split preserves the recipe without mixing training corpora.

## Exact comparison of public and archived sources

The local late research bundles were compared against the public #51 checkout.
`source_provenance.json` records raw file SHA-256, byte lengths, Python-AST
hashes, public Git blob identities, archive origins and exact copy checks.

- Every substantive model, index and hierarchical-training module in the
  backbone, prose-credit and utility-base bundles has the same Python AST as
  #51. Each late copy merely adds one trailing blank line.
- Public `relational_composition/__init__.py` exports `Config` and
  `RelationalLM`; archived initializers are empty. Direct module imports used
  by these experiments are unaffected.
- The per-document `focused_positions` training correction is present in
  public #51 and every archived `train_hierarchical.py` inspected. No later
  semantic review fix was found missing from the public source.
- `vendor/backbone_models.py`, `vendor/auxiliary.py`, `vendor/train_aux.py`
  and `vendor/utility/gated_model.py` were verified byte-identical to their
  named historical bundle files when this inventory was written.
- The copied base modules are the exact late source bytes; the V1/V3
  `train.py`, absent from the late bundle, is copied from public #51.

### Pointer-ladder source must remain diagnostic

The pointer-ladder's modified `hierarchical_model.py` is materially different.
It adds `query_only=True`, recognizes categorical token-14/token-13 query
markers, skips other query positions, and optionally accepts oracle candidate
pools. Its SHA-256 is
`275ee975564723a2a5978dc4ca23e6a89df4336c3f982ec5fbe3221f3cf14b63`.
The ordinary archived V4 file is
`47675971077b8bf983d86c7a408c31035f347ae1950eaf25e119a4accf5c7171`.
The benchmark uses ordinary V4 semantics; it does not import the pointer
scaffold as if it were the general language model.

## Historical stages covered without fictitious LM rows

| Stage | Disposition |
| --- | --- |
| LRC-002 oracle/answer-focus diagnostics | Normal LM is covered by V1; supplied semantic endpoints are not available for arbitrary Shakespeare. |
| Fixed-bucket page-selection pilot | V4 diagnostic with supplied bucket structure, not a new general LM. |
| Pointer ladder | Categorical oracle-pool/query-mask scaffold; source difference is retained in provenance, excluded from raw-text prediction. |
| Controlled entity/version credit transfer | Unchanged V4 inference with privileged categorical teachers; applicable prose recipes are explicitly included above. |
| Cache confirmation and long-context pilot | Repeat or extend workloads for models already in the matrix. |
| NumPy inference adapter | Same checkpoint/output semantics, an execution optimization rather than another trained model. |
| Capacity audit and Lean formalization | Index/action-space or ideal operational checks, with no new neural language model. |
| `VersionedProgramEngine` | Exact executor for supplied relation maps; neither a learned parser nor a next-byte LM. |
| V5/V6 | No implementation recovered in the inspected lineage and bundles. |

Earlier LRC language comparisons used Alice/Looking-Glass, Sherlock Holmes
and fresh Gutenberg author excerpts. No inspected LRC source/report contains
a Tiny Shakespeare benchmark. In particular, historical backbone losses near
2.93–2.98 bits per byte must not be relabeled as Shakespeare results.

This inventory establishes coverage and source identity. Successful training,
runtime parity, language quality, memory benefit, efficiency and broader
relational competence remain separate measurements.
