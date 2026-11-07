# Review: learned relational composition

Date: 2026-11-07. Verdict: worth an isolated, falsifiable prototype; neither a
qualified language backbone nor an established new architecture family.

## Evidence checked

The current GitHub reports support the numerical claims in the proposal:

| Source | Verified report | What it does not establish |
| --- | --- | --- |
| [LRR-011, PR 47](https://github.com/FPC-effortless/cdl-attention-experiment/pull/47) | Regular H32 treatment: 758/1280 correct, 522 shared-address overwrites despite complete relation agreement and gold admission | General raw-language identity learning |
| [LRR-012, PR 48](https://github.com/FPC-effortless/cdl-attention-experiment/pull/48) | Primary C−A −0.546875pp; no adoption criterion met; fresh control 98.125% leaves less than the registered 3pp target | Impossibility of separation losses; the cohort had a ceiling limitation |
| [CASM-X20R, PR 26](https://github.com/FPC-effortless/cdl-attention-experiment/pull/26) | Straight-through hard competence 0/6, all-absent collapse | Failure of every possible discrete estimator |
| [CASM-X20U, PR 29](https://github.com/FPC-effortless/cdl-attention-experiment/pull/29) | Local-credit treatment 6/6 seen and strong unseen; structure-blind 0/6 | Discovery of graph structure from text; graph/candidates were supplied |

These are checks against recorded reports, not independent replications of the
fits. The inspected LRR register source at `1588b60cd2f9c8ef5ea4b2dc3c1b25b1baf779e2`
indeed uses recurrent entity/relation registers, overwrite dictionary inference,
and dense answer-position compatibility during training. This prototype does not
import that training scaffold or modify any frozen experiment. PRS and USEF-X
motivate the submitted proposal but are not claimed as source-level integrations
or independently reproduced results here.

## Required architectural corrections

1. **Occurrence identity is not entity identity.** A pointer to one Nora mention
   does not identify the Nora mention in another sentence. Use the referenced
   feature in the next address query, and measure generalization to names and
   paraphrases. Cross-mention linking remains learned and unproven. The reference
   stores `(document row, token position)`; each layer explicitly dereferences that
   occurrence's previous-layer representation. It is not a claim that records at
   different layers have identical values. Branching references are limited to
   null, previous anchor, left operand or right operand.
2. **Non-destructive storage is necessary, not sufficient.** Append-only posting
   lists preserve all occurrences. Lookup returns only the most recent C earlier
   occurrences; rank choice includes null. Older records beyond C remain stored
   but unavailable to this request. Omitted counts are exposed, never searched
   away. A finer address or different budget is needed to access them.
3. **Learn writes as well as reads.** The integer writer has no ordinary loss
   gradient through a dictionary. Include write namespace/address in the discrete
   intervention space. One changed decision triggers a full replay with identical
   Gumbel noise and recomputed downstream decisions. Holding all later decisions
   fixed would give continuation no credit: its only role is to change later
   queries. This was caught during implementation review before the pilot.
4. **State the estimator honestly.** Loss-weighted pairwise log-probability
   contrast is a local, biased policy-improvement heuristic. Uniform alternative
   proposals, clipping, one-coordinate coverage, delayed effects, and sampled
   training versus greedy evaluation are unresolved limitations. It is not an
   unbiased REINFORCE gradient and does not reproduce CASM-X20U.
5. **Make coupling interventions comparable.** All three models instantiate the
   same weights/shapes. Coupled averages two relation logit heads into one action;
   independent uses them separately. Relation conditioning affects ordered query
   generation, lookup namespace, operator and continuation. No-reference disables
   the anchor route. Allocated parameter counts match; numbers of independent
   actions and effective degrees of freedom do not, by design. The no-reference
   continuation head is inactive. Report this rather than pretending every
   parameter has the same functional role.
6. **Bounded feature reads do not imply constant total query time.** The CPU
   reference builds postings in O(N), binary-searches predecessors in O(log N),
   and gathers at most C candidates for each of two reads. Total lookup cost is
   O(LNK(log N+C)), plus neural compute and O(LNd) retained training activations.
   Address prediction costs depend on the fixed codebook sizes. Reference
   dereferences are additional O(LN) gathers and counted separately. No N×N score
   matrix or history softmax is used. Python loops, CPU transfers and scalar
   assignments preclude a GPU performance claim. Prefix decoding currently
   recomputes the whole prefix; there is no incremental decode cache.

## Prior art and novelty boundary

- [Neural Random-Access Machines](https://arxiv.org/abs/1511.06392) already learn
  pointer manipulation and dereferencing from input/output examples.
- [Neural Production Systems](https://arxiv.org/abs/2103.01937) already bind rule
  variables to entities; their published system uses attention for binding.
- [Engram](https://arxiv.org/abs/2601.07372) uses scalable lookup for static n-gram
  memory alongside a language backbone. Its efficient lookup is not this dynamic
  occurrence-and-continuation mechanism.

The narrow candidate contribution is the coupling of dynamic indexed lookup,
ordered composition and reference continuation **if** that coupling improves
held-out composition at matched budgets. This source check is not an exhaustive
novelty search, and no uniqueness, efficiency or neural superiority is asserted.

## Falsifiers and remaining gaps

If independent or no-reference performs equally well, the central coupling claim
has not earned support. If disabling/corrupting memory barely changes answers,
the model may solve the dataset through the local path. Low whole-text NLL can
reflect punctuation/grammar learning; answer NLL and exact answers are separate.
The present `disabled` and `shuffled` checks are inference interventions, not
separately trained local models or competent attention/SSM comparators.

The generator produces random relational functions, withheld relation order,
unseen name strings, a held-out query paraphrase, explicit updates and explicitly
requested old versions. Inputs are UTF-8 bytes, without entity labels, token-role
labels, graph edges or answer-position supervision. Training uses all next bytes.
This is still templated text, not natural-language evidence. Two operations may
require more than two layers when recognition/linking itself requires steps.

`final_query_fact_end_hits` is only a structural diagnostic: a correct learned
representation may live at another byte. It is not semantic retrieval recall.
Oracle routing, controlled operand-role correctness, free-running answer
generation, natural corpora, a trained local baseline, attention/SSM comparisons,
peak-memory profiling and optimized incremental decoding remain later work.
