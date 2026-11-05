# LRR-002A — Post-result zero-parameter symbolic control (2026-11-05)

**Outcome:** Symbolic control MATCHES the neural treatment under this complete pair-label protocol. **LRR-002 does not establish neural advantage over a simpler algorithm.** This control was added after the LRR-002 confirmatory run, so its interpretation is diagnostic/posthoc, not part of the original preregistered test.

## Source and provenance

- Implementation: `relational_router/symbolic_control.py` and `test_symbolic_control.py`.
- Workflow: [run 42071049497](https://github.com/FPC-effortless/cdl-attention-experiment/actions/runs/42071049497), **success**.
- Evaluated exact head: `f0a330eb46db4975851ad9d5585e5bae87536575`.
- Artifact: `lrr-002a-symbolic-control`, ID `13436611523`.
- ZIP SHA256: `e12a0f8a6d70eec19968fe4cf5925c7c70a1b2a47d66956c02b9964e099186b5`.
- Tests: **2 passed**.
- Same 276 binary pair outcomes and the exact same six evaluation seeds
  (101–106), 32 trials per seed per context size, as LRR-002.

## Mechanism

Build connected components from positive fact/question answerability pairs
using union-find. Validate that no negative pair lies in the same component.
Assign an integer address to each connected component, and use the same
`CausalRelationIndex` as LRR-001 and LRR-002.

**Trainable parameters: 0; optimizer steps: 0; inferred components: 6.**

No relation-class labels or Transformer attention are needed, but all
pairwise answerability labels are still supplied and fully cover the
training vocabulary.

## Controlled results

| Facts H | LRR-002 balanced learned recall@1 | LRR-002A symbolic recall@1 | Symbolic median CPU query time (µs) |
|---:|---:|---:|---:|
| 8 | 1.0000 | **1.0000** | 2.574 |
| 64 | 1.0000 | **1.0000** | 3.0545 |
| 256 | 1.0000 | **1.0000** | 3.6655 |
| 1024 | 1.0000 | **1.0000** | 6.885 |

Both treatments attained 100% in all six evaluation seeds at every
context length. CPU query times are from a separate CI execution and are
descriptive, not matched-hardware comparisons. Both use dictionary lookup;
the symbolic coder also avoids running a neural MLP to resolve each
relation word at query time.

## Research interpretation

The complete positive-equivalence graph explicitly determines the answer
codes. A union-find algorithm recovers those groups exactly with no neural
learning. The neural code-use balance objective mitigated a real
optimization failure relative to the *unbalanced neural control* (positive
LRR-002 finding), but **it is not required to solve the information problem
given complete pair outcomes**.

Thus, code balancing is a validated controlled optimizer intervention, not
a demonstrated novel semantic-retrieval advantage or attention replacement.

## Decision

Preserve both original frozen LRR-001/002 results and append this
qualification; do not rewrite the original pre-registrations. Keep PRs
draft and do not promote the neural router to a language-model architecture.
The next defensible test must eliminate complete pair supervision,
generalize to unseen relation compositions and entity references in
unsegmented causal text, and compare against this union-find ceiling and
a credible attention baseline.

This is **more important** than simply increasing context length on the
current synthetic grammar, where all methods already saturate.
