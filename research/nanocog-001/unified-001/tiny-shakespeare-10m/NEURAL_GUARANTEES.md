# Guarantees and limits of the implemented neural model

This note concerns **the actual `cog-unified-tiny-latent-v1` implementation**, not the richer typed executor proposed in COG-UNIFIED-001. Its fixed source is [model.py](model.py), SHA-256 `f96bd967b4bc4ece9d2750a5cf1afdf1cc819a6b007e2d638af633376c7eccaa`. The default 65-character model has 9,725,682 trainable parameters, six width-400 blocks, and 16 latent slots of width 64. These implementation facts and constructed preflight checks are recorded in [review_checks.json](review_checks.json).

The arguments below are source-derived mathematics under stated assumptions. They are **not formal verification of PyTorch, floating-point kernels, learned parameters, or generated language**. The earlier Lean theorems do not certify this neural implementation. Model quality and the value of the board are separate empirical outcomes.

## 1. Assumptions and notation

Fix the model parameters, input-token stream, batch/stream ownership, and initial state. For deterministic inference, use evaluation mode and disable gradient recording. Training statements additionally require matching dropout masks on the compared prefix; the same random seed alone does not establish identical masks under different tensor shapes or chunking.

Written algebra initially uses exact real arithmetic. Implementation interpretations require finite inputs, parameters and intermediates, valid float32 recurrent/board states, supported tensor geometry, and no concurrent external mutation. `_validate_state` checks geometry, device and floating-point type; it does not enforce finiteness or restrict every supplied state tensor to float32. Arbitrary caller-provided states therefore need these additional premises.

At block level, let \(x_t\) be the current input feature and \(u_t=\operatorname{RMSNorm}(x_t)\). The stored previous feature supplies \(u_{t-1}\). Learned projections of their mixture produce

\[
a_t=0.90+0.0998\,\sigma(d_t),\qquad z_t=\tanh(v_t),
\qquad h_t=a_t\odot h_{t-1}+(1-a_t)\odot z_t.
\]

The final board receives the causal backbone features. Its query, write address, commit decision and proposed value are functions of those features, rather than of a token-to-token query/key matrix. [Implementation description](ARCHITECTURE_IMPLEMENTED.md)

## 2. Prefix causality

**Claim, under the deterministic interface above:** an output at position \(t\) depends only on the supplied initial state and tokens through \(t\).

Token embeddings, projections, normalization over feature coordinates, and gates are position-local. The previous-feature path uses only the preceding position. The affine scan at position \(t\) combines transforms with indices at most \(t\). Induction through the blocks preserves this causal dependency. Board queries use current causal features and the board before the current write. The output head is position-local again. Consequently, changing a later token cannot introduce a mathematical dependency into an earlier output.

Read-before-write does not remove the current token from the prediction: its local feature still participates in the logits for the next character. It only excludes the current proposal from the board read at that position.

This is a dependency argument, not a promise of bitwise equality across arbitrary devices, batching kernels or training dropout schedules. The existing fixed-shape suffix-replacement check observed exactly unchanged prefix logits on its fixture. [Preflight evidence](review_checks.json)

## 3. Local bounds and conditional contraction

In exact arithmetic, each coordinate has \(0.90\le a_{t,j}\le0.9998<1\) and \(|z_{t,j}|\le1\). Its update is a convex combination. Thus

\[
|h_{t,j}|\le \max\{|h_{0,j}|,1\}.
\]

In particular, the default zero initialization keeps the exact-arithmetic hidden state inside \([-1,1]^d\), for any sequence length and any finite projection weights. The argument follows by induction: a convex combination cannot exceed the larger bound of its two operands.

The stored normalized previous feature has a different bound. For RMSNorm scale \(\gamma_j\) and positive epsilon,

\[
|u_{t,j}|=
\frac{|\gamma_j x_{t,j}|}{\sqrt{d^{-1}\sum_i x_{t,i}^2+\varepsilon}}
\le\sqrt d\,|\gamma_j|.
\]

This bound is parameter-dependent and applies after the feature has been computed. It does not make arbitrary supplied initial previous-feature tensors small. Nor does it constrain learned scales uniformly throughout optimization.

Now compare two runs of **one local recurrence** with identical coefficient/innovation sequences and initial hidden states differing by \(\Delta h_0\). This requires the same block inputs and compatible initial previous-feature state. Subtracting gives

\[
\Delta h_t=\left(\bigodot_{i=1}^t a_i\right)\odot\Delta h_0,
\qquad
\|\Delta h_t\|_\infty\le0.9998^t\|\Delta h_0\|_\infty.
\]

This is **conditional local contraction**, not contraction of the complete model. A changed lower-layer state changes upper-layer inputs, gates and innovations. Learned output projections can amplify differences. Hard query decisions can be discontinuous. An untouched board slot preserves its previous difference rather than contracting it.

The same product describes the local derivative with respect to that block's initial hidden state while its inputs are fixed. It does not bound the full parameter gradient, all cross-layer Jacobians, the straight-through estimator, or the optimizer. Floating-point rounding also prevents treating the displayed real-arithmetic inequalities as machine-certified bounds.

## 4. Exact-forward latent VALUE/HOLD semantics

For finite probabilities, `hard_categorical` has a one-hot forward value; its detached subtraction supplies a surrogate derivative without changing that value. The product of the entity and relation choices selects one of \(S=16\) latent coordinates. The binary commit selects write or HOLD. Therefore the forward indicator satisfies

\[
w_{t,s}\in\{0,1\},\qquad \sum_s w_{t,s}\le1.
\]

For proposed vector \(v_t\), the board equation is

\[
M_{t,s}=(1-w_{t,s})M_{t-1,s}+w_{t,s}v_t.
\]

Let \(\tau_t(s)\) be the latest committed write to slot \(s\) through position \(t\), if any. Induction yields

\[
M_{t,s}=
\begin{cases}
v_{\tau_t(s)},&\text{if such a write exists},\\
M_{0,s},&\text{otherwise}.
\end{cases}
\]

Thus HOLD preserves every coordinate; a write replaces one coordinate and preserves the others; the latest write to the same coordinate wins. Since proposed values pass through tanh, zero-initialized exact-arithmetic board coordinates stay in \([-1,1]\). With other finite initial values, the bound is the maximum of their magnitude and one.

The read is

\[
r_t=\sum_s q_{t,s}M_{t-1,s},
\]

so it selects the latest **strictly earlier** write to the queried slot, or the supplied initial slot value if there was no such write. The code implements this timing by shifting the inclusive `post_write` scan and prepending incoming memory.

These are latent-coordinate laws. Two unrelated facts routed to the same coordinate overwrite one another in the board. The theorem does not establish correct entity identity, a useful query, or preservation of an earlier fact elsewhere in the model. For example, writing value A and then value B to the same slot leaves B there, even if the intended facts concern different entities.

For fixed proposals, differences between two board states are nonexpansive: each addressed coordinate is replaced equally, and other coordinates retain their old difference. This assumes identical routes and values; it says nothing about autoregressive runs whose different outputs lead to different later input tokens. The controlled board preflight checks observed VALUE/HOLD, pre-write reading and untouched-slot behavior, without constituting a floating-point refinement proof. [Checked fixtures](verify_training.py)

## 5. Scan algebra and numerical scope

Represent a coordinatewise affine transform by \((a,b)\), acting as \(h\mapsto a\odot h+b\). Composition is

\[
(a_2,b_2)\circ(a_1,b_1)
=(a_2\odot a_1,\ b_2+a_2\odot b_1).
\]

Associativity follows by expanding either association and applying distributivity. Both the local recurrence and the latent board have this form because their proposals are available from causal features independently of their own evolving state. This justifies the exact-arithmetic prefix calculation.

`affine_scan` implements parallel doubling, with \(O(T\log T)\) elementwise work for length \(T\), multiplied by the relevant feature dimensions and batch size. The board uses dense small-board tensors, even though forward decisions are hard. This algebra does not establish a work-optimal implementation or a speed advantage.

The scan accumulates in float32. Reassociation is not floating-point associativity, and float32 can still round, overflow or underflow. For example, in float32, \((10^{20}+(-10^{20}))+1\) and \(10^{20}+((-10^{20})+1)\) need not agree. A small discrepancy may also change a threshold or argmax near a tie, producing a larger downstream change.

The existing CPU preflight observed full-versus-incremental maximum differences of approximately \(1.31\times10^{-6}\) in logits and \(2.59\times10^{-6}\) in state, below its \(2\times10^{-5}\) tolerance. Those figures concern its tested fixture and source, not every trained checkpoint, input or device. [Numerical receipt](review_checks.json)

## 6. Fixed logical inference state versus total memory

For batch size \(B\), \(L\) layers, width \(d\), \(S\) slots and board width \(m\), the returned state contains

\[
B(2Ld+Sm)
\]

scalars: hidden and previous-feature vectors per block, plus the board. For the default one-stream float32 path this is

\[
2(6)(400)+16(64)=5{,}824\text{ scalars}
=23{,}296\text{ bytes}.
\]

This count is independent of processed sequence length. It excludes weights, logits, transient scan tensors, allocator overhead, token/output buffers, and other retained snapshots or streams.

The normal native path clones the final hidden, previous-feature and board slices. Under inference mode, these clones avoid retaining an entire prefix tensor merely through a final-slice view. **Cloning does not detach an autograd graph:** with gradient recording enabled, a returned state can still retain previous computation. `eval()` alone does not disable gradients. `detach_state` removes graph history but shares its tensor storage; callers must respect ownership. An externally supplied oversized backing-storage view, or the memory-disabled path returning the supplied memory object, needs separate accounting.

Keeping every returned state in a Python list consumes growing memory even though each state's shape is fixed. The sampler also keeps generated text for return. Thus 23,296 bytes is a tensor-state measurement under a specified usage pattern, not a total-process or serving-memory guarantee. [State and generation implementations](sample.py)

## 7. Finite capacity is not a practical context cutoff

Assume fixed parameters and that the only history-dependent information available to future queries is a state with at most \(K\) bits; count any additional history-bearing random state or external store if used. There are at most \(2^K\) distinguishable states. Arbitrary length-\(n\) strings over an alphabet of size \(V\) have \(V^n\) possibilities. Exact recall of every such string requires

\[
2^K\ge V^n,\qquad K\ge n\log_2 V.
\]

Otherwise two histories collide, yet differ at a position a query can ask about. The same state/query cannot deterministically produce both required answers. Randomization cannot provide probability-one correctness for both from the same available information.

For the default float32 tensors, 186,368 stored bits is an upper bound on their available bit patterns; accessible or useful states can be far fewer. The counting result **does not identify a practical context-length cutoff**. Structured or compressible information may remain useful far longer than arbitrary independent facts. Growing external history, emitted-text storage, or changing weights alters the retained-information premise and must be counted explicitly.

## 8. Guarantees that do not follow

- **Semantic routing:** stable coordinate numbers do not establish entity identity, paraphrase invariance or correct fact binding.
- **Useful board contribution:** setting `read_value.weight` to zero makes outputs independent of the board while leaving legal writes intact. Connected gradients in a preflight fixture do not prove useful final reliance; a selected-checkpoint intervention is separate evidence.
- **Whole-model smoothness or gradient bounds:** two slots holding opposite values can produce a finite read jump when an arbitrarily small logit perturbation switches the chosen slot. The local decay bound does not prevent that discontinuity.
- **Learning convergence:** hard forward choices use a biased straight-through derivative. Neither that estimator nor the language-model objective guarantees an optimum, correct credit assignment or semantic acquisition.
- **Universal retention:** an overwrite can destroy an old slot value; finite state cannot encode every arbitrary unbounded history exactly. Local hidden-state contraction also attenuates differences under its stated conditions.
- **Parent executor contracts:** this token path does not implement semantic COPY/LINK/DELETE, evidence provenance, versioned transactions, dependency-certified caches, or learned procedure compilation. Their earlier abstract proofs cannot be inherited by naming the latent board similarly.
- **Attention superiority:** bounded tensor state, algebraic scan structure and a trained checkpoint do not establish better language quality, total memory, latency, or training efficiency than competent alternatives.

The defensible result is a causal neural architecture with a bounded declared inference state and explicit latent update semantics, together with conditional mathematical arguments and finite implementation checks. Its learned capabilities and resource advantages must be established by the actual measurements appropriate to each claim.
