# LRC-V4 mathematical contract (Python + Lean)

**Status:** executable Python tests pass; Lean sources written, **not yet kernel-checked**.
No proof of learned multi-hop competence is claimed.

## 1. Strictly causal paged addressing

Let `w_l,j = (r_l,j, a_l,j)` be the write namespace/address for layer `l`,
position `j`. Define the newest-first eligible occurrence list

\[
E_l(t,r,a)=\operatorname{sort}_{\mathrm{descending}}\{j<t\mid w_{l,j}=(r,a)\}.
\]

For cap `C`, page number `p>=0`, and slot `s` with `0<=s<C`,

\[
P_l(t,r,a,p)=\operatorname{take}_C
   \left(\operatorname{drop}_{pC}(E_l(t,r,a))\right),\qquad
R_l(t,r,a,p,s)=\begin{cases} P_l(t,r,a,p)[s],&s<|P_l|,\\
\bot,&\text{otherwise}.\end{cases}
\]

**Causality:** `R_l(...) = j` implies `j < t`; current/future writes
cannot affect any read at `t`. **Cap:** `|P_l|<=C`.
**Key isolation:** a result must have exactly the requested namespace/address.
**Non-destructive writes:** all eligible occurrences remain in the posting list.
**Page zero compatibility:** `P_l(t,r,a,0)` equals V3's latest-C list.

**Reachability theorem:** For any eligible occurrence of reverse-chronological
rank `q` (`0<=q<m`), let `p=q//C` and `s=q%C`. Then
`q = p*C+s` and `0<=s<C`, so `R_l(t,r,a,p,s)` returns that occurrence.
This requires an address coder with enough pages: a `B`-bit coder covers all
eligible occurrences whenever `ceil(m/C) <= 2^B`. Our V4 model defaults to
`B=16` in its constructor, and benchmark instances use `B=20`; with `C=2`,
`B=20` covers up to **2,097,152 colliding records per key**. Above that,
older pages are not addressable without increasing B or using a variable-length
page code. This is an addressability property, **not retrieval accuracy**.

A query with a page outside the currently existing range returns null, rather
than aliasing a different page. Inactive high bits are masked by the bit length
of the number of existing pages. Low bits encode the page ordinal. The neural
routing policy must learn which page to request; no oracle page is given during
normal `forward` or cached inference.

Operational implementation: `relational_composition/hierarchical_index.py`.
Lean specification and lemmas: `lean/Relational/Paged.lean`.

## 2. Dependent two-hop routing

At position `t`, layer input `h`, and previous-layer anchor `b`, choose
three relation roles `(r_1, r_2, o)` and first address/slot/page:

\[
(r_1,r_2,o)=\mathrm{Route}_\theta([h;b]);\quad
(a_1,p_1,s_1)=Q_{1,\theta}([h;b;E(r_1)]).
\]

Use a typed memory read to get `j_1`; dereference `v_1=h_{j_1}` or zero on null.
Second-hop decisions depend on the **actual returned value**, not merely the
first query's control state:

\[
(a_2,p_2,s_2)=Q_{2,\theta}([h;b;v_1;E(r_2)]),\quad
j_2=R_l(t,r_2,a_2,p_2,s_2).
\]

The operator consumes ordered `v_1,v_2`, their relation embeddings and null
flags. A continuation pointer chooses among `{null, old anchor, j_1, j_2}` for
the **next layer at the same position**. Different relation namespaces for
operand 1 and operand 2 are allowed. The reference and streaming models are
in `hierarchical_model.py`; cached/bulk parity is tested separately.

## 3. Hard policy learning, selectively applied

For a page bit `b∈{0,1}` with conditional probability `p=P(b=1)`, compare two
full causal forward replays with the same non-intervened Gumbel draws:

\[
J(p)=pL_1+(1-p)L_0,\qquad
\partial_{\theta}J\big|_{L_0,L_1\ \mathrm{fixed}}
=(L_1-L_0)\,\partial_\theta p.
\]

The implementation `train_hierarchical.py` uses the surrogate
`stopgrad(L1-L0)*p`, selecting **active** page bits only. This is the exact
derivative of the **conditional two-action surrogate** holding counterfactual
losses fixed. It is **not** an unbiased gradient of the full multi-decision
training objective, and it does not solve cross-mention identity, writer
organization or long-delay credit assignment. Other multi-class actions use a
separate, biased pairwise intervention heuristic.

## 4. Parallel composition

When a **known** finite directed functional graph has successor functions
`f_r : V -> V`, an H-hop program `[r1,...,rH]` denotes
`f_rH∘...∘f_r1`. Since composition is associative, a balanced reduction has
`O(log H)` parallel dependency depth with enough processing resources, and
`O(H |V|)` total work to materialize the full composed successor map.
Applying the compiled map to `Q` input entities is a vectorized gather;
compilation becomes worthwhile only when many queries or repeated reuse
amortize construction. For one query, dependent serial O(H) chasing generally
wins. These estimates exclude model-to-graph parsing and training.

`program_runtime.py` compares both methods with the same vectorized backend,
invalidates only cached paths depending on updated relations, and protects
cached semantics from accidental caller modification of the underlying graph.
Lean proof candidates are in `lean/Relational/Programs.lean` (extends Basic).

## 5. Cost and selection, with inference weighted 12:1

For candidate `c` relative to baseline `b`, optimize only after expressivity,
causality, memory and answer-accuracy gates pass:

\[
S(c)=12\ln\left(\frac{T_{\mathrm{inference},c}}{T_{\mathrm{inference},b}}\right)
     +\ln\left(\frac{T_{\mathrm{training},c}}{T_{\mathrm{training},b}}\right).
\]

Lower is better. Ratios make the units consistent. The coefficient 12 is a
*design preference*, not a derived physical constant. Tests in
`design_selection.py` reject models that fail semantic correctness, historical
reachability, accuracy or memory limits, no matter how fast they are. **There
is no qualified V4 model yet:** learned multi-hop accuracy has not been proven.

## 6. Complexity and qualification boundary

For `L` layers, `N` positions, width `d`, cap `C` and `K=2` reads, postings
storage is O(LN), neural state storage O(LNd) plus capacity slack. Appending a
record is amortized O(1); historical page reads use an O(log N) predecessor
search and O(C) output copy, i.e. O(log N+C). The cached reference can avoid
the binary search when reading at the just-written current position. Neural
matmul costs and CPU/Python overhead remain significant. There is no N×N
attention matrix in the implementation. Recomputing a full prefix on each step
is not required by the cached engine, but prefix prefill still incurs CPU
index-building work.

Lean proofs are only partially formalized: causal and page-cap results are
encoded with proof-carrying `Hit` records, page zero and division reconstruction
lemmas are supplied, and functional-program semantics are specified. The
Python operational implementation has been differentially checked against its
bulk model. A compiler/kernel run, bridge proof between Python and Lean, and
formal statement of floating-point parity remain outstanding.
