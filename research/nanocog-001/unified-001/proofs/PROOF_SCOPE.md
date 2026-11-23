# COG-UNIFIED-001: checked mathematical scope

## Status and exact boundary

**46 abstract theorem statements were checked by Lean 4.19.0.** The complete
source is `UnifiedLaws.lean`; `verification.json` binds the exact source hash,
theorem inventory, compiler version, return codes and audit logs. The source
contains no proof placeholders and declares no custom axioms. The dependency
audit uses only Lean's standard `propext`, `Quot.sound` and `Classical.choice`
foundations. “No custom axioms” does not mean “no logical foundations.”

This is a new abstract contract study. It is **not** verification of the
Python reference, PyTorch, tokenization, a trained model, learned grounding,
optimizer convergence, benchmark correctness, latency, or hardware arithmetic.
No historical model/checkpoint or protected evaluation was executed here.

## What is proved

### Exact stores and operations

The store is a mathematical function from keys to values. A `write` changes
exactly the named key, its latest overwrite wins, and writes of fixed values
to two distinct keys commute as extensional stores. Protected raw cells are
unchanged when a write targets a key outside the protected predicate.

These laws concern **supplied exact keys and values**. They do not prove that
a learned entity resolver, binder, operator selector or value writer inferred
the right key/value. The direct readout theorem explicitly requires correct
grounding, correct stored content and a correct decoder for that content.

The typed interpreter has `VALUE`, `LINK` and `deleted` records. SET resolves
to its new value; DELETE resolves to no value; LINK dereferences the current
store. COPY first resolves on the old snapshot and then writes a materialized
VALUE, refusing an unresolved source. A materialized snapshot value survives
an arbitrary later write to a different address. This includes a later write
to the source address. It does not claim the destination itself is immutable.

The bounded interpreter merges all unresolved outcomes into `none`. The
Python reference's distinctions among cycles, missing records and budget
exhaustion are more detailed and are not refined by these proofs.

### Versions, dependencies and caches

The per-key version is a mathematical natural number. A versioned write
strictly increases the target version, and no key's version decreases.
Finite machine counters require a separate overflow/reuse policy.

The global cache theorem uses `DependsOn`: the declared dependency list must
suffice for the reader on every possible store pair. The weaker local theorem
uses `LocallyDependsOn`: it need only suffice for the reader at the one
snapshot being cached. Both also require **version soundness**, meaning that
equal version identifiers denote equal payloads for the relevant records.
That is an explicit history-integrity obligation, not inferred from a hash
or counter merely being present. An additional theorem requires the reader's
epoch to remain unchanged; its mathematical epoch can stand for a correctly
encoded tuple of all relevant model/tokenizer/schema/policy versions.

For the particular deterministic, fuel-bounded LINK interpreter, dependency
completeness is itself proved: `readDependencies` lists every record actually
read in the old snapshot; if those record contents remain unchanged, the
resolver follows the same path and returns the same result in the new store.
This instantiates `LocallyDependsOn` for that interpreter. It does not prove
that an arbitrary neural selector, candidate index, recurrent hidden state or
procedural program has the same dependencies. Such readers may depend on new
competitors, absent records, model parameters, packets or hidden state.

### Composition, causality and conditional end-to-end correctness

Two fixed affine maps compose associatively over exact natural-number
arithmetic, with an identity map. This is a scalar algebraic instance of the
scan identity. It is not a floating-point associativity theorem, a speedup
theorem, or permission to precompute coefficients that depend on the same
layer's evolving state. It does not make the full relational system parallel.

A pure state fold over tokens obeys a concatenation law. Reading only a
prefix yields the same output under any replacement of the unseen suffix.
Application to a model requires that its implementation really obey that
pure causal interface and does not read hidden future information.

`run_preserves_simulation` proves a finite-run induction: related initial
actual/ideal states remain related if **every one-step transition preserves
the relation**. `simulated_run_has_same_readout` adds readout agreement on
related states. This establishes a compositional route to end-to-end
correctness without assuming the entire final run is correct. It does not
discharge the one-step semantic refinement obligation for a learned model or
the Python implementation.

### Selective changes and bounded work

A scalar residual edit preserves an output exactly when its residual is zero
there (the checked scalar arithmetic is natural-number addition). A generic
hard-routed adapter also preserves every input outside its frozen route.
Exact preservation over a declared domain follows only if the residual is
zero, or the adapter is excluded, on **all inputs in that domain**. Sampling
a finite regression panel does not establish that universal premise.

The LINK interpreter performs at most `fuel` record reads. This bounds its
logical read count and ensures finite recursion; it does not bound neural
candidate discovery, hash/index maintenance, disk transfer or wall time.

## Counterexamples that are also checked

1. A finite regression panel can agree everywhere tested while a new input
   changes. For any test range `0..limit`, a patch that changes only
   `limit + 1` passes the range and still changes another output.
2. A LINK record can retain identical bytes while its answer changes after
   its target is updated. Raw cell retention is weaker than answer retention.
3. Two operations can write different keys and still fail to commute because
   each reads what the other writes. Disjoint write destinations alone do not
   establish conflict freedom or serializability.
4. A bounded candidate reader cannot return the needed record if that key was
   omitted. An exact readout does not fix a retrieval miss.
5. Three distinct histories encoded in two Boolean states must collide.
   More generally, if two histories encode to the same state but require
   different answers to some query, a deterministic decoder cannot answer
   both correctly. The general `b >= N log2(V)` cardinality bound requires the
   separate counting derivation in the mathematical report; it is not fully
   formalized in this file.

## Claims explicitly outside the checked result

- Correct natural-language binding, alias identity, paraphrase invariance,
  learned temporal interpretation or discovery of semantic roles.
- Unlimited exact retention in finite-bit memory.
- Universal correctness or truth of an admitted fact or cited evidence.
- Complete runtime dependency discovery for a general reader.
- Deployed database atomicity, isolation, crash durability or distributed
  consistency. The Boolean commit lemmas prove branch selection only.
- A complete parallel or sublinear-cost neural architecture.
- Floating-point equality, gradients, estimator validity or convergence.
- Universal preservation after weight changes or a finite test suite.
- Quality, speed, memory, sample-efficiency, novelty or scaling superiority
  over attention, Mamba, a pointer slot or another baseline.

## Reproduce

Install the official Lean 4.19.0 toolchain. No Mathlib or external Lean package
is required. In this directory run:

```bash
python verify.py --lean /absolute/path/to/lean
```

The checker compiles `UnifiedLaws.lean`, generates and executes the exact
theorem axiom audit, rejects proof placeholders or nonstandard/custom axiom
dependencies, and updates the receipt and logs. `lean-toolchain` records the
version. `TOOLCHAIN.json` records the official download URL and the observed
archive digest used for this run. That digest identifies downloaded bytes;
it is not a claim of a separate vendor signature validation.

The local `.runtime/` folder and generated `.olean` file are tooling, not
deliverables. They must be excluded from the research package.
