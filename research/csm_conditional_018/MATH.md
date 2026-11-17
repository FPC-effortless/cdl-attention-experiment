# Conditional executor contract and proof boundary
Let M map four identities to Absent, Value(v), or Ref(j). Read follows references up to capacity; absence/cycles produce unknown. Define
C(M,d,s,e,v) = Set(M,d,v) if Read(M,s)=e, otherwise M.
The predicate uses the pre-operation state. Values are immutable integers1..4. Set replaces destination with Value(v), detaching its outgoing link.

Direct consequences of this definition:
1. False predicate: C(M,...)=M (all descriptors unchanged).
2. True predicate: destination becomes Value(v); every other descriptor unchanged.
3. Query is pure. COPY snapshots Read(source) before mutation; LINK stores source identity.
4. Descriptor preservation does NOT imply effective-value preservation for live references pointing directly/indirectly at destination. Their changed reads are required semantics.
5. Learned selection errors are outside these laws. Incorrect destination/source/predicate/gate can violate intended natural-language behavior while executor still obeys this contract.

The Python conditional path is differentially checked against a separate reference over2000 deterministic random transitions and through all18DEV panel oracle-head paths. Oracle heads only test execution, never learning. The existing MATH-008 Lean proofs are not extended by this file. No new Lean compilation or whole-neural-model proof is claimed.
