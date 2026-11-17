# CSM-CONDITIONAL-019 prospective protocol — NOT YET ADMITTED

New study after018 was halted before endpoint inspection for source-identity nonidentifiability and a global-value-existence shortcut. Retain all018attempts, receipts and costs separately; never reuse its partial weights as019initialization. This study does not repair or reinterpret018results.

Question: can a learned predicate gate use its own predicted source's effective value reliably, and does a state-conditioned parser earn its added cost over a local parser with the same gate or an exact comparator?

## Arms
- local_read: local clause parser; learned gate takes the effective value read from OWN predicted source and OWN predicted expected value.
- state_read: same learned gate; parser also receives OWN predicted complete memory state.
- local_exact: local parser; supplied exact predicate comparison on OWN predicted operands/state. This is the competent simpler control.
The learned gate is a9→16→2MLP (one-hot source value inclunknown5 +expected4); it receives no gold source/value/state. All arms use the same supplied typed store/read, identity lexer, newline isolation and punctuation guard. No claim that the network discovered memory semantics. Existing main-head unused gate outputs are retained and counted; implementation is not optimized.

All receive identical parsing labels/exposure. For learned gates, auxiliary targets are equality of OWN predicted operands on OWN state at gold conditional-event positions. Gold positions select the loss only, never execution. Exact comparator needs no gate loss. These are different loss terms, disclosed rather than equal-optimization/FLOP claims. Parser labels separate from gate competence. Free-running discrete executor has no differentiable state gradient; no end-to-end credit-assignment claim.

## Discriminating data
512TRAIN:4 familiar structures×2wordings×64cases.1152DEV:6structures×3wordings×64cases, including2new structures and2untrained wordings. Each panel has16counterbalanced quartets (4identity rotations×4cyclic value permutations).
Each quartet contains (a) identical current clause/suffix with differing source assignments and opposite decisions and (b) identical entire precondition memory with switched source clauses requiring opposite decisions. False cases always contain expected value at another identity; global value presence cannot distinguish the branch. Every identity and expected/new value marginal is balanced. Only4value permutations and four familiar one-letter identities are covered; no arbitrary-transition, unseen-name, open-text or broad generalization claim. Both true/false cases and both pair types are reported; unknown-source cases are included.

Names/text/program overlaps checked. TRAIN/DEV identity permutation rosters differ; exposed constructed DEV is not protected TEST. No paid resources. Exact executor laws supplied; no new Lean proof claimed.

## Budget/gates
3seeds101/202/303,1024fixed AdamW updates, batch16,lr.003,clipnorm1, oneCPUthread; endpoint only, no outcome-selection. All9fits required. Each seed×DEVpanel must reach95% query accuracy, final complete map, final state, uninterrupted complete session, exact conditional decision, conditional commit precision/recall. Conditional decision must also reach95% separately on true/false cases. Paired whole-state/source-binding diagnostics reported separately. Zero new incorrect answers against corresponding local_exact is a separate preservation gate. Failed seeds remain; aggregate rescue forbidden. Fixed-budget failure is not incapacity/convergence.

## Integrity/cost
Independent review binds full source/data/runtime and operator; native Forge private freeze→TrainLab adequacy/execution→Veritas saved-record scoring→Research Lab indexing. Every64updates checkpoint includes optimizer/RNG/sampler; verify private durable archive byte roundtrip before resume. No private evidence on public GitHub; source-only backup before fits. Final independent checkpoint reload and pure-Python rescoring required. Earlier018costs remain interruption costs, not019training.
Full complete-session inference includes parsing, actual gate/conditional comparison, reads, all operations and queries. Compare all arms at same cases. Timing selects first quartet (all4cases) per18DEVpanels=72sessions, randomized interleaving,1warmup+5warm repeats,3coldprocess repeats/arm, failedattempts retained. Cost outputs checked against independently reviewed checkpoint outputs. No timing while training. Warm RSS is shared-process and cold Linux high-water includesimports; neither is tensor bytes. Quality gate first, then12:1 preference; no equalparameter/equalFLOP or optimized-serving claim.

Predeclared auxiliary endpoint diagnostic: exhaustively probe the20finite source-value/expected-value combinations in the learned predicate gate. This tests only isolated comparison on supplied operand representations, not end-to-end parsing/memory; exact comparator is an engineered reference. It is not an additional primary gate or permission to select a different checkpoint.
