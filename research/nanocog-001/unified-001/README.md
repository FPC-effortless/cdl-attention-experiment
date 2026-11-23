# COG-UNIFIED-001

**A unified architecture proposal, mathematical guarantee study, and executable typed-state reference.**

The design combines the reusable mechanisms behind prior COG/CSM wins without claiming that separate checkpoints or measured speedups compose automatically. It preserves the strict nanoCOG candidate boundary and the prospective comparison protocol. No unified neural model was trained or benchmarked in this work.

## Read first

- [Architecture and proposed ADR](ARCHITECTURE.md)
- [Mathematical study and guarantee ledger](MATHEMATICS.md)
- [Evidence-to-component map](evidence/mechanism_map.md)
- [Exact source manifest](evidence/source_manifest.json)
- [Independent mathematical review](review/guarantee_analysis.md)
- [Formal scope](proofs/PROOF_SCOPE.md) and [theorem inventory](proofs/THEOREMS.md)

## What has been established

The abstract Lean development checks exact state laws, causal execution, a bounded LINK dependency theorem, conditional cache and simulation laws, affine algebra, and counterexamples to overly broad guarantees. The semantic Python reference exercises COPY versus LINK, read-before-write, snapshot guards, capacity, explicit unresolved outcomes and pure typed caching. Independent review caught and repaired numeric-revision reuse across divergent forks; full immutable observations now guard dependencies.

The formal proofs do not certify the Python implementation, learned binding, neural readout, optimizer, hardware speed or universal no-forgetting. General finite-information and efficiency analysis is written mathematics where marked. Actual receipts determine check counts and exact source identities; see `VERIFICATION.json` and the component receipts.

## Reproduce the bounded checks

Use a standard Python interpreter; no torch, dataset, trained checkpoint or network is required for the semantic reference:

```sh
python reference/check_contracts.py
python review/reference_independent.py
```

Install the official pinned Lean toolchain through your normal trusted installation route, then run:

```sh
cd proofs
python verify.py --lean /absolute/path/to/lean
```

`TOOLCHAIN.json` records the official compiler release used. The large runtime itself is intentionally excluded. The independent reviewer used a fresh direct compiler invocation rather than the author's helper.

## Integration status

The reference accepts predicted typed operations as an interface. It is not a raw-language binder or a fused neural training implementation. The design specifies the shared causal backbone, native learned key/event interfaces, objectives, storage and invalidation contracts, and necessary comparative gates. The current tokenization/training/serving integration and learned capability gaps remain explicit; no protected evaluation, new fit, paid compute, deployment or scientific promotion is part of this study.

The accompanying results-index entry labels this work **FORMAL_SPECIFICATION_AND_REFERENCE_CHECKS**, not an admitted model result. Existing historical entries and negatives are preserved.
