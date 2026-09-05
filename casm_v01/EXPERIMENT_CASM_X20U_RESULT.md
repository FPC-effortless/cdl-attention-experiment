# CASM-X20U frozen scientific result

## Verdict

**VALID. Positive control PASS 6/6. `dual_credit_replication` PASS 5/6 seen and 5/6 strong unseen. `local_counterfactual_credit` PASS 6/6 seen and 6/6 strong unseen. Structure-blind local credit FAIL 0/6. Local-credit repair criterion PASS. Reuse/merge is authorized.**

X20U tested whether per-record same-target counterfactual credit resolves the two residual X20T failure modes: under-instantiation and distractor over-instantiation.

## Provenance

- frozen X20T base: `a122c447efc31054b61134b3271cbc282f167ddb`
- preregistration commit: `524b372f89e2b3fd554131782c87278c58bf0552`
- authorized executable head: `d7130eb6b1b9c8cf432ff903d8682e9a29827f1e`
- scientific workflow: `33945028131`
- contract/integrity gate: PASS
- all six train/evaluate jobs: PASS, including provenance validation and artifact upload
- Python: `3.11.16`
- PyTorch: `2.14.0+cpu`
- train seeds: `20261211..20261216`
- eval seeds: `20261291..20261296`

## Artifact digests

All downloaded artifact ZIP SHA-256 values match GitHub's recorded digests exactly.

| seed | artifact SHA-256 |
|---:|---|
| 20261211 | `78ba252c2248cfe550fd4a1a1c07c7d28b917405142bbc1fc831247a177cff7a` |
| 20261212 | `f5388a31f42165c3e36b93e2c54c0be3990279387477246f3270925bd9a4e6d0` |
| 20261213 | `c3e7bf7f84810d1f928827051ef26bf8783ccd6d74b14ffe4cc3c6084db769cd` |
| 20261214 | `47c9ed5cd054715ff1f4c46b91a6e4fc11b4a98caaacef6cd2ff55baeed85468` |
| 20261215 | `0d8a258b3be4e0fc5ec17347f2acc18d4b23e278607a197eb929f0047042460f` |
| 20261216 | `4a7617ecf2ba3282e616a52ba024d3fdd9fc4ba354bf2498577d744a241657fe` |

## Seed classification

| train seed | dual_credit_replication | local_counterfactual_credit | structure-blind local credit |
|---:|---|---|---|
| 20261211 | seen PASS; strong unseen PASS | seen PASS; strong unseen PASS | FAIL: over-instantiates all 8 candidates |
| 20261212 | seen FAIL: under-instantiates at n=3 | seen PASS; strong unseen PASS | FAIL: over-instantiates all 8 candidates |
| 20261213 | seen PASS; strong unseen PASS | seen PASS; strong unseen PASS | FAIL: over-instantiates all 8 candidates |
| 20261214 | seen PASS; strong unseen PASS | seen PASS; strong unseen PASS | FAIL: over-instantiates all 8 candidates |
| 20261215 | seen PASS; strong unseen PASS | seen PASS; strong unseen PASS | FAIL: over-instantiates all 8 candidates |
| 20261216 | seen PASS; strong unseen PASS | seen PASS; strong unseen PASS | FAIL: over-instantiates all 8 candidates |

## Regime classification

- `canonical_live_mask`: positive-control validity PASS 6/6.
- `dual_credit_replication`: 5/6 seen-competent and 5/6 strong unseen; fails the preregistered every-seed repair gate because seed `20261212` under-instantiates at `n=3`.
- `local_counterfactual_credit`: 6/6 seen competence and 6/6 strong unseen extension; all hard/raw-soft capability and selection thresholds pass on every seen and unseen cell.
- `local_counterfactual_credit_structure_blind`: 0/6; task answers can be correct, but all candidates are selected, violating precision, count error, and distractor-gate thresholds.

## Scientific conclusion

X20U supports the narrow claim that **per-record counterfactual answer credit repairs the frozen hard-forward discrete state-construction failure under the supplied graph-conditioned constructor**. The local treatment reaches the preregistered 6/6 seen plus 6/6 strong unseen gate, while the structure-blind treatment does not, so the result is not explained by a structure-blind shortcut under this contract.

The dual global hard+soft replication is not sufficient for the every-seed gate; local counterfactual credit is the qualifying treatment.

## Successor decision

The strict successor condition is satisfied. The next authorized work is **reuse/merge only**. The next branch must be created from this frozen result, and the reuse/merge experiment must be preregistered before any implementation. Lifecycle deletion, persistence, controller/program induction, and verifier-guided repair remain out of scope until reuse/merge is separately qualified.
