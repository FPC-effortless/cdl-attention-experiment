# COG-ATTN-097e — learned actions with outcome feedback

**Status: negative qualification.** The attention-free agent *was trained*, used its own retrieval actions and one-bit outcomes, then froze before unseen evaluation. It failed to beat a generic non-neural character index. No prior correct answer, relation field, oracle candidate or target ID was supplied to the learner. This is a read-only retrieval subsystem and not complete COG-ATTN.

## Findings

- 297 authored training documents; three incomplete archive experiences per seed; three seeds 1701/1707/1711.
- 2,673 total training requests, 5,788 chosen actions, 1,073 success bits.
- 184 unique unseen evaluation requests (161 answerable, 23 absent), each repeated across three seeds.
- With no abstention: trained neural **160/552** vs. generic char-TFIDF **216/552**.
- With development-only abstention: neural accepted 47 correct / 49 wrong (49.0% precision, 17.4% coverage); char accepted 78 correct / 36 wrong (68.4%, 20.7%). Neural generated 6 wrong answers on absent evidence; char generated 0 under the selected gate.
- Top-8 char ∪ top-8 word candidate recall 135/161 on the full test archive; learned candidate choice not qualified and continuous features did not establish novel semantics.

Read `PROTOCOL.md` for leakage/learning restrictions and `REPORT.md` for interpretation, costs, failures and qualification.

## Reproduce

Requires Python 3.13.5 standard library source matching the `source_sha` manifests; NumPy 2.3.5, scikit-learn 1.8.0, PyTorch 2.10.0 CPU and pytest 9.0.2.

```bash
bash reproduce.sh
# or:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiment.py --seed 1701 --epochs 3 --out result-1701.json --weights .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit.py --folder .
```

`reproduce.sh` trains from scratch for three seeds and writes frozen models. `audit.py` reloads the models and checks each evaluation decision; `test_experiment.py` checks transaction/identity isolation, no invented missing-target success, and gradient flow. `summarize.py` verifies aggregation after the results exist.

File checksums are in `SHA256SUMS`, excluding the checksum manifest itself. The complete offline archive retains `097e-*.pt`, per-seed full test and training-feedback traces, protocol, source and checkpoint digests; a public source-only draft must disclose if that archive is not included on GitHub.

## GitHub compact evidence

`EVIDENCE_COMPACT.json.gz.b64` decodes using `python unpack_evidence.py`. It preserves each observed action/reward and every held-out decision with **20-character digest prefixes**, and links to full-source-result SHA-256 hashes. The local 6 MB ZIP additionally retains the original full-hash JSON and frozen model weights. A fresh GitHub-hosted replay, when completed, can publish regenerated full JSON/checkpoints as workflow artifacts; **until then the GitHub branch is source-plus-compact-evidence, not a byte-identical binary archive**.
