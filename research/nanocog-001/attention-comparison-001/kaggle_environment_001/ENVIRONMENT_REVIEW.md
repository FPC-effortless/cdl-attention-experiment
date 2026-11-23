# Independent pre-execution review — Kaggle environment probe

**Decision: GO for the exact bounded private environment probe described below.** This is source-only engineering clearance, not a statement that Kaggle execution has occurred or succeeded, and not scientific campaign admission.

Reviewer: `attention_architectures`, independent of the probe author (`root`).

## Reviewed identity

- `probe.py` SHA-256: `70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719`.
- `PROTOCOL.md` SHA-256: `a0823963c45e98258e7306556126c3833cc6043cdb9a8c6ce7b280c54f128378`.
- Python source parsed successfully using standard-library `ast.parse`; the probe was not imported or executed by this reviewer.

## Findings

The source only obtains selected installed-package versions and hardware/framework metadata, then creates 16 floating-point values on each visible CUDA device and checks the analytic gradient of their squared sum. It imports no research model, loads no checkpoint, accesses no dataset, performs no optimizer step, and computes no model benchmark. The tensor sizes are fixed and negligible. A failed analytic check or missing CUDA device produces a recorded failure and a nonzero exit.

The inspected source contains no credential reads, environment-variable enumeration, home-directory scanning, network calls, package installation, subprocess execution, or remote mutation. It reads its own source for the provenance digest and writes only `environment.json` in its working directory. Its success label, `PASS_ENVIRONMENT_ONLY`, and the protocol correctly restrict the interpretation to runtime availability.

The probe's gradients are an analytic framework check. They provide no evidence about nanoCOG gradient coverage, learning, relative quality, or efficiency. Selected package version lookups do not import those packages. The actual allocated GPU identity, count, memory, CUDA version and elapsed time will be observations only after execution.

## Binding execution conditions

This GO is limited to **one** submission under the user's authorized free-quota continuation: a private Python script, requested `NvidiaTeslaT4`, worker internet disabled, no attached data/model/notebook/kernel sources, and an enforced outer remote timeout of **300 seconds**. No paid allocation, upgrade, automatic retry, accelerator substitution, protected evaluation, or model fit is admitted. The timeout is an external runner requirement; this small script does not implement its own watchdog.

Before submitting, the root runner must verify that the actual Kaggle metadata/CLI arguments enforce those conditions and that the upload is a narrowly allowlisted source bundle excluding credentials and unrelated files. Credentials remain solely in the local client boundary. This reviewer has not inspected any credential, private materialization directory, or Kaggle submission metadata, and does not attest that the unobserved remote settings already hold. The quota receipt reported by root is 30 GPU hours remaining; this review does not independently refresh account quota.

Retain the returned run/version receipt, actual settings, artifact hashes and failure record. Compare the returned source digest with the reviewed digest; any mismatch requires an explanation and re-review before treating the observation as evidence for this source. A startup or runtime failure is a failed attempt, not permission to silently retry.

## Review scope

Applicable AGENTS/FPC and research protocols were read. The user's instruction to continue with free Kaggle GPU execution supersedes the older CPU-only planning restriction for this stage. Other protected-data, cost and scientific-admission boundaries remain in force.

**No source defect blocks this exact environment-only probe under the binding conditions above.** The subsequent adapter/model probe requires a separate exact-source review.
