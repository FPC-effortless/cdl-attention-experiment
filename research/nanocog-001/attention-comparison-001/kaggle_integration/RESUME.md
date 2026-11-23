# Resume the accepted Kaggle jobs

Read EXECUTION_STATUS.json and both remote receipts before taking action. These
jobs already exist; a queued job is not an absent job. Preserve their versions,
private visibility, source hashes and resource envelopes.

## Read-only status and retrieval

Use Kaggle CLI 2.2.5 or a later version verified to preserve explicit version
selection. Authenticate using the user's authorized credential in a private local
configuration directory. Never add that file to a source bundle or repository.

```bash
kaggle kernels status eweewee2/cog-attention-comparison-environment-001/1
kaggle kernels status eweewee2/cog-attention-adapter-001/1
kaggle kernels logs eweewee2/cog-attention-comparison-environment-001/1
kaggle kernels logs eweewee2/cog-attention-adapter-001/1
```

Once terminal, retrieve exact-version output into fresh stage-specific directories:

```bash
kaggle kernels output eweewee2/cog-attention-comparison-environment-001/1 -p observed/environment001
kaggle kernels output eweewee2/cog-attention-adapter-001/1 -p observed/adapter001
```

Verify remote metadata/source with version-specific pull or SDK get_kernel. The
environment source SHA-256 is
`70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719`.
The adapter source SHA-256 is
`c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81`.
Use the returned environment ref above; Kaggle derived it from the submitted title.

## Interpret actual results

Environment output must identify actual GPU names/capabilities, Torch/CUDA versions
and analytic gradient outcomes. Server `machine_shape=NvidiaTeslaT4` is the requested
shape and is not a substitute for the worker's actual hardware receipt.

Adapter output includes bootstrap_result.json, dependency_install.json,
run_source_manifest.json, adapter_results/adapter_checks.json,
adapter_results/model_provenance.json and adapter_results/run_result.json, subject
to the stage reached. Missing later files mean unrun/incomplete stages, never PASS.
Read current_stage/current_arm, scheduled arm statuses and retained exceptions.
Hash all retrieved artifacts and independently check summary claims against them.

Keep native-initialization zeros separate from the activated gradient fixture.
Keep all-three-model-resident allocator peaks separate from deployment memory.
Keep requested architecture size separate from observed parameter counts.
The protocol includes no optimizer update or trained checkpoint.

## Failures and further execution

Do not rerun a completed passing probe to hunt a more favorable number. Preserve
failed runs and queued submissions. A second adapter attempt requires a concrete
retained defect, reviewed exact correction and separate version/receipt, with at
most two adapter submissions in this continuation and 900 seconds per attempt.
The environment probe had a separate one-submission/300-second allowance.

If a failure happens before core startup, examine bootstrap evidence; if the outer
handler labels a core exception FAILED_BOOTSTRAP, use the retained core stage and
traceback for the actual failure location. The worker is not an egress sandbox;
future bootstrap hardening can add pip --isolated before a separately reviewed build.

After result review, update this continuation's status and append an outcome to
the joint RESULTS_INDEX.md. Preserve this prospective protocol and its submitted
source hashes. Do not set the scientific campaign to admitted merely because the
engineering probe passes.
