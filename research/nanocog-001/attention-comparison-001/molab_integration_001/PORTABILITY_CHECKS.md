# Portability verification and remaining checks

## Source checks actually performed

The author used standard-library generation, AST parsing and in-memory archive/hash inspection. The environment probe, adapter kernel and all 33 embedded archive members match the reviewed identities. `SOURCE_CHECKS.json` records current supervisor/notebook/builder hashes and byte counts. No marimo application, model, tokenizer native package or job was imported or executed by the author. No external compute, account or credential was accessed.

## Conversion decisions

| Existing boundary | MoLab handling |
|---|---|
| `runner_core.py` derives tokenizer location from `__file__` | Original source is materialized unchanged as actual files; no model code is pasted into a notebook cell |
| Core uses `signal.signal`/`SIGALRM` and process-global import hooks | Original job runs in a fresh foreground child with its own main thread; notebook handlers and already-imported Torch modules are not reused |
| Staged kernel ends with `SystemExit` | Exit belongs to the child; the notebook captures return code and evidence |
| Notebook reactivity can rerun descendants | Two separate run buttons, `mo.stop` before side effects, persistent prelaunch claim and exclusive per-stage marker |
| Stop can cancel dependent cells | Recovery/export has its own button and no dependency on execution result variables |
| `start_new_session` child escapes the notebook's process group | Supervisor explicitly terminates only the owned child group on timeout/interruption/finalization, even after leader exit |
| Kaggle server timeout is unavailable | Parent monotonic supervisor provides 300/900-second job caps with a five-second teardown reserve; original adapter core deadline remains |
| Public/unlisted MoLab defaults differ from Kaggle | Notebook and outer receipt explicitly disclaim private visibility; embedded Kaggle fields are historical source metadata |
| Scratch output may disappear | Final/local ZIP on normal failure paths, independent recovery snapshots, optional cached snapshots and explicit download |
| Packages differ across providers | Actual environment inventory first, genuine CUDA gate, unchanged hash-pinned bootstrap; no silent Torch reinstall |
| Existing staged payload was credential-free | Exact bytes are reused; only driver/locale/certificate environment variables are forwarded to the child; export uses a fixed output list |

## Checks still required before claiming a MoLab execution result

- Exact-version independent review of supervisor, notebook generation and generated notebook.
- Actual marimo parse/reactivity validation. Standard Python AST success is not proof of marimo's cell dependency semantics or persistent-cache behavior.
- Authenticated MoLab access and visible server-backed notebook operation.
- Observed sharing settings and GPU attachment, followed by a successful environment receipt.
- Controlled wrapper checks for ordinary completion, timeout and interruption, including a group whose leader exits before a descendant, if the independent reviewer requests these before real compute.
- Successful native adapter and capacity receipts on the observed MoLab runtime, or retained explicit failure. The previous CPU result is not a MoLab result.
- Artifact downloads and independent comparison of downloaded ZIP hashes/manifests with notebook output.

## Known limits

The wrapper is designed for a live interactive POSIX notebook. It is not a durable workflow service. A hard kernel/VM kill can bypass Python finalization; cache/storage lifetime is provider-controlled. Old cached owner tickets fail closed, but cache eviction cannot serve as proof that an attempt never happened. No source-only check establishes current GPU availability, package compatibility, permanent privacy or successful process cleanup on that runtime.

No proposal/source state is relabeled PASS merely because authentication or a GPU is pending. No optimizer, real training corpus, protected evaluation, scientific comparison, or quality ranking is added by the conversion.
