# Independent source-bundle review

**Disposition: GO_BUNDLE_SOURCE_BOUNDARY_ONLY for the exact bytes below.** No blocking bundle defect was found for the private, bounded engineering submission. This is not scientific admission, GPU-runner admission, a successful installation receipt, or authorization to change the run scope. The separate runner/admission reviewer owns the execution contract.

Reviewer: `/root/candidate_readiness`. Bundle builder/staging author: `/root`. This reviewer authored the adapter implementation and tests, but did not author the builder, dependency bootstrap, tokenizer assets or root runner. This review is independent for the packaging and bootstrap boundary; the adapter source has a separate independent reviewer.

## Exact reviewed identities

| Object | SHA-256 |
|---|---|
| `build_bundle.py` | `3f2fe8382be6a33befd28206573fea7b0dc54b276134f148f2fa6c4526857c2f` |
| Staged `kernel.py` | `c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81` |
| Embedded ZIP archive | `65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5` |
| Staged `kernel-metadata.json` | `60f6f59614fb7889326d2fd9db6328bc0f382164d6e571953ca43c4d917fafa6` |
| Embedded bootstrap requirements | `e2a15d3b282e2939ab256979eab265a9eaa77166ad56d909a07c55e2ddbfffb3` |
| Embedded root runner | `25418c51065a301461ad0e8c468e83b299da569a78b44458e1a42e2ee4be03e8` |
| Embedded tokenizer adapter | `91095ddb5fbcbec91ec5b90761ac40bd346931d66e788352b241019e07cbac0b` |
| Frozen tokenizer encoding | `587a4639dc0483709fb06ae822d11a480a83743f5d18a94e17e03abee3c03601` |

Staging location: `/workspace/scratch/7148f407b210/kaggle_submissions/adapter001/`. This disposition binds these bytes, not a future rebuilt script. Rebuilds need corresponding digest verification and review of any changed behavior.

## Checks actually performed

Only standard-library source/AST/JSON/ZIP/hash inspection and read-only local `git show` were used. The staged Python script was parsed, never executed. Its base64 archive was decoded in memory and inspected without extracting or importing model code. No native model import, dependency installation, network request, remote mutation, credential-file access or credential-value inspection occurred in this review.

The archive has **33 unique regular-file members**, **476,755 compressed bytes**, and **1,307,756 uncompressed bytes**, below the worker's 10,000,000-byte cap. The declared whole-archive SHA-256, every member's byte count and SHA-256, unique member set, regular-file type and safe relative POSIX path were independently verified. The root `BUNDLE_RECEIPT.json` manifest and staged-script hash match the inspected script.

| Member category | Count | Verification |
|---|---:|---|
| Upstream Python source | 14 | Exact bytes equal `karpathy/nanochat` at `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd` through read-only local Git object reads |
| Candidate Python source | 8 | Exact bytes equal both PR90 `1a91168b216a49543cc839153fa299a262320730` and PR88 `39be434ff160d4e6507df8d2114979ba5529caff` |
| Upstream/candidate license notices | 2 | Exact pinned Git bytes |
| Integration source/protocol/requirements | 6 | Exact current files in the integration directory, including the independently reviewed adapter/test hashes |
| Frozen tokenizer assets | 3 | Encoding, manifest and GPT-2 license only; archive hashes verified |

The tokenizer encoding is structured data with exactly the declared encoding fields and 50,256 mergeable ranks. Its manifest labels the configuration as GPT-2-compatible engineering integration, vocab size 50,257, and explicitly says it does not reproduce the absent trained 32K nanochat tokenizer. There is no training/evaluation corpus in the manifest. This packaging review verifies custody of those frozen assets; it does not independently rerun tokenizer preparation or validate language-model quality.

## Extraction and source custody

The builder reads upstream and candidate source from exact Git revisions, rather than bundling an entire working directory. Candidate source equality across the two declared heads is checked before packaging. The six local integration filenames and three tokenizer filenames are explicitly selected. The builder rejects duplicate bundle paths and writes deterministic ZIP member metadata.

The worker verifies the archive digest before reading members, requires the exact reviewed member set, caps aggregate uncompressed bytes, rejects absolute/parent-traversal paths and nonregular entries, and verifies member hashes and lengths before writing each into a newly created private temporary directory. It does not call broad `extractall`. The inspected archive contains no traversal, symlink, duplicate, normalized-path collision or unexpected member.

The member allowlist contains public source, notices, reviewed integration files and frozen public tokenizer assets. It contains no Kaggle credential file, credential directory, checkpoint, private dataset, environment file, repository metadata or unrelated workspace material. The parent reported a separate secret-value scan PASS; this reviewer did not access the credential and does not claim to have repeated that value scan.

## Dependency bootstrap

The requirement file contains only `rustbpe==0.1.0` and `tiktoken==0.11.0`, each followed exclusively by valid SHA-256 hash options (32 and 30 permitted release-artifact hashes respectively). The worker invokes pip with an explicit PyPI simple index, `--no-deps`, `--only-binary=:all:`, `--require-hashes`, a fresh target directory, a machine-readable install report, and a 120-second subprocess timeout. Missing/incompatible wheels or hash mismatches fail the bootstrap instead of fetching unpinned dependencies or building source distributions.

Torch and transitive dependencies come from the base image and are therefore not fully pinned by this bootstrap. Their actual runtime identities must remain visible in the run receipt. This review verifies that hash checking is configured; it does not assert that installation has happened or independently attest every allowed hash to a publisher release.

The staged runner installs its optional-kernel import blocker before Torch/model imports and sets Hugging Face offline flags. Frozen tokenizer loading reconstructs the encoding from the bundled JSON instead of calling its network-enabled preparation function. Thus the inspected model path does not deliberately fetch external models, data or optional Hugging Face kernels.

**Nonblocking hardening:** the pip invocation does not use `--isolated`. An inherited pip configuration or environment setting can add indexes despite an explicit primary index. The required hashes still constrain installed bytes, but a strict claim that transport can contact only PyPI would be too strong without configuration isolation. Adding `--isolated` would improve that claim. Internet-enabled Kaggle metadata is not an enforced network-domain sandbox in either case.

## Submission boundary and limitations

Staged metadata requests a private Python script, GPU enabled, TPU disabled, internet enabled, and empty dataset/kernel/competition/model-source attachments. No attachment uploads unrelated user data. The metadata file itself does not encode the requested 900-second server timeout or particular accelerator; the submission command/service receipt must establish those separately. The manifest records a 720-second core bound and requested 900-second server bound, but declarations alone are not execution evidence.

The bootstrap reports failure rather than bypassing archive or dependency checks. A later runner failure may be caught by the outer bootstrap handler and labeled `FAILED_BOOTSTRAP` even after source verification; root should interpret the retained runner receipt for the actual failed stage. This does not create a false success.

No winner, trained competence, production speed claim, protected evaluation access, model fit or scientific stage is admitted by this packaging review.
