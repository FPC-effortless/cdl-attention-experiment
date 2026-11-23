# Independent review: COG Tiny Shakespeare 10M

## Scope

The user authorized an approximately ten-million-parameter model trained on the standard English Tiny Shakespeare corpus, with weights and a generation interface they can test. This is an exploratory language-model training run. It is not admission of a broad attention-replacement comparison or a claim that all COG-UNIFIED-001 mechanisms have been learned.

The architecture specification at `cog_unified_001/ARCHITECTURE.md` is the reference for the claimed interfaces. The trained checkpoint must identify which components execute in the native path, which are optional, and which remain unavailable or unqualified. Ordinary next-character text does not supply gold semantic identities, support records, COPY/LINK/DELETE labels, or preservation guarantees. Those cannot be inferred from a falling language-model loss.

## Necessary checks before the main fit

| Property | Falsifier | Current status |
|---|---|---|
| Actual model size | Count unique trainable and total parameters; report the actual count and configured width/depth/vocabulary | PASS: 9,725,682 total/trainable parameters |
| Active neural path | Execute next-token loss backward and inspect enabled parameter groups; distinguish zero-at-initialization effects from disconnected modules | PASS: every enabled parameter tensor received finite, nonzero LM-loss gradients on the constructed fixture |
| Prefix causality | Replace an unseen suffix and require unchanged earlier logits | PASS: exact unchanged tested prefix |
| Full/incremental agreement | Compare full-sequence, token-by-token and irregular chunk execution from equal state | PASS: logits and final states within declared 2e-5 absolute tolerance |
| State reset and isolation | Repeat equal prompts from fresh state; separate batch streams; prevent previous evaluation/training state leakage | PASS: fresh-state determinism, batch isolation, and supplied-state nonmutation |
| Read-before-write | Controlled record update must affect later reads rather than an earlier or same pre-write read | PASS: controlled VALUE/HOLD fixture, untouched slots, first-output write-gradient isolation |
| Valid training data | Hash exact corpus bytes, record splits, fit tokenizer only on training text, check held-out character coverage | PASS: pinned corpus, 65 training characters, exact split byte reconstruction |
| No split-crossing targets | Every sampled input and its next-token target remain within the selected split | PASS: source bounds reviewed and 4,096 position-coded sampled windows checked |
| Honest exposures | Report target presentations, unique training characters, elapsed work and repeated-corpus exposure | Source reviewed: actual microbatch presentations recorded; realized fit quantities remain a run outcome |
| Recoverable checkpoint | Restore model, configuration, tokenizer, optimizer, scheduler or schedule step, RNG and sampler state; reproduce the next update on constructed input | PASS: exact next-step parameter/loss/batch/RNG reproduction on constructed small-model fixture; best/resume pairing repair source reviewed |
| Usable generation | Reload a saved checkpoint in the user sampling entry point; deterministic seeded generation; handle unseen prompt characters explicitly | PASS: sample entrypoint reloaded and repeated seeded generation; unseen prompt character rejected |

## Fit and delivery interpretation

A successful run requires finite optimization, a recoverable checkpoint, held-out evaluation using the declared selection rule, and real generated samples. A best-validation checkpoint should be chosen without optimizing against final test text. Training and validation curves are evidence of this fit, not a population comparison. If budget or early stopping ends the run, record that actual stopping reason and actual work.

The final report must not call a proof of an abstract typed executor a proof of this neural implementation. Finite constructed checks establish their tested behavior only. In particular, numerical agreement has an explicit tolerance, and near-tied hard routes can amplify floating-point differences. Any real fit discovered to rely on future targets, bypassed memory, disconnected claimed heads, or an incompatible generation path must be repaired or relabeled before delivery.

## Independent execution results

**Exploratory fp32 fit admission: PASS.** This admission covers the user's requested prototype training and testing, not scientific promotion of an attention replacement.

- `review_checks.json`: 13/13 constructed neural-interface checks passed on the full 9,725,682-parameter model, using CPU float32, two threads, and seed 77129. No optimizer step was performed in that script.
- `pipeline_checks.json`: 9/9 independent pipeline checks passed. It reconstructed all split bytes, checked sampler/evaluator boundaries, and executed small-model optimizer continuations on an artificial integer sequence to verify exact checkpoint recovery. No Shakespeare targets were optimized in those checks.
- Both receipts bind exact inspected source hashes. `verify_training.py` and `verify_pipeline.py` are reproducible independent checking entry points.

The neural source SHA-256 checked was `f96bd967b4bc4ece9d2750a5cf1afdf1cc819a6b007e2d638af633376c7eccaa`. The corpus SHA-256 was `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`, with 1,003,854 training characters and 55,770 characters in each held-out split. The 90/5/5 contiguous split supplies 1,003,853 training targets and 55,769 targets per held-out split; these are source-defined text sections rather than independent random samples.

## Review findings and repairs

Source review found that initial resume handling checked only whether `best.pt` existed. A wrong, copied, or newer best checkpoint could then disagree with the saved resume selection metadata. The trainer now binds the exact best-file SHA-256 and checks its source, data, tokenizer, model configuration, selected step, and validation loss against the resume checkpoint. An inconsistent pair is rejected with an instruction to restore matching files; it is not silently reconciled. This repair was source reviewed. The independent executable recovery test separately verifies model, optimizer, schedule step, dropout randomness, Python/Torch randomness, and sampled batches on a valid constructed checkpoint.

The first fp16 overflow path would have aborted before GradScaler could lower its scale. The trainer added explicit skipped-update accounting and a bounded abort policy. The selected actual run is fp32; this review does not qualify the alternate mixed-precision CUDA path. CPU arithmetic checks also do not establish full-sequence/incremental parity on every accelerator or every near-tied hard route. The actual remote environment must record its software/hardware and successful finite execution.

The architecture note correctly identifies the native path as a causal language backbone with factorized latent categorical addresses and exact-forward VALUE/HOLD replacement. Those slot names are not a discovered semantic ontology. The typed semantic COPY/LINK/DELETE executor, evidence archive, external index, dependency cache, support-label objectives, separate broker, and compiled procedures are not represented as capabilities learned by this text-only run.

## Remaining outcome checks

The final checkpoint, held-out loss, actual training duration, generated samples, artifact hashes, and completed/paused stopping reason remain to be checked after the fit. Those are run outcomes and must not be marked complete by this preflight review. No trained quality, long-range semantic retention, or superiority claim follows from these checks.
