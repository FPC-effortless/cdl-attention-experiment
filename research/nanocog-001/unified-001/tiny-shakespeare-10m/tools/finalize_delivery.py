#!/usr/bin/env python3
"""Finalize local Tiny Shakespeare deliverables only after completed, audited training.

This script performs no training, network access, Library call, or publication.
It refuses paused/incomplete runs and preserves existing artifact inodes/xattrs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import re
from pathlib import Path, PurePosixPath
import shutil
import shlex
import tempfile
import zipfile

RUN_NAME = "tiny10m_seed20261122"
ARCHIVE_ROOT = "COG_TINY10M"

SOURCE_REQUIRED = (
    "model.py", "train.py", "data.py", "sample.py", "serve.py", "fetch_data.py",
    "verify_training.py", "verify_pipeline.py", "postfit_audit.py",
    "ARCHITECTURE_IMPLEMENTED.md", "NEURAL_GUARANTEES.md", "REVIEW.md", "RUN_PLAN.json",
    "review_checks.json", "pipeline_checks.json", "training_pipeline_checks.json",
    "web_tester_checks.json", "trained_web_tester_intermediate.json",
    "remote_source_verification.json", "tools/tiny_train_session.py",
    "data/dataset.json", "data/tokenizer.json", "data/source.json",
)
SOURCE_OPTIONAL = ("trained_web_tester_final.json", "postfit_archive_review.json", "finalizer_file_checks.json")
RUN_REQUIRED = (
    "run_config.json", "result.json", "metrics.jsonl", "sessions.jsonl",
    "tokenizer.json", "initial_evaluation.json", "interruption_events.jsonl",
    "allocator_recovery_receipt.json", "launcher_sessions.jsonl",
    "metrics_before_interruption_21959.jsonl", "discarded_metrics_21959.jsonl",
    "postfit_audit/audit.json", "postfit_audit/samples.json", "postfit_audit/samples.txt",
)
RUN_OPTIONAL = ("postfit_audit/memory_intervention.json", "launcher_model_config.json")
CORE_SOURCES = ("model.py", "train.py", "data.py", "sample.py")
AUDITED_INPUTS_REQUIRED = {
    "run_config.json", "result.json", "metrics.jsonl", "sessions.jsonl",
    "tokenizer.json", "initial_evaluation.json", "interruption_events.jsonl",
    "allocator_recovery_receipt.json", "launcher_sessions.jsonl",
    "metrics_before_interruption_21959.jsonl", "discarded_metrics_21959.jsonl",
    "best.pt", "resume.pt",
}


def fail(message):
    raise RuntimeError(message)


def require(condition, message):
    if not condition:
        fail(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def bytes_digest(data):
    return hashlib.sha256(data).hexdigest()


def validate_audit_binding(run, project, audit):
    """Bind the passing audit to its checker and every reported immutable input."""
    checks = audit.get("checks")
    require(isinstance(checks, list) and bool(checks), "Audit check inventory is missing")
    require(all(isinstance(check, dict) and type(check.get("passed")) is bool for check in checks),
            "Audit check entries are malformed")
    passed = sum(check["passed"] is True for check in checks)
    require(type(audit.get("total")) is int and audit["total"] == len(checks)
            and type(audit.get("passed")) is int and audit["passed"] == passed,
            "Audit declared check counts do not match its actual inventory")
    require(audit.get("status") == "PASS" and passed == len(checks),
            "Audit is not wholly passing")
    require(audit.get("audit_script_sha256") == digest(project / "postfit_audit.py"),
            "Audit was produced by a different/stale audit script")
    mapping = audit.get("input_sha256")
    require(isinstance(mapping, dict) and AUDITED_INPUTS_REQUIRED <= set(mapping),
            "Audit lacks required run metadata/checkpoint input identities")
    require(not run.is_symlink(), "Run directory may not be a symlink")
    resolved_run = run.resolve()
    for relative, wanted in mapping.items():
        require(isinstance(relative, str) and bool(relative) and "\\" not in relative,
                "Audit input path must be a canonical relative POSIX path")
        logical = PurePosixPath(relative)
        require(not logical.is_absolute() and ".." not in logical.parts
                and logical.as_posix() == relative and all(":" not in part for part in logical.parts),
                "Unsafe or noncanonical audit input path")
        require(isinstance(wanted, str) and re.fullmatch(r"[0-9a-f]{64}", wanted),
                "Malformed audited input SHA-256")
        actual = run
        for component in logical.parts:
            actual = actual / component
            require(not actual.is_symlink(), "Symlink in audited input path")
        require(actual.is_file() and actual.resolve().is_relative_to(resolved_run),
                "Audited input is missing or escapes its run directory")
        require(digest(actual) == wanted, f"Audited input bytes changed: {relative}")
    require(audit.get("discarded_history_audit", {}).get("passed") is True,
            "Discarded-trajectory archive audit is missing or failed")
    return mapping


def default_workspace():
    for parent in Path(__file__).resolve().parents:
        if (parent / "cog_tiny_10m").is_dir():
            return parent
    fail("Cannot locate cog_tiny_10m; pass --workspace explicitly")


def read_xattrs(path):
    if not path.exists() or not hasattr(os, "listxattr"):
        return {}
    return {name: os.getxattr(path, name) for name in os.listxattr(path)}


def write_in_place(target, source):
    """Write bytes through an existing inode, keeping attached Library metadata."""
    require(not target.is_symlink(), f"Refusing symlink destination: {target}")
    existed = target.exists()
    old_stat = target.stat() if existed else None
    attrs = read_xattrs(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("r+b" if existed else "w+b") as writer:
        writer.seek(0)
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
        writer.truncate()
        writer.flush()
        os.fsync(writer.fileno())
    if existed:
        current = target.stat()
        require((current.st_dev, current.st_ino) == (old_stat.st_dev, old_stat.st_ino),
                f"Existing file identity changed: {target}")
        require(read_xattrs(target) == attrs, f"Extended attributes changed: {target}")
    require(digest(target) == digest(source), f"In-place write hash mismatch: {target}")
    return {"existed": existed, "inode_preserved": existed,
            "xattr_names_preserved": sorted(attrs), "bytes": target.stat().st_size}


def zip_payload(path, entries):
    """Explicit entry mapping: each value is a Path or generated UTF-8 bytes."""
    require(len(entries) == len(set(entries)), "Duplicate archive entries")
    expected = {}
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as out:
        for relative, value in sorted(entries.items()):
            require(not relative.startswith("/") and ".." not in Path(relative).parts,
                    f"Unsafe ZIP path: {relative}")
            member = f"{ARCHIVE_ROOT}/{relative}"
            if isinstance(value, bytes):
                out.writestr(member, value)
                expected[member] = bytes_digest(value)
            else:
                require(value.is_file() and not value.is_symlink(), f"Missing or linked input: {value}")
                out.write(value, member)
                expected[member] = digest(value)
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == set(expected), "ZIP entry whitelist differs")
        require(archive.testzip() is None, "ZIP CRC verification failed")
        for member, wanted in expected.items():
            h = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            require(h.hexdigest() == wanted, f"ZIP content hash mismatch: {member}")
    return {"sha256": digest(path), "bytes": path.stat().st_size,
            "members": len(expected), "member_sha256": expected}


def format_metric(value):
    return f"{value:.6f}"


def cli_command(cfg, threads, cap, allocator_env=None):
    allocator_env = allocator_env or {"MALLOC_ARENA_MAX": "2"}
    allowed = {"MALLOC_ARENA_MAX", "MALLOC_CONF", "LD_PRELOAD"}
    require(set(allocator_env) <= allowed, "Unexpected allocator environment keys")
    prefix = " ".join(f"{key}={shlex.quote(str(value))}" for key, value in sorted(allocator_env.items()))
    return (
        f"{prefix} python train.py --device cpu --precision {cfg['precision']} "
        f"--threads {threads} --batch-size {cfg['batch_size']} "
        f"--gradient-accumulation {cfg['gradient_accumulation']} "
        f"--context-length {cfg['context_length']} --max-steps {cfg['max_steps']} "
        f"--eval-every {cfg['eval_every']} --minimum-steps {cfg['minimum_steps']} "
        f"--patience {cfg['patience']} --max-wall-seconds {cap} --log-every 1 "
        f"--out-dir runs/{RUN_NAME}"
    )


def render_reports(manifest, result, audit, best, sessions, interruptions, launcher, allocator, hashes, remote):
    cfg, mc, dataset = manifest["training_config"], manifest["model_config"], manifest["dataset"]
    selected = best["val"]
    final_test = result["test"]
    initial = result["initial_evaluation"]["val"]
    uv, ut = result["unigram_validation"], result["unigram_test"]
    split = dataset["splits"]
    thread_count = sessions[-1]["threads"]
    final_argv = launcher[-1]["argv"]
    cap = final_argv[final_argv.index("--max-wall-seconds") + 1] if "--max-wall-seconds" in final_argv else "1200"
    command = cli_command(cfg, thread_count, cap, launcher[-1].get("allocator_env"))
    arena_command = cli_command(cfg, 2, "120", {"MALLOC_ARENA_MAX": "2"})
    arena_resume_command = arena_command + f" --resume runs/{RUN_NAME}/resume.pt"
    resume_command = command + f" --resume runs/{RUN_NAME}/resume.pt"
    exit137 = [event for event in interruptions if event.get("exit_code") == 137]
    optional = audit.get("memory_intervention")
    intentional = [event for event in interruptions if event.get("event") == "intentional_runtime_stop"]

    readme = f"""# COG Unified Tiny Shakespeare 10M

A trained, user-requested text-generation prototype from the COG-UNIFIED-001 study.

**{result['parameters']:,} parameters.** The validation-selected checkpoint is from update **{result['best_step']:,}**. The restored training trajectory stopped at update **{result['step']:,}** of its {cfg['max_steps']:,}-update budget, with stop reason `{result['stop_reason']}`.

This is a character-level Shakespeare-style continuation model, not an instruction-following chatbot. It has {mc['n_layers']} width-{mc['d_model']} causal recurrent blocks and a {mc['n_entities'] * mc['n_relations']}-slot factorized latent VALUE/HOLD board. No token-to-token QK attention or borrowed SSM implementation is used.

## Test the delivered checkpoint

Extract `COG_TINY10M_CODE.zip`, then put the separately supplied `COG_UNIFIED_TINY10M.pt` inside its `COG_TINY10M` folder. Open a terminal in that folder. The checkpoint embeds its model configuration and {dataset['vocab_size']}-character tokenizer.

Install PyTorch 2.10.0 if needed. The checked training and inference environment was **Python 3.12 on Linux CPU**; macOS execution was not checked.

**Linux or Windows, CPU installation:**

```sh
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
```

**macOS, using the normal package index:**

```sh
python -m pip install torch==2.10.0
```

These installation commands follow the [official PyTorch 2.10.0 platform instructions](https://pytorch.org/get-started/previous-versions/).

Start the local browser tester:

```sh
python serve.py --checkpoint COG_UNIFIED_TINY10M.pt --threads 4
```

Open **http://127.0.0.1:8765/** on the same computer. Enter a character name or dialogue, choose a length and temperature, and generate a continuation.

For an interactive terminal:

```sh
python sample.py --checkpoint COG_UNIFIED_TINY10M.pt --interactive
```

For a reproducible sample:

```sh
python sample.py --checkpoint COG_UNIFIED_TINY10M.pt --prompt "ROMEO:" --max-new-tokens 500 --temperature 0.8 --seed 42
```

Temperature 0 selects greedy output. Unsupported characters are rejected explicitly. Sampling can repeat, lose coherence, or resemble the small training corpus.

## Observed held-out results

| Model/checkpoint | Validation NLL | Validation character perplexity | Final-test NLL | Final-test character perplexity |
|---|---:|---:|---:|---:|
| Random initialization | {initial['mean_nll']:.6f} | {initial['character_perplexity']:.4f} | Not scored | Not scored |
| Training-frequency unigram | {uv['mean_nll']:.6f} | {uv['character_perplexity']:.4f} | {ut['mean_nll']:.6f} | {ut['character_perplexity']:.4f} |
| Selected COG checkpoint, update {result['best_step']} | {selected['mean_nll']:.6f} | {selected['character_perplexity']:.4f} | {final_test['mean_nll']:.6f} | {final_test['character_perplexity']:.4f} |

Validation and final test each score **{final_test['targets']:,}** next-character targets, with state reset for each nonoverlapping {cfg['context_length']}-character window. Validation selected the weights; the final test was scored after the declared training stop. Character perplexity is not directly comparable to subword perplexity.

See [RESULTS.md](RESULTS.md) for provenance, actual exposure accounting, recovery limits, runtime receipts, and post-fit checks. Generated examples are in [post-fit samples](runs/{RUN_NAME}/postfit_audit/samples.txt).

## Reproduce or recover the declared training run

The pinned corpus has {dataset['characters']:,} characters. The contiguous training/validation/test splits contain {split['train']['characters']:,}/{split['val']['characters']:,}/{split['test']['characters']:,} characters. The tokenizer uses training characters only; targets never cross a split boundary.

Fetch and verify the corpus:

```sh
python fetch_data.py
```

The following command derives its thread count and allocator environment from the final successful launcher receipt. For a fresh reproduction, choose a new output directory. If it includes `LD_PRELOAD`, first verify that the named allocator library exists on that Linux machine:

```sh
{command}
```

For continuation from a matching interrupted checkpoint pair:

```sh
{resume_command}
```

`MALLOC_ARENA_MAX=2` records the allocator setting used in successful bounded recovery. It is an observed runtime mitigation, not a guarantee against future out-of-memory termination. The environment prefix is a Linux shell form. A recorded `LD_PRELOAD` library path is machine-specific and should not be copied to an incompatible system. The earlier two-thread arena-limited recovery command was:

```sh
{arena_resume_command}
```

On Windows, ordinary CPU inference does not need these training allocator settings.

The separate `COG_UNIFIED_TINY10M_RESUME.zip` contains matching `best.pt` and `resume.pt`, optimizer/RNG state, and run evidence, under the same `COG_TINY10M` folder. Extract it alongside this source package and fetch the corpus before resuming. The inference checkpoint alone does not contain the optimizer state. A completed resume checkpoint has already reached its declared stop; extending the budget requires a separately declared continuation configuration.

## Scope

The latent board learns routing and VALUE/HOLD updates from next-character prediction. Tiny Shakespeare provides no semantic entity, support, COPY/LINK/DELETE, or transaction labels. The richer typed executor and its abstract proofs are not certified neural capabilities of this checkpoint. See [ARCHITECTURE_IMPLEMENTED.md](ARCHITECTURE_IMPLEMENTED.md).

[NEURAL_GUARANTEES.md](NEURAL_GUARANTEES.md) studies the implemented network's causal dependencies, conditional recurrence bounds, latent update semantics, and bounded logical inference state. These are source-derived structural arguments under stated arithmetic and interface assumptions; they do not prove learned semantic accuracy or verify PyTorch's numerical kernels.

This one-seed run establishes a trained, testable artifact and its measured results on this corpus. It does not establish superiority over attention, broad language competence, or general semantic memory. Optional private online notebook submission was not verified; the supported delivery is the local browser and CLI.

The source/evidence review is tracked in [PR #93](https://github.com/FPC-effortless/cdl-attention-experiment/pull/93). Weights are delivered separately.
"""

    session_lines = []
    for i, session in enumerate(sessions, 1):
        session_lines.append(
            f"| {i} | {session['resume_step']} | {session['threads']} | {session['max_wall_seconds']:g} | {session['python']} / {session['torch']} |"
        )
    wrapper_lines = []
    for i, row in enumerate([allocator] + launcher, 1):
        elapsed = row["end_unix"] - row["start_unix"]
        wrapper_lines.append(
            f"| {i}: {row.get('kind', 'runtime receipt')} | {row['return_code']} | {elapsed:.3f} | {row['max_child_rss_kib'] / (1024 * 1024):.3f} | "
            + "; ".join(f"{key}={value}" for key, value in sorted(row.get("allocator_env", {}).items())).replace("|", "\\|") + " |"
        )
    interruptions_text = "\n".join(
        f"- Exit-137 interruption {i}: last matching checkpoint **{event.get('last_matching_checkpoint_step', 'unrecorded')}**. Cause field: {event.get('cause', 'not established')}. Uncheckpointed work: {event.get('discarded_uncheckpointed_updates', 'unknown')}."
        for i, event in enumerate(exit137, 1)
    )
    intentional_text = "\n".join(
        f"- Intentional stop {i}: tool exit code **{event.get('tool_exit_code', 'unrecorded')}**, retained checkpoint **{event.get('last_retained_checkpoint_step', 'unrecorded')}**, last discarded logged update **{event.get('last_logged_discarded_step', 'unrecorded')}**. Reason: {event.get('reason', 'not recorded')}. At least **{event.get('minimum_discarded_completed_updates', 0)} completed updates** were abandoned; additional uncheckpointed work is {event.get('actual_discarded_uncheckpointed_updates', 'unknown')}. Exact original metrics: `{event.get('full_original_metrics_archive', 'unrecorded')}`; discarded tail: `{event.get('discarded_tail_file', 'unrecorded')}`."
        for i, event in enumerate(intentional, 1)
    )
    source_lines = "\n".join(f"| `{name}` | `{value}` |" for name, value in manifest["source_sha256"].items())
    memory_text = """No matched memory-removal diagnostic is included in this completed audit. No memory-benefit claim is inferred from the language loss alone."""
    if optional is not None:
        delta = optional["disabled_minus_native_nll"]
        direction = "worsened" if delta > 0 else ("improved" if delta < 0 else "did not change")
        memory_text = f"""The optional validation-only inference intervention recorded native NLL **{optional['native_cpu_replay']['mean_nll']:.6f}** and memory-disabled NLL **{optional['memory_disabled_cpu']['mean_nll']:.6f}**, a disabled-minus-native difference of **{delta:+.6f}**. Removing the trained board contribution {direction} this checkpoint's validation prediction quality in this replay.

This is an inference intervention on one selected checkpoint. It is not a retrained local-only ablation, an attention comparison, or proof of semantic binding. The final test was not used for this diagnostic. See `runs/{RUN_NAME}/postfit_audit/memory_intervention.json`."""
    selected_target_count = result["best_checkpoint_training_targets"]
    total_attempts = result["attempted_updates"]
    report = f"""# COG Unified Tiny Shakespeare 10M — completed-run results

## Outcome and selection

The run completed with status **{result['status']}** and stop reason **`{result['stop_reason']}`**. The final restored trajectory reached **{result['step']:,}** successful optimizer updates; validation selected update **{result['best_step']:,}**. The delivered inference artifact contains that selected checkpoint, not necessarily the last optimizer state.

- Actual total and trainable parameters: **{result['parameters']:,}**.
- Model identifier: `{mc['architecture_id']}`.
- Seed: **{cfg['seed']}**; precision: **{cfg['precision']}**.
- Frozen configuration: {mc['n_layers']} layers, width {mc['d_model']}, FFN multiplier {mc['ffn_mult']}, dropout {mc['dropout']}, {mc['n_entities']}×{mc['n_relations']} latent addresses, payload width {mc['memory_dim']}.
- Selected checkpoint SHA-256: `{hashes['best.pt']}`.
- Matching final resume checkpoint SHA-256: `{hashes['resume.pt']}`.
- Post-fit audit: **{audit['passed']}/{audit['total']} checks passed**.

## Measured language results

| Measurement | Mean NLL, nats/character | Bits/character | Character perplexity | Next-character accuracy | Scored targets |
|---|---:|---:|---:|---:|---:|
| Random initialization, validation | {initial['mean_nll']:.6f} | {initial['bits_per_character']:.6f} | {initial['character_perplexity']:.4f} | {100*initial['next_character_accuracy']:.3f}% | {initial['targets']:,} |
| Train-frequency unigram, validation | {uv['mean_nll']:.6f} | {uv['bits_per_character']:.6f} | {uv['character_perplexity']:.4f} | {100*uv['next_character_accuracy']:.3f}% | {uv['targets']:,} |
| Selected COG, validation | {selected['mean_nll']:.6f} | {selected['bits_per_character']:.6f} | {selected['character_perplexity']:.4f} | {100*selected['next_character_accuracy']:.3f}% | {selected['targets']:,} |
| Train-frequency unigram, final test | {ut['mean_nll']:.6f} | {ut['bits_per_character']:.6f} | {ut['character_perplexity']:.4f} | {100*ut['next_character_accuracy']:.3f}% | {ut['targets']:,} |
| Selected COG, final test | {final_test['mean_nll']:.6f} | {final_test['bits_per_character']:.6f} | {final_test['character_perplexity']:.4f} | {100*final_test['next_character_accuracy']:.3f}% | {final_test['targets']:,} |

Each split's next-character targets were scored once, using reset state in nonoverlapping {cfg['context_length']}-character windows. The partial final window is masked correctly. Native continuous-state generation has a different context policy from this fixed-window NLL measurement.

The unigram control uses only training-character frequencies. It is a weak sanity baseline, not a competitive neural comparator. No trained attention baseline or architecture ranking is established by this run. The independent audit validates the recorded final-test receipt and denominators; it does not recompute the test or use it to select another checkpoint.

## Data provenance

- Dataset: standard English Tiny Shakespeare from `{dataset['source_repository']}`.
- Pinned source commit: `{dataset['source_commit']}`.
- [Exact source text]({dataset['source_url']}).
- Corpus SHA-256: `{dataset['input_sha256']}`.
- Corpus characters/bytes: **{dataset['characters']:,} / {dataset['input_bytes']:,}**.
- Tokenizer: **{dataset['vocab_size']}** reversible characters, fitted on training text only.
- Tokenizer SHA-256: `{dataset['tokenizer_sha256']}`.
- Split policy: **{dataset['split_policy']}**.
- Training/validation/test characters: **{split['train']['characters']:,} / {split['val']['characters']:,} / {split['test']['characters']:,}**.
- Training/validation/test next-character targets available: **{split['train']['available_next_character_targets']:,} / {split['val']['available_next_character_targets']:,} / {split['test']['available_next_character_targets']:,}**.
- No normalization, unknown-character substitution, or cross-split prediction target was introduced. Held-out vocabulary inspection was only a coverage check, not tokenizer fitting.

The raw corpus is intentionally absent from the delivery archives. `fetch_data.py` retrieves the pinned bytes and verifies their hash.

## Frozen training and exposure accounting

The optimizer was AdamW with learning rate {cfg['learning_rate']}, betas {cfg['betas']}, weight decay {cfg['weight_decay']}, gradient clipping {cfg['grad_clip']}, {cfg['warmup_steps']} warmup updates, and cosine decay to {cfg['min_lr_ratio']} of peak learning rate. Each attempt uses {cfg['batch_size']} sequences × {cfg['gradient_accumulation']} accumulated microbatches × {cfg['context_length']} characters: **{cfg['effective_targets_per_step']:,} target presentations**.

- Planned maximum successful updates: **{cfg['max_steps']:,}**.
- Actual successful updates on the restored trajectory: **{result['step']:,}**.
- Recorded optimizer attempts on that trajectory: **{total_attempts:,}**.
- Skipped updates on that trajectory: **{result['skipped_updates']:,}**.
- Recorded target presentations on the restored trajectory: **{result['training_targets_seen']:,}**.
- Equivalent training-target passes: **{result['train_target_equivalent_passes']:.6f}**.
- Target presentations at the selected checkpoint: **{selected_target_count:,}**.
- Recorded optimizer-attempt time on the restored trajectory: **{result['training_seconds']:.3f} seconds** ({result['training_seconds']/60:.3f} minutes).

Presentation counts are repeated exposures, not new unique corpus tokens. Recorded optimizer-attempt time excludes validation, checkpoint I/O, initialization, and other overhead. It also does not recover uncheckpointed work discarded by process termination.

Validation is evaluated every {cfg['eval_every']} successful updates and selects the strict lowest observed NLL. The declared early-stop rule has patience {cfg['patience']}, minimum step {cfg['minimum_steps']}, and improvement threshold {cfg['minimum_improvement']}. The reported stop reason above determines whether the maximum budget or that rule ended this run.

## Interruptions, recovery, and limits of cost accounting

There are **{len(exit137)} recorded exit-137 interruptions**. The preserved evidence does not establish an exact process-specific cause for either original termination. The second receipt records system OOM activity, and the memory diagnostic reports an OOM-kill counter of 2. These observations support investigating memory pressure; they do not prove which process caused or received each OOM event.

{interruptions_text}

There are **{len(intentional)} separately recorded intentional runtime stops**. These are distinct from the exit-137 events:

{intentional_text}

The archived discarded tail preserves the abandoned logged updates, while canonical metrics were rolled back to the retained checkpoint and include a trajectory-recovery event. Discarded updates are excluded from the restored-trajectory counters above.

The independent archive audit establishes at least **{audit['accounting']['minimum_documented_discarded_completed_updates']:,} discarded completed updates**, or **{audit['accounting']['minimum_documented_discarded_target_presentations']:,} discarded target presentations**. Adding these documented discarded presentations to the restored trajectory yields a **lower bound of {audit['accounting']['total_physical_target_presentations_lower_bound']:,} physical target presentations**. This is a lower bound, not an exact total; additional unlogged work remains unknown.

Both model/optimizer state and the saved Python, Torch, and batch-sampler RNG trajectory were restored from matching snapshots. Restarts are continuations of one selected seed, not independent fitted seeds. The intentional-stop receipt gives a lower bound on its discarded completed updates; any additional work after the last observed log is **unknown**. Work discarded around the earlier unlogged kills is also **unknown**. Total physical target presentations and compute time across those discarded attempts are therefore **not established**. The displayed exposure/time counters describe the restored trajectory only.

The initial allocator-bounded recovery used `MALLOC_ARENA_MAX=2` without editing the frozen model, data, or trainer. A two-thread bounded continuation was observed before later launcher continuations; any later jemalloc configuration is reported directly from their environment receipts below. Successful return codes and measured maximum resident sizes belong to the individual receipts below; they are not a general OOM guarantee.

### Trainer session records

| Session | Resume update | CPU threads | Requested wall cap, seconds | Python / Torch |
|---|---:|---:|---:|---|
{chr(10).join(session_lines)}

### Completed wrapper receipts

| Receipt | Return code | Observed wall duration, seconds | Maximum child RSS, GiB | Recorded allocator environment |
|---|---:|---:|---:|---|
{chr(10).join(wrapper_lines)}

The wrapper durations include their observed command lifetime. They cover only these recorded invocations and must not be summed into an asserted whole-task or all-attempt compute bill. The final successful launcher receipt is included in the archive. `sessions.jsonl`, `launcher_sessions.jsonl`, `allocator_recovery_receipt.json`, `interruption_events.jsonl`, and `evidence/resume_memory_diagnosis.json` preserve the evidence.

The trainer's last-session wall-time field is **{result['session_wall_seconds']:.3f} seconds**, recorded before its final test evaluation. Use the final wrapper receipt for that command's full observed lifetime.

## Post-fit review and generation

The completed audit reports **{audit['passed']}/{audit['total']} passing checks**. It validates source/data identities, parameter count, finite weights, the matching best/resume pair, validation selection, declared stopping, restored-trajectory exposure accounting, held-out summaries, and reproducible generation.

The fixed sample prompts were **ROMEO:** and **First Citizen:**, with temperature 0.8 and seeds 42/43. Their full outputs and timing are in `postfit_audit/samples.json` and `samples.txt`. The checked sample hash is `{audit['sampling']['samples_json_sha256']}`. Samples demonstrate executable continuation; they do not certify factual or semantic understanding.

## Memory intervention

{memory_text}

## Source identities and reproduction

| Source | SHA-256 |
|---|---|
{source_lines}

All four core source files were independently fetched from published commit [`{remote['commit']}`](https://github.com/{remote['repository']}/commit/{remote['commit']}) and matched their local Git blob and SHA-256 identities. `remote_source_verification.json` preserves that read-only verification.

For local interactive use, follow [README.md](README.md). For an interrupted matching run, the recovery command is:

```sh
{resume_command}
```

The finalized resume archive contains the selected `best.pt`, final `resume.pt`, and all whitelisted run metadata/evidence. It contains no source code, credentials, or raw corpus. The source archive contains an explicit whitelist of source, reports, checks, and data metadata; it contains no checkpoint weights, credentials, or raw corpus. Neither archive creates a hosted service.

The separate [implemented neural guarantee study](NEURAL_GUARANTEES.md) analyzes prefix causality, conditional local contraction, hard VALUE/HOLD semantics, and the logical state-size bound. It distinguishes exact-real arguments from floating-point execution and excludes logits, temporary tensors, retained histories, and autograd graphs from the small state count. These structural arguments do not establish learned entity binding, useful board content, or generated-language correctness.

This is a single-seed, single-corpus exploratory language model. The exact typed semantic executor from COG-UNIFIED-001 is not fully instantiated or qualified by unannotated Shakespeare training. Attention superiority, unrestricted no-forgetting, semantic COPY/LINK, and broad language capability remain outside this result.
"""
    return readme.encode("utf-8"), report.encode("utf-8"), resume_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    workspace = (args.workspace or default_workspace()).resolve()
    project = workspace / "cog_tiny_10m"
    run = project / "runs" / RUN_NAME
    delivery = workspace / "deliverables"
    audit_path = run / "postfit_audit" / "audit.json"

    # Refuse incomplete work before any write, archive operation, or Torch import.
    require((run / "result.json").is_file(), "No result.json; training is incomplete")
    result = read_json(run / "result.json")
    require(result.get("status") == "completed" and result.get("test_evaluated") is True,
            "Training is paused/incomplete or its selected final test is missing")
    require(audit_path.is_file(), "Passing completed post-fit audit is required")
    audit = read_json(audit_path)
    audited_inputs = validate_audit_binding(run, project, audit)
    require("sampling" in audit, "Audit metadata-only stage is incomplete; wait for generation/final audit")
    manifest = read_json(run / "run_config.json")
    dataset = read_json(project / "data" / "dataset.json")
    require(manifest["dataset"] == dataset, "Dataset manifests differ")
    cfg = manifest["training_config"]
    require(cfg["max_steps"] == 2000 and cfg["precision"] == "fp32", "Unexpected planned run or precision")
    require(result["planned_max_steps"] == cfg["max_steps"], "Budget metadata mismatch")
    history = audit["discarded_history_audit"]
    accounting = audit["accounting"]
    require(accounting["minimum_documented_discarded_completed_updates"] == history["minimum_documented_discarded_completed_updates"]
            and accounting["minimum_documented_discarded_target_presentations"] == history["minimum_documented_discarded_target_presentations"]
            and accounting["total_physical_target_presentations_lower_bound"] == result["training_targets_seen"] + history["minimum_documented_discarded_target_presentations"],
            "Discarded-history accounting lower bounds are inconsistent")
    for name, wanted in manifest["source_sha256"].items():
        require(digest(project / name) == wanted, f"Frozen training source changed: {name}")
    frozen_before = {name: digest(project / name) for name in CORE_SOURCES}
    remote = read_json(project / "remote_source_verification.json")
    require({row["file"] for row in remote["files"]} == set(CORE_SOURCES)
            and all(row["match"] is True and row["sha256"] == frozen_before[row["file"]]
                    and row["remote_blob_sha"] == row["local_blob_sha"] for row in remote["files"]),
            "Published core-source verification differs")

    run_files = list(RUN_REQUIRED) + [p for p in RUN_OPTIONAL if (run / p).is_file()]
    for rel in run_files:
        require((run / rel).is_file(), f"Required run evidence missing: {rel}")
    sessions = read_jsonl(run / "sessions.jsonl")
    interruptions = read_jsonl(run / "interruption_events.jsonl")
    launcher_raw = read_jsonl(run / "launcher_sessions.jsonl")
    require(launcher_raw and launcher_raw[-1].get("event") != "launcher_start",
            "Final launcher has not written its exit receipt")
    launcher = []
    declared_logs = set()
    for row in launcher_raw:
        if "log" in row:
            log = Path(row["log"])
            log = log.resolve() if log.is_absolute() else (run / log).resolve()
            require(log.parent == run.resolve() and re.fullmatch(r"launcher_[0-9]+-[0-9]+\.log", log.name),
                    "Launcher log path is outside the agreed whitelist pattern")
            require(log.is_file() and not log.is_symlink(), "Declared launcher log missing or linked")
            declared_logs.add(log.name)
        if row.get("event") == "launcher_start":
            continue
        normalized = dict(row)
        if row.get("event") == "launcher_end":
            normalized["return_code"] = row["returncode"]
            env = row.get("allocator_environment", {})
            normalized["allocator_env"] = {k: v for k, v in env.items() if k in {"LD_PRELOAD", "MALLOC_ARENA_MAX", "MALLOC_CONF"}}
            normalized["kind"] = "launcher_end: " + str(row.get("session_token", "recorded"))
        require("return_code" in normalized and "end_unix" in normalized, "Unrecognized incomplete runtime receipt")
        launcher.append(normalized)
    run_files.extend(sorted(declared_logs))
    # Catch new evidence rather than quietly dropping it or copying an unreviewed file.
    observed_metadata = {p.relative_to(run).as_posix() for p in run.rglob("*")
                         if p.is_file() and p.suffix in (".json", ".jsonl", ".txt", ".log")}
    require(observed_metadata == set(run_files),
            "Unexpected/missing run evidence; update explicit whitelist after review: "
            + repr(sorted(observed_metadata.symmetric_difference(run_files))))
    if (run / "launcher_model_config.json").exists():
        require(read_json(run / "launcher_model_config.json") == manifest["model_config"],
                "Launcher model configuration differs from admitted run")
    for event in interruptions:
        if event.get("event") == "intentional_runtime_stop":
            for path_key, hash_key in (("full_original_metrics_archive", "full_original_metrics_sha256"),
                                       ("discarded_tail_file", "discarded_tail_sha256")):
                require(event[path_key] in run_files and digest(run / event[path_key]) == event[hash_key],
                        "Original/discarded metrics archive identity differs")
    allocator = read_json(run / "allocator_recovery_receipt.json")
    require(sessions and launcher, "Completed runtime receipts missing")
    require(launcher[-1].get("return_code") == 0
            and launcher[-1]["end_unix"] >= sessions[-1]["wall_start_unix"],
            "Final launcher has not recorded a successful completed command")
    require(bool(launcher[-1].get("allocator_env")), "Final allocator environment receipt is missing")
    require(audit["sessions"] == sessions and audit["interruption_events"] == interruptions,
            "Audit and runtime history differ")
    require(sum(e.get("exit_code") == 137 for e in interruptions) >= 2,
            "Previously recorded interruption evidence is missing")
    external_memory = workspace / "resume_memory_diagnosis.json"
    require(external_memory.is_file() and external_memory.stat().st_size < 1024 * 1024,
            "Small authorized memory-diagnosis receipt missing")
    memory_bytes = external_memory.read_bytes()
    json.loads(memory_bytes)
    if (run / "postfit_audit" / "memory_intervention.json").exists():
        require(read_json(run / "postfit_audit" / "memory_intervention.json") == audit.get("memory_intervention"),
                "Memory intervention file differs from completed audit")
    samples = read_json(run / "postfit_audit" / "samples.json")
    require(digest(run / "postfit_audit" / "samples.json") == audit["sampling"]["samples_json_sha256"]
            and digest(run / "postfit_audit" / "samples.txt") == audit["sampling"]["samples_text_sha256"],
            "Audit sample identities differ")

    import torch
    best_path, resume_path = run / "best.pt", run / "resume.pt"
    hashes = {"best.pt": digest(best_path), "resume.pt": digest(resume_path)}
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    resume = torch.load(resume_path, map_location="cpu", weights_only=True)
    require(hashes["best.pt"] == result["best_checkpoint_sha256"] == resume["best_checkpoint_sha256"]
            == audit["sampling"]["checkpoint_sha256"] == samples["checkpoint_sha256"],
            "Selected best checkpoint hash pair differs")
    require(resume["checkpoint_status"] == "resumable"
            and resume["step"] == result["step"] and resume["best_step"] == result["best_step"]
            and best["step"] == result["best_step"] <= result["step"]
            and best["val"]["mean_nll"] == result["best_val_nll"] == resume["best_val_nll"],
            "Final/selected checkpoint metadata differs")
    require(best["model_config"] == resume["model_config"] == manifest["model_config"]
            and resume["training_config"] == cfg, "Checkpoint configurations differ")
    for key in ("training_targets_seen", "attempted_updates", "skipped_updates", "training_seconds"):
        require(resume[key] == result[key], f"Restored-trajectory {key} differs")
    require(result["best_checkpoint_training_targets"] == best["training_targets_seen"],
            "Selected checkpoint exposure count differs")
    require(best["source_sha256"] == resume["source_sha256"] == manifest["source_sha256"],
            "Checkpoint source provenance differs")
    require(best["dataset_sha256"] == resume["dataset_sha256"] == dataset["input_sha256"]
            and best["tokenizer_sha256"] == resume["tokenizer_sha256"] == dataset["tokenizer_sha256"],
            "Checkpoint data/tokenizer identity differs")
    del resume
    readme, report, resume_command = render_reports(
        manifest, result, audit, best, sessions, interruptions, launcher, allocator, hashes, remote)
    del best

    metadata = {
        "status": "COMPLETED_AND_AUDITED", "study": manifest["study"],
        "selected_step": result["best_step"], "completed_step": result["step"],
        "parameters": result["parameters"], "best_checkpoint_sha256": hashes["best.pt"],
        "resume_checkpoint_sha256": hashes["resume.pt"],
        "accounting_scope": "restored trajectory; discarded uncheckpointed work/total physical compute unknown",
        "restored_trajectory_targets": result["training_targets_seen"],
        "selected_checkpoint_targets": result["best_checkpoint_training_targets"],
        "postfit_audit_sha256": digest(audit_path), "postfit_checks": [audit["passed"], audit["total"]],
        "audit_script_sha256": audit["audit_script_sha256"],
        "audited_input_sha256": audited_inputs,
        "documented_discarded_targets_lower_bound": audit["accounting"]["minimum_documented_discarded_target_presentations"],
        "physical_targets_lower_bound": audit["accounting"]["total_physical_target_presentations_lower_bound"],
        "frozen_source_sha256": manifest["source_sha256"],
        "verified_source_commit": remote["commit"],
        "corpus_sha256": dataset["input_sha256"],
        "no_network_or_library_actions": True,
    }
    metadata_bytes = (json.dumps(metadata, indent=2) + "\n").encode("utf-8")
    generated = {
        "README.md": readme, "RESULTS.md": report,
        "FINAL_DELIVERY_MANIFEST.json": metadata_bytes,
        "evidence/resume_memory_diagnosis.json": memory_bytes,
    }
    source_files = list(SOURCE_REQUIRED) + [p for p in SOURCE_OPTIONAL if (project / p).is_file()]
    for rel in source_files:
        require((project / rel).is_file(), f"Whitelisted source/evidence missing: {rel}")
    code_entries = {rel: project / rel for rel in source_files}
    code_entries.update(generated)
    run_prefix = f"runs/{RUN_NAME}/"
    run_entries = {run_prefix + rel: run / rel for rel in run_files}
    code_entries.update(run_entries)
    require(all(not name.endswith((".pt", ".bin")) and not name.endswith("input.txt")
                and "credential" not in name.lower() and "kaggle" not in name.lower()
                for name in code_entries), "Disallowed source archive entry")

    resume_readme = f"""# Matching completed COG Tiny10M resume pair

Extract this archive into the same parent folder as COG_TINY10M_CODE.zip. Both use the COG_TINY10M top folder.

The run reached update {result['step']}; validation selected update {result['best_step']}.
best.pt SHA-256: {hashes['best.pt']}
resume.pt SHA-256: {hashes['resume.pt']}

This archive contains weights and run metadata/evidence only: no source, raw corpus, or credentials.
Use the separate source package and fetch_data.py for the pinned corpus.

Recovery command for a matching interrupted snapshot:
{resume_command}

This supplied snapshot is already completed. It contains the final optimizer/RNG state but does not extend the original 2,000-update plan. Recorded exposure/time counters cover the restored trajectory. Discarded work across exit-137 terminations remains unknown.
""".encode("utf-8")
    resume_entries = dict(run_entries)
    resume_entries.update({
        run_prefix + "best.pt": best_path,
        run_prefix + "resume.pt": resume_path,
        "evidence/resume_memory_diagnosis.json": memory_bytes,
        "RESUME_README.md": resume_readme,
    })
    require(all(not name.endswith((".py", ".bin")) and not name.endswith("input.txt")
                and "credential" not in name.lower() and "kaggle" not in name.lower()
                for name in resume_entries), "Disallowed resume archive entry")

    delivery.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tiny-finalize-", dir=workspace) as temporary:
        temp = Path(temporary)
        staged_weights = temp / "COG_UNIFIED_TINY10M.pt"
        shutil.copyfile(best_path, staged_weights)
        require(digest(staged_weights) == hashes["best.pt"], "Staged inference weights differ")
        staged_resume = temp / "COG_UNIFIED_TINY10M_RESUME.zip"
        staged_code = temp / "COG_TINY10M_CODE.zip"
        resume_info = zip_payload(staged_resume, resume_entries)
        code_info = zip_payload(staged_code, code_entries)
        staged_writes = {
            delivery / staged_weights.name: staged_weights,
            delivery / staged_resume.name: staged_resume,
            delivery / staged_code.name: staged_code,
        }
        for i, (rel, content) in enumerate(generated.items()):
            path = temp / f"generated-{i}.dat"
            path.write_bytes(content)
            staged_writes[project / rel] = path
        # Stage and verify everything before changing any existing deliverable.
        require(digest(best_path) == hashes["best.pt"] and digest(resume_path) == hashes["resume.pt"]
                and read_json(run / "result.json") == result and read_json(audit_path) == audit,
                "Run changed during finalization; refusing inconsistent delivery")
        require(validate_audit_binding(run, project, audit) == audited_inputs,
                "Audited input mapping changed during staging")
        backups = {}
        original_attrs = {}
        for i, target in enumerate(staged_writes):
            require(not target.is_symlink(), f"Refusing destination symlink: {target}")
            original_attrs[target] = read_xattrs(target)
            if target.exists():
                backup = temp / f"backup-{i}.dat"
                shutil.copyfile(target, backup)
                backups[target] = backup
            else:
                backups[target] = None
        changed, identity = [], {}
        try:
            for target, source in staged_writes.items():
                changed.append(target)
                identity[str(target.relative_to(workspace))] = write_in_place(target, source)
            require(all(digest(project / name) == value for name, value in frozen_before.items()),
                    "A frozen source changed during finalization")
        except BaseException:
            for target in reversed(changed):
                if backups[target] is None:
                    target.unlink(missing_ok=True)
                else:
                    write_in_place(target, backups[target])
                    require(read_xattrs(target) == original_attrs[target],
                            f"Rollback attribute mismatch: {target}")
            raise

        receipt = {
            **metadata, "finalized_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                staged_weights.name: {"sha256": hashes["best.pt"], "bytes": staged_weights.stat().st_size},
                staged_resume.name: resume_info,
                staged_code.name: code_info,
            },
            "in_place_identity": identity,
            "reports": ["cog_tiny_10m/README.md", "cog_tiny_10m/RESULTS.md"],
            "source_helper": str(Path(__file__).resolve()),
            "source_helper_sha256": digest(Path(__file__).resolve()),
        }
        receipt_path = delivery / "tiny_final_delivery.json"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "LOCAL_DELIVERY_FINALIZED", "selected_step": result["best_step"],
                          "completed_step": result["step"], "receipt": str(receipt_path),
                          "artifacts": list(receipt["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()

