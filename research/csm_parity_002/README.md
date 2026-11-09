# CSM-PARITY-002

Pinned starting commit: b5ff1410905dc5dfb11a383ebb60362247899070.

This follow-up reproduces the four Python verification suites without changing
the original receipts, checks the archived PyTorch hard-forward implementation,
and adds an eager regulated adapter plus an exploratory real-text pilot.

## Reproduce

Install `requirements-lock.txt` in a Python 3.12 environment, from the CPU PyTorch
index for the CPU wheel. From the repository root:

```
python research/csm_parity_002/verify_parity.py
python research/csm_parity_002/verify_regulated.py
python research/csm_parity_002/replay.py
```

`pilot.py` reruns fitting and overwrites its local checkpoints/results; run in a
fresh checkout to preserve original evidence. The archive is read directly; its
model source is executed unchanged. No network dataset fetch is necessary.

## Verification scope

- Original package manifest: all 25 entries passed before execution.
- Reference checks: 18 zero-injection rollouts, 5,184 bound/frame steps, 54 each
  prefix/padding/reset/independent-stream cases; reproduced.
- Exact trace algebra, 14 stability examples, 64 gradient coordinates: passed.
  Fresh max gradient discrepancy 5.49e-10, versus originally recorded 4.04e-10.
- Archived baseline parity: 36 rollouts, 3,456 tokens, 13,824 categorical route
  checks; FP64 and FP32. Largest numerical difference 1.54405e-7 overall.
- Regulated FP64 parity: 27 rollouts / 2,592 tokens, all modes and trace variants,
  document resets and padding; max discrepancy 3.88578e-16.
- Added driver/injection receive nonzero task gradients; exact zero injection
  blocks driver gradients as predicted. This does not certify the straight-through
  estimator against derivatives of hard choices.
- Lean was not rerun: no Lean executable installed. Original Lean receipts remain
  intact and hash-verified; this is not a new kernel verification claim.

The original diagnostic reset preserves position. The adapter's explicit document
reset initializes all state before the token. These are distinct contracts.
Adapter flags apply to a whole batch; asynchronous stream boundaries require
separate scheduling. The scoped controller hook supports ordinary eager autograd,
not concurrent calls on the same model, compilation, or activation checkpointing.
No optimized inference timing claim is made.

## Pilot limitations

See PROTOCOL.md for prospectively fixed design. All arms contain 34,479 parameters
because the pilot uses 256 byte IDs, rather than the specification's 19-symbol
fixture. Same parameters, token schedule, initialization, optimizer and 32 updates.
Actual hardware FLOPs were not measured. Training uses 64-token truncated gradient
windows despite carrying state across 2,048 tokens. Slow-state learning can therefore
be limited by truncation. Development scoring covers only the first 1,024 targets
of the exposed split; no held-out test evaluation, significance claim, or model
selection. Checkpoints include optimizer, RNG, stream offsets and final state.

## Observed pilot results

| Variant | Mean DEV NLL | Mean trace-reset NLL |
|---|---:|---:|
| instant | 4.152728 | 4.152728 |
| single | 4.166674 | 4.148593 |
| multi | 4.163046 | 4.157991 |

Lower is better. Persistent regulation did not show a consistent advantage in this
small pilot. Resetting changes downstream routing too, so the intervention measures
the total effect of removing trace history, not an isolated direct output effect.
All nine checkpoints passed exact cold-process score replay.
Next: longer-gradient-window study with broader preregistered DEV coverage and
trace-swap controls before increasing training scale. Preserve these negative results.
