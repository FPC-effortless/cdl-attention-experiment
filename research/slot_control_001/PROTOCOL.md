# SLOT-CONTROL-001 — equally supervised fixed-budget calibration

Frozen prospectively before new fits and before reading CHANGING-STATE-001 neural
predictions. This is an exploratory process-supervised control, not a qualified
architecture campaign. No paid/GPU compute. Three new small CPU fits.

Question: does a conventional learned slot store solve the same changing-SET
problem with the same semantic controller labels, without CSM's operators,
pointers and content-dependent controller? If it matches the CSM's reliability,
the task does not establish a need for those extra mechanisms.

Model: 8 slots x32, 19-token embedding, identical three-token frontend shape,
learned destination/edit/gain, a 32->208->32 tanh proposal and linear answer head.
Hard destination/edit forward, straight-through softmax derivatives, identical
convex replacement formula. No learned operator, link traversal or content-based
controller. All weights learned; no parser, gold destinations or gold write masks
at inference. Same physical-slot labels during training as supervised CSM.
18,254 allocated parameters versus CSM 18,294 (40 fewer, 0.219%).
Initializations differ because architectures differ; do not claim paired weights.

Use exactly the public CSM-ADDRESS-002 TRAIN bank and seed-specific index schedules:
seeds101/202/303, 256 AdamW steps, batch32, lr0.002, decay0.01, clip1.0,
answer+destination+admission+0.25 gain losses. NumPy/Torch seeds fixed.
No tuning, checkpoint selection or extension. Save all 256-step training curves,
checkpoints, schedules' hashes, provenance and failures. Final checkpoint only.
All three fits close before ANY new development scoring in either experiment.
No old held-out scores used for selection and no protected TEST opened.

Evaluate on the unchanged CHANGING-STATE-001 worlds and exact metrics. Share
labels and supervision INFORMATION, not merely parameter counts. Count all
exposures and compare schedule hashes with original CSM receipts. The historical
Transformer remains answer-only and cannot establish an equally informed win.
Do not infer a training-efficiency advantage from runtime on different hosts.

Resource bound: one CPU thread, three fits x256 steps, no retries after model
results without a new recorded protocol. A numerical failure is retained.
Runtime versions Torch2.5.1+cpu/NumPy2.3.5, matching historical dependencies.
Unit and independent pre-fit review required. Check model never accepts labels;
check causal prefix, untouched slot preservation when admission is off, parameter
count, forward parity between sequence and step, and closure/schedule provenance.

Exploratory timing (only if both learned models meet CHANGING-STATE continuation
screens): original uninstrumented RecordRewrite vs this slot model, common step
loop, CPU one thread, batch1, inference_mode; whole fresh query session includes
public event encoding/tensor construction, fresh state, every token step and
argmax. Exclude imports/checkpoint loading; include no unmeasured retry/restore
benefits. 8 predeclared block representatives (shift0), regimes both, depths1/16/256,
3 seeds, one warmup each model/cell, then three repetitions alternating model
order. Save every duration and correctness; report per-seed/depth/regime median
session latency and descriptive ratios, retained tensor bytes separately.
This is single-machine exploratory latency with development data, no confidence
or production-speed claim. It is NOT a continuous multiquery workload, nor a
full comparison with optimized MemNet/Transformer/SSM implementations.

Next criterion: if the conventional store meets the same reliability screen,
prioritize COPY-vs-LINK/deletion and shared learned parsing before architectural
scaling. If it fails, inspect learnability/optimization before declaring a CSM
advantage. Either outcome leaves natural language, larger active state, staged
learning, strong tuned baselines and full session reuse unestablished.
