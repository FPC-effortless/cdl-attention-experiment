# Engineering check record

CPU, PyTorch 2.10.0+cpu, Python 3.12, one configured OpenMP thread. Checks used inert generated input. From this directory:

```sh
OMP_NUM_THREADS=1 python -m pytest -q nanocog
```

Expected fixture-only suite: 23 passed, one optional exact-source test skipped. With the exact separately supplied PULSE-006 source set via `PULSE006_SOURCE`, 24 passed. The optional test validates hashed source interfaces using forced heads and synthesized inputs, with no trained checkpoints, scientific scoring or replication claim.

Checks cover suffix noninterference; chunk/single-token versus full passage; immutable prefix reuse and branch separation; reset; fixed FIFO overwrite; padding; gradient plumbing and hard-routing gradient boundaries; atomic update and one-hop reads; and differential checks against a simple typed-store interpreter. They do not exercise nanochat training, tokenizer state, a serving engine, or trained capabilities.

## Pinned upstream reference

[karpathy/nanochat commit 92d63d4e8bb4df75c3b71618f31ddde2378b2bcd](https://github.com/karpathy/nanochat/commit/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd), inspected Git blob identities:

| Source | Git blob |
| --- | --- |
| `nanochat/gpt.py` | `1e89378b1ae4fcba97462960b5a11695b1fe6a8e` |
| `nanochat/engine.py` | `9680d830fa8ebcd5d365482dc85eaec576757997` |
| `nanochat/tokenizer.py` | `36880f58dc168e45d947313739b697304a4e00ed` |
| `scripts/base_train.py` | `d1af271a4fd70411dd844d4bc7f46a3ce068f730` |
| `LICENSE` | `72d95c190c0d14341d5b0720e28c88fb58690821` |
