# CSM historical replay through Research Lab OS

TrainLab ran Study 041 (including the Study 040 baseline) and SKILL-001 through SKILL-005 from CSM commit b7bbfd52b6ffeb339eed471ecc1f860bd04b2fe9. Each run has a historical-replay recipe, a verified receipt, input hashes, logs and output artifacts. Forge froze the six generated input inventories. Its inventory splits have no role in the historical studies' train/test assignments. The coordinator compared the deterministic artifacts to the pinned Git objects, validated fresh timing coverage and recorded six Veritas shared evidence envelopes in the local Research Lab OS graph.

All six runs succeeded. The score tables, pilot JSON, refitted trees and model descriptions reproduced exactly. Study 041 passed its ten scope checks. Study 003 recorded 90 fresh timing batches with median microseconds per goal of 2.62 deferred, 2.74 primitive and 3.06 eager. The archive includes the SQLite lab, graph export and every receipt. Veritas records these observations under historical replay. Scientific qualification remains NOT_QUALIFIED. This run uses the local lab instance and a CSM replay verifier.

## Resume from a clone

Run from this directory in a full CSM checkout with Python 3.12.14. Use a new output directory for each execution.

```sh
git clone https://github.com/FPC-effortless/veritas.git /tmp/csm-veritas-src
git -C /tmp/csm-veritas-src checkout --detach eec439492407b0aa26922c23b5ae40d95289f4eb
python -m venv /tmp/csm-replay-env
. /tmp/csm-replay-env/bin/activate
python -m pip install -r ../requirements.txt /tmp/csm-veritas-src /tmp/csm-veritas-src/integrations/forge /tmp/csm-veritas-src/integrations/trainlab /tmp/csm-veritas-src/integrations/research_lab
sha256sum -c SHA256SUMS
python -m zipfile -e historical-replay.zip /tmp/csm-saved
python verify.py /tmp/csm-saved
python pipeline.py /tmp/csm-new-run
python verify.py /tmp/csm-new-run
```

## Second execution

[Download the second execution bundle](historical-replay-2.zip) to inspect six fresh TrainLab receipts and the corresponding Research Lab OS database. All six input inventories and deterministic artifact hashes matched the first execution. Study 003 collected another 90 timing batches. Use the same extraction and verification commands above with the second archive and a fresh destination.
