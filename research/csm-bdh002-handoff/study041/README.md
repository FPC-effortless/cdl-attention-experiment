# Study 041 — Governing subject and predicate scope

Study 040 exposed two occurrence-selection failures. The accepted status belongs to the head subject in “the former rather than the latter receives accepted.” In “accepted as denied to the former takes effect for the latter,” the governing predicate supplies the recipient and the denial supplies a competing occurrence.

The new selector recognizes a bounded set of subject and predicate constructions. It requires one accepted status and one supported attachment in its clause. It abstains on the ambiguous and unsupported forms checked by the companion script. Binding, frozen checkpoints, evaluation rosters, conditional COPY/LINK execution, and the scoring gate are inherited from Study 040.

| Layout | Study 040 best joint quartets per seed | Scope grammar per seed | Scope grammar unauthorized operations |
| --- | ---: | ---: | ---: |
| Intervening consultation | 96/96 | 96/96 | 0 |
| Postposed consultation | 96/96 | 96/96 | 0 |
| Unseen connector | 96/96 | 96/96 | 0 |
| Contrastive subject | 0/96 | 96/96 | 0 |
| Indirect scope | 0/96 | 96/96 | 0 |

The selector cleared the 92/96 gate in all 15 layout-by-seed cells, with zero abstentions on the replay. The prior arms reproduce the committed score table exactly. The companion checks cover four authored paraphrases and six abstention cases. These cases were designed with the grammar in view. The original test layouts were known during implementation. Independent paraphrase evaluation and comparison with a compact dependency parser remain open. The lexical scope checks cover a limited grammar and cannot establish correct treatment of unrestricted denial, embedding, or coordination. No comparative latency or deployment claim is supported.

From this directory, with the handoff dependencies installed:

```sh
OPENBLAS_NUM_THREADS=1 python replay.py > /tmp/csm-study041.csv
diff -u scores.csv /tmp/csm-study041.csv
python scope_checks.py
```
