# Reproduction and resource accounting

Use Python 3.12 and standard library. Run in repository root:

```bash
cd research/cog-081-086/extreme
python run_ladder.py --output results
python -m unittest test_ladder -v
sha256sum -c CASE_SHA256SUMS
```

This recreates two deterministic JSONL outputs. `summary.json` includes `wall_ns`, which is intentionally **not** hash-stable; compare task counts and case/measurement digests, not wall time. Approximate local wall time on a single CPU was about 29 seconds; timing varies between hosts.

`results/cases.jsonl` and `results/reference_metrics.jsonl` are intentionally separate to prevent answer leakage. The reference file is published for independent auditing only; evaluated agents must not load it or import generator/search implementation files.

Scoring JSONL for a task uses `{ "id": "...", "actions": [ ... ] }` with the original pilot's domain-specific action formats. For chess mate-in-N submit `{ "id": "...", "strategy": { "FEN at each White turn": ["Q", FROM_INDEX, TO_INDEX] } }`; the scorer evaluates all Black replies rather than following one line. Maze shift tasks allow an additional `shift_actions` sequence for the changed-world task. Hanoi levels 8–10 require a separate future streaming checker; the offline scorer labels these `requires_streaming_evaluator`.

All scenario parameters, source files, and two reserved seed streams remain inspectable; any unchanged bound puzzle should yield an identical `input_sha256` in a separate-machine replay.
