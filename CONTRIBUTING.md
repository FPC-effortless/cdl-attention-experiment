# Contributing

This repository is intentionally falsification-first.

When changing the benchmark:

1. preserve previous result files;
2. document prompt, template, model, or scoring changes;
3. do not tune against held-out cases without recording the tuning;
4. report negative results as well as positive results;
5. keep retrieval metrics separate from downstream answer NLL.

The central hypothesis is defined in `EXPERIMENT.md`.
