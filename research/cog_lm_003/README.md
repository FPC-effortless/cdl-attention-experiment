# COG-LM-003

Read REPORT.md for the results and PROTOCOL.md for the primary design and subsequent exploratory intervention. This cycle reuses the seed-17 COG-LM-002 jointly trained systems. The frozen backbone cache accelerates repeated fits. It preserves FP32 token-level representations. Generated revisions and natural-text fillers are reproducible from the released code and pinned data.

## Reproduce

Run from the repository root with the pinned environment in research/cog_lm_001/requirements.txt. Prepare the data by following the 001 README. The two annotated MultiWOZ files are data/MultiWOZ_2.2/test/dialogues_002.json and data/MultiWOZ_2.2/dev/dialogues_001.json from budzianowski/multiwoz commit fe0c8e65cfcd8462bd33c86e35f21addc84ca82b. The code verifies both source hashes. The release includes the three 002 parent pairs and fit records in parents/ and the ten new readout checkpoints in heads/.

```bash
set -eu
py=python
n=research/cog_lm_003
parents=/path/to/parents
data=/path/to/prepared/data
known=/path/to/test/dialogues_002.json
fresh=/path/to/dev/dialogues_001.json
out=/path/to/results
cache=/path/to/cache
mkdir -p "$out" "$cache"
for a in cog_revision cog_lookup mamba_lookup; do
  "$py" "$n/cache.py" --arm "$a" --parents "$parents" --data "$data" --out "$cache"
done
for name in short_text long_text short_revision long_revision route_revision cog_control route_cog_control mamba_control route_mamba_control floor_revision; do
  args=(--name "$name" --parents "$parents" --data "$data" --out "$out" --cache "$cache")
  "$py" "$n/fit.py" "${args[@]}"
  "$py" "$n/assess.py" "${args[@]}"
done
common=(--parents "$parents" --data "$data" --out "$out")
for a in cog_lookup mamba_lookup cog_revision; do
  "$py" "$n/conversations.py" "${common[@]}" --threads 1 --arm "$a" --known "$known" --fresh "$fresh"
done
for script in attribution usage replay timing; do "$py" "$n/$script.py" "${common[@]}"; done
for script in seen mechanism; do "$py" "$n/$script.py" "${common[@]}" --cache "$cache"; done
"$py" "$n/parity.py"
"$py" "$n/summarize.py" --out "$out"
```

Use an empty cache directory for a different experiment. For generation, supply the trained readout and its corresponding frozen base to the existing language runtime. The fit receipt and summary identify the required base for every head.

```bash
python research/cog_lm_002/runtime.py --base parents/joint_cog_revision.base.pt \
  --adapter heads/route_revision.pt --tokenizer research/cog_lm_001/tokenizer.json \
  --prompt 'The train left the station' --count 32
```
