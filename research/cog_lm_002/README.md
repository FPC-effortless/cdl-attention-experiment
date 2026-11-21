# COG-LM-002

This directory fits and evaluates a content-addressed language memory on the pretrained 001 models. See PROTOCOL.md for the registered comparisons and REPORT.md for the measured outcome. It reuses the pinned 001 CPU environment, byte-level BPE tokenizer, WikiText-2 articles, MultiWOZ dialogues and exact successor table. The new bank stores eight persistent records. Each 16-token block proposes a learned address and value. Matching keys compete for a gated overwrite, and a token-conditioned query reads the preceding records. Each stream starts empty at the document boundary. The system receives only token IDs.

## Reproduction

Start with the pinned data preparations in ../cog_lm_001/README.md and the published seed-17 checkpoints. Use Python 3.12 and the 001 pinned dependencies. Run the following from the repository root. Select paths to the prepared data, original checkpoint folder, downloaded English LAMBADA parquet and two writable result directories.

```bash
py=python
old=research/cog_lm_001
new=research/cog_lm_002
data=/path/to/prepared/data
original=/path/to/001/checkpoints
lambada=/path/to/en.parquet
dialogue=/path/to/MultiWOZ_2.2/test/dialogues_002.json
raw=/path/to/002/receipts
joint=/path/to/002/joint-fits
mkdir -p "$raw" "$joint"

"$py" "$old/train.py" --arm mamba2 --scale scale --seed 17 --threads 2 \
  --resume "$original/scale-mamba2-17.pt" --data "$data" --out "$raw"
mamba="$raw/scale-mamba2-17-continued.pt"
cog="$original/scale-local-17-continued.pt"
"$py" "$old/evaluate.py" --weights "$mamba" --data "$data" --out "$raw" --threads 2
for base in "$mamba" "$cog"; do
  "$py" "$new/adapt.py" --weights "$base" --data "$data" --out "$raw" --threads 3
done
for arm in cog_revision cog_lookup mamba_lookup; do
  base="$cog"
  threads=3
  microbatch=4
  if [ "$arm" = mamba_lookup ]; then base="$mamba"; threads=2; microbatch=2; fi
  "$py" "$new/joint.py" --arm "$arm" --base "$base" --data "$data" --out "$joint" --threads "$threads" --microbatch "$microbatch"
  for regime in frozen joint; do
    head="$raw/$arm.pt"
    trained_base="$base"
    if [ "$regime" = joint ]; then head="$joint/joint_$arm.pt"; trained_base="$joint/joint_$arm.base.pt"; fi
    args=(--base "$trained_base" --adapter "$head" --data "$data" --out "$raw")
    "$py" "$new/panels.py" "${args[@]}" --lambada "$lambada"
    if [ "$arm" = cog_revision ]; then
      "$py" "$new/panels.py" "${args[@]}" --lambada "$lambada" --disabled
      "$py" "$new/matched_cost.py" "${args[@]}"
      "$py" "$new/memory_use.py" "${args[@]}"
    fi
    if [ "$regime" = joint ]; then
      "$py" "$new/dialogue.py" "${args[@]}" --dialogue "$dialogue"
      if [ "$arm" = cog_revision ]; then "$py" "$new/dialogue.py" "${args[@]}" --dialogue "$dialogue" --disabled; fi
      "$py" "$new/cold_request.py" "${args[@]}" --tokenizer "$data/tokenizer.json"
      "$py" "$new/economics.py" "${args[@]}"
      if [ "$arm" = cog_revision ]; then
        "$py" "$new/economics.py" "${args[@]}" --disabled
        "$py" "$new/cold_request.py" "${args[@]}" --tokenizer "$data/tokenizer.json" --disabled
      fi
    fi
  done
done
"$py" "$new/balanced_timing.py" --joint "$joint" --data "$data" --out "$raw"
"$py" "$new/audit.py" --data "$data" --original "$original" --raw "$raw" --joint "$joint" --out "$raw/audit.json"
"$py" "$new/summarize.py" --raw "$raw" --joint "$joint" --out "$raw/summary.json"
"$py" -m pytest -q "$old/test_lm.py" "$new/test_runtime.py"
```

## Generate a continuation

Use the matching base and adapter files as a pair. For example, the jointly trained COG revision model uses joint_cog_revision.base.pt and joint_cog_revision.pt. Frozen variants use the parent SHA recorded in their fit JSON.

```bash
python research/cog_lm_002/runtime.py --base joint_cog_revision.base.pt \
  --adapter joint_cog_revision.pt --tokenizer research/cog_lm_001/tokenizer.json \
  --prompt 'The history of the city' --count 48
```

Measured scores are based on full-vocabulary causal likelihoods. The probe compares two interventions that swap the assignments and leave all subsequent text unchanged. A fixed preference can score half of individual forced choices. It cannot get both directions of a counterfactual pair correct. Filler length is counted between the end of the second correction and the start of each query. Natural-text context scores use the same final 128 labels in all four conditions for each eligible article.
