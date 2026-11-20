#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
revision=1110a243fdf4706b3f48f1d95db1a4f5529b4d41
mkdir -p recipe/minilm
for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors README.md; do
    curl -fLsS --retry 2 "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$revision/$file" -o "recipe/minilm/$file"
done
(cd recipe/minilm && sha256sum config.json tokenizer.json tokenizer_config.json special_tokens_map.json model.safetensors README.md) > encoder.sha256
