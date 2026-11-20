#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
revision=a150876415327c80daeff35ca6f68f5ed8cf5c24
mkdir -p recipe/nli
for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json added_tokens.json README.md onnx/model_quint8_avx2.onnx; do
    curl -fLsS --retry 2 "https://huggingface.co/cross-encoder/nli-deberta-v3-xsmall/resolve/$revision/$file" -o "recipe/nli/${file##*/}"
done
(cd recipe/nli && sha256sum *.json *.onnx README.md) > nli.sha256
