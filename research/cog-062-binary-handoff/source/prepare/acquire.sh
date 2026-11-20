#!/bin/bash
set -euo pipefail
base='https://raw.githubusercontent.com/budzianowski/multiwoz/fe0c8e65cfcd8462bd33c86e35f21addc84ca82b'
mkdir -p publisher/train
export base
seq -f '%03g' 1 17 | xargs -P 4 -I @ bash -eu -c 'p=publisher/train/dialogues_$1.json; test -f "$p" || { curl --fail --silent --show-error --retry 1 --max-time 55 --max-filesize 12000000 "$base/data/MultiWOZ_2.2/train/dialogues_$1.json" --output "$p.part"; mv "$p.part" "$p"; }' _ @
for meta in LICENSE data/MultiWOZ_2.2/README.md data/MultiWOZ_2.2/schema.json; do
  p=publisher/"${meta##*/}"; test -f "$p" || { curl --fail --silent --show-error --max-time 30 --max-filesize 1000000 "$base/$meta" --output "$p.part"; mv "$p.part" "$p"; }
done
