"""Train-only reversible character encoding for the pinned Tiny Shakespeare text."""
from __future__ import annotations

import argparse
from array import array
import hashlib
import json
from pathlib import Path
import sys

SOURCE_COMMIT = "370cbcd448eb7daf32f21a6be560b70e0b33c4e3"
SOURCE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    + SOURCE_COMMIT + "/data/tinyshakespeare/input.txt"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CharTokenizer:
    def __init__(self, chars: list[str]):
        if not chars or len(chars) != len(set(chars)) or any(len(c) != 1 for c in chars):
            raise ValueError("Tokenizer needs unique single Unicode characters")
        self.chars = list(chars)
        self.stoi = {c: i for i, c in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        missing = sorted(set(text) - self.stoi.keys())
        if missing:
            raise ValueError(
                "Prompt contains characters outside the trained vocabulary: "
                + repr("".join(missing))
            )
        return [self.stoi[c] for c in text]

    def decode(self, ids) -> str:
        return "".join(self.chars[int(i)] for i in ids)

    @classmethod
    def load(cls, path: Path):
        return cls(json.loads(path.read_text(encoding="utf-8"))["chars"])


def prepare(data_dir: Path, expected_sha256: str | None = None) -> dict:
    source = data_dir / "input.txt"
    raw = source.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and source_hash != expected_sha256:
        raise ValueError("Corpus SHA-256 does not match the expected pinned bytes")
    text = raw.decode("utf-8")
    if len(text) < 10000:
        raise ValueError("Corpus unexpectedly small")
    n = len(text)
    edges = [0, int(n * 0.90), int(n * 0.95), n]
    names = ["train", "val", "test"]
    tokenizer = CharTokenizer(sorted(set(text[:edges[1]])))
    coverage = {
        name: sorted(set(text[a:b]) - tokenizer.stoi.keys())
        for name, a, b in zip(names, edges, edges[1:])
    }
    if any(coverage.values()):
        raise ValueError("Held-out characters not in training vocabulary: " + repr(coverage))
    tok_info = {
        "kind": "character",
        "fit_split": "train",
        "normalization": "none",
        "special_tokens": [],
        "chars": tokenizer.chars,
        "vocab_size": len(tokenizer.chars),
        "unknown_policy": "reject explicitly; no silent replacement",
    }
    (data_dir / "tokenizer.json").write_text(
        json.dumps(tok_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    splits = {}
    for name, a, b in zip(names, edges, edges[1:]):
        ids = tokenizer.encode(text[a:b])
        encoded = array("H", ids)
        if sys.byteorder != "little":
            encoded.byteswap()
        path = data_dir / (name + ".bin")
        path.write_bytes(encoded.tobytes())
        splits[name] = {
            "start_character_inclusive": a,
            "end_character_exclusive": b,
            "characters": b-a,
            "available_next_character_targets": max(0, b-a-1),
            "binary_sha256": sha256(path),
        }
    meta = {
        "dataset": "Tiny Shakespeare",
        "source_repository": "karpathy/char-rnn",
        "source_commit": SOURCE_COMMIT,
        "source_url": SOURCE_URL,
        "input_sha256": source_hash,
        "input_bytes": len(raw),
        "characters": n,
        "encoding": "utf-8",
        "binary_encoding": "little-endian unsigned uint16",
        "split_policy": "contiguous 90% train / 5% validation / 5% test",
        "boundary_policy": "no training/evaluation target crosses a split boundary",
        "tokenizer_fit": "training characters only",
        "vocab_size": len(tokenizer.chars),
        "tokenizer_sha256": sha256(data_dir / "tokenizer.json"),
        "unknown_characters_by_split": coverage,
        "heldout_text_usage": "character-coverage check only during preparation; no tokenizer fit",
        "splits": splits,
    }
    (data_dir / "dataset.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def load_split(data_dir: Path, name: str, verify: bool = True):
    import torch
    meta = json.loads((data_dir / "dataset.json").read_text(encoding="utf-8"))
    path = data_dir / (name + ".bin")
    if verify and sha256(path) != meta["splits"][name]["binary_sha256"]:
        raise ValueError("Split file hash mismatch: " + name)
    raw = array("H")
    raw.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        raw.byteswap()
    # Copy into an owned tensor, avoiding writable-buffer lifetime assumptions.
    ids = torch.tensor(raw.tolist(), dtype=torch.long)
    if len(ids) != meta["splits"][name]["characters"]:
        raise ValueError("Split length mismatch: " + name)
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(json.dumps(prepare(args.data_dir, args.expected_sha256), indent=2))


if __name__ == "__main__":
    main()

