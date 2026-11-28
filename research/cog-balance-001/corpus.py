"""Verify and sample the TRACE-007 WikiText corpus (Actions artifact 15273004264)."""
import hashlib
import json
from pathlib import Path

import numpy as np


MANIFEST_HASH = "2955e5b6e772cf4539615b0585c1b522f7cf168c3c8e47c95b29dc97dda40c35"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Corpus:
    def __init__(self, root):
        path = root / "manifest.json"
        if digest(path) != MANIFEST_HASH:
            raise ValueError("TRACE-007 corpus manifest hash mismatch")
        source = json.loads(path.read_text())
        for name, expected in source["files"].items():
            if digest(root / name) != expected:
                raise ValueError(f"Wrong corpus file hash for {name}")
        self.manifest = {
            "manifest_sha256": MANIFEST_HASH,
            "revision": source["revision"],
            "train": source["train"],
            "dev": source["dev"],
            "artifact_id": 15273004264,
            "sampling": "uniform over all eligible within-document byte starts with replacement",
        }
        self.bytes = {
            split: np.memmap(root / f"{split}.bin", dtype=np.uint8, mode="r")
            for split in ("train", "dev")
        }
        self.offsets = {
            split: np.load(root / f"{split}_offsets.npy", allow_pickle=False)
            for split in self.bytes
        }
        self.cumulative = {}

    def batch(self, split, rng, batch_size, length):
        """Draw a batch of (context length + one next-byte label) windows."""
        if length < 1 or batch_size < 1:
            raise ValueError("Expected positive batch size and context length")
        key = (split, length)
        if key not in self.cumulative:
            eligible = np.maximum(0, np.diff(self.offsets[split].astype(np.int64)) - length)
            self.cumulative[key] = np.cumsum(eligible)
        cumulative = self.cumulative[key]
        if cumulative[-1] == 0:
            raise ValueError("No documents long enough for the requested length")
        ranks = rng.integers(cumulative[-1], size=batch_size)
        docs = cumulative.searchsorted(ranks, side="right")
        previous = np.where(docs > 0, cumulative[docs - 1], 0)
        starts = self.offsets[split][docs].astype(np.int64) + ranks - previous
        return self.bytes[split][starts[:, None] + np.arange(length + 1)].astype(np.int64)


def load_corpus(root=None):
    """Use an explicitly supplied Actions download or the local staged corpus."""
    return Corpus(Path(root) if root is not None else Path.cwd() / "trace_corpus")
