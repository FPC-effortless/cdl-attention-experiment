"""Fetch the pinned public Tiny Shakespeare corpus and verify its exact bytes."""
from pathlib import Path
import hashlib
import urllib.request

from data import SOURCE_URL, prepare
from train import DEFAULT_CORPUS_SHA256

root = Path(__file__).resolve().parent / "data"
root.mkdir(exist_ok=True)
target = root / "input.txt"
if target.exists():
    payload = target.read_bytes()
else:
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read(2_000_000)
if hashlib.sha256(payload).hexdigest() != DEFAULT_CORPUS_SHA256:
    raise SystemExit("Corpus identity mismatch; existing data were not replaced.")
if not target.exists():
    target.write_bytes(payload)
metadata = prepare(root, DEFAULT_CORPUS_SHA256)
print(f"Verified {metadata['characters']:,} characters; {metadata['vocab_size']} training characters.")

