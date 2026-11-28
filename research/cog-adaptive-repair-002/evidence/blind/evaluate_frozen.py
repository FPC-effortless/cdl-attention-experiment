"""Score the locked taggers and shared C fallback against one sealed English bank."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch


SEEDS = (9731, 9743, 9769)
DIRS = {"narrow": "structured", "augmented": "structured_transfer"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


@torch.inference_mode()
def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--root", type=Path, default=Path.cwd())
    cli.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = cli.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    lock = json.loads((out / "evaluation_lock.json").read_text())
    for rel, expected in lock["files"].items():
        assert sha(root / rel) == expected, f"Hash mismatch for {rel}"
    frozen_dir = root / "local_result" / DIRS["augmented"]
    frozen = json.loads((frozen_dir / "FROZEN.json").read_text())
    assert frozen["seeds"] == list(SEEDS)
    assert [row["seed"] for row in frozen["checkpoints"]] == list(SEEDS)
    for row in frozen["checkpoints"]:
        assert row["filename"] == f"tagger_{row['seed']}.pt"
        assert sha(frozen_dir / row["filename"]) == row["sha256"]
    assert sha(root / "structured_transfer.py") == frozen["source_sha256"]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "balance_repo/research/unified-search-002"))
    from integrated import WholeCOG
    from structured_memory import StreamingStore, StructuredCOG, actions, parse_many, segments
    from structured_transfer import load_tagger
    import blind_transfer as bank
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    raw = []
    bank.evaluate_solver(lambda p: raw.append(p))
    assert len(raw) == 3072 and len(set(raw)) == len(raw)
    hashes = [hashlib.sha256(p).hexdigest() for p in raw]
    prompt_hash = hashlib.sha256(b"".join(bytes.fromhex(s) for s in hashes)).hexdigest()
    clauses = sorted({clause for p in raw for _, clause in segments(p)})
    sample = {group * 192 + (group % 3) for group in range(16)}

    def report(preds):
        return bank.evaluate_solver(dict(zip(raw, preds)).__getitem__)

    results = {"shared_prompt_sequence_sha256": prompt_hash, "unique_clauses": len(clauses),
               "template_sha256": bank._TEMPLATE_SHA256, "models": {}}
    predictions = {"prompt_sha256": hashes, "memory_only": {}}
    for variant, folder in DIRS.items():
        for seed in SEEDS:
            key = f"{variant}_{seed}"
            tagger = load_tagger(root / "local_result" / folder / f"tagger_{seed}.pt")
            parsed = parse_many(tagger, clauses)
            vals = [actions(p, parsed)[0].get(len(p) - 1) for p in raw]
            checked = sorted(sample | set(next_hits(vals, 4)))
            for i in checked:
                store = StreamingStore(tagger)
                edits = {j: y for j, b in enumerate(raw[i]) if (y := store.step(b)) is not None}
                assert edits == actions(raw[i], parsed)[0], f"Streaming mismatch in {key} at {i}"
            predictions["memory_only"][key] = vals
            results["models"][key] = {"sha256": sha(root / "local_result" / folder / f"tagger_{seed}.pt"),
                                      "streaming_checks": len(checked), "memory_only": report(vals)}
            save(out / "blind_results.json", results)
            print(key, results["models"][key]["memory_only"]["overall"], flush=True)

    base_file = root / "local_result/pilot/C-9731/matched_data_weights.pt"
    for seed in SEEDS:
        prior = root / "local_result" / DIRS["narrow"] / f"metrics_{seed}.json"
        assert json.loads(prior.read_text())["frozen_parent_sha256"] == sha(base_file)
    base = WholeCOG("SS_native_native",9731).eval()
    base.load_state_dict(torch.load(base_file, weights_only=True))
    vals = [None] * len(raw)
    order = sorted(range(len(raw)), key=lambda i: len(raw[i]))
    for at in range(0, len(raw), 12):
        group = order[at:at + 12]
        widths = [len(raw[i]) for i in group]
        batch = torch.zeros(len(group), max(widths), dtype=torch.long)
        for j, i in enumerate(group):
            batch[j, :widths[j]] = torch.tensor(list(raw[i]))
        state = None
        for block_start in range(0, batch.shape[1], 128):
            block = batch[:, block_start:block_start + 128]
            logits, state = base(block, state, carry=True)
            for j, i in enumerate(group):
                offset = widths[j] - block_start - 1
                if 0 <= offset < block.shape[1]:
                    vals[i] = int(logits[j, offset].argmax())
        if at % 480 == 0:
            print("C fallback prompts", at + len(group), "/", len(raw), flush=True)
    assert all(v is not None for v in vals)
    predictions["frozen_C"] = vals
    results["frozen_C"] = {"sha256": sha(base_file), "counts": report(vals)}
    for variant, folder in DIRS.items():
        for seed in SEEDS:
            key = f"{variant}_{seed}"
            memory = predictions["memory_only"][key]
            full = [v if read is None else read for v, read in zip(vals, memory)]
            wrapper = StructuredCOG(base, load_tagger(root / "local_result" / folder / f"tagger_{seed}.pt")).eval()
            # Include one model dispatch if available plus two fixed prompt indices.
            checked = sorted({0, len(raw) - 1} | set(next_hits(memory, 1)))
            for i in checked:
                state = None
                for block in torch.tensor([list(raw[i])]).split(128, dim=1):
                    logits, state = wrapper(block, state, carry=True)
                assert int(logits[0, -1].argmax()) == full[i], f"Full wrapper mismatch for {key} at {i}"
            results["models"][key].update(whole_wrapper=report(full), wrapper_checks=len(checked))
    save(out / "blind_results.json", results)
    with gzip.open(out / "blind_predictions.json.gz", "wt") as f:
        json.dump(predictions, f, separators=(",", ":"))
    print("BLIND EVALUATION COMPLETE", flush=True)


def next_hits(vals, count):
    hits = []
    for i, v in enumerate(vals):
        if v is not None:
            hits.append(i)
            if len(hits) == count:
                break
    return hits


if __name__ == "__main__":
    main()
