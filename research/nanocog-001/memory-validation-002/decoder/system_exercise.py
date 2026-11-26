"""CPU-only reference-model persistence exercises; no training or semantic score.

Every worker invocation is a new Python process. Configurations and tokens are
retained in the result so arbitrary-boundary/backup checks are reproducible.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import sqlite3
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from nanocog.interaction import PacketAndTypedProbe, PacketMixer
from nanocog.model import Config, Decoder, LocalOnly
from nanocog.memory.codec import decode, encode
from nanocog.memory.decoder import PersistentDecoder
from nanocog.memory.store import MemoryStore

CONFIG = Config(65, 128, 6)
SEED = 20261123
CHUNKS = [1, 2, 7, 31, 3, 64, 5, 97, 47]


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_for(kind):
    torch.manual_seed(SEED)
    cls = {"local": LocalOnly, "packet": PacketMixer, "probe": PacketAndTypedProbe}[kind]
    model = Decoder(CONFIG, cls).eval()
    with torch.no_grad():
        model.smear_gain.fill_(0.7)
    return model


def scoped_store(path):
    return MemoryStore(path, principal_id="synthetic-fixture", namespace="decoder-system-validation")


def adapter(model, store):
    return PersistentDecoder(model, store, session_id="stream", tokenizer_id="synthetic-id-vocab65-v1")


def worker(job_path):
    job = json.loads(Path(job_path).read_text())
    model = model_for(job["kind"])
    with scoped_store(job["database"]) as store:
        session = adapter(model, store)
        start = time.perf_counter()
        result = session.run(torch.tensor(job["tokens"], dtype=torch.long), request_id=job["request"])
        inference_and_persistence_seconds = time.perf_counter() - start
        current = store.load_checkpoint("stream")
        payload = dict(pid=os.getpid(), logits=encode(result.logits), revision=result.revision,
                       tokens_seen=result.tokens_seen, replayed=result.replayed,
                       checkpoint_revision=current["revision"], state=current["payload"]["state"],
                       manifest=current["manifest"],
                       inference_and_persistence_seconds=inference_and_persistence_seconds)
    Path(job["output"]).write_text(json.dumps(payload, separators=(",", ":")))


def launch(folder, database, kind, tokens, request, label):
    job = dict(database=str(database), kind=kind, tokens=tokens.tolist(), request=request,
               output=str(folder / f"{label}-output.json"))
    path = folder / f"{label}-job.json"
    path.write_text(json.dumps(job))
    start = time.perf_counter()
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(path)],
                            capture_output=True, text=True, timeout=60,
                            env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    if result.returncode:
        raise RuntimeError(f"worker {label}: {result.stderr[-5000:]}")
    value = json.loads(Path(job["output"]).read_text())
    value["process_wall_seconds"] = time.perf_counter() - start
    return value


def compare(x, y):
    diff = (x - y).abs()
    return dict(bit_exact=bool(torch.equal(x, y)),
                within_tolerance=bool(torch.allclose(x, y, rtol=1e-5, atol=1e-6)),
                max_abs_difference=float(diff.max()),
                argmax_changed_positions=int((x.argmax(-1) != y.argmax(-1)).sum()),
                positions=int(x.shape[1]))


def fact_influence_probe(folder, kind):
    results = []
    facts = []
    for colour in ("red", "blue"):
        with scoped_store(folder / f"facts-{kind}-{colour}.db") as store:
            store.append_message("observations", "user", f"The launch colour is {colour}.", event_id="source")
            store.propose("project.launch_colour", colour, "source", proposal_id="proposal")
            store.accept("proposal", authority="fixture-user")
            facts.append(store.facts("project.launch_colour")[0]["value"])
            results.append(adapter(model_for(kind), store).run(torch.tensor([[2, 3, 4]]),
                                                              request_id="first").logits)
    return dict(committed_values=facts, comparison=compare(*results),
                interpretation="PersistentDecoder does not select or tokenize semantic facts. Identical token input has identical output despite changed stored facts; a host integration is required.")


def exercise_kind(folder, kind, tokens):
    model = model_for(kind)
    report = dict(kind=kind, parameters=sum(p.numel() for p in model.parameters()),
                  config=asdict(CONFIG), seed=SEED, trained=False, chunks=CHUNKS,
                  fresh_process_chunks=[])
    database = folder / f"{kind}.db"
    restored_database = folder / f"{kind}-restored.db"
    backup_at = 5
    with torch.inference_mode():
        uninterrupted_logits, uninterrupted_state = model(tokens)
    state = None
    offset = 0
    outputs = []
    reference_chunks = []
    saved = []
    for index, length in enumerate(CHUNKS):
        chunk = tokens[:, offset:offset + length]
        with torch.inference_mode():
            expected, state = model(chunk, state)
        reference_chunks.append(expected)
        observed = launch(folder, database, kind, chunk, f"chunk-{index:03d}", f"{kind}-{index:03d}")
        logits = decode(observed["logits"])
        saved.append(observed)
        outputs.append(logits)
        row = dict(index=index, length=length, tokens_seen=observed["tokens_seen"],
                   revision=observed["revision"], pid=observed["pid"],
                   expected_tokens_seen=offset + length, expected_revision=index + 1,
                   output=compare(logits, expected),
                   state_bit_exact=observed["state"] == encode(state),
                   state_sha256=canonical_sha(observed["state"]),
                   model_sha256=observed["manifest"]["model_sha256"],
                   inference_and_persistence_seconds=observed["inference_and_persistence_seconds"],
                   process_wall_seconds=observed["process_wall_seconds"])
        row["passed"] = (row["output"]["bit_exact"] and row["state_bit_exact"]
                         and row["tokens_seen"] == row["expected_tokens_seen"]
                         and row["revision"] == row["expected_revision"] and not observed["replayed"])
        report["fresh_process_chunks"].append(row)
        offset += length
        if index + 1 == backup_at:
            backup = folder / f"{kind}-backup.db"
            with scoped_store(database) as store:
                metadata = store.backup(backup)
            MemoryStore.restore_backup(backup, restored_database)
            with scoped_store(restored_database) as store:
                actual = store.load_checkpoint("stream")
                report["backup_restore"] = dict(backup_after_chunk=backup_at, backup_bytes=metadata["bytes"],
                                                backup_sha256=metadata["sha256"],
                                                restored_state_bit_exact=actual["payload"]["state"] == observed["state"],
                                                continuation=[])
    combined = torch.cat(outputs, dim=1)
    report["uninterrupted_comparison"] = compare(combined, uninterrupted_logits)
    report["chunked_reference_comparison"] = compare(combined, torch.cat(reference_chunks, dim=1))
    offset = sum(CHUNKS[:backup_at])
    for index in range(backup_at, len(CHUNKS)):
        length = CHUNKS[index]
        restored = launch(folder, restored_database, kind, tokens[:, offset:offset + length],
                          f"chunk-{index:03d}", f"{kind}-restored-{index:03d}")
        original = saved[index]
        row = dict(index=index, pid=restored["pid"],
                   state_bit_exact=restored["state"] == original["state"],
                   logits_bit_exact=restored["logits"] == original["logits"],
                   chronology_equal=(restored["revision"], restored["tokens_seen"]) ==
                                    (original["revision"], original["tokens_seen"]))
        row["passed"] = row["state_bit_exact"] and row["logits_bit_exact"] and row["chronology_equal"]
        report["backup_restore"]["continuation"].append(row)
        offset += length
    retry = launch(folder, database, kind, tokens[:, :CHUNKS[0]], "chunk-000", f"{kind}-old-retry")
    report["old_retry_in_new_process"] = dict(pid=retry["pid"], replayed=retry["replayed"],
                                               result_revision=retry["revision"],
                                               result_tokens_seen=retry["tokens_seen"],
                                               current_revision=retry["checkpoint_revision"],
                                               logits_bit_exact=retry["logits"] == saved[0]["logits"],
                                               state_unchanged=retry["state"] == saved[-1]["state"])
    generator = torch.Generator().manual_seed(9981)
    wrong_tokens = torch.randint(CONFIG.vocab_size, tokens.shape, generator=generator)
    suffix = torch.randint(CONFIG.vocab_size, (1, 31), generator=generator)
    with torch.inference_mode():
        _, wrong_state = model(wrong_tokens)
        correct, _ = model(suffix, state)
        reset, _ = model(suffix)
        wrong, _ = model(suffix, wrong_state)
    report["numerical_state_influence"] = dict(correct_vs_reset=compare(correct, reset),
                                               correct_vs_wrong_history=compare(correct, wrong),
                                               semantics_evaluated=False,
                                               interpretation="A numerical change only. Tokens and weights are synthetic; there are no labeled semantic answers and no useful-memory accuracy claim.")
    report["semantic_store_integration_probe"] = fact_influence_probe(folder, kind)
    retry_ok = (retry["replayed"] and retry["revision"] == 1 and retry["tokens_seen"] == CHUNKS[0]
                and retry["checkpoint_revision"] == len(CHUNKS) and retry["state"] == saved[-1]["state"]
                and retry["logits"] == saved[0]["logits"])
    report["passed"] = (all(r["passed"] for r in report["fresh_process_chunks"])
                         and report["uninterrupted_comparison"]["within_tolerance"]
                         and report["backup_restore"]["restored_state_bit_exact"]
                         and all(r["passed"] for r in report["backup_restore"]["continuation"])
                         and retry_ok)
    return report


def source_hashes():
    paths = ("nanocog/memory/decoder.py", "nanocog/memory/store.py", "nanocog/memory/codec.py",
             "nanocog/model.py", "nanocog/interaction.py", "nanocog/comm/packet.py", "nanocog/state/records.py")
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.worker:
        worker(args.worker)
        return 0
    folder = Path(__file__).resolve().parent
    tokens = torch.randint(CONFIG.vocab_size, (1, sum(CHUNKS)), generator=torch.Generator().manual_seed(81352))
    evidence = dict(purpose="reference inference persistence engineering validation",
                    trained=False, semantic_accuracy_measured=False, paid_compute=False,
                    runtime=dict(python=sys.version, torch=str(torch.__version__), sqlite=sqlite3.sqlite_version,
                                 platform=platform.platform(), device="cpu", threads=torch.get_num_threads()),
                    source_sha256_before=source_hashes(), tokens=tokens.tolist(), models=[])
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="nanocog-decoder-system-") as tmp:
        for kind in ("local", "packet", "probe"):
            try:
                report = exercise_kind(Path(tmp), kind, tokens)
            except Exception as exc:
                import traceback
                report = dict(kind=kind, passed=False, error=str(exc), traceback=traceback.format_exc())
            evidence["models"].append(report)
            evidence["elapsed_seconds"] = time.perf_counter() - start
            (folder / "system_results.json").write_text(json.dumps(evidence, indent=2) + "\n")
            print(json.dumps({k: report[k] for k in ("kind", "passed", "parameters") if k in report}), flush=True)
    evidence.update(elapsed_seconds=time.perf_counter() - start, source_sha256_after=source_hashes(),
                    passed=all(row["passed"] for row in evidence["models"]))
    (folder / "system_results.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
