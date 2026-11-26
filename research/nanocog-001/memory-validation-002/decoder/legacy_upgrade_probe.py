"""Exercise a genuine published v1 decoder/store database against the v2 runtime."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[1]
BASE = "56cb4eeb04d68f5f6c210c756f0790f34cc2fd5f"
PREFIX = "research/nanocog-001/"

LEGACY_WORKER = r'''
import json,sys,torch
from nanocog.model import Config,Decoder,LocalOnly
from nanocog.memory.decoder import PersistentDecoder
from nanocog.memory.store import MemoryStore
torch.set_num_threads(1)
torch.manual_seed(731)
model=Decoder(Config(31,32,2),LocalOnly).eval()
with torch.no_grad(): model.smear_gain.fill_(0.7)
with MemoryStore(sys.argv[1],principal_id="upgrade-fixture",namespace="validation") as store:
    store.append_message("notes","user","The fixture project colour is green.",event_id="source")
    store.propose("project.colour","green","source",proposal_id="proposal")
    store.accept("proposal",authority="fixture-user")
    adapter=PersistentDecoder(model,store,session_id="legacy-stream",tokenizer_id="synthetic-vocab31-v1")
    result=adapter.run(torch.tensor([[2,3,4,5,7]]),request_id="legacy-first")
    state=store.load_checkpoint("legacy-stream")
    print(json.dumps({"schema_version":store.db.execute("PRAGMA user_version").fetchone()[0],
                      "manifest":state["manifest"],"revision":result.revision,
                      "tokens_seen":result.tokens_seen,"fact":store.facts("project.colour")[0]["value"]}))
'''


def main():
    report = dict(baseline_commit=BASE, trained=False,
                  purpose="actual v1 SQL migration and intentional native decoder identity break")
    current_paths = ("nanocog/memory/decoder.py", "nanocog/memory/store.py", "nanocog/memory/codec.py")
    report["source_sha256_before"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in current_paths}
    with tempfile.TemporaryDirectory(prefix="nanocog-legacy-upgrade-") as name:
        tmp = Path(name)
        old_root = tmp / "legacy"
        files = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", BASE,
                                         PREFIX + "nanocog"], cwd=REPO, text=True).splitlines()
        copied = []
        for filename in files:
            if not filename.endswith(".py") or "/test_" in filename:
                continue
            data = subprocess.check_output(["git", "show", f"{BASE}:{filename}"], cwd=REPO)
            target = old_root / filename.removeprefix(PREFIX)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            copied.append(dict(path=filename, sha256=hashlib.sha256(data).hexdigest()))
        database = tmp / "legacy.db"
        legacy = subprocess.run([sys.executable, "-c", LEGACY_WORKER, str(database)], cwd=tmp,
                                env=dict(os.environ, PYTHONPATH=str(old_root), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"),
                                capture_output=True, text=True, timeout=60)
        if legacy.returncode:
            raise RuntimeError(legacy.stderr)
        report["legacy_creation"] = json.loads(legacy.stdout)
        report["legacy_source_sha256"] = copied
        sys.path.insert(0, str(ROOT))
        import torch
        from nanocog.model import Config, Decoder, LocalOnly
        from nanocog.memory.codec import decode
        from nanocog.memory.decoder import PersistentDecoder, validate_decode_state
        from nanocog.memory.store import IntegrityError, MemoryStore
        torch.set_num_threads(1)
        torch.manual_seed(731)
        model = Decoder(Config(31, 32, 2), LocalOnly).eval()
        with torch.no_grad():
            model.smear_gain.fill_(0.7)
        with MemoryStore(database, principal_id="upgrade-fixture", namespace="validation") as store:
            current = store.load_checkpoint("legacy-stream")
            report["migration"] = dict(schema_version=store.db.execute("PRAGMA user_version").fetchone()[0],
                                        fact=store.facts("project.colour")[0]["value"],
                                        observation_retained=store.get_message("source")["content"] ==
                                                             "The fixture project colour is green.",
                                        checkpoint_retained=current is not None,
                                        old_state_shape_valid=validate_decode_state(decode(current["payload"]["state"]), model) is not None)
            try:
                PersistentDecoder(model, store, session_id="legacy-stream", tokenizer_id="synthetic-vocab31-v1")
            except IntegrityError as exc:
                report["legacy_attach"] = dict(accepted=False, error=type(exc).__name__, message=str(exc))
            else:
                report["legacy_attach"] = dict(accepted=True)
            new = PersistentDecoder(model, store, session_id="fresh-stream", tokenizer_id="synthetic-vocab31-v1")
            result = new.run(torch.tensor([[11, 13]]), request_id="new-first")
            report["fresh_stream"] = dict(manifest=new._manifest, tokens_seen=result.tokens_seen,
                                           revision=result.revision, replayed=result.replayed)
        report["passed"] = (report["legacy_creation"]["schema_version"] == 1
                             and report["legacy_creation"]["manifest"]["version"] == 1
                             and report["migration"]["schema_version"] == 2
                             and report["migration"]["fact"] == "green"
                             and report["migration"]["observation_retained"]
                             and report["migration"]["checkpoint_retained"]
                             and report["migration"]["old_state_shape_valid"]
                             and not report["legacy_attach"]["accepted"]
                             and report["fresh_stream"]["manifest"]["version"] == 2
                             and report["fresh_stream"]["tokens_seen"] == 2
                             and report["fresh_stream"]["revision"] == 1)
    report["source_sha256_after"] = {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in current_paths}
    report["source_unchanged"] = report["source_sha256_before"] == report["source_sha256_after"]
    report["passed"] = report["passed"] and report["source_unchanged"]
    (Path(__file__).resolve().parent / "legacy_upgrade.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "legacy_attach": report["legacy_attach"],
                      "facts_preserved": report["migration"]["fact"] == "green",
                      "new_manifest_version": report["fresh_stream"]["manifest"]["version"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
