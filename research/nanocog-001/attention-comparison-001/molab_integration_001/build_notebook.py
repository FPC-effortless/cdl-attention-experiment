"""Build a self-contained interactive notebook; never imports or runs the jobs."""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


ENV_SHA = "70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719"
KERNEL_SHA = "c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81"
ARCHIVE_SHA = "65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5"


TEMPLATE = '''# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo"]
# ///
"""Interactive MoLab engineering notebook. Compute requires explicit buttons."""
import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Attention replacement: bounded engineering checks

    This notebook runs the reviewed environment and adapter checks on **the GPU
    actually attached to this MoLab session**. It does not train a model or rank
    architectures. Each stage needs its own explicit button press: environment
    at most 5 minutes, then adapter engineering at most 15 minutes.

    **Visibility:** MoLab notebooks are public/unlisted by default. This file
    contains public source, public tokenizer assets and inert fixtures only.
    It contains no credentials, private data or model checkpoints, and it makes
    no assertion that this notebook is private.

    The inner programs are preserved byte-for-byte from the reviewed Kaggle
    bundle. Their old Kaggle IDs, private flag, T4 request and server-timeout
    declarations are historical metadata. The outer receipt records MoLab,
    observed hardware, and this notebook's wall-clock supervisor.

    **One attempt per stage.** A cached claim plus an exclusive local marker
    blocks ordinary reactive repeats. An old-kernel claim is refused. MoLab
    storage/cache may expire; this is not a permanent global execution lock.
    Do not retry by clearing markers or caches. Preserve failures and review
    any new attempt separately.

    Enable a GPU in MoLab's notebook-specs control before the environment stage.
    After completion **or interruption**, use the independent export button
    below to download the selected evidence ZIPs before ending the session.
    """)
    return


@app.cell(hide_code=True)
def _():
    import json
    payload_envelope = json.loads(__ENVELOPE_LITERAL__)
    supervisor_source_b64 = __SUPERVISOR_LITERAL__
    supervisor_source_sha256 = __SUPERVISOR_SHA_LITERAL__
    return payload_envelope, supervisor_source_b64, supervisor_source_sha256


@app.cell
def _(mo, payload_envelope, supervisor_source_b64, supervisor_source_sha256):
    import base64
    import hashlib
    import importlib.util
    import os
    from pathlib import Path
    import sys
    mo.stop(not mo.running_in_notebook(), mo.md("Open this file in an interactive marimo/MoLab notebook."))
    if os.name != "posix" or sys.platform == "emscripten":
        raise RuntimeError("Use a server-backed POSIX MoLab notebook, not a WebAssembly preview.")
    _notebook_parent = mo.notebook_dir()
    if _notebook_parent is None:
        raise RuntimeError("Notebook directory unavailable; no jobs will be started.")
    runtime_root = Path(_notebook_parent) / "molab_integration_001_runtime"
    runtime_root.mkdir(exist_ok=True)
    _helper_bytes = base64.b64decode(supervisor_source_b64, validate=True)
    if hashlib.sha256(_helper_bytes).hexdigest() != supervisor_source_sha256:
        raise RuntimeError("Embedded supervisor hash mismatch")
    _helper_path = runtime_root / ("supervisor_" + supervisor_source_sha256 + ".py")
    if _helper_path.exists():
        if _helper_path.is_symlink() or _helper_path.read_bytes() != _helper_bytes:
            raise RuntimeError("Existing helper differs from the reviewed bytes")
    else:
        with _helper_path.open("xb") as _output:
            _output.write(_helper_bytes)
    _module_name = "cog_molab_supervisor_" + supervisor_source_sha256
    if _module_name in sys.modules:
        supervisor = sys.modules[_module_name]
    else:
        _spec = importlib.util.spec_from_file_location(_module_name, _helper_path)
        supervisor = importlib.util.module_from_spec(_spec)
        sys.modules[_module_name] = supervisor
        _spec.loader.exec_module(supervisor)
    supervisor.verify_payloads(payload_envelope)
    supervisor.runtime_owner()
    run_root = str(runtime_root / "attempts")
    mo.md("Source hashes verified. No compute has launched. Native packages will be inspected by the environment stage.")
    return run_root, supervisor


@app.cell
def _(mo):
    @mo.persistent_cache(method="json")
    def persisted_attempt_claim(stage, source_sha256):
        # Imports live inside the cached function: the nonce is a return value,
        # not a changing function argument/cache key. A cache hit from an older
        # kernel therefore returns an old owner and is refused by the supervisor.
        import sys
        import time
        import uuid
        _attribute = "_cog_molab_001_owner_nonce"
        if not hasattr(sys, _attribute):
            setattr(sys, _attribute, uuid.uuid4().hex)
        return {"stage": stage, "source_sha256": source_sha256,
                "owner_nonce": getattr(sys, _attribute), "claimed_unix": time.time()}

    @mo.persistent_cache(method="json")
    def cache_export_snapshot(snapshot):
        # Convenience persistence only. The explicit download is the evidence
        # handoff; cache TTL and provider shutdown behavior are not guaranteed.
        return snapshot
    return cache_export_snapshot, persisted_attempt_claim


@app.cell
def _(mo):
    environment_start = mo.ui.run_button(label="Run environment check · max 5 min")
    environment_start
    return (environment_start,)


@app.cell
def _(environment_start, mo, payload_envelope, persisted_attempt_claim, run_root, supervisor):
    # Keep mo.stop outside any broad exception handler and before side effects.
    mo.stop(not environment_start.value, mo.md("Environment stage waits for its button."))
    _claim = persisted_attempt_claim("environment", supervisor.ENV_SHA256)
    environment_result = supervisor.run_stage("environment", payload_envelope, run_root, _claim)
    mo.md("Environment result: **" + environment_result["receipt"]["status"] + "**. See live console output and the export action.")
    return (environment_result,)


@app.cell
def _(environment_result, mo):
    _receipt = environment_result["receipt"]
    _observation = _receipt.get("environment_observation", {})
    mo.vstack([
        mo.md("## Observed accelerator inventory"),
        mo.ui.table(_observation.get("devices", [])),
        mo.md("CUDA available: **" + str(_observation.get("cuda_available", False)) + "**. The model stage requires a passing receipt from this notebook kernel."),
    ])
    return


@app.cell
def _(environment_result, mo):
    mo.stop(environment_result["receipt"]["status"] != "PASS",
            mo.md("Adapter start remains unavailable until the environment stage passes."))
    adapter_start = mo.ui.run_button(label="Run adapter engineering · max 15 min")
    adapter_start
    return (adapter_start,)


@app.cell
def _(adapter_start, environment_result, mo, payload_envelope, persisted_attempt_claim, run_root, supervisor):
    mo.stop(not adapter_start.value, mo.md("Adapter stage waits for its own button after the environment check."))
    _claim = persisted_attempt_claim("adapter", supervisor.KERNEL_SHA256)
    adapter_result = supervisor.run_stage("adapter", payload_envelope, run_root, _claim,
                                          environment_result=environment_result)
    mo.md("Adapter result: **" + adapter_result["receipt"]["status"] + "**. No optimizer updates or quality ranking were performed.")
    return (adapter_result,)


@app.cell
def _(mo):
    export_start = mo.ui.run_button(label="Refresh and export evidence · works after interruption")
    export_start
    return (export_start,)


@app.cell
def _(cache_export_snapshot, export_start, mo, run_root, supervisor):
    # Deliberately independent of environment_result / adapter_result: marimo can
    # cancel their descendants on Stop. This recovery action remains runnable.
    import base64 as _export_base64
    mo.stop(not export_start.value, mo.md("Use this button after success, failure or an interrupted cell."))
    export_snapshot = cache_export_snapshot(supervisor.export_existing(run_root))
    _downloads = []
    for _item in export_snapshot["snapshots"]:
        _archive = _item["archive"]
        _downloads.append(mo.vstack([
            mo.md("**" + _item["receipt"].get("stage", "stage") + ": " + _item["receipt"]["status"] + "**"),
            mo.download(data=_export_base64.b64decode(_archive["archive_b64"]),
                        filename=_archive["filename"], mimetype="application/zip",
                        label="Download " + _item["receipt"].get("stage", "stage") + " evidence"),
            mo.md("ZIP SHA-256: `" + _archive["sha256"] + "`"),
        ]))
    mo.vstack(_downloads) if _downloads else mo.md("No attempt has been started; no job outputs exist.")
    return (export_snapshot,)


if __name__ == "__main__":
    app.run()
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_archive(kernel):
    values = {}
    for node in ast.parse(kernel).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("PAYLOAD", "ARCHIVE_SHA256"):
                values[name] = ast.literal_eval(node.value)
            elif name == "MANIFEST":
                values[name] = json.loads(ast.literal_eval(node.value.args[0]))
    raw = base64.b64decode(values["PAYLOAD"], validate=True)
    if sha(raw) != ARCHIVE_SHA or values["ARCHIVE_SHA256"] != ARCHIVE_SHA:
        raise ValueError("Reviewed archive identity mismatch")
    specs = {row["bundle_path"]: row for row in values["MANIFEST"]["files"]}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        members = archive.infolist()
        if len(members) != 33 or len(specs) != 33 or len(values["MANIFEST"]["files"]) != 33:
            raise ValueError("Archive member count differs from reviewed allowlist")
        if {m.filename for m in members} != set(specs) or sum(m.file_size for m in members) > 10_000_000:
            raise ValueError("Archive membership or size cap mismatch")
        for member in members:
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or not stat.S_ISREG(member.external_attr >> 16):
                raise ValueError("Unsafe archive member")
            data = archive.read(member)
            if len(data) != specs[member.filename]["size_bytes"] or sha(data) != specs[member.filename]["sha256"]:
                raise ValueError("Archive member hash/length mismatch")
    return {"members_verified": 33, "archive_sha256": ARCHIVE_SHA, "archive_bytes": len(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    environment, kernel = args.environment.read_bytes(), args.kernel.read_bytes()
    if sha(environment) != ENV_SHA or sha(kernel) != KERNEL_SHA:
        raise ValueError("Input programs are not the exact independently reviewed bytes")
    archive_check = validate_archive(kernel)
    helper = (destination / "supervisor.py").read_bytes()
    ast.parse(helper)
    envelope = {"schema": "cog.molab-public-payloads.v1", "embedded_archive_sha256": ARCHIVE_SHA,
                "environment_b64": base64.b64encode(environment).decode(),
                "kernel_b64": base64.b64encode(kernel).decode()}
    notebook = TEMPLATE.replace("__ENVELOPE_LITERAL__", repr(json.dumps(envelope, sort_keys=True)))
    notebook = notebook.replace("__SUPERVISOR_LITERAL__", repr(base64.b64encode(helper).decode()))
    notebook = notebook.replace("__SUPERVISOR_SHA_LITERAL__", repr(sha(helper)))
    ast.parse(notebook)
    (destination / "notebook.py").write_text(notebook)
    receipt = {"status": "SOURCE_CHECKS_ONLY_NO_JOB_EXECUTION", "provider_context": "MoLab",
               "environment_sha256": ENV_SHA, "kernel_sha256": KERNEL_SHA,
               "supervisor_sha256": sha(helper), "notebook_sha256": sha(notebook.encode()),
               "builder_sha256": sha(Path(__file__).read_bytes()), "archive_check": archive_check,
               "native_imports": 0, "optimizer_updates": 0, "remote_calls": 0,
               "runtime_marimo_validation": "NOT_RUN", "notebook_bytes": len(notebook.encode())}
    (destination / "SOURCE_CHECKS.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
