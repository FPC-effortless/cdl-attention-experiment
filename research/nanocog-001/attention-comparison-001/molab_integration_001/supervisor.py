"""Standard-library supervisor for a visible, user-started marimo notebook cell.

Import is inert: no files, subprocesses, network calls, or model imports. There
is deliberately no CLI, SSH, detached-job endpoint, or background service.
"""
from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid
import zipfile


ENV_SHA256 = "70c81e91d05c38a7cd487c2d6a74b639ff2ce1802ce4b27a93071a2ffefc7719"
KERNEL_SHA256 = "c7cd17d088294cf56f62a372328921ed4e29cf083f8bfc897f7040a374ee4a81"
ARCHIVE_SHA256 = "65f19af445dc368e382faa32fd006941969c6df26f5c4b5330bd898472f555f5"
SCHEMA = "cog.molab-interactive-supervisor.v1"
MAX_LOG_BYTES = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES = 24 * 1024 * 1024
STAGES = {
    "environment": {"sha256": ENV_SHA256, "payload": "environment_b64",
                    "script": "environment_probe.py", "seconds": 300,
                    "run_id": "COG-ATTENTION-MOLAB-ENV-001"},
    "adapter": {"sha256": KERNEL_SHA256, "payload": "kernel_b64",
                "script": "reviewed_adapter_kernel.py", "seconds": 900,
                "run_id": "COG-ATTENTION-MOLAB-ADAPTER-001"},
}
OUTPUTS = {
    "environment": ("environment.json",),
    "adapter": ("environment_gate.json", "run_source_manifest.json", "dependency_install.json", "bootstrap_result.json",
                "adapter_results/run_result.json", "adapter_results/adapter_checks.json",
                "adapter_results/model_provenance.json"),
}
COMMON_OUTPUTS = ("wrapper_claim.json", "wrapper_result.json", "stdout.log")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_owner() -> str:
    """Stable within this notebook Python process, fresh after kernel restart."""
    name = "_cog_molab_001_owner_nonce"
    if not hasattr(sys, name):
        setattr(sys, name, uuid.uuid4().hex)
    return getattr(sys, name)


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_new(path: Path, content: bytes) -> None:
    """Exclusive writes; existing attempt artifacts are never replaced."""
    with path.open("xb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _read_regular(path: Path, limit: int = MAX_ARTIFACT_BYTES) -> bytes:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
        raise ValueError("Artifact is not an allowed bounded regular file: " + path.name)
    with path.open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("Artifact grew beyond its byte cap")
    return data


def _read_json(path: Path) -> dict:
    value = json.loads(_read_regular(path))
    if not isinstance(value, dict):
        raise ValueError("Expected object receipt")
    return value


def verify_payloads(envelope: dict) -> dict[str, bytes]:
    """Decode only two explicit public files and require their reviewed hashes."""
    result = {}
    for stage, spec in STAGES.items():
        content = base64.b64decode(envelope[spec["payload"]], validate=True)
        if hashlib.sha256(content).hexdigest() != spec["sha256"]:
            raise ValueError(stage + " payload differs from its reviewed source")
        result[stage] = content
    if envelope.get("embedded_archive_sha256") != ARCHIVE_SHA256:
        raise ValueError("Archive identity differs from reviewed bundle")
    return result


def _root(path: str | Path) -> Path:
    root = Path(path).absolute()
    if root.exists() and root.is_symlink():
        raise ValueError("Run root must not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _environment_gate(result: dict | None) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Run and pass the environment stage before the adapter stage")
    receipt = result.get("receipt", {})
    observation = receipt.get("environment_observation", {})
    valid = (receipt.get("schema") == SCHEMA
             and receipt.get("stage") == "environment"
             and receipt.get("provider_context") == "MoLab"
             and receipt.get("status") == "PASS"
             and receipt.get("returncode") == 0
             and receipt.get("source_sha256") == ENV_SHA256
             and receipt.get("notebook_kernel_owner") == runtime_owner()
             and observation.get("source_sha256") == ENV_SHA256
             and observation.get("status") == "PASS_ENVIRONMENT_ONLY"
             and observation.get("cuda_available") is True
             and isinstance(observation.get("device_count"), int)
             and observation["device_count"] > 0
             and len(observation.get("devices", [])) == observation["device_count"]
             and all(d.get("analytic_gradient_check") is True for d in observation["devices"]))
    if not valid:
        raise ValueError("Missing, failed, stale-kernel, or non-CUDA environment receipt; no model launched")
    return {"source_sha256": ENV_SHA256, "environment_run_id": receipt["run_id"],
            "environment_attempt_id": receipt["attempt_id"],
            "notebook_kernel_owner": receipt["notebook_kernel_owner"],
            "environment_observation": observation}


def _child_environment(run_dir: Path) -> dict[str, str]:
    # Forward runtime settings only. Do not enumerate, print or forward account
    # credentials, API keys, tokens, arbitrary PYTHONPATH, or pip index settings.
    names = ("PATH", "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES",
             "NVIDIA_DRIVER_CAPABILITIES", "LANG", "LC_ALL", "TZ", "SSL_CERT_FILE", "SSL_CERT_DIR")
    env = {name: os.environ[name] for name in names if name in os.environ}
    temp = run_dir / "temporary"
    temp.mkdir()
    env.update({"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1", "PIP_CONFIG_FILE": os.devnull, "PIP_NO_INPUT": "1",
                "NETRC": os.devnull, "PIP_KEYRING_PROVIDER": "disabled",
                "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
                "NANOCHAT_DTYPE": "float32", "TMPDIR": str(temp)})
    return env


def _terminate_group(process: subprocess.Popen, grace: float = 2.0) -> dict:
    """Own process group only, even if the original group leader has exited."""
    result = {"group_id": process.pid, "term_sent": False, "kill_sent": False, "leader_reaped": False}
    try:
        os.killpg(process.pid, signal.SIGTERM)
        result["term_sent"] = True
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # A reaped leader does not imply every descendant exited. Kill the owned
    # group if it still exists; never use a broad pkill or a notebook group ID.
    try:
        os.killpg(process.pid, signal.SIGKILL)
        result["kill_sent"] = True
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=3.0)
        result["leader_reaped"] = True
    except subprocess.TimeoutExpired:
        result["leader_reaped"] = False
    return result


def _parse_observations(stage: str, run_dir: Path, receipt: dict) -> None:
    """Pass requires both exit success and the exact expected evidence status."""
    try:
        if stage == "environment":
            observed = _read_json(run_dir / "environment.json")
            receipt["environment_observation"] = observed
            complete = (observed.get("status") == "PASS_ENVIRONMENT_ONLY"
                        and observed.get("source_sha256") == ENV_SHA256
                        and observed.get("cuda_available") is True
                        and observed.get("device_count", 0) > 0
                        and len(observed.get("devices", [])) == observed["device_count"]
                        and all(x.get("analytic_gradient_check") is True for x in observed["devices"]))
        else:
            observed = _read_json(run_dir / "adapter_results/run_result.json")
            checks = _read_json(run_dir / "adapter_results/adapter_checks.json")
            receipt["adapter_observation"] = observed
            receipt["adapter_checks_status"] = checks.get("status")
            complete = (observed.get("status") == "PASS_ENGINEERING_ONLY"
                        and observed.get("optimizer_updates") == 0
                        and observed.get("scientific_comparison_admitted") is False
                        and checks.get("status") == "PASS")
        if receipt["status"] == "EXITED":
            receipt["status"] = "PASS" if receipt["returncode"] == 0 and complete else "FAILED"
    except Exception as error:
        receipt["observation_error"] = {"type": type(error).__name__, "message": str(error)}
        if receipt["status"] == "EXITED":
            receipt["status"] = "FAILED_MISSING_OR_INVALID_RECEIPT"


def _collect_zip(stage: str, run_dir: Path, *, label: str) -> dict:
    """Only explicit receipt/log names; never zip a working directory broadly."""
    artifacts, entries, missing, skipped = [], [], [], []
    for relative in COMMON_OUTPUTS + OUTPUTS[stage]:
        path = run_dir / relative
        try:
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                skipped.append({"path": relative, "reason": "nonregular file"})
                continue
            cap = MAX_LOG_BYTES if relative == "stdout.log" else MAX_ARTIFACT_BYTES
            with path.open("rb") as stream:
                data = stream.read(cap)
        except FileNotFoundError:
            missing.append(relative)
            continue
        except OSError as error:
            skipped.append({"path": relative, "reason": type(error).__name__, "message": str(error)})
            continue
        if sum(len(value) for _, value in artifacts) + len(data) > MAX_ARTIFACT_BYTES:
            skipped.append({"path": relative, "reason": "export total byte cap"})
            continue
        artifacts.append((relative, data))
        entries.append({"path": relative, "size_bytes": len(data),
                        "original_size_bytes": metadata.st_size,
                        "truncated": metadata.st_size > len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
    if sum(len(data) for _, data in artifacts) > MAX_ARTIFACT_BYTES:
        raise ValueError("Selected artifacts exceed the export cap")
    manifest = {"schema": "cog.molab-artifacts.v1", "stage": stage, "snapshot_kind": label,
                "files": entries, "missing": missing, "skipped": skipped, "credentials_included": False,
                "training_checkpoint_included": False,
                "export_status": "PARTIAL" if missing or skipped or any(x["truncated"] for x in entries) else "COMPLETE_SELECTED_FILES"}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, data in artifacts:
            archive.writestr(relative, data)
        archive.writestr("artifact_manifest.json", _json_bytes(manifest))
    raw = buffer.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    filename = f"molab-{stage}-{run_dir.name}-{label}-{sha[:12]}.zip"
    _write_new(run_dir / filename, raw)
    return {"filename": filename, "sha256": sha, "bytes": len(raw),
            "archive_b64": base64.b64encode(raw).decode(), "manifest": manifest}


def run_stage(stage: str, envelope: dict, root_path: str, persistent_claim: dict,
              environment_result: dict | None = None) -> dict:
    """Call only from a visible notebook run-button cell, never automatically.

    A persistent cache ticket plus an exclusive local marker prevents ordinary
    reactive repeats. MoLab may evict caches/storage; this is not a permanent
    global lock. Interrupted calls finalize local evidence and then re-raise.
    """
    if os.name != "posix" or not hasattr(os, "killpg"):
        raise RuntimeError("This supervisor requires a POSIX process-group runtime")
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    spec = STAGES[stage]
    payloads = verify_payloads(envelope)
    if (persistent_claim.get("stage") != stage
            or persistent_claim.get("source_sha256") != spec["sha256"]
            or persistent_claim.get("owner_nonce") != runtime_owner()):
        raise RuntimeError("Prior or invalid cached attempt ticket: no automatic retry in a new notebook kernel")
    gate = _environment_gate(environment_result) if stage == "adapter" else None
    root = _root(root_path)
    marker = root / (stage + ".attempt.json")
    if marker.exists():
        raise RuntimeError("This stage already has an attempt. Use the independent export button; do not rerun.")
    attempt_id = uuid.uuid4().hex
    run_dir = root / (stage + "_" + attempt_id)
    run_dir.mkdir(exist_ok=False)
    receipt = {
        "schema": SCHEMA, "provider_context": "MoLab", "stage": stage,
        "run_id": spec["run_id"], "attempt_id": attempt_id,
        "notebook_kernel_owner": runtime_owner(), "started_utc": utc_now(),
        "source_sha256": spec["sha256"], "embedded_archive_sha256": ARCHIVE_SHA256 if stage == "adapter" else None,
        "wall_cap_seconds": spec["seconds"], "status": "CLAIMED", "returncode": None,
        "work_cutoff_seconds": spec["seconds"] - 5,
        "teardown_reserve_seconds": 5,
        "visibility": "public/unlisted default; no private visibility assertion",
        "hosting_origin_verified_by_code": False,
        "historical_inner_metadata": "Kaggle IDs/private/T4/server-timeout fields are preserved source-bundle history, not MoLab observations or current privacy/hardware guarantees.",
        "timeout_authority": "interactive notebook supervisor monotonic wall clock; not a Kaggle server limit",
        "optimizer_updates": 0, "scientific_comparison_admitted": False,
        "training_or_evaluation_corpus": None,
        "marimo_version": importlib.metadata.version("marimo"),
        "python": sys.version, "supervisor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    marker_record = {"schema": SCHEMA, "stage": stage, "attempt_id": attempt_id,
                     "run_dir_name": run_dir.name, "owner_nonce": runtime_owner(),
                     "source_sha256": spec["sha256"], "claim": persistent_claim}
    # Atomic exclusive marker is written before any child can be launched.
    _write_new(marker, _json_bytes(marker_record))
    _write_new(run_dir / "wrapper_claim.json", _json_bytes(receipt))
    _write_new(run_dir / spec["script"], payloads[stage])
    if gate is not None:
        _write_new(run_dir / "environment_gate.json", _json_bytes(gate))
    started = time.monotonic()
    process = None
    interrupted = None
    display_bytes = 0
    log_offset = 0
    archive = None

    def show_output() -> None:
        nonlocal log_offset, display_bytes
        log = run_dir / "stdout.log"
        if not log.exists():
            return
        with log.open("rb") as stream:
            stream.seek(log_offset)
            chunk = stream.read(65536)
            log_offset += len(chunk)
        if chunk and display_bytes < 1024 * 1024:
            display = chunk[:1024 * 1024 - display_bytes]
            print(display.decode("utf-8", errors="replace"), end="", flush=True)
            display_bytes += len(display)

    try:
        env = _child_environment(run_dir)
        with (run_dir / "stdout.log").open("xb", buffering=0) as output:
            process = subprocess.Popen([sys.executable, "-u", str(run_dir / spec["script"])],
                                       cwd=run_dir, env=env, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            receipt["status"], receipt["child_pid"] = "RUNNING", process.pid
            while True:
                show_output()
                returncode = process.poll()
                if returncode is not None:
                    receipt["status"], receipt["returncode"] = "EXITED", returncode
                    break
                # Reserve TERM/KILL/reap time inside the advertised job cap.
                if time.monotonic() - started >= spec["seconds"] - 5:
                    receipt["status"] = "TIMEOUT"
                    break
                if (run_dir / "stdout.log").stat().st_size > MAX_LOG_BYTES:
                    receipt["status"] = "OUTPUT_LIMIT"
                    break
                time.sleep(0.2)
    except BaseException as error:
        receipt["status"] = "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "SUPERVISOR_FAILED"
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        if not isinstance(error, Exception):
            interrupted = error
    finally:
        if process is not None:
            try:
                receipt["group_cleanup"] = _terminate_group(process)
                receipt["returncode"] = process.returncode
                if not receipt["group_cleanup"]["leader_reaped"]:
                    receipt["execution_status_before_cleanup"] = receipt["status"]
                    receipt["status"] = "FAILED_PROCESS_CLEANUP"
            except BaseException as cleanup_error:
                receipt["execution_status_before_cleanup"] = receipt["status"]
                receipt["status"] = "FAILED_PROCESS_CLEANUP"
                receipt["group_cleanup"] = {"error_type": type(cleanup_error).__name__,
                                            "error": str(cleanup_error), "leader_reaped": False}
                if interrupted is None and not isinstance(cleanup_error, Exception):
                    interrupted = cleanup_error
        receipt["ended_utc"] = utc_now()
        receipt["elapsed_seconds"] = time.monotonic() - started
        _parse_observations(stage, run_dir, receipt)
        _write_new(run_dir / "wrapper_result.json", _json_bytes(receipt))
        # This final local ZIP exists even if marimo cancels dependent cells.
        # A separately gated export cell reads it without referencing run results.
        try:
            archive = _collect_zip(stage, run_dir, label="final")
        except Exception as error:
            archive = {"error": {"type": type(error).__name__, "message": str(error)}}
        show_output()
        print(json.dumps({"stage": stage, "status": receipt["status"],
                          "elapsed_seconds": receipt["elapsed_seconds"],
                          "export_available": "archive_b64" in archive}), flush=True)
    if interrupted is not None:
        raise interrupted
    return {"receipt": receipt, "archive": archive}


def export_existing(root_path: str) -> dict:
    """Independent recovery action: selected attempt paths only, no execution."""
    root = _root(root_path)
    results = []
    for stage in STAGES:
        marker = root / (stage + ".attempt.json")
        if not marker.exists():
            continue
        claim = _read_json(marker)
        name = claim["run_dir_name"]
        if Path(name).name != name or not name.startswith(stage + "_"):
            raise ValueError("Invalid owned attempt path")
        run_dir = root / name
        if run_dir.is_symlink() or not run_dir.is_dir():
            raise ValueError("Owned attempt directory is unavailable")
        receipt_path = run_dir / "wrapper_result.json"
        try:
            receipt = _read_json(receipt_path)
        except FileNotFoundError:
            receipt = {"stage": stage, "status": "INCOMPLETE_NO_FINAL_RECEIPT",
                       "note": "A missing final receipt is not evidence that the child finished or that compute stopped."}
        except Exception as error:
            receipt = {"stage": stage, "status": "INCOMPLETE_INVALID_FINAL_RECEIPT",
                       "error": {"type": type(error).__name__, "message": str(error)},
                       "note": "Raw allowlisted evidence is retained; invalid JSON is not a successful execution receipt."}
        # Recovery snapshots get unique labels and do not overwrite final ZIPs.
        archive = _collect_zip(stage, run_dir, label="recovery-" + uuid.uuid4().hex[:12])
        results.append({"receipt": receipt, "archive": archive})
    return {"schema": "cog.molab-recovery.v1", "snapshots": results,
            "persistence_limit": "MoLab cache/storage lifetime is discretionary. Download ZIPs before ending the session."}
