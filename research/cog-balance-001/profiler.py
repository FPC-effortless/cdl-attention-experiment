"""Whole-model CPU profile of a fixed batch-one prefill and carried decode."""
import ctypes
import os
import platform
import resource
import statistics
import struct
import time
from pathlib import Path

import torch


def _storage_sizes(state):
    backing = {}
    logical = 0

    def visit(x):
        nonlocal logical
        if torch.is_tensor(x):
            storage = x.untyped_storage()
            backing[(str(x.device), storage.data_ptr())] = storage.nbytes()
            logical += x.numel() * x.element_size()
        elif isinstance(x, dict):
            for v in x.values():
                visit(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                visit(v)
        elif x is not None and not isinstance(x, (int, float, bool)):
            raise TypeError(f"Unexpected carried state type {type(x).__name__}")

    visit(state)
    return {"logical_bytes": logical, "backing_tensor_bytes": sum(backing.values())}


def _energy_meters():
    meters = []
    for p in sorted(Path("/sys/class/powercap").glob("*:*")):
        try:
            if p.joinpath("name").read_text().strip().startswith("package-"):
                meter = p / "energy_uj"
                maximum = int((p / "max_energy_range_uj").read_text())
                int(meter.read_text())
                meters.append((meter, maximum))
        except (OSError, ValueError):
            return []
    return meters


def _energy_snapshot(meters):
    try:
        return [int(p.read_text()) for p, _ in meters]
    except (OSError, ValueError):
        return []


def _read_imc_fd(pmu, event_name, cpu):
    # Linux's perf_event_open ABI. Accept only Intel's simple event/umask
    # format and units explicitly exposed by sysfs for this IMC counter.
    syscall_number = {"x86_64": 298, "aarch64": 241}.get(platform.machine())
    if syscall_number is None:
        raise ValueError("unsupported perf_event_open ABI")
    spec = dict(field.split("=") for field in (pmu / "events" / event_name).read_text().strip().split(","))
    if set(spec) != {"event", "umask"}:
        raise ValueError("unsupported IMC event encoding")
    for key, field in (("event", "config:0-7"), ("umask", "config:8-15")):
        if (pmu / "format" / key).read_text().strip() != field:
            raise ValueError("unsupported IMC bitfields")
    scale_file = pmu / "events" / (event_name + ".scale")
    unit_file = pmu / "events" / (event_name + ".unit")
    if unit_file.read_text().strip() != "MiB":
        raise ValueError("unsupported IMC count unit")
    bytes_per_event = float(scale_file.read_text()) * (1024 ** 2)
    attr = ctypes.create_string_buffer(64)
    config = int(spec["event"], 0) | (int(spec["umask"], 0) << 8)
    struct.pack_into("IIQ", attr, 0, int((pmu / "type").read_text()), 64, config)
    # Read the actual enabled and running times to detect multiplexing.
    struct.pack_into("Q", attr, 32, 3)
    fd = ctypes.CDLL(None, use_errno=True).syscall(
        syscall_number, ctypes.byref(attr), -1, cpu, -1, 8,
    )
    if fd < 0:
        raise OSError(ctypes.get_errno(), "could not open package-wide IMC counter")
    return fd, bytes_per_event


def _imc_meters():
    meters = []
    try:
        for pmu in Path("/sys/bus/event_source/devices").glob("uncore_imc_[0-9]*"):
            cpus = (pmu / "cpumask").read_text().strip().split(",")
            for cpu in cpus:
                for event in ("cas_count_read", "cas_count_write"):
                    meters.append(_read_imc_fd(pmu, event, int(cpu)))
    except (OSError, ValueError):
        for fd, _ in meters:
            os.close(fd)
        return []
    return meters


def _imc_snapshot(meters):
    try:
        return [struct.unpack("QQQ", os.read(fd, 24)) for fd, _ in meters]
    except (OSError, struct.error):
        return []


def _traffic_delta(start, stop, meters):
    if not meters or len(start) != len(meters) or len(stop) != len(meters):
        return None
    total = 0.0
    for (v0, e0, r0), (v1, e1, r1), (_, scale) in zip(start, stop, meters):
        if r1 <= r0 or e1 < e0 or v1 < v0:
            return None
        total += scale * (v1 - v0) * (e1 - e0) / (r1 - r0)
    return round(total)


@torch.inference_mode()
def profile_model(model, token_ids):
    """Accept [T] or [B,T] CPU token IDs. Use at most 48 IDs of row zero."""
    tokens = torch.as_tensor(token_ids, dtype=torch.long, device="cpu")
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.size(0) == 0 or tokens.size(1) < 2:
        raise ValueError("need one sequence with at least two tokens")
    x = tokens[:1, :48].contiguous()
    split = min(32, max(1, x.size(1) * 2 // 3))
    was_training = model.training
    model.eval()
    imc = []
    try:
        reference = model(x)
        # Warm both the batched prefill and single-token paths before measuring.
        _, state = model(x[:, :split], carry=True)
        model(x[:, split:split + 1], state, carry=True)
        energy = _energy_meters()
        imc = _imc_meters()
        joules = []
        memory_bytes = []
        decode_times = []
        prefill_times = []
        per_step = []
        abs_err = 0.0
        for _ in range(3):
            start = time.perf_counter_ns()
            _, state = model(x[:, :split], carry=True)
            prefill_times.append((time.perf_counter_ns() - start) / 1e9)
            energy_before = _energy_snapshot(energy)
            traffic_before = _imc_snapshot(imc)
            start = time.perf_counter_ns()
            outputs = []
            for t in range(split, x.size(1)):
                step_start = time.perf_counter_ns()
                logits, state = model(x[:, t:t + 1], state, carry=True)
                per_step.append((time.perf_counter_ns() - step_start) / 1e6)
                outputs.append(logits)
            decode_times.append((time.perf_counter_ns() - start) / 1e9)
            traffic_after = _imc_snapshot(imc)
            energy_after = _energy_snapshot(energy)
            memory_bytes.append(_traffic_delta(traffic_before, traffic_after, imc))
            if energy and len(energy_before) == len(energy_after) == len(energy):
                joules.append(sum((b - a) % mx for a, b, (_, mx) in zip(
                    energy_before, energy_after, energy
                )) / 1e6)
            abs_err = max(abs_err, (torch.cat(outputs, 1) - reference[:, split:]).abs().max().item())

        n = x.size(1) - split
        return {
            "device": "cpu", "torch_threads": torch.get_num_threads(),
            "batch": 1, "repeats": 3, "prefill_tokens": split, "decode_tokens": n,
            "prefill_tokens_s": split / statistics.median(prefill_times),
            "decode_tokens_s": n / statistics.median(decode_times),
            "decode_step_median_ms": statistics.median(per_step),
            "decode_step_p95_ms": statistics.quantiles(per_step, n=100, method="inclusive")[94],
            "decode_seconds_by_repeat": decode_times,
            "batch_vs_stream_max_abs_logit_error": abs_err,
            "final_carried_tensor_state": _storage_sizes(state),
            "process_lifetime_peak_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "package_energy_j_per_decode_token": statistics.median(joules) / n if len(joules) == 3 else None,
            "package_dram_read_plus_write_bytes_per_decode_token": (
                statistics.median(memory_bytes) / n if all(v is not None for v in memory_bytes) else None
            ),
            "hardware_counter_scope": "host package totals during decode; includes co-tenants and sampling overhead. Null when counters are unavailable.",
        }
    finally:
        for fd, _ in imc:
            os.close(fd)
        model.train(was_training)
