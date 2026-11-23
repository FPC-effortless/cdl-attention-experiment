"""Bounded, non-executable JSON encoding for the reference decoder's state.

This is deliberately an allowlist, not a general Python object serializer. The
decoder adapter checks model-specific shape and chronology after decoding.
"""
import base64
import binascii
import hashlib
import math
import sys
from dataclasses import dataclass

import torch

from ..comm.packet import PacketState
from ..model import DecodeState
from ..state.records import RecordState


class CodecError(ValueError):
    """The input is malformed, unsupported, or exceeds the configured bounds."""


@dataclass(frozen=True)
class CodecLimits:
    max_bytes: int = 16 * 1024 * 1024
    max_tensors: int = 256
    max_depth: int = 16
    max_dimensions: int = 8
    max_items: int = 512

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in self.__dict__.values()):
            raise ValueError("codec limits must be positive integers")


_DTYPES = {str(d).removeprefix("torch."): d for d in (
    torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
)}
_CLASSES = {
    "DecodeState": (DecodeState, ("mixer_states", "previous_embedding", "tokens_seen")),
    "PacketState": (PacketState, ("local", "packets", "present", "positions")),
    "RecordState": (RecordState, ("value", "revision", "clock", "last_change")),
}


def _tensor_buffer(tensor, *, check_finite=True):
    """Canonical byte view, retained by its private array/tensor ownership chain.

    The identity checker can omit a repeated finite scan only when it compares
    the complete resulting digest to an identity whose finiteness was validated.
    Serialization always uses the finite-checked public ``tensor_bytes`` path.
    """
    if sys.byteorder != "little":
        raise CodecError("the v1 tensor format requires a little-endian host")
    if type(tensor) is not torch.Tensor or tensor.layout != torch.strided:
        raise CodecError("only ordinary dense tensors are supported")
    if str(tensor.dtype).removeprefix("torch.") not in _DTYPES:
        raise CodecError("unsupported tensor dtype")
    if check_finite and tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
        raise CodecError("nonfinite tensor")
    array = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8).numpy()
    return memoryview(array).cast("B")


def tensor_bytes(tensor):
    """Canonical owned bytes on little-endian hosts, with no pickle path."""
    return _tensor_buffer(tensor).tobytes()


class _Budget:
    def __init__(self, limits):
        self.limits = limits
        self.bytes = self.tensors = self.items = 0

    def visit(self, depth):
        self.items += 1
        if depth > self.limits.max_depth or self.items > self.limits.max_items:
            raise CodecError("state nesting/item limit exceeded")

    def tensor(self, shape, dtype):
        if (type(shape) is not list or len(shape) > self.limits.max_dimensions or
                any(type(n) is not int or n < 0 or n > self.limits.max_bytes for n in shape)):
            raise CodecError("invalid or oversized tensor shape")
        itemsize = torch.empty((), dtype=dtype).element_size()
        # Zero-sized dimensions must not disguise overflowing stride geometry.
        if math.prod(max(1, n) for n in shape) * itemsize > self.limits.max_bytes:
            raise CodecError("tensor shape exceeds geometry limit")
        size = math.prod(shape) * itemsize
        self.bytes += size
        self.tensors += 1
        if self.bytes > self.limits.max_bytes or self.tensors > self.limits.max_tensors:
            raise CodecError("tensor byte/count limit exceeded")
        return size


def encode(value, *, limits=None):
    """Encode known state classes, tuples, integers, None and finite tensors."""
    budget = _Budget(limits or CodecLimits())

    def visit(obj, depth):
        budget.visit(depth)
        if obj is None:
            return {"type": "none"}
        if type(obj) is int:
            if not -(2**63) <= obj < 2**63:
                raise CodecError("integer outside int64 range")
            return {"type": "int", "value": obj}
        if type(obj) is torch.Tensor:
            name = str(obj.dtype).removeprefix("torch.")
            if name not in _DTYPES:
                raise CodecError("unsupported tensor dtype")
            budget.tensor(list(obj.shape), obj.dtype)
            raw = tensor_bytes(obj)
            return {"type": "tensor", "dtype": name, "shape": list(obj.shape),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "data": base64.b64encode(raw).decode("ascii")}
        if type(obj) is tuple:
            if len(obj) > budget.limits.max_items:
                raise CodecError("tuple item limit exceeded")
            return {"type": "tuple", "items": [visit(v, depth + 1) for v in obj]}
        for name, (cls, fields) in _CLASSES.items():
            if type(obj) is cls:
                return {"type": name, "fields": {k: visit(getattr(obj, k), depth + 1) for k in fields}}
        raise CodecError("unsupported state type")

    return {"format": "nanocog-state", "version": 1, "value": visit(value, 0)}


def decode(document, *, device="cpu", limits=None):
    """Validate sizes and hashes before constructing allowlisted state objects.

    Hashes detect damage, not a malicious writer who can replace both data and
    digest. Authentication and access controls belong to the containing store.
    """
    if (type(document) is not dict or set(document) != {"format", "version", "value"}
            or document["format"] != "nanocog-state" or type(document["version"]) is not int
            or document["version"] != 1):
        raise CodecError("unsupported state format")
    if sys.byteorder != "little":
        raise CodecError("the v1 tensor format requires a little-endian host")
    budget = _Budget(limits or CodecLimits())

    def visit(obj, depth):
        budget.visit(depth)
        if type(obj) is not dict or type(obj.get("type")) is not str:
            raise CodecError("invalid state node")
        tag = obj["type"]
        if tag == "none" and set(obj) == {"type"}:
            return None
        if tag == "int" and set(obj) == {"type", "value"}:
            val = obj["value"]
            if type(val) is not int or not -(2**63) <= val < 2**63:
                raise CodecError("invalid int64 value")
            return val
        if tag == "tuple" and set(obj) == {"type", "items"}:
            if type(obj["items"]) is not list or len(obj["items"]) > budget.limits.max_items:
                raise CodecError("invalid tuple")
            return tuple(visit(v, depth + 1) for v in obj["items"])
        if tag in _CLASSES and set(obj) == {"type", "fields"}:
            cls, fields = _CLASSES[tag]
            if type(obj["fields"]) is not dict or set(obj["fields"]) != set(fields):
                raise CodecError("invalid state fields")
            return cls(**{k: visit(obj["fields"][k], depth + 1) for k in fields})
        if tag != "tensor" or set(obj) != {"type", "dtype", "shape", "sha256", "data"}:
            raise CodecError("unknown or malformed state node")
        if type(obj["dtype"]) is not str or obj["dtype"] not in _DTYPES:
            raise CodecError("unsupported tensor dtype")
        dtype = _DTYPES[obj["dtype"]]
        nbytes = budget.tensor(obj["shape"], dtype)
        data = obj["data"]
        digest = obj["sha256"]
        if (type(data) is not str or len(data) != 4 * ((nbytes + 2) // 3)
                or type(digest) is not str or len(digest) != 64):
            raise CodecError("invalid tensor byte length or digest")
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise CodecError("invalid tensor base64") from exc
        if (len(raw) != nbytes or hashlib.sha256(raw).hexdigest() != digest or
                base64.b64encode(raw).decode("ascii") != data):
            raise CodecError("tensor checksum or canonical encoding mismatch")
        if dtype == torch.bool and any(b not in (0, 1) for b in raw):
            raise CodecError("invalid Boolean tensor bytes")
        if nbytes:
            tensor = torch.frombuffer(bytearray(raw), dtype=dtype).clone().reshape(obj["shape"])
        else:
            tensor = torch.empty(obj["shape"], dtype=dtype)
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise CodecError("nonfinite tensor")
        return tensor.to(device=device)

    return visit(document["value"], 0)
