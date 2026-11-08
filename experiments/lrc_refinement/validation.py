"""Independent, audit-only checks for the existing location-payload task.

These functions may inspect semantic metadata and support labels to verify the
experiment. They must never be used to form keys, choose probes, filter memory,
or otherwise supply information to inference. Bridge quartets retain the
original validator in ``experiments.lrc_route_repair.bridge_data``.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib

import torch

from lrc_vb001_comparison.source import study as vb
from experiments.lrc_route_repair.bridge_data import _parse_record


DATA_KEYS = frozenset({"records", "query", "answer", "support", "era", "entity", "person"})
LOGICAL_FACTS = frozenset((era, person, relation)
                          for era in range(2)
                          for person in range(4)
                          for relation in range(2))


def _require_cpu_long(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.dtype != torch.long:
        raise ValueError(f"{name} must be a torch.long tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for the dataset audit")
    return value


def validate_location_quartets(data: Mapping[str, torch.Tensor], heldout: bool) -> dict:
    """Verify the unmodified ``vb.dataset`` semantics, returning an audit receipt.

    A quartet is four *separate* worlds in the generator's canonical answer
    order 0, 1, 2, 3. Only the final supporting location's payload token may
    change. The held-out rule concerns the queried entity/partner pair, not
    every unrelated partnership appearing in the snapshot.
    """
    if type(heldout) is not bool:
        raise ValueError("heldout must be a bool")
    if not isinstance(data, Mapping) or set(data) != DATA_KEYS:
        raise ValueError("wrong dataset keys")
    for name, value in data.items():
        _require_cpu_long(value, name)
    if data["answer"].ndim != 1:
        raise ValueError("answer must be one-dimensional")
    n = data["answer"].shape[0]
    expected_shapes = {
        "records": (n, vb.N_RECORDS, vb.MAX_WORDS),
        "query": (n, vb.MAX_WORDS),
        "support": (n, 2),
        **{name: (n,) for name in ("answer", "era", "entity", "person")},
    }
    if n == 0 or n % 4 or any(tuple(data[name].shape) != shape
                              for name, shape in expected_shapes.items()):
        raise ValueError("invalid quartet count or tensor shapes")

    changed_token_count = 0
    expected_answers = torch.arange(4, dtype=torch.long)
    for start in range(0, n, 4):
        end = start + 4
        records = data["records"][start:end]
        queries = data["query"][start:end]
        supports = data["support"][start:end]
        era = int(data["era"][start])
        entity = int(data["entity"][start])
        person = int(data["person"][start])
        if not (0 <= era < 2 and 0 <= entity < 4 and 0 <= person < 4):
            raise ValueError("metadata out of range")
        if any(not bool(torch.all(data[name][start:end] == value))
               for name, value in (("era", era), ("entity", entity), ("person", person))):
            raise ValueError("location quartet metadata changes")
        expected_query = torch.tensor(vb.encode([
            "During", "the", vb.ERAS[era], "era", "where", "did", vb.NAMES[entity],
            "'s", "partner", "live", "?",
        ]), dtype=torch.long)
        if not bool(torch.all(queries == expected_query[None])):
            raise ValueError("query changes or disagrees with metadata")
        if not torch.equal(data["answer"][start:end], expected_answers):
            raise ValueError("location quartet must enumerate answers 0, 1, 2, 3")
        if not bool(torch.all(supports == supports[0][None])):
            raise ValueError("support occurrences change within location quartet")
        first, second = (int(value) for value in supports[0])
        if not (0 <= first < vb.N_RECORDS and 0 <= second < vb.N_RECORDS and first != second):
            raise ValueError("invalid support occurrences")
        is_withheld_pair = person == (entity + 1) % 4
        if is_withheld_pair != heldout:
            raise ValueError("queried entity/partner pair violates the requested split")

        for variant in range(4):
            differences = records[variant] != records[0]
            allowed = torch.zeros_like(differences)
            if variant:
                allowed[second, 7] = True
            if not torch.equal(differences, allowed):
                raise ValueError("only the second support's location payload may change")
            changed_token_count += int(differences.sum())

            parsed = [_parse_record(record) for record in records[variant]]
            keys = [fact[:3] for fact in parsed]
            if set(keys) != LOGICAL_FACTS or len(set(keys)) != vb.N_RECORDS:
                raise ValueError("missing or duplicate logical fact")
            if any(fact[0] != position // 8 for position, fact in enumerate(parsed)):
                raise ValueError("records do not preserve the two era blocks")
            lookup = {fact[:3]: (position, fact[3])
                      for position, fact in enumerate(parsed)}
            expected_first, expected_person = lookup[era, entity, 0]
            expected_second, expected_answer = lookup[era, expected_person, 1]
            if expected_person != person or (first, second) != (expected_first, expected_second):
                raise ValueError("metadata or support labels disagree with the two-hop rule")
            if expected_answer != variant or int(data["answer"][start + variant]) != expected_answer:
                raise ValueError("answer label disagrees with the selected location")

    return {
        "status": "passed",
        "kind": "location_payload",
        "heldout": heldout,
        "examples": n,
        "quartets": n // 4,
        "only_location_payload_changes": True,
        "changed_tokens_vs_quartet_reference": changed_token_count,
        "query_first_value_supports_and_other_facts_fixed": True,
        "all_four_answers_per_quartet": True,
        "unique_gold_supports_and_rule": True,
        "split_rule": "cyclic_next_pair" if heldout else "excluding_cyclic_next_pair",
        "same_schema_as_vb_dataset": True,
        "data_sha256": vb.digest_data(data),
    }


def _input_tensors(data: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(data, Mapping) or not {"records", "query"}.issubset(data):
        raise ValueError("records and query are required for input signatures")
    records = _require_cpu_long(data["records"], "records")
    queries = _require_cpu_long(data["query"], "query")
    if records.ndim != 3 or tuple(records.shape[1:]) != (vb.N_RECORDS, vb.MAX_WORDS):
        raise ValueError("invalid record shape for input signatures")
    if tuple(queries.shape) != (records.shape[0], vb.MAX_WORDS):
        raise ValueError("invalid query shape for input signatures")
    return records, queries


def input_signatures(data: Mapping[str, torch.Tensor]) -> set[bytes]:
    """SHA-256 identities of complete ordered model inputs, excluding all labels.

    Inputs have the fixed task shapes. Signed 64-bit little-endian serialization
    makes the digest unambiguous and independent of tensor strides. An empty
    correctly shaped input panel produces an empty set. Exact repeated inputs
    collapse here; ``split_overlap`` reports the resulting duplicate counts.
    """
    records, queries = _input_tensors(data)
    record_array = records.numpy().astype("<i8", copy=False)
    query_array = queries.numpy().astype("<i8", copy=False)
    signatures = set()
    for record_row, query_row in zip(record_array, query_array):
        digest = hashlib.sha256(b"LRC-REFINEMENT-001/model-input/v1\0")
        digest.update(b"records\0")
        digest.update(record_row.tobytes(order="C"))
        digest.update(b"query\0")
        digest.update(query_row.tobytes(order="C"))
        signatures.add(digest.digest())
    return signatures


def split_overlap(train: Mapping[str, torch.Tensor],
                  datasets: Mapping[str, Mapping[str, torch.Tensor]]) -> dict:
    """Count exact input overlap with TRAIN and between each pair of fresh panels.

    Counts refer to unique complete inputs, not to labels or just questions.
    ``all_disjoint`` concerns cross-panel intersections; within-panel duplicate
    counts remain separate so a caller can apply its declared duplicate policy.
    """
    if not isinstance(datasets, Mapping) or any(not isinstance(name, str) for name in datasets):
        raise ValueError("datasets must map string names to input panels")
    train_records, _ = _input_tensors(train)
    train_signatures = input_signatures(train)
    names = sorted(datasets)
    signatures = {name: input_signatures(datasets[name]) for name in names}
    with_train = {name: len(train_signatures & signatures[name]) for name in names}
    between_fresh = {
        left: {right: len(signatures[left] & signatures[right]) for right in names[i + 1:]}
        for i, left in enumerate(names)
    }
    return {
        "train_examples": train_records.shape[0],
        "train_unique_inputs": len(train_signatures),
        "train_duplicate_inputs": train_records.shape[0] - len(train_signatures),
        "dataset_unique_inputs": {name: len(signatures[name]) for name in names},
        "within_dataset_duplicate_inputs": {
            name: datasets[name]["records"].shape[0] - len(signatures[name]) for name in names
        },
        "with_train": with_train,
        "between_fresh": between_fresh,
        "all_disjoint": not any(with_train.values()) and not any(
            count for row in between_fresh.values() for count in row.values()),
    }
