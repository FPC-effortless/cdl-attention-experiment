#!/usr/bin/env python3
"""Explicit opt-in CLI for the unqualified, bounded categorical experiment.

Run with the admitted private kit and one saved factual checkpoint. The host
policy accepts the model's typed predictions; acceptance does not prove truth.
This command does not change the ordinary memory CLI or native language model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from runner import load_adapter
from nanocog.memory.learned_archive import FactualMemory
from nanocog.memory.store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--principal", required=True)
    parser.add_argument("--namespace", default="factual-experiment")
    parser.add_argument("--session", required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--authority", default="experimental-model-policy")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--event")
    group.add_argument("--question")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    adapter = load_adapter(args.kit, args.checkpoint)
    with MemoryStore(args.db, principal_id=args.principal, namespace=args.namespace) as store:
        memory = FactualMemory(store, adapter, session_id=args.session, authority=args.authority)
        if args.event is not None:
            result = memory.observe(args.event, request_id=args.request_id)
        else:
            result = memory.ask(args.question, request_id=args.request_id)
    print(json.dumps({"mode": "experimental-categorical-factual-memory",
                      "unfamiliar_wording_qualified": False,
                      "result": asdict(result)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
