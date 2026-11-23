"""Command-line access to a real trained checkpoint and persisted session."""
import argparse
import json
from pathlib import Path
import uuid

import torch

from .session import Session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--scope", default="world")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("action", choices=("observe", "ask", "status"))
    parser.add_argument("text", nargs="?")
    args = parser.parse_args()
    if args.action != "status" and not args.text:
        parser.error("observe and ask require text")
    torch.set_num_threads(1)
    with Session.from_checkpoint(args.checkpoint, args.db, scope=args.scope, seed=args.seed) as session:
        request = args.request_id or uuid.uuid4().hex
        if args.action == "observe":
            result = session.observe(args.text, request)
        elif args.action == "ask":
            result = session.ask(args.text, request, temperature=args.temperature)
        else:
            result = {"revision": session.revision, "counters": session.counters,
                      "model_identity": session.model_id, "scope": session.scope}
        if args.json or args.action == "status":
            print(json.dumps(result, indent=2))
        elif args.action == "ask":
            print(result["text"])
        else:
            print("Observation committed. Predicted operation:", result["predicted_operation"])


if __name__ == "__main__":
    main()
