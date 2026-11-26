"""Local memory inspection and integration CLI; this is not a language model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import uuid

from .session import MemorySession
from .store import (ConflictError, DeletedError, IntegrityError, MemoryStore,
                    ReceiptExpiredError, default_path)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Durable NanoCOG memory. Records and recalls data; does not generate AI answers.")
    p.add_argument("--db", type=Path, default=None,
                   help="SQLite file; default is the configured persistent NanoCOG state path")
    p.add_argument("--principal", required=True,
                   help="stable application-provided identity; a local scope, not authentication")
    p.add_argument("--namespace", default="default", help="project or memory scope")
    p.add_argument("--session", help="conversation identity; record creates one when omitted")
    commands = p.add_subparsers(dest="command", required=True)

    record = commands.add_parser("record", help="save an observation automatically")
    record.add_argument("text")
    record.add_argument("--role", choices=("user", "assistant", "tool"), default="user")
    record.add_argument("--event-id", help="reuse for an exact retry")

    recall = commands.add_parser("recall", help="retrieve source-labelled memory across chats")
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=8)
    recall.add_argument("--max-chars", type=int, default=8000)

    proposal = commands.add_parser("propose", help="record an unaccepted interpretation")
    proposal.add_argument("key")
    proposal.add_argument("value_json", help="JSON value, quoted according to your shell")
    proposal.add_argument("source_event_id")
    proposal.add_argument("--proposal-id", default=None)
    proposal.add_argument("--expected-revision", type=int, default=0)

    accept = commands.add_parser("accept", help="explicitly accept a proposal through host authority")
    accept.add_argument("proposal_id")
    accept.add_argument("--authority", required=True,
                        help="host-controlled provenance for this acceptance, not a model confidence score")

    facts = commands.add_parser("facts", help="inspect current accepted facts")
    facts.add_argument("key", nargs="?")

    forget = commands.add_parser("forget", help="redact one source and its dependent stored claims")
    forget.add_argument("event_id")
    forget_session = commands.add_parser("forget-session", help="forget one conversation's records")
    forget_session.add_argument("session_id")

    export = commands.add_parser("export", help="export this principal/namespace as inspectable JSON")
    export.add_argument("destination", type=Path)
    backup = commands.add_parser("backup", help="make a local backup of the entire database, all scopes")
    backup.add_argument("destination", type=Path)
    restore = commands.add_parser("restore", help="restore a verified full backup to a new database path")
    restore.add_argument("source", type=Path)
    restore.add_argument("destination", type=Path)
    return p


def _reject_nonfinite(value: str):
    raise ValueError(f"nonfinite JSON constant is not allowed: {value}")


def execute(args: argparse.Namespace):
    if args.command == "restore":
        return MemoryStore.restore_backup(args.source, args.destination)
    with MemoryStore(args.db if args.db is not None else default_path(),
                     principal_id=args.principal, namespace=args.namespace) as store:
        if args.command == "record":
            return store.append_message(args.session or str(uuid.uuid4()), args.role, args.text,
                                        event_id=args.event_id or str(uuid.uuid4()))
        if args.command == "recall":
            return MemorySession(store, session_id=args.session, recall_limit=args.limit,
                                 max_context_chars=args.max_chars).context(args.query)
        if args.command == "propose":
            value = json.loads(args.value_json, parse_constant=_reject_nonfinite)
            return store.propose(args.key, value, args.source_event_id,
                                 proposal_id=args.proposal_id or str(uuid.uuid4()),
                                 expected_revision=args.expected_revision)
        if args.command == "accept":
            return store.accept(args.proposal_id, authority=args.authority)
        if args.command == "facts":
            return store.facts(args.key)
        if args.command == "forget":
            return store.forget_event(args.event_id)
        if args.command == "forget-session":
            return store.forget_session(args.session_id)
        if args.command == "export":
            return store.export_json(args.destination)
        if args.command == "backup":
            return store.backup(args.destination)
        raise ValueError("unsupported command")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = execute(args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return 0
    except (ConflictError, DeletedError, IntegrityError, ReceiptExpiredError,
            ValueError, TypeError, OSError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
