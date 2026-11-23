"""Independent, inference-free corpus audit; uses no candidate executor code."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def audit(corpus_path: Path, manifest_path: Path):
    raw = corpus_path.read_bytes()
    corpus = json.loads(raw)
    manifest = json.loads(manifest_path.read_text())
    entities = tuple(manifest["entities"])
    unknown = manifest["unknown"]
    seen_semantics, seen_text = {}, {}
    failures, counts, words = [], {}, {}
    for split, families in corpus.items():
        count = Counter(families=len(families), worlds=2 * len(families))
        split_words = set()
        if sha(families) != manifest["split_hashes"][split]:
            failures.append([split, "manifest_split_hash"])
        for family in families:
            rendered = []
            world_answers = []
            for world in family["worlds"]:
                ledger, meaningful_operations = {}, []
                for position, event in enumerate(world["events"]):
                    op, entity, relation, value = (event[x] for x in ("operation", "entity", "relation", "value"))
                    if op == 1:
                        ledger[(entity, relation)] = value
                        meaningful_operations.append([op, entity, relation, value])
                    elif op == 2:
                        ledger.pop((entity, relation), None)
                        meaningful_operations.append([op, entity, relation])
                    elif op != 0:
                        failures.append([split, family["family_id"], "invalid_gold_operation", position])
                    expected = [[e, r, v] for (e, r), v in sorted(ledger.items())]
                    if expected != world["post_event_ledgers"][position]:
                        failures.append([split, family["family_id"], "post_event_ledger", position])
                    count["events"] += 1
                    split_words.update(re.findall(r"[A-Za-z]+", event["text"]))
                entity, plan = world["query_entity"], world["query_plan"]
                first_key = (entity, 0 if plan == 0 else 1)
                value = ledger.get(first_key, unknown)
                supports = [list(first_key)]
                evidence = [value]
                if plan == 2:
                    if value != unknown:
                        supports.append([value, 0])
                        value = ledger.get((value, 0), unknown)
                    evidence.append(value)
                text = ("unknown" if value == unknown else entities[value]) + "."
                if (value, text, supports, evidence) != (
                        world["answer"], world["answer_text"], world["supports"], world["evidence_values"]):
                    failures.append([split, family["family_id"], "query_labels"])
                world_answers.append(value)
                signature = sha([meaningful_operations, entity, plan])
                if signature in seen_semantics and seen_semantics[signature][0] != split:
                    failures.append([split, family["family_id"], "cross_split_semantics", seen_semantics[signature]])
                seen_semantics[signature] = [split, family["family_id"]]
                text_signature = sha([[event["text"] for event in world["events"]], world["query"]])
                if text_signature in seen_text and seen_text[text_signature][0] != split:
                    failures.append([split, family["family_id"], "cross_split_text", seen_text[text_signature]])
                seen_text[text_signature] = [split, family["family_id"]]
                rendered.append(text_signature)
                split_words.update(re.findall(r"[A-Za-z]+", world["query"]))
            if rendered[0] == rendered[1]:
                failures.append([split, family["family_id"], "identical_world_text"])
            count["answer_changing" if world_answers[0] != world_answers[1] else "invariant_or_unknown"] += 1
        counts[split] = dict(count)
        words[split] = split_words
    observed_sha = hashlib.sha256(raw).hexdigest()
    if observed_sha != manifest["sha256"]:
        failures.append(["corpus", "manifest_hash"])
    train_templates = {t for values in manifest["training_templates"].values() for t in values}
    fresh_templates = {t for values in manifest["held_syntax_templates"].values() for t in values}
    if train_templates & fresh_templates:
        failures.append(["templates", "held_wording_overlap"])
    return {
        "schema": "cog-fusion-independent-data-review-v1",
        "status": "PASS" if not failures else "FAIL",
        "corpus_sha256": observed_sha,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "reviewer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": counts,
        "failures": failures,
        "fresh_words_absent_from_task_train": sorted(words["fresh"] - words["train"]),
        "interpretation": "Fresh wording changes vocabulary and syntax; entity vocabulary remains fixed.",
        "scope": "No candidate module imports, model calls, fitting, threshold selection or checkpoint selection.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.corpus, args.manifest)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
