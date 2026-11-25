#!/usr/bin/env python3
"""No-model, bounded, local constructed COG-COMM-001 Forge intake."""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

SCHEMA = "cog-comm-001-intake-plan.v1"
SPLITS = ("train", "cal", "dev")
SPECIAL = ("<pad>", "<bos>", "<end>", "<word>", "</word>", "<answer>")
WORD = re.compile(r"[A-Za-z0-9]+|[^\w\s]", re.ASCII)
WRITE = re.compile(r"The clerk filed the ([A-Za-z]+) badge as ([a-z]+)\.")
NOOP = re.compile(r"The clerk proposed ([a-z]+) for the ([A-Za-z]+) badge but withdrew it\.")
ASK = re.compile(r"What color is the badge for ([A-Za-z]+)\?")
HASH = re.compile(r"[a-f0-9]{64}")
MAX_BYTES = 16 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def emit(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def pairs(keys):
    out = {}
    for key, value in keys:
        require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def reject_constant(value):
    raise ValueError("nonfinite JSON constant: " + value)


def read(path, bound=MAX_BYTES):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "regular input required")
        raw = stream.read(bound + 1)
    require(len(raw) <= bound, "input exceeded bounded read")
    return raw


def parse(raw):
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)


def exact(obj, names, message):
    require(type(obj) is dict and set(obj) == set(names), message)


def natural(x, lo, hi):
    return type(x) is int and lo <= x <= hi


def source_files(protocol):
    base = Path(__file__).resolve().parent
    spec = importlib.util.find_spec("forge_data")
    require(spec is not None and spec.origin is not None, "pinned native Forge not installed/on PYTHONPATH")
    forge = Path(spec.origin).parent
    return {"intake": base / "forge_intake.py", "generator": base / "cog_comm/data.py",
            "tokenizer_source": base / "cog_comm/objective.py",
            "project_init": base / "cog_comm/__init__.py", "protocol": Path(protocol),
            **{"forge_" + name: forge / (name + ".py") for name in ("__init__", "diagnostics", "dataset", "core")}}


def runtime_pin():
    exe = Path(sys.executable).resolve(strict=True)
    return {"implementation": sys.implementation.name, "version": sys.version,
            "entrypoint": sys.executable, "binary_sha256": digest(read(exe, 64*1024*1024)),
            "dont_write_bytecode": bool(sys.dont_write_bytecode), "optimize": sys.flags.optimize,
            "utf8_mode": sys.flags.utf8_mode}


def plan_checks(doc):
    exact(doc, ("schema", "source_kind", "sources", "runtime", "panels", "policy"), "unexpected plan fields")
    require(doc["schema"] == SCHEMA and doc["source_kind"] == "CONSTRUCTED",
            "only constructed project data is admitted")
    require(doc["runtime"] == runtime_pin(), "runtime/config pin mismatch")
    panels = doc["panels"]
    require(type(panels) is dict and set(panels) in ({"train", "dev"}, set(SPLITS)),
            "only TRAIN and DEV, with optional separate CAL; no TEST")
    seeds = []
    for p in panels.values():
        exact(p, ("count", "seed", "delays"), "invalid panel schema")
        require(natural(p["count"], 1, 2000) and natural(p["seed"], 0, 2**32-1),
                "invalid finite family/seed bound")
        require(type(p["delays"]) is list and 1 <= len(p["delays"]) <= 32
                and all(natural(d, 0, 256) for d in p["delays"])
                and len(p["delays"]) == len(set(p["delays"])), "distinct finite delays required")
        seeds.append(p["seed"])
    require(len(seeds) == len(set(seeds)), "panel generator seeds must differ")
    p = doc["policy"]
    exact(p, ("vocab_minimum", "min_families_per_delay", "min_worlds_per_answer",
              "local_suffix_tokens", "min_local_equal_dev_families", "block_tokens",
              "retained_blocks", "max_encoded_tokens"), "invalid policy schema")
    require(natural(p["vocab_minimum"], 1, 100), "bad vocabulary minimum")
    for k in ("min_families_per_delay", "min_worlds_per_answer"):
        exact(p[k], panels, "panel support must be specified for each panel")
        require(all(natural(x, 1, 2000) for x in p[k].values()), "invalid nonzero support")
    require(natural(p["block_tokens"], 1, 256) and natural(p["retained_blocks"], 1, 256)
            and natural(p["max_encoded_tokens"], 1, 4096), "invalid static geometry bounds")
    width, minimum = p["local_suffix_tokens"], p["min_local_equal_dev_families"]
    require((width is None and minimum is None) or
            (natural(width, 1, 4096) and natural(minimum, 1, panels["dev"]["count"])),
            "local-suffix width and positive eligibility support must be pinned together")


def inspect_prompt(text, colors):
    """Independent fixed grammar latest-write and alias-normalized signatures."""
    require(type(text) is str and text.isascii() and len(text) <= 25000, "prompt grammar bytes")
    parts = tuple(re.split(r"(?<=[.?]) ", text))
    require(len(parts) >= 5 and parts[-2] == "The clerk closed the ledger.", "closing grammar")
    asked = ASK.fullmatch(parts[-1])
    require(asked is not None, "question grammar")
    target = asked[1].lower()
    aliases = {target: 0}
    records = []
    state = {}
    for pos, sentence in enumerate(parts[:-2]):
        written, canceled = WRITE.fullmatch(sentence), NOOP.fullmatch(sentence)
        require((written is not None) != (canceled is not None), "unsupported event")
        if written:
            name, value = written[1].lower(), written[2]
            state[name] = value, pos
            kind = "write"
        else:
            name, value = canceled[2].lower(), canceled[1]
            kind = "noop"
        require(value in colors and name not in colors, "value/alias namespace conflict")
        key = aliases.setdefault(name, len(aliases))
        records.append((kind, key, value))
    require(target in state, "queried record lacks observed write")
    return parts, target, aliases, tuple(records), state[target]


def vocab_from_train(texts, answers, minimum):
    counts = collections.Counter(w for text in texts for w in WORD.findall(text) if w == w.lower())
    return tuple(sorted({w for w, n in counts.items() if n >= minimum} | set(answers)))


def encode(text, index):
    result = [1]
    for word in WORD.findall(text):
        if word in index:
            result.append(index[word])
        else:
            raw = word.lower().encode("utf8")
            require(raw.decode("utf8") == word.lower(), "unknown word byte representation")
            result.extend((3, *(len(SPECIAL) + byte for byte in raw), 4))
    result.append(5)
    return tuple(result)


def common_tail(a, b):
    for n, (x, y) in enumerate(zip(reversed(a), reversed(b))):
        if x != y:
            return n
    return min(len(a), len(b))


def write_raw(path, raw):
    require(len(raw) <= MAX_BYTES, "would exceed output bound")
    with path.open("xb") as out:
        out.write(raw)
    os.chmod(path, 0o600)
    return digest(raw)


def write(path, obj):
    return write_raw(path, emit(obj))


def write_jsonl(path, rows):
    return write_raw(path, b"".join(emit(row) for row in rows))


def role(out, split, actor, labels, audit_rows, counts):
    # Prediction input records only literal prompt text, not IDs or pair orientations.
    folder = out / split
    folder.mkdir(mode=0o700)
    if split != "train":
        write_jsonl(folder / "actor.jsonl", actor)
    truth = out / "operator-only" / "truth"
    write_jsonl(truth / (split + ".jsonl"), labels)
    counts[split]["actor_prompt_count"] = len(actor)
    counts[split]["checker_row_count"] = len(audit_rows)


def inspect_panels(generated, data, plan, out):
    policy, panels = plan["policy"], plan["panels"]
    answers = tuple(data.ANSWERS)
    require(len(answers) == len(set(answers)) == 8, "expected fixed eight-answer alphabet")
    names = {s: set(n.lower() for n in data.ENTITY_BANKS[s]) for s in generated}
    require(all(not names[a] & names[b] for a in names for b in names if a != b), "alias-bank overlap")
    counts, checks = {}, {}
    flat = [family for s in SPLITS for family in generated.get(s, ())]
    generator_report = data.audit_families(flat)
    words = vocab_from_train(data.model_view(generated["train"]), answers, policy["vocab_minimum"])
    index = {w: i for i, w in enumerate(words, start=len(SPECIAL)+256)}
    seen = {name: set() for name in ("family", "prompt", "encoded", "paired_content")}
    aliases_norm, structure_norm = {}, {}
    report = {}
    width = policy["local_suffix_tokens"]
    block, retention = policy["block_tokens"], policy["retained_blocks"]
    for s in SPLITS:
        if s not in generated:
            continue
        rows, actor, truth, diagnostics = generated[s], [], [], []
        require(len(rows) == panels[s]["count"], "generator under-run")
        n_delay, n_value = collections.Counter(), collections.Counter()
        pair_values, oriented, target_names = collections.Counter(), collections.Counter(), collections.Counter()
        eligible, local_equal, age_eligible = (collections.Counter() for _ in range(3))
        tails, distances, block_distances = (collections.defaultdict(list) for _ in range(3))
        orientation_order_control = collections.Counter()
        methods = ("first_color", "last_color", "color_bag", "suffix_hash", "position_hash")
        shortcut_counts = {k: collections.Counter() for k in methods}
        lengths = []
        alias_shape, struct_shape = set(), set()
        slot_values = collections.Counter()
        for f in rows:
            require(f.split == s and HASH.fullmatch(f.family_id) and f.family_id not in seen["family"],
                    "invalid or overlapping family identity")
            seen["family"].add(f.family_id)
            require(f.delay in panels[s]["delays"] and len(f.worlds) == 2, "missing or wrong family delay")
            a, b = f.worlds
            ra, rb = (inspect_prompt(w.prompt, answers) for w in (a, b))
            ea, eb = (encode(w.prompt, index) for w in (a, b))
            require(len(ea) <= policy["max_encoded_tokens"], "encoded input exceeds no-clip audit bound")
            lengths.extend((len(ea), len(eb)))
            require(len(ea) == len(eb) and collections.Counter(ea) == collections.Counter(eb),
                    "tokenized pair lengths or multisets differ")
            require(ra[0][-2:] == rb[0][-2:] and a.answer != b.answer and
                    a.support_positions == b.support_positions, "pair tail, opposing answer or support differs")
            require((a.world_id, b.world_id) == (0, 1), "world orientation changed")
            deltas = [(j, u, v) for j, (u, v) in enumerate(zip(ra[3], rb[3])) if u != v]
            require(len(ra[3]) == len(rb[3]) and len(deltas) == 2
                    and all(u[:2] == v[:2] and u[0] == "write" for _, u, v in deltas)
                    and len({u[1] for _, u, _ in deltas}) == 2 and 0 in {u[1] for _, u, _ in deltas}
                    and deltas[0][1][2] == deltas[1][2][2] and deltas[0][2][2] == deltas[1][1][2],
                    "independent swap structure mismatch")
            paired_sig = digest(emit(sorted((a.prompt, b.prompt))))
            require(paired_sig not in seen["paired_content"], "repeated or orientation-inverted exact family")
            seen["paired_content"].add(paired_sig)
            alias_shape.add(digest(emit(sorted((ra[3], rb[3])))))
            struct_shape.add(digest(emit(tuple((kind, key) for kind, key, _ in ra[3]))))
            tail = common_tail(ea, eb)
            tails[f.delay].append(tail)
            same_tail = width is not None and tail >= width
            if same_tail:
                local_equal[f.delay] += 1
            for k in methods:
                hits = [data.shortcut_predict(w.prompt, k) == w.answer for w in f.worlds]
                shortcut_counts[k]["both_correct_families"] += all(hits)
                shortcut_counts[k]["correct_worlds"] += sum(hits)
            n_delay[f.delay] += 1
            pair_values["/".join(sorted((a.answer, b.answer)))] += 1
            target_names[ra[1]] += 1
            offsets = []
            for w, render, enc in zip(f.worlds, (ra, rb), (ea, eb)):
                parts, target, aliases, _, (answer, pos) = render
                require(w.answer == answer and w.support_positions == (pos,)
                        and set(aliases) <= names[s], "independent answer, support or alias scope mismatch")
                require(f.family_id not in w.prompt and w.prompt not in seen["prompt"]
                        and enc not in seen["encoded"], "prompt metadata leak or exact/token overlap")
                seen["prompt"].add(w.prompt)
                seen["encoded"].add(enc)
                require(w.answer in index and index[w.answer] >= len(SPECIAL)+256,
                        "answer not an exact one-token vocabulary entry")
                source_before = " ".join(parts[:pos])
                start = len(encode(source_before, index)) - 1
                end = len(encode(" ".join(parts[:pos+1]), index)) - 2
                pos_offset = len(encode(parts[pos].rsplit(" ", 1)[0], index)) - 1
                source_token = start + pos_offset - 1  # document BOS counted only once
                require(enc[source_token] == index[answer] and start <= source_token <= end,
                        "answer-bearing source token alignment failed")
                query = len(enc) - 1  # actor consumes delimiter then predicts answer; gold not appended
                age = query - source_token
                blocks = query // block - source_token // block
                reachable = 0 < age and blocks < retention
                offsets.append((start, end, source_token, age, blocks, reachable))
                oriented[f"{w.world_id}:{w.answer}"] += 1
                slot_values[f"{f.delay}:{pos}:{w.answer}"] += 1
                n_value[w.answer] += 1
                actor.append({"family_id": f.family_id, "world_index": w.world_id, "prompt": w.prompt})
                truth.append({"family_id": f.family_id, "world_index": w.world_id,
                              "answer": w.answer, "delay": f.delay, "support_sentence": pos,
                              "support_start_token": start, "support_end_token": end,
                              "answer_source_token": source_token, "local_suffix_equal": same_tail,
                              "age_eligible": reachable})
                diagnostics.append({"id": f"{f.family_id}:{w.world_id}", "features": {"prompt": w.prompt},
                                    "groups": {"family": f.family_id, "queried_alias": target}})
            require(offsets[0] == offsets[1], "paired token support start/age mismatch")
            # Also check token start of both changed writes; equality preserves any fixed chunk boundary.
            for j, _, _ in deltas:
                require(len(encode(" ".join(ra[0][:j]), index)) ==
                        len(encode(" ".join(rb[0][:j]), index)), "changed-write token alignment")
            _, _, _, age, blocks, reachable = offsets[0]
            distances[f.delay].append(age)
            block_distances[f.delay].append(blocks)
            age_eligible[f.delay] += reachable
            eligible[f.delay] += bool(same_tail and reachable and age >= width)
            orientation_order_control[f"{a.answer}:{b.answer}"] += 1
        require(all(n_delay[d] >= policy["min_families_per_delay"][s] for d in panels[s]["delays"]),
                "panel below prospectively pinned delay support")
        require(all(n_value[v] >= policy["min_worlds_per_answer"][s] for v in answers),
                "panel below prospectively pinned answer support")
        if s != "train":
            require(not names[s] & set(words), "held alias appears in TRAIN vocabulary")
        if s == "dev" and width is not None:
            require(sum(eligible.values()) >= policy["min_local_equal_dev_families"],
                    "DEV below prospectively pinned identical-suffix support")
        aliases_norm[s], structure_norm[s] = alias_shape, struct_shape
        counts[s] = {"families": len(rows), "families_by_declared_delay": dict(n_delay),
                     "worlds_by_answer": dict(n_value), "unordered_answer_pair_support": dict(pair_values),
                     "answer_by_orientation": dict(oriented), "query_alias_count": len(target_names),
                     "query_alias_family_support_min": min(target_names.values()),
                     "answer_by_delay_and_support_slot": dict(slot_values),
                     "local_equal_families_by_delay": None if width is None else dict(local_equal),
                     "source_age_eligible_families_by_delay": dict(age_eligible),
                     "local_equal_and_source_age_eligible_by_delay": None if width is None else dict(eligible),
                     "orientation_pair_support": dict(orientation_order_control),
                     "encoded_length": {"min": min(lengths), "p50": sorted(lengths)[len(lengths)//2],
                                        "p95": sorted(lengths)[(95*len(lengths)+99)//100-1],
                                        "max": max(lengths), "sum": sum(lengths)},
                     "shortcut_counts": {k: dict(v) for k,v in shortcut_counts.items()},
                     "fixed_grammar_reference": {"correct_worlds": len(actor), "both_correct_families": len(rows)},
                     "answer_source_block_age_minmax": {k: [min(v), max(v)] for k,v in block_distances.items()},
                     "support_age_minmax_by_delay": {k: [min(v), max(v)] for k, v in distances.items()},
                     "encoded_common_tail_minmax_by_delay": {k: [min(v), max(v)] for k, v in tails.items()}}
        role(out, s, actor, truth, diagnostics, counts)
        checks[s] = diagnostics
    for i, s in enumerate(SPLITS):
        for t in SPLITS[i+1:]:
            if s in checks and t in checks:
                report[f"{s}:{t}"] = {"alias_normalized_family_overlap": len(aliases_norm[s] & aliases_norm[t]),
                                      "event_role_structure_overlap": len(structure_norm[s] & structure_norm[t])}
    # The trainer may consume train answers, never support positions or hidden evaluation truth.
    write_jsonl(out / "train" / "families.jsonl",
                ({"split": "train", "prompts": [w.prompt for w in f.worlds],
                  "answers": [w.answer for w in f.worlds]} for f in generated["train"]))
    write(out / "train" / "vocabulary.json", {"special": SPECIAL, "words": words, "minimum": policy["vocab_minimum"],
                                            "answers": answers, "shape": len(SPECIAL)+256+len(words)})
    return counts, checks, report, generator_report


def run(plan, protocol, output, expected_plan):
    raw = read(plan, 65536)
    require(HASH.fullmatch(expected_plan) and digest(raw) == expected_plan, "expected plan digest mismatch")
    doc = parse(raw)
    plan_checks(doc)
    files = source_files(protocol)
    exact(doc["sources"], files, "missing or extraneous source digest")
    for name, path in files.items():
        require(doc["sources"][name] == digest(read(path)), "source pin mismatch: " + name)
    # These checkout modules are trusted experiment operator code, not sandboxed.
    from cog_comm import data
    from forge_data.diagnostics import diagnose
    dest = Path(output).absolute()
    require(not dest.exists() and not dest.is_symlink(), "existing output refused")
    require(dest.parent.is_dir() and not dest.parent.is_symlink(), "trusted real parent directory required")
    with tempfile.TemporaryDirectory(prefix=".cog-comm-audit-", dir=dest.parent) as tmp:
        stage = Path(tmp) / "candidate"
        stage.mkdir(mode=0o700)
        (stage / "operator-only" / "truth").mkdir(mode=0o700, parents=True)
        os.chmod(stage / "operator-only", 0o700)
        generated = {s: data.generate(s, p["count"], p["seed"], delays=p["delays"])
                     for s, p in doc["panels"].items()}
        counts, diagnostics, templates, generator_report = inspect_panels(generated, data, doc, stage)
        forge = {}
        for evaluation in ("cal", "dev"):
            if evaluation not in diagnostics:
                continue
            ds = diagnostics["train"] + diagnostics[evaluation]
            for j, row in enumerate(ds):
                row["split"] = "train" if j < len(diagnostics["train"]) else "validation"
            d = {"schema": "forge.diagnostic-input.v1", "feature_fields": ["prompt"],
                 "group_fields": ["family", "queried_alias"], "rows": ds}
            fp = stage / "operator-only" / f"forge-train-{evaluation}-input.json"
            pin = write(fp, d)
            native = diagnose(fp, sha256=pin, max_rows=len(ds), max_findings=0)
            coverage = {c["name"]: c["status"] for c in native["coverage"]}
            require(native["state"] == "COMPLETE" and native["findings_total"] == 0
                    and all(coverage.get(k) == "TESTED" for k in
                            ("completeness", "split_content", "group:family", "group:queried_alias"))
                    and coverage.get("labels") == "UNSUPPORTED", "native Forge rejected exact role audit")
            op = stage / "operator-only" / f"forge-train-{evaluation}-receipt.json"
            forge[evaluation] = {"input_sha256": pin, "receipt_sha256": write(op, native),
                                 "coverage": coverage}
        summary = {"schema": "cog-comm-001-intake-report.v1", "state": "PREPARED_ONLY",
                   "source_kind": "CONSTRUCTED", "claim_eligibility": "NOT_QUALIFIED",
                   "independent_review": "PENDING", "fit_authorized": False, "fit_count": 0,
                   "protected_test_access": False, "plan_sha256": expected_plan, "sources": doc["sources"],
                   "runtime": doc["runtime"],
                   "scope": "One fixed grammar, held aliases; not new event structure, fit or model evidence.",
                   "local_suffix_width": doc["policy"]["local_suffix_tokens"],
                   "block_tokens": doc["policy"]["block_tokens"],
                   "retained_blocks": doc["policy"]["retained_blocks"],
                   "encoded_no_clip_audit_bound": doc["policy"]["max_encoded_tokens"],
                   "source_age_scope": "Color token to causal answer delimiter; age eligibility does not demonstrate admission or survival.",
                   "panels": counts, "normalized_content_signatures": templates,
                   "generator_internal_checks": generator_report, "forge_native": forge}
        summary["artifact_sha256"] = {p.relative_to(stage).as_posix(): digest(read(p))
                                      for p in sorted(stage.rglob("*")) if p.is_file()}
        write(stage / "INTAKE_REPORT.json", summary)
        # mkdir is an exclusive output reservation. No overwrite of another receipt on repeated run.
        dest.mkdir(mode=0o700)
        os.replace(stage, dest)
    return {"state": summary["state"], "independent_review": summary["independent_review"],
            "report_sha256": digest(read(dest / "INTAKE_REPORT.json")),
            "families": {s: p["families"] for s, p in counts.items()}, "fit_count": 0}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--protocol", required=True, type=Path)
    cli.add_argument("--source-pins", action="store_true", help="print source hashes only; no data generated")
    cli.add_argument("--runtime-pin", action="store_true", help="print runtime pin only; no data generated")
    cli.add_argument("--plan", type=Path)
    cli.add_argument("--plan-sha256")
    cli.add_argument("--out", type=Path)
    a = cli.parse_args()
    if a.source_pins or a.runtime_pin:
        require(a.plan is None and a.plan_sha256 is None and a.out is None
                and not (a.source_pins and a.runtime_pin), "pins-only is separate")
    if a.source_pins:
        print(json.dumps({k: digest(read(path)) for k, path in source_files(a.protocol).items()}, sort_keys=True))
    elif a.runtime_pin:
        print(json.dumps(runtime_pin(), sort_keys=True))
    else:
        require(a.plan is not None and a.plan_sha256 is not None and a.out is not None, "plan, pin and out required")
        print(json.dumps(run(a.plan, a.protocol, a.out, a.plan_sha256), sort_keys=True))


if __name__ == "__main__":
    main()
