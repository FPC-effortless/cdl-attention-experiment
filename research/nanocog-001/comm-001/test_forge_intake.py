"""Inert no-model fixtures exercising exact Forge integration and fail-closed guards."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import forge_intake as audit
from cog_comm import data


class IntakeTest(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name)
        self.protocol = self.root / "fixture-protocol.md"
        self.protocol.write_text("Inert diagnostic; not production split seeds.\n")
        self.plan = {
            "schema": audit.SCHEMA, "source_kind": "CONSTRUCTED", "runtime": audit.runtime_pin(),
            "sources": {k: audit.digest(audit.read(path)) for k, path in audit.source_files(self.protocol).items()},
            "panels": {s: {"count": 16, "seed": 181 + i, "delays": [0, 2]} for i,s in enumerate(audit.SPLITS)},
            "policy": {"vocab_minimum": 5, "local_suffix_tokens": 8, "min_local_equal_dev_families": 16,
                       "min_families_per_delay": dict.fromkeys(audit.SPLITS, 8),
                       "min_worlds_per_answer": dict.fromkeys(audit.SPLITS, 4),
                       "block_tokens": 8, "retained_blocks": 64, "max_encoded_tokens": 1024}}
        self.plan_path = self.root / "plan.json"
        self.out = self.root / "out"

    def invoke(self):
        raw = audit.emit(self.plan)
        self.plan_path.write_bytes(raw)
        return audit.run(self.plan_path, self.protocol, self.out, audit.digest(raw))

    def test_roles_native_and_no_model(self):
        summary = self.invoke()
        self.assertEqual(summary["state"], "PREPARED_ONLY")
        report = audit.parse(audit.read(self.out / "INTAKE_REPORT.json"))
        self.assertFalse(report["fit_authorized"])
        self.assertEqual(report["forge_native"]["dev"]["coverage"]["labels"], "UNSUPPORTED")
        self.assertEqual(report["panels"]["dev"]["fixed_grammar_reference"]["correct_worlds"], 32)
        read_lines = lambda f: [audit.parse(x) for x in audit.read(self.out / f).splitlines()]
        trainers = read_lines("train/families.jsonl")
        self.assertTrue(all(r["split"] == "train" and set(r) == {"split", "prompts", "answers"} for r in trainers))
        actor = read_lines("dev/actor.jsonl")
        truth = read_lines("operator-only/truth/dev.jsonl")
        self.assertTrue(all(set(r) == {"family_id", "world_index", "prompt"} for r in actor))
        self.assertTrue(all(r["local_suffix_equal"] and r["age_eligible"] for r in truth))
        for record, gold in zip(actor, truth):
            self.assertEqual((record["family_id"], record["world_index"]), (gold["family_id"], gold["world_index"]))
        for path, sha in report["artifact_sha256"].items():
            self.assertEqual(audit.digest(audit.read(self.out / path)), sha)
        with self.assertRaisesRegex(ValueError, "existing output"):
            self.invoke()

    def test_wrong_source_and_test_roles_refused(self):
        self.plan["sources"]["generator"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "source pin mismatch"):
            self.invoke()
        self.plan["panels"]["test"] = dict(self.plan["panels"]["dev"])
        with self.assertRaisesRegex(ValueError, "no TEST"):
            self.invoke()
        self.assertFalse(self.out.exists())

    def test_duplicate_inverted_pair_and_wrong_truth_refused(self):
        g = data.generate
        for case in ("duplicate", "bad truth"):
            def mutated(split, *args, **kw):
                rows = g(split, *args, **kw)
                if split != "dev":
                    return rows
                if case == "duplicate":
                    rows = (replace(rows[1], worlds=tuple(replace(w, world_id=i)
                            for i,w in enumerate(reversed(rows[0].worlds)))), *rows[1:])
                else:
                    old = rows[0].worlds
                    rows = (replace(rows[0], worlds=(replace(old[0], answer=old[1].answer), old[1])), *rows[1:])
                return rows
            with patch.object(data, "generate", mutated), self.assertRaises(ValueError):
                self.invoke()
            self.assertFalse(self.out.exists())

    def test_insufficient_support_and_age_ineligible_refused(self):
        self.plan["policy"]["min_worlds_per_answer"]["train"] = 5
        with self.assertRaisesRegex(ValueError, "answer support"):
            self.invoke()
        self.plan["policy"]["min_worlds_per_answer"]["train"] = 4
        self.plan["policy"]["retained_blocks"] = 1
        with self.assertRaisesRegex(ValueError, "identical-suffix support"):
            self.invoke()
        self.assertFalse(self.out.exists())

    def test_runtime_and_nonretained_forge_finding_refused(self):
        original = self.plan["runtime"]
        self.plan["runtime"] = {**original, "optimize": original["optimize"] + 1}
        with self.assertRaisesRegex(ValueError, "runtime/config pin mismatch"):
            self.invoke()
        self.plan["runtime"] = original
        from forge_data.diagnostics import diagnose
        def nonretained(*a, **kw):
            return {**diagnose(*a, **kw), "findings_total": 1}
        with patch("forge_data.diagnostics.diagnose", nonretained), self.assertRaisesRegex(ValueError, "native Forge rejected"):
            self.invoke()
        self.assertFalse(self.out.exists())

    def test_exact_byte_word_pieces(self):
        vocab = audit.vocab_from_train(["The Ada badge blue."] * 5, ("blue", "red"), 5)
        self.assertNotIn("ada", vocab)
        self.assertNotIn("the", vocab)
        self.assertIn("badge", vocab)
        self.assertIn("red", vocab)
        positions = {w:i+len(audit.SPECIAL)+256 for i,w in enumerate(vocab)}
        a = audit.encode("Ada", positions)
        b = audit.encode("Mara", positions)
        self.assertNotEqual(a, b)
        self.assertEqual(bytes(i-len(audit.SPECIAL) for i in a[2:-2]).decode(), "ada")


if __name__ == '__main__':
    unittest.main()
