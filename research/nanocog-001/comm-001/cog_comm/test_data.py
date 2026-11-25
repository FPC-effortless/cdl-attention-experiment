"""Unit fixtures only; no actual future validation panel or training."""
from collections import Counter, defaultdict
from dataclasses import replace
import unittest

from cog_comm.data import (
    ANSWERS, ENTITY_BANKS, generate, model_view, sentences, shortcut_fixture_summary,
    tokenize, tokenize_chars, audit_families, NOOP_RE, WRITE_RE,
)


def independent_latest_value(prompt):
    """Independent reconstruction from raw text, without importing task regexes."""
    statements = prompt.split(". ")
    asked = statements[-1].split()[-1].rstrip("?")
    sequence = []
    for i, stmt in enumerate(statements[:-1]):
        words = stmt.removesuffix(".").split()
        if words[:4] == ["The", "clerk", "filed", "the"] and words[4] == asked:
            sequence.append((words[-1], i))
    return sequence[-1]


class DataTest(unittest.TestCase):
    def test_determinism_and_single_raw_view(self):
        a = generate("train", 12, 44)
        self.assertEqual(a, generate("train", 12, 44))
        self.assertNotEqual(a, generate("train", 12, 45))
        self.assertEqual(len(model_view(a)), 24)
        self.assertEqual(len(set(model_view(a))), 24)
        self.assertTrue(all(type(x) is str for x in model_view(a)))
        self.assertTrue(all(f.family_id not in w.prompt for f in a for w in f.worlds))

    def test_same_bag_same_suffix_exact_balanced_factorization(self):
        rows = generate("train", 28, 221, delays=(0, 1, 4, 8, 16))
        self.assertEqual(audit_families(rows)["worlds"], 56)
        pairs = set()
        counts = defaultdict(Counter)
        for f in rows:
            a, b = f.worlds
            self.assertEqual(sentences(a.prompt)[-3:], sentences(b.prompt)[-3:])
            self.assertEqual(Counter(tokenize(a.prompt)), Counter(tokenize(b.prompt)))
            self.assertEqual(len(tokenize(a.prompt)), len(tokenize(b.prompt)))
            self.assertNotEqual(a.answer, b.answer)
            self.assertEqual(a.support_positions, b.support_positions)
            counts[f.delay].update((a.answer, b.answer))
            pairs.add(frozenset((a.answer, b.answer)))
            for w in f.worlds:
                value, position = independent_latest_value(w.prompt)
                self.assertEqual((w.answer, w.support_positions), (value, (position,)))
                self.assertTrue(any(NOOP_RE.fullmatch(s) for s in sentences(w.prompt)))
            changed = [i for i, (x, y) in enumerate(zip(sentences(a.prompt), sentences(b.prompt)))
                       if x != y]
            self.assertEqual(len(changed), 2)
            self.assertIn(a.support_positions[0], changed)
            self.assertTrue(all(WRITE_RE.fullmatch(sentences(a.prompt)[i]) for i in changed))
        self.assertEqual(len(pairs), 28)
        self.assertEqual(Counter(w.answer for f in rows for w in f.worlds),
                         Counter(dict.fromkeys(ANSWERS, 7)))
        for counts_for_delay in counts.values():
            self.assertEqual(len(set(counts_for_delay.values())), 1)

    def test_split_aliases_never_cross_but_answer_alphabet_shared(self):
        sets = {split: set(aliases) for split, aliases in ENTITY_BANKS.items()}
        for i, (name, known) in enumerate(sets.items()):
            for other, aliens in sets.items():
                if name != other:
                    self.assertFalse(known & aliens)
            rows = generate(name, 8, 51+i)
            self.assertTrue(audit_families(rows))
            self.assertEqual({w.answer for f in rows for w in f.worlds}, set(ANSWERS))
            self.assertEqual(len({f.family_id for f in rows}), 8)
        combined = [f for split in sets for f in generate(split, 8, 51)]
        self.assertEqual(audit_families(combined)["families"], 24)
        self.assertNotEqual(tokenize_chars("Ada"), tokenize_chars("Mara"))
        self.assertEqual(tokenize("Badge? BLUE."), ("badge", "?", "blue", "."))
        with self.assertRaises(ValueError):
            generate("test", 8, 11)

    def test_shortcuts_fail_complete_pair_not_world_accuracy(self):
        rows = generate("train", 12, 33)
        result = shortcut_fixture_summary(rows)
        for rule, score in result.items():
            self.assertEqual(score["pairs"], 12)
            self.assertEqual(score["complete_pairs"], 12 if rule == "grammar" else 0)
            if rule == "grammar":
                self.assertEqual(score["correct_worlds"], 24)

    def test_falsifiers(self):
        a, b = generate("train", 2, 17)
        x, y = a.worlds
        with self.assertRaisesRegex(ValueError, "duplicate family"):
            audit_families((a, a))
        for mutant in (
            replace(a, worlds=(x, replace(y, answer=x.answer))),
            replace(a, worlds=(replace(x, support_positions=(999,)), y)),
            replace(a, worlds=(replace(x, prompt=x.prompt + " Bad suffix?"), y)),
            replace(a, split="dev"),
            replace(a, worlds=(replace(x, prompt=x.prompt.replace("withdrew", "accepted", 1)), y)),
        ):
            with self.subTest(mutant=mutant.split), self.assertRaises(ValueError):
                audit_families((mutant, b))


if __name__ == "__main__":
    unittest.main()
