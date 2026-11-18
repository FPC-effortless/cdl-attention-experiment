"""Replay Study 040 with one additional selector in an isolated copy."""
import csv
import importlib
import sys
import tempfile
from pathlib import Path
from shutil import copytree

import pandas as pd

from scope_grammar import select

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "study040"


def replay():
    with tempfile.TemporaryDirectory() as directory:
        study = Path(directory) / "study040"
        copytree(SOURCE, study)
        sys.path.insert(0, str(study))
        run = importlib.import_module("run_study")
        raw = study / "replay.csv"
        with raw.open("w") as output:
            writer = csv.writer(output)
            writer.writerow("seed episode layout reforder approval text arm gold ref goldslot slot effect wrong".split())
            for seed in run.protocol["seeds"]:
                run.evaluate(seed, writer, extra_selector=select)
        rows = pd.read_csv(raw)
    quartet = ["seed", "layout", "arm", "episode"]
    assert not rows.duplicated(quartet + ["reforder", "approval"]).any()
    assert rows.groupby(quartet).size().eq(4).all()
    assert rows.groupby(quartet).goldslot.nunique().eq(2).all()
    assert not rows.text.str.contains("*", regex=False).any()
    assert rows.effect.eq(rows.slot.eq(rows.goldslot)).all()
    assert rows.wrong.eq(2 * (rows.slot.ge(0) & rows.slot.ne(rows.goldslot))).all()
    rows["joint"] = rows.ref.eq(rows.gold) & rows.slot.eq(rows.goldslot) & rows.effect.eq(1)
    rows["abstain"] = rows.slot.lt(0)
    cells = quartet[:-1]
    scores = rows.groupby(quartet).joint.all().groupby(cells).sum().to_frame()
    scores = scores.join(rows.groupby(cells)[["wrong", "abstain"]].sum())
    prior = pd.read_csv(SOURCE / "scores.csv", index_col=cells)
    pd.testing.assert_frame_equal(scores.drop("scope_grammar", level="arm"), prior)
    print(scores.xs("scope_grammar", level="arm").to_csv(), end="")


if __name__ == "__main__":
    replay()
