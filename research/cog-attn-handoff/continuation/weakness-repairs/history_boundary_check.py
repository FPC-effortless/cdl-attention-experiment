import sqlite3
from tempfile import TemporaryDirectory
from pathlib import Path

from retained_history import as_of, build_bounded_snapshot
from storage import populate, value

with TemporaryDirectory() as directory:
    source = Path(directory) / "full.db"
    dest = Path(directory) / "bounded.db"
    db, _ = populate(source, [(0, 1, 1, 110), (0, 2, 2, 122), (0, 2, 7, 127),
                              (0, 4, 4, 144), (1, 0, 3, 103)], True)
    db.close()
    build_bounded_snapshot(source, dest, 3, 5)
    with sqlite3.connect(dest) as retained, sqlite3.connect(source) as full:
        for vt, tx in [(3, 5), (3, 7), (4, 5), (4, 7)]:
            for key in (0, 1, 2):
                assert as_of(retained, key, vt, tx) == value(full, (key, vt, tx))
        for vt, tx in [(2, 5), (3, 4)]:
            try:
                as_of(retained, 0, vt, tx)
            except ValueError:
                continue
            raise AssertionError((vt, tx))
print("12 in-domain equivalence checks, including a late correction. Both unsupported cutoffs rejected.")
