"""Create a separate, bounded bitemporal snapshot with an explicit read floor."""
import sqlite3
from storage import value


def build_bounded_snapshot(source, destination, valid_floor, transaction_floor):
    if destination.exists():
        raise FileExistsError(destination)
    complete=sqlite3.connect(f"{source.resolve().as_uri()}?mode=ro",uri=True)
    retained=sqlite3.connect(destination)
    try:
        complete.execute("BEGIN")
        with retained:
            retained.execute("CREATE TABLE events(k INTEGER,valid INTEGER,tx INTEGER,v INTEGER)")
            retained.execute("CREATE TABLE read_floor(valid INTEGER,tx INTEGER)")
            retained.execute("INSERT INTO read_floor VALUES(?,?)",(valid_floor,transaction_floor))
            retained.executemany("INSERT INTO events VALUES(?,?,?,?)",complete.execute("""
                SELECT k,valid,tx,v FROM events WHERE valid>? OR tx>?
                UNION ALL
                SELECT k,valid,tx,v FROM (
                    SELECT *,row_number() OVER (PARTITION BY k ORDER BY valid DESC,tx DESC) AS rank
                    FROM events WHERE valid<=? AND tx<=?
                ) WHERE rank=1""",(valid_floor,transaction_floor,valid_floor,transaction_floor)))
            retained.execute("CREATE INDEX temporal ON events(k,valid DESC,tx DESC)")
    finally:
        retained.close()
        complete.close()


def as_of(connection, key, valid, transaction):
    valid_min,transaction_min=connection.execute("SELECT valid,tx FROM read_floor").fetchone()
    if valid<valid_min or transaction<transaction_min:
        raise ValueError("query predates the retained history boundary")
    return value(connection,(key,valid,transaction))
