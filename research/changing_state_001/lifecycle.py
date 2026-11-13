"""Exact public development semantics; no learned model or privileged parser claim."""
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Event:
    op: str
    key: int = 0
    arg: int = 0

    def __post_init__(self):
        if self.op not in {'SET', 'DELETE', 'LINK', 'COPY', 'NOP'}:
            raise ValueError('unknown operation')
        if type(self.key) is not int or type(self.arg) is not int or min(self.key, self.arg) < 0:
            raise ValueError('operands must be nonnegative integers')

class CurrentState:
    def __init__(self):
        self.records = {}

    def query(self, key: int) -> Optional[int]:
        seen = set()
        while key not in seen:
            seen.add(key)
            cell = self.records.get(key)
            if cell is None:
                return None
            kind, value = cell
            if kind == 'value':
                return value
            key = value
        return None

    def apply(self, event: Event):
        op, key, arg = event.op, event.key, event.arg
        if op == 'SET':
            self.records[key] = ('value', arg)
        elif op == 'LINK':
            self.records[key] = ('link', arg)
        elif op == 'DELETE':
            self.records.pop(key, None)
        elif op == 'COPY':
            value = self.query(arg)  # Resolve before changing even a self-copy.
            if value is None:
                self.records.pop(key, None)
            else:
                self.records[key] = ('value', value)


def history_query(events, key):
    """Independent oracle: search last authored event at a historical time.

    A live link is resolved at current time; COPY resolves at the time immediately
    before copying. Cycle identity includes time to distinguish old snapshots.
    """
    def resolve(target, end, visiting):
        marker = (target, end)
        if marker in visiting:
            return None
        visiting = visiting | {marker}
        for index in range(end - 1, -1, -1):
            event = events[index]
            if event.op == 'NOP' or event.key != target:
                continue
            if event.op == 'SET':
                return event.arg
            if event.op == 'DELETE':
                return None
            if event.op == 'LINK':
                return resolve(event.arg, end, visiting)
            if event.op == 'COPY':
                return resolve(event.arg, index, visiting)
        return None
    return resolve(key, len(events), set())
