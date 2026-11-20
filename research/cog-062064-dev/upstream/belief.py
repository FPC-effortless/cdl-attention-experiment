"""Sequential two-slot runtime. Provisional reads are estimates, never patches."""
import numpy as np
from .state_store import World

HOLD = '<hold>'
EMPTY = '<empty>'


class DialogueState:
    def __init__(self, labels, cutoffs, method, power=1., hold_weight=1., cutoff=0.):
        self.labels = labels
        self.hold = [v.index(HOLD) for v in labels.values()]
        self.cutoffs = cutoffs
        self.method = method
        self.power = power
        self.hold_weight = hold_weight
        self.cutoff = cutoff
        self._store = World()
        self.last_turn = -1
        self.mass = [self.point(slot, EMPTY) for slot in labels]

    def point(self, slot, value):
        labels = self.labels[slot]
        p = np.zeros(len(labels))
        p[labels.index(value)] = 1.
        return p

    def observe(self, turn, probabilities):
        if turn <= self.last_turn:
            raise ValueError('Each observation must have a new, increasing turn ID.')
        patch = {}
        updated = []
        for i, (slot, labels) in enumerate(self.labels.items()):
            p = probabilities[i]
            ix = int(p.argmax())
            best = labels[ix]
            commit = best != HOLD and p[ix] >= self.cutoffs[slot]
            if commit:
                patch[slot] = None if best == EMPTY else best
            previous = self.mass[i]
            if commit:
                current = self.point(slot, best)
            elif self.method == 'soft':
                weights = p ** self.power
                weights[self.hold[i]] *= self.hold_weight
                weights /= weights.sum()
                carry = weights[self.hold[i]]
                weights[self.hold[i]] = 0.
                current = weights + carry * previous
            elif self.method == 'hard' and best != HOLD and p[ix] >= self.cutoff:
                current = self.point(slot, best)
            else:
                current = previous
            updated.append(current)
        if patch and self._store.commit(str(turn), self._store.epoch(), patch) != 'committed':
            raise RuntimeError('Verified update failed')
        self.mass = updated
        self.last_turn = turn
        return patch

    def verified(self):
        return self._store.state()

    def estimates(self):
        verified = self.verified()
        out = {}
        for (slot, labels), p in zip(self.labels.items(), self.mass):
            value = labels[int(p.argmax())]
            out[slot] = (value, value != verified.get(slot, EMPTY), float(p.max()))
        return out

    def close(self):
        self._store.db.close()

