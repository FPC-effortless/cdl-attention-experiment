"""Frozen language fixtures with a held-out clarity and argument-revision split."""
import random
import re

import numpy as np

from binding import KNOWN, rows
from language import NAMES, apply

TRAIN_CLEAR = (
    "The amount kept in {s} is the input, and I want the {v} result written on {d}. Please leave {u} alone.",
    "{d} is the account to change. Take the number from {s}, perform {v}, and keep {u} as it is.",
    "Use whatever {s} holds for {v}. Record the new figure in {d}; {u} has no part in it.",
    "Leave {u} aside. Adjust {d} by using the {v} method on the figure taken from {s}.",
    "Either {u} or {s} was suggested earlier. I settled on {s} as the source for {v}; update only {d}.",
    "Not {u}: I want the amount in {s} as input to {v}, and the result belongs in {d}.",
    "It was unclear where the value should go, so use {d}. Read {s} as the input for {v}; leave {u}.",
)
TRAIN_AMBIG = (
    "{v} into {d} using the amount from {s} or {u}; I have not picked which one.",
    "Read {s} and {v} it, but I don't know if the result should go to {d} or {u}.",
    "One of {s} and {u} has the input for {v}. Update {d} once someone chooses the source.",
    "{s} and {u} have competing figures. Until we decide which one to use, the {v} update in {d} is uncertain.",
)
CAL_CLEAR = (
    "The {v} adjustment is for {d}, based on {s}. It is not for {u}.",
    "I considered {u} or {d}; settle on {d} as the result. Apply {v} using the amount in {s}.",
    "Let's {v} the value held by {s} and put the result in {d}, without using {u}.",
)
CAL_AMBIG = (
    "Choose {s} or {u} for {v} into {d}, but wait until I say which.",
    "The {v} input is {s}. I still need to choose between updating {u} and {d}.",
    "Apply {v} to {d}. The source of the amount could be {s} or {u}.",
)
IMPLICIT = (
    "{s} is upstream here. When the {v} calculation is done {d} ought to reflect it. {u} has nothing to do with this.",
    "Keep the entry on {u} intact. The {v} outcome belongs on the sheet for {d}, based on what you find under {s}.",
    "For the final tally under {d}, do {v} against the figure sitting with {s}. {u} is a separate job.",
    "The reference for this {v} is {s}. The place that should end up different is {d}. Leave the number on {u}.",
    "Only {d} needs a fresh tally. Run {v} using the balance shown for {s}; keep {u} as it stands.",
    "Use the tally posted to {s} as the operand for {v}. Post the new tally to {d}. Ignore {u}.",
    "The last figure, {d}, is the place for {v}. Use the amount from the middle one, {s}, rather than the first one, {u}.",
    "The ledger for {u} can wait. Work out {v} from the amount under {s} and save the changed total under {d}.",
)
CORRECTIONS = (
    "I said {u} at first. Correction: the number comes from {s}. Use {v} to update {d}.",
    "Should that result go to {u} or {d}? Use {d}, we decided. Do {v} with the number in {s}.",
    "Earlier I wrote down {u} as the destination. Make that {d} instead. Use {s} as the operand for {v}.",
    "The source is {s}. There's no uncertainty about the receiving ledger anymore: {d} is right, {u} is wrong. Proceed with {v}.",
)
AMBIGUOUS = (
    "We have not agreed which of {s} and {u} supplies the amount, though we do know {d} is where the {v} result belongs.",
    "The operand for {v} comes from {s}, but {d} and {u} are both listed as the place to post it.",
    "One note calls {s} the source and another says {u}. I cannot tell which governs the {v} adjustment to {d}.",
    "No decision yet about the receiving ledger. Take the value in {s} for {v}; both {u} and {d} are still candidates.",
    "{s} was originally the operand and {u} was proposed afterward. Nothing says the change was approved. Hold the {v} entry for {d}.",
    "If {u} is the current sheet, write there. Otherwise write to {d}. I have no way to determine which is current. Use {s} for {v}.",
)
COHORTS = ("layout", "implicit", "correction", "ambiguous")


def generate(seed, n, verbs, patterns):
    rng = random.Random(seed)
    output = []
    for _ in range(n):
        op = rng.randrange(4)
        src, dst, other = rng.sample(NAMES, 3)
        rendered = rng.choice(patterns).format(v=verbs[op], s=src, d=dst, u=other)
        tokens = re.findall(r"[a-z]+|[^\w\s]", rendered.lower())
        names = list(dict.fromkeys(t for t in tokens if t in NAMES))
        s, d = names.index(src), names.index(dst)
        state = np.array([rng.randrange(8) for _ in names], dtype=np.int64)
        result = state.copy()
        result[d] = apply(op,state[s],state[d])
        output.append((tokens,names,state,result,op,s,d))
    return output


def fixture(seed, extra):
    train = rows(seed,extra,2048,"familiar") + generate(seed+1,2048,KNOWN,TRAIN_CLEAR)
    support = rows(seed+2,extra,96,"support")
    train_ambig = generate(seed+3,768,KNOWN,TRAIN_AMBIG)
    cal = {"clear":generate(seed+5,384,extra,CAL_CLEAR), "ambiguous":generate(seed+6,384,extra,CAL_AMBIG)}
    test = {
        "layout":rows(seed+11,extra,512,"joint"),
        "implicit":generate(seed+12,512,extra,IMPLICIT),
        "correction":generate(seed+13,512,extra,CORRECTIONS),
        "ambiguous":generate(seed+14,512,extra,AMBIGUOUS),
    }
    return train, support, train_ambig, cal, test
