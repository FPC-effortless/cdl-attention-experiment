"""Constructed, paired, raw-text state-binding fixtures. No held test split."""
from collections import Counter
from dataclasses import dataclass
import hashlib
import random
import re

ANSWERS = ("blue", "gold", "green", "orange", "pink", "purple", "red", "white")
DEFAULT_DELAYS = (0, 2, 4, 8, 16)
ENTITY_BANKS = {
    "train": ("Ada", "Bola", "Cato", "Dara", "Elin", "Femi", "Gina", "Hugo",
              "Idris", "Juno", "Kemi", "Leto"),
    "dev": ("Mara", "Niko", "Ola", "Pia", "Quin", "Rafi", "Sade", "Tobi",
            "Una", "Vera", "Wren", "Xena"),
    "cal": ("Yara", "Zeno", "Anya", "Bran", "Cleo", "Dion", "Esme", "Finn",
            "Greta", "Hana", "Ivo", "Jade"),
}
WRITE = "The clerk filed the {name} badge as {value}."
NOOP = "The clerk proposed {value} for the {name} badge but withdrew it."
SUFFIX = "The clerk closed the ledger."
QUESTION = "What color is the badge for {name}?"
WRITE_RE = re.compile(r"The clerk filed the ([A-Za-z]+) badge as ([a-z]+)\.")
NOOP_RE = re.compile(r"The clerk proposed ([a-z]+) for the ([A-Za-z]+) badge but withdrew it\.")
QUESTION_RE = re.compile(r"What color is the badge for ([A-Za-z]+)\?")


@dataclass(frozen=True)
class World:
    prompt: str
    answer: str
    support_positions: tuple[int, ...]
    world_id: int


@dataclass(frozen=True)
class Family:
    family_id: str
    split: str
    delay: int
    worlds: tuple[World, World]


def tokenize(text: str) -> tuple[str, ...]:
    """Stable lowercase word/punctuation tokenizer; no metadata or IDs."""
    return tuple(re.findall(r"[A-Za-z]+|[^\w\s]", text.lower()))


def tokenize_chars(text: str) -> tuple[str, ...]:
    """Lossless except case/whitespace normalization; preserves unseen alias distinctions."""
    return tuple(" ".join(text.lower().split()))


def model_view(families) -> tuple[str, ...]:
    """The full model interface: raw prompt strings only, one per world."""
    return tuple(w.prompt for f in families for w in f.worlds)


def sentences(prompt: str) -> tuple[str, ...]:
    return tuple(re.split(r"(?<=[.?]) ", prompt))


def _id(split, seed, index):
    return hashlib.sha256(f"COG-COMM-001/v1:{split}:{seed}:{index}".encode()).hexdigest()


def _event(rng, names):
    """A non-target statement. Proposals do not update the ledger."""
    pattern = NOOP if rng.randrange(2) else WRITE
    return pattern.format(name=rng.choice(names), value=rng.choice(ANSWERS))


def _one_family(rng, split, i, seed, delay, values):
    target, sibling, *others = rng.sample(ENTITY_BANKS[split], 6)
    prefix = [
        WRITE.format(name=target, value=rng.choice(ANSWERS)),
        WRITE.format(name=sibling, value=rng.choice(ANSWERS)),
        *[_event(rng, others) for _ in range(rng.randint(1, 4))],
    ]
    # A canceled proposal about one swapped entity is present in every family.
    # Timing and its unrelated color are common within the pair.
    proposal = NOOP.format(name=rng.choice((target, sibling)), value=rng.choice(ANSWERS))
    rng.shuffle(prefix)
    middle = [_event(rng, others) for _ in range(rng.randrange(4))]
    tail = [_event(rng, others) for _ in range(delay)]
    if rng.randrange(2):
        order = (target, sibling)
    else:
        order = (sibling, target)
    # Put the canceled proposal before, between, or after the two evidence
    # writes. Recompute positions after insertion for verifier custody only.
    mark = rng.randrange(3)
    if mark == 0:
        prefix.insert(rng.randrange(len(prefix) + 1), proposal)
    elif mark == 1:
        middle.insert(rng.randrange(len(middle) + 1), proposal)
    else:
        tail.insert(rng.randrange(len(tail) + 1), proposal)
    positions = (len(prefix), len(prefix) + 1 + len(middle))
    latest_target = positions[order.index(target)]
    suffix_proposal = NOOP.format(name=rng.choice(others), value=rng.choice(ANSWERS))
    end = [*tail, suffix_proposal, SUFFIX, QUESTION.format(name=target)]
    rendered = []
    for j, answer in enumerate(values):
        assigns = {target: answer, sibling: values[1 - j]}
        text = " ".join((*prefix, WRITE.format(name=order[0], value=assigns[order[0]]),
                         *middle, WRITE.format(name=order[1], value=assigns[order[1]]), *end))
        rendered.append(World(text, answer, (latest_target,), j))
    return Family(_id(split, seed, i), split, delay, tuple(rendered))


def _matchings(colors):
    """Seven perfect matchings of the eight colors, covering 28 distinct pairs."""
    pivot, *wheel = colors
    for _ in range(len(wheel)):
        yield [(pivot, wheel[0]), *zip(wheel[1:4], reversed(wheel[4:]))]
        wheel = wheel[-1:] + wheel[:-1]


def generate(split: str, n: int, seed: int, *, delays=DEFAULT_DELAYS) -> tuple[Family, ...]:
    """Deterministic corpus. Delay counts distractor writes/proposals in the
    post-evidence tail; canceled proposals and closing sentence add further
    distance. Both worlds always share exact support positions. Each complete
    four-family block exactly balances answers, at a single delay.
    """
    if split not in ENTITY_BANKS:
        raise ValueError("supported constructed splits are train, dev and cal")
    if type(n) is not int or n < 1 or type(seed) is not int:
        raise ValueError("positive family count and integer seed required")
    delays = tuple(delays)
    if (not delays or any(type(x) is not int or not 0 <= x <= 256 for x in delays)
            or len(set(delays)) != len(delays)):
        raise ValueError("delay schedule must be nonempty with integer values in [0,256]")
    rng = random.Random(seed)
    colors = list(ANSWERS)
    rng.shuffle(colors)
    matchings = list(_matchings(colors))
    rng.shuffle(matchings)
    for matching in matchings:
        rng.shuffle(matching)
    order = list(delays)
    rng.shuffle(order)
    rows = []
    seen = set()
    for i in range(n):
        block = i // 4
        # When delay count is divisible by seven, rotate each delay to a new
        # matching upon returning to it rather than freezing that delay's pairs.
        cycle_offset = block // len(order) if len(order) % 7 == 0 else 0
        pair = matchings[(block + cycle_offset) % 7][i % 4]
        values = pair if rng.randrange(2) else pair[::-1]
        for _ in range(100):
            f = _one_family(rng, split, i, seed, order[block % len(order)], values)
            texts = {w.prompt for w in f.worlds}
            if texts.isdisjoint(seen):
                seen.update(texts)
                rows.append(f)
                break
        else:
            raise RuntimeError("could not generate a unique family")
    rng.shuffle(rows)
    return tuple(rows)


make_families = generate


def resolve_prompt(prompt: str) -> tuple[str, int]:
    """Exact deterministic grammar control, not a trained or language-general model."""
    *events, closing, question = sentences(prompt)
    ask = QUESTION_RE.fullmatch(question)
    if closing != SUFFIX or ask is None:
        raise ValueError("invalid end or question")
    values = {}
    for position, sent in enumerate(events):
        match = WRITE_RE.fullmatch(sent)
        if match is not None and match[2] in ANSWERS:
            values[match[1]] = (match[2], position)
        elif (proposal := NOOP_RE.fullmatch(sent)) is None or proposal[1] not in ANSWERS:
            raise ValueError("unsupported statement")
    if ask[1] not in values:
        raise ValueError("queried record has no accepted write")
    return values[ask[1]]


def shortcut_predict(prompt: str, rule: str) -> str:
    """Fixed raw-text shortcuts; none takes verifier-only metadata."""
    toks = tokenize(prompt)
    colors = [t for t in toks if t in ANSWERS]
    if rule == "first_color":
        return colors[0] if colors else ANSWERS[0]
    if rule == "last_color":
        return colors[-1] if colors else ANSWERS[0]
    if rule == "color_bag":
        c = Counter(colors)
        return max(ANSWERS, key=lambda a: (c[a], -ANSWERS.index(a)))
    if rule == "suffix_hash":
        key = " ".join(sentences(prompt)[-3:]).encode()
        return ANSWERS[int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(ANSWERS)]
    if rule == "position_hash":
        # Only length and positions of color tokens; their identities removed.
        key = repr((len(toks), tuple(i for i, t in enumerate(toks) if t in ANSWERS))).encode()
        return ANSWERS[int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % len(ANSWERS)]
    if rule == "grammar":
        return resolve_prompt(prompt)[0]
    raise ValueError(f"unknown shortcut: {rule}")


def shortcut_fixture_summary(families):
    """Score fixed controls on caller-supplied construction fixtures only."""
    rows = tuple(families)
    out = {}
    for rule in ("first_color", "last_color", "color_bag", "suffix_hash", "position_hash", "grammar"):
        hits = [tuple(shortcut_predict(w.prompt, rule) == w.answer for w in f.worlds)
                for f in rows]
        out[rule] = {"pairs": len(hits), "complete_pairs": sum(all(h) for h in hits),
                     "correct_worlds": sum(sum(h) for h in hits)}
    return out


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_families(families):
    """Structural leakage/semantics checks. Does not run a trained model."""
    rows = tuple(families)
    keys, seen = set(), set()
    rosters = {}
    for f in rows:
        _require(f.family_id not in keys, "duplicate family")
        keys.add(f.family_id)
        _require(f.split in ENTITY_BANKS, "unknown lexical split")
        _require(len(f.worlds) == 2, "need two worlds")
        a, b = f.worlds
        ws = [sentences(w.prompt) for w in f.worlds]
        _require((a.world_id, b.world_id) == (0, 1) and a.answer != b.answer,
                 "bad paired roster or unchanged answer")
        _require(ws[0][-3:] == ws[1][-3:], "paired suffix or question changed")
        _require(Counter(tokenize(a.prompt)) == Counter(tokenize(b.prompt)), "unequal word bag")
        _require(len(ws[0]) == len(ws[1]), "unaligned statements")
        changed = [(p, WRITE_RE.fullmatch(x), WRITE_RE.fullmatch(y))
                   for p, (x, y) in enumerate(zip(*ws)) if x != y]
        _require(len(changed) == 2 and all(x and y and x[1] == y[1] for _, x, y in changed),
                 "two aligned assignments must be the only changed statements")
        _require(changed[0][1][2] == changed[1][2][2] and changed[1][1][2] == changed[0][2][2],
                 "assignment values were not swapped")
        _require(a.support_positions == b.support_positions, "support positions differ")
        for w in f.worlds:
            _require(w.prompt not in seen, "duplicate world text")
            seen.add(w.prompt)
            expected, pos = resolve_prompt(w.prompt)
            _require(w.answer == expected and w.support_positions == (pos,), "bad truth or support offset")
            _require(f.family_id not in w.prompt, "family ID leaked into prompt")
            stmts = sentences(w.prompt)
            mentioned = {QUESTION_RE.fullmatch(stmts[-1])[1]}
            for sent in stmts[:-2]:
                if (match := WRITE_RE.fullmatch(sent)) is not None:
                    mentioned.add(match[1])
                elif (match := NOOP_RE.fullmatch(sent)) is not None:
                    mentioned.add(match[2])
            _require(mentioned <= set(ENTITY_BANKS[f.split]), "alias crossed split")
        counts = rosters.setdefault(f.split, Counter())
        counts.update((a.answer, b.answer))
    return {"families": len(rows), "worlds": len(seen),
            "answer_counts": {s: dict(c) for s, c in rosters.items()}}
