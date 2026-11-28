"""Frozen blind English transfer items for a four-person digit task."""

# BEGIN SEALED TEMPLATES
FAMILIES = (
    {
        "id": "lists",
        "sources": ("blue", "green"),
        "open": (
            "Use the signed list, the {good} one. The {bad} list was a first guess.",
            "The {bad} list was a guess. It's the {good} list that was signed off.",
        ),
        "firm": (
            "The {good} list gives {who} a {n}.",
            "Under {who} on the {good} list: {n}.",
        ),
        "weak": (
            "Under {who} on the {bad} list: {n}.",
            "The {bad} list gives {who} a {n}.",
        ),
        "ask": ("Which digit should I copy for {who}?", "What goes next to {who} on my copy?"),
    },
    {
        "id": "witness",
        "sources": ("the host", "the guest"),
        "open": (
            "Only {good} saw the draw. All {bad} can offer is a guess.",
            "{Bad} missed the draw and guessed. {Good} was there and saw every result.",
        ),
        "firm": (
            '"{Who} drew {n}," said {good}.',
            '{Good} said, "{Who} drew {n}."',
        ),
        "weak": (
            '{Bad} said, "{Who} drew {n}."',
            '"{Who} drew {n}," said {bad}.',
        ),
        "ask": ("Remind me what {who} drew.", "Which digit did {who} actually draw?"),
    },
    {
        "id": "marks",
        "sources": ("star", "circle"),
        "open": (
            "A {bad} marks a guess. A {good} marks a checked entry.",
            "I've marked the checked entries with a {good} and the guesses with a {bad}.",
        ),
        "firm": (
            "{Who} got {n} ({good}).",
            "{Good}: {n} for {who}.",
        ),
        "weak": (
            "{Bad}: {n} for {who}.",
            "{Who} got {n} ({bad}).",
        ),
        "ask": ("What was the checked entry for {who}?", "Which digit can we put down for {who}?"),
    },
    {
        "id": "callbacks",
        "sources": ("aunt", "uncle"),
        "open": (
            "We rang around to find out which ticket each person had.",
            "Everyone got one numbered ticket. Here's what we heard back.",
        ),
        "firm": (
            "{Who}'s {good} read the ticket aloud. It was {n}.",
            "On {who}'s ticket the printed digit was {n}, their {good} confirmed.",
        ),
        "weak": (
            '"Maybe {n}?" was the {bad}\'s guess about {who}.',
            "{Who} might have had {n}, their {bad} thought.",
        ),
        "ask": ("What was printed on {who}'s ticket?", "Which ticket did {who} end up with?"),
    },
    {
        "id": "draw_days",
        "sources": ("Tuesday", "Friday"),
        "open": (
            "The draw was on {good}. On {bad} the numbers were just guesses for a different exercise.",
            "They used guesses for the {bad} exercise. The real draw took place on {good}.",
        ),
        "firm": (
            "{Good}'s entry for {who} was {n}.",
            "{Who} had {n} under {good}.",
        ),
        "weak": (
            "{Who} had {n} under {bad}.",
            "{Bad}'s entry for {who} was {n}.",
        ),
        "ask": ("In the real draw, what did {who} get?", "Which digit did the draw give {who}?"),
    },
    {
        "id": "tickets",
        "sources": ("desk", "door"),
        "open": (
            "People picked up numbered passes. I made a note of what I saw and what I only heard.",
            "Here's how I pieced together the pass numbers.",
        ),
        "firm": (
            "I saw {n} on {who}'s pass at the {good}.",
            "At the {good}, {who} showed me a pass bearing {n}.",
        ),
        "weak": (
            "By the {bad} someone wondered if {who}'s number was {n}.",
            "{Who} may have got {n}, someone at the {bad} thought.",
        ),
        "ask": ("Which number did I see on {who}'s pass?", "For {who}, which digit can I vouch for?"),
    },
    {
        "id": "transcript",
        "sources": ("left", "right"),
        "open": (
            "Two mics picked up two conversations. The {good} mic caught the judge reading the scores. The {bad} mic caught fans guessing.",
            "Fans were guessing into the {bad} mic. The judge read the official scores into the {good} mic.",
        ),
        "firm": (
            '{Good} mic: "{Who} gets {n}."',
            '"{n} for {who}." ({Good} mic.)',
        ),
        "weak": (
            '"{n} for {who}." ({Bad} mic.)',
            '{Bad} mic: "{Who} gets {n}."',
        ),
        "ask": ("How many points did the judge give {who}?", "What's {who}'s score according to the judge?"),
    },
    {
        "id": "edits",
        "sources": ("slip", "card"),
        "open": (
            "Someone mixed the proposed numbers in with the assigned ones.",
            "Some numbers came up for discussion. Each person eventually got one number.",
        ),
        "firm": (
            "They settled on {n} for {who} and filed the {good}.",
            "{Who}'s filed {good} gives the assigned number as {n}.",
        ),
        "weak": (
            "A pencilled {bad} suggests giving {who} a {n}.",
            "For {who} someone floated {n} on a spare {bad}.",
        ),
        "ask": ("What number did they settle on for {who}?", "Which digit did {who} end up getting?"),
    },
)
# END SEALED TEMPLATES

import hashlib
import itertools
import operator
import random


_TEMPLATE_SHA256 = "acd7ca0940dd5aceb76e1b5e4825e6e27cb2a1e2516fbdc46337aa758927dc70"
_NAMES = {
    "familiar": ("Ada", "Ben", "Cia", "Dan"),
    "novel": ("Pim", "Rui", "Taj", "Zev"),
}
_ARMS = ("base", "evidence_edit", "query_swap")


def _worlds(family, names):
    # Restarting from the same local seed makes the two name groups exact renamings.
    seed = hashlib.sha256(("blind-v1/" + family["id"]).encode("ascii")).digest()
    rng = random.Random(int.from_bytes(seed[:16], "big"))
    # Each name/digit combination is the base question exactly four times.
    schedule = list(itertools.product(range(4), range(4))) * 4
    rng.shuffle(schedule)
    for target, answer in schedule:
        other_target = rng.choice([i for i in range(4) if i != target])
        other_answer = rng.choice([i for i in range(4) if i != answer])
        edited_answer = rng.choice([i for i in range(4) if i != answer])
        values = [rng.randrange(4) for _ in names]
        values[target] = answer
        values[other_target] = other_answer
        guesses = []
        for i, value in enumerate(values):
            excluded = {value}
            if i == target:
                excluded.add(edited_answer)
            guesses.append(rng.choice([n for n in range(4) if n not in excluded]))

        good, bad = rng.sample(family["sources"], 2)
        context = dict(good=good, Good=good[0].upper() + good[1:],
                       bad=bad, Bad=bad[0].upper() + bad[1:])
        opening = rng.choice(family["open"]).format(**context)
        question = rng.choice(family["ask"])
        claims = []
        for i, name in enumerate(names):
            for status, n in (("firm", values[i]), ("weak", guesses[i])):
                line = rng.choice(family[status])
                args = dict(context, who=name, Who=name)
                original = line.format(n=n, **args)
                modified = (line.format(n=edited_answer, **args)
                            if i == target and status == "firm" else original)
                claims.append((original, modified))
        rng.shuffle(claims)
        join = rng.choice((" ", "\n"))
        prefix = opening + join
        base_text = prefix + join.join(x[0] for x in claims)
        edit_text = prefix + join.join(x[1] for x in claims)
        ask = question.format(who=names[target])
        swap_ask = question.format(who=names[other_target])
        yield (
            (base_text + "\n" + ask, answer, guesses[target]),
            (edit_text + "\n" + ask, edited_answer, guesses[target]),
            (base_text + "\n" + swap_ask, other_answer, guesses[other_target]),
        )


def _empty_counts():
    return {
        "worlds": 0,
        "prompts": 0,
        "correct": 0,
        "abstained": 0,
        "wrong_nonabstention": 0,
        "wrong_matching_weak_digit": 0,
        "non_digit_nonabstention": 0,
        "evidence_pair_correct": 0,
        "query_pair_correct": 0,
        "both_pairs_correct": 0,
        **{arm + "_correct": 0 for arm in _ARMS},
    }


def _score(solver, world):
    delta = _empty_counts()
    delta.update(worlds=1, prompts=3)
    correct = []
    for arm, (prompt, target, guess) in zip(_ARMS, world):
        raw = solver(prompt.encode("utf-8"))
        # operator.index permits integer scalars returned by array libraries.
        byte = None if raw is None else operator.index(raw)
        hit = byte == 48 + target
        correct.append(hit)
        delta[arm + "_correct"] += int(hit)
        if hit:
            delta["correct"] += 1
        elif byte is None:
            delta["abstained"] += 1
        else:
            delta["wrong_nonabstention"] += 1
            delta["wrong_matching_weak_digit"] += int(byte == 48 + guess)
            delta["non_digit_nonabstention"] += int(not 48 <= byte <= 51)
    delta["evidence_pair_correct"] = int(correct[0] and correct[1])
    delta["query_pair_correct"] = int(correct[0] and correct[2])
    delta["both_pairs_correct"] = int(all(correct))
    return delta


def evaluate_solver(solver):
    """Score a callable from UTF-8 bytes to an ASCII digit byte integer or None.

    Each pair requires a correct answer on both its baseline and its changed
    prompt. Both-pairs success requires all three answers in the world.
    The two name groups use the same latent worlds and identical wording.
    Outcomes are integer counts suitable for JSON serialization.
    """
    by_family = {f["id"]: {group: _empty_counts() for group in _NAMES}
                 for f in FAMILIES}
    by_names = {group: _empty_counts() for group in _NAMES}
    total = _empty_counts()
    for f in FAMILIES:
        for group, names in _NAMES.items():
            tallies = (by_family[f["id"]][group], by_names[group], total)
            for world in _worlds(f, names):
                delta = _score(solver, world)
                for tally in tallies:
                    for key, value in delta.items():
                        tally[key] += value
    return dict(template_sha256=_TEMPLATE_SHA256, overall=total,
                by_name_group=by_names, by_family=by_family)
