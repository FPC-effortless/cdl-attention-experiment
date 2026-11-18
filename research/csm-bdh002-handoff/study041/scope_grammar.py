"""Resolve explicit approval constructions in the synthetic CSM grammar."""
import re

REFERENCE = r"(?:former|latter)"
SUBJECT = re.compile(
    rf"\bthe (?P<reference>{REFERENCE})"
    rf"(?: (?:rather than|instead of|as opposed to) the {REFERENCE})?"
    r" (?:receives|gets|is given) (?P<status>accepted|discarded)\b"
)
PREDICATE = re.compile(
    rf"\b(?:applies to|is assigned to|is granted to|is awarded to|pertains to|"
    rf"takes effect for|is valid for|becomes effective for) the (?P<reference>{REFERENCE})\b"
)
STATUS = re.compile(r"\b(?:accepted|discarded)\b")
UNSUPPORTED_SCOPE = re.compile(r"\b(?:not|never|unless|if|whether)\b")


def select(words):
    """Return the governing token offset, or abstain with -1."""
    if words.count("accepted") != 1:
        return -1
    start = 0
    for stop, word in enumerate([*words, ";"]):
        if word not in {";", ".", "!"}:
            continue
        text = " ".join(words[start:stop])
        statuses = list(STATUS.finditer(text))
        if "accepted" in words[start:stop]:
            if len(statuses) != 1 or UNSUPPORTED_SCOPE.search(text):
                return -1
            status = statuses[0]
            targets = [
                match.start("reference") for match in SUBJECT.finditer(text)
                if match.start("status") == status.start()
            ]
            targets.extend(
                match.start("reference") for match in PREDICATE.finditer(text)
                if match.start() >= status.end()
            )
            if len(targets) == 1:
                return start + text.count(" ", 0, targets[0])
            return -1
        start = stop + 1
    return -1
